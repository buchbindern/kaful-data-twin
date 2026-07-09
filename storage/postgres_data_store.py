"""
PostgresDataStore — the same DataStore interface as SQLiteDataStore, backed by
PostgreSQL (psycopg 3). This is the fleet/production backend: persistent (survives
an ephemeral deploy filesystem) and concurrency-safe for many machines.

Design mirrors SQLiteDataStore exactly (same tables, same ISO-8601 TEXT datetimes,
same JSON feature blobs) so behaviour is identical — only the driver differs:
  * placeholders are %s (not ?),
  * BLOB -> BYTEA, REAL -> DOUBLE PRECISION,
  * a single autocommit connection guarded by a lock (single-writer model, matching
    the SQLite store; the multi-worker path is a psycopg connection pool).
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Optional

import psycopg
from psycopg.rows import dict_row

from domain.models import (
    Machine, Run, Cut, FeatureRecord, RULPrediction, TwinState, WearLabel, User, Session,
    CutResult)
from domain.stores import DataStore


def _dt(value: datetime) -> str:
    return value.isoformat()


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    return datetime.fromisoformat(value) if value is not None else None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS machines (
    machine_id   TEXT PRIMARY KEY, machine_type TEXT NOT NULL, name TEXT,
    created_at   TEXT NOT NULL, owner_id TEXT);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, machine_id TEXT NOT NULL REFERENCES machines(machine_id),
    started_at TEXT NOT NULL, ended_at TEXT, tool_id TEXT);
CREATE TABLE IF NOT EXISTS cuts (
    run_id TEXT NOT NULL REFERENCES runs(run_id), cut_index INTEGER NOT NULL,
    waveform_key TEXT NOT NULL, ingested_at TEXT NOT NULL, PRIMARY KEY (run_id, cut_index));
CREATE TABLE IF NOT EXISTS features (
    run_id TEXT NOT NULL, cut_index INTEGER NOT NULL, features_json TEXT NOT NULL,
    extracted_at TEXT NOT NULL, PRIMARY KEY (run_id, cut_index),
    FOREIGN KEY (run_id, cut_index) REFERENCES cuts(run_id, cut_index));
CREATE TABLE IF NOT EXISTS rul_predictions (
    run_id TEXT NOT NULL, cut_index INTEGER NOT NULL, rul_median DOUBLE PRECISION NOT NULL,
    rul_lower DOUBLE PRECISION NOT NULL, rul_upper DOUBLE PRECISION NOT NULL,
    ci_level DOUBLE PRECISION NOT NULL, predicted_at TEXT NOT NULL,
    PRIMARY KEY (run_id, cut_index),
    FOREIGN KEY (run_id, cut_index) REFERENCES cuts(run_id, cut_index));
CREATE TABLE IF NOT EXISTS twin_state (
    run_id TEXT PRIMARY KEY REFERENCES runs(run_id), cut_index INTEGER NOT NULL,
    params_json TEXT NOT NULL, particles BYTEA, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY, email TEXT NOT NULL UNIQUE, password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(user_id),
    created_at TEXT NOT NULL, expires_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS cut_results (
    run_id TEXT NOT NULL, cut_index INTEGER NOT NULL,
    wear_mean DOUBLE PRECISION NOT NULL, wear_lo DOUBLE PRECISION NOT NULL,
    wear_hi DOUBLE PRECISION NOT NULL, wear_true DOUBLE PRECISION,
    rul_median DOUBLE PRECISION, rul_lo DOUBLE PRECISION, rul_hi DOUBLE PRECISION,
    censored DOUBLE PRECISION NOT NULL, computed_at TEXT NOT NULL,
    PRIMARY KEY (run_id, cut_index));
CREATE TABLE IF NOT EXISTS wear_labels (
    run_id TEXT NOT NULL REFERENCES runs(run_id), cut_index INTEGER NOT NULL,
    wear_mm DOUBLE PRECISION NOT NULL, PRIMARY KEY (run_id, cut_index));
"""


class PostgresDataStore(DataStore):
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._lock = threading.RLock()          # single-writer model (matches SQLite store)
        self._conn = self._connect()
        with self._lock, self._conn.cursor() as cur:
            for stmt in _SCHEMA.split(";"):
                if stmt.strip():
                    cur.execute(stmt)

    def _connect(self):
        return psycopg.connect(self._dsn, autocommit=True, row_factory=dict_row)

    def _query(self, sql, params, fetch):
        """Run a statement, reconnecting once if the connection was dropped (Neon
        auto-suspend / server restart terminates idle connections)."""
        with self._lock:
            for attempt in (1, 2):
                try:
                    with self._conn.cursor() as cur:
                        cur.execute(sql, params)
                        if fetch == "one":
                            return cur.fetchone()
                        if fetch == "all":
                            return cur.fetchall()
                        return None
                except psycopg.OperationalError:
                    if attempt == 2:
                        raise
                    try:
                        self._conn.close()
                    except Exception:
                        pass
                    self._conn = self._connect()

    def close(self) -> None:
        self._conn.close()

    def _one(self, sql, params=()):
        return self._query(sql, params, "one")

    def _all(self, sql, params=()):
        return self._query(sql, params, "all")

    def _exec(self, sql, params=()):
        return self._query(sql, params, "none")

    # ---------------- Machine ----------------
    def create_machine(self, machine: Machine) -> None:
        self._exec("INSERT INTO machines (machine_id,machine_type,name,created_at,owner_id) "
                   "VALUES (%s,%s,%s,%s,%s)",
                   (machine.machine_id, machine.machine_type, machine.name,
                    _dt(machine.created_at), machine.owner_id))

    def rename_machine(self, machine_id: str, name) -> None:
        self._exec("UPDATE machines SET name=%s WHERE machine_id=%s", (name, machine_id))

    def _many(self, sql, rows):
        with self._lock:
            for attempt in (1, 2):
                try:
                    with self._conn.cursor() as cur:
                        cur.executemany(sql, rows)
                    return
                except psycopg.OperationalError:
                    if attempt == 2:
                        raise
                    try:
                        self._conn.close()
                    except Exception:
                        pass
                    self._conn = self._connect()

    def get_machine(self, machine_id: str) -> Optional[Machine]:
        r = self._one("SELECT * FROM machines WHERE machine_id=%s", (machine_id,))
        if r is None:
            return None
        return Machine(r["machine_id"], r["machine_type"], r["name"],
                       _parse_dt(r["created_at"]), r["owner_id"])

    def list_machines(self):
        rows = self._all("SELECT * FROM machines ORDER BY machine_id")
        return [Machine(r["machine_id"], r["machine_type"], r["name"],
                        _parse_dt(r["created_at"]), r["owner_id"]) for r in rows]

    # ---------------- Run ----------------
    def create_run(self, run: Run) -> None:
        self._exec("INSERT INTO runs (run_id,machine_id,started_at,ended_at,tool_id) "
                   "VALUES (%s,%s,%s,%s,%s)",
                   (run.run_id, run.machine_id, _dt(run.started_at),
                    _dt(run.ended_at) if run.ended_at else None, run.tool_id))

    def _row_to_run(self, r) -> Run:
        return Run(r["run_id"], r["machine_id"], _parse_dt(r["started_at"]),
                   _parse_dt(r["ended_at"]), r["tool_id"])

    def get_run(self, run_id: str) -> Optional[Run]:
        r = self._one("SELECT * FROM runs WHERE run_id=%s", (run_id,))
        return self._row_to_run(r) if r else None

    def get_active_run(self, machine_id: str) -> Optional[Run]:
        r = self._one("SELECT * FROM runs WHERE machine_id=%s AND ended_at IS NULL "
                      "ORDER BY started_at DESC LIMIT 1", (machine_id,))
        return self._row_to_run(r) if r else None

    def end_run(self, run_id: str, ended_at: datetime) -> None:
        self._exec("UPDATE runs SET ended_at=%s WHERE run_id=%s", (_dt(ended_at), run_id))

    def delete_run(self, run_id: str) -> None:
        for t in ("cut_results", "rul_predictions", "twin_state", "features", "wear_labels", "cuts"):
            self._exec(f"DELETE FROM {t} WHERE run_id=%s", (run_id,))
        self._exec("DELETE FROM runs WHERE run_id=%s", (run_id,))

    def delete_machine(self, machine_id: str) -> None:
        for r in self.list_runs(machine_id):
            self.delete_run(r.run_id)
        self._exec("DELETE FROM machines WHERE machine_id=%s", (machine_id,))

    def rename_run(self, run_id: str, label) -> None:
        self._exec("UPDATE runs SET tool_id=%s WHERE run_id=%s", (label, run_id))

    def list_runs(self, machine_id: str):
        rows = self._all("SELECT * FROM runs WHERE machine_id=%s ORDER BY started_at DESC",
                         (machine_id,))
        return [self._row_to_run(r) for r in rows]

    # ---------------- Auth ----------------
    def create_user(self, user: User) -> None:
        self._exec("INSERT INTO users (user_id,email,password_hash,created_at) "
                   "VALUES (%s,%s,%s,%s)",
                   (user.user_id, user.email, user.password_hash, _dt(user.created_at)))

    def _row_to_user(self, r) -> User:
        return User(r["user_id"], r["email"], r["password_hash"], _parse_dt(r["created_at"]))

    def get_user_by_email(self, email: str):
        r = self._one("SELECT * FROM users WHERE email=%s", (email,))
        return self._row_to_user(r) if r else None

    def get_user(self, user_id: str):
        r = self._one("SELECT * FROM users WHERE user_id=%s", (user_id,))
        return self._row_to_user(r) if r else None

    def create_session(self, session: Session) -> None:
        self._exec("INSERT INTO sessions (token,user_id,created_at,expires_at) "
                   "VALUES (%s,%s,%s,%s)",
                   (session.token, session.user_id, _dt(session.created_at),
                    _dt(session.expires_at)))

    def get_valid_session(self, token: str, now=None):
        r = self._one("SELECT * FROM sessions WHERE token=%s", (token,))
        if r is None:
            return None
        sess = Session(r["token"], r["user_id"], _parse_dt(r["created_at"]),
                       _parse_dt(r["expires_at"]))
        if sess.expires_at <= (now or datetime.now(timezone.utc)):
            return None
        return sess

    def delete_session(self, token: str) -> None:
        self._exec("DELETE FROM sessions WHERE token=%s", (token,))

    def delete_user_sessions(self, user_id: str) -> None:
        self._exec("DELETE FROM sessions WHERE user_id=%s", (user_id,))

    def delete_expired_sessions(self, now) -> None:
        self._exec("DELETE FROM sessions WHERE expires_at <= %s", (_dt(now),))

    # ---------------- Cut ----------------
    def append_cut(self, cut: Cut) -> None:
        self._exec("INSERT INTO cuts (run_id,cut_index,waveform_key,ingested_at) "
                   "VALUES (%s,%s,%s,%s)",
                   (cut.run_id, cut.cut_index, cut.waveform_key, _dt(cut.ingested_at)))

    def append_cuts_bulk(self, cuts) -> None:
        self._many("INSERT INTO cuts (run_id,cut_index,waveform_key,ingested_at) VALUES (%s,%s,%s,%s)",
                   [(c.run_id, c.cut_index, c.waveform_key, _dt(c.ingested_at)) for c in cuts])

    def get_cut(self, run_id: str, cut_index: int) -> Optional[Cut]:
        r = self._one("SELECT * FROM cuts WHERE run_id=%s AND cut_index=%s", (run_id, cut_index))
        if r is None:
            return None
        return Cut(r["run_id"], r["cut_index"], r["waveform_key"], _parse_dt(r["ingested_at"]))

    # ---------------- Features ----------------
    def append_features(self, record: FeatureRecord) -> None:
        self._exec("INSERT INTO features (run_id,cut_index,features_json,extracted_at) "
                   "VALUES (%s,%s,%s,%s)",
                   (record.run_id, record.cut_index, json.dumps(record.features),
                    _dt(record.extracted_at)))

    def append_features_bulk(self, records) -> None:
        self._many("INSERT INTO features (run_id,cut_index,features_json,extracted_at) "
                   "VALUES (%s,%s,%s,%s)",
                   [(r.run_id, r.cut_index, json.dumps(r.features), _dt(r.extracted_at))
                    for r in records])

    def _row_to_features(self, r) -> FeatureRecord:
        return FeatureRecord(r["run_id"], r["cut_index"], json.loads(r["features_json"]),
                             _parse_dt(r["extracted_at"]))

    def get_features(self, run_id: str, cut_index: int) -> Optional[FeatureRecord]:
        r = self._one("SELECT * FROM features WHERE run_id=%s AND cut_index=%s",
                      (run_id, cut_index))
        return self._row_to_features(r) if r else None

    def read_all_features(self, run_id: str) -> list[FeatureRecord]:
        rows = self._all("SELECT * FROM features WHERE run_id=%s ORDER BY cut_index", (run_id,))
        return [self._row_to_features(r) for r in rows]

    def count_features(self, run_id: str) -> int:
        r = self._one("SELECT COUNT(*) AS n FROM features WHERE run_id=%s", (run_id,))
        return int(r["n"]) if r else 0

    # ---------------- RUL ----------------
    def read_all_cuts(self, run_id: str) -> list[Cut]:
        rows = self._all("SELECT * FROM cuts WHERE run_id=%s ORDER BY cut_index", (run_id,))
        return [Cut(r["run_id"], r["cut_index"], r["waveform_key"], _parse_dt(r["ingested_at"]))
                for r in rows]

    def save_cut_results(self, results) -> None:
        self._many(
            "INSERT INTO cut_results VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
            "ON CONFLICT(run_id,cut_index) DO UPDATE SET wear_mean=excluded.wear_mean, "
            "wear_lo=excluded.wear_lo, wear_hi=excluded.wear_hi, wear_true=excluded.wear_true, "
            "rul_median=excluded.rul_median, rul_lo=excluded.rul_lo, rul_hi=excluded.rul_hi, "
            "censored=excluded.censored, computed_at=excluded.computed_at",
            [(r.run_id, r.cut_index, r.wear_mean, r.wear_lo, r.wear_hi, r.wear_true,
              r.rul_median, r.rul_lo, r.rul_hi, r.censored, _dt(r.computed_at)) for r in results])

    def read_cut_results(self, run_id: str) -> list[CutResult]:
        rows = self._all("SELECT * FROM cut_results WHERE run_id=%s ORDER BY cut_index", (run_id,))
        return [CutResult(r["run_id"], r["cut_index"], r["wear_mean"], r["wear_lo"], r["wear_hi"],
                          r["wear_true"], r["rul_median"], r["rul_lo"], r["rul_hi"], r["censored"],
                          _parse_dt(r["computed_at"])) for r in rows]

    def clear_cut_results(self, run_id: str) -> None:
        self._exec("DELETE FROM cut_results WHERE run_id=%s", (run_id,))

    def append_rul(self, p: RULPrediction) -> None:
        self._exec("INSERT INTO rul_predictions "
                   "(run_id,cut_index,rul_median,rul_lower,rul_upper,ci_level,predicted_at) "
                   "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                   (p.run_id, p.cut_index, p.rul_median, p.rul_lower, p.rul_upper,
                    p.ci_level, _dt(p.predicted_at)))

    def read_all_rul(self, run_id: str) -> list[RULPrediction]:
        rows = self._all("SELECT * FROM rul_predictions WHERE run_id=%s ORDER BY cut_index",
                         (run_id,))
        return [RULPrediction(r["run_id"], r["cut_index"], r["rul_median"], r["rul_lower"],
                              r["rul_upper"], r["ci_level"], _parse_dt(r["predicted_at"]))
                for r in rows]

    def clear_rul(self, run_id: str) -> None:
        self._exec("DELETE FROM rul_predictions WHERE run_id=%s", (run_id,))

    # ---------------- Twin state ----------------
    def save_twin_state(self, state: TwinState) -> None:
        self._exec("INSERT INTO twin_state (run_id,cut_index,params_json,particles,updated_at) "
                   "VALUES (%s,%s,%s,%s,%s) "
                   "ON CONFLICT(run_id) DO UPDATE SET cut_index=excluded.cut_index, "
                   "params_json=excluded.params_json, particles=excluded.particles, "
                   "updated_at=excluded.updated_at",
                   (state.run_id, state.cut_index, json.dumps(state.params),
                    state.particles, _dt(state.updated_at)))

    def has_twin_state(self, run_id: str) -> bool:
        return self._one("SELECT 1 AS ok FROM twin_state WHERE run_id=%s", (run_id,)) is not None

    def load_twin_state(self, run_id: str) -> Optional[TwinState]:
        r = self._one("SELECT * FROM twin_state WHERE run_id=%s", (run_id,))
        if r is None:
            return None
        particles = r["particles"]
        return TwinState(r["run_id"], r["cut_index"], json.loads(r["params_json"]),
                         bytes(particles) if particles is not None else None,
                         _parse_dt(r["updated_at"]))

    # ---------------- Wear labels ----------------
    def append_wear_label(self, label: WearLabel) -> None:
        self._exec("INSERT INTO wear_labels (run_id,cut_index,wear_mm) VALUES (%s,%s,%s)",
                   (label.run_id, label.cut_index, label.wear_mm))

    def read_wear_labels(self, run_id: str) -> list[WearLabel]:
        rows = self._all("SELECT * FROM wear_labels WHERE run_id=%s ORDER BY cut_index", (run_id,))
        return [WearLabel(r["run_id"], r["cut_index"], r["wear_mm"]) for r in rows]
