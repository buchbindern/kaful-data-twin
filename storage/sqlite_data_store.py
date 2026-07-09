"""
SQLiteDataStore (M2b) — a DataStore backed by a single SQLite file.

Why SQLite to start (handoff decision #5): embedded (zero infra), ACID (a crash
mid-cut can't corrupt already-committed cuts), O(1) appends, and a CNC machine
emits a cut every few minutes — far under SQLite's write ceiling. Its one-writer
limit is a non-issue for a single machine. Swapping to Postgres/TimescaleDB for a
fleet later touches only this file; nothing above the DataStore interface changes.

Schema notes:
  * Six tables mirror the six domain records; the machine->run->cut hierarchy is
    baked in as foreign keys, so a tool swap can't orphan cuts and you can't
    append a cut for a run that doesn't exist.
  * Datetimes are stored as ISO-8601 UTC TEXT. SQLite has no datetime type; ISO
    strings sort chronologically and round-trip losslessly.
  * A FeatureRecord's dict is stored as JSON TEXT — no schema migration when the
    (still-being-discovered) feature set changes. This is the whole reason
    features live in a dict, not 42 columns.
  * twin_state has ONE row per run (keyed by run_id), overwritten each cut via
    upsert — that's the "persist the posterior between cuts" requirement.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

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
    machine_id   TEXT PRIMARY KEY,
    machine_type TEXT NOT NULL,
    name         TEXT,
    created_at   TEXT NOT NULL,
    owner_id     TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    run_id     TEXT PRIMARY KEY,
    machine_id TEXT NOT NULL REFERENCES machines(machine_id),
    started_at TEXT NOT NULL,
    ended_at   TEXT,
    tool_id    TEXT
);

CREATE TABLE IF NOT EXISTS cuts (
    run_id       TEXT    NOT NULL REFERENCES runs(run_id),
    cut_index    INTEGER NOT NULL,
    waveform_key TEXT    NOT NULL,
    ingested_at  TEXT    NOT NULL,
    PRIMARY KEY (run_id, cut_index)
);

CREATE TABLE IF NOT EXISTS features (
    run_id        TEXT    NOT NULL,
    cut_index     INTEGER NOT NULL,
    features_json TEXT    NOT NULL,
    extracted_at  TEXT    NOT NULL,
    PRIMARY KEY (run_id, cut_index),
    FOREIGN KEY (run_id, cut_index) REFERENCES cuts(run_id, cut_index)
);

CREATE TABLE IF NOT EXISTS rul_predictions (
    run_id       TEXT    NOT NULL,
    cut_index    INTEGER NOT NULL,
    rul_median   REAL    NOT NULL,
    rul_lower    REAL    NOT NULL,
    rul_upper    REAL    NOT NULL,
    ci_level     REAL    NOT NULL,
    predicted_at TEXT    NOT NULL,
    PRIMARY KEY (run_id, cut_index),
    FOREIGN KEY (run_id, cut_index) REFERENCES cuts(run_id, cut_index)
);

CREATE TABLE IF NOT EXISTS twin_state (
    run_id     TEXT PRIMARY KEY REFERENCES runs(run_id),
    cut_index  INTEGER NOT NULL,
    params_json TEXT   NOT NULL,
    particles  BLOB,
    updated_at TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    user_id       TEXT PRIMARY KEY,
    email         TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    token      TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL REFERENCES users(user_id),
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cut_results (
    run_id TEXT NOT NULL, cut_index INTEGER NOT NULL,
    wear_mean REAL NOT NULL, wear_lo REAL NOT NULL, wear_hi REAL NOT NULL,
    wear_true REAL, rul_median REAL, rul_lo REAL, rul_hi REAL,
    censored REAL NOT NULL, computed_at TEXT NOT NULL,
    PRIMARY KEY (run_id, cut_index)
);

CREATE TABLE IF NOT EXISTS wear_labels (
    run_id    TEXT    NOT NULL REFERENCES runs(run_id),
    cut_index INTEGER NOT NULL,
    wear_mm   REAL    NOT NULL,
    PRIMARY KEY (run_id, cut_index)
);
"""


class SQLiteDataStore(DataStore):
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)  # create ./var etc. if absent
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")  # SQLite needs this per-connection
        self._conn.executescript(_SCHEMA)
        try:                                   # migrate pre-existing DBs
            self._conn.execute("ALTER TABLE machines ADD COLUMN owner_id TEXT")
        except sqlite3.OperationalError:
            pass                               # column already present
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ---------------- Machine ----------------
    def create_machine(self, machine: Machine) -> None:
        self._conn.execute(
            "INSERT INTO machines VALUES (?,?,?,?,?)",
            (machine.machine_id, machine.machine_type, machine.name, _dt(machine.created_at),
             machine.owner_id),
        )
        self._conn.commit()

    def rename_machine(self, machine_id: str, name) -> None:
        self._conn.execute("UPDATE machines SET name=? WHERE machine_id=?", (name, machine_id))
        self._conn.commit()

    def get_machine(self, machine_id: str) -> Optional[Machine]:
        row = self._conn.execute(
            "SELECT * FROM machines WHERE machine_id=?", (machine_id,)
        ).fetchone()
        if row is None:
            return None
        return Machine(row["machine_id"], row["machine_type"], row["name"],
                       _parse_dt(row["created_at"]), row["owner_id"])

    # ---------------- Run ----------------
    def create_run(self, run: Run) -> None:
        self._conn.execute(
            "INSERT INTO runs VALUES (?,?,?,?,?)",
            (run.run_id, run.machine_id, _dt(run.started_at),
             _dt(run.ended_at) if run.ended_at else None, run.tool_id),
        )
        self._conn.commit()

    def _row_to_run(self, row: sqlite3.Row) -> Run:
        return Run(row["run_id"], row["machine_id"], _parse_dt(row["started_at"]),
                   _parse_dt(row["ended_at"]), row["tool_id"])

    def get_run(self, run_id: str) -> Optional[Run]:
        row = self._conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return self._row_to_run(row) if row else None

    def get_active_run(self, machine_id: str) -> Optional[Run]:
        row = self._conn.execute(
            "SELECT * FROM runs WHERE machine_id=? AND ended_at IS NULL "
            "ORDER BY started_at DESC LIMIT 1",
            (machine_id,),
        ).fetchone()
        return self._row_to_run(row) if row else None

    def end_run(self, run_id: str, ended_at: datetime) -> None:
        self._conn.execute("UPDATE runs SET ended_at=? WHERE run_id=?",
                           (_dt(ended_at), run_id))
        self._conn.commit()

    def delete_run(self, run_id: str) -> None:
        for t in ("cut_results", "rul_predictions", "twin_state", "features", "wear_labels", "cuts"):
            self._conn.execute(f"DELETE FROM {t} WHERE run_id=?", (run_id,))
        self._conn.execute("DELETE FROM runs WHERE run_id=?", (run_id,))
        self._conn.commit()

    def delete_machine(self, machine_id: str) -> None:
        for r in self.list_runs(machine_id):
            self.delete_run(r.run_id)
        self._conn.execute("DELETE FROM machines WHERE machine_id=?", (machine_id,))
        self._conn.commit()

    def rename_run(self, run_id: str, label) -> None:
        self._conn.execute("UPDATE runs SET tool_id=? WHERE run_id=?", (label, run_id))
        self._conn.commit()

    def list_runs(self, machine_id: str):
        rows = self._conn.execute(
            "SELECT * FROM runs WHERE machine_id=? ORDER BY started_at DESC", (machine_id,)
        ).fetchall()
        return [self._row_to_run(r) for r in rows]

    # ---------------- Auth (users + sessions) ----------------
    def create_user(self, user: User) -> None:
        self._conn.execute("INSERT INTO users VALUES (?,?,?,?)",
            (user.user_id, user.email, user.password_hash, _dt(user.created_at)))
        self._conn.commit()

    def get_user_by_email(self, email: str):
        row = self._conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        return self._row_to_user(row) if row else None

    def get_user(self, user_id: str):
        row = self._conn.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()
        return self._row_to_user(row) if row else None

    def create_session(self, session: Session) -> None:
        self._conn.execute("INSERT INTO sessions VALUES (?,?,?,?)",
            (session.token, session.user_id, _dt(session.created_at), _dt(session.expires_at)))
        self._conn.commit()

    def get_valid_session(self, token: str, now=None):
        from datetime import datetime, timezone
        row = self._conn.execute("SELECT * FROM sessions WHERE token=?", (token,)).fetchone()
        if row is None:
            return None
        sess = Session(row["token"], row["user_id"], _parse_dt(row["created_at"]),
                       _parse_dt(row["expires_at"]))
        if sess.expires_at <= (now or datetime.now(timezone.utc)):
            return None
        return sess

    def delete_session(self, token: str) -> None:
        self._conn.execute("DELETE FROM sessions WHERE token=?", (token,))
        self._conn.commit()

    def delete_user_sessions(self, user_id: str) -> None:
        self._conn.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
        self._conn.commit()

    def delete_expired_sessions(self, now) -> None:
        self._conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (_dt(now),))
        self._conn.commit()

    def _row_to_user(self, row) -> User:
        return User(row["user_id"], row["email"], row["password_hash"],
                    _parse_dt(row["created_at"]))

    def list_machines(self):
        rows = self._conn.execute("SELECT * FROM machines ORDER BY machine_id").fetchall()
        return [Machine(r["machine_id"], r["machine_type"], r["name"], _parse_dt(r["created_at"]),
                        r["owner_id"])
                for r in rows]

    # ---------------- Cut ----------------
    def append_cut(self, cut: Cut) -> None:
        self._conn.execute(
            "INSERT INTO cuts VALUES (?,?,?,?)",
            (cut.run_id, cut.cut_index, cut.waveform_key, _dt(cut.ingested_at)),
        )
        self._conn.commit()

    def append_cuts_bulk(self, cuts) -> None:
        self._conn.executemany(
            "INSERT INTO cuts VALUES (?,?,?,?)",
            [(c.run_id, c.cut_index, c.waveform_key, _dt(c.ingested_at)) for c in cuts])
        self._conn.commit()

    def get_cut(self, run_id: str, cut_index: int) -> Optional[Cut]:
        row = self._conn.execute(
            "SELECT * FROM cuts WHERE run_id=? AND cut_index=?", (run_id, cut_index)
        ).fetchone()
        if row is None:
            return None
        return Cut(row["run_id"], row["cut_index"], row["waveform_key"],
                   _parse_dt(row["ingested_at"]))

    # ---------------- Features ----------------
    def append_features(self, record: FeatureRecord) -> None:
        self._conn.execute(
            "INSERT INTO features VALUES (?,?,?,?)",
            (record.run_id, record.cut_index, json.dumps(record.features),
             _dt(record.extracted_at)),
        )
        self._conn.commit()

    def append_features_bulk(self, records) -> None:
        self._conn.executemany(
            "INSERT INTO features VALUES (?,?,?,?)",
            [(r.run_id, r.cut_index, json.dumps(r.features), _dt(r.extracted_at)) for r in records])
        self._conn.commit()

    def _row_to_features(self, row: sqlite3.Row) -> FeatureRecord:
        return FeatureRecord(row["run_id"], row["cut_index"],
                             json.loads(row["features_json"]),
                             _parse_dt(row["extracted_at"]))

    def get_features(self, run_id: str, cut_index: int) -> Optional[FeatureRecord]:
        row = self._conn.execute(
            "SELECT * FROM features WHERE run_id=? AND cut_index=?", (run_id, cut_index)
        ).fetchone()
        return self._row_to_features(row) if row else None

    def read_all_features(self, run_id: str) -> list[FeatureRecord]:
        rows = self._conn.execute(
            "SELECT * FROM features WHERE run_id=? ORDER BY cut_index", (run_id,)
        ).fetchall()
        return [self._row_to_features(r) for r in rows]

    def count_features(self, run_id: str) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM features WHERE run_id=?", (run_id,)
        ).fetchone()[0]

    # ---------------- RUL predictions ----------------
    def read_all_cuts(self, run_id: str) -> list[Cut]:
        rows = self._conn.execute(
            "SELECT * FROM cuts WHERE run_id=? ORDER BY cut_index", (run_id,)).fetchall()
        return [Cut(r["run_id"], r["cut_index"], r["waveform_key"], _parse_dt(r["ingested_at"]))
                for r in rows]

    def save_cut_results(self, results) -> None:
        self._conn.executemany(
            "INSERT OR REPLACE INTO cut_results VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [(r.run_id, r.cut_index, r.wear_mean, r.wear_lo, r.wear_hi, r.wear_true,
              r.rul_median, r.rul_lo, r.rul_hi, r.censored, _dt(r.computed_at)) for r in results])
        self._conn.commit()

    def read_cut_results(self, run_id: str) -> list[CutResult]:
        rows = self._conn.execute(
            "SELECT * FROM cut_results WHERE run_id=? ORDER BY cut_index", (run_id,)).fetchall()
        return [CutResult(r["run_id"], r["cut_index"], r["wear_mean"], r["wear_lo"], r["wear_hi"],
                          r["wear_true"], r["rul_median"], r["rul_lo"], r["rul_hi"], r["censored"],
                          _parse_dt(r["computed_at"])) for r in rows]

    def clear_cut_results(self, run_id: str) -> None:
        self._conn.execute("DELETE FROM cut_results WHERE run_id=?", (run_id,))
        self._conn.commit()

    def append_rul(self, prediction: RULPrediction) -> None:
        self._conn.execute(
            "INSERT INTO rul_predictions VALUES (?,?,?,?,?,?,?)",
            (prediction.run_id, prediction.cut_index, prediction.rul_median,
             prediction.rul_lower, prediction.rul_upper, prediction.ci_level,
             _dt(prediction.predicted_at)),
        )
        self._conn.commit()

    def read_all_rul(self, run_id: str) -> list[RULPrediction]:
        rows = self._conn.execute(
            "SELECT * FROM rul_predictions WHERE run_id=? ORDER BY cut_index", (run_id,)
        ).fetchall()
        return [
            RULPrediction(r["run_id"], r["cut_index"], r["rul_median"], r["rul_lower"],
                          r["rul_upper"], r["ci_level"], _parse_dt(r["predicted_at"]))
            for r in rows
        ]

    # ---------------- Twin state ----------------
    def save_twin_state(self, state: TwinState) -> None:
        # One row per run, overwritten each cut.
        self._conn.execute(
            "INSERT INTO twin_state VALUES (?,?,?,?,?) "
            "ON CONFLICT(run_id) DO UPDATE SET "
            "cut_index=excluded.cut_index, params_json=excluded.params_json, "
            "particles=excluded.particles, updated_at=excluded.updated_at",
            (state.run_id, state.cut_index, json.dumps(state.params),
             state.particles, _dt(state.updated_at)),
        )
        self._conn.commit()

    def has_twin_state(self, run_id: str) -> bool:
        return self._conn.execute(
            "SELECT 1 FROM twin_state WHERE run_id=?", (run_id,)
        ).fetchone() is not None

    def load_twin_state(self, run_id: str) -> Optional[TwinState]:
        row = self._conn.execute(
            "SELECT * FROM twin_state WHERE run_id=?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        particles = row["particles"]
        return TwinState(row["run_id"], row["cut_index"], json.loads(row["params_json"]),
                         bytes(particles) if particles is not None else None,
                         _parse_dt(row["updated_at"]))

    # ---------------- Wear labels (reference/validation only) ----------------
    def append_wear_label(self, label: WearLabel) -> None:
        self._conn.execute(
            "INSERT INTO wear_labels VALUES (?,?,?)",
            (label.run_id, label.cut_index, label.wear_mm),
        )
        self._conn.commit()

    def read_wear_labels(self, run_id: str) -> list[WearLabel]:
        rows = self._conn.execute(
            "SELECT * FROM wear_labels WHERE run_id=? ORDER BY cut_index", (run_id,)
        ).fetchall()
        return [WearLabel(r["run_id"], r["cut_index"], r["wear_mm"]) for r in rows]

    def clear_rul(self, run_id: str) -> None:
        self._conn.execute("DELETE FROM rul_predictions WHERE run_id=?", (run_id,))
        self._conn.commit()
