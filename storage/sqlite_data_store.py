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
    Machine, Run, Cut, FeatureRecord, RULPrediction, TwinState,
)
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
    created_at   TEXT NOT NULL
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
"""


class SQLiteDataStore(DataStore):
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")  # SQLite needs this per-connection
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ---------------- Machine ----------------
    def create_machine(self, machine: Machine) -> None:
        self._conn.execute(
            "INSERT INTO machines VALUES (?,?,?,?)",
            (machine.machine_id, machine.machine_type, machine.name, _dt(machine.created_at)),
        )
        self._conn.commit()

    def get_machine(self, machine_id: str) -> Optional[Machine]:
        row = self._conn.execute(
            "SELECT * FROM machines WHERE machine_id=?", (machine_id,)
        ).fetchone()
        if row is None:
            return None
        return Machine(row["machine_id"], row["machine_type"], row["name"],
                       _parse_dt(row["created_at"]))

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

    # ---------------- Cut ----------------
    def append_cut(self, cut: Cut) -> None:
        self._conn.execute(
            "INSERT INTO cuts VALUES (?,?,?,?)",
            (cut.run_id, cut.cut_index, cut.waveform_key, _dt(cut.ingested_at)),
        )
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

    # ---------------- RUL predictions ----------------
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
