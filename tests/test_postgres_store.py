"""Parity tests: PostgresDataStore must behave identically to SQLiteDataStore.
Skipped unless a Postgres is reachable at $KAFUL_TEST_PG (or DATABASE_URL)."""

import os
import uuid
from datetime import datetime, timezone, timedelta

import numpy as np
import pytest

from domain.models import (Machine, Run, Cut, FeatureRecord, RULPrediction, TwinState,
                            WearLabel, User, Session)
from storage import SQLiteDataStore

_DSN = os.environ.get("KAFUL_TEST_PG") or os.environ.get("DATABASE_URL")

pytestmark = pytest.mark.skipif(not _DSN, reason="no Postgres DSN in env")


def _fresh_pg():
    import psycopg
    from storage.postgres_data_store import PostgresDataStore
    with psycopg.connect(_DSN, autocommit=True) as c:
        c.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
    return PostgresDataStore(_DSN)


def _exercise(ds, now):
    """Run a full round-trip through every method; return a snapshot dict to compare."""
    ds.create_machine(Machine("phm2010", "phm2010_milling"))                 # system
    ds.create_machine(Machine("m1", "cnc", name="Mine", owner_id="u1"))       # owned
    ds.create_run(Run("c1", "phm2010", now))
    ds.append_cut(Cut("c1", 1, "k1", now))
    ds.append_features(FeatureRecord("c1", 1, {"vibration_x_mean_abs": 3.14}, now))
    ds.append_wear_label(WearLabel("c1", 1, 0.055))
    ds.append_rul(RULPrediction("c1", 1, 120.0, 90.0, 200.0, 0.9, now))
    ds.save_twin_state(TwinState("c1", 1, {"a": 1, "feature_name": "vibration_x_mean_abs"},
                                 b"\x00\x01\x02particles", now))
    ds.save_twin_state(TwinState("c1", 2, {"a": 2}, b"\xff\xfe", now))         # upsert overwrite
    # auth
    real = datetime.now(timezone.utc)                 # session validity is vs real now
    ds.create_user(User("u1", "a@b.com", "hash", now))
    ds.create_session(Session("tok", "u1", real, real + timedelta(days=1)))
    ds.create_session(Session("old", "u1", real - timedelta(days=2), real - timedelta(days=1)))

    return {
        "machine_sys_owner": ds.get_machine("phm2010").owner_id,
        "machine_owned_owner": ds.get_machine("m1").owner_id,
        "machines": [m.machine_id for m in ds.list_machines()],
        "run_machine": ds.get_run("c1").machine_id,
        "active_run": ds.get_active_run("phm2010").run_id,
        "cut_key": ds.get_cut("c1", 1).waveform_key,
        "features": ds.get_features("c1", 1).features,
        "n_features": len(ds.read_all_features("c1")),
        "labels": [l.wear_mm for l in ds.read_wear_labels("c1")],
        "rul_median": ds.read_all_rul("c1")[0].rul_median,
        "twin_cut": ds.load_twin_state("c1").cut_index,          # 2 after upsert
        "twin_particles": ds.load_twin_state("c1").particles,    # b"\xff\xfe"
        "user_by_email": ds.get_user_by_email("a@b.com").user_id,
        "valid_session": ds.get_valid_session("tok").user_id,
        "expired_session": ds.get_valid_session("old"),          # None
    }


def test_postgres_roundtrip_and_parity(tmp_path):
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    pg = _fresh_pg()
    sq = SQLiteDataStore(tmp_path / "kaful.db")
    pg_snap = _exercise(pg, now)
    sq_snap = _exercise(sq, now)
    assert pg_snap == sq_snap, f"\nPG: {pg_snap}\nSQLite: {sq_snap}"
    # spot-check the values that matter
    assert pg_snap["twin_cut"] == 2 and pg_snap["twin_particles"] == b"\xff\xfe"  # upsert + bytea
    assert pg_snap["expired_session"] is None
    assert pg_snap["valid_session"] == "u1"
    pg.close(); sq.close()


def test_postgres_delete_sessions():
    now = datetime.now(timezone.utc)
    pg = _fresh_pg()
    pg.create_machine(Machine("s", "t"))
    pg.create_user(User("u", "x@y.com", "h", now))
    pg.create_session(Session("t1", "u", now, now + timedelta(days=1)))
    pg.delete_session("t1")
    assert pg.get_valid_session("t1") is None
    pg.close()
