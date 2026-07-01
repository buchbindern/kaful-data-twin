"""M2b: tests for SQLiteDataStore. Each test gets a fresh DB file under a temp dir
(pytest's tmp_path), so tests are isolated and nothing leaks between them."""

from datetime import datetime, timezone

import pytest

from domain import Machine, Run, Cut, FeatureRecord, RULPrediction, TwinState
from storage import SQLiteDataStore


@pytest.fixture
def store(tmp_path):
    s = SQLiteDataStore(tmp_path / "kaful.db")
    yield s
    s.close()


def _seed_machine_and_run(store, run_id="c1"):
    store.create_machine(Machine("phm2010", "phm2010_milling"))
    store.create_run(Run(run_id, "phm2010"))


# ---------------- Machine ----------------
def test_machine_roundtrip(store):
    store.create_machine(Machine("phm2010", "phm2010_milling", name="Kern Micro"))
    got = store.get_machine("phm2010")
    assert got.machine_id == "phm2010"
    assert got.machine_type == "phm2010_milling"
    assert got.name == "Kern Micro"

def test_get_missing_machine_is_none(store):
    assert store.get_machine("nope") is None


# ---------------- Run + active-run logic ----------------
def test_active_run_lifecycle(store):
    store.create_machine(Machine("phm2010", "phm2010_milling"))
    store.create_run(Run("c1", "phm2010"))
    assert store.get_active_run("phm2010").run_id == "c1"

    store.end_run("c1", datetime.now(timezone.utc))
    assert store.get_run("c1").ended_at is not None
    assert store.get_active_run("phm2010") is None  # no active run after tool removed

    store.create_run(Run("c4", "phm2010"))
    assert store.get_active_run("phm2010").run_id == "c4"  # new tool is now active


# ---------------- Cut ----------------
def test_cut_roundtrip(store):
    _seed_machine_and_run(store)
    store.append_cut(Cut("c1", 1, "phm2010/c1/000001.npy.gz"))
    got = store.get_cut("c1", 1)
    assert got.cut_index == 1
    assert got.waveform_key.endswith("000001.npy.gz")

def test_cut_for_missing_run_rejected(store):
    # foreign key: can't append a cut whose run doesn't exist
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):
        store.append_cut(Cut("ghost_run", 1, "k"))

def test_duplicate_cut_rejected(store):
    _seed_machine_and_run(store)
    store.append_cut(Cut("c1", 1, "k"))
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):
        store.append_cut(Cut("c1", 1, "k2"))  # same (run, cut_index) written twice


# ---------------- Features ----------------
def test_features_roundtrip_and_ordering(store):
    _seed_machine_and_run(store)
    for i in (3, 1, 2):  # insert out of order on purpose
        store.append_cut(Cut("c1", i, f"k{i}"))
        store.append_features(FeatureRecord("c1", i, {"force_z_rms": float(i), "ae_kurt": 4.0}))
    got = store.get_features("c1", 2)
    assert got.features == {"force_z_rms": 2.0, "ae_kurt": 4.0}  # dict survives JSON round-trip
    all_feats = store.read_all_features("c1")
    assert [f.cut_index for f in all_feats] == [1, 2, 3]  # returned ordered by cut_index


# ---------------- RUL ----------------
def test_rul_roundtrip_and_ordering(store):
    _seed_machine_and_run(store)
    for i in (2, 1):
        store.append_cut(Cut("c1", i, f"k{i}"))
        store.append_rul(RULPrediction("c1", i, 40.0 - i, 30.0, 55.0))
    preds = store.read_all_rul("c1")
    assert [p.cut_index for p in preds] == [1, 2]
    assert preds[0].ci_level == 0.9


# ---------------- Twin state ----------------
def test_twin_state_upsert_and_blob(store):
    _seed_machine_and_run(store)
    store.save_twin_state(TwinState("c1", 1, params={"exp": 1.5}, particles=b"\x00\x01cloud"))
    s1 = store.load_twin_state("c1")
    assert s1.cut_index == 1
    assert s1.params == {"exp": 1.5}
    assert s1.particles == b"\x00\x01cloud"  # BLOB round-trips as bytes

    # overwrite: same run, next cut -> one row, updated in place
    store.save_twin_state(TwinState("c1", 2, params={"exp": 1.7}, particles=b"newcloud"))
    s2 = store.load_twin_state("c1")
    assert s2.cut_index == 2
    assert s2.params == {"exp": 1.7}
    assert s2.particles == b"newcloud"

def test_load_missing_twin_state_is_none(store):
    assert store.load_twin_state("nope") is None


# ---------------- Durability across reopen ----------------
def test_data_survives_reopen(tmp_path):
    db = tmp_path / "kaful.db"
    s1 = SQLiteDataStore(db)
    s1.create_machine(Machine("phm2010", "phm2010_milling"))
    s1.create_run(Run("c1", "phm2010"))
    s1.append_cut(Cut("c1", 1, "k"))
    s1.close()

    s2 = SQLiteDataStore(db)  # reopen same file
    assert s2.get_machine("phm2010") is not None
    assert s2.get_cut("c1", 1).waveform_key == "k"
    s2.close()
