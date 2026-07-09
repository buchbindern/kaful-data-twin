"""
Phase G rung-1 fix — the two per-cut store queries the live ingest path now uses.

count_features replaces `len(read_all_features())` (O(n)/cut) for cut_index
assignment; has_twin_state replaces a full twin_state blob load used only as a
null-check. Both must agree with the slow methods they replace.
"""
from __future__ import annotations

from domain.models import Machine, Run, Cut, FeatureRecord, TwinState
from storage.sqlite_data_store import SQLiteDataStore


def _store(tmp_path):
    s = SQLiteDataStore(tmp_path / "t.db")
    s.create_machine(Machine("m1", "cnc_milling", name="t", owner_id=None))
    s.create_run(Run("r1", "m1", tool_id=None))
    return s


def test_count_features_matches_read_all(tmp_path):
    s = _store(tmp_path)
    assert s.count_features("r1") == 0
    for i in range(1, 6):
        s.append_cut(Cut("r1", i, f"k{i}"))
        s.append_features(FeatureRecord("r1", i, {"rms": float(i)}))
        assert s.count_features("r1") == i == len(s.read_all_features("r1"))
    # unknown run -> 0, never an error
    assert s.count_features("nope") == 0


def test_has_twin_state_matches_load(tmp_path):
    s = _store(tmp_path)
    assert s.has_twin_state("r1") is False
    assert s.has_twin_state("r1") == (s.load_twin_state("r1") is not None)
    s.save_twin_state(TwinState("r1", 1, params={"feature_name": "rms"}, particles=b"blob"))
    assert s.has_twin_state("r1") is True
    assert s.has_twin_state("r1") == (s.load_twin_state("r1") is not None)
