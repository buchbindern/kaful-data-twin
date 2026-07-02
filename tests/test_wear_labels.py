"""M5a: PHM wear-file loading + wear-label storage.

The real-data test reads your actual c1_wear.csv and SKIPS if absent."""

from pathlib import Path

import pytest

from domain import Machine, Run, WearLabel
from storage import SQLiteDataStore
from datasets import load_wear_labels


# ---------------- loader ----------------
def test_load_wear_labels_skips_header_and_converts_units(tmp_path):
    f = tmp_path / "c1_wear.csv"
    f.write_text(
        "cut,flute_1,flute_2,flute_3\n"   # header (non-numeric first field -> skipped)
        "1,100,110,120\n"                 # mean 110 (1e-3 mm) -> 0.110 mm
        "2,200,200,200\n"                 # mean 200 (1e-3 mm) -> 0.200 mm (== threshold)
    )
    labels = load_wear_labels(f)
    assert labels == [(1, 0.110), (2, 0.200)]

def test_load_wear_labels_sorts_by_cut(tmp_path):
    f = tmp_path / "w.csv"
    f.write_text("3,150,150,150\n1,50,50,50\n2,100,100,100\n")  # no header
    assert [c for c, _ in load_wear_labels(f)] == [1, 2, 3]


# ---------------- storage ----------------
def test_wear_label_roundtrip(tmp_path):
    ds = SQLiteDataStore(tmp_path / "kaful.db")
    ds.create_machine(Machine("phm2010", "phm2010_milling"))
    ds.create_run(Run("c1", "phm2010"))
    for cut, wear in [(1, 0.05), (2, 0.08), (3, 0.12)]:
        ds.append_wear_label(WearLabel("c1", cut, wear))
    labels = ds.read_wear_labels("c1")
    assert [(l.cut_index, l.wear_mm) for l in labels] == [(1, 0.05), (2, 0.08), (3, 0.12)]
    ds.close()

def test_wear_label_for_missing_run_rejected(tmp_path):
    import sqlite3
    ds = SQLiteDataStore(tmp_path / "kaful.db")
    with pytest.raises(sqlite3.IntegrityError):
        ds.append_wear_label(WearLabel("ghost", 1, 0.1))  # FK: run must exist
    ds.close()


# ---------------- real PHM data (skips if absent) ----------------
C1_WEAR = Path("data/phm2010/c1/c1_wear.csv")

@pytest.mark.skipif(not C1_WEAR.exists(), reason="PHM 2010 c1 wear file not present")
def test_real_c1_wear_is_plausible():
    labels = load_wear_labels(C1_WEAR)
    assert 300 <= len(labels) <= 320                 # ~315 cuts
    cuts = [c for c, _ in labels]
    assert cuts == sorted(cuts) and cuts[0] >= 1
    wears = [w for _, w in labels]
    assert all(0.0 <= w < 0.5 for w in wears)         # plausible mm range
    assert wears[-1] > wears[0]                        # tool wears over its life
