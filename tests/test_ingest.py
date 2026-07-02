"""M4: codec round-trip, the ingest handler wiring, and the replay driver.
Uses real storage backends on a temp dir + synthetic waveforms, so it needs no
PHM data and exercises the full 'it's alive' path."""

import numpy as np
import pytest

from domain import Machine, Run
from storage import SQLiteDataStore, FilesystemObjectStore
from features import FeatureExtractor
from datasets import PHM_CHANNELS, iter_cut_files
from ingest import encode_waveform, decode_waveform, IngestHandler, waveform_key, replay_run
from twin import StubTwin

RNG = np.random.default_rng(0)


# ---------------- codec ----------------
def test_codec_roundtrip_preserves_shape_and_values():
    wf = RNG.standard_normal((2000, 7))
    back = decode_waveform(encode_waveform(wf))
    assert back.shape == wf.shape
    assert np.allclose(back, wf, atol=1e-4)  # float32 storage tolerance


# ---------------- handler wiring ----------------
@pytest.fixture
def wired(tmp_path):
    ds = SQLiteDataStore(tmp_path / "kaful.db")
    os_ = FilesystemObjectStore(tmp_path / "obj")
    ds.create_machine(Machine("phm2010", "phm2010_milling"))
    ds.create_run(Run("c1", "phm2010"))
    handler = IngestHandler(ds, os_, FeatureExtractor(PHM_CHANNELS), StubTwin(rul_median=50.0))
    yield ds, os_, handler
    ds.close()

def test_ingest_cut_fills_every_table(wired):
    ds, os_, handler = wired
    raw = encode_waveform(RNG.standard_normal((1000, 7)))
    rul = handler.ingest_cut("phm2010", "c1", 1, raw)

    assert rul.rul_median == 50.0                      # came from the stub
    assert ds.get_cut("c1", 1) is not None             # cut row written
    assert os_.exists(waveform_key("phm2010", "c1", 1))# raw blob stored
    assert len(ds.read_all_features("c1")) == 1        # features written
    assert len(ds.read_all_rul("c1")) == 1             # rul written
    back = decode_waveform(os_.get(waveform_key("phm2010", "c1", 1)))
    assert back.shape == (1000, 7)                      # blob round-trips

def test_features_are_the_42_expected(wired):
    ds, _, handler = wired
    handler.ingest_cut("phm2010", "c1", 1, encode_waveform(RNG.standard_normal((1000, 7))))
    feats = ds.get_features("c1", 1).features
    assert len(feats) == 42
    assert "force_z_rms" in feats


# ---------------- replay driver ----------------
def _make_synthetic_phm(tmp_path, n_cuts=3, rows=500):
    folder = tmp_path / "c1" / "c1"
    folder.mkdir(parents=True)
    for n in range(1, n_cuts + 1):
        np.savetxt(folder / f"c_1_{n:03d}.csv", RNG.standard_normal((rows, 7)), delimiter=",")
    return folder

def test_iter_cut_files_orders_by_index(tmp_path):
    folder = _make_synthetic_phm(tmp_path, n_cuts=3)
    assert [c for c, _ in iter_cut_files(folder)] == [1, 2, 3]

def test_replay_fills_run_and_is_idempotent(tmp_path):
    folder = _make_synthetic_phm(tmp_path, n_cuts=3)
    ds = SQLiteDataStore(tmp_path / "kaful.db")
    os_ = FilesystemObjectStore(tmp_path / "obj")
    handler = IngestHandler(ds, os_, FeatureExtractor(PHM_CHANNELS), StubTwin())

    replay_run(handler, ds, cut_files=iter_cut_files(folder), machine_id="phm2010",
               run_id="c1", machine_type="phm2010_milling", progress=False)
    assert len(ds.read_all_rul("c1")) == 3

    replay_run(handler, ds, cut_files=iter_cut_files(folder), machine_id="phm2010",
               run_id="c1", machine_type="phm2010_milling", progress=False)
    assert len(ds.read_all_rul("c1")) == 3   # idempotent: still 3
    ds.close()
