"""M9: HTTP API over the ingest handler, driven in-process by FastAPI's TestClient
(no live server). Proves a real waveform round-trips: POST bytes -> handler -> twin
-> RUL, and that the error paths behave."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from domain import Machine, Run, Cut, FeatureRecord, WearLabel
from storage import SQLiteDataStore
from twin import build_twin
from ingest import encode_waveform
from api import create_app

RNG = np.random.default_rng(0)


def _seed_store_with_twin(store_dir, n=200):
    ds = SQLiteDataStore(store_dir / "kaful.db")
    ds.create_machine(Machine("phm2010", "phm2010_milling"))
    ds.create_run(Run("c1", "phm2010"))
    for cut in range(1, n + 1):
        wear = 0.05 + 0.12 * (cut / n) ** 2.5
        fz = 1521 * wear ** 2.0 + RNG.normal(0, 2.0)
        ds.append_cut(Cut("c1", cut, f"k{cut}"))
        ds.append_features(FeatureRecord("c1", cut, {"force_z_rms": fz}))
        ds.append_wear_label(WearLabel("c1", cut, wear))
    ds.save_twin_state(build_twin(ds, "c1", n_particles=1000))
    ds.close()


@pytest.fixture
def client(tmp_path):
    _seed_store_with_twin(tmp_path)
    return TestClient(create_app(store_dir=str(tmp_path)))


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"

def test_post_cut_returns_rul(client):
    wf = RNG.standard_normal((1000, 7))
    r = client.post("/machines/phm2010/runs/c1/cuts", content=encode_waveform(wf))
    assert r.status_code == 200
    body = r.json()
    assert "rul_median" in body and body["rul_lower"] <= body["rul_median"] <= body["rul_upper"]

def test_posted_cut_appears_in_rul_timeseries(client):
    wf = RNG.standard_normal((1000, 7))
    client.post("/machines/phm2010/runs/c1/cuts", content=encode_waveform(wf))
    r = client.get("/machines/phm2010/runs/c1/rul")
    assert r.status_code == 200 and r.json()["n"] >= 1

def test_empty_body_is_400(client):
    r = client.post("/machines/phm2010/runs/c1/cuts", content=b"")
    assert r.status_code == 400

def test_unknown_run_is_404(client):
    wf = encode_waveform(RNG.standard_normal((500, 7)))
    r = client.post("/machines/phm2010/runs/ghost/cuts", content=wf)
    assert r.status_code == 404

def test_run_without_twin_is_409(tmp_path):
    ds = SQLiteDataStore(tmp_path / "kaful.db")
    ds.create_machine(Machine("phm2010", "phm2010_milling"))
    ds.create_run(Run("c9", "phm2010"))
    ds.close()
    client = TestClient(create_app(store_dir=str(tmp_path)))
    r = client.post("/machines/phm2010/runs/c9/cuts", content=encode_waveform(RNG.standard_normal((500, 7))))
    assert r.status_code == 409
