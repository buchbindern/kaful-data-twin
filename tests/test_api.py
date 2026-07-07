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
        ds.append_features(FeatureRecord("c1", cut, {"force_z_rms": fz, "vibration_x_mean_abs": fz}))
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

def _live_run(client, run_id="live-tool"):
    r = client.post("/machines/phm2010/runs",
                    json={"run_id": run_id, "reference_run_id": "c1"})
    assert r.status_code == 200
    return run_id


def test_post_cut_returns_rul(client):
    run = _live_run(client)
    wf = RNG.standard_normal((1000, 7))
    r = client.post(f"/machines/phm2010/runs/{run}/cuts", content=encode_waveform(wf))
    assert r.status_code == 200
    body = r.json()
    assert "rul_median" in body and body["rul_lower"] <= body["rul_median"] <= body["rul_upper"]

def test_posted_cut_appears_in_rul_timeseries(client):
    run = _live_run(client)
    wf = RNG.standard_normal((1000, 7))
    client.post(f"/machines/phm2010/runs/{run}/cuts", content=encode_waveform(wf))
    r = client.get(f"/machines/phm2010/runs/{run}/rul")
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


def test_start_new_run_endpoint(client):
    r = client.post("/machines/phm2010/runs",
                    json={"run_id": "run2", "reference_run_id": "c1", "tool_id": "T-02"})
    assert r.status_code == 200 and r.json()["status"] == "active"
    import numpy as np
    from ingest import encode_waveform
    wf = encode_waveform(np.random.default_rng(1).standard_normal((800, 7)))
    r2 = client.post("/machines/phm2010/runs/run2/cuts", content=wf)
    assert r2.status_code == 200 and r2.json()["cut_index"] == 1   # cut_index reset

def test_start_run_bad_reference_is_400(client):
    r = client.post("/machines/phm2010/runs",
                    json={"run_id": "run3", "reference_run_id": "ghost"})
    assert r.status_code == 400


def test_root_serves_dashboard(client):
    r = client.get("/")
    assert r.status_code == 200 and "Tool Condition Monitoring" in r.text

def test_runs_list_endpoint(client):
    r = client.get("/machines/phm2010/runs")
    assert r.status_code == 200
    runs = r.json()["runs"]
    assert any(x["run_id"] == "c1" and x["has_labels"] for x in runs)


def test_all_runs_endpoint(client):
    r = client.get("/runs")
    assert r.status_code == 200
    runs = r.json()["runs"]
    assert any(x["run_id"] == "c1" and x["machine_id"] == "phm2010" for x in runs)


def test_ingest_rejects_labeled_reference_run(client):
    wf = encode_waveform(RNG.standard_normal((500, 7)))
    r = client.post("/machines/phm2010/runs/c1/cuts", content=wf)
    assert r.status_code == 409

def test_ingest_rejects_ended_run(client):
    client.post("/machines/phm2010/runs", json={"run_id": "r-old", "reference_run_id": "c1"})
    client.post("/machines/phm2010/runs", json={"run_id": "r-new", "reference_run_id": "c1"})
    wf = encode_waveform(RNG.standard_normal((500, 7)))
    r = client.post("/machines/phm2010/runs/r-old/cuts", content=wf)
    assert r.status_code == 409

def test_uploads_get_isolated_unique_runs(client):
    import numpy as np
    def files():
        return [("files", (f"c_9_{i:03d}.csv",
                 "\n".join(",".join(f"{v:.3f}" for v in row)
                           for row in np.random.default_rng(i).standard_normal((120, 7))),
                 "text/csv")) for i in range(1, 3)]
    r1 = client.post("/analyze", files=files()).json()
    r2 = client.post("/analyze", files=files()).json()
    assert r1["run_id"] != r2["run_id"] and r1["machine_id"] == "uploads"
    assert client.get("/machines/phm2010/runs/c1/rul").status_code == 200


def test_ingest_rejects_labeled_reference_run(client):
    wf = encode_waveform(RNG.standard_normal((500, 7)))
    r = client.post("/machines/phm2010/runs/c1/cuts", content=wf)
    assert r.status_code == 409

def test_ingest_rejects_ended_run(client):
    client.post("/machines/phm2010/runs", json={"run_id": "r-old", "reference_run_id": "c1"})
    client.post("/machines/phm2010/runs", json={"run_id": "r-new", "reference_run_id": "c1"})
    wf = encode_waveform(RNG.standard_normal((500, 7)))
    r = client.post("/machines/phm2010/runs/r-old/cuts", content=wf)
    assert r.status_code == 409

def test_uploads_get_isolated_unique_runs(client):
    import numpy as np
    def files():
        return [("files", (f"c_9_{i:03d}.csv",
                 "\n".join(",".join(f"{v:.3f}" for v in row)
                           for row in np.random.default_rng(i).standard_normal((120, 7))),
                 "text/csv")) for i in range(1, 3)]
    r1 = client.post("/analyze", files=files()).json()
    r2 = client.post("/analyze", files=files()).json()
    assert r1["run_id"] != r2["run_id"] and r1["machine_id"] == "uploads"
    assert client.get("/machines/phm2010/runs/c1/rul").status_code == 200
