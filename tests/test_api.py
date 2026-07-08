"""API tests under multi-tenancy: guest reads of system data, authed writes to
owned machines, and tenant isolation."""

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
    """A SYSTEM machine (owner_id None) holding the labeled reference run c1."""
    ds = SQLiteDataStore(store_dir / "kaful.db")
    ds.create_machine(Machine("phm2010", "phm2010_milling"))       # owner_id None -> system
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


def _signup(client, email="u@test.com"):
    r = client.post("/auth/signup", json={"email": email, "password": "password123"})
    assert r.status_code == 200          # cookies now persist on this client
    return r


def _owned_machine(client, email="u@test.com"):
    _signup(client, email)
    return client.post("/machines", json={"name": "My CNC"}).json()["machine_id"]


def _live_run(client, machine_id, run_id="live-tool"):
    r = client.post(f"/machines/{machine_id}/runs",
                    json={"run_id": run_id, "reference_run_id": "c1"})
    assert r.status_code == 200, r.text
    return run_id


# ---------------- reads (guest, system data) ----------------

def test_health(client):
    assert client.get("/health").json()["status"] == "ok"

def test_root_serves_dashboard(client):
    r = client.get("/")
    assert r.status_code == 200 and "Tool Condition Monitoring" in r.text

def test_guest_sees_system_runs(client):
    r = client.get("/runs")
    assert r.status_code == 200
    assert any(x["run_id"] == "c1" and x["machine_id"] == "phm2010" for x in r.json()["runs"])

def test_guest_can_list_system_machine_runs(client):
    r = client.get("/machines/phm2010/runs")
    assert r.status_code == 200
    assert any(x["run_id"] == "c1" and x["has_labels"] for x in r.json()["runs"])


# ---------------- writes (authed, owned machine) ----------------

def test_post_cut_returns_rul(client):
    mid = _owned_machine(client)
    run = _live_run(client, mid)
    r = client.post(f"/machines/{mid}/runs/{run}/cuts", content=encode_waveform(RNG.standard_normal((1000, 7))))
    assert r.status_code == 200
    body = r.json()
    assert body["rul_lower"] <= body["rul_median"] <= body["rul_upper"]

def test_posted_cut_appears_in_rul_timeseries(client):
    mid = _owned_machine(client)
    run = _live_run(client, mid)
    client.post(f"/machines/{mid}/runs/{run}/cuts", content=encode_waveform(RNG.standard_normal((1000, 7))))
    r = client.get(f"/machines/{mid}/runs/{run}/rul")
    assert r.status_code == 200 and r.json()["n"] >= 1

def test_empty_body_is_400(client):
    mid = _owned_machine(client)
    run = _live_run(client, mid)
    assert client.post(f"/machines/{mid}/runs/{run}/cuts", content=b"").status_code == 400

def test_unknown_run_is_404(client):
    mid = _owned_machine(client)
    r = client.post(f"/machines/{mid}/runs/ghost/cuts", content=encode_waveform(RNG.standard_normal((500, 7))))
    assert r.status_code == 404

def test_run_without_twin_is_409(client):
    mid = _owned_machine(client)
    client.app.state.data_store.create_run(Run("no-twin", mid))     # run with no twin
    r = client.post(f"/machines/{mid}/runs/no-twin/cuts", content=encode_waveform(RNG.standard_normal((500, 7))))
    assert r.status_code == 409

def test_start_new_run_endpoint(client):
    mid = _owned_machine(client)
    r = client.post(f"/machines/{mid}/runs",
                    json={"run_id": "run2", "reference_run_id": "c1", "tool_id": "T-02"})
    assert r.status_code == 200 and r.json()["status"] == "active"
    wf = encode_waveform(np.random.default_rng(1).standard_normal((800, 7)))
    r2 = client.post(f"/machines/{mid}/runs/run2/cuts", content=wf)
    assert r2.status_code == 200 and r2.json()["cut_index"] == 1

def test_start_run_bad_reference_is_400(client):
    mid = _owned_machine(client)
    r = client.post(f"/machines/{mid}/runs", json={"run_id": "run3", "reference_run_id": "ghost"})
    assert r.status_code == 400

def test_ingest_rejects_labeled_run(client):
    mid = _owned_machine(client)
    ds = client.app.state.data_store
    ds.create_run(Run("labeled", mid))
    ds.append_wear_label(WearLabel("labeled", 1, 0.05))
    r = client.post(f"/machines/{mid}/runs/labeled/cuts", content=encode_waveform(RNG.standard_normal((500, 7))))
    assert r.status_code == 409

def test_ingest_rejects_ended_run(client):
    from datetime import datetime, timezone
    mid = _owned_machine(client)
    ds = client.app.state.data_store
    ds.create_run(Run("ended", mid)); ds.end_run("ended", datetime.now(timezone.utc))
    r = client.post(f"/machines/{mid}/runs/ended/cuts", content=encode_waveform(RNG.standard_normal((500, 7))))
    assert r.status_code == 409

def _upload_files():
    return [("files", (f"c_9_{i:03d}.csv",
             "\n".join(",".join(f"{v:.3f}" for v in row)
                       for row in np.random.default_rng(i).standard_normal((120, 7))),
             "text/csv")) for i in range(1, 3)]

def test_uploads_to_same_tool_accumulate(client):
    mid = _owned_machine(client)
    t = client.post(f"/machines/{mid}/runs", json={}).json()["run_id"]
    r1 = client.post("/analyze", files=_upload_files(), data={"machine_id": mid, "run_id": t}).json()
    r2 = client.post("/analyze", files=_upload_files(), data={"machine_id": mid, "run_id": t}).json()
    assert r1["run_id"] == r2["run_id"] == t            # same tool
    assert r2["n_cuts"] == r1["n_cuts"] + r2["added"]   # cuts accumulate on it
    assert client.get("/machines/phm2010/runs/c1/rul").status_code == 200   # system data intact

def test_upload_without_tool_creates_new_tool(client):
    mid = _owned_machine(client)
    r1 = client.post("/analyze", files=_upload_files(), data={"machine_id": mid}).json()
    r2 = client.post("/analyze", files=_upload_files(), data={"machine_id": mid}).json()
    assert r1["run_id"] != r2["run_id"]                 # each is its own new tool

def test_rename_machine(client):
    mid = _owned_machine(client)
    r = client.patch(f"/machines/{mid}", json={"name": "Renamed Cell"})
    assert r.status_code == 200 and r.json()["name"] == "Renamed Cell"
    assert any(m["machine_id"] == mid and m["name"] == "Renamed Cell"
               for m in client.get("/machines").json()["machines"])

def test_rename_requires_ownership(tmp_path):
    _seed_store_with_twin(tmp_path)
    app = create_app(store_dir=str(tmp_path))
    a = TestClient(app); mid_a = _owned_machine(a, "a@test.com")
    b = TestClient(app); _signup(b, "b@test.com")
    assert b.patch(f"/machines/{mid_a}", json={"name": "hax"}).status_code == 404   # not B's machine


# ---------------- multi-tenancy isolation ----------------

def test_create_machine_requires_auth(client):
    assert client.post("/machines", json={"name": "x"}).status_code == 401

def test_write_endpoints_require_auth(client):
    assert client.post("/machines/phm2010/runs", json={"run_id": "r", "reference_run_id": "c1"}).status_code == 401
    assert client.post("/analyze", files=_upload_files()).status_code == 401

def test_cannot_write_to_system_machine(client):
    _owned_machine(client)          # authed, but phm2010 is system (not owned)
    r = client.post("/machines/phm2010/runs", json={"run_id": "x", "reference_run_id": "c1"})
    assert r.status_code == 404     # system machine hidden from writes

def test_user_cannot_access_others_machine(tmp_path):
    _seed_store_with_twin(tmp_path)
    app = create_app(store_dir=str(tmp_path))
    a = TestClient(app); mid_a = _owned_machine(a, "a@test.com")
    b = TestClient(app); _signup(b, "b@test.com")
    # B cannot see A's machine runs, nor write to it
    assert b.get(f"/machines/{mid_a}/runs").status_code == 404
    assert b.post(f"/machines/{mid_a}/runs", json={"run_id": "x", "reference_run_id": "c1"}).status_code == 404
    # and A's machine is absent from B's /runs view
    assert not any(x["machine_id"] == mid_a for x in b.get("/runs").json()["runs"])

def test_guest_sees_only_system_machines(tmp_path):
    _seed_store_with_twin(tmp_path)
    app = create_app(store_dir=str(tmp_path))
    a = TestClient(app); mid_a = _owned_machine(a, "a@test.com")
    guest = TestClient(app)
    runs = guest.get("/runs").json()["runs"]
    assert any(x["machine_id"] == "phm2010" for x in runs)     # system visible
    assert not any(x["machine_id"] == mid_a for x in runs)     # user machine hidden


# ---------------- hierarchy: tools, add-to-selected, delete, rename ----------------

def test_new_tool_creates_empty_run(client):
    mid = _owned_machine(client)
    r = client.post(f"/machines/{mid}/runs", json={})
    assert r.status_code == 200
    rid = r.json()["run_id"]
    assert any(x["run_id"] == rid and x["n_cuts"] == 0 for x in client.get("/runs").json()["runs"])

def test_cuts_go_to_selected_tool_not_active(client):
    mid = _owned_machine(client)
    t1 = client.post(f"/machines/{mid}/runs", json={}).json()["run_id"]
    t2 = client.post(f"/machines/{mid}/runs", json={}).json()["run_id"]      # t2 is now "active"
    # add cuts explicitly to t1 — must land on t1, NOT t2
    client.post("/analyze", files=_upload_files(), data={"machine_id": mid, "run_id": t1})
    runs = {r["run_id"]: r["n_cuts"] for r in client.get("/runs").json()["runs"]}
    assert runs[t1] == 2 and runs[t2] == 0

def test_delete_run_removes_it(client):
    mid = _owned_machine(client)
    t = client.post(f"/machines/{mid}/runs", json={}).json()["run_id"]
    assert client.delete(f"/machines/{mid}/runs/{t}").status_code == 200
    assert not any(x["run_id"] == t for x in client.get("/runs").json()["runs"])

def test_delete_machine_cascade(client):
    mid = _owned_machine(client)
    client.post(f"/machines/{mid}/runs", json={})
    assert client.delete(f"/machines/{mid}").status_code == 200
    assert not any(m["machine_id"] == mid for m in client.get("/machines").json()["machines"])

def test_rename_tool(client):
    mid = _owned_machine(client)
    t = client.post(f"/machines/{mid}/runs", json={}).json()["run_id"]
    client.patch(f"/machines/{mid}/runs/{t}", json={"name": "Endmill #4"})
    assert any(x["run_id"] == t and x["label"] == "Endmill #4" for x in client.get("/runs").json()["runs"])

def test_delete_and_addcuts_require_ownership(tmp_path):
    _seed_store_with_twin(tmp_path)
    app = create_app(store_dir=str(tmp_path))
    a = TestClient(app); mid_a = _owned_machine(a, "a@test.com")
    ta = a.post(f"/machines/{mid_a}/runs", json={}).json()["run_id"]
    b = TestClient(app); _signup(b, "b@test.com")
    assert b.delete(f"/machines/{mid_a}").status_code == 404
    assert b.delete(f"/machines/{mid_a}/runs/{ta}").status_code == 404
    assert b.post("/analyze", files=_upload_files(), data={"machine_id": mid_a, "run_id": ta}).status_code == 404

def test_cannot_addcuts_to_demo_tool(client):
    _signup(client)
    # c1 is a labeled system tool -> uploading to it is refused
    r = client.post("/analyze", files=_upload_files(), data={"machine_id": "phm2010", "run_id": "c1"})
    assert r.status_code in (404, 409)


# ---------------- stored results (static current-state view) ----------------

def test_results_endpoint_computes_and_caches_for_demo(client):
    r = client.get("/machines/phm2010/runs/c1/results")
    assert r.status_code == 200
    body = r.json()
    assert body["n"] > 0 and len(body["results"]) == body["n"]
    row = body["results"][-1]
    for k in ("cut", "wear_mean", "wear_lo", "wear_hi", "censored", "time"):
        assert k in row

def test_results_ready_after_upload_without_restream(client):
    mid = _owned_machine(client)
    t = client.post(f"/machines/{mid}/runs", json={}).json()["run_id"]
    up = client.post("/analyze", files=_upload_files(), data={"machine_id": mid, "run_id": t}).json()
    # results are already stored by /analyze — fetch returns them directly
    body = client.get(f"/machines/{mid}/runs/{t}/results").json()
    assert body["n"] == up["n_cuts"]
    assert all("wear_mean" in row for row in body["results"])

def test_results_deleted_with_run(client):
    mid = _owned_machine(client)
    t = client.post(f"/machines/{mid}/runs", json={}).json()["run_id"]
    client.post("/analyze", files=_upload_files(), data={"machine_id": mid, "run_id": t})
    ds = client.app.state.data_store
    assert len(ds.read_cut_results(t)) > 0
    client.delete(f"/machines/{mid}/runs/{t}")
    assert len(ds.read_cut_results(t)) == 0        # cascade cleared results
