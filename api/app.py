"""
HTTP API (M9) — a thin FastAPI shell over the existing IngestHandler.

This is the "same handler, different transport" payoff designed at M4:
POST a compressed waveform, the endpoint hands its RAW BYTES straight to
handler.ingest_cut() — the identical call the in-process replay driver makes.
No twin/handler/storage code changes; only the transport is new.

Concurrency note: SQLiteDataStore uses one connection and ParticleTwin does a
load->update->save on shared twin_state, which is NOT safe under simultaneous
cuts for the SAME run. A single machine emits one cut every few minutes, so this
is a non-issue here; the fleet path is per-run locking + Postgres (swap the
DataStore impl — the interface already allows it). Run uvicorn single-worker.
"""

from __future__ import annotations

from pathlib import Path

import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

from storage import SQLiteDataStore, object_store_from_env
from features import FeatureExtractor
from datasets import PHM_CHANNELS
from ingest import IngestHandler
from twin import ParticleTwin, start_new_run, deploy_from_reference


class NewRunRequest(BaseModel):
    run_id: str
    reference_run_id: str
    tool_id: str | None = None


def create_app(store_dir: str = "var") -> FastAPI:
    app = FastAPI(title="Kaful data-first twin", version="0.1.0")
    store_dir = Path(store_dir)

    data_store = SQLiteDataStore(store_dir / "kaful.db")
    object_store = object_store_from_env(store_dir)
    twin = ParticleTwin(data_store)            # uses the calibrated sigma_scale default
    handler = IngestHandler(data_store, object_store, FeatureExtractor(PHM_CHANNELS), twin)

    app.state.data_store = data_store
    app.state.handler = handler

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/machines/{machine_id}/runs")
    def start_run(machine_id: str, body: NewRunRequest):
        """Tool change: end the active run, start a fresh one, deploy a new twin."""
        if data_store.get_machine(machine_id) is None:
            raise HTTPException(status_code=404, detail=f"machine {machine_id!r} not found")
        if data_store.get_run(body.reference_run_id) is None:
            raise HTTPException(status_code=400,
                                detail=f"reference run {body.reference_run_id!r} not found")
        try:
            run = start_new_run(data_store, machine_id, body.run_id,
                                reference_run_id=body.reference_run_id, tool_id=body.tool_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return {"run_id": run.run_id, "machine_id": run.machine_id, "status": "active"}

    @app.post("/machines/{machine_id}/runs/{run_id}/cuts")
    async def ingest_cut(machine_id: str, run_id: str, request: Request):
        raw = await request.body()                      # the compressed waveform blob
        if not raw:
            raise HTTPException(status_code=400, detail="empty body: expected waveform bytes")
        if data_store.get_run(run_id) is None:
            raise HTTPException(status_code=404, detail=f"run {run_id!r} does not exist")
        if data_store.load_twin_state(run_id) is None:
            raise HTTPException(status_code=409,
                                detail=f"no twin built for run {run_id!r}; build_twin first")
        cut_index = len(data_store.read_all_features(run_id)) + 1
        try:
            rul = handler.ingest_cut(machine_id, run_id, cut_index, raw)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"cut_index": rul.cut_index, "rul_median": rul.rul_median,
                "rul_lower": rul.rul_lower, "rul_upper": rul.rul_upper,
                "ci_level": rul.ci_level}

    @app.get("/machines/{machine_id}/runs/{run_id}/rul")
    def get_rul(machine_id: str, run_id: str):
        preds = data_store.read_all_rul(run_id)
        return {"run_id": run_id, "n": len(preds),
                "predictions": [{"cut_index": p.cut_index, "rul_median": p.rul_median,
                                 "rul_lower": p.rul_lower, "rul_upper": p.rul_upper}
                                for p in preds]}

    _STATIC = Path(__file__).parent / "static"

    @app.get("/")
    def dashboard():
        return FileResponse(_STATIC / "index.html")

    @app.get("/runs")
    def all_runs():
        """Every run across all machines, for the monitor's tool rail."""
        out = []
        for m in data_store.list_machines():
            for r in data_store.list_runs(m.machine_id):
                n_cuts = len(data_store.read_all_features(r.run_id))
                n_labels = len(data_store.read_wear_labels(r.run_id))
                out.append({"machine_id": m.machine_id, "machine_type": m.machine_type,
                            "run_id": r.run_id, "active": r.ended_at is None,
                            "n_cuts": n_cuts, "has_labels": n_labels > 0})
        return {"runs": out}

    @app.get("/machines/{machine_id}/runs")
    def list_runs(machine_id: str):
        runs = data_store.list_runs(machine_id)
        out = []
        for r in runs:
            n_cuts = len(data_store.read_all_features(r.run_id))
            n_labels = len(data_store.read_wear_labels(r.run_id))
            out.append({"run_id": r.run_id, "active": r.ended_at is None,
                        "n_cuts": n_cuts, "has_labels": n_labels > 0})
        return {"machine_id": machine_id, "runs": out}

    @app.get("/machines/{machine_id}/runs/{run_id}/replay")
    def replay(machine_id: str, run_id: str, reference: str | None = None,
               sigma_scale: float = 2.5):
        """Stream the filter over a run's cuts as Server-Sent Events, for the live
        dashboard. Deploys a fresh twin first (fit on `reference`, default: self)."""
        if data_store.get_run(run_id) is None:
            raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")
        ref = reference or run_id
        try:
            data_store.save_twin_state(deploy_from_reference(data_store, ref, run_id))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        data_store.clear_rul(run_id)
        twin = ParticleTwin(data_store, sigma_scale=sigma_scale, seed=0)
        labels = {l.cut_index: l.wear_mm for l in data_store.read_wear_labels(run_id)}
        feats = data_store.read_all_features(run_id)
        threshold_um = 200.0

        def stream():
            yield f"data: {json.dumps({'type': 'meta', 'run': run_id, 'reference': ref, 'n': len(feats), 'threshold_um': threshold_um})}\n\n"
            for f in feats:
                rul = twin.update(run_id, f.cut_index, f.features)
                data_store.append_rul(rul)
                wt = labels.get(f.cut_index)
                ev = {"type": "cut", "cut": f.cut_index,
                      "wear_mean": twin.last_wear_mean * 1000,
                      "wear_lo": twin.last_wear_lo * 1000,
                      "wear_hi": twin.last_wear_hi * 1000,
                      "wear_true": (wt * 1000) if wt is not None else None,
                      "rul_median": rul.rul_median, "rul_lo": rul.rul_lower,
                      "rul_hi": rul.rul_upper, "censored": twin.last_rul_censored}
                yield f"data: {json.dumps(ev)}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app
