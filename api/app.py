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

from fastapi import FastAPI, Request, HTTPException

from storage import SQLiteDataStore, FilesystemObjectStore
from features import FeatureExtractor
from datasets import PHM_CHANNELS
from ingest import IngestHandler
from twin import ParticleTwin


def create_app(store_dir: str = "var") -> FastAPI:
    app = FastAPI(title="Kaful data-first twin", version="0.1.0")
    store_dir = Path(store_dir)

    data_store = SQLiteDataStore(store_dir / "kaful.db")
    object_store = FilesystemObjectStore(store_dir / "object_store")
    twin = ParticleTwin(data_store)            # uses the calibrated sigma_scale default
    handler = IngestHandler(data_store, object_store, FeatureExtractor(PHM_CHANNELS), twin)

    app.state.data_store = data_store
    app.state.handler = handler

    @app.get("/health")
    def health():
        return {"status": "ok"}

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
        # cut_index = next index for this run (server assigns; edge need not track it)
        cut_index = len(data_store.read_all_features(run_id)) + 1
        try:
            rul = handler.ingest_cut(machine_id, run_id, cut_index, raw)
        except Exception as exc:                        # bad payload, etc.
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

    return app
