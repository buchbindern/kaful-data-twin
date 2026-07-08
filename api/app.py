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

import io
import json
import os
import random
import re
import secrets
import uuid
from datetime import datetime, timezone

import numpy as np
from fastapi import FastAPI, Request, Response, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

from storage import SQLiteDataStore, object_store_from_env, data_store_from_env
from features import FeatureExtractor
from datasets import PHM_CHANNELS
from domain.models import Machine, Run, Cut, FeatureRecord, User, Session, CutResult
from ingest import IngestHandler
from twin import ParticleTwin, start_new_run, deploy_from_reference
from auth import (hash_password, verify_password, new_session_token, session_expiry,
                  SESSION_TTL, RateLimiter)


class Credentials(BaseModel):
    email: str
    password: str


class NewMachineRequest(BaseModel):
    name: str | None = None
    machine_type: str | None = None


class RenameRequest(BaseModel):
    name: str | None = None


class NewRunRequest(BaseModel):
    run_id: str | None = None
    reference_run_id: str | None = None
    tool_id: str | None = None


COOKIE_NAME = "kaful_session"
# secure=True requires HTTPS; keep False for local http dev, set KAFUL_COOKIE_SECURE=true on deploy.
COOKIE_SECURE = os.environ.get("KAFUL_COOKIE_SECURE", "false").lower() == "true"
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD_LEN = 8


def _normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def _client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(COOKIE_NAME, token, max_age=int(SESSION_TTL.total_seconds()),
                        httponly=True, secure=COOKIE_SECURE, samesite="lax", path="/")


def compute_and_store_results(data_store, run_id, reference, sigma_scale=2.5):
    """Run the filter over a run ONCE and store per-cut wear + RUL, so the dashboard
    shows current state instantly on view (no re-streaming)."""
    data_store.save_twin_state(deploy_from_reference(data_store, reference, run_id))
    data_store.clear_rul(run_id)
    data_store.clear_cut_results(run_id)
    twin = ParticleTwin(data_store, sigma_scale=sigma_scale, seed=0)
    labels = {l.cut_index: l.wear_mm for l in data_store.read_wear_labels(run_id)}
    now = datetime.now(timezone.utc)
    rows = []
    for f in data_store.read_all_features(run_id):
        rul = twin.update(run_id, f.cut_index, f.features)
        wt = labels.get(f.cut_index)
        rows.append(CutResult(run_id, f.cut_index, twin.last_wear_mean * 1000,
                              twin.last_wear_lo * 1000, twin.last_wear_hi * 1000,
                              (wt * 1000) if wt is not None else None,
                              rul.rul_median, rul.rul_lower, rul.rul_upper,
                              twin.last_rul_censored, now))
    data_store.save_cut_results(rows)
    return rows


def create_app(store_dir: str = "var") -> FastAPI:
    app = FastAPI(title="Kaful data-first twin", version="0.1.0")
    store_dir = Path(store_dir)

    data_store = data_store_from_env(store_dir)
    object_store = object_store_from_env(store_dir)
    twin = ParticleTwin(data_store)            # uses the calibrated sigma_scale default
    handler = IngestHandler(data_store, object_store, FeatureExtractor(PHM_CHANNELS), twin)

    app.state.data_store = data_store
    app.state.handler = handler

    login_limiter = RateLimiter(int(os.environ.get("KAFUL_LOGIN_RATE", "10")), 60)
    signup_limiter = RateLimiter(int(os.environ.get("KAFUL_SIGNUP_RATE", "5")), 60)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        resp = await call_next(request)
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return resp

    @app.get("/health")
    def health():
        store = "postgres" if type(data_store).__name__.startswith("Postgres") else "sqlite"
        return {"status": "ok", "store": store}

    # ---------------- Auth ----------------
    def optional_user(request: Request):
        token = request.cookies.get(COOKIE_NAME)
        if not token:
            return None
        sess = data_store.get_valid_session(token)
        if sess is None:
            return None
        return data_store.get_user(sess.user_id)

    def require_user(request: Request) -> User:
        user = optional_user(request)
        if user is None:
            raise HTTPException(status_code=401, detail="not authenticated")
        return user

    app.state.optional_user = optional_user
    app.state.require_user = require_user

    # ---- authorization: system machines (owner_id None) are readable by all, read-only ----
    def _visible(machine, user) -> bool:
        return machine.owner_id is None or (user is not None and machine.owner_id == user.user_id)

    def _readable_machine(machine_id, user):
        m = data_store.get_machine(machine_id)
        if m is None or not _visible(m, user):
            raise HTTPException(status_code=404, detail=f"machine {machine_id!r} not found")
        return m

    def _writable_machine(machine_id, user):
        m = data_store.get_machine(machine_id)
        # system machines are read-only; another user's machine is hidden (404, no leak)
        if m is None or m.owner_id != user.user_id:
            raise HTTPException(status_code=404, detail=f"machine {machine_id!r} not found")
        return m

    @app.get("/machines")
    def list_machines_endpoint(user: User = Depends(optional_user)):
        out = []
        for m in data_store.list_machines():
            if not _visible(m, user):
                continue
            out.append({"machine_id": m.machine_id, "name": m.name,
                        "machine_type": m.machine_type, "is_system": m.owner_id is None})
        return {"machines": out}

    @app.post("/machines")
    def create_machine_endpoint(body: NewMachineRequest, user: User = Depends(require_user)):
        machine_id = "m-" + secrets.token_hex(6)
        mtype = body.machine_type or "cnc_milling"
        data_store.create_machine(Machine(machine_id, mtype, name=body.name, owner_id=user.user_id))
        return {"machine_id": machine_id, "name": body.name, "machine_type": mtype}

    @app.patch("/machines/{machine_id}")
    def rename_machine_endpoint(machine_id: str, body: RenameRequest,
                                user: User = Depends(require_user)):
        _writable_machine(machine_id, user)
        name = (body.name or "").strip() or None
        data_store.rename_machine(machine_id, name)
        return {"machine_id": machine_id, "name": name}

    @app.post("/auth/signup")
    def signup(body: Credentials, request: Request, response: Response):
        if not signup_limiter.allow(_client_ip(request)):
            raise HTTPException(status_code=429, detail="too many signups, please try again later")
        email = _normalize_email(body.email)
        if not _EMAIL_RE.match(email) or len(email) > 254:
            raise HTTPException(status_code=400, detail="invalid email address")
        if not (MIN_PASSWORD_LEN <= len(body.password) <= 128):
            raise HTTPException(status_code=400,
                                detail=f"password must be {MIN_PASSWORD_LEN}-128 characters")
        if data_store.get_user_by_email(email) is not None:
            raise HTTPException(status_code=409, detail="email already registered")
        now = datetime.now(timezone.utc)
        user = User(uuid.uuid4().hex, email, hash_password(body.password), now)
        data_store.create_user(user)
        token = new_session_token()
        data_store.create_session(Session(token, user.user_id, now, session_expiry(now)))
        _set_session_cookie(response, token)
        return {"user_id": user.user_id, "email": user.email}

    @app.post("/auth/login")
    def login(body: Credentials, request: Request, response: Response):
        if not login_limiter.allow(_client_ip(request)):
            raise HTTPException(status_code=429, detail="too many attempts, please try again later")
        if random.random() < 0.05:                                       # occasional cleanup (keeps login fast)
            data_store.delete_expired_sessions(datetime.now(timezone.utc))
        email = _normalize_email(body.email)
        user = data_store.get_user_by_email(email)
        if user is None or not verify_password(body.password, user.password_hash):
            raise HTTPException(status_code=401, detail="invalid email or password")
        now = datetime.now(timezone.utc)
        token = new_session_token()
        data_store.create_session(Session(token, user.user_id, now, session_expiry(now)))
        _set_session_cookie(response, token)
        return {"user_id": user.user_id, "email": user.email}

    @app.post("/auth/logout")
    def logout(request: Request, response: Response):
        token = request.cookies.get(COOKIE_NAME)
        if token:
            data_store.delete_session(token)
        response.delete_cookie(COOKIE_NAME, path="/")
        return {"ok": True}

    @app.post("/auth/logout-all")
    def logout_all(response: Response, user: User = Depends(require_user)):
        data_store.delete_user_sessions(user.user_id)
        response.delete_cookie(COOKIE_NAME, path="/")
        return {"ok": True}

    @app.get("/auth/me")
    def me(user: User = Depends(require_user)):
        return {"user_id": user.user_id, "email": user.email}

    @app.post("/machines/{machine_id}/runs")
    def start_run(machine_id: str, body: NewRunRequest, user: User = Depends(require_user)):
        """Add a new tool (run) to the machine and deploy a twin. Tools coexist; this
        does not end other tools."""
        _writable_machine(machine_id, user)
        reference = body.reference_run_id or "c1"
        if data_store.get_run(reference) is None:
            raise HTTPException(status_code=400, detail=f"reference run {reference!r} not found")
        run_id = body.run_id or ("run-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
                                 + "-" + secrets.token_hex(3))
        if data_store.get_run(run_id) is not None:
            raise HTTPException(status_code=409, detail=f"run {run_id!r} already exists")
        data_store.create_run(Run(run_id, machine_id, tool_id=body.tool_id))
        data_store.save_twin_state(deploy_from_reference(data_store, reference, run_id))
        return {"run_id": run_id, "machine_id": machine_id, "status": "active"}

    @app.delete("/machines/{machine_id}")
    def delete_machine_endpoint(machine_id: str, user: User = Depends(require_user)):
        _writable_machine(machine_id, user)
        data_store.delete_machine(machine_id)
        return {"deleted": machine_id}

    @app.delete("/machines/{machine_id}/runs/{run_id}")
    def delete_run_endpoint(machine_id: str, run_id: str, user: User = Depends(require_user)):
        _writable_machine(machine_id, user)
        run = data_store.get_run(run_id)
        if run is None or run.machine_id != machine_id:
            raise HTTPException(status_code=404, detail="tool not found on this machine")
        data_store.delete_run(run_id)
        return {"deleted": run_id}

    @app.patch("/machines/{machine_id}/runs/{run_id}")
    def rename_run_endpoint(machine_id: str, run_id: str, body: RenameRequest,
                            user: User = Depends(require_user)):
        _writable_machine(machine_id, user)
        run = data_store.get_run(run_id)
        if run is None or run.machine_id != machine_id:
            raise HTTPException(status_code=404, detail="tool not found on this machine")
        label = (body.name or "").strip() or None
        data_store.rename_run(run_id, label)
        return {"run_id": run_id, "label": label}

    @app.post("/machines/{machine_id}/runs/{run_id}/cuts")
    async def ingest_cut(machine_id: str, run_id: str, request: Request,
                         user: User = Depends(require_user)):
        _writable_machine(machine_id, user)
        raw = await request.body()                      # the compressed waveform blob
        if not raw:
            raise HTTPException(status_code=400, detail="empty body: expected waveform bytes")
        run = data_store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"run {run_id!r} does not exist")
        if run.ended_at is not None:
            raise HTTPException(status_code=409,
                                detail=f"run {run_id!r} is archived (ended) and does not accept new cuts")
        if data_store.read_wear_labels(run_id):
            raise HTTPException(status_code=409,
                                detail=f"run {run_id!r} is a labeled reference run and does not accept live cuts")
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
    def get_rul(machine_id: str, run_id: str, user: User = Depends(optional_user)):
        _readable_machine(machine_id, user)
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
    def all_runs(user: User = Depends(optional_user)):
        """Runs across machines the caller may see (system + their own)."""
        out = []
        for m in data_store.list_machines():
            if not _visible(m, user):
                continue
            for r in data_store.list_runs(m.machine_id):
                n_cuts = len(data_store.read_all_features(r.run_id))
                n_labels = len(data_store.read_wear_labels(r.run_id))
                out.append({"machine_id": m.machine_id, "machine_type": m.machine_type,
                            "run_id": r.run_id, "label": r.tool_id, "active": r.ended_at is None,
                            "n_cuts": n_cuts, "has_labels": n_labels > 0})
        return {"runs": out}

    @app.get("/machines/{machine_id}/runs")
    def list_runs(machine_id: str, user: User = Depends(optional_user)):
        _readable_machine(machine_id, user)
        runs = data_store.list_runs(machine_id)
        out = []
        for r in runs:
            n_cuts = len(data_store.read_all_features(r.run_id))
            n_labels = len(data_store.read_wear_labels(r.run_id))
            out.append({"run_id": r.run_id, "active": r.ended_at is None,
                        "n_cuts": n_cuts, "has_labels": n_labels > 0})
        return {"machine_id": machine_id, "runs": out}

    @app.post("/analyze")
    async def analyze(files: list[UploadFile] = File(...), reference: str = "c1",
                      machine_id: str | None = Form(None), run_id: str | None = Form(None),
                      user: User = Depends(require_user)):
        """Upload cut files -> extract features -> create a run -> deploy the reference
        model. Returns a run_id the dashboard then streams via /replay."""
        if not files:
            raise HTTPException(status_code=400, detail="no files uploaded")
        if data_store.get_run(reference) is None or not data_store.read_wear_labels(reference):
            raise HTTPException(status_code=400,
                                detail=f"reference model {reference!r} not available (needs a labeled run)")
        if len(files) > 400:
            raise HTTPException(status_code=400, detail="too many files (max 400 cuts per upload)")

        if machine_id:
            target_id = _writable_machine(machine_id, user).machine_id
        else:
            target_id = f"uploads-{user.user_id}"
            if data_store.get_machine(target_id) is None:
                data_store.create_machine(Machine(target_id, "cnc_milling",
                                                  name="Uploaded tools", owner_id=user.user_id))

        # Append to the SELECTED tool (run_id); if none given, create a new tool.
        if run_id:
            run = data_store.get_run(run_id)
            if run is None or run.machine_id != target_id:
                raise HTTPException(status_code=404, detail=f"tool {run_id!r} not found on this machine")
            if run.ended_at is not None or data_store.read_wear_labels(run_id):
                raise HTTPException(status_code=409, detail="this tool does not accept new cuts")
            start = len(data_store.read_all_features(run_id))
            if data_store.load_twin_state(run_id) is None:
                data_store.save_twin_state(deploy_from_reference(data_store, reference, run_id))
        else:
            run_id = "run-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
            data_store.create_run(Run(run_id, target_id))
            data_store.save_twin_state(deploy_from_reference(data_store, reference, run_id))
            start = 0

        extractor = FeatureExtractor(PHM_CHANNELS)
        ordered = sorted(files, key=lambda f: f.filename or "")
        cuts, feats = [], []
        for offset, uf in enumerate(ordered, start=1):
            i = start + offset
            content = await uf.read()
            try:
                arr = np.loadtxt(io.StringIO(content.decode("utf-8", "ignore")), delimiter=",")
            except Exception:
                raise HTTPException(status_code=400,
                                    detail=f"could not parse {uf.filename!r}: expected a headerless CSV")
            if arr.ndim != 2 or arr.shape[1] != len(PHM_CHANNELS):
                raise HTTPException(status_code=400,
                                    detail=f"{uf.filename!r}: expected {len(PHM_CHANNELS)} columns, got shape {arr.shape}")
            cuts.append(Cut(run_id, i, f"upload/{run_id}/{i:06d}"))
            feats.append(FeatureRecord(run_id, i, extractor.extract(arr)))

        data_store.append_cuts_bulk(cuts)          # one round trip instead of N
        data_store.append_features_bulk(feats)
        default_ref = run_id if data_store.read_wear_labels(run_id) else reference
        compute_and_store_results(data_store, run_id, default_ref)   # compute once, ready to view
        return {"run_id": run_id, "machine_id": target_id, "n_cuts": start + len(cuts),
                "added": len(cuts), "reference": reference}

    @app.get("/machines/{machine_id}/runs/{run_id}/results")
    def get_results(machine_id: str, run_id: str, reference: str | None = None,
                    user: User = Depends(optional_user)):
        """Stored per-cut results for instant, static rendering (current state).
        Computes + caches on first view; recomputes if `reference` is given."""
        _readable_machine(machine_id, user)
        if data_store.get_run(run_id) is None:
            raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")
        default_ref = run_id if data_store.read_wear_labels(run_id) else "c1"
        ref = reference or default_ref
        results = data_store.read_cut_results(run_id)
        if not results or reference is not None:
            try:
                results = compute_and_store_results(data_store, run_id, ref)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
        times = {c.cut_index: (c.ingested_at.isoformat() if c.ingested_at else None)
                 for c in data_store.read_all_cuts(run_id)}
        series = [{"cut": r.cut_index, "wear_mean": r.wear_mean, "wear_lo": r.wear_lo,
                   "wear_hi": r.wear_hi, "wear_true": r.wear_true, "rul_median": r.rul_median,
                   "rul_lo": r.rul_lo, "rul_hi": r.rul_hi, "censored": r.censored,
                   "time": times.get(r.cut_index)} for r in results]
        return {"run": run_id, "reference": ref, "n": len(series),
                "threshold_um": 200.0, "results": series}

    @app.get("/machines/{machine_id}/runs/{run_id}/replay")
    def replay(machine_id: str, run_id: str, reference: str | None = None,
               sigma_scale: float = 2.5, user: User = Depends(optional_user)):
        """Stream the filter over a run's cuts as Server-Sent Events, for the live
        dashboard. Deploys a fresh twin first (fit on `reference`, default: self)."""
        _readable_machine(machine_id, user)
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
