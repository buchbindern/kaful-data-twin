"""
Load test (Phase G, step 1) — measure the ingest path under concurrent runs.

Drives the LIVE HTTP ingest path (POST .../cuts) exactly like replay_http.py, but
fans out across C concurrent runs to measure fleet-style concurrency. Each worker
OWNS one run and posts its cuts sequentially, so the server-assigned cut_index
never races; concurrency C = number of runs hammered at once.

MEASURE FIRST, fix second. Run against a LOCAL server on LOCAL Postgres so the
numbers reflect Kaful's architecture, not Render/Neon free-tier CPU/latency:

    # 1. seed the c1 reference locally (needs PHM data + local Postgres)
    KAFUL_DB=postgres DATABASE_URL=postgres://localhost/kaful python scripts/seed_postgres.py

    # 2. serve single-worker (as in prod) against the same DB
    KAFUL_DB=postgres DATABASE_URL=postgres://localhost/kaful python scripts/serve.py --port 8000

    # 3. sweep concurrency and capture the baseline
    python scripts/loadtest.py --concurrency 1,2,5,10,20 --cuts-per-run 30 --out baseline.csv

Prints a per-concurrency table (throughput, p50/p95/p99 latency, errors) and writes
a CSV. Re-run after the rung-1 fix with --out after.csv to get the before/after.

Stdlib + numpy only; reuses ingest.encode_waveform so the wire format is identical
to the real edge gateway.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ingest import encode_waveform


# ----------------------------- HTTP helpers -----------------------------
# One authenticated session is captured once, then every request is built fresh
# with an explicit Cookie header -> no shared client state, thread-safe by design.

def _request(method, url, data=None, cookie=None, ctype=None, timeout=180):
    headers = {}
    if cookie:
        headers["Cookie"] = cookie
    if ctype:
        headers["Content-Type"] = ctype
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        set_cookie = resp.headers.get_all("Set-Cookie") or []
        body = json.loads(raw) if raw else {}
        return resp.status, body, set_cookie


def _cookie_from(set_cookie_headers):
    # keep only the name=value part of each Set-Cookie, join for the Cookie header
    pairs = [h.split(";", 1)[0].strip() for h in set_cookie_headers if h.strip()]
    return "; ".join(pairs)


def authenticate(base, email, password):
    creds = json.dumps({"email": email, "password": password}).encode()
    for path in ("/auth/signup", "/auth/login"):
        try:
            status, _, set_cookie = _request("POST", base + path, data=creds,
                                             ctype="application/json")
        except urllib.error.HTTPError:
            continue  # e.g. signup 409 (user exists) -> fall through to login
        cookie = _cookie_from(set_cookie)
        if cookie:
            return cookie
    sys.exit("auth failed: could not obtain a session cookie via signup or login")


# ----------------------------- setup -----------------------------

def create_machine(base, cookie):
    body = json.dumps({"name": "loadtest", "machine_type": "cnc_milling"}).encode()
    _, resp, _ = _request("POST", base + "/machines", data=body, cookie=cookie,
                          ctype="application/json")
    return resp["machine_id"]


def create_run(base, cookie, machine_id, run_id, reference):
    body = json.dumps({"run_id": run_id, "reference_run_id": reference}).encode()
    url = f"{base}/machines/{machine_id}/runs"
    try:
        _request("POST", url, data=body, cookie=cookie, ctype="application/json")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        sys.exit(f"run setup failed ({exc.code}) for {run_id!r}: {detail}\n"
                 f"  -> is reference {reference!r} seeded in this DB? "
                 f"(run scripts/seed_postgres.py)")


def make_blob(samples, channels, rng):
    """Synthetic waveform of the right shape; contents are irrelevant to throughput."""
    arr = rng.standard_normal((samples, channels)).astype(np.float32)
    return encode_waveform(arr)


# ----------------------------- one worker -----------------------------

def drive_run(base, cookie, machine_id, run_id, blob, n_cuts):
    """Post n_cuts to one run, sequentially. Returns (latencies_ms, errors)."""
    url = f"{base}/machines/{machine_id}/runs/{run_id}/cuts"
    lat, errors = [], 0
    for _ in range(n_cuts):
        t0 = time.perf_counter()
        try:
            _request("POST", url, data=blob, cookie=cookie,
                     ctype="application/octet-stream")
            lat.append((time.perf_counter() - t0) * 1000.0)
        except Exception:
            errors += 1
    return lat, errors


# ----------------------------- sweep -----------------------------

def run_level(base, cookie, machine_id, concurrency, n_cuts, blob, tag):
    run_ids = [f"lt-{tag}-{i:03d}" for i in range(concurrency)]
    for rid in run_ids:
        create_run(base, cookie, machine_id, rid, args.reference)

    t0 = time.perf_counter()
    all_lat, all_err = [], 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(drive_run, base, cookie, machine_id, rid, blob, n_cuts)
                   for rid in run_ids]
        for f in futures:
            lat, err = f.result()
            all_lat.extend(lat)
            all_err += err
    wall = time.perf_counter() - t0

    ok = len(all_lat)
    thru = ok / wall if wall > 0 else 0.0
    pct = (lambda p: float(np.percentile(all_lat, p)) if all_lat else float("nan"))
    return {
        "concurrency": concurrency,
        "cuts_ok": ok,
        "errors": all_err,
        "wall_s": round(wall, 2),
        "throughput_cps": round(thru, 2),
        "p50_ms": round(pct(50), 1),
        "p95_ms": round(pct(95), 1),
        "p99_ms": round(pct(99), 1),
    }


def main():
    global args
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--email", default="loadtest@example.com")
    ap.add_argument("--password", default="loadtest-passw0rd")
    ap.add_argument("--reference", default="c1", help="seeded reference run for twin deploy")
    ap.add_argument("--concurrency", default="1,2,5,10,20",
                    help="comma-separated concurrency levels to sweep")
    ap.add_argument("--cuts-per-run", type=int, default=30)
    ap.add_argument("--samples", type=int, default=50000, help="waveform rows (blob size)")
    ap.add_argument("--channels", type=int, default=7, help="PHM channels")
    ap.add_argument("--out", default="loadtest.csv")
    args = ap.parse_args()

    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]
    rng = np.random.default_rng(0)
    blob = make_blob(args.samples, args.channels, rng)
    print(f"blob size: {len(blob)/1e6:.2f} MB  |  base: {args.base}  |  "
          f"cuts/run: {args.cuts_per_run}")

    cookie = authenticate(args.base, args.email, args.password)
    machine_id = create_machine(args.base, cookie)

    rows = []
    hdr = f"{'C':>4} {'cuts':>6} {'err':>4} {'wall_s':>8} {'cuts/s':>8} {'p50':>8} {'p95':>8} {'p99':>8}"
    print(hdr)
    print("-" * len(hdr))
    for c in levels:
        r = run_level(args.base, cookie, machine_id, c, args.cuts_per_run, blob, tag=f"c{c}")
        rows.append(r)
        print(f"{r['concurrency']:>4} {r['cuts_ok']:>6} {r['errors']:>4} "
              f"{r['wall_s']:>8} {r['throughput_cps']:>8} "
              f"{r['p50_ms']:>8} {r['p95_ms']:>8} {r['p99_ms']:>8}")

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
