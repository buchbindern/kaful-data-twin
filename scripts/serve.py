"""
Run the Kaful twin as an HTTP service (M9). From the repo root:

    python scripts/serve.py [--store-dir var] [--port 8000]

Single-worker on purpose (see the concurrency note in api/app.py). Endpoints:
    GET  /health
    POST /machines/{mid}/runs/{rid}/cuts   (body = compressed waveform bytes)
    GET  /machines/{mid}/runs/{rid}/rul
"""

from __future__ import annotations

import argparse
import os

import uvicorn

from api import create_app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store-dir", default="var")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    args = ap.parse_args()
    uvicorn.run(create_app(store_dir=args.store_dir), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
