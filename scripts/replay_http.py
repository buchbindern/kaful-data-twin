"""
Replay PHM cuts to a running Kaful server (M9) — the SAME replay as the in-process
driver, but over HTTP. From the repo root (with scripts/serve.py running):

    python scripts/replay_http.py --record c1 --limit 5

The server assigns cut_index and returns the RUL for each posted cut. Requires the
run to exist and its twin to be built (build_twin / run_filter) beforehand.
Uses only the stdlib (urllib) so it needs no client dependency.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from datasets import iter_cut_files, load_cut_waveform
from ingest import encode_waveform


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default="c1")
    ap.add_argument("--machine", default="phm2010")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--data-dir", default="data/phm2010")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    args = ap.parse_args()

    folder = Path(args.data_dir) / args.record / args.record
    cut_files = iter_cut_files(folder)[:args.limit]
    url = f"{args.base}/machines/{args.machine}/runs/{args.record}/cuts"
    for _, path in cut_files:
        raw = encode_waveform(load_cut_waveform(path))
        req = urllib.request.Request(url, data=raw, method="POST",
                                     headers={"Content-Type": "application/octet-stream"})
        with urllib.request.urlopen(req) as resp:
            body = json.load(resp)
        print(f"cut {body['cut_index']:>3}  ->  RUL {body['rul_median']:.0f} "
              f"[{body['rul_lower']:.0f}, {body['rul_upper']:.0f}]")


if __name__ == "__main__":
    main()
