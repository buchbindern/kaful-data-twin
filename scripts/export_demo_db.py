"""
Export a slim, committable demo database for deployment (e.g. Render).

Copies the feature/label data the live monitor needs (machines, runs, cuts,
features, wear_labels) and DROPS the regenerable tables (twin_state, rul_predictions
— rebuilt at runtime on every replay), then VACUUMs. Raw waveforms are NOT included;
the monitor runs off features, and raw lives in object storage.

    python scripts/export_demo_db.py            # var/kaful.db -> deploy/kaful.db
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="var/kaful.db")
    ap.add_argument("--dst", default="deploy/kaful.db")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if not src.exists():
        raise SystemExit(f"source db not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)

    con = sqlite3.connect(dst)
    for table in ("twin_state", "rul_predictions"):
        con.execute(f"DELETE FROM {table}")
    con.commit()
    con.execute("VACUUM")
    runs = [r[0] for r in con.execute("SELECT run_id FROM runs ORDER BY run_id")]
    con.close()
    print(f"wrote {dst} ({dst.stat().st_size/1e6:.1f} MB) with runs: {', '.join(runs) or '(none)'}")


if __name__ == "__main__":
    main()
