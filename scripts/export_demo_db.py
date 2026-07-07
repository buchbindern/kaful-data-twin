"""
Export a slim, committable demo database for deployment (e.g. Render).

Safeguards against shipping contamination:
  * WHITELIST — exports only labeled reference runs (or an explicit --runs list),
    never stray upload/live runs.
  * TRIPWIRE — refuses to export if any exported labeled run has feature rows PAST
    its last wear label (the signature of cuts wrongly appended to a reference run),
    and prints the exact SQL to purge them.
  * Marks exported runs as ENDED (archived) so the deployed API's ingest guard treats
    them as immutable reference runs.
  * Drops regenerable tables (twin_state, rul_predictions) and VACUUMs.

    python scripts/export_demo_db.py                 # all labeled runs -> deploy/kaful.db
    python scripts/export_demo_db.py --runs c1,c4    # explicit subset
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_CHILD_TABLES = ("rul_predictions", "twin_state", "features", "wear_labels", "cuts")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="var/kaful.db")
    ap.add_argument("--dst", default="deploy/kaful.db")
    ap.add_argument("--runs", default="", help="comma-separated run ids (default: all labeled runs)")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if not src.exists():
        raise SystemExit(f"source db not found: {src}")

    con = sqlite3.connect(src)
    labeled = [r[0] for r in con.execute("SELECT DISTINCT run_id FROM wear_labels ORDER BY run_id")]
    runs = [r.strip() for r in args.runs.split(",") if r.strip()] or labeled
    if not runs:
        raise SystemExit("no labeled runs found to export")

    problems = []
    for run in runs:
        ml = con.execute("SELECT max(cut_index) FROM wear_labels WHERE run_id=?", (run,)).fetchone()[0]
        mf = con.execute("SELECT max(cut_index) FROM features WHERE run_id=?", (run,)).fetchone()[0]
        if ml is None:
            raise SystemExit(f"run {run!r} has no wear labels — not a reference run; refuse to export")
        if mf is not None and mf > ml:
            problems.append((run, ml, mf))
    con.close()

    if problems:
        print("REFUSING TO EXPORT — contamination detected (features past last label):\n")
        for run, ml, mf in problems:
            print(f"  run {run!r}: labels end at cut {ml}, but features run to cut {mf} "
                  f"({mf - ml} stray cut(s)). Purge with:")
            for t in ("features", "cuts", "rul_predictions"):
                print(f"    DELETE FROM {t} WHERE run_id='{run}' AND cut_index>{ml};")
            print()
        raise SystemExit(1)

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)
    con = sqlite3.connect(dst)
    keep = set(runs)
    for run in [r[0] for r in con.execute("SELECT run_id FROM runs")]:
        if run not in keep:
            for t in _CHILD_TABLES:
                con.execute(f"DELETE FROM {t} WHERE run_id=?", (run,))
            con.execute("DELETE FROM runs WHERE run_id=?", (run,))
    con.execute("DELETE FROM twin_state")
    con.execute("DELETE FROM rul_predictions")
    con.execute("DELETE FROM machines WHERE machine_id NOT IN (SELECT DISTINCT machine_id FROM runs)")
    con.execute("UPDATE runs SET ended_at=? WHERE ended_at IS NULL",
                (datetime.now(timezone.utc).isoformat(),))
    con.commit()
    con.execute("VACUUM")
    kept = list(con.execute(
        "SELECT run_id, (SELECT count(*) FROM features f WHERE f.run_id=runs.run_id) "
        "FROM runs ORDER BY run_id"))
    con.close()
    print(f"wrote {dst} ({dst.stat().st_size/1e6:.1f} MB), reference runs (ended):")
    for r, n in kept:
        print(f"  {r}: {n} cuts")


if __name__ == "__main__":
    main()
