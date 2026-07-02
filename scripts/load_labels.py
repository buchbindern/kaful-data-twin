"""
Load a PHM record's wear labels into the runtime store. Run from the repo root:

    python scripts/load_labels.py --record c1

Idempotent: re-running skips labels already present. Labels are reference data and
are keyed to a run, so the run must exist (create it via run_replay.py first, or
this script will create a bare machine+run if missing).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from domain.models import Machine, Run, WearLabel
from storage import SQLiteDataStore
from datasets import load_wear_labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default="c1")
    ap.add_argument("--data-dir", default="data/phm2010")
    ap.add_argument("--store-dir", default="var")
    args = ap.parse_args()

    wear_path = Path(args.data_dir) / args.record / f"{args.record}_wear.csv"
    if not wear_path.exists():
        raise SystemExit(f"wear file not found: {wear_path}")
    labels = load_wear_labels(wear_path)
    print(f"loaded {len(labels)} labels from {wear_path}")
    print(f"  wear range: {labels[0][1]*1000:.0f}..{labels[-1][1]*1000:.0f} (1e-3 mm), "
          f"final {labels[-1][1]:.3f} mm  (threshold 0.200 mm)")

    ds = SQLiteDataStore(Path(args.store_dir) / "kaful.db")
    run_id = args.record
    if ds.get_machine("phm2010") is None:
        ds.create_machine(Machine("phm2010", "phm2010_milling"))
    if ds.get_run(run_id) is None:
        ds.create_run(Run(run_id, "phm2010"))

    existing = {w.cut_index for w in ds.read_wear_labels(run_id)}
    added = 0
    for cut, wear_mm in labels:
        if cut in existing:
            continue
        ds.append_wear_label(WearLabel(run_id, cut, wear_mm))
        added += 1
    print(f"stored {added} new labels ({len(existing)} already present)")
    ds.close()


if __name__ == "__main__":
    main()
