"""
M4 end-to-end replay. Run from the repo root:

    python scripts/run_replay.py --record c1 --limit 20      # quick first run
    python scripts/run_replay.py --record c1                 # full run (all cuts)

Writes the system's runtime store under ./var (gitignored). Ingest is idempotent;
to start over, delete the store:  rm -rf var
"""

from __future__ import annotations

import argparse
from pathlib import Path

from storage import SQLiteDataStore, object_store_from_env
from features import FeatureExtractor
from datasets import PHM_CHANNELS, iter_cut_files
from ingest import IngestHandler, replay_run
from twin import StubTwin


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default="c1", help="PHM record folder, e.g. c1")
    ap.add_argument("--limit", type=int, default=None, help="only the first N cuts")
    ap.add_argument("--data-dir", default="data/phm2010")
    ap.add_argument("--store-dir", default="var")
    args = ap.parse_args()

    csv_folder = Path(args.data_dir) / args.record / args.record  # Kaggle's doubled nesting
    cut_files = iter_cut_files(csv_folder)
    if not cut_files:
        raise SystemExit(f"no cut files found under {csv_folder}")
    print(f"found {len(cut_files)} cut files in {csv_folder}")

    store_dir = Path(args.store_dir)
    data_store = SQLiteDataStore(store_dir / "kaful.db")
    object_store = object_store_from_env(store_dir)
    handler = IngestHandler(data_store, object_store, FeatureExtractor(PHM_CHANNELS), StubTwin())

    run_id = args.record
    replay_run(handler, data_store, cut_files=cut_files, machine_id="phm2010",
               run_id=run_id, machine_type="phm2010_milling", limit=args.limit)

    print(f"\nfeature rows: {len(data_store.read_all_features(run_id))}"
          f"   rul rows: {len(data_store.read_all_rul(run_id))}")
    data_store.close()


if __name__ == "__main__":
    main()
