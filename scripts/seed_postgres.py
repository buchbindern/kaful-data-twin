"""
Seed a Postgres database with the committed reference data (c1/c4/c6 on the system
machine), copied from the slim SQLite seed (deploy/kaful.db). Idempotent: runs that
already exist are skipped. Runs as a deploy step so a fresh/empty Postgres gets the
demo tools; user accounts and machines live in Postgres and persist across deploys.

    DATABASE_URL=postgres://... python scripts/seed_postgres.py [--src deploy/kaful.db]
"""

from __future__ import annotations

import argparse
import os

from storage import SQLiteDataStore
from storage.postgres_data_store import PostgresDataStore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="deploy/kaful.db")
    ap.add_argument("--dsn", default=os.environ.get("DATABASE_URL"))
    args = ap.parse_args()
    if not args.dsn:
        raise SystemExit("no DATABASE_URL / --dsn")

    src = SQLiteDataStore(args.src)
    dst = PostgresDataStore(args.dsn)
    seeded = []
    for m in src.list_machines():
        if dst.get_machine(m.machine_id) is None:
            dst.create_machine(m)               # preserves owner_id (None = system)
        for r in src.list_runs(m.machine_id):
            if dst.get_run(r.run_id) is not None:
                continue                         # already seeded
            dst.create_run(r)                    # preserves ended_at
            for f in src.read_all_features(r.run_id):
                cut = src.get_cut(r.run_id, f.cut_index)
                if cut is not None:
                    dst.append_cut(cut)
                dst.append_features(f)
            for lab in src.read_wear_labels(r.run_id):
                dst.append_wear_label(lab)
            seeded.append(r.run_id)
    dst.close(); src.close()
    print(f"seeded runs: {', '.join(seeded) if seeded else '(none — already present)'}")


if __name__ == "__main__":
    main()
