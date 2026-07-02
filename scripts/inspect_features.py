"""
Rank features by how strongly they track wear on a labeled run. Run from repo root:

    python scripts/inspect_features.py --record c1

Aligns each cut's features with its wear label and computes Pearson correlation.
This is how we pick the observation model's health indicator(s) EMPIRICALLY,
rather than assuming force_z is best just because the literature says so.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from storage import SQLiteDataStore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default="c1")
    ap.add_argument("--store-dir", default="var")
    args = ap.parse_args()

    ds = SQLiteDataStore(Path(args.store_dir) / "kaful.db")
    feats = ds.read_all_features(args.record)
    labels = {l.cut_index: l.wear_mm for l in ds.read_wear_labels(args.record)}
    ds.close()

    rows = [(f.cut_index, f.features, labels[f.cut_index])
            for f in feats if f.cut_index in labels]
    if len(rows) < 10:
        raise SystemExit(f"only {len(rows)} cuts have BOTH features and labels; "
                         f"run a full ingest (run_replay.py) and load_labels.py first")
    print(f"aligned {len(rows)} cuts (features x labels)")

    wear = np.array([r[2] for r in rows])
    names = sorted(rows[0][1].keys())
    corrs = []
    for name in names:
        vals = np.array([r[1][name] for r in rows])
        r = 0.0 if np.std(vals) == 0 else float(np.corrcoef(vals, wear)[0, 1])
        corrs.append((name, r))
    corrs.sort(key=lambda x: -abs(x[1]))

    print("\ntop 12 features by |correlation| with wear:")
    for name, r in corrs[:12]:
        bar = "#" * int(abs(r) * 40)
        print(f"  {name:24s} r={r:+.3f}  {bar}")

    fz = next((r for n, r in corrs if n == "force_z_rms"), None)
    rank = next((i for i, (n, _) in enumerate(corrs) if n == "force_z_rms"), None)
    if fz is not None:
        print(f"\nforce_z_rms: r={fz:+.3f}, ranked #{rank+1} of {len(corrs)}")

    top = corrs[0][0]
    tv = np.array([r[1][top] for r in rows])
    order = np.argsort(wear)
    print(f"\n{top} vs wear (sorted by wear, ~10 buckets):")
    buckets = np.array_split(order, 10)
    for b in buckets:
        print(f"  wear {wear[b].mean()*1000:6.1f} um   {top} {tv[b].mean():10.4f}")


if __name__ == "__main__":
    main()
