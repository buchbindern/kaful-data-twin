"""
Fit the observation model (feature | wear) on a labeled run. Run from repo root:

    python scripts/fit_observation.py --record c1 [--feature force_z_rms]

Aligns stored features with wear labels, fits feature = c*wear^k + noise, and
reports the fit plus a sanity check that an observed feature makes the likelihood
peak near the wear that produced it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from storage import SQLiteDataStore
from twin import PowerLawObservation


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default="c1")
    ap.add_argument("--feature", default="force_z_rms")
    ap.add_argument("--store-dir", default="var")
    args = ap.parse_args()

    ds = SQLiteDataStore(Path(args.store_dir) / "kaful.db")
    feats = ds.read_all_features(args.record)
    labels = {l.cut_index: l.wear_mm for l in ds.read_wear_labels(args.record)}
    ds.close()

    pairs = [(labels[f.cut_index], f.features[args.feature])
             for f in feats if f.cut_index in labels and args.feature in f.features]
    if len(pairs) < 10:
        raise SystemExit("not enough aligned (feature, label) pairs; full ingest + load_labels first")
    wears, vals = map(np.array, zip(*pairs))

    om = PowerLawObservation.fit(wears, vals, args.feature)
    print(f"observation model for {args.feature!r} on {len(pairs)} cuts:")
    print(f"  g(wear) = {om.f0:.2f} + {om.c:.2f} * wear^{om.k:.3f}     noise sigma = {om.sigma:.3f}")
    print(f"  correlation(feature, wear) = {np.corrcoef(vals, wears)[0,1]:+.3f}")

    obs = vals[np.argmax(wears)]
    grid = np.linspace(wears.min(), 0.20, 200)
    implied = grid[np.argmax(om.log_likelihood(obs, grid))]
    print(f"  observed feature at max wear {wears.max():.3f} mm -> likelihood implies {implied:.3f} mm")


if __name__ == "__main__":
    main()
