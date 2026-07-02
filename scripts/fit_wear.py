"""
Fit the power-law wear model to a run's stored wear labels. Run from repo root:

    python scripts/fit_wear.py --record c1

Reads labels from the runtime store (populate them first with load_labels.py).
Reports the wear-out onset, fitted (a, p), fit RMSE, and the EXTRAPOLATED
threshold-crossing cut (pseudo-ground-truth RUL — never observed in c1).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from storage import SQLiteDataStore
from twin import PowerLawWear

THRESHOLD_MM = 0.200  # ISO 8688


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default="c1")
    ap.add_argument("--store-dir", default="var")
    ap.add_argument("--onset", type=float, default=None, help="override wear-out onset cut")
    args = ap.parse_args()

    ds = SQLiteDataStore(Path(args.store_dir) / "kaful.db")
    labels = ds.read_wear_labels(args.record)
    ds.close()
    if not labels:
        raise SystemExit(f"no wear labels for run {args.record!r}; run load_labels.py first")

    cuts = [l.cut_index for l in labels]
    wears = [l.wear_mm for l in labels]
    model, info = PowerLawWear.fit(cuts, wears, onset_cut=args.onset)

    last_cut, last_wear = cuts[-1], wears[-1]
    crossing = last_cut + model.cuts_to_threshold(last_wear, THRESHOLD_MM)
    print(f"fit on {info['n_fit_points']} points, wear-out onset at cut {info['onset_cut']:.0f}")
    print(f"  dw/dn = a*w^p   a={model.a:.3e}   p={model.p:.3f}")
    print(f"  fit RMSE = {info['rmse_um']:.2f} um")
    print(f"  last observed: cut {last_cut}, wear {last_wear:.3f} mm")
    print(f"  EXTRAPOLATED threshold (0.200 mm) crossing at cut ~{crossing:.0f}"
          f"   ->  pseudo-RUL at last cut ~{crossing - last_cut:.0f} cuts")


if __name__ == "__main__":
    main()
