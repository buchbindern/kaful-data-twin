"""
Run the particle filter over a run's stored features (M6). Run from repo root:

    python scripts/run_filter.py --record c1 [--process-noise 0.002]

Rebuilds a fresh cold-start twin, clears any prior RUL rows (e.g. the stub's),
then streams the stored features through the ParticleTwin cut by cut — updating
wear and writing real RUL. Reports wear tracking vs the ground-truth labels,
which is the M6 validation: does the filter recover wear from noisy force?
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from storage import SQLiteDataStore
from twin import build_twin, ParticleTwin


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default="c1")
    ap.add_argument("--feature", default="force_z_rms")
    ap.add_argument("--process-noise", type=float, default=0.002)
    ap.add_argument("--n-particles", type=int, default=2000)
    ap.add_argument("--onset", type=float, default=None)
    ap.add_argument("--store-dir", default="var")
    args = ap.parse_args()

    ds = SQLiteDataStore(Path(args.store_dir) / "kaful.db")
    labels = {l.cut_index: l.wear_mm for l in ds.read_wear_labels(args.record)}

    ds.save_twin_state(build_twin(ds, args.record, feature_name=args.feature,
                                  n_particles=args.n_particles, onset_cut=args.onset))
    ds.clear_rul(args.record)

    twin = ParticleTwin(ds, process_noise=args.process_noise, seed=0)
    feats = ds.read_all_features(args.record)

    rows = []
    for f in feats:
        rul = twin.update(args.record, f.cut_index, f.features)
        ds.append_rul(rul)
        rows.append((f.cut_index, twin.last_wear_mean, twin.last_wear_lo,
                     twin.last_wear_hi, labels.get(f.cut_index), rul.rul_median))

    errs = [(m - t) for _, m, _, _, t, _ in rows if t is not None]
    rmse = float(np.sqrt(np.mean(np.square(errs))) * 1000)
    print(f"ran filter over {len(rows)} cuts (process_noise={args.process_noise*1000:.1f} um)")
    print(f"wear-tracking RMSE vs ground truth: {rmse:.1f} um\n")
    print(f"{'cut':>4} {'true_um':>8} {'est_um':>8} {'est_90%_CI_um':>16} {'RUL_cuts':>9}")
    for cut, m, lo, hi, t, rul in rows[::21]:
        tt = f"{t*1000:6.1f}" if t is not None else "   -  "
        print(f"{cut:>4} {tt:>8} {m*1000:8.1f} {f'[{lo*1000:.1f},{hi*1000:.1f}]':>16} {rul:9.0f}")
    ds.close()


if __name__ == "__main__":
    main()
