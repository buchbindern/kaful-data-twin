"""
Calibrate the filter's observation-noise scale (M8). Run from repo root:

    python scripts/calibrate.py --record c1

Sweeps sigma_scale and reports wear-CI coverage (overall / running-in / wear-out)
and wear RMSE for each. Runs the FILTER only (skips RUL Monte Carlo) for speed.
Pick the smallest scale whose coverage is ~0.90 where it matters (wear-out) without
badly inflating RMSE — that's the honest, calibrated trust level for observations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from storage import SQLiteDataStore
from twin import (build_twin, models_from_state, ParticleCloud, filter_step, PowerLawWear)
from evaluation import coverage, rmse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default="c1")
    ap.add_argument("--feature", default="force_z_rms")
    ap.add_argument("--process-noise", type=float, default=0.002)
    ap.add_argument("--n-particles", type=int, default=2000)
    ap.add_argument("--scales", default="1,1.5,2,2.5,3,4")
    ap.add_argument("--store-dir", default="var")
    args = ap.parse_args()

    ds = SQLiteDataStore(Path(args.store_dir) / "kaful.db")
    labels = {l.cut_index: l.wear_mm for l in ds.read_wear_labels(args.record)}
    lc = sorted(labels); lw = [labels[c] for c in lc]
    _, info = PowerLawWear.fit(lc, lw)
    onset = info["onset_cut"]
    scales = [float(x) for x in args.scales.split(",")]

    print(f"calibrating {args.record!r} (process_noise={args.process_noise*1000:.1f}um, "
          f"wear-out onset cut {onset:.0f})\n")
    print(f"{'sig_scale':>9} {'cov_all':>8} {'cov_run-in':>11} {'cov_wear-out':>13} "
          f"{'rmse_all_um':>12} {'rmse_wo_um':>11}")

    feats = ds.read_all_features(args.record)
    best = None
    for ss in scales:
        state = build_twin(ds, args.record, feature_name=args.feature, n_particles=args.n_particles)
        cloud = ParticleCloud.from_bytes(state.particles)
        deg, obs = models_from_state(state)
        obs.sigma *= ss
        rng = np.random.default_rng(0)
        C, WT, WE, WLO, WHI = [], [], [], [], []
        for f in feats:
            cloud = filter_step(cloud, deg, obs, f.features[args.feature],
                                process_noise=args.process_noise, rng=rng)
            C.append(f.cut_index); WT.append(labels[f.cut_index])
            WE.append(cloud.mean_wear()); WLO.append(cloud.quantile_wear(0.05))
            WHI.append(cloud.quantile_wear(0.95))
        C, WT, WE, WLO, WHI = map(np.array, (C, WT, WE, WLO, WHI))
        ri = C < onset; wo = C >= onset
        cov_all = coverage(WT, WLO, WHI)
        print(f"{ss:>9.1f} {cov_all:>8.2f} {coverage(WT[ri],WLO[ri],WHI[ri]):>11.2f} "
              f"{coverage(WT[wo],WLO[wo],WHI[wo]):>13.2f} "
              f"{rmse(WE,WT)*1000:>12.1f} {rmse(WE[wo],WT[wo])*1000:>11.1f}")
        if best is None or abs(cov_all - 0.90) < abs(best[1] - 0.90):
            best = (ss, cov_all)
    ds.close()
    print(f"\nclosest overall coverage to 0.90: sigma_scale={best[0]:.1f} (coverage {best[1]:.2f})")


if __name__ == "__main__":
    main()
