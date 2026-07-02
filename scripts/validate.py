"""
Validate the twin (M8). Run from repo root:

    python scripts/validate.py --record c1

Runs the filter over a labeled run and scores it in TWO tiers:

  TIER 1 (trustworthy, non-circular): wear accuracy vs OBSERVED labels, plus
          wear-CI coverage (the honest calibration test of the filter).
  TIER 2 (flagged): RUL metrics vs EXTRAPOLATED pseudo-truth (c1 never reaches
          the threshold), scored only in the wear-out region where RUL is emitted.
          These share the degradation model with the ground-truth extrapolation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from storage import SQLiteDataStore
from twin import build_twin, ParticleTwin, PowerLawWear
from evaluation import rmse, mae, coverage, alpha_lambda_accuracy, prognostic_horizon

THRESHOLD = 0.200


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default="c1")
    ap.add_argument("--feature", default="force_z_rms")
    ap.add_argument("--process-noise", type=float, default=0.002)
    ap.add_argument("--n-particles", type=int, default=2000)
    ap.add_argument("--onset", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--store-dir", default="var")
    args = ap.parse_args()

    ds = SQLiteDataStore(Path(args.store_dir) / "kaful.db")
    labels = {l.cut_index: l.wear_mm for l in ds.read_wear_labels(args.record)}

    lc = sorted(labels); lw = [labels[c] for c in lc]
    deg_lab, info = PowerLawWear.fit(lc, lw, onset_cut=args.onset)
    onset = info["onset_cut"]
    L_star = lc[-1] + deg_lab.cuts_to_threshold(lw[-1], THRESHOLD)

    ds.save_twin_state(build_twin(ds, args.record, feature_name=args.feature,
                                  n_particles=args.n_particles, onset_cut=args.onset))
    ds.clear_rul(args.record)
    twin = ParticleTwin(ds, process_noise=args.process_noise, seed=0)

    C, WE, WLO, WHI, RM, RLO, RHI, CEN = ([] for _ in range(8))
    for f in ds.read_all_features(args.record):
        rul = twin.update(args.record, f.cut_index, f.features)
        ds.append_rul(rul)
        C.append(f.cut_index); WE.append(twin.last_wear_mean)
        WLO.append(twin.last_wear_lo); WHI.append(twin.last_wear_hi)
        RM.append(rul.rul_median); RLO.append(rul.rul_lower); RHI.append(rul.rul_upper)
        CEN.append(twin.last_rul_censored)
    ds.close()

    C = np.array(C); WE = np.array(WE); WLO = np.array(WLO); WHI = np.array(WHI)
    RM = np.array(RM); RLO = np.array(RLO); RHI = np.array(RHI); CEN = np.array(CEN)
    WT = np.array([labels[c] for c in C])
    RT = np.maximum(L_star - C, 0.0)

    um = 1000.0
    ri = C < onset
    wo = C >= onset

    print(f"=== VALIDATION: run {args.record!r}, {len(C)} cuts "
          f"(process_noise={args.process_noise*1000:.1f}um, alpha={args.alpha}) ===")
    print(f"wear-out onset cut {onset:.0f}; extrapolated life L* ~ cut {L_star:.0f} "
          f"(c1 never crosses {THRESHOLD} mm -> RUL truth is EXTRAPOLATED)\n")

    print("TIER 1 - wear accuracy vs OBSERVED labels (non-circular, trustworthy):")
    print(f"  wear RMSE  overall {rmse(WE,WT)*um:5.1f} um   "
          f"running-in {rmse(WE[ri],WT[ri])*um:5.1f} um   wear-out {rmse(WE[wo],WT[wo])*um:5.1f} um")
    print(f"  wear MAE   overall {mae(WE,WT)*um:5.1f} um")
    print(f"  wear 90% CI coverage: {coverage(WT,WLO,WHI):.2f}  (target ~0.90)")

    region = CEN < 0.2
    print(f"\nTIER 2 - RUL vs EXTRAPOLATED pseudo-truth, wear-out region "
          f"({int(region.sum())} cuts, censored<0.2) [FLAGGED: shares degradation model]:")
    if region.sum() >= 5:
        print(f"  RUL RMSE {rmse(RM[region],RT[region]):5.1f} cuts   MAE {mae(RM[region],RT[region]):5.1f} cuts")
        print(f"  alpha-lambda accuracy (+/-{int(args.alpha*100)}%): {alpha_lambda_accuracy(RM[region],RT[region],args.alpha):.2f}")
        print(f"  RUL 90% CI coverage: {coverage(RT[region],RLO[region],RHI[region]):.2f}")
        ph = prognostic_horizon(C[region], RM[region], RT[region], args.alpha)
        if ph is not None:
            print(f"  prognostic horizon: locks into +/-{int(args.alpha*100)}% band at cut {ph} "
                  f"-> {L_star-ph:.0f} cuts before end-of-life")
        else:
            print(f"  prognostic horizon: never stably locks into the band")
    else:
        print("  (too few non-censored cuts to score)")


if __name__ == "__main__":
    main()
