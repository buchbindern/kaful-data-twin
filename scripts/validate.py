"""
Validate the twin (M8, cross-run at M10). Run from repo root:

    python scripts/validate.py --record c1                      # self (fit & score on c1)
    python scripts/validate.py --record c4 --reference c1       # fit on c1, score on c4

TIER 1 (trustworthy, non-circular): wear accuracy vs OBSERVED labels + CI coverage.
TIER 2: RUL vs ground truth. If the run actually reaches the 0.2mm threshold, truth
        is OBSERVED (real accuracy). If not, truth is EXTRAPOLATED and — when fit and
        score are the SAME run — shares the degradation model (flagged). A cross-run
        test (reference != record) removes that circularity.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from storage import SQLiteDataStore
from twin import build_twin, deploy_from_reference, ParticleTwin, PowerLawWear
from evaluation import rmse, mae, coverage, alpha_lambda_accuracy, prognostic_horizon

THRESHOLD = 0.200


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default="c1", help="run to score (features + labels)")
    ap.add_argument("--reference", default=None, help="run to fit the twin on (default: --record)")
    ap.add_argument("--feature", default="force_z_rms")
    ap.add_argument("--process-noise", type=float, default=0.002)
    ap.add_argument("--sigma-scale", type=float, default=2.5)
    ap.add_argument("--n-particles", type=int, default=2000)
    ap.add_argument("--onset", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--store-dir", default="var")
    args = ap.parse_args()
    reference = args.reference or args.record
    cross = reference != args.record

    ds = SQLiteDataStore(Path(args.store_dir) / "kaful.db")
    labels = {l.cut_index: l.wear_mm for l in ds.read_wear_labels(args.record)}
    if not labels:
        raise SystemExit(f"no labels for {args.record!r}")

    lc = sorted(labels); lw = [labels[c] for c in lc]
    deg_lab, info = PowerLawWear.fit(lc, lw, onset_cut=args.onset)
    onset = info["onset_cut"]

    crossings = [c for c, w in zip(lc, lw) if w >= THRESHOLD]
    if crossings:
        L_star = float(crossings[0]); truth_kind = "OBSERVED"
    else:
        L_star = lc[-1] + deg_lab.cuts_to_threshold(lw[-1], THRESHOLD); truth_kind = "EXTRAPOLATED"

    if cross:
        ds.save_twin_state(deploy_from_reference(ds, reference, args.record,
                                                 feature_name=args.feature,
                                                 n_particles=args.n_particles, onset_cut=args.onset))
    else:
        ds.save_twin_state(build_twin(ds, args.record, feature_name=args.feature,
                                      n_particles=args.n_particles, onset_cut=args.onset))
    ds.clear_rul(args.record)
    twin = ParticleTwin(ds, process_noise=args.process_noise, sigma_scale=args.sigma_scale, seed=0)

    C, WE, WLO, WHI, RM, RLO, RHI, CEN = ([] for _ in range(8))
    for f in ds.read_all_features(args.record):
        if f.cut_index not in labels:
            continue  # score only labeled cuts (in a live run, labels can lag features)
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
    ri = C < onset; wo = C >= onset

    fit_note = f"fit on {reference!r} -> scored on {args.record!r}" if cross else f"self ({args.record!r})"
    print(f"=== VALIDATION: {fit_note}, {len(C)} cuts "
          f"(sigma_scale={args.sigma_scale}, alpha={args.alpha}) ===")
    print(f"wear-out onset cut {onset:.0f}; life {'at' if truth_kind=='OBSERVED' else '~'} "
          f"cut {L_star:.0f}  [RUL ground truth: {truth_kind}]\n")

    print("TIER 1 - wear accuracy vs OBSERVED labels (non-circular, trustworthy):")
    print(f"  wear RMSE  overall {rmse(WE,WT)*um:5.1f} um   "
          f"running-in {rmse(WE[ri],WT[ri])*um:5.1f} um   wear-out {rmse(WE[wo],WT[wo])*um:5.1f} um")
    print(f"  wear MAE   overall {mae(WE,WT)*um:5.1f} um")
    print(f"  wear 90% CI coverage: overall {coverage(WT,WLO,WHI):.2f}   "
          f"running-in {coverage(WT[ri],WLO[ri],WHI[ri]):.2f}   "
          f"wear-out {coverage(WT[wo],WLO[wo],WHI[wo]):.2f}   (target ~0.90)")

    region = CEN < 0.2
    if truth_kind == "OBSERVED":
        flag = "OBSERVED truth" + (" (cross-run: no circularity)" if cross else "")
    else:
        flag = "EXTRAPOLATED truth" + ("; cross-run" if cross else "; shares degradation model")
    print(f"\nTIER 2 - RUL vs ground truth, wear-out region "
          f"({int(region.sum())} cuts, censored<0.2) [{flag}]:")
    if region.sum() >= 5:
        print(f"  RUL RMSE {rmse(RM[region],RT[region]):5.1f} cuts   MAE {mae(RM[region],RT[region]):5.1f} cuts")
        print(f"  alpha-lambda accuracy (+/-{int(args.alpha*100)}%): {alpha_lambda_accuracy(RM[region],RT[region],args.alpha):.2f}")
        print(f"  RUL 90% CI coverage: {coverage(RT[region],RLO[region],RHI[region]):.2f}")
        ph = prognostic_horizon(C[region], RM[region], RT[region], args.alpha)
        print(f"  prognostic horizon: " + (f"locks into band at cut {ph} -> {L_star-ph:.0f} cuts before EOL"
                                           if ph is not None else "never stably locks into the band"))
    else:
        print("  (too few non-censored cuts to score)")


if __name__ == "__main__":
    main()