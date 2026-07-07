"""
Reproduce the cross-tool transfer findings (TRANSFER-FINDINGS.md). Run from repo root:

    python scripts/diagnose_transfer.py            # needs c1/c4/c6 features + labels

Runs three experiments that together explain why a reference model does not transfer
across tools, and what does fix it:
  1. RATE vs OBS-MAP decomposition — which one, when swapped, recovers accuracy.
  2. FEATURE transferability — which sensor feature transfers best across tools.
  3. FEW-SHOT calibration — how many target wear measurements recover accuracy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

from storage import SQLiteDataStore
from twin import (fit_model_spec, deploy_twin, deploy_with_measurements, models_from_state,
                  ParticleCloud, filter_step, PowerLawWear, PowerLawObservation, DEFAULT_FEATURE)
from evaluation import rmse, coverage

REFS = ("c1", "c4", "c6")


def onset(ds, run):
    lab = {l.cut_index: l.wear_mm for l in ds.read_wear_labels(run)}
    lc = sorted(lab); _, info = PowerLawWear.fit(lc, [lab[c] for c in lc])
    return info["onset_cut"]


def score(ds, target, deg_spec, obs_spec, feature, sigma_scale=2.5):
    hybrid = dict(deg_spec); hybrid["observation"] = obs_spec["observation"]
    hybrid["feature_name"] = feature
    state = deploy_twin(hybrid, target, n_particles=2000)
    cloud = ParticleCloud.from_bytes(state.particles); deg, obs = models_from_state(state)
    obs.sigma *= sigma_scale; rng = np.random.default_rng(0)
    lab = {l.cut_index: l.wear_mm for l in ds.read_wear_labels(target)}
    C, WE, WLO, WHI = [], [], [], []
    for f in ds.read_all_features(target):
        cloud = filter_step(cloud, deg, obs, f.features[feature], process_noise=0.002, rng=rng)
        C.append(f.cut_index); WE.append(cloud.mean_wear())
        WLO.append(cloud.quantile_wear(0.05)); WHI.append(cloud.quantile_wear(0.95))
    C, WE, WLO, WHI = map(np.array, (C, WE, WLO, WHI))
    WT = np.array([lab[c] for c in C]); wo = C >= onset(ds, target)
    return rmse(WE[wo], WT[wo]) * 1000, coverage(WT[wo], WLO[wo], WHI[wo])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store-dir", default="var")
    ap.add_argument("--feature", default=DEFAULT_FEATURE)
    args = ap.parse_args()
    ds = SQLiteDataStore(Path(args.store_dir) / "kaful.db")
    spec = {r: fit_model_spec(ds, r, feature_name=args.feature) for r in REFS}

    print(f"feature: {args.feature}\n")
    print("obs-map shape (k exponent) per tool:",
          {r: round(spec[r]["observation"]["k"], 2) for r in REFS})

    print("\n[1] RATE vs OBS-MAP (wear-out RMSE µm / coverage; reference c1)")
    for tgt in ("c4", "c6"):
        print(f"  target {tgt}:")
        for rate_lbl, rate in (("c1 rate", "c1"), (f"{tgt} rate", tgt)):
            cells = []
            for obs in ("c1", tgt):
                r, c = score(ds, tgt, spec[rate], spec[obs], args.feature)
                cells.append(f"{r:5.1f}/{c:.2f}")
            print(f"    {rate_lbl:10} | c1 obs {cells[0]}   own obs {cells[1]}")

    print("\n[2] FEATURE transferability (c1 map inverted on c4/c6, wear-out RMSE µm)")
    def transfer(feat):
        recs = {r: (np.array([l.wear_mm for l in sorted(ds.read_wear_labels(r), key=lambda x: x.cut_index)]),
                    np.array([f.features[feat] for f in ds.read_all_features(r)])) for r in REFS}
        w1, y1 = recs["c1"]
        if np.std(y1) < 1e-9 or abs(np.corrcoef(w1, y1)[0, 1]) < 0.5: return None
        g = lambda w, f0, c, k: f0 + c * np.power(np.clip(w, 1e-4, None), k)
        try: (f0, c, k), _ = curve_fit(g, w1, y1, p0=[y1.min(), 1.0, 1.0], maxfev=20000)
        except Exception: return None
        if c <= 0 or k <= 0: return None
        errs = []
        for r in ("c4", "c6"):
            w, y = recs[r]; on = onset(ds, r)
            cuts = [f.cut_index for f in ds.read_all_features(r)]
            iw = np.clip(np.clip((y - f0) / c, 1e-9, None) ** (1 / k), 1e-4, 0.5)
            wo = np.array([cc >= on for cc in cuts]); errs.append(rmse(iw[wo], w[wo]) * 1000)
        return float(np.mean(errs))
    feats = sorted(ds.read_all_features("c1")[0].features.keys())
    ranked = sorted(((transfer(f), f) for f in feats), key=lambda t: (t[0] is None, t[0]))
    for avg, f in [r for r in ranked if r[0] is not None][:6]:
        print(f"    {f:26} {avg:5.1f}")

    print("\n[3] FEW-SHOT calibration (K target wear measurements; wear-out RMSE µm / coverage)")
    for tgt in ("c4", "c6"):
        lab = {l.cut_index: l.wear_mm for l in ds.read_wear_labels(tgt)}
        cuts = sorted(lab)
        row = []
        for K in (0, 5, 10, 20):
            if K == 0:
                r, c = score(ds, tgt, spec["c1"], spec["c1"], args.feature)   # reference map
            else:
                idx = np.linspace(0, len(cuts) - 1, K).round().astype(int)
                meas = {cuts[i]: lab[cuts[i]] for i in idx}
                st = deploy_with_measurements(ds, tgt, meas, reference_run_id="c1", feature_name=args.feature)
                ds.save_twin_state(st)
                _, obs = models_from_state(st)
                r, c = score(ds, tgt, spec["c1"], {"observation": {"c": obs.c, "k": obs.k, "sigma": obs.sigma, "f0": obs.f0}}, args.feature)
            row.append(f"K={K}:{r:.1f}/{c:.2f}")
        print(f"    {tgt}: " + "   ".join(row))
    ds.close()


if __name__ == "__main__":
    main()
