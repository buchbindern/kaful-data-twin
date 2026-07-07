# Cross-Tool Transfer — Findings

## Summary

The committed milestone (`HANDOFF-FILTER.md`) planned to fix cross-tool transfer
failure by estimating the degradation parameters `(a, p)` per tool (joint
`(wear, a, p)` particle filtering with Liu-West shrinkage), on the theory that a
fixed degradation *rate* was the root cause.

**That theory is wrong, and the milestone is cancelled.** Measured three ways, the
degradation rate is irrelevant to transfer. The failure is entirely in the
**observation map** — the wear→signal relationship, which is tool-specific *in shape*
and not identifiable from signals alone. This document records the experiments
(reproducible via `scripts/diagnose_transfer.py`) and the two changes they motivated:
a better observation feature, and a few-shot calibration path.

## The problem

A twin fit on one tool (c1) and deployed on another fails badly. Scored against each
tool's own *observed* failure (c4 fails at cut 313, c6 at 291):

| deployment | wear-out RMSE | coverage |
|---|--:|--:|
| c1 → c1 (self) | ~6 µm | ~0.9 |
| c1 → c4 | 23 µm | 0.16 |
| c1 → c6 | 34 µm | 0.00 |

## Experiment 1 — rate vs observation map

For each target, run the filter under every combination of {c1 rate, target's own
rate} × {c1 obs map, target's own obs map}. Result (wear-out RMSE / coverage):

- Swapping in the target's own **rate** changes nothing (c6: 33.5 → 33.1, coverage
  0.00 → 0.01).
- Swapping in the target's own **observation map** — even keeping c1's *wrong* rate —
  recovers almost everything (c6: 33.5 → ~8–11, coverage 0.00 → ~0.7–1.0; c4 similar).

**Conclusion:** the transfer failure is an observation-map problem, not a rate
problem. Estimating `(a, p)` cannot fix it. Mechanism: in the wear-out regime the
observation likelihood dominates the posterior, so wear ≈ the observed signal inverted
through the obs map; the rate only shapes forward RUL projection.

The obs maps differ in *shape*: the fitted exponent `k` in `signal = f0 + c·wᵏ` is
roughly **2.7 / 1.4 / 1.4** across c1 / c4 / c6 (feature-dependent, but always
tool-specific). That is a different curve, not a rescaling — which is why the next two
experiments (normalization, mixture) don't help.

## Experiment 2 — what does NOT help

- **Fresh-baseline normalization** (divide each signal by its unworn-tool value, free
  since every tool starts fresh): no improvement. It removes tool-specific *scale*, but
  the difference is *shape*.
- **Mixture over reference maps** (deploy an ensemble of the known tools' maps):
  improves honesty slightly (leave-one-out coverage 0.30 → 0.69 on c1) but never
  calibrates and stays inaccurate (26–32 µm). You cannot average your way to a map you
  do not have.
- **Multi-channel fusion** (average implied wear across several features): modest —
  plateaus around ~18 µm.

## Experiment 3 — what DOES help

**(a) A more transferable feature.** `force_z_rms` is the best wear indicator *within*
a tool but among the *least* transferable across tools. Scanning all 42 features,
**`vibration_x_mean_abs`** transfers best and is also slightly better within-tool:

| feature | c1 self | c1 → c4 | c1 → c6 |
|---|--:|--:|--:|
| force_z_rms | 6.2 / 0.93 | 23.2 / 0.16 | 33.5 / 0.00 |
| **vibration_x_mean_abs** | **5.3 / 0.94** | **16.3 / 0.33** | **21.0 / 0.09** |

This is now the default observation feature. It is a free ~35–40% improvement on
unlabeled tools with no downside within-tool. (Caveat: chosen on 3 tools; re-validate
as more labeled tools become available.)

**(b) Few-shot calibration.** A handful of actual wear measurements on the target tool
identify *its own* observation map — the one thing signals alone cannot. With the
target's own map, accuracy is fully recovered (own-map c4: 6.2 µm / 0.99). In practice
~15–20 periodic wear checks are enough to fit a stable map and recover calibrated
accuracy; very few (≤5) can overfit the 3-parameter curve. Implemented as
`twin.deploy_with_measurements(...)`.

## The identifiability result

Why can't the map be learned online from signals alone? Fitting `(w0, a, p, f0, c, k)`
to a force trajectory with no wear labels is under-determined: fits within a few
percent of the best signal-RMSE imply final wear anywhere from ~0.3 mm to enormous
values (true ~0.2 mm). There is a gauge freedom in `signal = f0 + c·wᵏ` — you can
trade the wear scale against the map parameters and fit the signal equally well. So the
map is **unidentifiable from signals alone**; it needs an external anchor (wear
measurements) or, potentially, a genuinely tool-invariant feature (not found here).

## Deployment consequence (two modes)

- **Unlabeled new tool:** deploy the `vibration_x_mean_abs` reference model with
  honest (inflated) uncertainty. Best-effort ~16–21 µm — a real, measured improvement
  — with intervals wide enough to be truthful about what they don't know.
- **Tool with ~15–20 periodic wear checks:** `deploy_with_measurements` fits the
  tool's own map → calibrated accuracy. The interval visibly tightens as measurements
  arrive.

The unlabeled ceiling is *information-theoretic*, not a modeling shortfall: the
wear→signal map is tool-specific and unidentifiable from signals alone. The honest
system is one that is accurate where it can be and wide where it must be.

## Reproduce

    python scripts/diagnose_transfer.py     # experiments 1–3 on c1/c4/c6
