"""
Twin build / deploy (M5d, split into fit + deploy at M10).

Two separable operations, per the handoff's fit-once / deploy-per-tool story:

  fit_model_spec(reference_run)  -> a reusable dict of fitted params (degradation,
                                    observation, threshold). Fit ONCE on a labeled
                                    reference tool.
  deploy_twin(spec, target_run)  -> a fresh cold-start TwinState for a new tool
                                    life (new particle cloud at ~zero wear),
                                    carrying the reference spec.

build_twin() keeps the M5d behavior (fit AND deploy on the same run) as a
convenience; deploy_from_reference() is the cross-tool path (fit on c1, deploy to
c4/c6, or to a fresh run after a tool change).

The initial wear prior is deliberately broad — early RUL is wide and tightens as
data accumulates (handoff decision #8), not a defect.
"""

from __future__ import annotations

import numpy as np

from domain.models import TwinState
from twin.degradation import PowerLawWear
from twin.observation import PowerLawObservation
from twin.cloud import ParticleCloud

# Most transferable wear indicator across tools (see TRANSFER-FINDINGS.md).
# vibration_x_mean_abs beats force_z_rms both within-tool and, especially,
# when a reference model is deployed onto a different tool.
DEFAULT_FEATURE = "vibration_x_mean_abs"


def fit_model_spec(data_store, reference_run_id: str, *, feature_name: str = DEFAULT_FEATURE,
                   threshold_mm: float = 0.200, onset_cut: float | None = None) -> dict:
    """Fit degradation + observation models on a labeled reference run. Returns a
    reusable model spec (no particle cloud, not stamped to any run)."""
    labels = data_store.read_wear_labels(reference_run_id)
    if not labels:
        raise ValueError(f"no wear labels for run {reference_run_id!r}")
    cuts = [l.cut_index for l in labels]
    wears = [l.wear_mm for l in labels]
    deg, deg_info = PowerLawWear.fit(cuts, wears, onset_cut=onset_cut)

    feats = data_store.read_all_features(reference_run_id)
    lab = {l.cut_index: l.wear_mm for l in labels}
    pairs = [(lab[f.cut_index], f.features[feature_name])
             for f in feats if f.cut_index in lab and feature_name in f.features]
    if len(pairs) < 10:
        raise ValueError("not enough aligned (feature, label) pairs to fit observation model")
    ws, vs = map(np.array, zip(*pairs))
    obs = PowerLawObservation.fit(ws, vs, feature_name)

    return {
        "feature_name": feature_name,
        "threshold_mm": threshold_mm,
        "degradation": {"a": deg.a, "p": deg.p},
        "observation": {"c": obs.c, "k": obs.k, "sigma": obs.sigma, "f0": obs.f0},
        "onset_cut": deg_info["onset_cut"],
        "reference_run_id": reference_run_id,   # provenance: which run this was fit on
    }


def deploy_twin(model_spec: dict, target_run_id: str, *, n_particles: int = 2000,
                seed: int = 0) -> TwinState:
    """Seed a fresh cold-start TwinState for `target_run_id` from a fitted spec."""
    rng = np.random.default_rng(seed)
    wear0 = rng.uniform(0.005, 0.08, n_particles)
    cloud = ParticleCloud(wear0, np.full(n_particles, 1.0 / n_particles))
    params = dict(model_spec)
    params["n_particles"] = n_particles
    return TwinState(run_id=target_run_id, cut_index=0, params=params,
                     particles=cloud.to_bytes())


def deploy_from_reference(data_store, reference_run_id: str, target_run_id: str, *,
                          feature_name: str = DEFAULT_FEATURE, n_particles: int = 2000,
                          threshold_mm: float = 0.200, onset_cut: float | None = None,
                          seed: int = 0) -> TwinState:
    """Fit on `reference_run_id`, deploy a fresh twin onto `target_run_id`."""
    spec = fit_model_spec(data_store, reference_run_id, feature_name=feature_name,
                          threshold_mm=threshold_mm, onset_cut=onset_cut)
    return deploy_twin(spec, target_run_id, n_particles=n_particles, seed=seed)


def build_twin(data_store, reference_run_id: str, *, feature_name: str = DEFAULT_FEATURE,
               n_particles: int = 2000, threshold_mm: float = 0.200,
               onset_cut: float | None = None, seed: int = 0) -> TwinState:
    """M5d convenience: fit AND deploy on the SAME run (self-reference)."""
    return deploy_from_reference(data_store, reference_run_id, reference_run_id,
                                 feature_name=feature_name, n_particles=n_particles,
                                 threshold_mm=threshold_mm, onset_cut=onset_cut, seed=seed)


def deploy_with_measurements(data_store, target_run_id: str,
                             wear_measurements: dict[int, float], *, reference_run_id: str,
                             feature_name: str = DEFAULT_FEATURE, n_particles: int = 2000,
                             threshold_mm: float = 0.200, seed: int = 0) -> TwinState:
    """Few-shot calibration (TRANSFER-FINDINGS.md).

    A reference model does NOT transfer to a new tool: the wear->signal map is
    tool-specific in shape and unidentifiable from signals alone. But a handful of
    actual wear measurements on the target tool identify ITS OWN observation map,
    which recovers calibrated accuracy (~10 measurements: broken -> ~0.90 coverage).
    The degradation RATE is taken from a reference run (rate does not limit transfer,
    measured three ways). `wear_measurements` maps cut_index -> measured wear (mm).
    """
    feats = {f.cut_index: f.features for f in data_store.read_all_features(target_run_id)}
    pairs = [(w, feats[c][feature_name]) for c, w in wear_measurements.items()
             if c in feats and feature_name in feats[c]]
    if len(pairs) < 4:
        raise ValueError(f"need >=4 aligned (measurement, feature) pairs; got {len(pairs)}")
    ws, vs = map(np.array, zip(*pairs))
    obs = PowerLawObservation.fit(ws, vs, feature_name)

    ref_labels = data_store.read_wear_labels(reference_run_id)
    if not ref_labels:
        raise ValueError(f"reference run {reference_run_id!r} has no labels for a degradation rate")
    deg, deg_info = PowerLawWear.fit([l.cut_index for l in ref_labels],
                                     [l.wear_mm for l in ref_labels])
    spec = {
        "feature_name": feature_name,
        "threshold_mm": threshold_mm,
        "degradation": {"a": deg.a, "p": deg.p},
        "observation": {"c": obs.c, "k": obs.k, "sigma": obs.sigma, "f0": obs.f0},
        "onset_cut": deg_info["onset_cut"],
        "reference_run_id": reference_run_id,
        "calibrated_from": f"{len(pairs)} wear measurements on {target_run_id}",
    }
    return deploy_twin(spec, target_run_id, n_particles=n_particles, seed=seed)


def models_from_state(state: TwinState) -> tuple[PowerLawWear, PowerLawObservation]:
    """Reconstruct the fixed models from a persisted TwinState (used by M6/M7)."""
    d = state.params
    deg = PowerLawWear(d["degradation"]["a"], d["degradation"]["p"])
    o = d["observation"]
    obs = PowerLawObservation(d["feature_name"], o["c"], o["k"], o["sigma"], o.get("f0", 0.0))
    return deg, obs
