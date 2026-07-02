"""
Cold-start twin build (M5d).

build_twin() fits both models on a LABELED reference run and produces the initial
TwinState (cut_index 0, before any deployed cut): a cold-start particle cloud plus
the fixed model parameters. M6 loads this and updates it cut by cut.

The initial wear prior is deliberately broad (a fresh tool with uncertain running-in
wear) — early RUL will therefore be wide and tighten as data accumulates, which is
the correct honest behavior (handoff decision #8), not a defect.
"""

from __future__ import annotations

import numpy as np

from domain.models import TwinState
from twin.degradation import PowerLawWear
from twin.observation import PowerLawObservation
from twin.cloud import ParticleCloud


def build_twin(data_store, reference_run_id: str, *, feature_name: str = "force_z_rms",
               n_particles: int = 2000, threshold_mm: float = 0.200,
               onset_cut: float | None = None, seed: int = 0) -> TwinState:
    # 1. degradation model from wear labels
    labels = data_store.read_wear_labels(reference_run_id)
    if not labels:
        raise ValueError(f"no wear labels for run {reference_run_id!r}")
    cuts = [l.cut_index for l in labels]
    wears = [l.wear_mm for l in labels]
    deg, deg_info = PowerLawWear.fit(cuts, wears, onset_cut=onset_cut)

    # 2. observation model from aligned features + labels
    feats = data_store.read_all_features(reference_run_id)
    lab = {l.cut_index: l.wear_mm for l in labels}
    pairs = [(lab[f.cut_index], f.features[feature_name])
             for f in feats if f.cut_index in lab and feature_name in f.features]
    if len(pairs) < 10:
        raise ValueError("not enough aligned (feature, label) pairs to fit observation model")
    ws, vs = map(np.array, zip(*pairs))
    obs = PowerLawObservation.fit(ws, vs, feature_name)

    # 3. cold-start particle cloud: broad prior over initial wear
    rng = np.random.default_rng(seed)
    wear0 = rng.uniform(0.005, 0.08, n_particles)
    cloud = ParticleCloud(wear0, np.full(n_particles, 1.0 / n_particles))

    # 4. pack fixed models + cloud into a TwinState
    params = {
        "feature_name": feature_name,
        "threshold_mm": threshold_mm,
        "degradation": {"a": deg.a, "p": deg.p},
        "observation": {"c": obs.c, "k": obs.k, "sigma": obs.sigma},
        "n_particles": n_particles,
        "onset_cut": deg_info["onset_cut"],
    }
    return TwinState(run_id=reference_run_id, cut_index=0, params=params,
                     particles=cloud.to_bytes())


def models_from_state(state: TwinState) -> tuple[PowerLawWear, PowerLawObservation]:
    """Reconstruct the fixed models from a persisted TwinState (used by M6/M7)."""
    d = state.params
    deg = PowerLawWear(d["degradation"]["a"], d["degradation"]["p"])
    obs = PowerLawObservation(d["feature_name"], d["observation"]["c"],
                              d["observation"]["k"], d["observation"]["sigma"])
    return deg, obs
