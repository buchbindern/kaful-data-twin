"""
Particle filter mechanics (M6) — one predict/update/resample step.

Pure functions over a ParticleCloud + the two models. This is the conceptual
heart of the system:

  PREDICT   advance every wear particle one cut via the degradation model,
            then add process noise (models degradation-dynamics uncertainty and
            keeps the cloud diverse enough to track running-in, where the
            wear-out model under-drives the motion).
  UPDATE    reweight each particle by how well its wear explains the observed
            force, via the observation model's likelihood.
  RESAMPLE  when the effective sample size collapses, resample so particles
            concentrate where the posterior mass is (kills degeneracy).
"""

from __future__ import annotations

import numpy as np

from twin.cloud import ParticleCloud

_WEAR_MIN = 1e-4
_WEAR_MAX = 1.0     # mm; caps the finite-time singularity so no inf leaks in


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = weights.size
    positions = (rng.random() + np.arange(n)) / n
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0
    return np.searchsorted(cumsum, positions)


def filter_step(cloud: ParticleCloud, deg, obs, observed_feature: float, *,
                process_noise: float, rng: np.random.Generator,
                resample_frac: float = 0.5) -> ParticleCloud:
    # PREDICT
    w = deg.advance(cloud.wear, 1.0)
    w = np.clip(w + rng.normal(0.0, process_noise, w.size), _WEAR_MIN, _WEAR_MAX)

    # UPDATE (work in log space for numerical stability)
    logw = np.log(np.clip(cloud.weights, 1e-300, None)) + obs.log_likelihood(observed_feature, w)
    logw -= logw.max()
    weights = np.exp(logw)
    total = weights.sum()
    weights = weights / total if total > 0 else np.full(w.size, 1.0 / w.size)

    # RESAMPLE if effective sample size is low
    ess = 1.0 / np.sum(weights ** 2)
    if ess < resample_frac * w.size:
        idx = systematic_resample(weights, rng)
        return ParticleCloud(w[idx], np.full(w.size, 1.0 / w.size))
    return ParticleCloud(w, weights)
