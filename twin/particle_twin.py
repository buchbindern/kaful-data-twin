"""
ParticleTwin (M6/M7) — the real Twin. Replaces StubTwin.

Each cut: load the persisted TwinState, run one filter step against the observed
feature (M6), project the posterior cloud to an RUL distribution via Monte Carlo
(M7), and persist the updated state. Satisfies the same Twin.update() contract the
handler already calls, so nothing upstream changes.
"""

from __future__ import annotations

import numpy as np

from domain.models import RULPrediction, TwinState
from twin.base import Twin
from twin.cloud import ParticleCloud
from twin.build import models_from_state
from twin.filter import filter_step
from twin.rul import project_rul


class ParticleTwin(Twin):
    def __init__(self, data_store, *, process_noise: float = 0.002,
                 resample_frac: float = 0.5, horizon: int = 500,
                 rul_samples: int = 1000, seed: int = 0) -> None:
        self.ds = data_store
        self.process_noise = process_noise
        self.resample_frac = resample_frac
        self.horizon = horizon
        self.rul_samples = rul_samples
        self.rng = np.random.default_rng(seed)
        # exposed for inspection after each update()
        self.last_wear_mean = None
        self.last_wear_lo = None
        self.last_wear_hi = None
        self.last_rul_censored = None

    def update(self, run_id: str, cut_index: int, features: dict[str, float]) -> RULPrediction:
        state = self.ds.load_twin_state(run_id)
        if state is None:
            raise ValueError(f"no twin state for run {run_id!r}; run build_twin first")
        cloud = ParticleCloud.from_bytes(state.particles)
        deg, obs = models_from_state(state)
        feature_name = state.params["feature_name"]
        threshold = state.params["threshold_mm"]
        observed = features[feature_name]

        cloud = filter_step(cloud, deg, obs, observed,
                            process_noise=self.process_noise, rng=self.rng,
                            resample_frac=self.resample_frac)

        self.last_wear_mean = cloud.mean_wear()
        self.last_wear_lo = cloud.quantile_wear(0.05)
        self.last_wear_hi = cloud.quantile_wear(0.95)

        # RUL: Monte Carlo forward simulation of the posterior cloud (M7)
        dist = project_rul(cloud, deg, threshold=threshold, process_noise=self.process_noise,
                           rng=self.rng, horizon=self.horizon, n_samples=self.rul_samples)
        self.last_rul_censored = dist.censored_frac

        self.ds.save_twin_state(TwinState(run_id, cut_index, params=state.params,
                                          particles=cloud.to_bytes()))
        return RULPrediction(run_id, cut_index, dist.median, dist.lower, dist.upper, ci_level=0.9)
