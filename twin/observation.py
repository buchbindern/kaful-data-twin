"""
PowerLawObservation (M5c) — the observation model.

Maps hidden wear to an expected sensor feature and gives the likelihood the
particle filter uses to weight particles:

    feature ~ Normal( g(wear), sigma ),   g(wear) = c * wear**k

Chosen because on the c1 reference run the force health indicator vs wear is
monotonic and convex (k>1), and a power law captures that with two parameters
while staying invertible (each wear -> a unique expected feature). Fitted on a
LABELED reference run (c1); an unlabeled deployment reuses the fitted mapping.

Primary indicator: force_z_rms (thrust). The handoff flags force_x/force_y as
process-confounded and AE as weak on c1 — force_z is the principled choice.
The model is single-feature for now; multiple independent modalities sum their
log-likelihoods, a later extension.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

_LOG_2PI = float(np.log(2 * np.pi))


@dataclass
class PowerLawObservation:
    feature_name: str
    c: float
    k: float
    sigma: float          # residual std of the feature around g(wear)

    def expected(self, wear):
        """E[feature | wear] = c * wear**k. Scalar or array."""
        return self.c * np.power(np.asarray(wear, float), self.k)

    def log_likelihood(self, observed_feature: float, wear):
        """log p(observed_feature | wear), vectorized over a wear particle cloud."""
        mu = self.expected(wear)
        z = (observed_feature - mu) / self.sigma
        return -0.5 * z * z - np.log(self.sigma) - 0.5 * _LOG_2PI

    @classmethod
    def fit(cls, wears, feature_values, feature_name: str) -> "PowerLawObservation":
        wears = np.asarray(wears, float)
        feats = np.asarray(feature_values, float)
        pw = lambda w, c, k: c * np.power(w, k)
        (c, k), _ = curve_fit(pw, wears, feats, p0=[100.0, 1.3], maxfev=200000)
        sigma = float(np.std(feats - pw(wears, c, k)))
        return cls(feature_name, float(c), float(k), sigma)
