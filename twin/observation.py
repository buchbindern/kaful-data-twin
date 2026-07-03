"""
PowerLawObservation (M5c, +intercept at M8) — the observation model.

    feature ~ Normal( g(wear), sigma ),   g(wear) = f0 + c * wear**k

The intercept f0 was added at M8: real cutting force has a nonzero baseline even
on a sharp tool, so a pure power law through the origin mis-maps force to wear
(biased low at low wear, high in mid wear-out). f0 = 0 recovers the original
pure-power-law model, so persisted states without an f0 still load correctly.

Fitted on a LABELED reference run; an unlabeled deployment reuses the mapping.
sigma is the residual std of the feature around g(wear) — note it reflects
training scatter, NOT model bias, which is why calibrating the filter may still
need to inflate the effective noise (a separate M8 lever).
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
    f0: float = 0.0       # baseline (intercept); 0.0 = pure power law

    def expected(self, wear):
        """E[feature | wear] = f0 + c * wear**k. Scalar or array."""
        return self.f0 + self.c * np.power(np.asarray(wear, float), self.k)

    def log_likelihood(self, observed_feature: float, wear):
        """log p(observed_feature | wear), vectorized over a wear particle cloud."""
        mu = self.expected(wear)
        z = (observed_feature - mu) / self.sigma
        return -0.5 * z * z - np.log(self.sigma) - 0.5 * _LOG_2PI

    @classmethod
    def fit(cls, wears, feature_values, feature_name: str) -> "PowerLawObservation":
        wears = np.asarray(wears, float)
        feats = np.asarray(feature_values, float)
        g = lambda w, f0, c, k: f0 + c * np.power(w, k)
        p0 = [max(feats.min(), 0.0), 100.0, 1.3]
        (f0, c, k), _ = curve_fit(g, wears, feats, p0=p0,
                                  bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
                                  maxfev=200000)
        sigma = float(np.std(feats - g(wears, f0, c, k)))
        return cls(feature_name, float(c), float(k), sigma, float(f0))
