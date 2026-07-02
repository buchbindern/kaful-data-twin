"""
ParticleCloud (M5d) — the twin's latent-state representation.

A weighted set of candidate wear values. The particle filter (M6) updates it each
cut; Monte Carlo (M7) projects it to the failure threshold for an RUL distribution.

State is wear only in this version — degradation (a,p) and observation (c,k,sigma)
are held fixed from the reference fit. The array layout leaves obvious room to add
(a,p) columns later for joint state-parameter estimation.

Serialized as gzip'd .npz (wear + weights) into the TwinState.particles BLOB.
"""

from __future__ import annotations

import gzip
import io
from dataclasses import dataclass

import numpy as np


@dataclass
class ParticleCloud:
    wear: np.ndarray      # (N,) candidate wear per particle
    weights: np.ndarray   # (N,) normalized weights (sum to 1)

    @property
    def n(self) -> int:
        return int(self.wear.size)

    def mean_wear(self) -> float:
        return float(np.sum(self.weights * self.wear))

    def quantile_wear(self, q: float) -> float:
        """Weighted quantile of wear (q in [0,1])."""
        order = np.argsort(self.wear)
        w = self.wear[order]
        cw = np.cumsum(self.weights[order])
        return float(np.interp(q, cw, w))

    def to_bytes(self) -> bytes:
        buf = io.BytesIO()
        np.savez(buf, wear=self.wear.astype(np.float64),
                 weights=self.weights.astype(np.float64))
        return gzip.compress(buf.getvalue())

    @classmethod
    def from_bytes(cls, data: bytes) -> "ParticleCloud":
        d = np.load(io.BytesIO(gzip.decompress(data)))
        return cls(d["wear"], d["weights"])
