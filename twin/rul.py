"""
Monte Carlo RUL projection (M7).

Turns the posterior wear cloud into a PREDICTIVE remaining-useful-life
distribution by simulating each particle's future forward, cut by cut, adding
process noise at every step, until it crosses the failure threshold. This is the
honest interval: it compounds present wear uncertainty WITH future accumulation
randomness, unlike M6's closed-form point-projection which used only today's spread.

Degradation params (a,p) are fixed here, so the interval captures process/wear
uncertainty but NOT rate uncertainty (the "this tool wears faster than c1" risk) —
that gap is where joint parameter estimation would plug in. Particles that don't
reach threshold within `horizon` are censored and reported as RUL = horizon, rather
than emitting a fake-precise huge number for a tool not yet in the wear-out regime.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from twin.cloud import ParticleCloud

_WEAR_MIN = 1e-4
_WEAR_MAX = 1.0


@dataclass
class RULDistribution:
    median: float
    lower: float          # 5th percentile
    upper: float          # 95th percentile
    censored_frac: float  # fraction of futures that didn't fail within the horizon


def project_rul(cloud: ParticleCloud, deg, *, threshold: float, process_noise: float,
                rng: np.random.Generator, horizon: int = 500,
                n_samples: int | None = None) -> RULDistribution:
    n = n_samples or cloud.n
    # resample to an equal-weight ensemble (fair samples for quantiles)
    idx = rng.choice(cloud.n, size=n, p=cloud.weights)
    w = cloud.wear[idx].astype(float).copy()

    rul = np.full(n, float(horizon))     # default: censored at horizon
    alive = w < threshold
    rul[~alive] = 0.0                     # already at/past threshold

    for step in range(1, horizon + 1):
        if not alive.any():
            break
        m = alive
        w[m] = np.clip(deg.advance(w[m], 1.0) + rng.normal(0.0, process_noise, int(m.sum())),
                       _WEAR_MIN, _WEAR_MAX)
        just = m & (w >= threshold)
        rul[just] = step
        alive &= ~just

    censored_frac = float(alive.mean())
    return RULDistribution(
        median=float(np.median(rul)),
        lower=float(np.quantile(rul, 0.05)),
        upper=float(np.quantile(rul, 0.95)),
        censored_frac=censored_frac,
    )
