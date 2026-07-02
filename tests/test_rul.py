"""M7: Monte Carlo RUL projection — correctness, spread, censoring, calibration."""

import numpy as np
import pytest

from twin import PowerLawWear, ParticleCloud, project_rul

DEG = PowerLawWear(9.883e-2, 2.727)
TH = 0.200


def _cloud(w, n=3000):
    return ParticleCloud(np.full(n, w), np.full(n, 1 / n))


def test_zero_noise_median_matches_closed_form():
    d = project_rul(_cloud(0.15), DEG, threshold=TH, process_noise=0.0,
                    rng=np.random.default_rng(0), horizon=400)
    assert d.median == pytest.approx(DEG.cuts_to_threshold(0.15, TH), abs=2)

def test_process_noise_widens_interval():
    rng = np.random.default_rng(1)
    narrow = project_rul(_cloud(0.15), DEG, threshold=TH, process_noise=0.0, rng=rng, horizon=400)
    wide = project_rul(_cloud(0.15), DEG, threshold=TH, process_noise=0.003, rng=rng, horizon=400)
    assert (wide.upper - wide.lower) > (narrow.upper - narrow.lower)

def test_already_past_threshold_is_zero():
    d = project_rul(_cloud(0.25), DEG, threshold=TH, process_noise=0.002,
                    rng=np.random.default_rng(2), horizon=100)
    assert d.median == 0.0

def test_low_wear_is_heavily_censored():
    d = project_rul(_cloud(0.05), DEG, threshold=TH, process_noise=0.002,
                    rng=np.random.default_rng(3), horizon=200)
    assert d.censored_frac > 0.5

def test_interval_is_calibrated_when_model_matches_reality():
    rng = np.random.default_rng(4)
    covered = 0; trials = 150
    for _ in range(trials):
        w0 = rng.uniform(0.12, 0.17)
        w, steps = w0, 0
        while w < TH and steps < 400:
            w = np.clip(DEG.advance(np.array([w]), 1.0)[0] + rng.normal(0, 0.002), 1e-4, 1.0)
            steps += 1
        d = project_rul(_cloud(w0, 1200), DEG, threshold=TH, process_noise=0.002,
                        rng=rng, horizon=400)
        covered += d.lower <= steps <= d.upper
    assert 0.80 <= covered / trials <= 1.0
