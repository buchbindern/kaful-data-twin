"""M5c: the PowerLawObservation model — fit, likelihood, and real-data shape."""

from pathlib import Path

import numpy as np
import pytest

from datasets import PHM_CHANNELS, load_cut_waveform, iter_cut_files, load_wear_labels
from features import FeatureExtractor
from twin import PowerLawObservation

RNG = np.random.default_rng(0)


def test_fit_recovers_parameters():
    wear = np.linspace(0.06, 0.16, 300)
    feat = 300.0 * wear**1.47 + RNG.normal(0, 0.4, wear.size)
    om = PowerLawObservation.fit(wear, feat, "force_z_rms")
    assert om.c == pytest.approx(300, rel=0.1)
    assert om.k == pytest.approx(1.47, rel=0.05)
    assert om.sigma == pytest.approx(0.4, rel=0.2)

def test_likelihood_peaks_at_generating_wear():
    om = PowerLawObservation("force_z_rms", c=300.0, k=1.47, sigma=0.3)
    observed = om.expected(0.13)
    cloud = np.linspace(0.06, 0.20, 200)
    assert cloud[np.argmax(om.log_likelihood(observed, cloud))] == pytest.approx(0.13, abs=0.005)

def test_log_likelihood_is_vectorized():
    om = PowerLawObservation("force_z_rms", c=300.0, k=1.47, sigma=0.3)
    cloud = np.array([0.08, 0.10, 0.12, 0.15])
    assert om.log_likelihood(9.0, cloud).shape == cloud.shape


C1_DIR = Path("data/phm2010/c1/c1")
C1_WEAR = Path("data/phm2010/c1/c1_wear.csv")

@pytest.mark.skipif(not (C1_DIR.exists() and C1_WEAR.exists()),
                    reason="PHM 2010 c1 data not present")
def test_real_c1_observation_is_convex_and_monotonic():
    # self-contained: subsample ~20 cuts, extract force_z_rms, fit against labels
    labels = {c: w for c, w in load_wear_labels(C1_WEAR)}
    ex = FeatureExtractor(PHM_CHANNELS)
    cuts_paths = iter_cut_files(C1_DIR)[::15]     # ~21 cuts across the run
    wears, vals = [], []
    for cut_index, path in cuts_paths:
        if cut_index not in labels:
            continue
        feats = ex.extract(load_cut_waveform(path))
        wears.append(labels[cut_index]); vals.append(feats["force_z_rms"])
    om = PowerLawObservation.fit(wears, vals, "force_z_rms")
    assert om.k > 1.0                                       # convex (accelerating)
    assert np.corrcoef(vals, wears)[0, 1] > 0.9            # strong wear indicator
