"""M5b: the PowerLawWear degradation model — forward math, fitting, extrapolation."""

import numpy as np
import pytest

from datasets import load_wear_labels
from pathlib import Path
from twin import PowerLawWear


def test_advance_then_threshold_is_self_consistent():
    m = PowerLawWear(a=7.5e-3, p=1.45)
    full = m.cuts_to_threshold(0.095, 0.200)
    after = m.cuts_to_threshold(m.advance(0.095, 30), 0.200)
    assert full - after == pytest.approx(30.0, abs=1e-6)  # advancing 30 cuts costs 30 cuts of RUL

def test_threshold_already_reached_is_zero():
    m = PowerLawWear(a=7.5e-3, p=1.45)
    assert m.cuts_to_threshold(0.25, 0.200) == 0.0

def test_fit_recovers_known_parameters():
    true = PowerLawWear(a=6e-3, p=1.5)
    cuts = np.arange(100, 316, 3.0)
    wears = np.array([true.advance(0.095, c - 100) for c in cuts])
    fit, info = PowerLawWear.fit(cuts, wears, onset_cut=100)
    assert fit.a == pytest.approx(6e-3, rel=1e-3)
    assert fit.p == pytest.approx(1.5, rel=1e-3)
    assert info["rmse_um"] < 0.01

def test_advance_and_threshold_are_vectorized():
    m = PowerLawWear(a=7.5e-3, p=1.45)
    cloud = np.array([0.09, 0.10, 0.12, 0.15])
    adv = m.advance(cloud, 20)
    assert adv.shape == cloud.shape and np.all(adv > cloud)   # all worn more
    rul = m.cuts_to_threshold(cloud, 0.2)
    assert rul.shape == cloud.shape and np.all(np.diff(rul) < 0)  # more wear -> less RUL


C1_WEAR = Path("data/phm2010/c1/c1_wear.csv")

@pytest.mark.skipif(not C1_WEAR.exists(), reason="PHM 2010 c1 wear file not present")
def test_fit_real_c1_extrapolates_past_observed_life():
    labels = load_wear_labels(C1_WEAR)
    cuts = [c for c, _ in labels]; wears = [w for _, w in labels]
    model, info = PowerLawWear.fit(cuts, wears)
    assert info["rmse_um"] < 3.0            # sub-3-micron fit on the wear-out region
    assert model.p > 1.0                    # accelerating wear-out
    crossing = cuts[-1] + model.cuts_to_threshold(wears[-1], 0.200)
    assert 330 < crossing < 450             # sane extrapolation past cut 315
