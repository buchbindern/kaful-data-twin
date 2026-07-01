"""M3: tests for the FeatureExtractor, feature-naming, and PHM adapter.

The real-data test at the bottom reads one actual c1 cut file and SKIPS cleanly
if the PHM data isn't present (so this suite passes in CI / a fresh clone, and
exercises real data on a machine that has it)."""

import math
from pathlib import Path

import numpy as np
import pytest

from features import FeatureExtractor, STAT_FUNCS, feature_name, split_feature_name
from datasets import PHM_CHANNELS, load_cut_waveform


# ---------------- individual statistics ----------------
def test_stats_on_known_signal():
    x = np.array([3.0, -3.0, 3.0, -3.0])  # |x|=3 everywhere, mean 0
    f = STAT_FUNCS
    assert f["rms"](x) == pytest.approx(3.0)
    assert f["mean_abs"](x) == pytest.approx(3.0)
    assert f["peak"](x) == pytest.approx(3.0)
    assert f["std"](x) == pytest.approx(3.0)
    assert f["crest_factor"](x) == pytest.approx(1.0)   # peak / rms = 3/3

def test_constant_channel_has_no_nan():
    x = np.zeros(1000)  # degenerate: rms=0, var=0
    for name, fn in STAT_FUNCS.items():
        v = fn(x)
        assert math.isfinite(v), f"{name} produced non-finite {v} on a constant signal"
    assert STAT_FUNCS["crest_factor"](x) == 0.0  # guarded, not 0/0
    assert STAT_FUNCS["kurtosis"](x) == 0.0      # guarded, not 0/0


# ---------------- feature naming (the underscore landmine) ----------------
def test_feature_name_builds():
    assert feature_name("force_z", "rms") == "force_z_rms"
    assert feature_name("ae_rms", "crest_factor") == "ae_rms_crest_factor"

def test_split_handles_multiword_stats_and_underscore_channels():
    # naive rsplit('_', 1) gets ALL THREE of these wrong; ours must not.
    assert split_feature_name("force_x_mean_abs") == ("force_x", "mean_abs")
    assert split_feature_name("ae_rms_crest_factor") == ("ae_rms", "crest_factor")
    assert split_feature_name("ae_rms_rms") == ("ae_rms", "rms")

def test_name_split_roundtrips_for_all_phm_features():
    ex = FeatureExtractor(PHM_CHANNELS)
    for name in ex.feature_names:
        ch, st = split_feature_name(name)
        assert feature_name(ch, st) == name
        assert ch in PHM_CHANNELS and st in STAT_FUNCS

def test_split_rejects_unknown_stat():
    with pytest.raises(ValueError):
        split_feature_name("force_x_bogus")


# ---------------- the extractor ----------------
def test_extract_produces_42_named_features():
    ex = FeatureExtractor(PHM_CHANNELS)
    wf = np.random.default_rng(0).standard_normal((5000, 7))
    feats = ex.extract(wf)
    assert len(feats) == 42                      # 6 stats x 7 channels
    assert set(feats) == set(ex.feature_names)   # exactly the expected names
    assert all(math.isfinite(v) for v in feats.values())

def test_extract_column_mapping_is_correct():
    # build a waveform where each channel is a distinct constant, so rms == that constant
    channels = PHM_CHANNELS
    consts = np.array([1., 2., 3., 4., 5., 6., 7.])
    wf = np.tile(consts, (100, 1))              # shape (100, 7), column i is constant i+1
    feats = FeatureExtractor(channels).extract(wf)
    for i, ch in enumerate(channels):
        assert feats[f"{ch}_rms"] == pytest.approx(consts[i])   # right column -> right channel

def test_extract_rejects_wrong_shape():
    ex = FeatureExtractor(PHM_CHANNELS)
    with pytest.raises(ValueError):
        ex.extract(np.zeros((100, 3)))   # 3 columns != 7 channels
    with pytest.raises(ValueError):
        ex.extract(np.zeros(100))        # 1-D, not 2-D


# ---------------- real PHM data (skips if absent) ----------------
PHM_C1_CUT1 = Path("data/phm2010/c1/c1/c_1_001.csv")

@pytest.mark.skipif(not PHM_C1_CUT1.exists(), reason="PHM 2010 c1 data not present")
def test_extract_on_real_phm_cut():
    wf = load_cut_waveform(PHM_C1_CUT1)
    assert wf.shape[1] == 7
    assert wf.shape[0] > 100_000            # ~127k rows expected
    feats = FeatureExtractor(PHM_CHANNELS).extract(wf)
    assert len(feats) == 42
    assert all(math.isfinite(v) for v in feats.values())
    assert feats["force_z_rms"] > 0        # a real cutting-force channel is nonzero
