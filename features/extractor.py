"""
FeatureExtractor (M3) — turns one cut's raw waveform into scalar features.

Dataset-agnostic: it takes a 2-D waveform array (n_samples, n_channels) plus the
channel names, and returns a flat {feature_name: value} dict. For PHM that's
6 stats x 7 channels = 42 features. Ported from the prior work's `normalize.py`.

The one subtlety worth understanding is feature NAMING (see split_feature_name):
both channel names ('ae_rms') and stat names ('mean_abs', 'crest_factor') can
contain underscores, so you cannot parse a feature name back into (channel, stat)
with a naive rsplit('_', 1) — 'force_x_mean_abs' would wrongly split into
('force_x_mean', 'abs'). We match against the known stat suffixes, longest first.
"""

from __future__ import annotations

import numpy as np


# ---- the 6 per-channel statistics (each takes a 1-D array, returns a float) ----
# Guards return 0.0 on degenerate (constant) channels so we never emit NaN/inf.

def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x)))

def _mean_abs(x: np.ndarray) -> float:
    return float(np.mean(np.abs(x)))

def _peak(x: np.ndarray) -> float:
    return float(np.max(np.abs(x)))  # peak absolute amplitude

def _kurtosis(x: np.ndarray) -> float:
    m = x.mean()
    var = np.mean((x - m) ** 2)
    if var == 0.0:
        return 0.0
    return float(np.mean((x - m) ** 4) / (var ** 2))  # Pearson kurtosis (=3 for Gaussian)

def _crest_factor(x: np.ndarray) -> float:
    rms = _rms(x)
    if rms == 0.0:
        return 0.0
    return float(np.max(np.abs(x)) / rms)

def _std(x: np.ndarray) -> float:
    return float(np.std(x))  # population std (ddof=0)


# Insertion order fixes the feature-name order; keep it stable for reproducibility.
STAT_FUNCS = {
    "rms": _rms,
    "mean_abs": _mean_abs,
    "peak": _peak,
    "kurtosis": _kurtosis,
    "crest_factor": _crest_factor,
    "std": _std,
}


def feature_name(channel: str, stat: str) -> str:
    return f"{channel}_{stat}"


def split_feature_name(name: str, stats=STAT_FUNCS) -> tuple[str, str]:
    """Inverse of feature_name. Matches known stat suffixes longest-first so that
    multi-word stats ('mean_abs', 'crest_factor') and underscore-bearing channels
    ('ae_rms') parse correctly. Raises ValueError if no known stat suffix matches.
    """
    for stat in sorted(stats, key=len, reverse=True):
        suffix = "_" + stat
        if name.endswith(suffix):
            return name[: -len(suffix)], stat
    raise ValueError(f"cannot split feature name {name!r} with known stats {sorted(stats)}")


class FeatureExtractor:
    def __init__(self, channels: list[str], stats=STAT_FUNCS) -> None:
        self.channels = list(channels)
        self.stats = dict(stats)

    @property
    def feature_names(self) -> list[str]:
        return [feature_name(ch, st) for ch in self.channels for st in self.stats]

    def extract(self, waveform: np.ndarray) -> dict[str, float]:
        """waveform: shape (n_samples, n_channels), column i == self.channels[i]."""
        arr = np.asarray(waveform, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != len(self.channels):
            raise ValueError(
                f"expected waveform of shape (n_samples, {len(self.channels)}), "
                f"got {arr.shape}"
            )
        out: dict[str, float] = {}
        for i, ch in enumerate(self.channels):
            col = arr[:, i]
            for st, fn in self.stats.items():
                out[feature_name(ch, st)] = fn(col)
        return out
