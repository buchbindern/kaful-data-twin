"""
PHM 2010 dataset adapter (M3).

Isolates everything PHM-specific — channel order and how to read one cut file —
from the dataset-agnostic FeatureExtractor. A second dataset (e.g. Mondragon
MU-TCM later) gets its own adapter here but produces the same waveform shape, so
the feature/twin code downstream never changes.

Per-cut file: headerless CSV, 7 columns in this fixed order, ~127k rows @ 50 kHz.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Column order in every PHM 2010 per-cut CSV (no header in the files themselves).
PHM_CHANNELS = [
    "force_x", "force_y", "force_z",           # dynamometer, N
    "vibration_x", "vibration_y", "vibration_z",  # accelerometer, g
    "ae_rms",                                   # acoustic emission RMS, V
]


def load_cut_waveform(path: str | Path) -> np.ndarray:
    """Read one PHM cut CSV into a (n_samples, 7) float array."""
    arr = np.loadtxt(path, delimiter=",")
    if arr.ndim != 2 or arr.shape[1] != len(PHM_CHANNELS):
        raise ValueError(
            f"{path}: expected a headerless CSV with {len(PHM_CHANNELS)} columns, "
            f"got array of shape {arr.shape}"
        )
    return arr
