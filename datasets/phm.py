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


def iter_cut_files(folder: str | Path) -> list[tuple[int, Path]]:
    """Find PHM per-cut CSVs in a folder, returned as (cut_index, path) sorted by
    cut_index. Parses the index from filenames like 'c_1_001.csv' -> 1.
    The caller supplies the exact folder, so the (Kaggle-specific) doubled
    'c1/c1' nesting is not baked into this adapter."""
    folder = Path(folder)
    out: list[tuple[int, Path]] = []
    for p in folder.glob("c_*_*.csv"):
        out.append((int(p.stem.split("_")[-1]), p))
    out.sort()
    return out


def load_wear_labels(path: str | Path) -> list[tuple[int, float]]:
    """Read a PHM wear file into (cut_index, wear_mm) pairs, sorted by cut_index.

    File columns: cut, flute_1, flute_2, flute_3, with wear in 1e-3 mm. The primary
    wear label is the mean of the three flutes, converted to mm. A header row
    (non-numeric first field) is auto-detected and skipped.
    """
    out: list[tuple[int, float]] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        try:
            cut = int(float(parts[0]))
        except ValueError:
            continue  # header row
        flutes = [float(x) for x in parts[1:4]]     # VB1, VB2, VB3 in 1e-3 mm
        wear_mm = (sum(flutes) / len(flutes)) / 1000.0
        out.append((cut, wear_mm))
    out.sort()
    return out
