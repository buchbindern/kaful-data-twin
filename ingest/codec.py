"""
Waveform codec (M4) — array <-> compressed bytes.

The edge gateway sends a *compressed binary* payload, not a numpy array, so this
is the (de)serialization at that boundary. Stored to float32: PHM values are
small and float32 is ample precision for the statistics, while halving the size
of the ~5 MB blob. The '.npy' format embeds shape + dtype, so decode is exact.
"""

from __future__ import annotations

import gzip
import io

import numpy as np


def encode_waveform(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.asarray(array, dtype=np.float32))
    return gzip.compress(buf.getvalue())


def decode_waveform(data: bytes) -> np.ndarray:
    return np.load(io.BytesIO(gzip.decompress(data)))
