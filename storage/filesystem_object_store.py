"""
FilesystemObjectStore (M2a) — an ObjectStore backed by the local filesystem.

Stores each blob as a file at <root>/<key>. It is deliberately a *dumb byte
store*: it knows nothing about numpy or gzip. Callers hand it already-serialized
bytes, and the '.gz' in a key like 'phm2010/c1/000001.npy.gz' is just a naming
convention, not something this class acts on. That serialize/compress step lives
above it (added at M4).

Two robustness details worth noting:
  * Writes are atomic (write to a temp file, then os.replace). A crash mid-write
    can't leave a half-written 5 MB waveform that later reads as corrupt.
  * Keys are validated against path traversal ('..') and absolute paths, so a
    bad key can't write outside the store root.

Swap-target later: S3 / MinIO, which expose the same put/get/exists-by-key shape.
"""

from __future__ import annotations

import os
from pathlib import Path

from domain.stores import ObjectStore


class FilesystemObjectStore(ObjectStore):
    def __init__(self, root: str | os.PathLike) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        # Reject absolute keys and any parent directory traversal.
        if key.startswith("/") or ".." in Path(key).parts:
            raise ValueError(f"unsafe object key: {key!r}")
        return self.root / key

    def put(self, key: str, data: bytes) -> None:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)  # atomic rename on the same filesystem

    def get(self, key: str) -> bytes:
        path = self._resolve(key)
        try:
            return path.read_bytes()
        except FileNotFoundError:
            raise KeyError(key) from None

    def exists(self, key: str) -> bool:
        return self._resolve(key).is_file()
