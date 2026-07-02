"""
Storage interfaces for the Kaful data-first twin (M1b).

Signatures only — no implementations. These abstract base classes are the seam
that keeps the rest of the system off any specific storage technology
(handoff decision #5). Phases call `store.append_cut(...)` /
`store.read_all_features(...)`, never `pd.read_parquet(...)` or
`sqlite3.connect(...)` directly.

Two stores, because two kinds of data with different access patterns:

  DataStore    -> small structured records (machines, runs, cuts, features,
                  RUL predictions, twin state). Needs O(1) append and queries.
                  SQLite now; Postgres/TimescaleDB for a fleet later.

  ObjectStore  -> big immutable raw-waveform blobs (~5 MB/cut), addressed by key.
                  Local filesystem now; S3/MinIO later.

Swapping either implementation must touch ZERO twin logic. That is the whole
point of moving off parquet.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from domain.models import (
    Machine,
    Run,
    Cut,
    FeatureRecord,
    RULPrediction,
    TwinState,
    WearLabel,
)


class DataStore(ABC):
    """Persistence for the small, structured records of the system.

    Implementations must be crash-safe per operation (a mid-cut crash must not
    corrupt already-committed cuts) and provide O(1) appends.

    NOTE: wear-label storage (validation-only) is added deliberately at M5, once
    the PHM wear-file format is nailed down. It is intentionally absent here — we
    don't guess an API before we've seen the data it stores.
    """

    # --- Machine ---
    @abstractmethod
    def create_machine(self, machine: Machine) -> None: ...

    @abstractmethod
    def get_machine(self, machine_id: str) -> Optional[Machine]: ...

    # --- Run (one tool life) ---
    @abstractmethod
    def create_run(self, run: Run) -> None: ...

    @abstractmethod
    def get_run(self, run_id: str) -> Optional[Run]: ...

    @abstractmethod
    def get_active_run(self, machine_id: str) -> Optional[Run]:
        """The run with ended_at is None for this machine, if any."""
        ...

    @abstractmethod
    def end_run(self, run_id: str, ended_at: datetime) -> None:
        """Mark a run finished (tool change). Used at M10."""
        ...

    # --- Cut (the transaction unit) ---
    @abstractmethod
    def append_cut(self, cut: Cut) -> None: ...

    @abstractmethod
    def get_cut(self, run_id: str, cut_index: int) -> Optional[Cut]: ...

    # --- Features (append per cut; read all for fitting) ---
    @abstractmethod
    def append_features(self, record: FeatureRecord) -> None: ...

    @abstractmethod
    def get_features(self, run_id: str, cut_index: int) -> Optional[FeatureRecord]: ...

    @abstractmethod
    def read_all_features(self, run_id: str) -> list[FeatureRecord]:
        """All features for a run, ordered by cut_index. Used to fit the twin (M5)."""
        ...

    # --- RUL predictions (append per cut; read all for dashboard/validation) ---
    @abstractmethod
    def append_rul(self, prediction: RULPrediction) -> None: ...

    @abstractmethod
    def read_all_rul(self, run_id: str) -> list[RULPrediction]:
        """All RUL predictions for a run, ordered by cut_index."""
        ...

    @abstractmethod
    def clear_rul(self, run_id: str) -> None:
        """Delete all RUL predictions for a run (e.g. before re-running the filter)."""
        ...

    # --- Twin state (overwrite once per cut; must survive between cuts) ---
    @abstractmethod
    def save_twin_state(self, state: TwinState) -> None:
        """Persist (overwrite) the latest twin posterior for a run."""
        ...

    @abstractmethod
    def load_twin_state(self, run_id: str) -> Optional[TwinState]:
        """Load the most recent twin posterior for a run, or None if not built yet."""
        ...

    # --- Wear labels (reference/validation only; independent of ingest) ---
    @abstractmethod
    def append_wear_label(self, label: WearLabel) -> None: ...

    @abstractmethod
    def read_wear_labels(self, run_id: str) -> list[WearLabel]:
        """All wear labels for a run, ordered by cut_index. Used at M8 (validation)
        and to fit the observation model on a labeled reference run (M5c)."""
        ...


class ObjectStore(ABC):
    """Persistence for large, immutable raw-waveform blobs, addressed by key.

    A dumb byte store on purpose: it knows nothing about numpy or gzip. Callers
    serialize a waveform to bytes and hand it a key (e.g. 'machine/run/cut').
    That ignorance is what makes the local-fs -> S3/MinIO swap trivial.
    """

    @abstractmethod
    def put(self, key: str, data: bytes) -> None: ...

    @abstractmethod
    def get(self, key: str) -> bytes: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...
