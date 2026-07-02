"""
IngestHandler (M4) — the orchestrator for one cut. THE load bearing artifact.

ingest_cut() is a plain method so that at M9 a FastAPI endpoint can wrap this
exact call: "same handler, different transport." The order matters — raw blob is
stored first (durable record of what arrived), then everything derived from it.
"""

from __future__ import annotations

from domain.models import Cut, FeatureRecord, RULPrediction
from domain.stores import DataStore, ObjectStore
from features.extractor import FeatureExtractor
from ingest.codec import decode_waveform
from twin.base import Twin


def waveform_key(machine_id: str, run_id: str, cut_index: int) -> str:
    """Object-store key: machine/run/cut (handoff decision #6)."""
    return f"{machine_id}/{run_id}/{cut_index:06d}.npy.gz"


class IngestHandler:
    def __init__(self, data_store: DataStore, object_store: ObjectStore,
                 extractor: FeatureExtractor, twin: Twin) -> None:
        self.data_store = data_store
        self.object_store = object_store
        self.extractor = extractor
        self.twin = twin

    def ingest_cut(self, machine_id: str, run_id: str, cut_index: int,
                   raw_bytes: bytes) -> RULPrediction:
        # 1. persist the raw waveform blob (immutable record of what arrived)
        key = waveform_key(machine_id, run_id, cut_index)
        self.object_store.put(key, raw_bytes)
        self.data_store.append_cut(Cut(run_id, cut_index, key))

        # 2. decode -> extract -> persist features
        waveform = decode_waveform(raw_bytes)
        features = self.extractor.extract(waveform)
        self.data_store.append_features(FeatureRecord(run_id, cut_index, features))

        # 3. twin produces the RUL for this cut; handler persists it
        rul = self.twin.update(run_id, cut_index, features)
        self.data_store.append_rul(rul)
        return rul
