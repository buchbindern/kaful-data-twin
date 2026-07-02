"""
Replay driver (M4) — simulates the edge gateway.

Reads PHM cut files one at a time, encodes each waveform to compressed bytes
(what the edge would POST), and calls the handler in-process. Ingest is
idempotent: an already-ingested cut is skipped, so a re-run resumes rather than
erroring on the duplicate-cut constraint.
"""

from __future__ import annotations

from pathlib import Path

from domain.models import Machine, Run
from datasets.phm import load_cut_waveform
from ingest.codec import encode_waveform
from ingest.handler import IngestHandler


def replay_run(handler: IngestHandler, data_store, *, cut_files, machine_id: str,
               run_id: str, machine_type: str, tool_id=None, limit=None,
               progress: bool = True) -> None:
    # idempotent setup of the machine and run
    if data_store.get_machine(machine_id) is None:
        data_store.create_machine(Machine(machine_id, machine_type))
    if data_store.get_run(run_id) is None:
        data_store.create_run(Run(run_id, machine_id, tool_id=tool_id))

    files = cut_files if limit is None else cut_files[:limit]
    for cut_index, path in files:
        if data_store.get_cut(run_id, cut_index) is not None:
            if progress:
                print(f"cut {cut_index:>3}  ->  (already ingested, skipping)")
            continue
        raw = encode_waveform(load_cut_waveform(path))
        rul = handler.ingest_cut(machine_id, run_id, cut_index, raw)
        if progress:
            print(f"cut {cut_index:>3}  ->  RUL {rul.rul_median:5.1f} "
                  f"[{rul.rul_lower:.1f}, {rul.rul_upper:.1f}] cuts")
