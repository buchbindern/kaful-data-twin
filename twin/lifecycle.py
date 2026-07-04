"""
Run lifecycle (M10) — tool changes.

A tool change is a first-class operation: end the machine's active run (archive =
keep its history, mark it ended), create a fresh run (cut_index resets to 0 for the
new tool life), and deploy a fresh cold-start twin from the reference model spec.

This activates the Machine->Run->Cut boundary built at M1: without it, a tool swap
would look like wear collapsing from ~0.16mm back to ~0.04mm mid-stream and the
filter would break. With it, a tool change is a clean reset.
"""

from __future__ import annotations

from datetime import datetime, timezone

from domain.models import Run
from twin.build import deploy_from_reference


def start_new_run(data_store, machine_id: str, new_run_id: str, *, reference_run_id: str,
                  feature_name: str = "force_z_rms", n_particles: int = 2000,
                  threshold_mm: float = 0.200, onset_cut: float | None = None,
                  tool_id: str | None = None, seed: int = 0) -> Run:
    if data_store.get_run(new_run_id) is not None:
        raise ValueError(f"run {new_run_id!r} already exists")

    # 1. end the machine's active run (tool removed); history is retained
    active = data_store.get_active_run(machine_id)
    if active is not None:
        data_store.end_run(active.run_id, datetime.now(timezone.utc))

    # 2. create the fresh run (its cut_index starts at 0)
    data_store.create_run(Run(new_run_id, machine_id, tool_id=tool_id))

    # 3. deploy a fresh cold-start twin from the reference spec
    state = deploy_from_reference(data_store, reference_run_id, new_run_id,
                                  feature_name=feature_name, n_particles=n_particles,
                                  threshold_mm=threshold_mm, onset_cut=onset_cut, seed=seed)
    data_store.save_twin_state(state)
    return data_store.get_run(new_run_id)
