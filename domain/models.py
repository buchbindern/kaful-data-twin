"""
Domain models for the Kaful data-first twin (M1a).

These are pure data records — no behavior, no I/O, no heavy dependencies.
Everything else in the system imports its vocabulary from here.

The entity hierarchy is load-bearing (handoff decision #3):

    Machine  (physical CNC, persistent)
      └─ Run  (one tool installation / tool life; cut_index resets here)
           └─ Cut  (one milling pass; carries the raw waveform + features)
"""

from __future__ import annotations # for forward references in dataclasses

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


def _utcnow() -> datetime:
    """Timezone aware UTC 'now'. Used as the default timestamp everywhere."""
    return datetime.now(timezone.utc)


@dataclass
class Machine:
    """A physical CNC machine. Persistent — it outlives the tools installed in it.

    `machine_type` points at a machine-type config (channels, feature list,
    degradation-model form, failure threshold) that is generated once at
    onboarding and is static per type (handoff decision #4).
    """
    machine_id: str
    machine_type: str
    name: Optional[str] = None
    created_at: datetime = field(default_factory=_utcnow)
    owner_id: Optional[str] = None   # None = system machine (readable by all, read-only)


@dataclass
class Run:
    """One tool installation = one tool life = one degradation trajectory from zero wear.

    A fresh Run is created at every tool change. `cut_index` resets within a run,
    so a tool swap does NOT look like wear collapsing to zero (which would break
    the filter — handoff decision #3).

    `ended_at is None` marks the currently active run for a machine.
    """
    run_id: str
    machine_id: str
    started_at: datetime = field(default_factory=_utcnow)
    ended_at: Optional[datetime] = None
    tool_id: Optional[str] = None


@dataclass
class Cut:
    """One milling pass — the transaction unit of the whole system (decision #1).

    A Cut carries a *pointer* to its raw waveform (`waveform_key` into the
    ObjectStore), never the ~5 MB array itself. The machine is reachable via
    run_id -> Run.machine_id, so we don't denormalize machine_id onto the Cut.
    """
    run_id: str
    cut_index: int
    waveform_key: str
    ingested_at: datetime = field(default_factory=_utcnow)


@dataclass
class FeatureRecord:
    """Scalar features extracted from one cut's waveform (6 stats x 7 channels = 42 for PHM).

    Stored as a name->value map on purpose: the feature set is still being
    discovered, and server-side extraction exists precisely so it can change
    without redeploying edge gateways (handoff decision #7). A rigid column set
    would fight that.
    """
    run_id: str
    cut_index: int
    features: dict[str, float]
    extracted_at: datetime = field(default_factory=_utcnow)


@dataclass
class RULPrediction:
    """Remaining useful life after one cut, as a distribution summary.

    Units: cuts remaining until the wear threshold (VB >= 0.2 mm for PHM, ISO 8688).
    We keep the median plus a credible interval rather than a point estimate,
    because honest early-life uncertainty is a design goal (decision #8).
    """
    run_id: str
    cut_index: int
    rul_median: float
    rul_lower: float
    rul_upper: float
    ci_level: float = 0.9
    predicted_at: datetime = field(default_factory=_utcnow)


@dataclass
class TwinState:
    """Persisted posterior of the per-run twin, overwritten once per cut.

    Structure = identity + a model-specific payload. The exact particle-cloud
    representation is defined in M5; here it is an opaque `particles: bytes` blob
    plus a `params` dict, so the storage layer (M2) can round-trip twin state
    WITHOUT depending on any M5 modeling internals. This is what lets the filter
    posterior survive between cuts (decision #5).
    """
    run_id: str
    cut_index: int
    params: dict[str, Any] = field(default_factory=dict)
    particles: Optional[bytes] = None
    updated_at: datetime = field(default_factory=_utcnow)


@dataclass
class WearLabel:
    """Ground-truth tool wear after one cut, in mm. REFERENCE / VALIDATION ONLY —
    the twin never consumes these to run (decision #2). For PHM this is the mean of
    the three flute measurements (VB1, VB2, VB3), converted from 1e-3 mm to mm.
    Stored per run so a labeled reference run (c1) can score the twin at M8.
    """
    run_id: str
    cut_index: int
    wear_mm: float


@dataclass(frozen=True)
class User:
    """An account. Owns machines (multi-tenancy)."""
    user_id: str
    email: str
    password_hash: str
    created_at: datetime


@dataclass(frozen=True)
class Session:
    """A server-side login session; its token lives in an httponly cookie."""
    token: str
    user_id: str
    created_at: datetime
    expires_at: datetime


@dataclass(frozen=True)
class CutResult:
    """Stored per-cut filter output, so the dashboard shows current state without recomputing."""
    run_id: str
    cut_index: int
    wear_mean: float
    wear_lo: float
    wear_hi: float
    wear_true: Optional[float]
    rul_median: float
    rul_lo: float
    rul_hi: float
    censored: float
    computed_at: datetime = field(default_factory=_utcnow)
