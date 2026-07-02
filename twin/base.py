"""
Twin interface (M4).

The digital twin, viewed from the handler, is anything that can turn this cut's
features into an RUL prediction. The StubTwin (M4) and the real particle-filter
twin (M6) both satisfy this — swapping one for the other changes no handler code.

A twin that maintains state persists its own posterior between cuts (via the
DataStore's twin_state table); the handler owns cut/features/RUL persistence but
NOT the twin's private state. That boundary is why the stub needs no store at all.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from domain.models import RULPrediction


class Twin(ABC):
    @abstractmethod
    def update(self, run_id: str, cut_index: int, features: dict[str, float]) -> RULPrediction:
        """Given this cut's features, return an RUL prediction for it."""
        ...
