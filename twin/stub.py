"""
StubTwin (M4) — a placeholder that returns a CONSTANT RUL.

Its entire job is to prove the pipeline end-to-end without any modeling. It is
intentionally dumb: it ignores the features and returns the same numbers every
cut. If you ever see a flat RUL line in the output, that's the stub — the real
science arrives at M6 and this class gets deleted from the wiring.
"""

from __future__ import annotations

from domain.models import RULPrediction
from twin.base import Twin


class StubTwin(Twin):
    def __init__(self, rul_median: float = 50.0, ci_halfwidth: float = 25.0) -> None:
        self.rul_median = rul_median
        self.ci_halfwidth = ci_halfwidth

    def update(self, run_id: str, cut_index: int, features: dict[str, float]) -> RULPrediction:
        return RULPrediction(
            run_id=run_id,
            cut_index=cut_index,
            rul_median=self.rul_median,
            rul_lower=self.rul_median - self.ci_halfwidth,
            rul_upper=self.rul_median + self.ci_halfwidth,
        )
