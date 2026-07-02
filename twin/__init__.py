"""The digital twin: interface, stub, and modeling components."""

from twin.base import Twin
from twin.stub import StubTwin
from twin.degradation import PowerLawWear
from twin.observation import PowerLawObservation

__all__ = ["Twin", "StubTwin", "PowerLawWear", "PowerLawObservation"]
