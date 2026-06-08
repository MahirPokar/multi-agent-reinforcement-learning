"""Exploration schedules shared across agents."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EpsilonSchedule:
    start: float = 1.0
    minimum: float = 0.1
    decay: float = 1e-5

    def __post_init__(self) -> None:
        self.value = float(self.start)

    def step(self) -> float:
        self.value = max(self.minimum, self.value - self.decay)
        return self.value
