"""Tabular Q-learning utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable

import numpy as np

from marl_lbf.schedules import EpsilonSchedule


def state_key(observation) -> Hashable:
    """Convert an observation into a stable dictionary key."""

    values = np.asarray(observation, dtype=np.float32).round(4)
    return tuple(float(value) for value in values)


def q_learning_update(
    current_q: float,
    reward: float,
    next_max_q: float,
    *,
    alpha: float,
    gamma: float,
    done: bool = False,
) -> float:
    """Return the one-step Q-learning update."""

    bootstrap = 0.0 if done else gamma * next_max_q
    return (1.0 - alpha) * current_q + alpha * (reward + bootstrap)


@dataclass
class TabularQAgent:
    n_actions: int
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon: EpsilonSchedule = field(default_factory=EpsilonSchedule)
    seed: int | None = None

    def __post_init__(self) -> None:
        self.q_table: dict[Hashable, np.ndarray] = {}
        self.rng = np.random.default_rng(self.seed)

    def q_values(self, observation) -> np.ndarray:
        key = state_key(observation)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.n_actions, dtype=np.float32)
        return self.q_table[key]

    def choose_action(self, observation) -> int:
        if self.rng.random() < self.epsilon.value:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q_values(observation)))

    def learn(
        self,
        observation,
        action: int,
        reward: float,
        next_observation,
        done: bool,
    ) -> None:
        values = self.q_values(observation)
        next_max = float(np.max(self.q_values(next_observation)))
        values[action] = q_learning_update(
            float(values[action]),
            float(reward),
            next_max,
            alpha=self.alpha,
            gamma=self.gamma,
            done=done,
        )
        self.epsilon.step()
