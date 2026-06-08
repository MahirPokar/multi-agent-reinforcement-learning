"""Experience replay buffer for DQN experiments."""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Deque

import numpy as np


@dataclass(frozen=True)
class Experience:
    state: np.ndarray
    action: int
    reward: float
    done: bool
    next_state: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int | None = None) -> None:
        self.buffer: Deque[Experience] = collections.deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        self.buffer.append(experience)

    def sample(self, batch_size: int):
        if batch_size > len(self.buffer):
            raise ValueError("Cannot sample more experiences than the buffer contains.")

        indices = self.rng.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[int(index)] for index in indices]
        states = np.asarray([exp.state for exp in batch], dtype=np.float32)
        actions = np.asarray([exp.action for exp in batch], dtype=np.int64)
        rewards = np.asarray([exp.reward for exp in batch], dtype=np.float32)
        dones = np.asarray([exp.done for exp in batch], dtype=np.float32)
        next_states = np.asarray([exp.next_state for exp in batch], dtype=np.float32)
        return states, actions, rewards, dones, next_states
