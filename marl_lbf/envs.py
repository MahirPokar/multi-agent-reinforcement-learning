"""Environment helpers used by the training entrypoints.

The full experiments use the external Level-Based Foraging package. The
TinyForagingEnv fallback keeps smoke tests fast and self-contained.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


DEFAULT_LBF_ENV_ID = "Foraging-8x8-2p-2f-v2"
DEFAULT_FORCED_COOP_ENV_ID = "Foraging-8x8-2p-2f-coop-v2"


@dataclass(frozen=True)
class EnvSpec:
    n_agents: int = 2
    obs_dim: int = 12
    n_actions: int = 6


class TinyForagingEnv:
    """A small deterministic grid-world used for smoke tests.

    Actions follow the LBF convention closely enough for code-path checks:
    0 noop, 1 north, 2 south, 3 west, 4 east, 5 load.
    """

    spec = EnvSpec()

    def __init__(
        self,
        grid_size: int = 5,
        max_steps: int = 25,
        seed: int | None = None,
        forced_coop: bool = False,
    ) -> None:
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.forced_coop = forced_coop
        self.rng = np.random.default_rng(seed)
        self.agent_positions = np.zeros((2, 2), dtype=np.int64)
        self.food_position = np.zeros(2, dtype=np.int64)
        self.step_count = 0

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.agent_positions = np.array(
            [[0, 0], [self.grid_size - 1, 0]], dtype=np.int64
        )
        self.food_position = np.array(
            [self.grid_size // 2, self.grid_size // 2], dtype=np.int64
        )
        self.step_count = 0
        return self._observations()

    def step(self, actions: Sequence[int]):
        actions = [int(action) for action in actions]
        self.step_count += 1

        for agent_idx, action in enumerate(actions):
            self.agent_positions[agent_idx] = self._move(
                self.agent_positions[agent_idx], action
            )

        rewards = np.full(2, -0.01, dtype=np.float32)
        adjacent = [
            self._manhattan(self.agent_positions[i], self.food_position) <= 1
            for i in range(2)
        ]
        loading = [action == 5 for action in actions]

        if self.forced_coop:
            success = all(adjacent) and all(loading)
        else:
            success = any(
                is_adjacent and is_loading
                for is_adjacent, is_loading in zip(adjacent, loading)
            )

        if success:
            rewards[:] = 1.0
            done = [True, True]
        else:
            for i, is_loading in enumerate(loading):
                if is_loading:
                    rewards[i] -= 0.05
            done = [self.step_count >= self.max_steps] * 2

        info = {
            "food_collected": bool(success),
            "smoke_env": True,
            "forced_coop": self.forced_coop,
        }
        return self._observations(), rewards.tolist(), done, info

    def render(self) -> None:
        grid = np.full((self.grid_size, self.grid_size), ".", dtype=object)
        grid[tuple(self.food_position)] = "F"
        for idx, pos in enumerate(self.agent_positions):
            grid[tuple(pos)] = str(idx + 1)
        print("\n".join(" ".join(row) for row in grid))

    def _move(self, position: np.ndarray, action: int) -> np.ndarray:
        delta = {
            0: (0, 0),
            1: (-1, 0),
            2: (1, 0),
            3: (0, -1),
            4: (0, 1),
            5: (0, 0),
        }.get(action, (0, 0))
        next_position = position + np.asarray(delta, dtype=np.int64)
        return np.clip(next_position, 0, self.grid_size - 1)

    def _observations(self) -> list[np.ndarray]:
        return [self._observation(agent_idx) for agent_idx in range(2)]

    def _observation(self, agent_idx: int) -> np.ndarray:
        partner_idx = 1 - agent_idx
        agent = self.agent_positions[agent_idx]
        partner = self.agent_positions[partner_idx]
        food = self.food_position
        scale = max(self.grid_size - 1, 1)
        agent_food_dist = self._manhattan(agent, food)
        partner_food_dist = self._manhattan(partner, food)
        agent_partner_dist = self._manhattan(agent, partner)
        adjacent_count = int(agent_food_dist <= 1) + int(partner_food_dist <= 1)

        return np.array(
            [
                agent[0] / scale,
                agent[1] / scale,
                partner[0] / scale,
                partner[1] / scale,
                food[0] / scale,
                food[1] / scale,
                agent_food_dist / (2 * scale),
                partner_food_dist / (2 * scale),
                agent_partner_dist / (2 * scale),
                self.step_count / max(self.max_steps, 1),
                float(adjacent_count),
                float(self.forced_coop),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _manhattan(left: Iterable[int], right: Iterable[int]) -> int:
        left_arr = np.asarray(left)
        right_arr = np.asarray(right)
        return int(np.abs(left_arr - right_arr).sum())


def make_env(
    env_id: str | None = None,
    *,
    seed: int | None = None,
    smoke: bool = False,
    forced_coop: bool = False,
):
    """Create either the real LBF environment or the built-in smoke env."""

    if smoke:
        return TinyForagingEnv(seed=seed, forced_coop=forced_coop)

    selected_env = env_id or (
        DEFAULT_FORCED_COOP_ENV_ID if forced_coop else DEFAULT_LBF_ENV_ID
    )
    try:
        import gym  # type: ignore
        import lbforaging  # noqa: F401  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Full Level-Based Foraging runs require gym and lbforaging. "
            "Install requirements.txt, or pass --smoke for a fast local check."
        ) from exc

    env = gym.make(selected_env)
    if seed is not None:
        if hasattr(env, "seed"):
            env.seed(seed)
        else:
            try:
                env.reset(seed=seed)
            except TypeError:
                pass
    return env


def reset_env(env, *, seed: int | None = None):
    """Reset old- or new-style Gym environments."""

    if seed is None:
        result = env.reset()
    else:
        try:
            result = env.reset(seed=seed)
        except TypeError:
            result = env.reset()

    if isinstance(result, tuple) and len(result) == 2:
        observations, _info = result
        return observations
    return result


def step_env(env, actions: Sequence[int]):
    """Step old- or new-style Gym environments."""

    result = env.step(tuple(actions))
    if isinstance(result, tuple) and len(result) == 5:
        observations, rewards, terminated, truncated, info = result
        done = np.logical_or(terminated, truncated)
        return observations, rewards, done, info
    return result
