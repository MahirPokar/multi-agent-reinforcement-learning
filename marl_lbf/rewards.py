"""Reward transforms used by the reconstructed experiments."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def agent_distance_from_lbf_observation(
    observation,
    *,
    agent_xy: tuple[int, int] = (6, 7),
    partner_xy: tuple[int, int] = (9, 10),
) -> float:
    """Return the Manhattan distance between agents in an LBF observation."""

    obs = np.asarray(observation, dtype=np.float32)
    if obs.size <= max(*agent_xy, *partner_xy):
        raise ValueError("Observation is too short for the requested position indices.")
    return float(
        abs(obs[agent_xy[0]] - obs[partner_xy[0]])
        + abs(obs[agent_xy[1]] - obs[partner_xy[1]])
    )


def shape_forced_coop_rewards(
    observation,
    actions: Sequence[int],
    rewards: Sequence[float],
    *,
    reward_scale: float = 100.0,
    movement_penalty: float = 2.0,
    stay_penalty: float = 1.0,
    proximity_threshold: float = 4.0,
    proximity_bonus: float = 0.5,
) -> list[float]:
    """Reward shaping used in the forced-cooperation experiment."""

    distance = agent_distance_from_lbf_observation(observation)
    shaped: list[float] = []
    for action, reward in zip(actions, rewards):
        penalty = stay_penalty if int(action) == 0 else movement_penalty
        value = float(reward) * reward_scale - penalty
        if distance < proximity_threshold:
            value += proximity_bonus
        shaped.append(value)
    return shaped


def scale_mixed_coop_rewards(
    actions: Sequence[int],
    rewards: Sequence[float],
    *,
    reward_scale: float = 50.0,
    movement_penalty: float = 1.0,
) -> list[float]:
    """Reward transform used by the replay-buffer DQN experiment."""

    shaped = []
    for action, reward in zip(actions, rewards):
        value = float(reward) * reward_scale
        if int(action) != 0:
            value -= movement_penalty
        shaped.append(value)
    return shaped
