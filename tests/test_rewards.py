import numpy as np

from marl_lbf.rewards import agent_distance_from_lbf_observation
from marl_lbf.rewards import shape_forced_coop_rewards


def test_agent_distance_from_lbf_observation():
    observation = np.zeros(12, dtype=np.float32)
    observation[6] = 1
    observation[7] = 2
    observation[9] = 4
    observation[10] = 3

    assert agent_distance_from_lbf_observation(observation) == 4.0


def test_forced_coop_reward_shaping_adds_proximity_bonus():
    observation = np.zeros(12, dtype=np.float32)
    observation[6] = 1
    observation[7] = 1
    observation[9] = 2
    observation[10] = 2

    shaped = shape_forced_coop_rewards(observation, actions=[0, 5], rewards=[0.1, 0.1])

    assert shaped == [9.5, 8.5]
