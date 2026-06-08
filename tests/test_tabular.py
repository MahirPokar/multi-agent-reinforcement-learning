import numpy as np

from marl_lbf.tabular import TabularQAgent, q_learning_update


def test_q_learning_update_uses_bootstrap_when_not_done():
    updated = q_learning_update(1.0, 2.0, 4.0, alpha=0.5, gamma=0.9)

    assert updated == 3.3


def test_q_learning_update_drops_bootstrap_when_done():
    updated = q_learning_update(1.0, 2.0, 4.0, alpha=0.5, gamma=0.9, done=True)

    assert updated == 1.5


def test_tabular_agent_learns_state_action_value():
    agent = TabularQAgent(n_actions=2, alpha=1.0, gamma=0.0, seed=1)
    observation = np.zeros(12, dtype=np.float32)
    next_observation = np.ones(12, dtype=np.float32)

    agent.learn(
        observation,
        action=1,
        reward=3.0,
        next_observation=next_observation,
        done=False,
    )

    assert agent.q_values(observation)[1] == 3.0
