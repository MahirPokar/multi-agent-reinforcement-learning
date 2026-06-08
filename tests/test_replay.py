import numpy as np
import pytest

from marl_lbf.replay import Experience, ReplayBuffer


def test_replay_buffer_samples_expected_shapes():
    buffer = ReplayBuffer(capacity=10, seed=1)
    for idx in range(5):
        state = np.full(12, idx, dtype=np.float32)
        buffer.append(Experience(state, idx % 6, float(idx), False, state + 1))

    states, actions, rewards, dones, next_states = buffer.sample(3)

    assert states.shape == (3, 12)
    assert actions.shape == (3,)
    assert rewards.shape == (3,)
    assert dones.shape == (3,)
    assert next_states.shape == (3, 12)


def test_replay_buffer_rejects_oversampling():
    buffer = ReplayBuffer(capacity=2)

    with pytest.raises(ValueError):
        buffer.sample(1)
