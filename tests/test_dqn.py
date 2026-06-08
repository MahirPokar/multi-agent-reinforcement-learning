import pytest

torch = pytest.importorskip("torch")

from marl_lbf.dqn import DeepQNetwork


def test_dqn_network_output_shape():
    network = DeepQNetwork(input_dim=12, n_actions=6)
    batch = torch.zeros((2, 12), dtype=torch.float32)

    output = network(batch)

    assert tuple(output.shape) == (2, 6)
