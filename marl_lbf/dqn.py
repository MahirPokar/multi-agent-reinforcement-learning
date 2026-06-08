"""DQN model and independent-agent learner."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from marl_lbf.schedules import EpsilonSchedule

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
except ModuleNotFoundError:
    torch = None
    nn = None
    F = None
    optim = None


def require_torch() -> None:
    if torch is None or nn is None or F is None or optim is None:
        raise RuntimeError(
            "DQN experiments require PyTorch. Install requirements.txt before "
            "running train_dqn or train_forced_coop."
        )


if nn is not None:

    class DeepQNetwork(nn.Module):
        def __init__(
            self,
            input_dim: int,
            n_actions: int,
            *,
            hidden_dim: int = 128,
            lr: float = 1e-4,
        ) -> None:
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, n_actions)
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
            self.loss_fn = nn.MSELoss()

        def forward(self, state):
            hidden = F.relu(self.fc1(state))
            return self.fc2(hidden)

else:

    class DeepQNetwork:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            require_torch()


@dataclass
class DQNAgent:
    input_dim: int
    n_actions: int
    lr: float = 1e-4
    gamma: float = 0.999
    epsilon: EpsilonSchedule = field(
        default_factory=lambda: EpsilonSchedule(
            start=1.0, minimum=0.1, decay=0.25e-5
        )
    )
    target_sync_interval: int = 1000
    seed: int | None = None

    def __post_init__(self) -> None:
        require_torch()
        self.rng = np.random.default_rng(self.seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.q_network = DeepQNetwork(self.input_dim, self.n_actions, lr=self.lr).to(
            self.device
        )
        self.target_network = DeepQNetwork(
            self.input_dim, self.n_actions, lr=self.lr
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.learn_steps = 0

    def choose_action(self, observation) -> int:
        if self.rng.random() < self.epsilon.value:
            return int(self.rng.integers(self.n_actions))

        state = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            actions = self.q_network(state)
        return int(torch.argmax(actions, dim=1).item())

    def learn(self, batch) -> float:
        states, actions, rewards, dones, next_states = batch
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(
            next_states, dtype=torch.float32, device=self.device
        )

        q_pred = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.target_network(next_states_t).max(dim=1).values
            q_target = rewards_t + self.gamma * q_next * (1.0 - dones_t)

        loss = self.q_network.loss_fn(q_pred, q_target)
        self.q_network.optimizer.zero_grad()
        loss.backward()
        self.q_network.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_sync_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_network.eval()
        self.epsilon.step()
        return float(loss.item())

    def save(self, path) -> None:
        torch.save(self.q_network.state_dict(), path)

    def load(self, path) -> None:
        state = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(state)
        self.target_network.load_state_dict(state)
        self.target_network.eval()
