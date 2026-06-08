"""Command-line runner for independent DQN with replay and target networks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np

from marl_lbf.dqn import DQNAgent
from marl_lbf.envs import DEFAULT_FORCED_COOP_ENV_ID
from marl_lbf.envs import DEFAULT_LBF_ENV_ID
from marl_lbf.envs import make_env, reset_env, step_env
from marl_lbf.logging import CsvMetricLogger, make_run_dir, write_json
from marl_lbf.replay import Experience, ReplayBuffer
from marl_lbf.rewards import scale_mixed_coop_rewards

RewardTransform = Callable[[list[int], list[float], object], list[float]]


def build_parser(description: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description or __doc__)
    parser.add_argument("--episodes", type=int, default=500, help="Training episodes.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--render", action="store_true", help="Render the environment.")
    parser.add_argument("--output-dir", default=None, help="Directory for metrics/checkpoints.")
    parser.add_argument(
        "--load-checkpoint",
        default=None,
        help="Directory containing agent1.pt and agent2.pt.",
    )
    parser.add_argument("--smoke", action="store_true", help="Use a tiny env for a fast check.")
    parser.add_argument("--env-id", default=DEFAULT_LBF_ENV_ID, help="Gym/LBF environment id.")
    parser.add_argument("--max-steps", type=int, default=200, help="Safety cap per episode.")
    parser.add_argument("--batch-size", type=int, default=32, help="Replay minibatch size.")
    parser.add_argument("--buffer-capacity", type=int, default=10000, help="Replay buffer capacity.")
    parser.add_argument(
        "--min-buffer-size",
        type=int,
        default=5000,
        help="Experiences before learning starts.",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate.")
    return parser


def default_reward_transform(
    actions: list[int],
    rewards: list[float],
    observation,
) -> list[float]:
    return scale_mixed_coop_rewards(actions, rewards)


def run_training(
    argv: list[str] | None = None,
    *,
    experiment_name: str = "dqn",
    forced_coop: bool = False,
    reward_transform: RewardTransform = default_reward_transform,
    parser_description: str | None = None,
) -> dict:
    args = build_parser(parser_description).parse_args(argv)
    if forced_coop and args.env_id == DEFAULT_LBF_ENV_ID:
        args.env_id = DEFAULT_FORCED_COOP_ENV_ID
    if args.smoke:
        args.max_steps = min(args.max_steps, 25)
        args.batch_size = min(args.batch_size, 4)
        args.min_buffer_size = min(args.min_buffer_size, args.batch_size)

    env = make_env(args.env_id, seed=args.seed, smoke=args.smoke, forced_coop=forced_coop)
    run_dir = make_run_dir(args.output_dir, experiment_name)
    agents = [
        DQNAgent(input_dim=12, n_actions=6, lr=args.lr, seed=args.seed + idx)
        for idx in range(2)
    ]
    buffers = [ReplayBuffer(args.buffer_capacity, seed=args.seed + idx) for idx in range(2)]

    if args.load_checkpoint:
        checkpoint_dir = Path(args.load_checkpoint)
        agents[0].load(checkpoint_dir / "agent1.pt")
        agents[1].load(checkpoint_dir / "agent2.pt")

    metrics_path = run_dir / "metrics.csv"
    with CsvMetricLogger(
        metrics_path,
        [
            "episode",
            "reward_agent_1",
            "reward_agent_2",
            "epsilon",
            "loss_agent_1",
            "loss_agent_2",
        ],
    ) as metrics:
        for episode in range(args.episodes):
            observations = reset_env(env, seed=args.seed + episode if args.smoke else None)
            done = [False, False]
            episode_rewards = np.zeros(2, dtype=np.float32)
            last_losses = [None, None]
            steps = 0

            while not all(done) and steps < args.max_steps:
                actions = [
                    agent.choose_action(observations[idx])
                    for idx, agent in enumerate(agents)
                ]
                next_observations, raw_rewards, done, _info = step_env(env, actions)
                rewards = reward_transform(actions, list(raw_rewards), observations[0])

                for idx, buffer in enumerate(buffers):
                    buffer.append(
                        Experience(
                            state=np.asarray(observations[idx], dtype=np.float32),
                            action=actions[idx],
                            reward=float(rewards[idx]),
                            done=bool(done[idx]),
                            next_state=np.asarray(next_observations[idx], dtype=np.float32),
                        )
                    )
                    episode_rewards[idx] += float(raw_rewards[idx])

                    if len(buffer) >= args.min_buffer_size and len(buffer) >= args.batch_size:
                        last_losses[idx] = agents[idx].learn(buffer.sample(args.batch_size))

                observations = next_observations
                steps += 1
                if args.render:
                    env.render()

            metrics.write(
                {
                    "episode": episode,
                    "reward_agent_1": float(episode_rewards[0]),
                    "reward_agent_2": float(episode_rewards[1]),
                    "epsilon": agents[0].epsilon.value,
                    "loss_agent_1": "" if last_losses[0] is None else last_losses[0],
                    "loss_agent_2": "" if last_losses[1] is None else last_losses[1],
                }
            )

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    agents[0].save(checkpoint_dir / "agent1.pt")
    agents[1].save(checkpoint_dir / "agent2.pt")

    summary = {
        "episodes": args.episodes,
        "metrics": str(metrics_path),
        "checkpoint_dir": str(checkpoint_dir),
        "smoke": args.smoke,
        "forced_coop": forced_coop,
    }
    write_json(run_dir / "summary.json", summary)
    print(f"wrote metrics to {metrics_path}")
    return summary


def run(argv: list[str] | None = None) -> dict:
    return run_training(argv, experiment_name="dqn")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
