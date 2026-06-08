"""Command-line runner for independent tabular Q-learning."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from marl_lbf.envs import DEFAULT_LBF_ENV_ID, make_env, reset_env, step_env
from marl_lbf.logging import CsvMetricLogger, make_run_dir, write_json
from marl_lbf.schedules import EpsilonSchedule
from marl_lbf.tabular import TabularQAgent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=500, help="Training episodes.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--render", action="store_true", help="Render the environment.")
    parser.add_argument("--output-dir", default=None, help="Directory for metrics/checkpoints.")
    parser.add_argument(
        "--load-checkpoint",
        default=None,
        help="Optional pickle checkpoint containing two Q-tables.",
    )
    parser.add_argument("--smoke", action="store_true", help="Use a tiny env for a fast check.")
    parser.add_argument("--env-id", default=DEFAULT_LBF_ENV_ID, help="Gym/LBF environment id.")
    parser.add_argument("--max-steps", type=int, default=200, help="Safety cap per episode.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Q-learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    return parser


def _load_q_tables(path: str, agents: list[TabularQAgent]) -> None:
    with Path(path).open("rb") as handle:
        tables = pickle.load(handle)
    for agent, table in zip(agents, tables):
        agent.q_table = table


def run(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    if args.smoke:
        args.max_steps = min(args.max_steps, 25)

    env = make_env(args.env_id, seed=args.seed, smoke=args.smoke)
    run_dir = make_run_dir(args.output_dir, "tabular")
    agents = [
        TabularQAgent(
            n_actions=6,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=EpsilonSchedule(
                start=1.0,
                minimum=0.1,
                decay=1e-3 if args.smoke else 1e-5,
            ),
            seed=args.seed + idx,
        )
        for idx in range(2)
    ]

    if args.load_checkpoint:
        _load_q_tables(args.load_checkpoint, agents)

    metrics_path = run_dir / "metrics.csv"
    with CsvMetricLogger(
        metrics_path,
        ["episode", "reward_agent_1", "reward_agent_2", "epsilon"],
    ) as metrics:
        for episode in range(args.episodes):
            observations = reset_env(env, seed=args.seed + episode if args.smoke else None)
            done = [False, False]
            episode_rewards = np.zeros(2, dtype=np.float32)
            steps = 0

            while not all(done) and steps < args.max_steps:
                actions = [
                    agent.choose_action(observations[idx])
                    for idx, agent in enumerate(agents)
                ]
                next_observations, rewards, done, _info = step_env(env, actions)

                for idx, agent in enumerate(agents):
                    agent.learn(
                        observations[idx],
                        actions[idx],
                        float(rewards[idx]),
                        next_observations[idx],
                        bool(done[idx]),
                    )
                    episode_rewards[idx] += float(rewards[idx])

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
                }
            )

    checkpoint_path = run_dir / "q_tables.pkl"
    with checkpoint_path.open("wb") as handle:
        pickle.dump([agent.q_table for agent in agents], handle)

    summary = {
        "episodes": args.episodes,
        "metrics": str(metrics_path),
        "checkpoint": str(checkpoint_path),
        "smoke": args.smoke,
    }
    write_json(run_dir / "summary.json", summary)
    print(f"wrote metrics to {metrics_path}")
    return summary


def main() -> None:
    run()


if __name__ == "__main__":
    main()
