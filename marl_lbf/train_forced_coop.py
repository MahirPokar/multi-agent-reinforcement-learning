"""Command-line runner for the forced-cooperation DQN experiment."""

from __future__ import annotations

from marl_lbf.rewards import shape_forced_coop_rewards
from marl_lbf.train_dqn import run_training


def forced_coop_reward_transform(actions, rewards, observation):
    return shape_forced_coop_rewards(observation, actions, rewards)


def run(argv: list[str] | None = None) -> dict:
    return run_training(
        argv,
        experiment_name="forced-coop-dqn",
        forced_coop=True,
        reward_transform=forced_coop_reward_transform,
        parser_description=__doc__,
    )


def main() -> None:
    run()


if __name__ == "__main__":
    main()
