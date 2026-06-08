import pytest

from marl_lbf.train_tabular import run as run_tabular


def test_tabular_smoke_run_writes_metrics(tmp_path):
    output_dir = tmp_path / "tabular"

    summary = run_tabular(["--episodes", "2", "--smoke", "--output-dir", str(output_dir)])

    assert summary["smoke"] is True
    assert (output_dir / "metrics.csv").exists()
    assert (output_dir / "q_tables.pkl").exists()


def test_dqn_smoke_run_writes_metrics(tmp_path):
    pytest.importorskip("torch")
    from marl_lbf.train_dqn import run as run_dqn

    output_dir = tmp_path / "dqn"
    summary = run_dqn(["--episodes", "1", "--smoke", "--output-dir", str(output_dir)])

    assert summary["smoke"] is True
    assert (output_dir / "metrics.csv").exists()


def test_forced_coop_smoke_run_writes_metrics(tmp_path):
    pytest.importorskip("torch")
    from marl_lbf.train_forced_coop import run as run_forced_coop

    output_dir = tmp_path / "forced"
    summary = run_forced_coop(["--episodes", "1", "--smoke", "--output-dir", str(output_dir)])

    assert summary["forced_coop"] is True
    assert (output_dir / "metrics.csv").exists()
