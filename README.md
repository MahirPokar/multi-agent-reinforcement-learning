# Multi-Agent Reinforcement Learning (Q-learning → DQN) on Level-Based Foraging

> A practical, from-scratch exploration of **multi-agent Q-learning** and **Deep Q-Networks (DQN)** in a cooperative grid-world, highlighting **non-stationarity**, **scalability limits**, and why **independent value-based agents struggle in fully cooperative tasks without communication**.

<p align="center">
  <img src="assets/demo.gif" width="820" alt="Demo: two agents learning to collect items in Level-Based Foraging" />
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.7%2B-blue">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-used-red">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

---

## What this project is

This repo contains my **third-year individual project** on applying **Q-learning-based multi-agent reinforcement learning** to cooperative environments.

I implement and compare:

- **Tabular Q-learning**
  - Single-agent baseline (custom toy grid env)
  - Multi-agent **joint-action** tabular Q-learning (custom env)
- **Deep Q-learning**
  - Naïve independent DQN (fails to learn reliably)
  - **DQN + target network + experience replay** (stabilises learning)
  - Forced-cooperation setting + **reward shaping** (works, but brittle)

The core evaluation environment is **Level-Based Foraging (LBF)**: a cooperative task where agents must combine their “levels” to collect higher-level items.

---

## Key results (what matters)

### 1) Tabular Q-learning works… until the state/action space explodes
- Single-agent tabular learns efficiently in a small grid.
- Multi-agent joint-action tabular can work in a simplified setting, but storage grows fast (state/action blow-up).

<p align="center">
  <img src="assets/fig8_tabular_training.png" width="820" alt="Figure 8: Tabular Q-learning training curves (single-agent and multi-agent)" />
</p>

---

### 2) Naïve independent DQN is unstable (moving target + correlated samples)
The straightforward “replace the Q-table with a network” approach fails to converge in this MARL setting.

<p align="center">
  <img src="assets/fig9_naive_dqn.png" width="820" alt="Figure 9: Naïve DQN training results" />
</p>

---

### 3) DQN + target network + replay buffer learns in mixed-cooperative LBF
Adding the classic DQN stabilisation tricks yields strong learning in the mixed cooperative scenario.

<p align="center">
  <img src="assets/fig10_dqn_replay_target.png" width="820" alt="Figure 10: DQN with target network + replay buffer (mixed cooperative)" />
</p>

Also observed: **high sensitivity to learning rate** — an unsuitable LR can cause collapse / local minima.

<p align="center">
  <img src="assets/fig11_lr_sensitivity.png" width="820" alt="Figure 11: Training instability for lr=0.005" />
</p>

---

### 4) Fully cooperative (forced cooperation) breaks independent DQN — unless you “hand-hold” with reward shaping
When agents must cooperate *every time* (no communication, decentralised training/execution), learning fails with sparse rewards:

<p align="center">
  <img src="assets/fig12_forced_coop_no_shaping.png" width="820" alt="Figure 12: Forced cooperation without reward shaping" />
</p>

Reward shaping can make it work (but it’s heuristic-heavy and not robust):

<p align="center">
  <img src="assets/fig13_forced_coop_with_shaping.png" width="820" alt="Figure 13: Forced cooperation with reward shaping" />
</p>

---

## Environment: Level-Based Foraging (LBF)

LBF is a grid-world with **n agents** and **m items**, each with a **level**.
Agents can collect an item only if they stand adjacent to it and the **sum of cooperating agent levels ≥ item level**.

This environment is a useful abstraction for **distributed robotics** tasks (e.g., multiple robots coordinating to pick, carry, or complete missions).

<p align="center">
  <img src="assets/fig7_lbf_render.png" width="650" alt="Figure 7: LBF render (2 agents, 2 items)" />
</p>

---

## Methods (high-level)

### Tabular Q-learning (custom envs)
- Stores action-values in a Python dictionary keyed by state.
- Single-agent: 4 actions `{up, down, left, right}`.
- Multi-agent: joint-action policy with 16 action pairs (4×4).

### Deep Q-learning (PyTorch)
- Naïve DQN: independent networks per agent, bootstrapping on the same network → unstable.
- DQN improvements:
  - **Target network** to reduce moving-target instability
  - **Replay buffer** to de-correlate samples
  - Epsilon-greedy exploration with decay
- Forced cooperation:
  - Sparse reward makes learning collapse
  - Reward shaping used to encourage agents to stay close and coordinate

---

## Repo structure

- `IMPLEMENTATIONS/`
  - `Tabular.py` — tabular Q-learning (single + multi-agent joint-action)
  - `DQN_agent.py` — DQN with replay buffer + target network
  - `DQN_forced_coop.py` — forced cooperation experiments (+ reward shaping)
  - *(helper modules for network, replay buffer, plotting, CSV logging)*
- `assets/` — images/gifs used by this README
- `requirements.txt` — python dependencies (note: may contain extra packages)

---

## Quickstart (local)

> Notes:
> - The LBF environment is typically easiest to run in a **Python 3.7 (or older) virtual environment**.
> - Install the LBF environment first, then run the scripts in this repo.

```bash
# 1) Create and activate a venv (example)
python3.7 -m venv .venv
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run implementations (examples)
python IMPLEMENTATIONS/Tabular.py
python IMPLEMENTATIONS/DQN_agent.py
python IMPLEMENTATIONS/DQN_forced_coop.py
