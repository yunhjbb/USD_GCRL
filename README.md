#  LAVL: Latent-Aliged Value Learning

The official implementation for \<Latent Representation Alignment for Offline Goal-Conditioned Reinforcement Learning\>

## Overview

<p align="center">
  <img src="assets/lavl_benchmark.png" width="800">
</p>

In offline GCRL, existing value-based methods often suffer from **erroneous value generalization**: the learned value function assigns similar values to states that are close in Euclidean distance, even when they are far apart in temporal distance.

We propose **Latent-Aligned Value Learning (LAVL)**, an offline GCRL algorithm built on a new value function architecture, **Latent Alignment Network (LAN)**. LAN parameterizes the goal-conditioned value as the negative distance between learned state and goal representations:

$V(s, g) = - \|\phi_S(s) - \phi_G(g)\|_2.$

This simple architecture significantly improves value generalization in offline GCRL!

## Requirements
Our code is based on the OGBench [1], a comprehensive benchmark for GCRL including baseline implementations.
The requirements and installation guide are presented in the OGBench [repository](https://github.com/seohongpark/ogbench/tree/master)

## Training
The implementation of LAVL algorithm is contained in the `impls/agents/lavl.py` file.
The hyperparameters and exact commands for reproduction are in the `impls/hyperparameters.sh` file.
For instance, the following command runs LAVL in antmaze-giant-navigate
```
python main.py --env_name=antmaze-giant-navigate-v0 --eval_episodes=50 --agent=agents/lavl.py --agent.discount=0.999 --agent.expectile=0.9 --agent.smoothness_weight=10.0 --agent.low_actor_rep_grad=True --agent.high_actor_rep_grad=True --agent.high_alpha=1.0 --agent.low_alpha=3.0 --agent.subgoal_steps=25
```

LAVL also supports a categorical latent geometry. Add `--agent.value_geometry=categorical` to the command to use
`-sqrt(2 * KL(q_goal || q_state))`, where each latent distribution is the softmax of its state or goal representation.
The default `--agent.value_geometry=euclidean` keeps the original LAN distance.

## Offline METRA-style goal-directed skills

`impls/agents/offline_metra.py` implements iterative representation fitting with future-goal directions and an
independent IQL skill learner. Both updates sample only from the fixed OGBench dataset. Training never steps the
environment; the environment is used for periodic evaluation. It preserves METRA's linear directional-progress
objective and clipped constraint. The three experiment changes are fixed offline data, hindsight-computed skills for
representation fitting, and IQL in place of SAC. The representation skill is detached during each representation
gradient update; policy rewards use the current representation directly, without an EMA copy.

Run the AntMaze Giant Navigate experiment from the repository root after activating the Conda environment:

```bash
bash impls/run_offline_metra.sh
```

The script evaluates two tasks with five episodes each every 10,000 updates, without recording videos or running
random-skill evaluation. Checkpoints are saved every 100,000 updates. `--skill_eval_horizon=1` recomputes the
goal-induced skill direction each step; set it to a larger integer to hold each direction for that many evaluation
steps. OGBench exposes episode-end markers without separate timeout labels, so this implementation treats every
marked end as terminal for the IQL bootstrap mask.

The representation network follows the neighboring METRA implementation's two 1024-unit ReLU layers, 2-D unit
skills, learning rate `1e-4`, initial multiplier 30, and `1e-3` constraint slack. IQL uses the standard AntMaze
settings of two 256-unit ReLU layers, learning rate `3e-4`, expectile `0.9`, AWR inverse temperature `10`, discount
`0.99`, target update `0.005`, batch size 256, and advantage-weight cap 100. This adopts AntMaze IQL's temperature;
LAVL's AWR temperature of 3 is not used for this AntMaze configuration.

Evaluation performs real environment rollouts: the agent observes the current state and environment goal, infers a
skill direction from the learned representation, and executes the IQL actor's actions until the episode ends. The
default `--eval_temperature=0` uses the actor's deterministic action. To evaluate a saved checkpoint once without
training, run from `impls/` and set the checkpoint directory and step:

```bash
python main.py \
  --project=OfflineMETRA-IQL \
  --env_name=antmaze-giant-navigate-v0 \
  --agent=agents/offline_metra.py \
  --agent.latent_dim=2 \
  --restore_path='exp/OfflineMETRA-IQL/antmaze-giant-navigate-v0/YOUR_RUN_NAME' \
  --restore_epoch=1000000 \
  --eval_only=True \
  --eval_episodes=50 \
  --skill_eval_episodes=0
```

Replace `YOUR_RUN_NAME` and the epoch with the directory and `params_<epoch>.pkl` you want to load. This writes a
one-time `eval.csv` under a new run directory. For deployment or custom inference, call
`agent.compute_skill(observation, goal)` and then `agent.sample_actions(observations=observation, goals=goal,
skill=skill, temperature=0)` at each step; `skill` can be recomputed every step or held for a chosen horizon.

## Online METRA on OGBench

`impls/run_online_metra_ogbench.sh` trains a separate online METRA agent directly in the OGBench AntMaze simulator.
It creates the simulator without loading or downloading the offline dataset. Each collected episode uses one skill
sampled uniformly from the unit circle; representation updates use that rollout skill and transition, and the
skill-conditioned policy is trained with SAC from replay. Network sizes, learning rates, constraint, entropy target,
rollout batch, replay warmup, update count, and epoch/log/evaluation/checkpoint intervals follow the sibling METRA
implementation and `run_metra_ant.sh`. Set `N_EPOCHS` before launching to shorten the run for a smoke run.

Evaluation infers `normalize(phi(goal) - phi(state))` from the OGBench goal observation and runs the policy
deterministically. The script evaluates two tasks for five episodes each without video. Training and evaluation use a
200-step horizon to match METRA's runner; reaching that sampler cutoff bootstraps the critic, while an environment
termination masks it.

## Reference
[1] Park, S., Frans, K., Eysenbach, B., and Levine, S. OGBench: Benchmarking offline goal-conditioned RL. In The Thirteenth International Conference on Learning Representations, 2025.
[2] Kostrikov, I., Nair, A., and Levine, S. Offline Reinforcement Learning with Implicit Q-Learning. [arXiv:2110.06169](https://arxiv.org/abs/2110.06169).
