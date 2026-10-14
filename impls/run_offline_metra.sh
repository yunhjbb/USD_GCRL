#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

python main.py \
  --project=OfflineMETRA-IQL \
  --seed=0 \
  --env_name=antmaze-giant-navigate-v0 \
  --train_steps=1000000 \
  --eval_interval=5000 \
  --eval_tasks=2 \
  --eval_episodes=5 \
  --video_episodes=0 \
  --skill_eval_episodes=0 \
  --skill_eval_horizon=1 \
  --save_interval=100000 \
  --agent=agents/offline_metra.py \
  --agent.batch_size=256 \
  --agent.updates_per_step=50 \
  --agent.latent_dim=2 \
  --agent.representation_lr=1e-4 \
  --agent.iql_lr=3e-4 \
  --agent.lambda_lr=1e-4 \
  --agent.lambda_init=30.0 \
  --agent.constraint_epsilon=1e-3 \
  --agent.discount=0.99 \
  --agent.expectile=0.9 \
  --agent.awr_beta=10.0 \
  --agent.min_future_gap=1
