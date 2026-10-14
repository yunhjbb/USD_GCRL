#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

python main.py \
  --project=OnlineMETRA \
  --seed=0 \
  --env_name=antmaze-giant-navigate-v0 \
  --train_steps="${N_EPOCHS:-1000000}" \
  --log_interval=100 \
  --eval_interval=1000 \
  --eval_tasks=2 \
  --eval_episodes=5 \
  --eval_max_episode_steps=200 \
  --eval_temperature=0 \
  --video_episodes=0 \
  --skill_eval_episodes=0 \
  --skill_eval_horizon=1 \
  --save_interval=10000 \
  --agent=agents/online_metra.py \
  --agent.latent_dim=2 \
  --agent.hidden_dims='(1024,1024)' \
  --agent.learning_rate=1e-4 \
  --agent.batch_size=256 \
  --agent.traj_batch_size=8 \
  --agent.trans_optimization_epochs=50 \
  --agent.min_buffer_size=10000 \
  --agent.max_buffer_size=1000000 \
  --agent.max_episode_steps=200 \
  --agent.alpha_init=0.01 \
  --agent.target_entropy_coef=1.0 \
  --agent.discount=0.99 \
  --agent.tau=0.005 \
  --agent.lambda_init=30.0 \
  --agent.dual_slack=1e-3
