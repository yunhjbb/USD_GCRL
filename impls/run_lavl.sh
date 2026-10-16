#!/usr/bin/env bash
set -euo pipefail

# Run from this script's directory so the agent and dataset paths resolve.
cd "$(dirname "${BASH_SOURCE[0]}")"

python main.py \
  --project=LAVL \
  --seed=0 \
  --env_name=pointmaze-medium-navigate-v0 \
  --train_steps=1000000 \
  --eval_interval=100000 \
  --eval_episodes=50 \
  --video_episodes=0 \
  --save_interval=100000 \
  --agent=agents/lavl.py \
  --agent.discount=0.999 \
  --agent.expectile=0.9 \
  --agent.smoothness_weight=10.0 \
  --agent.low_actor_rep_grad=True \
  --agent.high_actor_rep_grad=True \
  --agent.high_alpha=1.0 \
  --agent.low_alpha=3.0 \
  --agent.subgoal_steps=25
