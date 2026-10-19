#!/usr/bin/env bash
set -euo pipefail

cuda_device=0
cuda_requested=true
while (($#)); do
  case "$1" in
    --cuda)
      if (($# < 2)); then
        echo "--cuda requires a GPU number" >&2
        exit 2
      fi
      cuda_device="$2"
      cuda_requested=true
      shift 2
      ;;
    --cuda=*)
      cuda_device="${1#*=}"
      cuda_requested=true
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$cuda_requested" == true ]]; then
  if [[ ! "$cuda_device" =~ ^[0-9]+$ ]]; then
    echo "--cuda must be a non-negative GPU number" >&2
    exit 2
  fi
  export CUDA_VISIBLE_DEVICES="$cuda_device"
fi

cd "$(dirname "${BASH_SOURCE[0]}")"

python main.py \
  --project=OfflineMETRA-IQL-subgoal \
  --seed=0 \
  --env_name=pointmaze-medium-navigate-v0 \
  --train_steps="${N_EPOCHS:-1000000}" \
  --log_interval=100 \
  --eval_interval=1000 \
  --eval_episodes=50 \
  --eval_temperature=0 \
  --save_failed_eval_trajectories \
  --metra_goal_map \
  --save_representation \
  --video_episodes=0 \
  --skill_eval_episodes=0 \
  --skill_eval_horizon=1 \
  --save_interval=10000 \
  --agent=agents/offline_metra.py \
  --agent.batch_size=256 \
  --agent.updates_per_step=50 \
  --agent.latent_dim=2 \
  --agent.representation_hidden_dims='(1024,1024)' \
  --agent.high_level_hidden_dims='(1024,1024)' \
  --agent.actor_hidden_dims='(1024,1024)' \
  --agent.value_hidden_dims='(1024,1024)' \
  --agent.actor_action_epsilon=1e-6 \
  --agent.representation_lr=1e-4 \
  --agent.high_level_lr=1e-4 \
  --agent.high_level_beta=1.0 \
  --agent.high_level_max_weight=100.0 \
  --agent.iql_lr=1e-4 \
  --agent.lambda_lr=1e-4 \
  --agent.lambda_init=30.0 \
  --agent.constraint_epsilon=1e-3 \
  --agent.discount=0.999 \
  --agent.expectile=0.9 \
  --agent.awr_beta=10.0 \
  --agent.rep_future_steps=25
