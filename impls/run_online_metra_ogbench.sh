#!/usr/bin/env bash
set -euo pipefail

cuda_device=1
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
  --project=OnlineMETRA \
  --seed=0 \
  --env_name=pointmaze-medium-navigate-v0 \
  --train_steps="${N_EPOCHS:-1000000}" \
  --log_interval=100 \
  --eval_interval=200 \
  --eval_tasks=2 \
  --eval_episodes=5 \
  --eval_max_episode_steps=1000 \
  --metra_goal_map \
  --save_representation \
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
  --agent.max_episode_steps=1000 \
  --agent.alpha_init=0.01 \
  --agent.target_entropy_coef=1.0 \
  --agent.discount=0.999 \
  --agent.tau=0.005 \
  --agent.lambda_init=30.0 \
  --agent.dual_slack=1e-3
