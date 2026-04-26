#!/usr/bin/env bash
# Stage 2: seller 训练，高显存利用率版本（约 80%+ GPU memory on A800 80GB）
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
source "${script_dir}/common.sh"

cd "${script_dir}/../../.."

load_swanlab_env

stage1_buyer_adapter="$(resolve_stage1_buyer_adapter "$(pwd)")"
echo "[stage2] using buyer adapter: ${stage1_buyer_adapter}" >&2

ts="$(date -u +%Y%m%d_%H%M%S)"
exp_name="grpo-stage2-250-mem80b-${ts}"

python3 -u -m Final_project.grpo.train \
  --config Final_project/grpo/configs/default.yaml \
  --stage stage2 \
  --override "adapter_init.buyer=${stage1_buyer_adapter}" \
  --override "stage2.total_steps=250" \
  --override "model.max_model_len=2048" \
  --override "vllm.enforce_eager=true" \
  --override "vllm.gpu_memory_utilization=0.70" \
  --override "rollout.group_size=8" \
  --override "rollout.max_new_tokens=96" \
  --override "train.per_device_train_batch_size=1" \
  --override "train.policy_mini_batch_size=2" \
  --override "train.max_prompt_length=1536" \
  --override "train.max_completion_length=96" \
  --override "swanlab.experiment_name=${exp_name}" \
  "$@"
