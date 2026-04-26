#!/usr/bin/env bash
# Stage 1: buyer 训练，高显存利用率版本（约 80%+ GPU memory on A800 80GB）
set -euo pipefail

cd "$(dirname "$0")/../../.."

# ~/.bashrc 在非交互 shell 里会提前 return，这里用交互 bash 取出 SwanLab 相关环境变量。
if [[ -z "${SWANLAB_API:-}" && -z "${SWANLAB_API_KEY:-}" && -f "$HOME/.bashrc" ]]; then
  eval "$(
    bash -ic '
      for name in SWANLAB_API SWANLAB_API_KEY; do
        value="${!name-}"
        if [[ -n "$value" ]]; then
          printf "export %s=%q\n" "$name" "$value"
        fi
      done
    ' 2>/dev/null
  )"
fi

ts="$(date -u +%Y%m%d_%H%M%S)"
exp_name="grpo-stage1-250-mem80b-${ts}"

python3 -u -m Final_project.grpo.train \
  --config Final_project/grpo/configs/default.yaml \
  --stage stage1 \
  --override "stage1.total_steps=250" \
  --override "model.max_model_len=2048" \
  --override "vllm.enforce_eager=true" \
  --override "vllm.gpu_memory_utilization=0.65" \
  --override "rollout.group_size=8" \
  --override "rollout.max_new_tokens=96" \
  --override "train.per_device_train_batch_size=1" \
  --override "train.policy_mini_batch_size=2" \
  --override "train.max_prompt_length=1536" \
  --override "train.max_completion_length=96" \
  --override "swanlab.experiment_name=${exp_name}" \
  "$@"
