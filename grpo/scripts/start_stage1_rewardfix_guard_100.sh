#!/usr/bin/env bash
# Stage 1 short buyer run with reward fixes + cold-start seller guard.
# Starts detached, writes pid/log metadata, and keeps old stage1 checkpoints intact.
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
source "${script_dir}/common.sh"

cd "${script_dir}/../../.."

load_swanlab_env

if pgrep -af "Final_project.grpo.train.*stage1_rewardfix_guard_100" >/dev/null; then
  echo "[stage1] stage1_rewardfix_guard_100 already appears to be running:" >&2
  pgrep -af "Final_project.grpo.train.*stage1_rewardfix_guard_100" >&2
  exit 1
fi

mkdir -p Final_project/logs

ts="$(date -u +%Y%m%d_%H%M%S)"
exp_name="grpo-stage1-rewardfix-guard-100-${ts}"
log_file="/root/autodl-tmp/Final_project/logs/grpo_stage1_rewardfix_guard_100_${ts}.log"

cmd=(
  python3 -u -m Final_project.grpo.train
  --config Final_project/grpo/configs/default.yaml
  --stage stage1
  --override "stage1.output_dir=./checkpoints/grpo/stage1_rewardfix_guard_100"
  --override "stage1.total_steps=100"
  --override "stage1.eval_every=25"
  --override "stage1.save_every=50"
  --override "stage1.seller_cold_guard.enabled=true"
  --override "stage1.seller_cold_guard.min_cost_ratio=0.8"
  --override "stage1.seller_cold_guard.consecutive_below_cost=2"
  --override "stage1.seller_cold_guard.walkaway_on_bad_deal=true"
  --override "stage1.seller_cold_guard.walkaway_on_invalid_after_low=true"
  --override "model.max_model_len=2048"
  --override "vllm.enforce_eager=true"
  --override "vllm.gpu_memory_utilization=0.65"
  --override "rollout.group_size=8"
  --override "rollout.max_new_tokens=96"
  --override "train.per_device_train_batch_size=1"
  --override "train.policy_mini_batch_size=2"
  --override "train.max_prompt_length=1536"
  --override "train.max_completion_length=96"
  --override "swanlab.experiment_name=${exp_name}"
)

setsid bash -lc "cd /root/autodl-tmp && exec \"\${@}\" > '${log_file}' 2>&1" bash "${cmd[@]}" < /dev/null &
pid=$!

printf '%s\n' "${pid}" > Final_project/logs/grpo_stage1_rewardfix_guard_100.pid
printf '%s\n' "${log_file}" > Final_project/logs/grpo_stage1_rewardfix_guard_100.logpath

echo "[stage1] started rewardfix+guard buyer run"
echo "[stage1] pid: ${pid}"
echo "[stage1] log: ${log_file}"
echo "[stage1] output_dir: /root/autodl-tmp/checkpoints/grpo/stage1_rewardfix_guard_100"
