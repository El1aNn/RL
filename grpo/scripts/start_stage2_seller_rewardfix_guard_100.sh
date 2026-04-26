#!/usr/bin/env bash
# Stage 2 short seller run against the reward-fixed + guarded stage1 buyer.
# Starts detached, writes pid/log metadata, and keeps old stage2 checkpoints intact.
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
source "${script_dir}/common.sh"

cd "${script_dir}/../../.."

load_swanlab_env

resolve_rewardfix_guard_buyer_adapter() {
  local repo_root="${1:-$(pwd)}"

  if [[ -n "${STAGE1_BUYER_ADAPTER:-}" ]]; then
    printf '%s\n' "${STAGE1_BUYER_ADAPTER}"
    return 0
  fi

  local candidate
  for candidate in \
    "${repo_root}/checkpoints/grpo/stage1_rewardfix_guard_100/best/buyer" \
    "${repo_root}/checkpoints/grpo/stage1_rewardfix_guard_100/final/buyer"; do
    if [[ -d "${candidate}" && -f "${candidate}/adapter_config.json" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  echo "[stage2] cannot find rewardfix+guard stage1 buyer adapter." >&2
  echo "[stage2] finish stage1_rewardfix_guard_100 first, or set STAGE1_BUYER_ADAPTER." >&2
  return 1
}

if pgrep -af "Final_project.grpo.train.*stage2_seller_rewardfix_guard_100" >/dev/null; then
  echo "[stage2] stage2_seller_rewardfix_guard_100 already appears to be running:" >&2
  pgrep -af "Final_project.grpo.train.*stage2_seller_rewardfix_guard_100" >&2
  exit 1
fi

stage1_buyer_adapter="$(resolve_rewardfix_guard_buyer_adapter "$(pwd)")"
echo "[stage2] using buyer adapter: ${stage1_buyer_adapter}" >&2

mkdir -p Final_project/logs

ts="$(date -u +%Y%m%d_%H%M%S)"
exp_name="grpo-stage2-seller-rewardfix-guard-100-${ts}"
log_file="/root/autodl-tmp/Final_project/logs/grpo_stage2_seller_rewardfix_guard_100_${ts}.log"

cmd=(
  python3 -u -m Final_project.grpo.train
  --config Final_project/grpo/configs/default.yaml
  --stage stage2
  --override "adapter_init.buyer=${stage1_buyer_adapter}"
  --override "stage2.output_dir=./checkpoints/grpo/stage2_seller_rewardfix_guard_100"
  --override "stage2.total_steps=100"
  --override "stage2.eval_every=25"
  --override "stage2.save_every=50"
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

printf '%s\n' "${pid}" > Final_project/logs/grpo_stage2_seller_rewardfix_guard_100.pid
printf '%s\n' "${log_file}" > Final_project/logs/grpo_stage2_seller_rewardfix_guard_100.logpath

echo "[stage2] started rewardfix+guard seller run"
echo "[stage2] pid: ${pid}"
echo "[stage2] log: ${log_file}"
echo "[stage2] output_dir: /root/autodl-tmp/checkpoints/grpo/stage2_seller_rewardfix_guard_100"
