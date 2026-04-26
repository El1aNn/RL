#!/usr/bin/env bash
# Stage 3 short alternating self-play run using the reward-fixed guarded chain.
# Starts detached, writes pid/log metadata, and keeps old stage3 checkpoints intact.
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
    "${repo_root}/checkpoints/grpo/stage1_2_buyer_vs_stage2_seller_100/best/buyer" \
    "${repo_root}/checkpoints/grpo/stage1_2_buyer_vs_stage2_seller_100/final/buyer" \
    "${repo_root}/checkpoints/grpo/stage1_2_buyer_vs_stage2_seller_100/step_100/buyer" \
    "${repo_root}/checkpoints/grpo/stage1_rewardfix_guard_100/best/buyer" \
    "${repo_root}/checkpoints/grpo/stage1_rewardfix_guard_100/final/buyer" \
    "${repo_root}/checkpoints/grpo/stage1_rewardfix_guard_100/step_100/buyer"; do
    if [[ -d "${candidate}" && -f "${candidate}/adapter_config.json" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  echo "[stage3] cannot find rewardfix+guard stage1 buyer adapter." >&2
  echo "[stage3] finish stage1_2_buyer_vs_stage2_seller_100 or stage1_rewardfix_guard_100 first, or set STAGE1_BUYER_ADAPTER." >&2
  return 1
}

resolve_rewardfix_guard_seller_adapter() {
  local repo_root="${1:-$(pwd)}"

  if [[ -n "${STAGE2_SELLER_ADAPTER:-}" ]]; then
    printf '%s\n' "${STAGE2_SELLER_ADAPTER}"
    return 0
  fi

  local root="${repo_root}/checkpoints/grpo/stage2_seller_rewardfix_guard_100"
  local candidate
  for candidate in \
    "${root}/best/seller" \
    "${root}/final/seller" \
    "${root}/step_100/seller" \
    "${root}/step_50/seller"; do
    if [[ -d "${candidate}" && -f "${candidate}/adapter_config.json" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  local latest_path=""
  local latest_step=-1
  local step_dir=""
  local step_num=-1
  for candidate in "${root}"/step_*/seller; do
    [[ -d "${candidate}" && -f "${candidate}/adapter_config.json" ]] || continue
    step_dir="$(basename "$(dirname "${candidate}")")"
    step_num="${step_dir#step_}"
    if [[ "${step_num}" =~ ^[0-9]+$ ]] && (( step_num > latest_step )); then
      latest_step="${step_num}"
      latest_path="${candidate}"
    fi
  done

  if [[ -n "${latest_path}" ]]; then
    printf '%s\n' "${latest_path}"
    return 0
  fi

  echo "[stage3] cannot find rewardfix+guard stage2 seller adapter." >&2
  echo "[stage3] finish stage2_seller_rewardfix_guard_100 first, or set STAGE2_SELLER_ADAPTER." >&2
  return 1
}

if pgrep -af "Final_project.grpo.train.*stage2_seller_rewardfix_guard_100" >/dev/null; then
  echo "[stage3] stage2_seller_rewardfix_guard_100 is still running; wait for it to finish before stage3." >&2
  pgrep -af "Final_project.grpo.train.*stage2_seller_rewardfix_guard_100" >&2
  exit 1
fi

if pgrep -af "Final_project.grpo.train.*stage3_alternating_rewardfix_guard_100" >/dev/null; then
  echo "[stage3] stage3_alternating_rewardfix_guard_100 already appears to be running:" >&2
  pgrep -af "Final_project.grpo.train.*stage3_alternating_rewardfix_guard_100" >&2
  exit 1
fi

stage1_buyer_adapter="$(resolve_rewardfix_guard_buyer_adapter "$(pwd)")"
stage2_seller_adapter="$(resolve_rewardfix_guard_seller_adapter "$(pwd)")"

echo "[stage3] using buyer adapter: ${stage1_buyer_adapter}" >&2
echo "[stage3] using seller adapter: ${stage2_seller_adapter}" >&2

mkdir -p Final_project/logs

ts="$(date -u +%Y%m%d_%H%M%S)"
exp_name="grpo-stage3-alternating-rewardfix-guard-100-${ts}"
log_file="/root/autodl-tmp/Final_project/logs/grpo_stage3_alternating_rewardfix_guard_100_${ts}.log"

per_device_train_batch_size="${STAGE3_PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
gradient_accumulation_steps="${STAGE3_GRADIENT_ACCUMULATION_STEPS:-2}"
policy_mini_batch_size="${STAGE3_POLICY_MINI_BATCH_SIZE:-1}"

echo "[stage3] per_device_train_batch_size: ${per_device_train_batch_size}" >&2
echo "[stage3] gradient_accumulation_steps: ${gradient_accumulation_steps}" >&2
echo "[stage3] policy_mini_batch_size: ${policy_mini_batch_size}" >&2

cmd=(
  python3 -u -m Final_project.grpo.train
  --config Final_project/grpo/configs/default.yaml
  --stage stage3
  --override "adapter_init.buyer=${stage1_buyer_adapter}"
  --override "adapter_init.seller=${stage2_seller_adapter}"
  --override "stage3.output_dir=./checkpoints/grpo/stage3_alternating_rewardfix_guard_100"
  --override "stage3.total_steps=100"
  --override "stage3.eval_every=20"
  --override "stage3.save_every=40"
  --override "stage3.learning_rate=1.0e-6"
  --override "stage3.beta_kl=0.08"
  --override "model.max_model_len=2048"
  --override "vllm.enforce_eager=true"
  --override "vllm.gpu_memory_utilization=0.60"
  --override "rollout.group_size=8"
  --override "rollout.max_new_tokens=96"
  --override "train.per_device_train_batch_size=${per_device_train_batch_size}"
  --override "train.gradient_accumulation_steps=${gradient_accumulation_steps}"
  --override "train.policy_mini_batch_size=${policy_mini_batch_size}"
  --override "train.max_prompt_length=1536"
  --override "train.max_completion_length=96"
  --override "swanlab.experiment_name=${exp_name}"
)

cmd+=("$@")

setsid bash -lc "cd /root/autodl-tmp && exec \"\${@}\" > '${log_file}' 2>&1" bash "${cmd[@]}" < /dev/null &
pid=$!

printf '%s\n' "${pid}" > Final_project/logs/grpo_stage3_alternating_rewardfix_guard_100.pid
printf '%s\n' "${log_file}" > Final_project/logs/grpo_stage3_alternating_rewardfix_guard_100.logpath

echo "[stage3] started rewardfix+guard alternating run"
echo "[stage3] pid: ${pid}"
echo "[stage3] log: ${log_file}"
echo "[stage3] output_dir: /root/autodl-tmp/checkpoints/grpo/stage3_alternating_rewardfix_guard_100"
