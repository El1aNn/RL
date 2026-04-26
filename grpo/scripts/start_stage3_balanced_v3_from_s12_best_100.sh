#!/usr/bin/env bash
# Stage 3 v3: restart balanced alternating training from stage1_2 best buyer
# and stage2 best seller. V3 keeps the v1 shared reward backbone and adds
# softer leakage / unfair-split / low-utility-deal protections.
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
source "${script_dir}/common.sh"

cd "${script_dir}/../../.."

load_swanlab_env

resolve_s12_best_buyer_adapter() {
  local repo_root="${1:-$(pwd)}"

  if [[ -n "${STAGE1_BUYER_ADAPTER:-}" ]]; then
    printf '%s\n' "${STAGE1_BUYER_ADAPTER}"
    return 0
  fi

  local candidate
  for candidate in \
    "${repo_root}/checkpoints/grpo/stage1_2_buyer_vs_stage2_seller_100/best/buyer" \
    "${repo_root}/checkpoints/grpo/stage1_2_buyer_vs_stage2_seller_100/step_40/buyer" \
    "${repo_root}/checkpoints/grpo/stage1_rewardfix_guard_100/best/buyer"; do
    if [[ -d "${candidate}" && -f "${candidate}/adapter_config.json" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  echo "[stage3-v3] cannot find a buyer adapter." >&2
  echo "[stage3-v3] set STAGE1_BUYER_ADAPTER or finish stage1_2 first." >&2
  return 1
}

resolve_stage2_best_seller_adapter() {
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

  echo "[stage3-v3] cannot find a seller adapter." >&2
  echo "[stage3-v3] set STAGE2_SELLER_ADAPTER or finish stage2 first." >&2
  return 1
}

if pgrep -af "Final_project.grpo.train" >/dev/null; then
  echo "[stage3-v3] another GRPO training process is already running:" >&2
  pgrep -af "Final_project.grpo.train" >&2
  exit 1
fi

output_name="${STAGE3_V3_OUTPUT_NAME:-stage3_balanced_v3_from_s12_best_100}"
output_dir="/root/autodl-tmp/checkpoints/grpo/${output_name}"
if [[ -e "${output_dir}" && "${FORCE_STAGE3_V3:-0}" != "1" ]]; then
  echo "[stage3-v3] output_dir already exists: ${output_dir}" >&2
  echo "[stage3-v3] set FORCE_STAGE3_V3=1 to intentionally reuse it." >&2
  exit 1
fi

buyer_adapter="$(resolve_s12_best_buyer_adapter "$(pwd)")"
seller_adapter="$(resolve_stage2_best_seller_adapter "$(pwd)")"

balance_alpha="${STAGE3_BALANCE_ALPHA:-0.45}"
balance_scale="${STAGE3_BALANCE_SCALE:-1.0}"
balance_eps="${STAGE3_BALANCE_EPS:-1.0e-3}"
balance_gap_threshold="${STAGE3_BALANCE_GAP_THRESHOLD:-0.25}"
balance_gap_penalty="${STAGE3_BALANCE_GAP_PENALTY:--25.0}"
early_deal_min_rounds="${STAGE3_EARLY_DEAL_MIN_ROUNDS:-2}"
early_deal_penalty="${STAGE3_EARLY_DEAL_PENALTY:--4.0}"
seller_min_deal_util="${STAGE3_SELLER_MIN_DEAL_UTIL:-0.30}"
seller_low_util_deal_penalty="${STAGE3_SELLER_LOW_UTIL_DEAL_PENALTY:--25.0}"
buyer_min_deal_util="${STAGE3_BUYER_MIN_DEAL_UTIL:-0.25}"
buyer_low_util_deal_penalty="${STAGE3_BUYER_LOW_UTIL_DEAL_PENALTY:--20.0}"
leak_penalty="${STAGE3_LEAK_PENALTY:--25.0}"
learning_rate="${STAGE3_LEARNING_RATE:-7.0e-7}"
beta_kl="${STAGE3_BETA_KL:-0.10}"
per_device_train_batch_size="${STAGE3_PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
gradient_accumulation_steps="${STAGE3_GRADIENT_ACCUMULATION_STEPS:-2}"
policy_mini_batch_size="${STAGE3_POLICY_MINI_BATCH_SIZE:-1}"

echo "[stage3-v3] using buyer adapter: ${buyer_adapter}" >&2
echo "[stage3-v3] using seller adapter: ${seller_adapter}" >&2
echo "[stage3-v3] output_dir: ${output_dir}" >&2
echo "[stage3-v3] shared balance alpha: ${balance_alpha}" >&2
echo "[stage3-v3] balance gap threshold: ${balance_gap_threshold}" >&2
echo "[stage3-v3] balance gap penalty: ${balance_gap_penalty}" >&2
echo "[stage3-v3] early deal min rounds: ${early_deal_min_rounds}" >&2
echo "[stage3-v3] early deal penalty: ${early_deal_penalty}" >&2
echo "[stage3-v3] seller min deal util: ${seller_min_deal_util}" >&2
echo "[stage3-v3] buyer min deal util: ${buyer_min_deal_util}" >&2
echo "[stage3-v3] leak penalty: ${leak_penalty}" >&2
echo "[stage3-v3] learning_rate: ${learning_rate}" >&2
echo "[stage3-v3] beta_kl: ${beta_kl}" >&2

mkdir -p Final_project/logs

ts="$(date -u +%Y%m%d_%H%M%S)"
exp_name="grpo-stage3-balanced-v3-from-s12-best-100-${ts}"
log_file="/root/autodl-tmp/Final_project/logs/grpo_stage3_balanced_v3_from_s12_best_100_${ts}.log"

cmd=(
  python3 -u -m Final_project.grpo.train
  --config Final_project/grpo/configs/default.yaml
  --stage stage3
  --override "adapter_init.buyer=${buyer_adapter}"
  --override "adapter_init.seller=${seller_adapter}"
  --override "stage3.output_dir=./checkpoints/grpo/${output_name}"
  --override "stage3.total_steps=100"
  --override "stage3.eval_every=20"
  --override "stage3.save_every=40"
  --override "stage3.learning_rate=${learning_rate}"
  --override "stage3.beta_kl=${beta_kl}"
  --override "reward.enable_shared_balance_reward=true"
  --override "reward.shared_balance_alpha=${balance_alpha}"
  --override "reward.shared_balance_scale=${balance_scale}"
  --override "reward.shared_balance_eps=${balance_eps}"
  --override "reward.enable_deal_balance_penalty=true"
  --override "reward.deal_balance_gap_threshold=${balance_gap_threshold}"
  --override "reward.deal_balance_gap_penalty=${balance_gap_penalty}"
  --override "reward.enable_early_deal_penalty=true"
  --override "reward.early_deal_min_rounds=${early_deal_min_rounds}"
  --override "reward.early_deal_penalty=${early_deal_penalty}"
  --override "reward.enable_low_utility_deal_penalty=true"
  --override "reward.seller_min_deal_util=${seller_min_deal_util}"
  --override "reward.seller_low_util_deal_penalty=${seller_low_util_deal_penalty}"
  --override "reward.buyer_min_deal_util=${buyer_min_deal_util}"
  --override "reward.buyer_low_util_deal_penalty=${buyer_low_util_deal_penalty}"
  --override "reward.leak_penalty=${leak_penalty}"
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

printf '%s\n' "${pid}" > Final_project/logs/grpo_stage3_balanced_v3_from_s12_best_100.pid
printf '%s\n' "${log_file}" > Final_project/logs/grpo_stage3_balanced_v3_from_s12_best_100.logpath

echo "[stage3-v3] started balanced v3 alternating run"
echo "[stage3-v3] pid: ${pid}"
echo "[stage3-v3] log: ${log_file}"
echo "[stage3-v3] output_dir: ${output_dir}"
