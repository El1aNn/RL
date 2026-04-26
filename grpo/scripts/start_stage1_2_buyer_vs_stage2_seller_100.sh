#!/usr/bin/env bash
# Stage 1.2: continue buyer training against a frozen strong stage2 seller.
# Starts detached, writes pid/log metadata, and keeps old checkpoints intact.
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
source "${script_dir}/common.sh"

cd "${script_dir}/../../.."

load_swanlab_env

resolve_stage1_2_buyer_adapter() {
  local repo_root="${1:-$(pwd)}"

  if [[ -n "${STAGE1_2_BUYER_ADAPTER:-}" ]]; then
    printf '%s\n' "${STAGE1_2_BUYER_ADAPTER}"
    return 0
  fi

  local candidate
  for candidate in \
    "${repo_root}/checkpoints/grpo/stage1_rewardfix_guard_100/best/buyer" \
    "${repo_root}/checkpoints/grpo/stage1_rewardfix_guard_100/final/buyer" \
    "${repo_root}/checkpoints/grpo/stage1_rewardfix_guard_100/step_100/buyer"; do
    if [[ -d "${candidate}" && -f "${candidate}/adapter_config.json" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  echo "[stage1_2] cannot find stage1 rewardfix+guard buyer adapter." >&2
  echo "[stage1_2] set STAGE1_2_BUYER_ADAPTER or finish stage1_rewardfix_guard_100 first." >&2
  return 1
}

resolve_stage2_seller_adapter() {
  local repo_root="${1:-$(pwd)}"

  if [[ -n "${STAGE2_SELLER_ADAPTER:-}" ]]; then
    printf '%s\n' "${STAGE2_SELLER_ADAPTER}"
    return 0
  fi

  local root="${repo_root}/checkpoints/grpo/stage2_seller_rewardfix_guard_100"
  local candidate
  for candidate in \
    "${root}/step_100/seller" \
    "${root}/final/seller" \
    "${root}/step_50/seller" \
    "${root}/best/seller"; do
    if [[ -d "${candidate}" && -f "${candidate}/adapter_config.json" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  echo "[stage1_2] cannot find stage2 seller adapter." >&2
  echo "[stage1_2] set STAGE2_SELLER_ADAPTER or finish stage2_seller_rewardfix_guard_100 first." >&2
  return 1
}

if pgrep -af "Final_project.grpo.train" >/dev/null; then
  echo "[stage1_2] another GRPO training process is already running:" >&2
  pgrep -af "Final_project.grpo.train" >&2
  exit 1
fi

if pgrep -af "Final_project.grpo.train.*stage1_2_buyer_vs_stage2_seller_100" >/dev/null; then
  echo "[stage1_2] stage1_2_buyer_vs_stage2_seller_100 already appears to be running:" >&2
  pgrep -af "Final_project.grpo.train.*stage1_2_buyer_vs_stage2_seller_100" >&2
  exit 1
fi

output_name="${STAGE1_2_OUTPUT_NAME:-stage1_2_buyer_vs_stage2_seller_100}"
output_dir="/root/autodl-tmp/checkpoints/grpo/${output_name}"
if [[ -e "${output_dir}" && "${FORCE_STAGE1_2:-0}" != "1" ]]; then
  echo "[stage1_2] output_dir already exists: ${output_dir}" >&2
  echo "[stage1_2] set FORCE_STAGE1_2=1 to intentionally reuse it." >&2
  exit 1
fi

buyer_adapter="$(resolve_stage1_2_buyer_adapter "$(pwd)")"
seller_adapter="$(resolve_stage2_seller_adapter "$(pwd)")"

echo "[stage1_2] using buyer adapter: ${buyer_adapter}" >&2
echo "[stage1_2] using frozen seller adapter: ${seller_adapter}" >&2

mkdir -p Final_project/logs

ts="$(date -u +%Y%m%d_%H%M%S)"
exp_name="grpo-stage1-2-buyer-vs-stage2-seller-100-${ts}"
log_file="/root/autodl-tmp/Final_project/logs/grpo_stage1_2_buyer_vs_stage2_seller_100_${ts}.log"

cmd=(
  python3 -u -m Final_project.grpo.train
  --config Final_project/grpo/configs/default.yaml
  --stage stage1_2
  --override "adapter_init.buyer=${buyer_adapter}"
  --override "adapter_init.seller=${seller_adapter}"
  --override "stage1_2.output_dir=./checkpoints/grpo/${output_name}"
  --override "stage1_2.total_steps=100"
  --override "stage1_2.eval_every=20"
  --override "stage1_2.save_every=40"
  --override "stage1_2.learning_rate=3.0e-6"
  --override "stage1_2.beta_kl=0.06"
  --override "reward.enable_buyer_budget_pressure_penalty=true"
  --override "reward.format_bonus=0.2"
  --override "reward.leak_penalty=-30.0"
  --override "reward.buyer_near_budget_offer_ratio=0.98"
  --override "reward.buyer_near_budget_offer_penalty=-8.0"
  --override "reward.buyer_first_offer_budget_ratio=0.95"
  --override "reward.buyer_first_offer_budget_penalty=-15.0"
  --override "reward.buyer_near_budget_deal_ratio=0.98"
  --override "reward.buyer_near_budget_deal_penalty=-20.0"
  --override "model.max_model_len=2048"
  --override "vllm.enforce_eager=true"
  --override "vllm.gpu_memory_utilization=0.60"
  --override "rollout.group_size=8"
  --override "rollout.max_new_tokens=96"
  --override "train.per_device_train_batch_size=2"
  --override "train.gradient_accumulation_steps=1"
  --override "train.policy_mini_batch_size=1"
  --override "train.max_prompt_length=1536"
  --override "train.max_completion_length=96"
  --override "swanlab.experiment_name=${exp_name}"
)

cmd+=("$@")

setsid bash -lc "cd /root/autodl-tmp && exec \"\${@}\" > '${log_file}' 2>&1" bash "${cmd[@]}" < /dev/null &
pid=$!

printf '%s\n' "${pid}" > Final_project/logs/grpo_stage1_2_buyer_vs_stage2_seller_100.pid
printf '%s\n' "${log_file}" > Final_project/logs/grpo_stage1_2_buyer_vs_stage2_seller_100.logpath

echo "[stage1_2] started buyer-vs-stage2-seller run"
echo "[stage1_2] pid: ${pid}"
echo "[stage1_2] log: ${log_file}"
echo "[stage1_2] output_dir: ${output_dir}"
