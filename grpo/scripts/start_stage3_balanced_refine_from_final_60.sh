#!/usr/bin/env bash
# Continue balanced stage3 from the current balanced final adapters.
# This short refine run teaches the new leakage / early-deal / balance penalties.
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
source "${script_dir}/common.sh"

cd "${script_dir}/../../.."

load_swanlab_env

if pgrep -af "Final_project.grpo.train" >/dev/null; then
  echo "[stage3-refine] another GRPO training process is already running:" >&2
  pgrep -af "Final_project.grpo.train" >&2
  exit 1
fi

buyer_adapter="${STAGE3_REFINE_BUYER_ADAPTER:-/root/autodl-tmp/checkpoints/grpo/stage3_balanced_from_s12_best_100/final/buyer}"
seller_adapter="${STAGE3_REFINE_SELLER_ADAPTER:-/root/autodl-tmp/checkpoints/grpo/stage3_balanced_from_s12_best_100/final/seller}"

for adapter in "${buyer_adapter}" "${seller_adapter}"; do
  if [[ ! -f "${adapter}/adapter_config.json" ]]; then
    echo "[stage3-refine] missing adapter: ${adapter}" >&2
    exit 1
  fi
done

output_name="${STAGE3_REFINE_OUTPUT_NAME:-stage3_balanced_refine_final_60}"
output_dir="/root/autodl-tmp/checkpoints/grpo/${output_name}"
if [[ -e "${output_dir}" && "${FORCE_STAGE3_REFINE:-0}" != "1" ]]; then
  echo "[stage3-refine] output_dir already exists: ${output_dir}" >&2
  echo "[stage3-refine] set FORCE_STAGE3_REFINE=1 to intentionally reuse it." >&2
  exit 1
fi

balance_alpha="${STAGE3_BALANCE_ALPHA:-0.45}"
balance_scale="${STAGE3_BALANCE_SCALE:-1.0}"
balance_eps="${STAGE3_BALANCE_EPS:-1.0e-3}"
balance_gap_threshold="${STAGE3_BALANCE_GAP_THRESHOLD:-0.22}"
balance_gap_penalty="${STAGE3_BALANCE_GAP_PENALTY:--50.0}"
early_deal_min_rounds="${STAGE3_EARLY_DEAL_MIN_ROUNDS:-2}"
early_deal_penalty="${STAGE3_EARLY_DEAL_PENALTY:--6.0}"
learning_rate="${STAGE3_LEARNING_RATE:-5.0e-7}"
beta_kl="${STAGE3_BETA_KL:-0.12}"

echo "[stage3-refine] using buyer adapter: ${buyer_adapter}" >&2
echo "[stage3-refine] using seller adapter: ${seller_adapter}" >&2
echo "[stage3-refine] output_dir: ${output_dir}" >&2
echo "[stage3-refine] balance alpha: ${balance_alpha}" >&2
echo "[stage3-refine] learning_rate: ${learning_rate}" >&2
echo "[stage3-refine] beta_kl: ${beta_kl}" >&2

mkdir -p Final_project/logs

ts="$(date -u +%Y%m%d_%H%M%S)"
exp_name="grpo-stage3-balanced-refine-final-60-${ts}"
log_file="/root/autodl-tmp/Final_project/logs/grpo_stage3_balanced_refine_final_60_${ts}.log"

cmd=(
  python3 -u -m Final_project.grpo.train
  --config Final_project/grpo/configs/default.yaml
  --stage stage3
  --override "adapter_init.buyer=${buyer_adapter}"
  --override "adapter_init.seller=${seller_adapter}"
  --override "stage3.output_dir=./checkpoints/grpo/${output_name}"
  --override "stage3.total_steps=60"
  --override "stage3.eval_every=20"
  --override "stage3.save_every=30"
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
  --override "model.max_model_len=2048"
  --override "vllm.enforce_eager=true"
  --override "vllm.gpu_memory_utilization=0.60"
  --override "rollout.group_size=8"
  --override "rollout.max_new_tokens=96"
  --override "train.per_device_train_batch_size=2"
  --override "train.gradient_accumulation_steps=2"
  --override "train.policy_mini_batch_size=1"
  --override "train.max_prompt_length=1536"
  --override "train.max_completion_length=96"
  --override "swanlab.experiment_name=${exp_name}"
)

cmd+=("$@")

setsid bash -lc "cd /root/autodl-tmp && exec \"\${@}\" > '${log_file}' 2>&1" bash "${cmd[@]}" < /dev/null &
pid=$!

printf '%s\n' "${pid}" > Final_project/logs/grpo_stage3_balanced_refine_final_60.pid
printf '%s\n' "${log_file}" > Final_project/logs/grpo_stage3_balanced_refine_final_60.logpath

echo "[stage3-refine] started balanced refine run"
echo "[stage3-refine] pid: ${pid}"
echo "[stage3-refine] log: ${log_file}"
echo "[stage3-refine] output_dir: ${output_dir}"
