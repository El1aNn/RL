#!/usr/bin/env bash
# Evaluation on test scenarios
# 用法:
#   ./eval.sh                   # 评估 ours (buyer/seller adapter 互打)
#   ./eval.sh sft_only          # 评估 SFT baseline
set -euo pipefail

cd "$(dirname "$0")/../../.."

MODE="${1:-ours}"

if [[ "$MODE" == "ours" ]]; then
  python -m Final_project.grpo.eval.run_eval \
    --base-model ./checkpoints/sft_base \
    --buyer-adapter ./checkpoints/grpo/stage3/best \
    --seller-adapter ./checkpoints/grpo/stage3/best \
    --scenarios Final_project/data/scenarios_rl_5k/test.jsonl \
    --group-size 4 \
    --output eval_results_ours.json \
    --baseline-mode ours
elif [[ "$MODE" == "sft_only" ]]; then
  python -m Final_project.grpo.eval.run_eval \
    --base-model ./checkpoints/sft_base \
    --scenarios Final_project/data/scenarios_rl_5k/test.jsonl \
    --group-size 4 \
    --output eval_results_sft_only.json \
    --baseline-mode sft_only
elif [[ "$MODE" == "zero_shot" ]]; then
  python -m Final_project.grpo.eval.run_eval \
    --base-model Qwen/Qwen2.5-3B-Instruct \
    --scenarios Final_project/data/scenarios_rl_5k/test.jsonl \
    --group-size 4 \
    --output eval_results_zero_shot.json \
    --baseline-mode zero_shot
else
  echo "Unknown mode: $MODE (ours|sft_only|zero_shot)"
  exit 1
fi
