#!/usr/bin/env bash
# Stage 3: 交替 self-play（低 lr + 高 KL，stage1 stage2 都已完成）
set -euo pipefail

cd "$(dirname "$0")/../../.."

python -m Final_project.grpo.train \
  --config Final_project/grpo/configs/default.yaml \
  --stage stage3 \
  --override "adapter_init.buyer=./checkpoints/grpo/stage1/best" \
  --override "adapter_init.seller=./checkpoints/grpo/stage2/best" \
  "$@"
