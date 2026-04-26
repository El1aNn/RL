#!/usr/bin/env bash
# Stage 2: 训 seller adapter，buyer 冻结（使用 stage1 产出的 buyer）
set -euo pipefail

cd "$(dirname "$0")/../../.."

python -m Final_project.grpo.train \
  --config Final_project/grpo/configs/default.yaml \
  --stage stage2 \
  --override "adapter_init.buyer=./checkpoints/grpo/stage1/best" \
  "$@"
