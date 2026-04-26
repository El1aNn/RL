#!/usr/bin/env bash
# Stage 1: 训 buyer adapter，seller 冻结（从 SFT 初始化）
set -euo pipefail

cd "$(dirname "$0")/../../.."

python -m Final_project.grpo.train \
  --config Final_project/grpo/configs/default.yaml \
  --stage stage1 \
  "$@"
