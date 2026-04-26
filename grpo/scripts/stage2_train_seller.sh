#!/usr/bin/env bash
# Stage 2: 训 seller adapter，buyer 冻结（使用 stage1 产出的 buyer）
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
source "${script_dir}/common.sh"

cd "${script_dir}/../../.."

load_swanlab_env

stage1_buyer_adapter="$(resolve_stage1_buyer_adapter "$(pwd)")"
echo "[stage2] using buyer adapter: ${stage1_buyer_adapter}" >&2

python3 -u -m Final_project.grpo.train \
  --config Final_project/grpo/configs/default.yaml \
  --stage stage2 \
  --override "adapter_init.buyer=${stage1_buyer_adapter}" \
  "$@"
