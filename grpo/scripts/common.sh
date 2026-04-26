#!/usr/bin/env bash

load_swanlab_env() {
  if [[ -n "${SWANLAB_API:-}" || -n "${SWANLAB_API_KEY:-}" || ! -f "$HOME/.bashrc" ]]; then
    return 0
  fi

  eval "$(
    bash -ic '
      for name in SWANLAB_API SWANLAB_API_KEY; do
        value="${!name-}"
        if [[ -n "$value" ]]; then
          printf "export %s=%q\n" "$name" "$value"
        fi
      done
    ' 2>/dev/null
  )"
}

resolve_stage1_buyer_adapter() {
  local repo_root="${1:-$(pwd)}"

  if [[ -n "${STAGE1_BUYER_ADAPTER:-}" ]]; then
    printf '%s\n' "${STAGE1_BUYER_ADAPTER}"
    return 0
  fi

  local candidate
  for candidate in \
    "${repo_root}/checkpoints/grpo/stage1/best/buyer" \
    "${repo_root}/checkpoints/grpo/stage1/final/buyer"; do
    if [[ -d "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  local latest_path=""
  local latest_step=-1
  local step_dir=""
  local step_num=-1
  for candidate in "${repo_root}"/checkpoints/grpo/stage1/step_*/buyer; do
    [[ -d "${candidate}" ]] || continue
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

  echo "[stage2] cannot find a stage1 buyer adapter. Set STAGE1_BUYER_ADAPTER or finish stage1 first." >&2
  return 1
}
