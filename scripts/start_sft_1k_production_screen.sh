#!/bin/zsh

set -euo pipefail

ROOT="${ROOT:-/Users/bytedance/cuhksz/CSC6129/Final_project}"
RUN_SCRIPT="$ROOT/scripts/run_sft_2k_production.sh"
LOG_DIR="$ROOT/logs"
SESSION_PREFIX="${SESSION_PREFIX:-sft_1k_prod}"
SESSION="${1:-${SESSION_PREFIX}_$(date +%Y%m%d_%H%M%S)}"
LOG="$LOG_DIR/${SESSION}.log"

LATEST_SESSION_FILE="$LOG_DIR/sft_1k_production.session"
LATEST_LOG_FILE="$LOG_DIR/sft_1k_production.logpath"
LATEST_PID_FILE="$LOG_DIR/sft_1k_production.pid"

if [[ -z "${ARK_API_KEY:-}" ]]; then
  print -u2 "ERROR: ARK_API_KEY is not set. Export it before starting production."
  exit 1
fi

if ! command -v screen >/dev/null 2>&1; then
  print -u2 "ERROR: screen is not installed or not on PATH."
  exit 1
fi

if [[ ! -f "$RUN_SCRIPT" ]]; then
  print -u2 "ERROR: production script not found: $RUN_SCRIPT"
  exit 1
fi

worker_count="$(grep -c -- "--max-workers 15" "$RUN_SCRIPT" || true)"
if [[ "$worker_count" -lt 3 ]]; then
  print -u2 "ERROR: expected train/val/test commands in $RUN_SCRIPT to use --max-workers 15."
  exit 1
fi

mkdir -p "$LOG_DIR"

screen_list="$(screen -ls || true)"
if printf "%s\n" "$screen_list" | awk -v suffix=".$SESSION" 'index($1, suffix) { found = 1 } END { exit found ? 0 : 1 }'; then
  print -u2 "ERROR: screen session already exists: $SESSION"
  exit 1
fi

screen -dmS "$SESSION" env \
  ROOT="$ROOT" \
  RUN_SCRIPT="$RUN_SCRIPT" \
  SESSION_NAME="$SESSION" \
  LOG="$LOG" \
  zsh -lc '
    {
      printf "SESSION=%s\n" "$SESSION_NAME"
      printf "LOG=%s\n" "$LOG"
      printf "RUN_SCRIPT=%s\n" "$RUN_SCRIPT"
      printf "STARTED_AT=%s\n\n" "$(date "+%Y-%m-%d %H:%M:%S %Z")"

      cd "$ROOT"
      cd_status=$?
      if [[ "$cd_status" -ne 0 ]]; then
        printf "\nFINISHED_AT=%s\nEXIT_STATUS=%s\n" "$(date "+%Y-%m-%d %H:%M:%S %Z")" "$cd_status"
        exit "$cd_status"
      fi

      zsh "$RUN_SCRIPT"
      exit_status=$?
      printf "\nFINISHED_AT=%s\nEXIT_STATUS=%s\n" "$(date "+%Y-%m-%d %H:%M:%S %Z")" "$exit_status"
      exit "$exit_status"
    } > "$LOG" 2>&1
  '

sleep 1

screen_list="$(screen -ls || true)"
screen_token="$(printf "%s\n" "$screen_list" | awk -v suffix=".$SESSION" 'index($1, suffix) { print $1; exit }')"
if [[ -z "$screen_token" ]]; then
  print -u2 "ERROR: screen session exited immediately: $SESSION"
  print -u2 "Log: $LOG"
  tail -n 40 "$LOG" >&2 || true
  exit 1
fi

screen_pid="${screen_token%%.*}"
printf "%s\n" "$SESSION" > "$LATEST_SESSION_FILE"
printf "%s\n" "$LOG" > "$LATEST_LOG_FILE"
printf "%s\n" "$screen_pid" > "$LATEST_PID_FILE"

print "Started screen session: $SESSION"
print "Screen pid: $screen_pid"
print "Log: $LOG"
print "Attach: screen -r $SESSION"
print "Tail: tail -f $LOG"
