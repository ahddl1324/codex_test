#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
SWEEP_PID="${1:?usage: $0 SWEEP_PID}"
BASE_DIR="$ROOT_DIR/sweep_results_10epoch"
LOG_DIR="$BASE_DIR/logs"
STATUS_DIR="$BASE_DIR/status"
RESULT_FILE="$BASE_DIR/results.txt"

while kill -0 "$SWEEP_PID" 2>/dev/null; do
  sleep 60
done

ids=(e01 e02 e03 e04 e05 e06 e07 e08)
conv_filters=(20 40 40 80 20 20 20 20)
local_strides=(10 10 20 40 10 10 10 10)
conv2_kernels=(19 17 9 4 19 19 19 19)
n_hids=(32 32 32 32 16 64 32 32)
gru_hiddens=(32 32 32 32 32 64 64 32)
batch_sizes=(8 8 8 8 8 4 8 16)
lrs=(5e-4 5e-4 5e-4 5e-4 5e-4 5e-4 1e-4 1e-3)

{
  printf '10-epoch hyperparameter sweep results\n'
  printf 'wandb: disabled\n'
  printf 'effective receptive field constraint: conv_filter_size + local_conv_stride * (conv2_time_kernel - 1) = 200\n'
  printf 'selection metric: best validation loss (lower is better)\n\n'
  printf '%-4s %-5s %-6s %-6s %-6s %-6s %-6s %-8s %-8s %-14s %-14s\n' \
    ID CF LS C2K NH GRU BATCH LR STATUS BEST_VAL TEST_LOSS

  for i in "${!ids[@]}"; do
    id="${ids[$i]}"
    log_file="$LOG_DIR/$id.log"
    status_file="$STATUS_DIR/$id.status"
    exit_code="missing"
    best_val="N/A"
    test_loss="N/A"

    if [[ -f "$status_file" ]]; then
      exit_code="$(tr -d '[:space:]' < "$status_file")"
    fi

    if [[ -f "$log_file" ]]; then
      read -r best_val test_loss < <(
        awk '/Best val:/ {best=$3; test=$6} END {if (best == "") best="N/A"; if (test == "") test="N/A"; print best, test}' "$log_file"
      )
    fi

    if [[ "$exit_code" == "0" ]]; then
      status="OK"
    elif [[ "$exit_code" == "missing" ]]; then
      status="INCOMPLETE"
    else
      status="FAILED($exit_code)"
    fi

    printf '%-4s %-5s %-6s %-6s %-6s %-6s %-6s %-8s %-8s %-14s %-14s\n' \
      "$id" "${conv_filters[$i]}" "${local_strides[$i]}" "${conv2_kernels[$i]}" \
      "${n_hids[$i]}" "${gru_hiddens[$i]}" "${batch_sizes[$i]}" "${lrs[$i]}" \
      "$status" "$best_val" "$test_loss"
  done

  printf '\nRaw logs: %s\n' "$LOG_DIR"
  printf 'Checkpoints: %s\n' "$ROOT_DIR/newSTDP/exp/model_ckpt"
} > "$RESULT_FILE"

printf 'result collection complete: %s\n' "$RESULT_FILE"
