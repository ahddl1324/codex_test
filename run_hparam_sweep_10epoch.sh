#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$ROOT_DIR/sweep_results_10epoch/logs"
STATUS_DIR="$ROOT_DIR/sweep_results_10epoch/status"
mkdir -p "$LOG_DIR" "$STATUS_DIR"

run_one() {
  local gpu_group="$1"
  local id="$2"
  local conv_filter="$3"
  local local_stride="$4"
  local conv2_kernel="$5"
  local n_hid="$6"
  local gru_hidden="$7"
  local batch_size="$8"
  local lr="$9"
  local exp_name="sweep10_${id}_cf${conv_filter}_cs${local_stride}_ck${conv2_kernel}_nh${n_hid}_gh${gru_hidden}_b${batch_size}_lr${lr}"
  local log_file="$LOG_DIR/${id}.log"

  printf '%s\n' "START $exp_name GPUs=$gpu_group" | tee "$log_file"

  CUDA_VISIBLE_DEVICES="$gpu_group" torchrun --standalone --nproc_per_node=4 \
    "$ROOT_DIR/main_newSTDP_multitrial_recurrent_concat_mlp_2conv_chunk_sp_chunk.py" \
    --distributed \
    --exp_name "$exp_name" \
    --train_list \
      "$ROOT_DIR/data/data10/spike_seed6_300k.npy" \
      "$ROOT_DIR/data/data10/spike_seed16_300k.npy" \
      "$ROOT_DIR/data/data10/spike_seed30_300k.npy" \
      "$ROOT_DIR/data/data10/spike_seed37_300k.npy" \
      "$ROOT_DIR/data/data10/spike_seed73_300k.npy" \
      "$ROOT_DIR/data/data10/spike_seed131_300k.npy" \
      "$ROOT_DIR/data/data10/spike_seed133_300k.npy" \
      "$ROOT_DIR/data/data10/spike_seed134_300k.npy" \
    --val_list "$ROOT_DIR/data/data10/spike_seed142_300k.npy" \
    --test_list "$ROOT_DIR/data/data10/spike_seed143_300k.npy" \
    --totalsteps 300000 \
    --skipsteps 0 \
    --seq_len 400 \
    --seq_stride 200 \
    --history 200 \
    --conv_filter_size "$conv_filter" \
    --local_conv_stride "$local_stride" \
    --conv2_time_kernel "$conv2_kernel" \
    --conv_stride 1 \
    --embed_mode none \
    --posemb_dim 0 \
    --edge_chunk_num 20 \
    --out_channel 32 \
    --n_hid "$n_hid" \
    --gru_hidden "$gru_hidden" \
    --gru_layers 1 \
    --batch_size "$batch_size" \
    --lr "$lr" \
    --max_epoch 10 \
    2>&1 | tee -a "$log_file"

  local exit_code=${PIPESTATUS[0]}
  printf '%s\n' "$exit_code" > "$STATUS_DIR/${id}.status"
  printf '%s\n' "END $exp_name exit_code=$exit_code" | tee -a "$log_file"
}

worker_a() {
  run_one "0,1,2,3" e01 20 10 19 32 32 8 5e-4
  run_one "0,1,2,3" e03 40 20 9 32 32 8 5e-4
  run_one "0,1,2,3" e05 20 10 19 16 32 8 5e-4
  run_one "0,1,2,3" e07 20 10 19 32 64 8 1e-4
}

worker_b() {
  run_one "4,5,6,7" e02 40 10 17 32 32 8 5e-4
  run_one "4,5,6,7" e04 80 40 4 32 32 8 5e-4
  run_one "4,5,6,7" e06 20 10 19 64 64 4 5e-4
  run_one "4,5,6,7" e08 20 10 19 32 32 16 1e-3
}

worker_a &
pid_a=$!
worker_b &
pid_b=$!

wait "$pid_a"
status_a=$?
wait "$pid_b"
status_b=$?

printf '%s\n' "workers_finished status_a=$status_a status_b=$status_b"
