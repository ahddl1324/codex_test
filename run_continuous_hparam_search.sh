#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASELINE_VAL="0.17451598026355108"
RUN_DIR="$ROOT_DIR/continuous_sweep_10epoch"
LOG_DIR="$RUN_DIR/logs"
STATUS_DIR="$RUN_DIR/status"
RESULT_FILE="$ROOT_DIR/experiment_results.tsv"
MAIN="$ROOT_DIR/main_newSTDP_multitrial_recurrent_concat_mlp_2conv_chunk_sp_chunk.py"

mkdir -p "$LOG_DIR" "$STATUS_DIR"

already_recorded() {
  local id="$1"
  awk -F '\t' -v id="$id" '$1 == id { found=1 } END { exit !found }' "$RESULT_FILE"
}

record_result() {
  local id="$1" status="$2" best_val="$3" test_loss="$4"
  local cf="$5" cs="$6" ck="$7" oc="$8" nh="$9"
  local gh="${10}" lr="${11}" bs="${12}" changes="${13}"
  local delta="NA"

  if [[ "$best_val" != "NA" ]]; then
    delta="$(awk -v v="$best_val" -v b="$BASELINE_VAL" 'BEGIN { printf "%+.10f", v-b }')"
  fi

  (
    flock 9
    if ! already_recorded "$id"; then
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$id" "$status" "$best_val" "$delta" "$test_loss" \
        "$cf" "$cs" "$ck" "$oc" "$nh" "$gh" "$lr" "$bs" "$changes" >> "$RESULT_FILE"
    fi
  ) 9>"$RUN_DIR/results.lock"
}

run_one() {
  local gpu_group="$1" id="$2" cf="$3" cs="$4" ck="$5" oc="$6"
  local nh="$7" gh="$8" bs="$9" lr="${10}" changes="${11}"
  local exp_name="search10_${id}_cf${cf}_cs${cs}_ck${ck}_oc${oc}_nh${nh}_gh${gh}_b${bs}_lr${lr}"
  local log_file="$LOG_DIR/${id}.log"

  if already_recorded "$id"; then
    printf 'SKIP %s already recorded\n' "$id"
    return 0
  fi

  printf 'START %s GPUs=%s\n' "$exp_name" "$gpu_group" | tee "$log_file"

  CUDA_VISIBLE_DEVICES="$gpu_group" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  torchrun --standalone --nproc_per_node=4 "$MAIN" \
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
    --conv_filter_size "$cf" \
    --local_conv_stride "$cs" \
    --conv2_time_kernel "$ck" \
    --conv_stride 1 \
    --embed_mode none \
    --posemb_dim 0 \
    --edge_chunk_num 20 \
    --out_channel "$oc" \
    --n_hid "$nh" \
    --gru_hidden "$gh" \
    --gru_layers 1 \
    --batch_size "$bs" \
    --lr "$lr" \
    --max_epoch 10 \
    2>&1 | tee -a "$log_file"

  local exit_code=${PIPESTATUS[0]}
  printf '%s\n' "$exit_code" > "$STATUS_DIR/${id}.status"

  local best_val test_loss status
  best_val="$(awk '/Best val:/ { print $3; exit }' "$log_file")"
  test_loss="$(awk '/Best val:/ { print $6; exit }' "$log_file")"

  if [[ "$exit_code" -eq 0 && -n "$best_val" ]]; then
    status="ok"
    [[ -n "$test_loss" ]] || test_loss="NA"
  elif rg -q "OutOfMemoryError|CUDA out of memory" "$log_file"; then
    status="oom"
    best_val="NA"
    test_loss="NA"
  else
    status="failed"
    best_val="NA"
    test_loss="NA"
  fi

  record_result "$id" "$status" "$best_val" "$test_loss" \
    "$cf" "$cs" "$ck" "$oc" "$nh" "$gh" "$lr" "$bs" "$changes"
  printf 'END %s status=%s best_val=%s\n' "$exp_name" "$status" "$best_val" | tee -a "$log_file"
}

# id|conv_filter|local_stride|conv2_kernel|out_channel|n_hid|gru_hidden|batch|lr|changes
# Rows are launched in pairs on GPUs 0-3 and 4-7.
EXPERIMENTS=(
  'e09|20|10|19|32|64|32|4|5e-4|n_hid:32->64,batch_size:8->4'
  'e10|20|10|19|32|32|64|4|5e-4|gru_hidden:32->64,batch_size:8->4'
  'e11|20|10|19|32|64|64|8|5e-4|n_hid:32->64,gru_hidden:32->64'
  'e12|20|10|19|32|64|64|4|2.5e-4|n_hid:32->64,gru_hidden:32->64,batch_size:8->4,lr:5e-4->2.5e-4'
  'e13|20|10|19|32|64|64|4|7.5e-4|n_hid:32->64,gru_hidden:32->64,batch_size:8->4,lr:5e-4->7.5e-4'
  'e14|20|10|19|32|64|64|4|1e-3|n_hid:32->64,gru_hidden:32->64,batch_size:8->4,lr:5e-4->1e-3'
  'e15|20|10|19|16|64|64|4|5e-4|out_channel:32->16,n_hid:32->64,gru_hidden:32->64,batch_size:8->4'
  'e16|20|10|19|64|64|64|4|5e-4|out_channel:32->64,n_hid:32->64,gru_hidden:32->64,batch_size:8->4'
  'e17|20|10|19|32|96|64|4|5e-4|n_hid:32->96,gru_hidden:32->64,batch_size:8->4'
  'e18|20|10|19|32|64|96|4|5e-4|n_hid:32->64,gru_hidden:32->96,batch_size:8->4'
  'e19|40|10|17|32|64|64|4|5e-4|conv_filter_size:20->40,conv2_time_kernel:19->17,n_hid:32->64,gru_hidden:32->64,batch_size:8->4'
  'e20|40|20|9|32|64|64|4|5e-4|conv_filter_size:20->40,local_conv_stride:10->20,conv2_time_kernel:19->9,n_hid:32->64,gru_hidden:32->64,batch_size:8->4'
  'e21|80|40|4|32|64|64|4|5e-4|conv_filter_size:20->80,local_conv_stride:10->40,conv2_time_kernel:19->4,n_hid:32->64,gru_hidden:32->64,batch_size:8->4'
  'e22|20|20|10|32|64|64|4|5e-4|local_conv_stride:10->20,conv2_time_kernel:19->10,n_hid:32->64,gru_hidden:32->64,batch_size:8->4'
)

if (( $# > 0 )); then
  EXPERIMENTS=("$@")
fi

for ((i=0; i<${#EXPERIMENTS[@]}; i+=2)); do
  IFS='|' read -r id1 cf1 cs1 ck1 oc1 nh1 gh1 bs1 lr1 changes1 <<< "${EXPERIMENTS[$i]}"
  run_one '0,1,2,3' "$id1" "$cf1" "$cs1" "$ck1" "$oc1" "$nh1" "$gh1" "$bs1" "$lr1" "$changes1" &
  pid1=$!

  pid2=""
  if (( i + 1 < ${#EXPERIMENTS[@]} )); then
    IFS='|' read -r id2 cf2 cs2 ck2 oc2 nh2 gh2 bs2 lr2 changes2 <<< "${EXPERIMENTS[$((i+1))]}"
    run_one '4,5,6,7' "$id2" "$cf2" "$cs2" "$ck2" "$oc2" "$nh2" "$gh2" "$bs2" "$lr2" "$changes2" &
    pid2=$!
  fi

  wait "$pid1" || true
  if [[ -n "$pid2" ]]; then
    wait "$pid2" || true
  fi
done

printf 'PLANNED EXPERIMENT QUEUE COMPLETE\n'
