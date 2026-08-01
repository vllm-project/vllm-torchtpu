#!/bin/bash

# shellcheck disable=all
set -e



wait_for_server() {
  local port=$1
  local pid=$2
  timeout 1200 bash -c "
    until curl -s localhost:${port}/health > /dev/null; do
      if ! kill -0 $pid 2>/dev/null; then
        echo \"Error: vLLM server on port $port (PID $pid) crashed or failed to start!\" >&2
        exit 1
      fi
      sleep 1
    done" && return 0 || return 1
}

# Function to print logs on exit
print_logs_on_exit() {
  echo "--- Script exiting, displaying logs ---"

  # The logs are written inside containers to /root/logs, which is mapped from $LOG_DIR on the host.
  LOG_DIR=$HOME/logs

  if [ -d "$LOG_DIR" ]; then
    echo "--- Contents of $LOG_DIR/prefill_0.txt ---"
    if [ -f "$LOG_DIR/prefill_0.txt" ]; then
      cat "$LOG_DIR/prefill_0.txt"
    else
      echo "File not found."
    fi

    echo "--- Contents of $LOG_DIR/decode_0.txt ---"
    if [ -f "$LOG_DIR/decode_0.txt" ]; then
      cat "$LOG_DIR/decode_0.txt"
    else
      echo "File not found."
    fi

    echo "--- Contents of $LOG_DIR/benchmark_0.txt ---"
    if [ -f "$LOG_DIR/benchmark_0.txt" ]; then
      cat "$LOG_DIR/benchmark_0.txt"
    else
      echo "File not found."
    fi
  else
    echo "Log directory '$LOG_DIR' not found."
  fi
  echo "--- End of logs ---"
}

check_failed_requests() {
  local log_file="$1"
  local failed_requests
  failed_requests=$(grep "Failed requests:" "$log_file" | awk '{print $3}' || true)

  if [ -z "$failed_requests" ]; then
    echo "Error: Could not find 'Failed requests:' in the benchmark output." >&2
    return 1
  fi

  if [ "$failed_requests" -gt 0 ]; then
    echo "Error: Benchmark reported $failed_requests failed requests." >&2
    return 1
  fi

  echo "Success: Benchmark reported $failed_requests failed requests." >&2
  return 0
}

cleanup_instances() {
  echo "Cleaning up any running vLLM instances..."
  pkill -f "vllm" || true
  pkill -f "toy_proxy_server" || true
  sleep 5
  pkill -9 -f "vllm" || true
  pkill -9 -f "toy_proxy_server" || true
  fuser -k -9 /dev/vfio/* || true
  fuser -k -9 /dev/accel* || true
  rm -rf /tmp/jax_cache_* || true
  rm -f /tmp/libtpu_lockfile || true
}

# Register the cleanup function to be called on script exit (normal or error)
trap print_logs_on_exit EXIT

MODEL=${MODEL:="Qwen/Qwen3-0.6B"}
INPUT_LEN=${INPUT_LEN:=512}
OUTPUT_LEN=${OUTPUT_LEN:=128}
NUM_PROMPTS=${NUM_PROMPTS:=200}
REQUEST_RATE=${REQUEST_RATE:=4}

NUM_PREFILL_INSTANCES=1
NUM_DECODE_INSTANCES=1
PREFILLER_TP_SIZE=1
DECODER_TP_SIZE=1

PREFILL_HOSTS=()
PREFILL_PORTS=()
DECODE_HOSTS=()
DECODE_PORTS=()

# Retrieve per-chip vfio device paths (e.g. /dev/vfio/1) from `tpu-info`.
TPU_DEVICE_PATHS=($(tpu-info 2>/dev/null | grep -o '/dev/vfio/[0-9]\+'))

LOG_DIR=$HOME/logs

if [ ! -d $LOG_DIR ]; then
  mkdir -p $LOG_DIR
else
  # Delete old log files to avoid printing stale logs at the end
  rm -f $LOG_DIR/prefill_0.txt $LOG_DIR/decode_0.txt $LOG_DIR/benchmark_0.txt $LOG_DIR/proxy_0.txt
fi

cleanup_instances

# Start prefill instances
for i in $(seq 0 $((NUM_PREFILL_INSTANCES-1))); do
    PORT=$((8400 + i))
    KV_PORT=$((7100 + i))
    SIDE_PORT=$((6100 + i))
    CHIP_IDX=$i

    echo TPU_VISIBLE_DEVICE_PATHS will be set to ${TPU_DEVICE_PATHS[$CHIP_IDX]}

    TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 \
    TPU_PROCESS_BOUNDS=1,1,1 \
    TPU_VISIBLE_DEVICE_PATHS=${TPU_DEVICE_PATHS[$CHIP_IDX]} \
    \
    TPU_KV_TRANSFER_PORT=$KV_PORT \
    TPU_SIDE_CHANNEL_PORT=$SIDE_PORT \
    SKIP_JAX_PRECOMPILE=1 \
    \
    vllm serve $MODEL \
    --port $PORT \
    --gpu-memory-utilization 0.2 \
    --tensor-parallel-size $PREFILLER_TP_SIZE \
    --attention-backend CUSTOM \
    --kv-transfer-config "{\"kv_connector\":\"TPUConnector\",\"kv_connector_module_path\":\"vllm_torchtpu.distributed.kv_transfer.tpu_connector\",\"kv_role\":\"kv_producer\"}" \
    > $LOG_DIR/prefill_$i.txt 2>&1 &

    PREFILL_HOSTS+=("localhost")
    PREFILL_PORTS+=($PORT)
    PREFILL_PIDS+=($!)
done


# Start decode instances
for i in $(seq 0 $((NUM_DECODE_INSTANCES-1))); do
    PORT=$((9400 + i))
    KV_PORT=$((7200 + i))
    # Same as prefill SIDE_PORT
    SIDE_PORT=$((6100 + i))
    CHIP_IDX=$((NUM_PREFILL_INSTANCES + i))
    echo TPU_VISIBLE_DEVICE_PATHS will be set to ${TPU_DEVICE_PATHS[$CHIP_IDX]}


    TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 \
    TPU_PROCESS_BOUNDS=1,1,1 \
    TPU_VISIBLE_DEVICE_PATHS=${TPU_DEVICE_PATHS[$CHIP_IDX]} \
    \
    TPU_KV_TRANSFER_PORT=$KV_PORT \
    TPU_SIDE_CHANNEL_PORT=$SIDE_PORT \
    SKIP_JAX_PRECOMPILE=1 \
    \
    vllm serve $MODEL \
    --port $PORT \
    --gpu-memory-utilization 0.2 \
    --tensor-parallel-size $DECODER_TP_SIZE \
    --attention-backend CUSTOM \
    --kv-transfer-config "{\"kv_connector\":\"TPUConnector\",\"kv_connector_module_path\":\"vllm_torchtpu.distributed.kv_transfer.tpu_connector\",\"kv_role\":\"kv_consumer\"}" \
    > $LOG_DIR/decode_$i.txt 2>&1 &

    DECODE_HOSTS+=("localhost")
    DECODE_PORTS+=($PORT)
    DECODE_PIDS+=($!)
done

# Wait for all instances to start
# Wait for all instances to start
for i in "${!PREFILL_PORTS[@]}"; do
    PORT=${PREFILL_PORTS[$i]}
    echo "Waiting for prefill on port $PORT to start..."
    wait_for_server $PORT ${PREFILL_PIDS[$i]}
done

for i in "${!DECODE_PORTS[@]}"; do
    PORT=${DECODE_PORTS[$i]}
    echo "Waiting for decode on port $PORT to start..."
    wait_for_server $PORT ${DECODE_PIDS[$i]}
done

echo "starting proxy server"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
# Start proxy server
python $SCRIPT_DIR/toy_proxy_server.py \
--host localhost \
--port 8000 \
--prefiller-hosts ${PREFILL_HOSTS[@]} \
--prefiller-ports ${PREFILL_PORTS[@]} \
--decoder-hosts ${DECODE_HOSTS[@]} \
--decoder-ports ${DECODE_PORTS[@]} \
> $LOG_DIR/proxy_0.txt 2>&1 &

# run benchmark for both disagg and non-disagg
LOG_FILE="$LOG_DIR/benchmark_0.txt"
echo "--- Running Disagg Benchmark ---" > $LOG_FILE

# run ben for disagg
set -x
vllm bench serve \
  --model=$MODEL \
  --num-warmups=3 \
  --dataset-name=random \
  --random-input-len=${INPUT_LEN} \
  --random-output-len=${OUTPUT_LEN} \
  --num-prompts=${NUM_PROMPTS} \
  --ignore-eos \
  --host=localhost \
  --port 8000 \
  --request-rate=${REQUEST_RATE} \
  >> $LOG_FILE 2>&1
set +x

check_failed_requests "$LOG_FILE"

cat <<'EOF'
The proxy server has been launched on: 127.0.0.1:7080

>> Send example request:

curl -X POST \
http://127.0.0.1:7080/v1/completions \
-H "Content-Type: application/json" \
-d '{"prompt": "We hold these truths to be self-evident, that all men are created equal, that they are endowed by their Creator with certain unalienable Rights, that among these are Life, Liberty and the pursuit of Happiness.--That to secure these rights, Governments are instituted among Men, deriving their just powers from the consent of the governed,  ", "max_tokens": 10}'

>> Stop the proxy server and all prefill/decode instances:

pkill -f "vllm serve" && pkill -f "toy_proxy_server" && pkill -f "run_disagg_single_host"
EOF
