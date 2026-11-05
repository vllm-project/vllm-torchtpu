#!/usr/bin/env bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -ex

echo "=== 1. Setting up HuggingFace token and editable install ==="
HF_TOKEN=$(cat /root/.cache/huggingface/token 2>/dev/null || echo "")
export HF_TOKEN
if [ "$HF_TOKEN" = "your_huggingface_token" ] || [ -z "$HF_TOKEN" ]; then
    unset HF_TOKEN
fi
export HF_HOME=/root/.cache/huggingface

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}"

echo "=== 1a. Syncing workspace src from $WORK_DIR ==="
mkdir -p /root/cloud-devkit/vllm-torchtpu/src /workspace/vllm-torchtpu/src
cp -a "$WORK_DIR/src/"* /root/cloud-devkit/vllm-torchtpu/src/ 2>/dev/null || true
cp -a "$WORK_DIR/src/"* /workspace/vllm-torchtpu/src/ 2>/dev/null || true
export PYTHONPATH="$WORK_DIR/src:/workspace/vllm-torchtpu/src:/root/cloud-devkit/vllm-torchtpu/src:${PYTHONPATH:-}"

pip install --no-deps -e "$WORK_DIR"

SITE_PACKAGES="/usr/local/lib/python3.12/site-packages"
if [ -d "$SITE_PACKAGES" ]; then
    cp -a "$WORK_DIR/src/sitecustomize.py" "$SITE_PACKAGES/sitecustomize.py" 2>/dev/null || true
fi

echo "=== 2. Checking TPU hardware ==="
tpu-info

if [ -d "/cdk-outputs" ]; then
    OUT_DIR="${CDK_OUTPUT_DIR:-/cdk-outputs}"
else
    OUT_DIR="${CDK_OUTPUT_DIR:-/tmp/cdk-outputs}"
fi
mkdir -p "$OUT_DIR/profile"

export VLLM_ENGINE_READY_TIMEOUT_S=3600

echo "=== 3. Starting vLLM Server (Gemma-4 with DP=4, TP=2, data mode) ==="
vllm serve RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --data-parallel-size 4 \
    --api-server-count 1 \
    --gpu-memory-utilization 0.90 \
    --async-scheduling \
    --attention-backend CUSTOM \
    --mm-encoder-tp-mode data \
    --mm-processor-kwargs '{"size":{"longest_edge":262144,"shortest_edge":3136}}' \
    --compilation-config '{"cudagraph_mm_encoder":true,"encoder_cudagraph_max_vision_items_per_batch":8,"encoder_cudagraph_max_frames_per_batch":8}' \
    --disable-chunked-mm-input \
    --max-model-len 4096 \
    --max-num-batched-tokens 16384 \
    --kv-cache-dtype fp8 \
    --max-num-seqs 512 \
    --block-size 256 \
    --hf-overrides '{"allow_global_per_layer_attribute_access": true, "text_config": {"allow_global_per_layer_attribute_access": true}}' \
    --profiler-config.profiler torch \
    --profiler-config.torch_profiler_dir "$OUT_DIR/profile" \
    --profiler-config.max_iterations 15 \
    --profiler-config.ignore_frontend true \
    2>&1 | tee "$OUT_DIR/vllm_server.log" &
SERVER_PID=$!

echo "Waiting for vLLM server to be ready on port 8000..."
until curl -sf http://localhost:8000/health > /dev/null; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Server exited unexpectedly. Checking last 50 lines of log:"
        tail -n 50 "$OUT_DIR/vllm_server.log"
        exit 1
    fi
    echo "Server initializing / compiling TPU graphs..."
    sleep 15
done
echo "Server is ready! Running benchmark..."

echo "=== 4. Running Client Benchmark (Kewei's 512x512 random-mm) ==="
vllm bench serve \
    --backend openai-chat \
    --model RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic \
    --dataset-name random-mm \
    --num-prompts 320 \
    --num-warmups 16 \
    --endpoint /v1/chat/completions \
    --base-url http://localhost:8000 \
    --random-mm-bucket-config '{(512, 512, 1): 1.0}' \
    --random-mm-limit-mm-per-prompt '{"image": 1}' \
    --random-mm-base-items-per-request 1 \
    --random-input-len 1024 \
    --random-output-len 500 \
    --temperature 0 \
    --ignore-eos \
    --profile \
    --save-result \
    --result-dir "$OUT_DIR" \
    --result-filename "benchmark_results.json" \
    2>&1 | tee "$OUT_DIR/benchmark_client.log"

CLIENT_EXIT=${PIPESTATUS[0]}

echo "=== 5. Shutting down server ==="
kill -TERM "$SERVER_PID" || true
sleep 10
kill -9 "$SERVER_PID" 2>/dev/null || true

if [ "$OUT_DIR" != "/cdk-outputs" ] && [ -d "/cdk-outputs" ]; then
    cp -rf "$OUT_DIR"/* /cdk-outputs/ 2>/dev/null || true
fi

echo "=== 6. Benchmark finished with exit code $CLIENT_EXIT ==="
exit "$CLIENT_EXIT"

