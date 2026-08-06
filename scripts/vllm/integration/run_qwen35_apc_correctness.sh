#!/bin/bash
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
#
# Single-instance prefix caching on the unified KV pool: a cache hit must
# restore the Mamba state, not just the attention KV. Covers the plain serving
# path, which the disagg job does not.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-35B-A3B-FP8}"
PORT="${PORT:-8000}"
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-2700}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# max_num_seqs=1 removes batching nondeterminism, so a cold/warm difference is
# the cache and nothing else. mamba_cache_mode resolves to align on its own.
vllm serve "$MODEL" \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --max-model-len 16384 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 1 \
  --port "$PORT" \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --quantization fp8 \
  --attention-backend CUSTOM \
  --block-size 256 \
  --language-model-only \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --default-chat-template-kwargs '{"enable_thinking":false}' &
SERVER_PID=$!
trap 'kill -TERM "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null' EXIT

deadline=$((SECONDS + STARTUP_TIMEOUT_S))
until curl -sf "localhost:$PORT/health" >/dev/null; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "APC_CORRECTNESS_FAIL: server exited during startup"
    exit 1
  fi
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "APC_CORRECTNESS_FAIL: server did not become healthy in time"
    exit 1
  fi
  sleep 10
done

python3 "$HERE/smoke_prefix_cache_e2e_divergence.py" \
  --host localhost --port "$PORT" --model "$MODEL"
python3 "$HERE/smoke_prefix_cache_correctness.py" \
  --host localhost --port "$PORT" --model "$MODEL"
# The one above repeat a single prompt, so a resumed request gets back a slot
# holding that same prompt's state and a missing restore stays invisible. This
# interleaves two prefixes so the slot always carries the wrong one.
python3 "$HERE/smoke_mamba_state_contamination.py" \
  --host localhost --port "$PORT" --model "$MODEL"

# A run that never hit the cache would pass both smokes vacuously.
hits=$(curl -sf "localhost:$PORT/metrics" |
  awk '/^vllm:prefix_cache_hits_total/ {s += $2} END {printf "%d", s}')
echo "prefix_cache_hits_total=$hits"
if [ "$hits" -le 0 ]; then
  echo "APC_CORRECTNESS_FAIL: prefix cache never hit"
  exit 1
fi

echo "APC_CORRECTNESS_OK"
