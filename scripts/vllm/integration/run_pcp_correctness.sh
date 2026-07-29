#!/usr/bin/env bash
# Correctness test for PCP (FA PC8, GDN TP8).
#
# Boots the vLLM server with Qwen3.5-35B-A3B-FP8 in the FA PCP8 + GDN TP8
# configuration and greedily continues a prompt of
# 32768 identical 'x' tokens; the only correct continuation is 8 more 'x'
# tokens. The prompt is long enough to engage the PCP streaming sequence
# layout, where GDN prefill must exchange token shards for head shards;
# without that exchange generation derails from the second token.
set -u

MODEL="Qwen/Qwen3.5-35B-A3B-FP8"
PORT=8000
EXPECTED="87 87 87 87 87 87 87 87"
STARTUP_TIMEOUT_S=1800
export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1

vllm serve "$MODEL" \
  --max-model-len 65536 \
  --kv-cache-dtype fp8 \
  --language-model-only \
  --enable-expert-parallel \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 2 \
  --compilation-config '{"compile_sizes":[4096]}' \
  --block-size 4096 \
  --attention-backend CUSTOM \
  --mamba-cache-mode align \
  --prefill-context-parallel-size 8 \
  --cp-kv-cache-interleave-size 256 &
SERVER_PID=$!
trap 'kill -TERM "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null' EXIT

deadline=$((SECONDS + STARTUP_TIMEOUT_S))
until curl -sf "localhost:$PORT/health" >/dev/null; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "PCP_CORRECTNESS_FAIL: server exited during startup"
    exit 1
  fi
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "PCP_CORRECTNESS_FAIL: server did not become healthy in time"
    exit 1
  fi
  sleep 10
done

req=$(mktemp)
python3 - "$MODEL" >"$req" <<'PY'
import json
import sys

print(json.dumps({
    "model": sys.argv[1],
    "prompt": [87] * 32768,  # 'x' * 32768
    "max_tokens": 8,
    "temperature": 0,
    "return_token_ids": True,
}))
PY

tokens=$(curl -sf -m 600 -H 'Content-Type: application/json' \
    -d @"$req" "localhost:$PORT/v1/completions" |
  python3 -c 'import json, sys
print(" ".join(str(t) for t in json.load(sys.stdin)["choices"][0]["token_ids"]))')
rm -f "$req"

echo "greedy continuation: [$tokens]"
if [ "$tokens" = "$EXPECTED" ]; then
  echo "PCP_CORRECTNESS_OK"
  exit 0
fi
echo "PCP_CORRECTNESS_FAIL: got [$tokens] expected [$EXPECTED]"
exit 1
