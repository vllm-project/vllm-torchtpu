#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# DeepSeek V4.1 Flash 100k decode with DSpark K=5 speculative decoding.
# Same workload as deepseek-v41-flash-tp8-ep-100k-decode; a verify
# step carries C * (1 + K) tokens, so add 6x buckets for C=4/8/16.
CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$CONFIG_DIR/deepseek-v41-flash-tp8-ep-100k-decode.sh"
DSPARK_K="${DSPARK_K:-5}"
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS//\[4,8,16,2048\]/[4,8,16,24,48,96,2048]}"
# With MODEL_URI the target's model path becomes a local config-only cache
# dir; the drafter inherits it and runai_streamer finds no safetensors there,
# so point the draft at the object-storage URI explicitly.
DSPARK_MODEL="${MODEL_URI:+,\"model\":\"${MODEL_URI}\"}"
EXTRA_SERVE_ARGS="$EXTRA_SERVE_ARGS --speculative-config={\"method\":\"dspark\",\"num_speculative_tokens\":${DSPARK_K}${DSPARK_MODEL}}"
