#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Kimi-K3 nightly serving config with a larger batch, for the
# eval-concurrency sweep (issue #841): same model, sharding, KV dtype, block
# size and serve flags as kimi-k3-tp32-ep, but max-num-seqs high enough that
# lm_eval's num_concurrent up to 256 is really served concurrently.
# Knobs (build env): K3_SWEEP_CTX (default 16384), K3_SWEEP_MAX_NUM_SEQS (256).
# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/kimi-k3-tp32-ep.sh"

MAX_MODEL_LEN="${K3_SWEEP_CTX:-16384}"
MAX_NUM_SEQS="${K3_SWEEP_MAX_NUM_SEQS:-256}"
# The bench pass that run_benchmarks.sh runs before handing over the live server
# is a smoke check here, not the nightly's 64-request latency probe.
NUM_PROMPTS=2

# Decode steps carry up to MAX_NUM_SEQS tokens; the nightly's [1,8,8192] buckets
# would pad every such step to 8192 tokens. Give the sweep matching buckets.
_before="$EXTRA_SERVE_ARGS"
EXTRA_SERVE_ARGS=$(printf '%s' "$EXTRA_SERVE_ARGS" | sed 's/"compile_sizes":\[1,8,8192\]/"compile_sizes":[1,8,16,32,64,128,256,1024,2048,4096,8192]/')
if [ "$EXTRA_SERVE_ARGS" = "$_before" ]; then
    echo "ERROR: kimi-k3-tp32-ep-concsweep.sh: the nightly config no longer sets compile_sizes [1,8,8192]; update the substitution above" >&2
    return 1
fi
