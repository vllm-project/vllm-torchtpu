#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by k3_1p1d.sh
# Kimi-K3 on a 1P1D pair serving the InferenceX AgentX trace replay.
#
# kimi-k3-tp8-dp4-ep-1p1d.sh is the base and almost all of it is inherited: the
# TP8 x DP4 x EP32 geometry, fp8 KV, prefix caching, the 1296-token block and
# the whole Raiden transfer environment are what make the pair work and do not
# change for a benchmark. What this file overrides is the WORKLOAD:
#
#   MAX_MODEL_LEN           10240 -> empty, i.e. the checkpoint's own 1M window
#   GPU_MEMORY_UTILIZATION  0.80  -> 0.90, which that window needs to boot
#   compile_sizes           adds 1,2,4 so the buckets bracket the ladder
#   serve flags             adds the tool-call parser the traces need
#   AGENTX_*                the cases and the row labels, which the base has no use for
#
# Nothing else differs. k3_agentx_client_pod.sh reads AGENTX_CONC_LIST from
# here, as agentx_sweep.sh does for the aggregate lane, so the cases live in one
# file.
# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/kimi-k3-tp8-dp4-ep-1p1d.sh"

# aiperf FILTERS whole traces above the window rather than truncating, so a
# smaller number swaps the corpus for the subset that fits (at 131072, 82 of
# 393 traces). It may not boot - _check_enough_kv_cache_memory is a hard
# ValueError (#1578) - and that failure is the result worth having.
MAX_MODEL_LEN=""

# The base's 0.80 is what #1271 was validated with, and it left 7.54 GiB of KV
# per rank against the 15.13 GiB one 1M request needs. The host transfer buffers
# are not carved out of this, so the extra 0.10 costs activation headroom.
GPU_MEMORY_UTILIZATION=0.90

# Two points: the pair holds 32 chips for the whole run and boots twice, so
# there is no wall clock for breadth. c1 and c4 were healthy on the aggregate.
AGENTX_CONC_LIST="1 4"

# The row labels, which the model-agnostic client takes as arguments.
AGENTX_MODEL_PREFIX=kimik3
AGENTX_PRECISION=fp4

# An assertion, not an override: the base is 8 too, but as a single-request
# probe's value, and it only happens to equal twice this ladder's top
# concurrency - what keeps the scheduler from being the limit. The client checks.
MAX_NUM_SEQS=8

# Without a 1, a single-sequence decode step runs the next bucket's shape. The
# prefill chunk stays the base's, tied to the pool's block geometry.
EXTRA_SERVE_ARGS=$(printf '%s' "$EXTRA_SERVE_ARGS" | sed "s/\"compile_sizes\":\[1,8,${MAX_NUM_BATCHED_TOKENS}\]/\"compile_sizes\":[1,2,4,8,${MAX_NUM_BATCHED_TOKENS}]/")
case "$EXTRA_SERVE_ARGS" in
    *'"compile_sizes":[1,2,4,8,'*) ;;
    *) echo "ERROR: kimi-k3-tp8-dp4-ep-agentx.sh: the base no longer sets" \
            "compile_sizes [1,8,\$MAX_NUM_BATCHED_TOKENS]; fix the substitution" >&2
       return 1 ;;
esac

# Every AgentX turn carries tool_calls and tool results; the base adds only the
# reasoning parser, its stages sending plain completions.
EXTRA_SERVE_ARGS="$EXTRA_SERVE_ARGS --enable-auto-tool-choice --tool-call-parser kimi_k3"
