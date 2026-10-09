#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Nightly structured-decoding gate: Qwen3-0.6B TP=1, every request constrained
# by the flat JSON grammar in structured_schemas/flat_json.json (an array of
# at least 10 flat 5-key records).
# Sibling of qwen3-0.6b-structured-complex-ast.sh; --extra-body applies to
# every request of a run, so each grammar is its own config and baseline.
MODEL="Qwen/Qwen3-0.6B"
TENSOR_PARALLELISM=1
DATA_PARALLELISM=1
ENABLE_EP=false
# Short prompt, 128 generated tokens: decode-dominated, so the per-step mask
# cost is what moves the numbers.
ISL_OSL_CONFIGS="128:128"
CONCURRENCY_OPTIONS="1 8 32"
RANDOM_RANGE_RATIO="0.0"
NUM_PROMPTS=128
GPU_MEMORY_UTILIZATION=0.9
ATTENTION_BACKEND="CUSTOM"
BENCHMARK_TEMPERATURE=0
EVAL_TOLERANCE="0.02"
# Concurrency 32 swings run to run on this small model (throughput by 9%,
# TPOT by 12% across the two calibration runs, and a later job hit +18.5% on
# c32 median TPOT). Perf is not gated in CI yet, so use a loose 0.20 band.
PERF_TOLERANCE="0.20"

BENCHMARK_EXTRA_BODY="$(cat "$SCRIPT_DIR/structured_schemas/flat_json.json")"
# The random-token prompts don't ask for JSON, so with xgrammar's default
# free whitespace most requests emit "{" and then only newlines. Disabling it
# makes every step emit keys and values. It is a server setting (xgrammar
# ignores the per-request one), and vLLM rejects it unless the backend is
# pinned rather than "auto". The schema's minItems=10 array keeps the JSON
# open for all 128 tokens; --ignore-eos would otherwise decode an
# unconstrained tail after it closes.
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--structured-outputs-config {\"backend\":\"xgrammar\",\"disable_any_whitespace\":true}"
# Keep the generated text and per-request output lengths in each cell's
# result JSON, to check that the output stays inside the grammar.
EXTRA_BENCH_ARGS="${EXTRA_BENCH_ARGS:+$EXTRA_BENCH_ARGS }--save-detailed"
