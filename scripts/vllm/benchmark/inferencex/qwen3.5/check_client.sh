#!/usr/bin/env bash
# Select and verify the client without changing the caller's shell options.
prepare_inferencex_client() {
  export INFERENCEX_COMMIT="${INFERENCEX_COMMIT:-c8b8d5a30bdc415b6845466c5122f0a60324e161}"
  if [[ ! "$INFERENCEX_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: INFERENCEX_COMMIT must be a full 40-character commit SHA" >&2
    return 1
  fi
  if [ -z "${INFERENCEX_REPO:-}" ]; then
    INFERENCEX_REPO=$(mktemp -d "${TMPDIR:-/tmp}/qwen3.5-inferencex_XXXXXX") || return 1
    echo "--- fetching InferenceX $INFERENCEX_COMMIT into $INFERENCEX_REPO ---"
    git init -q "$INFERENCEX_REPO" || return 1
    git -C "$INFERENCEX_REPO" fetch -q --depth 1 https://github.com/SemiAnalysisAI/InferenceX.git "$INFERENCEX_COMMIT" || return 1
    git -C "$INFERENCEX_REPO" checkout -q --detach FETCH_HEAD || return 1
  fi
  export INFERENCEX_REPO
  local actual changes
  actual=$(git -C "$INFERENCEX_REPO" rev-parse HEAD) || return 1
  if [ "$actual" != "$INFERENCEX_COMMIT" ]; then
    echo "ERROR: InferenceX is at $actual; expected $INFERENCEX_COMMIT" >&2
    return 1
  fi
  changes=$(git -C "$INFERENCEX_REPO" status --porcelain --untracked-files=normal) || return 1
  if [ -n "$changes" ]; then
    echo "ERROR: InferenceX has local changes; use a clean client checkout" >&2
    return 1
  fi
  if [ ! -f "$INFERENCEX_REPO/utils/bench_serving/benchmark_serving.py" ]; then
    echo "ERROR: selected InferenceX checkout has no utils/bench_serving/benchmark_serving.py" >&2
    return 1
  fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then prepare_inferencex_client; fi
