# Parked: external cache reset + shm restart recovery

Both features were removed from `benliangw/raiden_controller` because the
global-prefix-cache + disagg/offloading path does not use them. The complete
pre-removal state is preserved on branch `benliangw/backup` in each repo
(vllm-torchtpu and tpu-raiden), and the shm allocator fix additionally lives
on tpu-raiden `benliangw/cl-shm`.

## What was removed

### vllm-torchtpu (this repo)

- `RaidenKVOffloadManager.reset_cache()` in `src/vllm_torchtpu/offload/raiden_store.py`
  (drain -> `store.clear()` -> advisory/quarantine reset). vLLM's
  `OffloadingManager.reset_cache` base hook is a no-op, so
  `POST /reset_prefix_cache?reset_external=true` now simply does not clear
  the external store.
- `derive_shm_model_uid()` and the spec's `RAIDEN_SHM_KEY` ->
  `RAIDEN_SHM_MODEL_UID` wiring. Replaced by a hard rejection: the spec
  raises when `RAIDEN_SHM_KEY` is set (see "Why shm is rejected" below).
- Tests: reset-cache tests, the shm-UID test class, and the legacy
  "reset raises without clear()" coverage.

### tpu-raiden (`benliangw/raiden_controller`)

- `KVCacheStore::Clear()` and the backend contract
  (`KVCacheStoreBackend::Clear` / `PreflightClear`,
  `HostOffloadBackend::{Clear,PreflightClear,CheckNothingPinned}`), the
  `clear` torch binding, and their tests (single-tier + multi-tier
  preflight semantics, clear-then-reuse on an exhausted pool).

## Why shm is rejected rather than just untested

On the current branch the shm allocator is upstream's single-segment
version: every `Allocate()` after the first returns the same base pointer,
so the per-layer/shard host mirrors all alias one segment, and under
multi-process TP every rank collides on the `_dev_0` key. Serving from a
`RAIDEN_SHM_KEY` store then returns full-looking external cache hits with
corrupt KV — the worst failure mode — so the offload spec fails
construction instead. Upstream's recovery e2e test allocates a single small
tensor and cannot reproduce this.

## To revive

1. tpu-raiden: cherry-pick the multi-segment shm allocator from
   `benliangw/cl-shm` ("Allocate one shm segment per host mirror with
   per-node keys") — also worth sending upstream — and restore `Clear()`
   from `benliangw/backup`; rebuild the wheel.
2. vllm-torchtpu: restore `reset_cache` / `derive_shm_model_uid` and their
   tests from `benliangw/backup`; drop the `RAIDEN_SHM_KEY` rejection.
3. Recipes: the retired reset and reset-recovery gates are in
   cloud-devkit-recipes git history on `benliangw/dev`.
