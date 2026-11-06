# Precompile Peer Replay and Subgroup-Aware Fingerprint Canonicalization on `partner/release-0.0.1`

## Executive Summary

On a single-host 8-chip TPU v7x node (`2x2x1` topology, 8 worker processes) serving `Qwen/Qwen3.5-397B-A17B-FP8` Decode (`DP=4, TP=2, EP=8`, `MTP=3` Eagle3 speculative decoding, `1M` context length, `TPURaidenConnector` `kv_consumer`, and 6 token shape buckets `[32, 64, 96, 128, 160, 192]`), cold-start precompilation on `partner/release-0.0.1` (`3e4d301eb`) exhibits a `+26.15 GiB` host DRAM (`Pss_Anon`) and `+20.97 GiB` `tcmalloc` active-allocation gap compared to warm-start cache loading:

- Without precompile peer replay, every rank on the host traces and lowers all 12 `(subgraph, bucket)` pairs (`6` backbone buckets + `6` Eagle3 drafter buckets = `96` Python/Dynamo/FX/AOTAutograd/Pallas/MLIR compilations across the 8 workers), even when `TPU_PARALLEL_PRECOMPILE=1` shares compiled PJRT executables through the `/dev/shm` Tier-2 cache after Wave 1.
- Furthermore, when `TPU_PRECOMPILE_PEER_REPLAY=1` (`#1810`) is enabled without subgroup-aware process-group canonicalization (`#2647`), PyTorch c10d and vLLM `GroupCoordinator` allocate distinct rank-local `group_name` string literals per DP/TP subgroup (`'1'`, `'3'`, `'5'`, `'7'` for the 4 TP=2 groups; `'13'`, `'15'` for the 2 DP=4 groups). Because `_graph_fingerprint` hashed raw `module.code`, all `84 / 84` cross-subgroup follower replay attempts failed with `Ignoring artifact ... built from a different graph`, falling back to `96` full Python frontend compilations.
- We backported the four peer-replay commits (`#1520`, `#1810`, `#2452`, and `#2647`) onto `partner/release-0.0.1` (`fix/peer-replay-c10d-fingerprint-rel001`) and validated the branch end-to-end on an 8-chip TPU v7x node:
  - **3-Pass E2E Cold-vs-Warm Host DRAM & Compilation Benchmark (`Qwen/Qwen3.5-397B-A17B-FP8` Decode, real `378.3 GiB` FP8 weights, `DP=4, TP=2, EP=8`, `MTP=3`, `1M` context, `TPURaidenConnector` `kv_consumer`, 6 buckets `[32..192]`)**:
    - **Pass 1 — Baseline Cold Start (`TPU_PRECOMPILE_BIGGEST_FIRST=0, TPU_PRECOMPILE_PEER_REPLAY=0`)**: `96` backend compiles, `96` saved artifacts, `0` peer replays; `compile_or_warm_up_model()` took **`521.4 s`** (`739.0 s` to `/health`); 8-worker `post_ready` `Pss_Anon` reached **`181.77 GiB`** (`VmRSS = 187.52 GiB`, `tcmalloc_allocated = 105.06 GiB`, `tcmalloc_heap_size = 121.06 GiB`), which is **`+26.15 GiB` `Pss_Anon` (`+20.97 GiB` `tcmalloc`)** above Warm Start.
    - **Pass 2 — Backport Peer-Replay Cold Start (`TPU_PRECOMPILE_BIGGEST_FIRST=1, TPU_PRECOMPILE_PEER_REPLAY=1`)**: `12` owner saves (`1` owner rank per bucket across `ranks 0..5`) and **`84 / 84` (`100%`) peer replays** (`0` fingerprint rejections); `compile_or_warm_up_model()` dropped to **`396.0 s`** (**`-125.4 s` / `-24.1%`** vs. Pass 1, and **`-668.7 s` / `-62.8%`** vs. unpatched `BIGGEST_FIRST=1` baseline's `1064.7 s`); 8-worker `post_ready` `Pss_Anon` dropped to **`161.61 GiB`** (**`-20.16 GiB`** vs. Pass 1), `VmRSS` to **`166.80 GiB`** (**`-20.72 GiB`**), `tcmalloc_allocated` to **`88.78 GiB`** (**`-16.28 GiB`**), and `tcmalloc_heap_size` to **`99.11 GiB`** (**`-21.95 GiB`**).
    - **Pass 3 — Warm Start Reference (`Reuse Disk Cache from Pass 2`)**: `0` backend compiles, `96` disk cache loads; `compile_or_warm_up_model()` took **`256.4 s`** (`427.0 s` to `/health`); 8-worker `post_ready` `Pss_Anon` settled at **`155.61 GiB`** (`VmRSS = 161.11 GiB`, `tcmalloc_allocated = 84.09 GiB`, `tcmalloc_heap_size = 92.17 GiB`). Enabling the backported peer replay reduced the Cold-vs-Warm host DRAM gap by **`77.1%` (`+26.15 GiB -> +5.99 GiB` `Pss_Anon`)** and **`77.6%` (`+20.97 GiB -> +4.69 GiB` `tcmalloc_allocated`)**, and eliminated the gap on pure follower ranks `rank 6..7` (`+3.04 / +2.99 GiB -> +0.06 / +0.03 GiB` `Pss_Anon`).
  - **8-Rank TPU Subgroup & Constant Collision Micro-Test Suite (`3 / 3` passed in `50.9 s`)**: Verified subgroup-bound replay on pure `4xTP2` graphs (`4` compiles + `4` replays, `0` device-count errors), world-tiled replay on `4xTP2 + EP8` graphs (`1` compile + `7` replays), and collision resistance on `65537`-element non-`nn.Parameter` `get_attr` constant tensors differing only at index `65536` (`2` compiles + `6` replays).
  - **End-to-End `vllm serve` & GSM8K Accuracy Evaluation (`Qwen/Qwen3.5-397B-A17B-FP8`, `DP=4, TP=2, EP=8`)**: Replayed `42 / 42` (`100%`) follower bucket compilations (`6` owner saves, `0` ignored artifacts) and achieved `0.9500 +- 0.0219` (`95.0%`) on 5-shot `gsm8k` (`100` samples, strict-match and flexible-extract), matching `main`.

---

## Motivation and Problem Statement

### Redundant Frontend Lowering Under `TPU_PARALLEL_PRECOMPILE=1`

On `partner/release-0.0.1` (`3e4d301eb`), `TPU_PARALLEL_PRECOMPILE=1` rotates the starting token bucket across the 8 local ranks so Wave 1 compiles 6 distinct buckets in parallel and populates `torch_tpu`'s `/dev/shm` Tier-2 compilation cache. However, the Tier-2 cache is keyed by the post-lowering StableHLO `CompilationCacheKey`. To compute that key in Waves 2 through 6, every follower rank still executes the entire Python and C++ frontend lowering pipeline for every bucket:

- `copy.deepcopy(graph)` and Dynamo/AOTAutograd decomposition (`TpuBackend.__call__`)
- FX graph passes (`StaticCompiler.__call__`)
- Pallas kernel lowering and JAX trace caches (`JaxCallable.__call__`)
- C++ MLIR module construction (`Traversal::BuildMlirModule`)

Across `2` compiled subgraphs (`backbone` and `eagle_head`) and `6` token shape buckets (`[32, 64, 96, 128, 160, 192]`), the 8 workers execute `8 * 12 = 96` full frontend compilations instead of `12` (`1` owner rank per bucket). When `TPUWorker.compile_or_warm_up_model()` subsequently calls `freeze_gc_heap()` (`gc.freeze()`), the Python/FX/Pallas/MLIR objects and C++ heap allocations created during those `84` redundant follower compilations are permanently retained in host DRAM (`+20.16 GiB` `Pss_Anon` and `+16.28 GiB` `tcmalloc_allocated` across the 8 workers) for the lifetime of the server.

### Why Naive Peer Replay (`#1810`) Fails on Multi-Subgroup Topologies (`DP > 1, TP > 1`)

Upstream PR `#1810` introduced `TPU_PRECOMPILE_PEER_REPLAY=1`, which assigns each shape bucket to a single owner rank on the host (`mine`) while the remaining 7 follower ranks (`others`) wait in `wait_for_peer_artifact` and load the owner's serialized artifact from vLLM's compilation cache (`<hash>/rank_<r>_<dp>/<prefix>/<key>`), hard-linking their own cache entry to the owner's file once `_graph_fingerprint(graph)` matches.

When tested on multi-subgroup topologies such as `DP=4, TP=2, EP=8`, we observed two critical failures and three subgroup/constant edge cases:

- **100% Fingerprint Mismatch Across DP/TP Subgroups**:
  - When vLLM initializes multi-subgroup process groups (4 TP=2 groups `[0, 1]`, `[2, 3]`, `[4, 5]`, `[6, 7]` and 2 DP=4 groups `[0, 2, 4, 6]`, `[1, 3, 5, 7]`), PyTorch c10d allocates sequential string handles per subgroup (`'1'`, `'3'`, `'5'`, `'7'` for TP; `'13'`, `'15'` for DP) and vLLM's `GroupCoordinator` allocates rank-local names (`'tp:0'`, `'tp:1'`, etc.).
  - Traced collective ops (`torch.ops._c10d_functional.all_reduce`, `reduce_scatter_tensor`, `all_gather_into_tensor`, `all_to_all_single`, `broadcast`, and `torch.ops.vllm.*`) embed those string literals directly into `module.code`:

    ```python
    # Rank 0..1 (TP subgroup 0):
    all_reduce = torch.ops._c10d_functional.all_reduce(x, 'sum', '1')
    # Rank 6..7 (TP subgroup 3):
    all_reduce = torch.ops._c10d_functional.all_reduce(x, 'sum', '7')
    ```

  - Because `_graph_fingerprint(graph)` hashed raw `module.code`, every subgroup produced a different SHA-256 digest and rejected all `84 / 84` peer artifacts with:

    ```text
    WARNING [TpuCompilerAdaptor] Ignoring artifact .../rank_0_0/backbone/artifact_compile_range_32_32_subgraph_0: built from a different graph (<owner_sha256> vs <follower_sha256>)
    ```

- **Unpicklable `entry` After `_deserialize_entry`**:
  - `_deserialize_entry(entry)` (`#2068`) sets `entry.__deepcopy__ = lambda _memo: entry` before calling `deserialize_bundled_cache_entry(entry)` so AOTAutograd reuses the loaded PJRT executable instead of loading a duplicate executable via `__reduce__`. Leaving that `<lambda>` on `entry.__dict__` caused subsequent `pickle.dumps(entry)` calls to fail with `PicklingError: Can't pickle <function _deserialize_entry.<locals>.<lambda>>`.
- **Subgroup-Bound `DeviceAssignment` on `TorchTPU 0.1.1` When No World-Sized Op Is Present**:
  - On `partner/release-0.0.1` (`torch_tpu 0.1.1`, `libtpu 0.0.47`), `ProcessGroupTpu::device_assignment()` returns only the subgroup's local devices (`size=2` for `TP=2`), and `Traversal::ResolveDeviceAssignment()` sizes the XLA `DeviceAssignment(1, num_partitions)` to the largest `op.device_assignment` in the FX graph.
  - When a graph contains a world-sized op (`size == world_size`, such as an `EP=8` `all_reduce`, `all_to_all_single`, or an 8-chip Pallas `torch.ops.tpu.custom_kernel` in `Qwen3.5-397B`), the executable is compiled for all 8 devices (`partition count: 8`) and tiles `TP=2` collectives globally across all 4 TP subgroups (`{{0,1},{2,3},{4,5},{6,7}}`).
  - Conversely, a synthetic micro-graph containing only `TP=2` collectives and no world-sized op compiles with `partition count: 2` bound to the compiling subgroup's 2 physical chips (e.g., devices `(0, 1)`). Replaying that 2-partition executable on Rank 2 (devices `(2, 3)`) fails at load time with:

    ```text
    RuntimeError: INVALID_ARGUMENT: Out of the 2plicas*2partitions devices used by this computation, local device count is 0 (total replica count: 1, partition count: 2)
    ```

- **Large Inline `get_attr` Constant Cutoff (`_FINGERPRINT_MAX_NUMEL = 65536`)**:
  - Hashing `get_attr` tensor contents only up to `65536` elements caused non-`nn.Parameter` inline constant tensors with `65537` elements differing at index `65536` to collide on the same fingerprint.
- **Single-Path Peer Scan (`paths[0]`) in `_replay_from_peer`**:
  - `_replay_from_peer` only checked `paths[0]` and returned `None` on mismatch instead of scanning all peer artifacts (`for path in self._peer_paths(key):`) and continuing to poll when a subgroup follower encounters an earlier subgroup's artifact before its own subgroup leader saves.

---

## What We Did: Root-Cause Analysis, Backported Commits, and Edge-Case Fixes

We structured `fix/peer-replay-c10d-fingerprint-rel001` cleanly on top of `origin/partner/release-0.0.1` (`3e4d301eba34d4325463fba95f1a82e7d4917c1e`) as four sequential code commits followed by this report commit:

| Commit | Upstream PR | Summary of Changes |
| :--- | :--- | :--- |
| `66573fdf7` | [`#1520`](https://github.com/vllm-project/vllm-torchtpu/pull/1520) | `fix(compilation): precompile the biggest range first (#1520)` — Adds `TPU_PRECOMPILE_BIGGEST_FIRST` in `envs.py`, `_TPU_COMPILE_ENV_IGNORED` in `tpu_compiler.py`, and biggest-first range sorting in `_patch_vllm_compile_all_ranges()`. |
| `5273c6e58` | [`#1810`](https://github.com/vllm-project/vllm-torchtpu/pull/1810) (+ [`#2068`](https://github.com/vllm-project/vllm-torchtpu/pull/2068)) | `Compilation time: replay a bucket another rank of the host compiled (#1810)` — Adds `TPU_PRECOMPILE_PEER_REPLAY`, snake-order bucket ownership in `_patch_vllm_compile_all_ranges()`, `TpuCompilationHandle.fingerprint`, `wait_for_peer_artifact`, `_graph_fingerprint`, `_deserialize_entry` (with `__deepcopy__` short-circuit from `#2068`), `_peer_paths`, `_replay_from_peer`, and `_link_artifact` while preserving `with fake_mode, open(path, "rb") as f:` in `_replay()`. |
| `f3d76fddc` | [`#2452`](https://github.com/vllm-project/vllm-torchtpu/pull/2452) | `[Core] Harden precompile peer replay waits and reject it without the compile cache (#2452)` — Adds `_validate_peer_replay_config()` in `tpu_platform.py`, `host_slot()` (`LOCAL_RANK - TPU_LOCAL_RANK_OFFSET`) in `_patch_vllm_compile_all_ranges()`, reduces `PEER_WAIT_S` to `600.0 s` and `_PEER_POLL_S` to `0.5 s`, and writes cache handles via PID-unique temporary files before `os.replace`. |
| `8b154f02b` | [`#2647`](https://github.com/vllm-project/vllm-torchtpu/pull/2647) | `[Compilation] Scope subgroup-bound peer replay, hash large constants, and scan all peer paths (#2647)` — Extracts graph fingerprinting into `src/vllm_torchtpu/compilation/graph_fingerprint.py` and process-group canonicalization into `src/vllm_torchtpu/distributed/utils.py` (with unit and multichip TPU tests in `tests/compilation/test_graph_fingerprint.py`), canonicalizes symmetric collective `group_name` arguments (`_SYMMETRIC_COLLECTIVE_SCHEMAS`, `GroupCoordinator.group_ranks` for wrap-around `tpu7x` subgroups, shared `canonical_by_raw` across `graph.named_modules()`, `"default" in target.overloads()`) in `_normalized_module_code`, scopes subgroup-bound graphs on `TorchTPU < 0.2.0` via `_uses_global_subgroup_tiling`, hashes non-`nn.Parameter` `get_attr` constants in full, scans all peer paths with subgroup-follower polling in `_replay_from_peer`, and pops transient `__deepcopy__` in `_deserialize_entry`. |

### Key Technical Details of `#2647`

- **World-Tiled vs. Subgroup-Bound Process-Group Canonicalization (`_uses_global_subgroup_tiling` & `_canonicalize_process_group_name`)**:
  - `_supports_global_subgroup_tiling()` checks whether `torch_tpu._internal.distributed.tpu_distributed` exposes `global_device_attributes` (`True` on `TorchTPU >= 0.2.0`, `False` on `TorchTPU 0.1.1` before `3b08b43fd`).
  - When `_supports_global_subgroup_tiling()` is `False`, `_graph_has_world_sized_op(graph, world_size)` inspects all `fx.GraphModule` submodules for either (a) a collective whose resolved group size or `group_size` / `world_size` argument equals `world_size`, or (b) a `torch.ops.tpu.custom_kernel` Pallas node whose `device_assignment` length equals `world_size`.
  - When `global_tiling` is `True` (all full-model `Qwen3.5-397B` `DP=4, TP=2, EP=8` graphs, and any graph on `TorchTPU >= 0.2.0`), subgroup process-group names canonicalize to rank-invariant structural tokens: when a vLLM `GroupCoordinator` matches `group_name` (either by `unique_name` or `device_group.group_name`), its full `group_ranks` partition (`vllm_pg:{ordinal}:{role}:size={size}:group_ranks={group_ranks}`) is encoded so wrap-around `tpu7x` subgroups (`tp:0=(0, 1, 6, 7)` and `tp:1=(4, 5, 2, 3)`) match across subgroups; raw `c10d` groups fall back to `c10d_pg:{ordinal}:size={size}:strides={strides}`.
  - When `global_tiling` is `False` (pure subgroup graphs without any world-sized op on `TorchTPU 0.1.1`), the canonical token includes `ranks={ranks}`, scoping fingerprint matches to peers within the same physical subgroup (`[0, 1]`, `[2, 3]`, `[4, 5]`, `[6, 7]`).
  - Only symmetric collectives explicitly listed in `_SYMMETRIC_COLLECTIVE_SCHEMAS` are canonicalized, a single `canonical_by_raw` map is shared across `graph.named_modules()`, `_node_schema` checks `"default" in target.overloads()` on `OpOverloadPacket` targets, and any unresolvable `group_name` or non-symmetric/unlisted collective raises `ValueError` so `_graph_fingerprint` logs a warning and returns `None`.
- **Full Content Hashing for Non-Parameter `get_attr` Constant Tensors**:
  - In `_graph_fingerprint(graph)`, `not isinstance(value, torch.nn.Parameter) or value.numel() <= _FINGERPRINT_MAX_NUMEL` ensures that `nn.Parameter` model weights larger than `65536` elements continue to hash by `dtype` and `shape` only, while non-`nn.Parameter` inline constant tensors are always hashed by full byte content.
- **Multi-Path Peer Scan and Subgroup-Follower Polling in `_replay_from_peer`**:
  - `_replay_from_peer` iterates over all paths returned by `self._peer_paths(key)` (`tried: set[str]`) instead of stopping at `paths[0]`. When `_has_subgroup_follower_role(graph)` is `True` (the current rank is a follower `ranks[0] != current_rank` in a subgroup-bound graph), it continues polling until `deadline` if only another subgroup's artifact has been written so far.

---

## Experiment Environment and Reproduction Steps

### Hardware and Runtime Environment

| Category | Setting | Value |
| :--- | :--- | :--- |
| **Accelerator & Topology** | Single-Host TPU v7x | `1x tpu7x` (`2x2x1` topology, `8` TPU v7x chips / `16` TensorCores) |
| **Host Resources** | Pod Memory & Shared Memory | `850 GiB` host DRAM, `64 GiB` `/dev/shm`, `1000 GiB` ephemeral PD |
| **Model Checkpoint** | `Qwen/Qwen3.5-397B-A17B-FP8` | `378.3 GiB` FP8 safetensors checkpoint |
| **Parallelism & Spec Decode** | Decode Topology | `DP=4, TP=2, EP=True` (`8` worker processes), `MTP=3` (`Qwen3_5MoeMTP`) |
| **KV Cache & Disagg** | `TPURaidenConnector` (`kv_consumer`) | `--max-model-len 1048576`, `--kv-cache-dtype fp8`, `--mamba-ssm-cache-dtype bfloat16`, `--mamba-cache-mode align`, `--block-size 2304`, `--kv-cache-memory-bytes 42467328000`, `TPU_USE_RAIDEN_KV_CACHE_MANAGER=1`, `TPU_RAIDEN_POOL_STAGING_LEASES=1`, `TPU_PREMAPPED_BUFFER_SIZE=4294967296` |
| **Compilation Config** | Decode Token Buckets | `compile_sizes: [32, 64, 96, 128, 160, 192]`, `--max-num-batched-tokens 192`, `--max-num-seqs 48` |
| **Software Stack** | `partner/release-0.0.1` | `vllm 0.29.0`, `torch_tpu 0.1.1+release.2026.9.22.20261003154636`, `libtpu 0.0.47`, `tpu-sync-torch 0.0.1.dev20261013083150`, `jax 0.10.2`, `torch 2.13.0+cpu`, `libtcmalloc.so.4` (`TCMALLOC_RELEASE_RATE=10.0`) |

### Step-by-Step Reproduction Commands

#### CPU and Multichip TPU Unit Test Suites

Run the unit test suites for range rotation, compiler cache handling, and subgroup-aware peer-replay fingerprinting on CPU (and the `@pytest.mark.multichip` real-TPU compilation test on a 4+ chip TPU node):

```bash
pytest -v -m "not tpu_test and not multichip and not nightly" \
  tests/compilation/test_compile_all_ranges_patch.py \
  tests/compilation/test_graph_fingerprint.py \
  tests/compilation/test_tpu_compiler.py \
  -k "not test_runtime_hash_covers_custom_op_boundaries"

pytest -sv -m "not nightly and multichip" \
  tests/compilation/test_graph_fingerprint.py
```

#### 3-Pass End-to-End Cold-vs-Warm Host DRAM and Compilation Benchmark on 8-Chip TPU

Run all three passes sequentially on the same 8-chip TPU v7x node so hardware and OS state remain identical across passes:

1. **Common Server Launch Command (`Qwen/Qwen3.5-397B-A17B-FP8` Decode)**:

   ```bash
   export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4
   export TCMALLOC_RELEASE_RATE=10.0
   export TPU_EVICT_WEIGHTS_PAGE_CACHE=true
   export TPU_USE_RAIDEN_KV_CACHE_MANAGER=1
   export TPU_RAIDEN_POOL_STAGING_LEASES=1
   export TPU_PREMAPPED_BUFFER_SIZE=4294967296
   export TPU_PARALLEL_PRECOMPILE=1

   vllm serve /models/Qwen/Qwen3.5-397B-A17B-FP8 \
     --served-model-name Qwen/Qwen3.5-397B-A17B-FP8 \
     --port 8000 \
     --trust-remote-code \
     --data-parallel-size 4 \
     --tensor-parallel-size 2 \
     --enable-expert-parallel \
     --max-model-len 1048576 \
     --max-num-batched-tokens 192 \
     --max-num-seqs 48 \
     --gpu-memory-utilization 0.9 \
     --kv-cache-dtype fp8 \
     --mamba-ssm-cache-dtype bfloat16 \
     --mamba-cache-mode align \
     --block-size 2304 \
     --kv-cache-memory-bytes 42467328000 \
     --no-enable-prefix-caching \
     --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
     --kv-transfer-config '{"kv_connector":"TPURaidenConnector","kv_role":"kv_consumer"}' \
     --compilation-config '{"compile_sizes":[32,64,96,128,160,192]}'
   ```

2. **Pass 1 — Baseline Cold Start (`TPU_PRECOMPILE_BIGGEST_FIRST=0, TPU_PRECOMPILE_PEER_REPLAY=0`)**:
   - Clear `/tmp/compile_cache_p1` and `/dev/shm/torch_tpu_cache*`.
   - Launch `vllm serve` with `VLLM_CACHE_ROOT=/tmp/compile_cache_p1`, `TPU_PRECOMPILE_BIGGEST_FIRST=0`, and `TPU_PRECOMPILE_PEER_REPLAY=0`.
   - Record per-rank and 8-rank aggregate `/proc/<pid>/smaps_rollup` (`Pss_Anon`), `/proc/<pid>/status` (`RssAnon`, `VmRSS`), and `libtcmalloc` (`generic.current_allocated_bytes`, `generic.heap_size`, `tcmalloc.pageheap_free_bytes`) at `post_weight_load`, `post_kv_init`, `post_compile`, and `post_ready` (after `/health` and 8 concurrent `/v1/completions` requests).
3. **Pass 2 — Backport Peer-Replay Cold Start (`TPU_PRECOMPILE_BIGGEST_FIRST=1, TPU_PRECOMPILE_PEER_REPLAY=1`)**:
   - Terminate Pass 1, clear `/tmp/compile_cache_p2` and `/dev/shm/torch_tpu_cache*`.
   - Launch `vllm serve` with `VLLM_CACHE_ROOT=/tmp/compile_cache_p2`, `TPU_PRECOMPILE_BIGGEST_FIRST=1`, and `TPU_PRECOMPILE_PEER_REPLAY=1`.
   - Record all 4 lifecycle stage memory snapshots, compilation/replay counters (`Compiling FX graph for range`, `Saved compiled artifact`, `Replayed range`, `Ignoring artifact`), `compile_or_warm_up_model()` duration, and `/v1/completions` verification.
4. **Pass 3 — Warm Start Reference (`Reuse Disk Cache from Pass 2`)**:
   - Terminate Pass 2, clear `/dev/shm/torch_tpu_cache*` while keeping `VLLM_CACHE_ROOT=/tmp/compile_cache_p2` populated from Pass 2.
   - Launch `vllm serve` with `VLLM_CACHE_ROOT=/tmp/compile_cache_p2`, `TPU_PRECOMPILE_BIGGEST_FIRST=1`, and `TPU_PRECOMPILE_PEER_REPLAY=1`.
   - Record all 4 lifecycle stage memory snapshots, disk-cache hit counters, `compile_or_warm_up_model()` duration, and `/v1/completions` verification.

#### End-to-End `vllm serve` and GSM8K Accuracy Evaluation

```bash
export TPU_PARALLEL_PRECOMPILE=1
export TPU_PRECOMPILE_BIGGEST_FIRST=1
export TPU_PRECOMPILE_PEER_REPLAY=1
export VLLM_CACHE_ROOT=/tmp/compile_cache/vllm

vllm serve /models/Qwen/Qwen3.5-397B-A17B-FP8 \
  --served-model-name Qwen/Qwen3.5-397B-A17B-FP8 \
  --port 8000 \
  --trust-remote-code \
  --data-parallel-size 4 \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --max-model-len 10240 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 64 \
  --kv-cache-memory-bytes 34359738368 \
  --no-enable-prefix-caching \
  --compilation-config '{"compile_sizes":[32,64,128,256,512,1024]}'

lm_eval --model local-completions \
  --model_args model=Qwen/Qwen3.5-397B-A17B-FP8,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=32,max_retries=3,tokenized_requests=False \
  --tasks gsm8k \
  --num_fewshot 5 \
  --limit 100 \
  --batch_size auto
```

---

## Experiment Results and Observations

### 3-Pass End-to-End Cold-vs-Warm Host DRAM and Compilation Benchmark (`fix/peer-replay-c10d-fingerprint-rel001`)

#### High-Level Summary Across All 3 Passes (8 TPU Workers)

| Metric | Pass 1: Baseline Cold (`PEER_REPLAY=0`) | Pass 2: Backport Peer-Replay Cold (`PEER_REPLAY=1`) | Pass 3: Warm Start Reference (`Reuse Cache`) | Pass 2 vs. Pass 1 Savings | Cold-vs-Warm Gap (`Pass 1 -> Pass 2`) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Backend Compile Calls (`Compiling FX graph`)** | `96` | `96` | `0` | `0` | — |
| **Saved Compiled Artifacts (`TpuCompilerAdaptor`)** | `96` | **`12`** (`6` backbone + `6` Eagle3) | `0` | **`-84` (`-87.5%`)** | — |
| **Peer Replay Hits (`Replayed range`)** | `0` | **`84 / 84` (`100%`)** | `0` | **`+84`** | — |
| **Disk Cache Loads (`Loaded range from cache`)** | `0` | `0` | `96` | — | — |
| **Fingerprint Mismatch Rejections (`Ignoring artifact`)** | `0` | **`0`** | `0` | `0` | — |
| **`compile_or_warm_up_model()` Duration** | `521.4 s` | **`396.0 s`** | `256.4 s` | **`-125.4 s` (`-24.1%`)** | `+265.0 s -> +139.6 s` (`-47.3%`) |
| **Wall Time to `/health` Ready** | `739.0 s` | **`579.0 s`** | `427.0 s` | **`-160.0 s` (`-21.7%`)** | `+312.0 s -> +152.0 s` (`-51.3%`) |
| **8-Worker `Pss_Anon` (`post_ready`)** | `181.77 GiB` | **`161.61 GiB`** | `155.61 GiB` | **`-20.16 GiB`** | **`+26.15 GiB -> +5.99 GiB` (`-77.1%`)** |
| **8-Worker `RssAnon` (`post_ready`)** | `181.77 GiB` | **`161.61 GiB`** | `155.61 GiB` | **`-20.16 GiB`** | **`+26.15 GiB -> +5.99 GiB` (`-77.1%`)** |
| **8-Worker `VmRSS` (`post_ready`)** | `187.52 GiB` | **`166.80 GiB`** | `161.11 GiB` | **`-20.72 GiB`** | **`+26.41 GiB -> +5.69 GiB` (`-78.5%`)** |
| **8-Worker `tcmalloc` Active (`post_ready`)** | `105.06 GiB` | **`88.78 GiB`** | `84.09 GiB` | **`-16.28 GiB`** | **`+20.97 GiB -> +4.69 GiB` (`-77.6%`)** |
| **8-Worker `tcmalloc` Heap Size (`post_ready`)** | `121.06 GiB` | **`99.11 GiB`** | `92.17 GiB` | **`-21.95 GiB`** | **`+28.89 GiB -> +6.94 GiB` (`-76.0%`)** |
| **Online `MTP=3` `/v1/completions` Smoke Check** | `8 / 8 PASSED` | `8 / 8 PASSED` | `8 / 8 PASSED` | — | — |

#### 4-Stage Lifecycle Progression (`Pss_Anon` / `tcmalloc_allocated` Across All 8 Workers)

| Stage | Pass 1 Baseline Cold (`Pss_Anon` / `tcm`) | Pass 2 Peer-Replay Cold (`Pss_Anon` / `tcm`) | Pass 3 Warm Ref (`Pss_Anon` / `tcm`) | Pass 1 - Pass 3 Gap (`Pss_Anon` / `tcm`) | Pass 2 - Pass 3 Gap (`Pss_Anon` / `tcm`) | Pass 2 vs. Pass 1 Savings (`Pss_Anon` / `tcm`) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `post_weight_load` | `50.32 / 45.89 GiB` | `50.17 / 45.78 GiB` | `50.21 / 45.80 GiB` | `+0.11 / +0.09 GiB` | `-0.04 / -0.01 GiB` | `-0.15 / -0.10 GiB` |
| `post_kv_init` | `116.23 / 47.08 GiB` | `116.19 / 47.06 GiB` | `116.15 / 47.00 GiB` | `+0.08 / +0.08 GiB` | `+0.04 / +0.06 GiB` | `-0.04 / -0.02 GiB` |
| `post_compile` | `181.78 / 105.06 GiB` | `161.61 / 88.78 GiB` | `155.62 / 84.09 GiB` | **`+26.15 / +20.97 GiB`** | **`+5.99 / +4.69 GiB`** | **`-20.16 / -16.28 GiB`** |
| `post_ready` | **`181.77 / 105.06 GiB`** | **`161.61 / 88.78 GiB`** | **`155.61 / 84.09 GiB`** | **`+26.15 / +20.97 GiB`** | **`+5.99 / +4.69 GiB` (`-77.1%` / `-77.6%`)** | **`-20.16 / -16.28 GiB`** |

#### Per-Worker `post_ready` Breakdown (`Pss_Anon` / `tcmalloc_allocated`)

| Worker (`env_rank`) | `DP` | `TP` | Buckets Compiled in Pass 2 (`Backbone + Eagle3`) | Pass 1 Baseline Cold (`Pss_Anon` / `tcm`) | Pass 2 Peer-Replay Cold (`Pss_Anon` / `tcm`) | Pass 3 Warm Ref (`Pss_Anon` / `tcm`) | Pass 1 - Warm (`Pss_Anon` / `tcm`) | Pass 2 - Warm (`Pss_Anon` / `tcm`) |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `rank 0` | `DP0` | `TP0` | `1` bucket (`192`, `2` saves + `10` replays) | `22.73 / 13.19 GiB` | `20.45 / 11.28 GiB` | `19.42 / 10.49 GiB` | `+3.31 / +2.70 GiB` | `+1.04 / +0.79 GiB` |
| `rank 1` | `DP0` | `TP1` | `1` bucket (`160`, `2` saves + `10` replays) | `22.79 / 13.21 GiB` | `20.47 / 11.29 GiB` | `19.44 / 10.52 GiB` | `+3.35 / +2.70 GiB` | `+1.02 / +0.77 GiB` |
| `rank 2` | `DP1` | `TP0` | `1` bucket (`128`, `2` saves + `10` replays) | `22.81 / 13.21 GiB` | `20.43 / 11.28 GiB` | `19.41 / 10.49 GiB` | `+3.40 / +2.72 GiB` | `+1.02 / +0.79 GiB` |
| `rank 3` | `DP1` | `TP1` | `1` bucket (`96`, `2` saves + `10` replays) | `22.82 / 13.22 GiB` | `20.43 / 11.28 GiB` | `19.45 / 10.53 GiB` | `+3.37 / +2.69 GiB` | `+0.97 / +0.75 GiB` |
| `rank 4` | `DP2` | `TP0` | `1` bucket (`64`, `2` saves + `10` replays) | `22.83 / 13.22 GiB` | `20.46 / 11.32 GiB` | `19.54 / 10.54 GiB` | `+3.29 / +2.67 GiB` | `+0.92 / +0.78 GiB` |
| `rank 5` | `DP2` | `TP1` | `1` bucket (`32`, `2` saves + `10` replays) | `22.88 / 13.24 GiB` | `20.41 / 11.28 GiB` | `19.48 / 10.51 GiB` | `+3.40 / +2.73 GiB` | `+0.93 / +0.77 GiB` |
| **`rank 6`** | **`DP3`** | **`TP0`** | **`0` buckets (`0` saves + `12` replays)** | `22.47 / 12.89 GiB` | **`19.49 / 10.52 GiB`** | `19.43 / 10.49 GiB` | `+3.04 / +2.40 GiB` | **`+0.06 / +0.03 GiB`** |
| **`rank 7`** | **`DP3`** | **`TP1`** | **`0` buckets (`0` saves + `12` replays)** | `22.44 / 12.88 GiB` | **`19.48 / 10.52 GiB`** | `19.45 / 10.52 GiB` | `+2.99 / +2.36 GiB` | **`+0.03 / +0.00 GiB`** |
| **Total (8 Workers)** | — | — | **`12` saves + `84` replays (`96` total)** | **`181.77 / 105.06 GiB`** | **`161.61 / 88.78 GiB`** | **`155.61 / 84.09 GiB`** | **`+26.15 / +20.97 GiB`** | **`+5.99 / +4.69 GiB`** |

- **Key Observations from the Per-Worker Breakdown**:
  - On `rank 6` and `rank 7` (which own `0` buckets under the 6-bucket schedule and replay all `12 / 12` subgraphs from peers `rank 0..5`), Pass 2 Cold Start (`19.49 GiB` and `19.48 GiB` `Pss_Anon`; `10.52 GiB` `tcmalloc_allocated`) is virtually indistinguishable from Pass 3 Warm Start (`19.43 GiB` and `19.45 GiB` `Pss_Anon`; `10.49 GiB` and `10.52 GiB` `tcmalloc_allocated`), confirming that peer replay eliminates `~100%` of follower-rank compilation memory overhead.
  - On `rank 0..5` (each owning `1` backbone bucket and `1` Eagle3 drafter bucket while replaying the other `10`), per-worker `Pss_Anon` drops by **`-2.28 to -2.47 GiB/worker`** (`22.73..22.88 GiB -> 20.41..20.47 GiB`), leaving only the `~0.92..1.04 GiB/worker` C++ `CompileAndLoad` metadata retention from the single bucket pair compiled by that owner rank.

### 8-Rank TPU Subgroup & Constant Collision Micro-Test Results

Running the 3 customer edge-case micro-tests across 8 TPU ranks (`torchrun --nproc_per_node=8`, `TPU_PARALLEL_PRECOMPILE=1`, `TPU_PRECOMPILE_PEER_REPLAY=1`) completed in `50.9 s` with `3 / 3` passing:

| Micro-Test Case (8 TPU Ranks) | Distinct Fingerprints | Compiler Ranks (`Compiles`) | Follower Ranks (`Peer Replays`) | Numerical & Hardlink Verification | Status |
| :--- | :---: | :---: | :---: | :--- | :---: |
| **`test_1a_pure_subgroup_4xTP2`**: Pure `4xTP2` subgroup `all_reduce` (`_supports_global_subgroup_tiling() -> False`) | `4` (`[0,1]`, `[2,3]`, `[4,5]`, `[6,7]`) | `[0, 2, 4, 6]` (`4`) | `[1, 3, 5, 7]` (`4`) | Outputs `[6.0, 6.0, 14.0, 14.0, 22.0, 22.0, 30.0, 30.0]`, `0` device-count errors, hardlinks verified | **`PASSED`** |
| **`test_1b_subgroup_plus_world_ep8`**: `4xTP2` subgroup + `EP=8` world `all_reduce` (`_graph_has_world_sized_op() -> True`) | `1` (all `8` ranks) | `[0]` (`1`) | `[1, 2, 3, 4, 5, 6, 7]` (`7`) | Outputs `[72.0] * 8`, `7 / 7` followers replayed `rank_0_0`, hardlinks verified | **`PASSED`** |
| **`test_2_constant_tensor_65537_idx65536`**: `65537`-element non-`nn.Parameter` `get_attr` constant differing at index `65536` | `2` (even `[0,2,4,6]` vs. odd `[1,3,5,7]`) | `[0, 1]` (`2`) | `[2, 3, 4, 5, 6, 7]` (`6`: `2,4,6 -> r0`, `3,5,7 -> r1`) | Index `65536` output `1.0` (even) vs. `2.0` (odd), `nn.Parameter` shape-only hash preserved, hardlinks verified | **`PASSED`** |

### End-to-End `vllm serve` & GSM8K Evaluation Results

| Metric | Observed Value |
| :--- | :--- |
| **`Compiling FX graph for range` (`6` buckets *`8` ranks)** | `48` |
| **`Saved compiled artifact` (`1` owner rank* `6` buckets)** | `6` |
| **`Replayed range` (`7` follower ranks * `6` buckets)** | `42 / 42` (`100%` peer replay hit rate) |
| **`Ignoring artifact` / Fingerprint Mismatches** | `0` |
| **Engine Init & Compilation Time (`6` buckets `[32..1024]`)** | `163.1 s` compilation (`322 s` total to `/health`) |
| **GSM8K (`100` samples, 5-shot, `strict-match`)** | `0.9500 +- 0.0219` (`95.0%`) |
| **GSM8K (`100` samples, 5-shot, `flexible-extract`)** | `0.9500 +- 0.0219` (`95.0%`) |

### Operational Observation: Explicit `--kv-cache-memory-bytes` vs. Auto-Sized `profile_run` on `partner/release-0.0.1`

When running `vllm serve` on `partner/release-0.0.1` with `TPU_PRECOMPILE_PEER_REPLAY=1` and `--gpu-memory-utilization 0.90` *without* `--kv-cache-memory-bytes`, `Worker_DP0_TP0_EP0` compiles the `(1024, 1024)` `profile_run` graph while the other 7 workers replay it from `profile_cache`. That single owner compilation leaves `DP0_TP0_EP0` with `0.14 GiB` less free HBM (`34.10 GiB` vs. `34.24 GiB` on `DP1..3`). On upstream `main`, PR `#1696` (`58c1d8e9b: Size EP-coupled DP engines' KV cache from the DP-wide minimum`) synchronizes `determine_available_memory()` across all `EP + DP` ranks via `utils.min_across_ep_dp(...)`. Because `#1696` is not included in `partner/release-0.0.1`, omitting `--kv-cache-memory-bytes` causes `DP0` (`37,224` blocks) and `DP1..3` (`37,389` blocks) to compute different `VllmConfig` cache hashes (`torch_compile_cache/<hash>/`), splitting peer replay into two separate cache directories (`DP0` vs. `DP1..3`). Setting `--kv-cache-memory-bytes` explicitly (as done in production Decode deployments and in the benchmarks above) keeps `VllmConfig` hashes identical across all 8 ranks on `partner/release-0.0.1`.
