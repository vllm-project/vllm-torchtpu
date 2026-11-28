# TorchTPU Compilation Cache

To enable persistent native C++ compilation caching in `vllm-torchtpu`, set `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT` to a persistent directory path:

```bash
export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="/path/to/persistent_cache"
```

## Cache Architecture & Environment Variables

`vllm-torchtpu` layers vLLM's Python-level framework cache on top of `torch_tpu`'s three-tier native C++ PJRT executable cache:

1. **Tier-1 In-Memory Cache** (Process-Local C++ `CompilationCache`):
    - **Purpose**: Holds loaded `PjRtLoadedExecutable` instances inside each worker process (`torch.tpu._get_cache_stats()`).
    - **Eviction**: `torch.tpu._clear_cache()` (`CompilationCache::EvictAll`) clears all Tier-1 entries in the process and waits a minimum of `1 s` whenever the cache is non-empty.

2. **`TORCH_TPU_TIER2_COMPILATION_CACHE`** (Host Shared-Memory PJRT Cache):
    - **Purpose**: Shares serialized PJRT executables across ranks on the same host under `/dev/shm/torch_tpu_cache/<name>`.
    - **Default Behavior**: Enabled automatically (`"default"`) when `world_size > 1`, or set to `"tpu_tier2_cache"` when `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT` is configured. Disabled by default on single-rank (`world_size == 1`) runs without Tier-3. Setting `"disabled"` turns off **both** Tier-2 and Tier-3.

3. **`TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT`** (Persistent On-Disk PJRT Binaries):
    - **Purpose**: Controls the persistent root directory for native C++ PJRT compiled binary executables (`.bin` files) across restarts (requires Tier-2 to remain enabled).

4. **`VLLM_CACHE_ROOT`** (vLLM Python-Level Framework Artifacts):
    - **Purpose**: Controls the root directory for vLLM Python-level framework tracing artifacts.
    - **Scope**: Stores PyTorch Dynamo FX graph bytecode, Guard expressions, AOTAutograd tracing artifacts (`torch_compile_cache/torch_aot_compile`), and `TpuCompilerAdaptor` bundled pickle handles (`artifact_compile_range_*`).
    - **Note**: `VLLM_CACHE_ROOT` manages Python framework artifacts and is independent of the C++ PJRT compiled binary executables managed by Tier-1/Tier-2/Tier-3.

### `VLLM_TPU_RELOAD_COMPILED_EXECUTABLE` (Cold-Start Host DRAM Reduction)

- **Default**: `0` (Disabled; opt-in temporary workaround until `torch_tpu` / `libtpu` frees compile-only PJRT host metadata natively).
- **Mechanism**: After `TpuCompilerAdaptor` finishes a fresh `CompileAndLoad` and saves its bundled artifact, it drops the fat `CompileAndLoad` executable references, calls `torch.tpu._clear_cache()` to evict Tier-1, and reloads the slim executable via `LoadSerializedExecutable` before serving (warm-start cache hits are unchanged; standalone `@torch.compile(backend="tpu")` helpers are not covered).
- **Trade-offs & Side Effects**:
    - Each newly compiled bucket pays one `LoadSerializedExecutable` pass plus `CompilationCache::EvictAll`'s minimum `1 s` wait when Tier-1 is non-empty.
    - Clearing Tier-1 evicts all earlier eager (`DEFER_AND_FUSE`) programs in the process. Each evicted eager program reloads from Tier-2 or Tier-3 on its next use — **which can occur on a live serving step** — or recompiles from scratch if both Tier-2 and Tier-3 are disabled.
    - Measured on 8-chip `tpu7x` serving `Qwen/Qwen3.5-397B-A17B-FP8`: steady-state host `Pss_Anon` drops from `224.8 GiB` to `210.8 GiB` (`-14.0 GiB`, `531 s` to `540 s` `KV-ready -> /health`) on `PCP=8` Prefill and from `276.6 GiB` to `262.9 GiB` (`-13.8 GiB`, `421 s` to `425 s`) on `DP=8` Decode with Tier-2 disabled (`-27.8 GiB` combined across 1P1D), and from `55.2 GiB` to `53.2 GiB` (`-2.0 GiB` host, `-1.2 GiB` max worker, `627 s` to `638 s`) at `TP=2, DP=4, EP=8, MTP=3` (`41.5 GiB` to `39.8 GiB`, `-1.7 GiB` host, `-1.6 GiB` max worker without MTP) with Tier-2 enabled.

## Multi-host Expert Parallelism

Programs built over the expert-parallel (EP) mesh, such as the fused EP MoE kernel, have the EP device placement compiled in: which TPU device each EP rank runs on. On a multi-host slice this placement can change between two starts of the same deployment, because the host that runs EP ranks 0-7 does not always hold the lowest device ids. The compile cache key includes the placement, so a program compiled for one placement is never reused under another one.

What this costs an operator:

- **Cold start per new placement**: a start with a placement the cache has not seen compiles from scratch. For DeepSeek-V4-Pro on a 2-host v7x-16 slice, that is about 20 minutes of compilation instead of under 2 minutes.
- **Disk**: each placement gets its own cache directory, so the cache grows with every placement seen. A slice with H hosts can have up to H! placements: 2 on a 2-host slice, 24 on a 4-host slice.
- **Single host**: the placement on one host does not change between starts, so it compiles once.

## Compiled helper functions and the `tpu_compile` section

Besides the model forward, the runner calls many small compiled programs every step: logits selection, sampling, rejection sampling, the MTP drafter's helpers, state copies. Each is a plain function compiled with `tpu_compile` (`src/vllm_torchtpu/compilation/decorators.py`), which calls `torch.compile(fn, backend="tpu", fullgraph=True, dynamic=False)` and wraps the result. The `tpu_compile` section of `--additional-config` changes how those functions are called:

| Option | Default | Effect when `true` |
|---|---|---|
| `fast_compiled_call` | `false` | Once a function has a compiled entry, each call checks the entries' Dynamo guards against its arguments and calls the matching entry's code directly, skipping Dynamo's per-call frame evaluation. A call that no entry accepts goes through Dynamo, which compiles a new entry if needed. The compiled programs and their guards are the same as with the option off. |

```bash
vllm serve <model> --additional-config '{"tpu_compile": {"fast_compiled_call": true}}'
```

- **When it applies**: each TPU worker arms the option when it is built, before any compiled function runs. `TpuPlatform.check_and_update_config` refuses a malformed section (an unknown key, or a value that is not a JSON boolean) at startup.
- **Compile-cache key**: the section is part of `VllmConfig.compute_hash()`, which keys vLLM's AOT compile cache and piecewise cache directory under `VLLM_CACHE_ROOT`; only the diagnostics keys in `runner.utils.HASH_IGNORED_ADDITIONAL_CONFIG_KEYS` are left out. A start with the option set the other way does not reuse those artifacts.
- **Calls that still go through Dynamo**: every call of a function whose signature takes `*args` or `**kwargs`, and any call under a non-default `torch.compiler.set_stance` or inside another compiled function. With `TORCHDYNAMO_DISABLE=1` the decorator returns the plain function, so nothing is compiled or dispatched.
- **Logs**: with the option on, each worker logs `tpu_compile: fast_compiled_call=True` at startup, `fast_compiled_call: first direct dispatch of <function>` once per function, and `fast_compiled_call: fast=<n> slow=<n>` (direct calls and calls through Dynamo, over the whole process) every 20,000 calls.

## Compilation Debugging & Telemetry

### `VLLM_XLA_CHECK_RECOMPILATION` (Runtime Recompilation Detector)

- **Default**: `0` (Disabled)
- **Purpose**: Detects unexpected compilation during serving — such as an input shape, dtype, or code path that AOT warmup never precompiled and that stalls the engine loop on a cache miss.
- **How It Works**: Queries `torch.tpu._get_cache_stats()` and advances the cached graph count watermark at each checkpoint (including every AOT precompile bucket and runtime forward step), logging newly compiled graphs whenever a cache miss occurs.
- **Overhead**: `0` (zero runtime overhead) vs `1` (queries cache stats and logs per step).
- **Recommendation**:
    - `1`: Enable in CI and test pipelines to catch compilation leaks and un-warmed code paths.
    - `0`: Keep disabled in production and benchmarking for maximum throughput.
- **Note**: With `VLLM_TPU_RELOAD_COMPILED_EXECUTABLE=1`, each newly compiled bucket clears `torch_tpu`'s Tier-1 compilation cache. This both resets the cached graph count watermark and causes evicted eager programs that execute again on later steps to re-enter Tier-1 with `read_count=1`, so `TPUModelRunner._update_num_xla_graphs` falsely reports those reloads as new graphs (showing `0.000s` compile time when served by Tier-2 or Tier-3). Use the `TpuCompilerAdaptor` reload logs or `torch.tpu._get_cache_stats().num_compilations` to count compilations instead.
