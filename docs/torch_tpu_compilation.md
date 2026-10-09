# TorchTPU Compilation Cache

To enable persistent native C++ compilation caching in `vllm-torchtpu`, set `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT` to a persistent directory path:

```bash
export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="/path/to/persistent_cache"
```

## Cache Architecture & Environment Variables

`vllm-torchtpu` separates Python-level framework tracing cache from native C++ compiled binary executables:

1. **`TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT`** (C++ Native PJRT Binaries):
   - **Purpose**: Controls the persistent root directory for native C++ PJRT compiled binary executables (`.bin` files).
   - **Scope**: Native `torch_tpu` C++ backend. This is the primary driver variable required to enable persistent executable binary caching on TPU.

2. **`VLLM_CACHE_ROOT`** (vLLM Python-Level Framework Artifacts):
   - **Purpose**: Controls the root directory for vLLM Python-level framework tracing artifacts.
   - **Scope**: Stores PyTorch Dynamo FX graph bytecode, Guard expressions, AOTAutograd tracing artifacts (`torch_compile_cache/torch_aot_compile`), and `TpuCompilerAdaptor` metadata pickle handles (`artifact_compile_range_*`).
   - **Note**: `VLLM_CACHE_ROOT` manages Python framework artifacts and is completely independent of the C++ PJRT compiled binary executables managed by `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT`.

## Host memory during compilation

Large graphs can leave freed allocations resident in the host allocator between
compilation buckets. If startup runs out of host memory, opt in to reclamation:

```bash
export TPU_RECLAIM_COMPILE_MEMORY=1
```

After each successful compiler `compile` or `load` call returns, the adaptor runs
Python garbage collection and glibc `malloc_trim(0)`. Waiting until the call
returns lets temporary compiler objects go out of scope before collection.
Returned executables and their live references remain intact. Cache loading
also uses this boundary because a cache miss can compile a graph.

The flag defaults to `0`. It requires Linux with glibc as the active host
allocator; enabling it with another allocator raises an error before the
compiler call. It adds cleanup time at each boundary and cannot reclaim live
allocations or prevent every host OOM. It does not change the compilation cache
key.

## Compilation Debugging & Telemetry

### `VLLM_XLA_CHECK_RECOMPILATION` (Runtime Recompilation Detector)

- **Default**: `0` (Disabled)
- **Purpose**: Detects unexpected compilation during serving — such as an input shape, dtype, or code path that AOT warmup never precompiled and that stalls the engine loop on a cache miss.
- **How It Works**: Queries `torch.tpu._get_cache_stats()` and advances the cached graph count watermark at each checkpoint (including every AOT precompile bucket and runtime forward step), logging newly compiled graphs whenever a cache miss occurs.
- **Overhead**: `0` (zero runtime overhead) vs `1` (queries cache stats and logs per step).
- **Recommendation**:
  - `1`: Enable in CI and test pipelines to catch compilation leaks and un-warmed code paths.
  - `0`: Keep disabled in production and benchmarking for maximum throughput.
