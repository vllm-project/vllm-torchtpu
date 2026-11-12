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

3. **`TPU_ARTIFACT_STORE`** (opt-in content-addressed store, under `VLLM_CACHE_ROOT/tpu_artifact_store`):
    - **Purpose**: Replays a compiled bucket when vLLM's `torch_compile_cache/<hash>` directory misses only because a vLLM env var or a traced file changed. Each artifact is keyed by the TPU compilation hash, vLLM's config hash, the graph fingerprint, the concrete inputs and the per-bucket key.
    - **Scope**: Hard links to the artifacts in `torch_compile_cache`, so it adds no disk on the run that writes it; it is not evicted. Off by default; see `TPU_ARTIFACT_STORE` in [Environment Variables](../reference/env_vars.md).

## Compilation Debugging & Telemetry

### `VLLM_XLA_CHECK_RECOMPILATION` (Runtime Recompilation Detector)

- **Default**: `0` (Disabled)
- **Purpose**: Detects unexpected compilation during serving — such as an input shape, dtype, or code path that AOT warmup never precompiled and that stalls the engine loop on a cache miss.
- **How It Works**: Queries `torch.tpu._get_cache_stats()` and advances the cached graph count watermark at each checkpoint (including every AOT precompile bucket and runtime forward step), logging newly compiled graphs whenever a cache miss occurs.
- **Overhead**: `0` (zero runtime overhead) vs `1` (queries cache stats and logs per step).
- **Recommendation**:
    - `1`: Enable in CI and test pipelines to catch compilation leaks and un-warmed code paths.
    - `0`: Keep disabled in production and benchmarking for maximum throughput.
