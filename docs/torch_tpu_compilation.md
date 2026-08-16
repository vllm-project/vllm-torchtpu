# TorchTPU Compilation Cache

To enable persistent native C++ compilation caching in `vllm-torchtpu`, set `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT` to a persistent directory path:

```bash
export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="/path/to/persistent_cache"
```

## Environment Variables

- **`TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT`**:
  Controls the persistent root directory for native C++ PJRT compiled binary executables (`.bin` files). This is the primary driver required to enable persistent executable binary caching across server restarts on TPU.
- **`VLLM_CACHE_ROOT`**:
  Controls the root directory for vLLM Python-level framework tracing artifacts (`torch_compile_cache`), completely independent of the C++ PJRT binaries.

## Compilation Debugging & Telemetry

### `VLLM_XLA_CHECK_RECOMPILATION` (Runtime Recompilation Detector)

- **Default**: `0` (Disabled)
- **Purpose**: Detects unexpected compilation during serving — such as an input shape, dtype, or code path that AOT warmup never precompiled and that stalls the engine loop on a cache miss.
- **How It Works**: Queries `torch.tpu._get_cache_stats()` and advances the cached graph count watermark at each checkpoint (including every AOT precompile bucket and runtime forward step), logging newly compiled graphs whenever a cache miss occurs.
- **Overhead**: `0` (zero runtime overhead) vs `1` (queries cache stats and logs per step).
- **Recommendation**:
  - `1`: Enable in CI and test pipelines to catch compilation leaks and un-warmed code paths.
  - `0`: Keep disabled in production and benchmarking for maximum throughput.
