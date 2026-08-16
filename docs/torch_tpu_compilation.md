# TorchTPU Compilation Cache

To enable persistent native C++ compilation caching in `vllm-torchtpu`, set `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT` to a persistent directory path:

```bash
export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="/path/to/persistent_cache"
```

## Environment Variables & Upstream Compatibility

- **`TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT`**:
  Controls the persistent root directory for native C++ PJRT compiled binary executables (`.bin` files). This is the primary driver required to enable persistent executable binary caching across server restarts on TPU.
- **`VLLM_CACHE_ROOT`**:
  Controls the root directory for vLLM Python-level framework tracing artifacts (`torch_compile_cache`), completely independent of the C++ PJRT binaries.
- **`VLLM_USE_AOT_COMPILE=1`** (Default on TPU):
  Enables vLLM's outer AOT compilation workflow.
- **`VLLM_USE_MEGA_AOT_ARTIFACT=0`** (Default on TPU):
  Upstream vLLM introduced Mega-AOT Artifacts (`VLLM_USE_MEGA_AOT_ARTIFACT=1`) to bundle Inductor C++ shared libraries (`.so`) into a single file for GPU fast startup. Because TPU uses native OpenXLA / PJRT executables and Tier-3 C++ binary caching rather than Inductor `.so` bundles, this is explicitly disabled on TPU (`=0`) to avoid Inductor-specific deserialization failures and unexpected recompilations.

## Compilation Debugging & Telemetry

### `VLLM_XLA_CHECK_RECOMPILATION` (Runtime Recompilation Detector)

- **Default**: `0` (Disabled)
- **Purpose**: Detects unexpected compilation during serving — such as an input shape, dtype, or code path that AOT warmup never precompiled and that stalls the engine loop on a cache miss.
- **How It Works**: Queries `torch.tpu._get_cache_stats()` and advances the cached graph count watermark at each checkpoint (including every AOT precompile bucket and runtime forward step), logging newly compiled graphs whenever a cache miss occurs.
- **Overhead**: `0` (zero runtime overhead) vs `1` (queries cache stats and logs per step).
- **Recommendation**:
  - `1`: Enable in CI and test pipelines to catch compilation leaks and un-warmed code paths.
  - `0`: Keep disabled in production and benchmarking for maximum throughput.
