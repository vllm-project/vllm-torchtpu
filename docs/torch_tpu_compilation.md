# TorchTPU Compilation Cache

To enable persistent native C++ compilation caching in `vllm-torchtpu`, set `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT` to a persistent directory path:

```bash
export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="/path/to/persistent_cache"
```

## Cache Architecture & Categories

`vllm-torchtpu` structures its compilation caching into three clear categories:

1. **Category 1 — Tier-3 Native C++ PJRT Binaries (`TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT`)**:
   - **Purpose**: Persistent root directory for native C++ PJRT compiled binary executables (`.bin` files).
   - **Scope**: Native `torch_tpu` C++ backend. This is the primary driver required to enable persistent executable binary caching across server restarts on TPU.

2. **Category 2 — vLLM Compiler Adapter Metadata Handles (`TpuCompilationHandle`)**:
   - **Location**: `$VLLM_CACHE_ROOT/torch_compile_cache/artifact_compile_range_*`
   - **Purpose**: Stores lightweight metadata handles (`TpuCompilationHandle`) created by `TpuCompilerAdaptor`.
   - **Storage Optimization**: Instead of pickling multi-megabyte `PjRtLoadedExecutable` binaries (which would duplicate Category 1 storage), it only stores tiny ~100-byte metadata pointers (`tier3_rel_path` and output reconstruction functions) totaling <50KB across the entire model. On warm load, `_unpickle_compiled_executable` directly reconstructs the executable from the Tier-3 binary in microseconds.

3. **Category 3 — PyTorch Dynamo / AOTAutograd Tracing Cache**:
   - **Location**: `$VLLM_CACHE_ROOT/torch_compile_cache/torch_aot_compile`
   - **Scope**: Stores PyTorch Dynamo FX graph bytecode, guards, and AOTAutograd tracing artifacts.

## Environment Variables & Upstream Compatibility

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
