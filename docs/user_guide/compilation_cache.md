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

## Multi-host Expert Parallelism

Programs built over the expert-parallel (EP) mesh, such as the fused EP MoE kernel, have the EP device placement compiled in: which TPU device each EP rank runs on. On a multi-host slice this placement can change between two starts of the same deployment, because the host that runs EP ranks 0-7 does not always hold the lowest device ids. The compile cache key includes the placement, so a program compiled for one placement is never reused under another one.

What this costs an operator:

- **Cold start per new placement**: a start with a placement the cache has not seen compiles from scratch. For DeepSeek-V4-Pro on a 2-host v7x-16 slice, that is about 20 minutes of compilation instead of under 2 minutes.
- **Disk**: each placement gets its own cache directory, so the cache grows with every placement seen. A slice with H hosts can have up to H! placements: 2 on a 2-host slice, 24 on a 4-host slice.
- **Single host**: the placement on one host does not change between starts, so it compiles once.

## Compilation Debugging & Telemetry

### `VLLM_XLA_CHECK_RECOMPILATION` (Runtime Recompilation Detector)

- **Default**: `0` (Disabled)
- **Purpose**: Detects unexpected compilation during serving — such as an input shape, dtype, or code path that AOT warmup never precompiled and that stalls the engine loop on a cache miss.
- **How It Works**: Queries `torch.tpu._get_cache_stats()` and advances the cached graph count watermark at each checkpoint (including every AOT precompile bucket and runtime forward step), logging newly compiled graphs whenever a cache miss occurs.
- **Overhead**: `0` (zero runtime overhead) vs `1` (queries cache stats and logs per step).
- **Recommendation**:
    - `1`: Enable in CI and test pipelines to catch compilation leaks and un-warmed code paths.
    - `0`: Keep disabled in production and benchmarking for maximum throughput.
