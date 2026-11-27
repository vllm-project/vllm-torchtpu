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
