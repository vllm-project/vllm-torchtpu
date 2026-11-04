# Qwen3.8-Flash-Next (`qwen38_flash`)

Self-contained TPU model directory for Qwen3.8-Flash-Next AgentX hill-climbing
optimizations.

- **Support tier**: `Experimental`
- **Owners**: `@DRZDONG`, `@caojx-google`
- **Tracking issue**:
  [#2766](https://github.com/vllm-project/vllm-torchtpu/issues/2766) (related:
  [#2630](https://github.com/vllm-project/vllm-torchtpu/issues/2630))

## Directory Layout

- `model.py`, `attention.py`, `moe.py`, `ple.py`, `quant.py`: specialized QSA,
  GDN, PLE, MoE, and MTP serving optimizations for Qwen3.8-Flash-Next.
- `kernels/`: model-owned copies of specialized JAX/Pallas and SparseCore
  kernels (each file records its upstream reference path and commit hash in its
  header).
- `ops/`: Torch-to-JAX custom op bridges scoped to `qwen38_flash`.
- `hooks.py`, `settings.py`: shared-core interface hooks and model-scoped
  configuration.
- `validation/`: AgentX serving configs, golden checks, and accuracy/performance
  baselines.

## Coordination

Check and claim optimization items on
[#2766](https://github.com/vllm-project/vllm-torchtpu/issues/2766) before
starting new hill-climbing work. Changes that require hooks in shared core
directories (`runner/`, `platforms/`, `distributed/`, `offload/`) must land
first as standalone shared PRs.
