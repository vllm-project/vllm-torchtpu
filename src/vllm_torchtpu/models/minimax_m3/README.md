# MiniMax-M3.1-Flash (`minimax_m3`)

Self-contained TPU model directory for
[`MiniMaxAI/MiniMax-M3.1-Flash`](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash)
(`MiniMaxM3SparseForConditionalGeneration` / `MiniMaxM3SparseForCausalLM`,
`model_type: minimax_m3_vl`).

- **Support tier**: `Experimental`
- **Owners**: `@DRZDONG`, `@caojx-google`, `@qizzzh`
- **Tracking issue**:
  [#2765](https://github.com/vllm-project/vllm-torchtpu/issues/2765)

## Directory Layout

- `model.py`, `attention.py`, `moe.py`, `quant.py`: TPU model wiring,
  block-sparse attention, MoE layers, and `mxfp8` / `NVFP4` weight loading.
- `kernels/`: model-owned copies of specialized JAX/Pallas kernels (each file
  records its upstream reference path and commit hash in its header).
- `ops/`: Torch-to-JAX custom op bridges scoped to `minimax_m3`.
- `hooks.py`, `settings.py`: shared-core interface hooks and model-scoped
  configuration.
- `validation/`: serving configs, golden checks, and accuracy/performance
  baselines.

The compile cache key covers the paths in `_RUNTIME_CACHE_KEY_PATHS` plus
every `kernels/` and `ops/` directory under `vllm_torchtpu/models/`, so keep
custom-op bridges and kernel copies in those directories. Another file the graphs
depend on, or a model setting that changes a graph, enters the key through
`register_compile_cache_key` (`compilation/tpu_compiler.py`), called from the
model's registration hook.

## Coordination

Check and claim work items on
[#2765](https://github.com/vllm-project/vllm-torchtpu/issues/2765) before
starting bring-up or optimization work. Changes that require hooks in shared
core directories (`runner/`, `platforms/`, `layers/`) must land first as
standalone shared PRs.
