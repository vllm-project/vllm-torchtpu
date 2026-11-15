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

## Coordination

Check and claim work items on
[#2765](https://github.com/vllm-project/vllm-torchtpu/issues/2765) before
starting bring-up or optimization work. Changes that require hooks in shared
core directories (`runner/`, `platforms/`, `layers/`) must land first as
standalone shared PRs.
