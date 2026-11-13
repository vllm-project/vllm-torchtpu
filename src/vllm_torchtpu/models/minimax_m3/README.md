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

`__init__.py` is import-free and registers the model at plugin load
(`models.vllm.register_model_directories`). Hill-climbing work lives under
`experimental/` (Experimental tier: not in the support matrix, must not
change any supported configuration):

- `experimental/model.py`, `attention.py`, `moe.py`, `quant.py`, `dspark.py`:
  TPU model wiring, block-sparse (MSA) attention, MoE layers, `mxfp8` /
  `NVFP4` weight loading, and the DSpark draft.
- `experimental/kernels/`: model-owned JAX/Pallas kernels (MSA attention,
  indexer, selection, DCP candidate merge, 4-bit caches, cache writes).
- `experimental/ops/`: Torch-to-JAX custom op bridges scoped to `minimax_m3`.
- `experimental/hooks.py`: the vLLM config hook (block size, KV cache dtype,
  DCP, speculative decoding checks).
- `experimental/vocab_parallel_argmax.py`: the DSpark draft's vocab-parallel
  greedy argmax.
- `experimental/validation/`: serving configs.

Shared-core hooks it uses: `register_tpu_quant_config`, `TpuOwnedKVCache`,
`register_model_kv_cache_dtype` (`int4_per_token_head`),
`register_multi_shapes_kv_architectures`,
`register_kv_head_spanning_dcp_architectures`.

Tests: `tests/models/minimax_m3/experimental/` (microbenchmarks under
`benchmarks/`).

## Coordination

Check and claim work items on
[#2765](https://github.com/vllm-project/vllm-torchtpu/issues/2765) before
starting bring-up or optimization work. Changes that require hooks in shared
core directories (`runner/`, `platforms/`, `layers/`) must land first as
standalone shared PRs.
