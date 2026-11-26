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
- `experimental/variants.py`, `profiles.py`, `plan.py`: the model plan. The
  default profile is the proven-best served recipe, so `vllm serve` needs no
  extra flags; `--additional-config '{"model_plan": {"profile": "<name>"}}'`
  picks another profile.
- `experimental/validation/`: serving configs and the default plan snapshot.

Shared-core hooks it uses: `register_tpu_quant_config`, `TpuOwnedKVCache`,
`register_kv_head_spanning_dcp_architectures`; through its plan, the model
attributes `model_all_reduce` / `spec_verify_sampler`.

## Serving (Experimental)

TP8 with expert parallelism, the DSpark draft and `--kv-cache-dtype fp8`:

- c4 / c16: `--decode-context-parallel-size 2`, default profile `agentx`
  (fp8 MSA caches).
- c32 / c48 / c64: `--decode-context-parallel-size 4` with KV offload and
  `--additional-config '{"model_plan": {"profile": "agentx-int4"}}'`: 4-bit
  MSA caches with per-token-head scales; the draft's KV stays fp8.

Tests: `tests/models/minimax_m3/experimental/` (microbenchmarks under
`benchmarks/`).

The compile cache key covers the paths in `_RUNTIME_CACHE_KEY_PATHS` plus
every `kernels/` and `ops/` directory under `vllm_torchtpu/models/`, so keep
custom-op bridges and kernel copies in those directories.

## Coordination

Check and claim work items on
[#2765](https://github.com/vllm-project/vllm-torchtpu/issues/2765) before
starting bring-up or optimization work. Changes that require hooks in shared
core directories (`runner/`, `platforms/`, `layers/`) must land first as
standalone shared PRs.
