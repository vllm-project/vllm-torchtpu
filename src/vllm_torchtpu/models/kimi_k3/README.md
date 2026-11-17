# Kimi-K3 family (`kimi_k3`)

Self-contained TPU model directory for the Kimi-K3 family, which shares one
megakernel code base:

- [`moonshotai/Kimi-K3`](https://huggingface.co/moonshotai/Kimi-K3)
  (`KimiK3ForConditionalGeneration` / `KimiLinearForCausalLM`)
- [`moonshotai/Kimi-K3.1-Pro`](https://huggingface.co/moonshotai/Kimi-K3.1-Pro)
  (`KimiK31ForConditionalGeneration`): Kimi-K3 plus a sparse-attention
  indexer on every MLA layer, selected by the checkpoint config.

- **Support tier**: `Experimental`
- **Owners**: `@yiw-wang`, `@QiliangCui`, `@qizzzh`
- **Tracking issue**:
  [#2595](https://github.com/vllm-project/vllm-torchtpu/issues/2595)

## Directory Layout

- `megakernel/`: the decode megakernel, prefill kernels, DSpark draft kernels
  and the program that drives them, migrated per #2595. Kimi-K3.1 code paths
  leave the Kimi-K3 path bit-exact.
- `kernels/`: model-owned copies of reference kernels (each file records its
  source path and commit hash in its header).
- `settings.py`: model-scoped settings.

The native Kimi-K3 model still lives in `models/vllm/kimi_k3/`.

## Coordination

Check and claim work items on
[#2595](https://github.com/vllm-project/vllm-torchtpu/issues/2595) before
starting migration or optimization work. Changes that need hooks in shared
core directories (`runner/`, `platforms/`, `offload/`, `distributed/`) land
first as standalone shared PRs.
