# Qwen3.8-Flash-Next (`qwen38_flash`)

Self-contained TPU model directory for Qwen3.8-Flash-Next AgentX
hill-climbing (`Qwen4ExpForCausalLM` /
`Qwen4ExpForConditionalGeneration` checkpoints).

- **Support tier**: `Experimental` (all code lives under `experimental/`)
- **Owners**: `@DRZDONG`, `@caojx-google`
- **Tracking issue**:
  [#2766](https://github.com/vllm-project/vllm-torchtpu/issues/2766) (related:
  [#2630](https://github.com/vllm-project/vllm-torchtpu/issues/2630))

## Selecting this model

The shared `models/vllm/qwen4_exp` implementation stays the default for the
Qwen4Exp architectures. Set `VLLM_TPU_QWEN38_FLASH_EXPERIMENTAL=1` (a
temporary migration selector, see `experimental/settings.py`) to serve them
with `experimental/` instead. Without the selector only the import-free
`register()` hook runs, so supported configurations are unaffected.

## Directory Layout

- `experimental/model.py`, `attention.py` (QSA), `gdn.py`, `mtp.py`,
  `hyperconnection.py`, `ple*.py`, `pool*.py`, `pd*.py`: the hill-climbing
  implementation of QSA, GDN, PLE, MoE wiring, MTP, pooled KV cache and PD
  transfer for Qwen3.8-Flash-Next.
- `experimental/kernels/`: model-owned copies of the JAX/Pallas and
  SparseCore kernels. Each file header records the source path and commit hash
  it was copied from; from that point the copy is owned here.
- `experimental/ops/`: Torch-to-JAX custom op bridges used only by this model
  (`qwen38_flash::` op namespace).
- `experimental/hooks.py`: this model's implementation of the shared-core
  interfaces (registration, config, runner, offload).
- `experimental/settings.py`: environment variables that apply only to this
  model. Shared `envs.py` keeps only flags read by shared code.
- `experimental/validation/`: AgentX serving configs, golden checks, and
  accuracy/performance baselines.

Tests live in `tests/models/qwen38_flash/experimental/`.

## Coordination

Check and claim optimization items on
[#2766](https://github.com/vllm-project/vllm-torchtpu/issues/2766) before
starting new hill-climbing work. Changes that need hooks in shared core
directories (`runner/`, `platforms/`, `spec_decode/`, `distributed/`,
`offload/`) land first as small standalone shared PRs that add a generic
interface; Qwen-specific logic stays in this directory.
