# Qwen3.5 MoE (`qwen3_5`)

TPU model directory for the Qwen3.5 mixture-of-experts checkpoints
(`Qwen3_5MoeForConditionalGeneration` / `Qwen3_5MoeForCausalLM`):

- [`Qwen/Qwen3.5-397B-A17B-FP8`](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8)
- [`Qwen/Qwen3.5-35B-A3B-FP8`](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8)

- **Support tier**: `Experimental`
- **Owners**: `@mshrihari3`, `@benliangw`, `@caojx-google`,
  `@kyuyeunk`
- **Tracking issue**:
  [#3210](https://github.com/vllm-project/vllm-torchtpu/issues/3210)

The native model code stays in `models/vllm/` for now (`qwen3_5_patch.py`,
`qwen3_5_mtp.py`, and `qwen3_next_patch.py`, which Qwen3.5 shares with
Qwen3-Next). Moving it is a separate change.

## Directory Layout

The top-level files are the path that works consistently on the production
configs below. Hill-climbing work (new kernels and features still being
measured) goes in `experimental/` and moves to the top level with `git mv` once
its end-to-end results are validated. Each entry is added by the first change
that needs it:

- `kernels/`: Qwen3.5-only kernels and model-owned copies of reference kernels
  (each copy records its source path and commit hash in its header).
- `settings.py`: settings that apply only to Qwen3.5. They are read from
  `--additional-config '{"qwen3_5": {...}}'` for
  `Qwen3_5MoeForConditionalGeneration`; a `qwen3_5` section given to another
  model is rejected. A setting is added by the change that implements it, and
  unknown keys fail at engine start (see [Settings](#settings)).
- `hooks.py`: this model's implementation of the shared-core hooks. Its config
  hook is registered in vLLM's `MODELS_CONFIG_MAP`: it keeps vLLM's own
  Qwen3.5 checks, validates the settings, and fills TPU serving defaults the
  user left unset (see its docstring).
- `validation/`: serving configs, golden outputs, and accuracy and performance
  baselines for this model's production configs. The first entries are the
  serving configs of the InferenceX (IX) prefill/decode points and the 1P1D
  accuracy gates (see `validation/README.md`).
- `experimental/`: hill-climbing kernels and features that are not yet part of
  the production configs.

## Settings

| Setting | Type | Default | What it does |
| --- | --- | --- | --- |

No setting exists yet. Each change that adds one adds its row here.

## Production Configs

| Config | Layout | Current entry |
| --- | --- | --- |
| C1: interleaved | Qwen3.5-397B-A17B-FP8, 1 host (4 TPU v7x chips, 8 devices), TP1 / DP8 + EP | `scripts/vllm/benchmarking/configs/qwen3.5-397b-fp8-tp1-dp8-ep.sh` (nightly perf and eval) |
| C2: long context | Not defined yet | None |
| C3: disaggregated | Qwen3.5-397B-A17B-FP8, 1 prefill + 1 decode | `.buildkite/kubernetes/manifests/workloads/qwen3.5-397b-1p1d.yaml` (nightly perf) |

Optimizations are folded into these configs once their end-to-end results are
validated, rather than kept as separate benchmark profiles.

## Coordination

Check and claim work items on
[#3210](https://github.com/vllm-project/vllm-torchtpu/issues/3210) before
starting bring-up or optimization work. Changes that need hooks in shared core
directories (`runner/`, `platforms/`, `layers/`, `core/`, `distributed/`,
`offload/`) land first as standalone shared PRs; this directory holds only
Qwen3.5-specific code.
