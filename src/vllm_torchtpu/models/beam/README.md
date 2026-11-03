# Reflection Beam (`beam`)

Self-contained TPU model directory for
[`reflection/Beam-1.0-501B-A23B`](https://huggingface.co/reflection/Beam-1.0-501B-A23B)
and its FP8 checkpoint
[`reflection/Beam-1.0-501B-A23B-FP8`](https://huggingface.co/reflection/Beam-1.0-501B-A23B-FP8)
(`BeamForCausalLM`).

- **Support tier**: `Experimental`
- **Owners**: `@foxxyanya`, `@HazyRaven`
- **Tracking issue**:
  [#3602](https://github.com/vllm-project/vllm-torchtpu/issues/3602)

## Directory Layout

- `model.py`: the TPU model, following the layer structure and parameter names
  of upstream `vllm/models/beam/nvidia/model.py`.
- `ops.py`: PyTorch versions of upstream's fused Beam kernels.
- `config.py`: `TPUBeamConfig`, which rejects unsupported configurations
  before the weights load.

## Coordination

Check and claim work items on
[#3602](https://github.com/vllm-project/vllm-torchtpu/issues/3602) before
starting new work on this model. Changes that need hooks in shared core
directories land first as standalone shared PRs.
