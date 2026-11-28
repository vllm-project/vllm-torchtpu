# Inkling 2 (`inkling2`)

Self-contained TPU model directory for Thinking Machines Lab's Inkling 2 family:
[`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
(`Inkling2LoopForConditionalGeneration`, `model_type: inkling2_loop`) and
[`thinkingmachines/Inkling-2-Small`](https://huggingface.co/thinkingmachines/Inkling-2-Small)
(`Inkling2ForConditionalGeneration`, `model_type: inkling2`).

- **Support tier**: `Experimental`
- **Tracking issue**: [#3650](https://github.com/vllm-project/vllm-torchtpu/issues/3650)

## Directory Layout

- `experimental/quant.py`: `hf_quant_config.json` per-layer precision map
  (`NVFP4` layers `0..19`, `MXFP8` layers `20..23`) and pure-PyTorch `MXFP8` /
  `NVFP4` dequantization utilities.
- `experimental/ops.py` *(follow-up PR)*: TPU-portable pure-PyTorch primitives
  (block `attn_res`, rank-64 cross-pass message controllers, static/dynamic
  short convolutions, softcapped `SiTU` MoE, and split relative-bias attention
  with exact LSE merge).
- `experimental/model.py` *(follow-up PR)*: looped/non-looped TPU model wiring
  and `ModelAdapter` integration once shared extension hooks (`#3125`, `#3277`)
  land on `main`.

## Coordination

Changes that require hooks in shared core directories (`runner/`, `platforms/`,
`layers/`) must land first as standalone shared PRs (tracking `#3125` for
model-registered checkpoint quant configs and `#3277` for model-owned input and
KV-cache adapters).
