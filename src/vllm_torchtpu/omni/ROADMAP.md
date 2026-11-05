# vLLM-Omni on TPU: Roadmap

This page lists planned work for the omni plugin. These are plans, not
commitments. To work on an item, open an issue first and follow
[Contributing](README.md#contributing).

Last updated: November 2026.

## New models

| Model | Modality | Effort |
| :--- | :--- | :--- |
| MiniMax-H3 (33B) | Video and audio | L |
| LingBot-World 2.0 (14B) | Interactive world model | L |
| Cosmos3-Nano (15.8B) | Video and world | M |
| Z-Image / Z-Image-Turbo (6.15B) | Image | S |
| Qwen3-Omni (30B-A3B) | Omni (any-to-any) | L |
| HunyuanVideo 1.5 | Video | M |
| Stable Diffusion 3.5 (Large, Medium) | Image | S |
| Qwen3-TTS (1.7B) | Text to speech | M |

Effort is a rough relative estimate: S (small), M (medium), L (large).

## New features

### **Entrypoint / API**

| Feature | Upstream vLLM-Omni | TPU today |
| :--- | :--- | :--- |
| Streaming output, step execution, batching | Per pipeline; streaming is experimental | Untested |
| Interactive world models | Experimental; see [RFC #9565](https://github.com/vllm-project/vllm-omni/issues/9565) | Not working: causal attention falls back to SDPA |


### **Connectors (Mooncake, NIXL, etc.)**

| Feature | Upstream vLLM-Omni | TPU today |
| :--- | :--- | :--- |
| Disaggregated multi-stage serving | Supported, e.g. Wan stages ([#6285](https://github.com/vllm-project/vllm-omni/issues/6285)) | Not working: multi-rank stages are unsupported |
| Mooncake device-to-device connector | Mooncake store and transfer-engine connectors (NIXL, Mori also) | Not yet: stage outputs pass through CPU shared memory |

### **Tensor caching / quantization**

| Feature | Upstream vLLM-Omni | TPU today |
| :--- | :--- | :--- |
| TeaCache / Cache-DiT | Supported | Untested; lossy, needs a quality check |
| FP8 linears and FP8 attention | Supported on CUDA | No FP8 path yet |

### **Parallelism / multi-host**

| Feature | Upstream vLLM-Omni | TPU today |
| :--- | :--- | :--- |
| Ring / 2-D ring attention | Supported | Not working: torch_tpu P2P send hangs |
| VAE tiling / parallel decode | Supported | Slow decode; faster decode in draft [#2459](https://github.com/vllm-project/vllm-torchtpu/pull/2459) |
| Multi-host support | Ray multi-node | Single host only |

### **Misc**

| Feature | Upstream vLLM-Omni | TPU today |
| :--- | :--- | :--- |
| Zero-recompile / shape bucketing | Inductor compile (not on TPU) | Partial: first requests still pay compile time |

## Hardware

We plan to support future hardware.

## Maintenance

- The `omni` extra pins a vLLM-Omni release; we bump it as new releases ship.
- At each pin bump we recheck the runtime patches in `patches.py` and
  `platform.py` and drop obsolete ones.
- Omni tests live in [`tests_omni/`](../../../tests_omni/).
- Omni owners in [CODEOWNERS](../../../.github/CODEOWNERS) triage omni issues.
