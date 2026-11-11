# vLLM-Omni on TPU: Roadmap

This page describes how we plan to support vLLM-Omni on TPU over the next one
to two quarters. These are plans, not commitments. To work on an item, open an
issue first and follow [Contributing](README.md#contributing). See also the
project [roadmap](../../../docs/roadmap.md).

Last updated: November 2026.

## Model Support Scope

Our focus is on video and audio generation, world models, and TTS and audio
models. The models below are examples, not a full list.

| Category | Example models |
| :--- | :--- |
| Video and audio generation | MiniMax-H3, HunyuanVideo 1.5 |
| World models | LingBot-World 2.0, Cosmos3-Nano |
| TTS and audio | Qwen3-TTS, Qwen3-Omni |
| Image generation | Z-Image, Stable Diffusion 3.5 |

## Feature Support Scope

| Area | Planned TPU support |
| :--- | :--- |
| Entrypoints / API | All vLLM-Omni entrypoints, including streaming output and interactive world models |
| Connectors | Multi-stage serving with the Mooncake connector; NIXL is not planned |
| Parallelism | All parallelism strategies, including ring attention, parallel VAE decode and multi-host |
| Caching / quantization | TeaCache and Cache-DiT after a quality check; FP8 linears and attention |
| Compilation | Shape bucketing so that serving does not recompile |

## Hardware

We plan to support future hardware.

## Maintenance

- The `omni` extra pins a vLLM-Omni release; we bump it as new releases ship.
- At each pin bump we recheck the runtime patches in `patches.py` and drop
  obsolete ones.
- Omni tests live in [`tests_omni/`](../../../tests_omni/).
- Omni owners in [CODEOWNERS](../../../.github/CODEOWNERS) triage omni issues.
