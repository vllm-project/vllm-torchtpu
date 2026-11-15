# vLLM TorchTPU

**The PyTorch-native TPU backend for [vLLM](https://github.com/vllm-project/vllm).** Run the same vLLM you already use — same CLI, same OpenAI-compatible server, same model definitions — on Google Cloud TPUs with no model code changes.

`vllm-torchtpu` is an out-of-tree vLLM platform plugin that executes vLLM's standard PyTorch models through **TorchTPU** (`torch.compile(backend="tpu")`) and a library of custom Pallas kernels for attention, Mixture-of-Experts, and quantized matmul.

| | |
| :--- | :--- |
| **Documentation** | <https://vllm-torchtpu.readthedocs.io/en/latest/> |
| **Source** | <https://github.com/vllm-project/vllm-torchtpu> |
| **Issues** | <https://github.com/vllm-project/vllm-torchtpu/issues> |
| **Community** | [vLLM Forum (TPU)](https://discuss.vllm.ai/c/hardware-support/google-tpu-support/27) · [vLLM Slack](https://slack.vllm.ai) `#sig-tpu` |
| **License** | Apache 2.0 |

## Tags

| Tag | Built from | Contents |
| :--- | :--- | :--- |
| `latest` | Most recent **release** | Production image: `vllm-torchtpu` and its pinned vLLM installed. Recommended for serving. |
| `vX.Y.Z` (e.g. `v0.29.0`) | That release | Same as `latest`, pinned to one release. Pre-release tags (`vX.Y.ZrcN`) never move `latest`. |
| `nightly` | Most recent nightly from `main` | Development image: repository source checked out and installed in editable mode, plus test and benchmarking dependencies. For trying unreleased features. |
| `nightly-YYYYMMDD-<commit>-<vllm-ref>` | One nightly build | A pinned nightly. The tag records the build date, `vllm-torchtpu` commit, and vLLM version it was built against. |

Images are `linux/amd64`, based on `python:3.12-slim-bookworm`, and start a shell (`/bin/bash`) by default; you run `vllm` inside the container.

## Supported hardware

| TPU generation | Status |
| :--- | :--- |
| **TPU v7x (Ironwood)** | Recommended — validated in CI on every pull request and nightly |
| TPU v6e / v5e / v5p | Experimental — not covered by CI |

The container must run **on a Cloud TPU VM**. See [Cloud TPU Setup](https://vllm-torchtpu.readthedocs.io/en/latest/getting_started/tpu_setup/) for project setup, quota, and provisioning.

## Quick start

On a Cloud TPU VM:

```bash
docker pull vllm/vllm-torchtpu:latest

docker run -it --rm \
  --privileged --net=host --shm-size=150gb \
  -e HF_TOKEN="$HF_TOKEN" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-torchtpu:latest
```

`--privileged` and `--net=host` give the container access to the TPU devices; `--shm-size` provides the shared memory vLLM's workers need. Mounting the Hugging Face cache avoids re-downloading model weights on every run.

Inside the container, start an OpenAI-compatible server exactly as on any other vLLM platform:

```bash
vllm serve meta-llama/Llama-3.2-1B-Instruct \
  --tensor-parallel-size 1 \
  --max-model-len 4096
```

Then, from the TPU VM:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [{"role": "user", "content": "Explain how TPUs accelerate matrix multiplication."}],
    "max_tokens": 128
  }'
```

> **Note:** the first startup compiles TPU graphs for the configured shapes and can take several minutes. Set `VLLM_XLA_CACHE_PATH` to a mounted volume to persist compiled artifacts across container restarts — see [Compilation & Warmup Cache](https://vllm-torchtpu.readthedocs.io/en/latest/user_guide/compilation_cache/).

## What's supported

First-class TPU kernel, quantization, and parallelism support for Qwen3 / Qwen3.5, DeepSeek-V3 / R1 / V4, Kimi-K2.5 / K3, GLM-5, Llama 3 / 4, GPT-OSS, Gemma 3 / 4, Mixtral, and the Qwen-VL and Gemma VL families, plus diffusion models via `vllm-omni`.

- **Kernels:** Ragged Paged Attention, dense and sparse MLA, Gated DeltaNet, fused MoE with SparseCore offload, FP8 / INT8 / MXFP4 quantized matmul.
- **Scale-out:** TP, DP, PP, EP, and prefill/decode context parallelism across single- and multi-host TPU slices, with disaggregated prefill/decode serving.
- **Standard vLLM features:** speculative decoding, prefix caching, multi-LoRA, structured output, multimodal inputs — all through the normal vLLM flags.

Full details: [Supported Models & Features](https://vllm-torchtpu.readthedocs.io/en/latest/user_guide/supported_models/) · [TPU v7x serving recipes](https://vllm-torchtpu.readthedocs.io/en/latest/deployment/v7x_recipes/) · [Kubernetes & GKE](https://vllm-torchtpu.readthedocs.io/en/latest/deployment/kubernetes/) · [Roadmap](https://vllm-torchtpu.readthedocs.io/en/latest/roadmap/)

## Building the image yourself

The Dockerfile lives in the repository under [`docker/`](https://github.com/vllm-project/vllm-torchtpu/tree/main/docker). The `prod` target is what `latest` and release tags are built from; `dev` is what `nightly` is built from.

```bash
git clone https://github.com/vllm-project/vllm-torchtpu.git
cd vllm-torchtpu
./docker/build_image.sh --target prod -t vllm-torchtpu:prod
```

See the [Docker guide](https://vllm-torchtpu.readthedocs.io/en/latest/getting_started/docker/) for the `ci` / `dev` / `prod` targets and build arguments.
