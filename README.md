<p align="center">
   <!-- This image will ONLY show up in GitHub's dark mode -->
  <img src="docs/assets/torch_tpu_dark_mode_short.png#gh-dark-mode-only" alt="vLLM TorchTPU" style="width: 86%;">
    <!-- This image will ONLY show up in GitHub's light mode (and on other platforms) -->
  <img src="docs/assets/torch_tpu_light_mode_short.png#gh-light-mode-only" alt="vLLM TorchTPU" style="width: 86%;">
</p>

<p align="center">
| <a href="https://vllm-torchtpu.readthedocs.io/en/latest/"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://discuss.vllm.ai/c/hardware-support/google-tpu-support/27"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> (#sig-tpu) |
</p>

---

## About

**vLLM TorchTPU** is the PyTorch-native TPU backend for [vLLM](https://github.com/vllm-project/vllm). It is an out-of-tree vLLM platform plugin, packaged as `vllm-torchtpu`, that runs vLLM's standard PyTorch model definitions on Google Cloud TPUs through **TorchTPU** (`torch_tpu`, via `torch.compile(backend="tpu")`) and a library of custom JAX/Pallas kernels for attention, Mixture-of-Experts, and quantized matmul.

The goal is simple: **the same vLLM you already use — same CLI, same OpenAI-compatible server, same model definitions — running at TPU speed with no model code changes.**

- **PyTorch first.** Models run through vLLM's native PyTorch runner; no re-implementation in another framework is required to serve on TPU.
- **TPU-tuned kernels.** Ragged Paged Attention, dense and sparse MLA, Gated DeltaNet, fused MoE (including SparseCore offload), and FP8/INT8/MXFP4 quantized matmuls written in Pallas.
- **Scale-out built in.** TP, DP, PP, EP, and prefill/decode context parallelism compose across single-host and multi-host TPU slices, with disaggregated prefill/decode serving.
- **Standard vLLM experience.** Speculative decoding, prefix caching, multi-LoRA, structured output, and multimodal inputs work through the normal vLLM flags.

> [!NOTE]
> **Looking for the JAX backend?** The original TPU backend for vLLM, [`tpu-inference`](https://github.com/vllm-project/tpu-inference), has been rebranded as **vLLM JAX**. It remains available for serving pure JAX model implementations on TPU. `vllm-torchtpu` is the recommended path for serving PyTorch model definitions on TPU.

<br>

## Supported hardware

| TPU generation | Status |
| :--- | :--- |
| **TPU v7x (Ironwood)** | Recommended — validated in CI on every pull request and nightly |
| **TPU v6e (Trillium)** | Supported — not covered by CI |
| TPU v5e / v5p | Experimental — not covered by CI |

See [Cloud TPU Setup](https://vllm-torchtpu.readthedocs.io/en/latest/getting_started/tpu_setup/) for project setup, quota, and provisioning commands.

<br>

## Get started

### Install

`vllm-torchtpu` requires **Python 3.12** on a Cloud TPU VM. Install [`vllm-torchtpu` from PyPI](https://pypi.org/project/vllm-torchtpu/) and a matching vLLM with a single command:

```bash
pip install \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  vllm-torchtpu
```

Or pull the pre-built container from [Docker Hub (`vllm/vllm-torchtpu`)](https://hub.docker.com/r/vllm/vllm-torchtpu/tags):

```bash
docker pull vllm/vllm-torchtpu:latest
```

For `uv`, editable source installs, cross-repo development against a local `vllm` checkout, and Docker images, see the full [Installation guide](https://vllm-torchtpu.readthedocs.io/en/latest/getting_started/installation/) and [Docker guide](https://vllm-torchtpu.readthedocs.io/en/latest/getting_started/docker/).

### Serve a model

Start an OpenAI-compatible server exactly as you would on any other vLLM platform:

```bash
vllm serve meta-llama/Llama-3.2-1B-Instruct \
  --tensor-parallel-size 1 \
  --max-model-len 4096
```

Then send a request:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [{"role": "user", "content": "Explain how TPUs accelerate matrix multiplication."}],
    "max_tokens": 128
  }'
```

> [!TIP]
> The first startup compiles TPU graphs for the configured shapes and can take several minutes. Set `VLLM_XLA_CACHE_PATH` to persist compiled artifacts across restarts — see [Compilation & Warmup Cache](https://vllm-torchtpu.readthedocs.io/en/latest/user_guide/compilation_cache/).

The [Quickstart](https://vllm-torchtpu.readthedocs.io/en/latest/getting_started/quickstart/) covers offline batch inference with `vllm.LLM`, multi-device serving with TP/EP, and multimodal examples.

<br>

## Supported models and features

`vllm-torchtpu` can serve most PyTorch models supported by vLLM, with first-class TPU kernel, quantization, and parallelism support for Qwen3 / Qwen3.5, DeepSeek-V3 / R1 / V4, Kimi-K2.5 / K3, GLM-5, Llama 3 / 4, GPT-OSS, Gemma 3 / 4, Mixtral, Qwen-VL and Gemma VL families, and diffusion models via `vllm-omni`.

- **[Supported Models & Features](https://vllm-torchtpu.readthedocs.io/en/latest/user_guide/supported_models/)** — validated model families, feature status, and quantization formats.
- **[Nightly support matrices](support_matrices/nightly/)** — TPU v7x unit, correctness, and benchmark status, updated automatically from nightly CI.

<br>

## Recipes and deployment

- [TPU v7x (Ironwood) serving recipes](https://vllm-torchtpu.readthedocs.io/en/latest/deployment/v7x_recipes/)
- [TPU v6e (Trillium) serving recipes](https://vllm-torchtpu.readthedocs.io/en/latest/deployment/v6e_recipes/)
- [Kubernetes deployment](https://vllm-torchtpu.readthedocs.io/en/latest/deployment/kubernetes/)
- [Multi-host serving](https://vllm-torchtpu.readthedocs.io/en/latest/user_guide/multihost/) · [Parallelism & scaling](https://vllm-torchtpu.readthedocs.io/en/latest/user_guide/parallelism/) · [Disaggregated serving (P/D)](https://vllm-torchtpu.readthedocs.io/en/latest/user_guide/disaggregated_serving/)
- [Quantization](https://vllm-torchtpu.readthedocs.io/en/latest/user_guide/quantization/) · [Speculative decoding](https://vllm-torchtpu.readthedocs.io/en/latest/user_guide/speculative_decoding/)

<br>

## 🤝 Contribute

We welcome contributions of all kinds — bug reports, model and feature requests, documentation fixes, and code.

- **🐞 Report bugs or request features:** open a [new issue](https://github.com/vllm-project/vllm-torchtpu/issues/new/choose).
- **👀 Review pull requests:** help us keep quality high by reviewing [open PRs](https://github.com/vllm-project/vllm-torchtpu/pulls).
- **💻 Write code:** start with the **[Contributing Guide](CONTRIBUTING.md)**, which covers environment setup, DCO sign-off, the `ready`-label CI gate, and the PR lifecycle. [Good first issues](https://github.com/vllm-project/vllm-torchtpu/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) are a great place to begin.

Developer references: [Testing & CI](https://vllm-torchtpu.readthedocs.io/en/latest/developers_guide/testing_and_ci/) · [Profiling](https://vllm-torchtpu.readthedocs.io/en/latest/developers_guide/profiling/) · [Benchmarking](https://vllm-torchtpu.readthedocs.io/en/latest/developers_guide/benchmarking/) · [Coding guidance](https://vllm-torchtpu.readthedocs.io/en/latest/developers_guide/coding_guidance/) · [Optimization best practices](https://vllm-torchtpu.readthedocs.io/en/latest/developers_guide/optimization_best_practices/)

<br>

## 💬 Contact us

- For technical questions and feature requests, open a [GitHub issue](https://github.com/vllm-project/vllm-torchtpu/issues).
- For discussion with fellow users, use the [TPU support topic on the vLLM Forum](https://discuss.vllm.ai/c/hardware-support/google-tpu-support/27).
- For coordinating contributions and development, join the [vLLM Developer Slack](https://slack.vllm.ai) (`#sig-tpu`).
