<p align="center">
   <!-- This image will ONLY show up in GitHub's dark mode -->
  <img src="docs/assets/torch_tpu_dark_mode_short.png#gh-dark-mode-only" alt="vLLM TorchTPU" style="width: 86%;">
    <!-- This image will ONLY show up in GitHub's light mode (and on other platforms) -->
  <img src="docs/assets/torch_tpu_light_mode_short.png#gh-light-mode-only" alt="vLLM TorchTPU" style="width: 86%;">
</p>

<p align="center">
| <a href="docs/README.md"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://discuss.vllm.ai/c/hardware-support/google-tpu-support/27"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> (#sig-tpu) |
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
| TPU v6e / v5e / v5p | Experimental — not covered by CI |

See [Cloud TPU Setup](docs/getting_started/tpu_setup.md) for project setup, quota, and provisioning commands.

<br>

## Get started

### Install

`vllm-torchtpu` requires **Python 3.12** on a Cloud TPU VM. Install the plugin and a matching vLLM with a single command:

```bash
pip install \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  vllm-torchtpu
```

For `uv`, editable source installs, cross-repo development against a local `vllm` checkout, and Docker images, see the full [Installation guide](docs/getting_started/installation.md) and [Docker guide](docs/getting_started/docker.md).

### Serve a model

Start an OpenAI-compatible server exactly as you would on any other vLLM platform:

```bash
vllm serve meta-llama/Llama-3.2-1B-Instruct \
  --tensor-parallel-size 1 \
  --max-model-len 4096
```

Then send a request:

A stage may instead span a prefill context-parallel group
(`--prefill-context-parallel-size`), which splits each chunk's attention over
the stage's cores; MoE models then need `--enable-expert-parallel`. This needs
a torch_tpu that compiles a stage's kernels and collectives for the stage's
devices only (the 2026-09-28 build of google-pytorch/torch_tpu#4387); on a
torch_tpu that builds world-size meshes only, the platform refuses such stages.
A two-core stage must not be the two cores of one chip: set
`VLLM_TPU_DEVICE_ORDER` to the device each local rank binds, e.g.
`0,2,1,3,4,6,5,7` for four stages of two cores on one host.

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
> The first startup compiles TPU graphs for the configured shapes and can take several minutes. Set `VLLM_XLA_CACHE_PATH` to persist compiled artifacts across restarts — see [Compilation & Warmup Cache](docs/user_guide/compilation_cache.md).

The [Quickstart](docs/getting_started/quickstart.md) covers offline batch inference with `vllm.LLM`, multi-device serving with TP/EP, and multimodal examples.

<br>

## Supported models and features

`vllm-torchtpu` can serve most PyTorch models supported by vLLM, with first-class TPU kernel, quantization, and parallelism support for Qwen3 / Qwen3.5, DeepSeek-V3 / R1 / V4, Kimi-K2.5 / K3, GLM-5, Llama 3 / 4, GPT-OSS, Gemma 3 / 4, Mixtral, Qwen-VL and Gemma VL families, and diffusion models via `vllm-omni`.

- **[Supported Models & Features](docs/user_guide/supported_models.md)** — validated model families, feature status, and quantization formats.
- **[Nightly support matrices](support_matrices/nightly/)** — TPU v7x unit, correctness, and benchmark status, updated automatically from nightly CI.

<br>

## Recipes and deployment

- [TPU v7x (Ironwood) serving recipes](docs/deployment/v7x_recipes.md)
- [TPU v6e (Trillium) serving recipes](docs/deployment/v6e_recipes.md)
- [Kubernetes deployment](docs/deployment/kubernetes.md)
- [Multi-host serving](docs/user_guide/multihost.md) · [Parallelism & scaling](docs/user_guide/parallelism.md) · [Disaggregated serving (P/D)](docs/user_guide/disaggregated_serving.md)
- [Quantization](docs/user_guide/quantization.md) · [Speculative decoding](docs/user_guide/speculative_decoding.md)

<br>

## 🤝 Contribute

We welcome contributions of all kinds — bug reports, model and feature requests, documentation fixes, and code.

- **🐞 Report bugs or request features:** open a [new issue](https://github.com/vllm-project/vllm-torchtpu/issues/new/choose).
- **👀 Review pull requests:** help us keep quality high by reviewing [open PRs](https://github.com/vllm-project/vllm-torchtpu/pulls).
- **💻 Write code:** start with the **[Contributing Guide](CONTRIBUTING.md)**, which covers environment setup, DCO sign-off, the `ready`-label CI gate, and the PR lifecycle. [Good first issues](https://github.com/vllm-project/vllm-torchtpu/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) are a great place to begin.

Developer references: [Testing & CI](docs/developers_guide/testing_and_ci.md) · [Profiling](docs/developers_guide/profiling/index.md) · [Benchmarking](docs/developers_guide/benchmarking.md) · [Coding guidance](docs/developers_guide/coding_guidance.md) · [Optimization best practices](docs/developers_guide/optimization_best_practices.md)

<br>

## 💬 Contact us

- For technical questions and feature requests, open a [GitHub issue](https://github.com/vllm-project/vllm-torchtpu/issues).
- For discussion with fellow users, use the [TPU support topic on the vLLM Forum](https://discuss.vllm.ai/c/hardware-support/google-tpu-support/27).
- For coordinating contributions and development, join the [vLLM Developer Slack](https://slack.vllm.ai) (`#sig-tpu`).
