# Roadmap

This page is the public roadmap for **vLLM TorchTPU** (`vllm-torchtpu`): what the project is working on over the coming quarter, published so that the community can see what is coming, plan around it, and get involved.

!!! info "Last updated: October 19, 2026"
    Items are grouped by **target window**. Dates are planning targets rather than commitments and may shift as upstream dependencies, model releases, and community priorities evolve. This page is refreshed at least once per quarter.

    `vllm-torchtpu` shares the `torch_tpu` runtime, the Tokamax kernel library, and the TPU recipes with its sibling project `sglang-torchtpu`, so platform-level milestones below benefit both projects. SGLang-specific plans are tracked by that project.

For what is supported **today**, see [Supported Models & Features](user_guide/supported_models.md).

---

## Q4 2026

### October 2026

| Milestone | Area | Description |
| :--- | :--- | :--- |
| **vLLM TorchTPU Launch** | Core OSS Repositories | Official public release of `vllm-project/vllm-torchtpu` and the `google-pytorch/torch_tpu` runtime on GitHub, with TorchTPU as a PyTorch Foundation supported accelerator, backed by a contributor TPU pool (256 v6e / 32 v7x chips). |
| **Inferact Co-Maintainership** | Partner Ecosystem & Governance | Establish a community-led governance model by onboarding Inferact maintainers to `vllm-torchtpu`, delivering custom megakernels and SOTA price-performance on InferenceX & AgentX. |
| **Qwen 3.5 397B & Gemma 4 Recipes** | Reproducible Recipes | Publish reproducible Ironwood (TPU v7x) FP8 W8A8 inference recipes for Qwen 3.5 397B and Gemma 4 in [AI-Hypercomputer/tpu-recipes](https://github.com/AI-Hypercomputer/tpu-recipes). |
| **TPU Agent Suite (Experimental) & Ray** | Agentic Tooling & Orchestration | Launch the TPU Agent Suite (experimental release) and native Ray integration to streamline distributed orchestration and agentic developer workflows on Cloud TPU. |
| **100% Tokamax PyTorch Interfaces** | Kernel Ecosystem | Deliver production PyTorch interfaces for 100% of [Tokamax](https://github.com/openxla/tokamax) kernels, giving `vllm-torchtpu` a single kernel integration layer shared with the rest of the TorchTPU ecosystem (SGLang, TorchTitan). |
| **Multi-Host Disaggregation (xP-yD)** | Distributed Serving | Enable low-latency multi-host prefill/decode disaggregated serving ([llm-d](https://llm-d.ai)) for frontier open models to unblock massive-scale, long-context, and agentic workloads. |
| **TPU-Sync & Lustre KV-Cache Offloading** | Memory & Context Scaling | Integrate zero-copy TPU-Sync KV-cache connectors across host RAM and Google Cloud Managed Lustre, scaling context windows from 128K to 1M+ tokens with throughput gains. |

### November 2026

| Milestone | Area | Description |
| :--- | :--- | :--- |
| **24-Hour Day-0 Model Onboarding Pipeline** | Community Enablement | Deploy an automated CI pipeline to ingest, benchmark, and publish support matrices for newly released open-weight models within 24 hours of launch. |
| **Qwen 3.5, Kimi K3 & GLM 5.2/5.3 Benchmarks** | SOTA Performance | Complete performance hill-climbing and publish competitive AgentX load-tester submissions (including llm-d xP/yD runs) for Qwen 3.5, Kimi K3, and GLM 5.2 / 5.3 (1M context). |
| **Speculative Decoding & Agentic Traces** | Latency Acceleration | Roll out Speculative Decoding (including Multi-Token Prediction) and publish AgentX benchmark submissions for multi-host speculative decoding and agentic trace distributions. |
| **Native Diffusion Model Serving** | Multimodal Serving | Integrate native serving support for major video and image diffusion models (e.g., Wan 2.2, Lightrick2) directly into `vllm-torchtpu` via `vllm-omni`. |
| **Public Nightly CI/CD Support Matrix** | Developer Tooling | Launch a public, nightly-updated CI/CD support matrix tracking model execution status, quantization formats (BF16, FP8, MXFP4), serving features, and automated evaluation scores. |

### December 2026

| Milestone | Area | Description |
| :--- | :--- | :--- |
| **DeepSeek V4 Readiness** | Frontier Model Expansion | Deliver Day-0 functional support, kernel performance optimization, and official AgentX benchmark submissions for DeepSeek V4. |
| **Smart Caching, Routing & Memory Tiers** | Advanced Serving | Roll out prefix-aware, cache-aware, and session-aware routing alongside dynamic length bucketing, llm-d TPU memory-tier routing, and ecosystem-driven KV-cache backends. |

---

## How to Get Involved

- **Request a feature or model.** Open a [GitHub issue](https://github.com/vllm-project/vllm-torchtpu/issues/new/choose). Community demand is a direct input into how roadmap items are prioritized.
- **Contribute.** Many of the items above are open to community contribution. Start with the [Contributing guide](developers_guide/contributing.md) and the [good first issues](https://github.com/vllm-project/vllm-torchtpu/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).
- **Discuss.** Join the conversation on the [vLLM Forum](https://discuss.vllm.ai/c/hardware-support/google-tpu-support/27) or the [vLLM Developer Slack](https://slack.vllm.ai) (`#sig-tpu`).
