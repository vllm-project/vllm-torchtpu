<p align="center">
   <!-- This image will ONLY show up in GitHub's dark mode -->
  <img src="assets/torch_tpu_dark_mode_short.png#gh-dark-mode-only" alt="vLLM TorchTPU" style="width: 86%;">
    <!-- This image will ONLY show up in GitHub's light mode (and on other platforms) -->
  <img src="assets/torch_tpu_light_mode_short.png#gh-light-mode-only" alt="vLLM TorchTPU" style="width: 86%;">
</p>
<p align="center">
| <a href="https://github.com/vllm-project/vllm-torchtpu"><b>GitHub</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://discuss.vllm.ai/c/hardware-support/google-tpu-support/27"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

---

## About

vLLM TorchTPU is the integration of **TorchTPU** and **vLLM**. It is a vLLM platform plugin packaged as `vllm_torchtpu`, with TPU kernels and runtime code for TorchTPU, so PyTorch model definitions run performantly on Cloud TPU with the same vLLM user experience, telemetry, and interface.

## What are you trying to do today?

<div class="grid cards two-columns" markdown>

- :material-rocket-launch:{ .lg .middle } __I'm New__

    Provision a Cloud TPU, install `vllm-torchtpu`, and send your first request.

    [:octicons-arrow-right-24: Cloud TPU Setup](getting_started/tpu_setup.md)

    [:octicons-arrow-right-24: Installation](getting_started/installation.md)

    [:octicons-arrow-right-24: Quickstart](getting_started/quickstart.md)

- :material-speedometer:{ .lg .middle } __I Want to Serve & Scale__

    Explore supported models, 6D parallelism, P/D disaggregation, and TPU recipes.

    [:octicons-arrow-right-24: Supported Models & Features](user_guide/supported_models.md)

    [:octicons-arrow-right-24: Parallelism & Scaling](user_guide/parallelism.md)

    [:octicons-arrow-right-24: Disaggregated Serving (P/D)](user_guide/disaggregated_serving.md)

    [:octicons-arrow-right-24: TPU v7x Serving Recipes](deployment/v7x_recipes.md)

- :material-code-tags:{ .lg .middle } __I Want to Contribute__

    Contribute code, dive into inference examples, and follow the submission flow.

    [:octicons-arrow-right-24: Inference Examples](api_and_code_examples/offline_inference.md)

    [:octicons-arrow-right-24: Contributing](developers_guide/contributing.md)

    [:octicons-arrow-right-24: Testing & CI](developers_guide/testing_and_ci.md)

- :material-lightbulb-on-outline:{ .lg .middle } __I Want to Learn More__

    Learn the architecture, run benchmarks, and profile TPU kernels.

    [:octicons-arrow-right-24: Coding Guidance](developers_guide/coding_guidance.md)

    [:octicons-arrow-right-24: Benchmarking](developers_guide/benchmarking.md)

    [:octicons-arrow-right-24: Profiling](developers_guide/profiling/index.md)

    [:octicons-arrow-right-24: Environment Variables](reference/env_vars.md)

</div>

## Contribute

We're always looking for ways to partner with the community to accelerate vLLM TorchTPU development. If you're interested in contributing, check out the [Contributing guide](developers_guide/contributing.md) and [Issues](https://github.com/vllm-project/vllm-torchtpu/issues) to start.
