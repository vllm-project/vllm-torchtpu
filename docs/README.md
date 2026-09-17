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

<div class="grid cards three-columns" markdown>

- :material-rocket-launch:{ .lg .middle } __I'm New__

    Install the plugin, start a server, and send your first request.

    [:octicons-arrow-right-24: Installation and Quickstart](https://github.com/vllm-project/vllm-torchtpu#readme)

- :material-speedometer:{ .lg .middle } __I Want to Tune__

    Cut compile times and capture TPU traces to find where time goes.

    [:octicons-arrow-right-24: TorchTPU Compilation Cache](torch_tpu_compilation.md)

    [:octicons-arrow-right-24: Profiling](developers_guide/profiling.md)

- :material-code-tags:{ .lg .middle } __I Want to Build__

    Contribute code and follow the submission flow.

    [:octicons-arrow-right-24: Contributing](developers_guide/contributing.md)

</div>

## Contribute

We're always looking for ways to partner with the community to accelerate vLLM TorchTPU development. If you're interested in contributing, check out the [Contributing guide](developers_guide/contributing.md) and [Issues](https://github.com/vllm-project/vllm-torchtpu/issues) to start.
