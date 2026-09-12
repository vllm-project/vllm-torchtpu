# Contributing to TorchTPU-vLLM

> [!IMPORTANT] Pre-Public Development Governance
> During the pre-public phase (July 2026 – Public Launch), we are operating with specialized repository rules due to missing automated GitHub branch protections. Please refer to [PRE_PUBLIC_DEV_GUIDE.md](docs/PRE_PUBLIC_DEV_GUIDE.md) for standard submission flows, merge checklists, and mandatory guidelines.

Thank you for your interest in contributing to TorchTPU-vLLM! Our community is open to everyone and welcomes all kinds of contributions, no matter how small or large. There are several ways you can contribute to the project:

* Identify and report any issues or bugs.
* Request or add support for a new model.
* Suggest or implement new features.
* Improve documentation or contribute a how-to guide.

We also believe in the power of community support; thus, answering queries, offering PR reviews, and assisting others are also highly regarded and beneficial contributions.

Finally, one of the most impactful ways to support us is by raising awareness about TorchTPU-vLLM. Talk about it in your blog posts and highlight how it's driving your projects. Express your support on social media if you're using TorchTPU-vLLM, or simply offer your appreciation by starring our repository.

## Getting Started
We recommend filtering on the “Good First Issue” tag in the [Issues](https://github.com/vllm-project/vllm-torchtpu/issues) section of GitHub if it's your first time contributing.

## Issues
If you encounter a bug or have a feature request, please search [existing issues](https://github.com/vllm-project/vllm-torchtpu/issues) first to see if it has already been reported. If not, please [file a new issue](https://github.com/vllm-project/vllm-torchtpu/issues/new/choose), providing as much relevant information as possible.

## Directory Structure
We choose to follow a similar directory structure as vLLM:
* `src/vllm_torchtpu/layers/`:
  * `common` contains shared layers and building blocks used across implementations
  * `vllm` contains custom layer adaptations and wrappers for vLLM models
* `src/vllm_torchtpu/models/`:
  * `vllm` contains model implementations and architectures supported on TPU

## Finding Reviewers
Not sure who to ask for a review? `scripts/find_owners.py` matches your changed files against [.github/CODEOWNERS](.github/CODEOWNERS) and prints the owners of each area you touched, plus the smallest set of reviewers that covers all of them.

```bash
# Owners of the changes on your current branch
scripts/find_owners.py

# Owners of an existing PR, or of specific paths
scripts/find_owners.py --pr 123
scripts/find_owners.py src/vllm_torchtpu/kernels/ docs/index.md

# Request their review (uses the current branch's PR, or --pr 123)
scripts/find_owners.py --request
```

## Testing
When checking in a new feature, we expect that you add relevant unit tests as well as CI tests. You can read more about the latter [here](https://github.com/vllm-project/vllm-torchtpu/tree/main/.buildkite#adding-a-new-feature-to-ci).

## Setting up linting, formatting, and static type checking

Make sure to activate the virtual environment created during [installation](README.md#2-install-vllm-torchtpu-and-dependencies) before installing and running `pre-commit`. Running `pre-commit` from a global installation or outside the virtual environment can cause hook version mismatches and missing dependency errors.

```bash
# Activate your virtual environment
source ~/uv_venv/bin/activate  # Or `source ~/pip_venv/bin/activate` / `source .venv/bin/activate`

# Install pre-commit in the active virtual environment
pip install pre-commit

# Linting, formatting and static type checking
pre-commit install --hook-type pre-commit --hook-type commit-msg

# You can manually run pre-commit with
pre-commit run --all-files
```

## Thank You!
We wanted to thank you for taking the time to read these guidelines and for your interest in contributing to TorchTPU-vLLM. All of your contributions help make TorchTPU-vLLM a better tool and community for everyone.
