# Contributing to vllm-torchtpu

Thank you for your interest in contributing to vllm-torchtpu! Our community is open to everyone and welcomes all kinds of contributions, no matter how small or large. There are several ways you can contribute to the project:

* Identify and report issues or bugs.
* Request or add support for a new model.
* Suggest or implement new performance optimizations or features.
* Improve documentation, examples, or tutorials.

We also believe in the power of community support: answering questions, offering PR reviews, and assisting fellow contributors are highly valued contributions.

---

## 1. Philosophy & Review Expectations

* **Small, Focused Pull Requests**: We strongly prioritize concise, well-tested pull requests that address a single clear objective over sprawling, multi-purpose rewrites. Small diffs are reviewed, tested, and merged significantly faster.
* **Value of Quality Bug Reports**: Submitting a clear bug report with a Minimal Reproducible Example (MRE) is a first-class contribution. Isolating the issue to a small reproduction script saves maintainer time and accelerates fixes.
* **Main Branch Stability (Revert-on-Breakage)**: We maintain a strict revert-on-breakage policy to keep `main` dependable for all developers. If a merged PR introduces regressions in nightly benchmarks or integration suites, it will be promptly reverted rather than fixed forward on `main`. The author is responsible for debugging locally before resubmitting.

---

## 2. Setting Up Your Development Environment

### 1. Developer Certificate of Origin (DCO)
Contributions to this project must be accompanied by a Developer Certificate of Origin sign-off. Commits must include a `Signed-off-by:` header certifying agreement with the DCO.

Create signed commits using `git commit -s`, which automatically appends the trailer:

```text
Signed-off-by: Your Name <your.email@example.com>
```

> [!TIP] Automatic Sign-Off in IDEs
> * **VS Code**: Enable `Git: Always Sign Off` (`git.alwaysSignOff: true`) in your settings.
> * **PyCharm**: In the Commit window, enable `Sign-off commit`.

If you forget to sign off a commit, amend it and force-push:

```bash
git commit --amend --signoff --no-edit
git push origin HEAD --force-with-lease
```

### 2. Google Cloud Authentication
<!-- TODO: Once torch_tpu and vllm-torchtpu are open-sourced, torch_tpu will be installable directly from PyPI without GCP authentication. -->
We authenticate with Google Cloud to access the Torch TPU Virtual Registry (`https://us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/`), which contains `torch-tpu`, `torch`, and other TPU dependencies.

Check your active account:

```bash
gcloud auth list
```

If needed, log in:

```bash
gcloud auth login
gcloud auth application-default login
```

> [!TIP] Troubleshooting Stale Credentials
> A common cause of authentication errors when using `keyring` with `uv` is an existing `~/.netrc` file containing stale credentials. Open `~/.netrc` and delete any expired block associated with `us-python.pkg.dev`.

### 3. Clone and Install Dependencies

Clone the repository:

```bash
git clone git@github.com:vllm-project/vllm-torchtpu.git && cd vllm-torchtpu
```

`VLLM_TARGET_DEVICE="empty"` is recognized by vLLM's build process to build `vllm-torchtpu` as an out-of-tree platform plugin.

#### Option A: Using `uv` (Recommended)

```bash
uv venv --python 3.12 ~/uv_venv
source ~/uv_venv/bin/activate

# Configure index authentication
export UV_INDEX_TORCH_TPU_REGISTRY_USERNAME="oauth2accesstoken"
uv tool install keyring --with keyrings.google-artifactregistry-auth
export UV_TORCH_BACKEND=cpu

# Install vllm-torchtpu and dependencies in editable mode
VLLM_TARGET_DEVICE="empty" uv pip install --pre -e .
```

#### Option B: Using `pip`

```bash
python3.12 -m venv ~/pip_venv --symlinks
source ~/pip_venv/bin/activate
pip install --upgrade pip

# Install keyring plugin for persistent auth
pip install keyring keyrings.google-artifactregistry-auth
# TODO: Replace with the public index once torch_tpu is open-sourced
export PIP_INDEX_URL="https://oauth2accesstoken@us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/"

# Install vllm-torchtpu and dependencies in editable mode
VLLM_TARGET_DEVICE="empty" pip install --pre -e .
```

### 4. [Optional] Developing with vLLM Source
If you are developing or debugging vLLM core in tandem with `vllm-torchtpu`:

```bash
# Clone matching vLLM version (e.g. v0.29.0)
git clone --depth 1 --branch v0.29.0 https://github.com/vllm-project/vllm.git ../vllm

# Install in editable mode without overwriting dependencies
VLLM_TARGET_DEVICE="empty" uv pip install --no-deps -e ../vllm
```

### 5. Pre-commit Hooks
Make sure to activate your configured virtual environment before installing and running `pre-commit`:

```bash
# Install pre-commit hooks in the active environment
pre-commit install --hook-type pre-commit --hook-type commit-msg

# Run manually across all files:
pre-commit run --all-files
```

The pre-commit `pyrefly` hook type-checks without third-party packages. CI also runs a full-environment check that resolves torch, jax and vllm; run it locally with `bash tools/typecheck_full.sh`. [Type checking](https://github.com/vllm-project/vllm-torchtpu/blob/main/docs/developers_guide/type_checking.md) covers both checks, what the local run needs, and how to fix new errors.

---

## 3. Issues and Duplicate Checks

### Check for Duplicate Work
Before opening a new issue or starting work on a feature, search existing issues and open pull requests:

```bash
gh issue view <issue_number> --comments
gh pr list --state open --search "<short area keywords>"
```

If an open PR already addresses the issue, please collaborate on the existing thread instead of opening a duplicate.

---

## 4. Codebase Architecture

The project source is organized under `src/vllm_torchtpu/`:

* `runner/`: TPU Model Runner, KV cache allocation pools, PagedAttention block tables, and execution loop.
* `layers/`:
  * `core/`: JAX-side numerical, attention metadata, and layout logic.
  * `adapter/`: Torch/vLLM-facing adapter layer (quantization methods, custom op bridges, attention backends).
* `kernels/`: JAX and Pallas custom hardware kernels for TPU.
* `models/`: Model implementations and architectures supported on TPU (e.g., Llama, Qwen, DeepSeek).
* `spec_decode/`: Speculative decoding engines (Draft models, Jump decoding, GaaD).
* `offload/`: KV cache offload subsystems (Raiden store connector, block-major memory layouts).
* `compilation/`: TorchTPU / XLA compilation pipeline, graph optimizations, and persistent compilation caches.
* `distributed/`: Tensor Parallel (TP) and Pipeline Parallel (PP) communication primitives for TPU.
* `platforms/`: Hardware platform detection, device initialization, and platform patch registry.
* `tests/`: Component unit tests and integration tests.

---

## 5. Development & Testing Guidelines

### Local Testing via Docker
For running development and unit tests inside containerized environments using Docker, refer to the [Docker Developer Guide](docs/docker.md).

### Physical TPU Tests & Buildkite CI (The `ready` Label Gate)
Tests requiring physical TPU hardware run in Buildkite CI.

* **Security & Privilege Isolation**: Because CI runners execute in environments with privileged GCP access and tokens, untrusted code from external forks cannot be run automatically. A repository collaborator or maintainer must review the PR for safety before attaching the `ready` label.
* **Resource Contention Mitigation**: Prevents scarce TPU accelerator runners from being held by work-in-progress drafts.
* **Triggering CI**: Apply the `ready` label to launch the test suite:

  ```bash
  gh pr edit <PR_NUMBER> --add-label ready
  ```

---

## 6. Pull Request & Review Lifecycle

### Submitting a Pull Request
When opening a pull request, please select the appropriate category prefix from the PR template (`[Bugfix]`, `[Core]`, `[Kernel]`, `[Model]`, `[Frontend]`, `[CI/Build]`, `[Doc]`, `[Misc]`) and complete the description and test plan checklists.

### Reviewer Routing
GitHub Actions automatically matches changed files against `.github/CODEOWNERS` and requests reviews from the appropriate area owners upon PR creation.

---

## 7. Building the Documentation

The documentation site is built with [MkDocs](https://www.mkdocs.org/) and the Material theme from the sources in `docs/`. Read the Docs builds it from `mkdocs.yml` on every push. To preview it locally:

```bash
# From the repository root, inside your virtual environment
pip install -r docs/requirements.txt

# Serve with live reload at http://127.0.0.1:8000
mkdocs serve

# Or build once with the same strict checks Read the Docs uses
mkdocs build --strict
```

Pages are listed in the `nav` section of `mkdocs.yml`. Because the build runs in strict mode, every page in `nav` must exist and every relative link must resolve to another page in `docs/`. Link to source files with full GitHub URLs.

---

## 8. Thank You!
Thank you for taking the time to read these guidelines and for your interest in contributing to vllm-torchtpu. All of your contributions help make vllm-torchtpu a better tool and community for everyone.
