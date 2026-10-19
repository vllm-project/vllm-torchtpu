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

> [!TIP]
> **Automatic Sign-Off in IDEs**:
>
> * **VS Code**: Enable `Git: Always Sign Off` (`git.alwaysSignOff: true`) in your settings.
> * **PyCharm**: In the Commit window, enable `Sign-off commit`.

If you forget to sign off a commit, amend it and force-push:

```bash
git commit --amend --signoff --no-edit
git push origin HEAD --force-with-lease
```

### 2. Clone and Install Dependencies

Clone the repository:

```bash
git clone git@github.com:vllm-project/vllm-torchtpu.git && cd vllm-torchtpu
```

`VLLM_TARGET_DEVICE="empty"` is recognized by vLLM's build process to build `vllm-torchtpu` as an out-of-tree platform plugin.

`torch-tpu` and the other dependencies come from PyPI; `torch` and `torchvision` come from the PyTorch CPU index (`https://download.pytorch.org/whl/cpu`). No authentication is required.

#### Option A: Using `uv` (Recommended)

```bash
uv venv --python 3.12 ~/uv_venv
source ~/uv_venv/bin/activate

# Use CPU Torch when building vLLM's editable TPU package.
export UV_TORCH_BACKEND=cpu

# Install vllm-torchtpu and dependencies in editable mode (--group vllm pulls vLLM from source)
VLLM_TARGET_DEVICE="empty" uv pip install --pre -e . --group vllm
```

#### Option B: Using `pip`

```bash
python3.12 -m venv ~/pip_venv --symlinks
source ~/pip_venv/bin/activate
pip install --upgrade pip

# Install vllm-torchtpu and dependencies in editable mode (--group vllm pulls vLLM from source;
# needs pip >= 25.1). torch and torchvision come from the PyTorch CPU index.
VLLM_TARGET_DEVICE="empty" pip install --pre -e . --group vllm \
  --extra-index-url https://download.pytorch.org/whl/cpu
```

### 3. [Optional] Developing with vLLM Source
If you are developing or debugging vLLM core in tandem with `vllm-torchtpu`:

```bash
# Clone matching vLLM version (e.g. v0.31.0)
git clone --depth 1 --branch v0.31.0 https://github.com/vllm-project/vllm.git ../vllm

# Install in editable mode without overwriting dependencies
VLLM_TARGET_DEVICE="empty" uv pip install --no-deps -e ../vllm
```

### 4. Pre-commit Hooks
Make sure to activate your configured virtual environment before installing and running `pre-commit`:

```bash
# Install pre-commit hooks in the active environment
pre-commit install --hook-type pre-commit --hook-type commit-msg

# Run manually across all files:
pre-commit run --all-files
```

The pre-commit `pyrefly` hook type-checks without third-party packages. CI also runs a full-environment check that resolves torch, jax and vllm; run it locally with `bash tools/typecheck_full.sh`. [Type checking](https://github.com/vllm-project/vllm-torchtpu/blob/main/docs/developers_guide/type_checking.md) covers both checks, what the local run needs, and how to fix new errors.

---

## 3. Issues, RFCs, and Duplicate Checks

### Check for Duplicate Work
Before opening a new issue or starting work on a feature, search existing issues and open pull requests:

```bash
gh issue view <issue_number> --comments
gh pr list --state open --search "<short area keywords>"
```

If an open PR already addresses the issue, please collaborate on the existing thread instead of opening a duplicate.

### Proposing Major Changes (>500 LOC RFC Requirement)
Following upstream vLLM conventions, any PR introducing major architectural changes or exceeding **500 lines of core logic** (excluding tests, configs, and autogenerated files) requires an initial GitHub Issue (RFC) to discuss technical design and justification before code submission. PRs opened without an approved RFC may be labeled `rfc-required` and placed on hold.

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
For running development and unit tests inside containerized environments using Docker, refer to the [Docker Developer Guide](./docs/getting_started/docker.md).

### Physical TPU Tests & Buildkite CI (The `ready` Label Gate)
Tests requiring physical TPU hardware run in Buildkite CI.

* **Security & Privilege Isolation**: Because CI runners execute in environments with privileged GCP access and tokens, untrusted code from external forks cannot be run automatically. A repository collaborator or maintainer must review the PR for safety before attaching the `ready` label.
* **Resource Contention Mitigation**: Prevents scarce TPU accelerator runners from being held by work-in-progress drafts.
* **Triggering CI**: Apply the `ready` label to launch the test suite:

  ```bash
  gh pr edit <PR_NUMBER> --add-label ready
  ```

* **New commits**: On a PR from a branch in this repository, every push runs TPU CI again while the label is on. On a PR from a fork, TPU CI runs only when the label is added: each push, including a rebase or a merge of `main`, takes it off, and a code owner adds it again after looking at the new commits.

* **On-call fixes**: A PR that fixes a broken main or nightly build can also take the `oncall-fix` label. Its TPU jobs then run ahead of every other build, including builds of `main`. Adding the label starts a new build and cancels the queued one. Use it only for on-call fixes.

---

## 6. Pull Request & Review Lifecycle

### Submitting a Pull Request
When opening a pull request, please select the appropriate category prefix from the PR template (`[Bugfix]`, `[Core]`, `[Kernel]`, `[Model]`, `[Frontend]`, `[CI/Build]`, `[Doc]`, `[Misc]`) and complete the description and test plan checklists.

### Reviewer Routing
GitHub Actions automatically matches changed files against `.github/CODEOWNERS` and requests reviews from the appropriate area owners upon PR creation.

---

## 7. Policy on AI-Assisted Code

We welcome the responsible use of AI tools to enhance engineering productivity, subject to upstream vLLM policies:

1. **Full Submitter Accountability**: The human contributor is 100% responsible for all submitted code. You must thoroughly review, understand, and validate every changed line. "An AI wrote it" is not an acceptable response during code review. If a reviewer asks why an operation is written a certain way, you must be able to explain and defend the implementation.
2. **No Low-Value "Busywork" Submissions**: Pull requests containing isolated, trivial edits (such as a single typo fix, solitary whitespace adjustments, or one-off formatting cleanups) waste reviewer bandwidth and will be closed. Mechanical cleanups must be bundled with substantive engineering contributions.
3. **Transparent Disclosure & Verification**: Always state in your pull request description when AI tools provided non-trivial assistance, and include the exact test commands and passing verification output.
4. **Commit Attribution**: Include standard commit trailers acknowledging AI assistance alongside your DCO sign-off:

```text
Signed-off-by: Your Name <your.email@example.com>
Co-authored-by: Claude <claude@anthropic.com>
Co-authored-by: GitHub Copilot <copilot@github.com>
Co-authored-by: Antigravity <antigravity@google.com>
```

---

## 8. Community & Communication Channels

For real-time coordination, technical questions, or getting reviewer attention across the community:
* **Slack**: <!-- TODO: insert Slack channel link once created --> A dedicated community Slack channel will be announced soon.
* **GitHub Issues**: Use [GitHub Issues](https://github.com/vllm-project/vllm-torchtpu/issues) for bug reports, model requests, and technical RFC proposals.

---

## 9. Reporting Security Vulnerabilities

If you discover a potential security vulnerability in vllm-torchtpu, please report it privately rather than opening a public issue.

* **How to Report**: We follow upstream [vLLM's Security Policy](https://github.com/vllm-project/vllm/blob/main/SECURITY.md). Please report security issues privately using [vLLM's vulnerability submission form](https://github.com/vllm-project/vllm/security/advisories/new). Once repository-level advisories are configured for vllm-torchtpu, reports can also be submitted directly via our repository's Security tab.
* **Scope**: Crashes caused by invalid inputs, excessive memory consumption, or numerical divergence from untrusted models are normal bugs rather than security vulnerabilities. Running untrusted model code is equivalent to running arbitrary Python code.

---

## 10. Building the Documentation

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

## 11. Thank You!
Thank you for taking the time to read these guidelines and for your interest in contributing to vllm-torchtpu. All of your contributions help make vllm-torchtpu a better tool and community for everyone.
