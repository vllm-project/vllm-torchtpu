# Installation

`vllm-torchtpu` is a platform plugin for [vLLM](https://docs.vllm.ai/) that enables PyTorch-native LLM and multimodal serving on Google Cloud TPUs via `torch_tpu` and custom JAX/Pallas kernels.

---

## Interactive Installation Generator

Choose your preferred installation method below:

<div class="command-generator-container">
  <div class="cg-row">
    <div class="cg-label">1. Installation Method</div>
    <div class="cg-options">
      <button type="button" class="cg-btn active" data-group="method" data-val="uv_pip" aria-pressed="true">uv (Recommended)</button>
      <button type="button" class="cg-btn" data-group="method" data-val="source" aria-pressed="false">Editable Source Install</button>
      <button type="button" class="cg-btn" data-group="method" data-val="docker" aria-pressed="false">Docker Container</button>
    </div>
  </div>
  <div class="cg-row" id="docker-image-group" style="display: none;">
    <div class="cg-label">2. Container Type</div>
    <div class="cg-options">
      <button type="button" class="cg-btn active" data-group="docker_img" data-val="latest" aria-pressed="true">Nightly Dev Image</button>
      <button type="button" class="cg-btn" data-group="docker_img" data-val="nightly" aria-pressed="false">Build from Dockerfile</button>
    </div>
  </div>
  <div class="cg-output-container">
    <div class="cg-instructions" id="cg-output-instructions"></div>
    <div class="highlight"><pre><code id="cg-output-command"></code></pre></div>
  </div>
</div>

---

## Prerequisites

- **Python 3.12** (`requires-python = ">=3.12,<3.13"` in `pyproject.toml`).
- **Access to the private Torch TPU registry**: `torch-tpu` is pulled from Google Cloud Artifact Registry (`us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/`). Log in to `gcloud` with an account that can read it (to request access, contact the Torch TPU team):

```bash
gcloud auth login
gcloud auth application-default login
```

!!! tip "Stale `~/.netrc` credentials"
    A `~/.netrc` entry for `us-python.pkg.dev` with an expired token is a common cause of authentication errors with `keyring`. Delete that block from `~/.netrc`.

!!! note "Migrating from the old `vllm_torchtpu` distribution"
    This repo is now distributed as `vllm-torchtpu-core` (the import package is still `vllm_torchtpu`). If your environment has the pre-rename `vllm_torchtpu` installed, run `pip uninstall -y vllm_torchtpu` (or `uv pip uninstall vllm_torchtpu`) first.

---

## Option 1: Install with `uv` (Recommended)

`pyproject.toml` pins `torch`, `torch-tpu`, `jax`, `libtpu`, and `vllm`, and uses the `torch-tpu-registry` index only for `torch-tpu`. Set `UV_TORCH_BACKEND=cpu` so `torch` resolves to the CPU build rather than the CUDA one.

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 2. Clone vllm-torchtpu and create a virtual environment
git clone https://github.com/vllm-project/vllm-torchtpu.git
cd vllm-torchtpu
uv venv --python 3.12
source .venv/bin/activate

# 3. Registry auth: username for the index, plus keyring for the token
export UV_INDEX_TORCH_TPU_REGISTRY_USERNAME="oauth2accesstoken"
uv tool install keyring --with keyrings.google-artifactregistry-auth

# 4. Install (--group vllm builds vLLM from source)
export UV_TORCH_BACKEND=cpu
VLLM_TARGET_DEVICE="empty" uv pip install --pre -e . --group vllm
```

To also install the test dependencies, add `--group test` to the last command.

---

## Option 2: Install with `pip`

If using standard `pip` (`pip >= 25.1` for `--group` support), install `keyring` and `keyrings.google-artifactregistry-auth` first so `pip` can authenticate with the Torch TPU registry:

```bash
python3.12 -m venv .venv --symlinks
source .venv/bin/activate

pip install --upgrade pip keyring keyrings.google-artifactregistry-auth

git clone https://github.com/vllm-project/vllm-torchtpu.git
cd vllm-torchtpu

export PIP_INDEX_URL="https://oauth2accesstoken@us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/"
VLLM_TARGET_DEVICE="empty" pip install --pre -e . --group vllm
```

---

## Option 3: Editable `vllm` Checkout (Cross-Repo Development)

By default, `--group vllm` builds and installs the pinned upstream `vllm` tag (`v0.29.0`) from GitHub. If you are developing changes across both `vllm` and `vllm-torchtpu`, replace the installed `vllm` package with a local editable checkout:

```bash
git clone --branch v0.29.0 --depth 1 https://github.com/vllm-project/vllm.git ../vllm
VLLM_TARGET_DEVICE=empty uv pip install -e ../vllm --no-deps
```

!!! tip "Why `VLLM_TARGET_DEVICE=empty`?"
    `vllm-torchtpu` registers itself through vLLM's out-of-tree platform plugin entry points (`vllm.platform_plugins` and `vllm.general_plugins`). Installing `vllm` with `VLLM_TARGET_DEVICE=empty --no-deps` skips CUDA C++ extension compilation while keeping all Python server and engine entrypoints editable.

---

## Optional Dependency Groups & Extras

| Flag | Purpose |
| :--- | :--- |
| `--group vllm` | Installs pinned upstream `vllm` (`v0.29.0`) from source. |
| `--group test` | Installs `pytest` (with `pytest-mock`, `pytest-timeout`, `pytest-cov`), `absl-py`, `parameterized`, and `vllm_test_utils`. |
| `--extra benchmarking` | Installs benchmarking and eval tools, including `lm-eval`, `evalscope`, `datasets`, and `runai-model-streamer[gcs]`. |
| `--extra omni` | Installs `vllm-omni` for multimodal diffusion models (e.g., FLUX). |

---

## Verify Installation

Verify that vLLM detects the `vllm-torchtpu` platform plugin:

```bash
python -c "from vllm.platforms import current_platform; print(current_platform.device_name)"
```
