# Installation

`vllm-torchtpu` is a platform plugin for [vLLM](https://docs.vllm.ai/) that enables PyTorch-native LLM and multimodal serving on Google Cloud TPUs via `torch_tpu` and custom JAX/Pallas kernels.

---

## Interactive Installation Generator

Choose your preferred installation method below:

<div class="command-generator-container">
  <div class="cg-row">
    <div class="cg-label">1. Installation Method</div>
    <div class="cg-options">
      <button type="button" class="cg-btn active" data-group="method" data-val="uv_pip" aria-pressed="true">uv sync (Recommended)</button>
      <button type="button" class="cg-btn" data-group="method" data-val="source" aria-pressed="false">Editable Source Install</button>
      <button type="button" class="cg-btn" data-group="method" data-val="docker" aria-pressed="false">Docker Container</button>
    </div>
  </div>
  <div class="cg-row" id="docker-image-group" style="display: none;">
    <div class="cg-label">2. Container Type</div>
    <div class="cg-options">
      <button type="button" class="cg-btn active" data-group="docker_img" data-val="latest" aria-pressed="true">Base TPU Dev Image</button>
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

- **Python 3.12** (`>=3.12,<3.13`). Pre-built `tpu-sync-torch` and `torch_tpu` wheels target CPython 3.12.
- **GCP Artifact Registry Authentication**: `torch_tpu` wheels are pulled from Google Cloud Artifact Registry (`us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/`). Authenticate your host before installing:

```bash
gcloud auth application-default login
```

---

## Option 1: Install with `uv` (Recommended)

`pyproject.toml` configures the `torch-tpu-registry` index and pins compatible `torch`, `torch-tpu`, `jax`, `libtpu`, and `vllm` versions automatically.

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 2. Clone vllm-torchtpu and sync the environment (including the vllm group)
git clone https://github.com/vllm-project/vllm-torchtpu.git
cd vllm-torchtpu
uv sync --group vllm

# 3. Activate the virtual environment
source .venv/bin/activate
```

To install in editable mode with test and development dependencies:

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install --prerelease=allow -e . --group vllm --group test
```

---

## Option 2: Install with `pip`

If using standard `pip` (`pip >= 25.1` for `--group` support), install `keyring` and `keyrings.google-artifactregistry-auth` first so `pip` can authenticate with the `torch_tpu` Artifact Registry index:

```bash
python3.12 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip keyring keyrings.google-artifactregistry-auth

git clone https://github.com/vllm-project/vllm-torchtpu.git
cd vllm-torchtpu

pip install -e . --group vllm \
  --extra-index-url https://us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/
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
| `--group test` | Installs `pytest`, `pytest-mock`, `pytest-cov`, `parameterized`, and `vllm_test_utils`. |
| `--extra benchmarking` | Installs `lm-eval`, `datasets`, and the other benchmarking dependencies. |
| `--extra runai` | Installs `runai-model-streamer[gcs]`, needed for `--load-format runai_streamer` (streaming `gs://` checkpoints). |
| `--group evalscope` | Installs `evalscope` for multimodal evals. Install it in a separate environment: its dependencies downgrade `openai` and `filelock`. |
| `--extra omni` | Installs `vllm-omni` for multimodal diffusion models (e.g., FLUX, Wan2.2). |

---

## Verify Installation

Verify that vLLM detects the `vllm-torchtpu` platform plugin:

```bash
python -c "from vllm.platforms import current_platform; print(current_platform.device_name)"
```
