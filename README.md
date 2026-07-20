<h1 align="center">vLLM TPU</h1>

This repository contains the integration of **TorchTPU** and **vLLM**. It is a vLLM platform plugin packaged as `vllm_torchtpu`, with TPU kernels and runtime code for TorchTPU.

> [!IMPORTANT] Pre-Public Development Governance
> During the pre-public phase (July 2026 – Public Launch), we are operating with specialized repository rules due to missing automated GitHub branch protections. Please refer to [PRE_PUBLIC_DEV_GUIDE.md](file:///usr/local/google/home/johnqiangzhang/projects/vllm-torchtpu/PRE_PUBLIC_DEV_GUIDE.md) for standard submission flows, merge checklists, and mandatory guidelines.

---

## 🛠️ Installation

To run the **Qwen3** model, follow these installation steps:

### 1. Google Cloud Authentication

We need to authenticate with Google Cloud to access the private Torch TPU Virtual Registry (`https://us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/`). This registry contains packages like `torch-tpu`, `torch`, and other dependencies required by `vllm-torchtpu`.

Make sure you are logged into `gcloud` using your corporate account (i.e., one that has read permissions for the Torch TPU registry). If you need access to this registry, please reach out to the Torch TPU team.

You can check your active account by running:

```bash
gcloud auth list
```

If the active account is not your corporate account, switch to it or log in by running:

```bash
gcloud auth login

# For keyring authentication, see https://github.com/GoogleCloudPlatform/artifact-registry-python-tools?tab=readme-ov-file#authentication
gcloud auth application-default login
```

### 2. Install vLLM-TorchTPU and dependencies

We recommend using `uv` for installing dependencies as it is significantly faster than standard `pip`.

#### Option A: Using `uv` (Recommended)

```bash
uv venv --python 3.12 ~/uv_venv
source ~/uv_venv/bin/activate

# Set the username for the index
export UV_INDEX_TORCH_TPU_REGISTRY_USERNAME="oauth2accesstoken"

# Install keyring and Google Artifact Registry plugin for persistent auth
uv tool install keyring --with keyrings.google-artifactregistry-auth

# Use CPU Torch when building vLLM's editable TPU package.
export UV_TORCH_BACKEND=cpu

# Clone vLLM to allow making local patches for debugging
git clone --depth 1 --branch v0.23.0 https://github.com/vllm-project/vllm.git ../vllm

# Patch vLLM's TPU requirements to avoid installing the upstream tpu-inference
# plugin alongside vllm-torchtpu.
sed -i '/tpu-inference/d' ../vllm/requirements/tpu.txt

# Install vLLM in editable mode (forcing the 0.23.0 base version to prevent .dev prerelease mismatch during dependency resolution)
SETUPTOOLS_SCM_PRETEND_VERSION=0.23.0 VLLM_TARGET_DEVICE="tpu" uv pip install -e ../vllm

# Install vLLM-TorchTPU and dependencies
uv pip install --pre -e .
```

#### Option B: Using `pip`

> **Note:** Currently, `vllm==0.23.0` is supported.

```bash
python3.12 -m venv ~/pip_venv --symlinks
source ~/pip_venv/bin/activate
pip install --upgrade pip

# Install keyring and Google Artifact Registry plugin for persistent auth
pip install keyring keyrings.google-artifactregistry-auth

# Set the PIP_INDEX_URL environment variable (using keyring)
export PIP_INDEX_URL="https://oauth2accesstoken@us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/"

# Clone vLLM to allow making local patches for debugging
git clone --depth 1 --branch v0.23.0 https://github.com/vllm-project/vllm.git ../vllm

# Patch vLLM's TPU requirements to avoid installing the upstream tpu-inference
# plugin alongside vllm-torchtpu.
sed -i '/tpu-inference/d' ../vllm/requirements/tpu.txt

# Install vLLM in editable mode (forcing the 0.23.0 base version to prevent .dev prerelease mismatch during dependency resolution)
SETUPTOOLS_SCM_PRETEND_VERSION=0.23.0 VLLM_TARGET_DEVICE="tpu" pip install -e ../vllm

# Install vLLM-TorchTPU and dependencies
pip install --pre -e .
```

> [!TIP]
> A common cause of authentication errors when using `keyring` with `uv` is an existing `~/.netrc` file containing stale credentials.
>
> To resolve this, open `~/.netrc` in your preferred text editor and delete the block associated with `us-python.pkg.dev`. It will look something like this:
>
> ```text
> machine us-python.pkg.dev
> login oauth2accesstoken
> password <your_expired_token>
> ```

> **Note:** Prioritize compile mode for better performance. The first startup may take several minutes while TPU graphs compile. Add `--enforce-eager` if you want eager mode.
> On TPUv7, Qwen3-Coder-30B can fit on a single device. On v6, use a smaller model like Qwen3-4B or test with TP/EP.

---

## 🌐 Online Serving

Start the server with the following command:

```bash
vllm serve "Qwen/Qwen3-Coder-30B-A3B-Instruct" \
  --tensor_parallel_size=2 \
  --max-model-len=256 \
  --max-num-batched-tokens=256 \
  --attention-backend CUSTOM
```

### Send a Request

Once the server is running, you can verify it by sending a request:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "prompt": "Hello, my name is",
        "max_tokens": 20,
        "temperature": 0
    }'
```

### 📉 Running Offline Inference

You can also run a simple offline inference script to verify the setup without starting a full server.

> [!NOTE]
> For current TorchTPU builds, set the temporary workarounds below or append `PYTHONPATH=$(pwd)/src` to the `python3` command:
>
> ```bash
> export TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false
> export TORCHINDUCTOR_AUTOGRAD_CACHE=0
> ```

#### Tensor Parallelism (TP)

```bash
python3 examples/offline_inference.py \
  --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --max-model-len 256 \
  --max-num-batched-tokens 256 \
  --tensor_parallel_size=2
```

#### Expert Parallelism (EP)

```bash
python3 examples/offline_inference.py \
  --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --max-model-len 256 \
  --max-num-batched-tokens 256 \
  --tensor_parallel_size=2 \
  --enable-expert-parallel
```
