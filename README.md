<h1 align="center">vLLM TPU</h1>

This repository contains the integration of **TorchTPU** and **vLLM**. It is a vLLM platform plugin packaged as `vllm_torchtpu`, with TPU kernels and runtime code for TorchTPU.

---

## 🛠️ Installation

To run the **Qwen3** model on a single device, follow these installation steps:

### 1. Google Cloud Authentication

We need to authenticate with Google Cloud to access the private Torch TPU Virtual Registry (`https://us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/`). This registry contains packages like `torch-tpu`, `torch`, and other dependencies required by `torchtpu-vllm`.

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

### 2. Install TorchTPU-vLLM and dependencies

We recommend using `uv` for installing dependencies as it is significantly faster than standard `pip`.

#### Option A: Using `uv` (Recommended)

```bash
uv venv --python 3.12 ~/uv_venv
source ~/uv_venv/bin/activate

# Set the username for the index
export UV_INDEX_TORCH_TPU_REGISTRY_USERNAME="oauth2accesstoken"

# Install keyring and Google Artifact Registry plugin for persistent auth
uv tool install keyring --with keyrings.google-artifactregistry-auth

# Clone vLLM to allow making local patches for debugging
git clone --depth 1 --branch v0.22.1 https://github.com/vllm-project/vllm.git ../vllm

# Patch vLLM's TPU requirements to avoid installing the upstream tpu-inference
# plugin alongside torchtpu-vllm.
sed -i '/tpu-inference/d' ../vllm/requirements/tpu.txt

# Install vLLM in editable mode (forcing the 0.22.1 base version to prevent .dev prerelease mismatch during dependency resolution)
SETUPTOOLS_SCM_PRETEND_VERSION=0.22.1 VLLM_TARGET_DEVICE="tpu" uv pip install -e ../vllm

# Install TorchTPU-vLLM and dependencies
uv pip install --pre -e .
```

#### Option B: Using `pip`

> **Note:** Currently, `vllm==0.22.1` is supported.

```bash
python3.12 -m venv ~/pip_venv --symlinks
source ~/pip_venv/bin/activate
pip install --upgrade pip

# Install keyring and Google Artifact Registry plugin for persistent auth
pip install keyring keyrings.google-artifactregistry-auth

# Set the PIP_INDEX_URL environment variable (using keyring)
export PIP_INDEX_URL="https://oauth2accesstoken@us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/"

# Clone vLLM to allow making local patches for debugging
git clone --depth 1 --branch v0.22.1 https://github.com/vllm-project/vllm.git ../vllm

# Patch vLLM's TPU requirements to avoid installing the upstream tpu-inference
# plugin alongside torchtpu-vllm.
sed -i '/tpu-inference/d' ../vllm/requirements/tpu.txt

# Install vLLM in editable mode (forcing the 0.22.1 base version to prevent .dev prerelease mismatch during dependency resolution)
SETUPTOOLS_SCM_PRETEND_VERSION=0.22.1 VLLM_TARGET_DEVICE="tpu" pip install -e ../vllm

# Install TorchTPU-vLLM and dependencies
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

> **Note:** Prioritize compile mode for better performance. Add `--enforce-eager` if you want eager mode.
> On TPUv7, Qwen3-Coder-30B can fit on a single device. On v6, use a smaller model like Qwen3-4B or test with TP/EP.

---

## 🌐 Online Serving

Start the server with the following command:

```bash
vllm serve "Qwen/Qwen3-Coder-30B-A3B-Instruct" \
  --tensor_parallel_size=1 \
  --max-model-len=256 \
  --max-num-batched-tokens=256 \
  --attention-backend CUSTOM
```

> [!TIP]
> If you see `RuntimeError: operator torchvision::nms does not exist`, run either `uv pip uninstall torchvision` or `pip uninstall torchvision`.

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

#### Single Device

```bash
python3 examples/offline_inference.py \
  --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --max-model-len 256 \
  --max-num-batched-tokens 256
```

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
