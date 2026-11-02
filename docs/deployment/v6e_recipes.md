# TPU v6e (Trillium) Serving Recipes

**TPU v6e (Trillium)** provides 1 TensorCore per physical chip with `32 GB` of high-bandwidth memory (HBM) per chip. Single-host TPU v6e VMs are available in `1-chip` (`v6e-1`, `32 GB` HBM), `4-chip` (`v6e-4`, `128 GB` HBM), and `8-chip` (`v6e-8`, `256 GB` HBM) topologies, with multi-host pod slices scaling to `v6e-16` (`512 GB` HBM) and beyond.

> [!NOTE]
> **Reference Configuration Examples**: Unlike the [TPU v7x Serving Recipes](v7x_recipes.md) (which mirror dedicated CI benchmark scripts in `scripts/vllm/benchmarking/configs/`), the `v6e` commands below are **reference configuration examples** sized for `v6e` HBM capacity (`32 GB` per chip). Check the nightly CI matrix in `support_matrices/nightly/v6e/model_support_matrix.csv` for the latest `v6e` validation status of each checkpoint, and [Supported Models](../user_guide/supported_models.md) for the validated model families and features.

---

## Recommended Topologies by Model Size

| Model Checkpoint | Precision | Recommended v6e Slice | Parallelism Layout |
| :--- | :--- | :--- | :--- |
| `meta-llama/Llama-3.2-1B-Instruct`, `Llama-3.2-3B-Instruct`, `Qwen/Qwen3-8B` | `bf16` or `fp8` | `v6e-1` (1 chip, 32 GB) | `TP=1` |
| `Qwen/Qwen2.5-VL-7B-Instruct`, `google/gemma-3-12b-it` | `bf16` or `fp8` | `v6e-1` or `v6e-4` | `TP=1` or `TP=4` |
| `Qwen/Qwen3-32B`, `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8`, `Qwen/Qwen3.5-35B-A3B-FP8` | `fp8` (`bf16` on `v6e-4`/`v6e-8`) | `v6e-4` (4 chips, 128 GB) | `TP=4` (+ `--enable-expert-parallel` for MoE) |
| `meta-llama/Llama-3.3-70B-Instruct`, `openai/gpt-oss-120b` | `fp8` / `mxfp4` (`bf16` on `v6e-8`) | `v6e-8` (8 chips, 256 GB) | `TP=8` (+ `--enable-expert-parallel` for MoE) |
| `Qwen/Qwen3.5-397B-A17B-FP8` | `fp8` | `v6e-16` (2 hosts, 512 GB) | `TP=16` or `DP=2, TP=8` + EP |

---

## 1. `v6e-1` (Single Chip, 32 GB HBM)

Ideal for 1B–8B dense models and development testing:

```bash
vllm serve Qwen/Qwen3-8B \
  --tensor-parallel-size 1 \
  --kv-cache-dtype fp8 \
  --max-model-len 8192 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.90 \
  --async-scheduling \
  --port 8000
```

---

## 2. `v6e-4` (4 Chips, 128 GB HBM)

Recommended for 30B–35B dense and MoE checkpoints:

```bash
vllm serve Qwen/Qwen3.5-35B-A3B-FP8 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --attention-backend CUSTOM \
  --max-model-len 16384 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.92 \
  --async-scheduling \
  --port 8000
```

---

## 3. `v6e-8` (8 Chips, 256 GB HBM)

Recommended for 70B–120B models on a single host without multi-host networking overhead:

### `meta-llama/Llama-3.3-70B-Instruct` (`TP=8`)

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --tensor-parallel-size 8 \
  --kv-cache-dtype fp8 \
  --max-model-len 16384 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.92 \
  --async-scheduling \
  --port 8000
```

### `openai/gpt-oss-120b` (`TP=8 + EP`, MXFP4)

```bash
vllm serve openai/gpt-oss-120b \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --kv-cache-dtype fp8 \
  --max-model-len 16384 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.92 \
  --async-scheduling \
  --port 8000
```

> [!TIP]
> **Maximizing KV Cache Capacity on `v6e` (`32 GB` HBM/chip)**: Always pass `--kv-cache-dtype fp8` on TPU v6e to double available KV cache blocks and prevent preemption under long-context or high-concurrency workloads.

---

## 4. Diffusion (vLLM-Omni)

`vllm-torchtpu` registers `OmniTpuPlatform` for [`vllm-omni`](https://github.com/vllm-project/vllm-omni), so FLUX pipelines run on TPU v6e through the offline `Omni` API and `vllm serve --omni`. The numbers below were measured on native single-host `v6e-1`, `v6e-4`, and `v6e-8` VMs with `torch-tpu 0.1.1+release.2026.9.22`, `libtpu 0.0.50`, and `vllm-omni 0.29.0rc1`.

| Model Checkpoint | Recommended v6e Slice | Parallelism Layout |
| :--- | :--- | :--- |
| `black-forest-labs/FLUX.2-klein-4B` | `v6e-1` (1 chip) | world size 1 (or Ulysses `SP=4` on `v6e-4`) |
| `black-forest-labs/FLUX.1-schnell` | `v6e-4` (4 chips) | `TP=4` |
| `black-forest-labs/FLUX.1-dev` | `v6e-4` or `v6e-8` | `TP=4` or `TP=8` |
| `Qwen/Qwen-Image` | `v6e-8` (8 chips) | `TP=8` with `vae_use_tiling=True` (experimental: almost no HBM headroom per chip) |
| `Qwen/Qwen-Image-Edit` | not supported on `v6e` | OOM while loading weights at `TP=8` (see Limitations) |

### Container Launch

The measurements used the CI `-omni` image and this launch command, with the Hugging Face cache on a host directory (`/mnt/hf`). `FLUX.1-dev` is gated, so `HF_TOKEN` must have access to it:

```bash
IMG=us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu-ci/vllm-torchtpu:a7ce631c23e80a41a3294aa41ef8ed914ca11e95-8039d3cbfd55be2aa172d544042c53b4e966369d-omni

sudo docker run -it --rm --privileged --net=host --shm-size=64g \
  -v /mnt/hf:/mnt/hf \
  -e HF_HOME=/mnt/hf \
  -e HF_TOKEN="${HF_TOKEN}" \
  "${IMG}" bash
```

Environment used inside the container for all runs below. Leave `TORCH_TPU_TOPOLOGY` unset; it is derived from the world size:

```bash
export DIFFUSION_ATTENTION_BACKEND=TPU_SDPA
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false
export TORCHINDUCTOR_AUTOGRAD_CACHE=0
export VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT=1800
unset TORCH_TPU_TOPOLOGY
```

> [!WARNING]
> **Known issues in the pinned `-omni` image**: The image above needs these in-container workarounds:
>
> - `cv2` fails to import (missing `libGL`) in spawned multi-rank workers. Install the system libraries with `apt-get install -y libgl1 libglib2.0-0` (tracked in #1939).
> - `RoutedExpertsLists` is missing from `vllm.v1.outputs` in the pinned vLLM, which breaks any `import vllm_omni`, and `AsyncOmni.release_kv_cache_memory` is abstract, so the `vllm serve --omni` engine cannot be created. Fixed by #1945.
> - `vllm serve --omni` imports `torchaudio`, which the image does not ship. Install the CPU wheel with `uv pip install --system --no-deps --index-url https://download.pytorch.org/whl/cpu torchaudio==2.11.0` (fixed by #1944).

### Offline Inference (`Omni` API)

`FLUX.2-klein-4B` on `v6e-1`. For `FLUX.1-*`, pass `tensor_parallel_size=4` (`v6e-4`) or `tensor_parallel_size=8` (`v6e-8`), and use `guidance_scale=0.0` for `FLUX.1-schnell` or `guidance_scale=3.5` for `FLUX.1-dev`. For Ulysses sequence parallelism, pass `ulysses_degree=4` instead:

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

if __name__ == "__main__":
    omni = Omni(
        "black-forest-labs/FLUX.2-klein-4B",
        init_timeout=1200,  # timeouts used for the measured runs
        stage_init_timeout=1200,
    )
    params = OmniDiffusionSamplingParams(
        height=1024,
        width=1024,
        num_inference_steps=4,
        guidance_scale=1.0,
        seed=42,
    )
    outputs = omni.generate("A glowing crystal monolith in a misty desert at night", params)
    outputs[0].images[0].save("flux2_klein.png")
    omni.shutdown()
```

### Online Serving (`vllm serve --omni`)

`FLUX.1-dev` with `TP=4` on `v6e-4`. For `FLUX.2-klein-4B` on `v6e-1`, drop `--tensor-parallel-size`:

```bash
vllm serve black-forest-labs/FLUX.1-dev --omni \
  --tensor-parallel-size 4 \
  --init-timeout 1200 \
  --stage-init-timeout 1200 \
  --port 8000
```

```bash
curl -s http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
        "prompt": "A glowing crystal monolith in a misty desert at night",
        "size": "1024x1024",
        "num_inference_steps": 28,
        "guidance_scale": 3.5,
        "seed": 42,
        "response_format": "b64_json"
      }' \
  | python3 -c 'import base64, json, sys; open("flux1_dev.png", "wb").write(base64.b64decode(json.load(sys.stdin)["data"][0]["b64_json"]))'
```

### Measured Performance on v6e

Warm latency is the second request at the same seed. Peak HBM is the maximum across the chips in use (`32 GB` per chip). First request includes XLA compilation.

| Model | Slice / Layout | Mode | Resolution | Steps (guidance) | Warm Latency | Warm s/step | Peak HBM / Chip | First Request |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `FLUX.2-klein-4B` | `v6e-1`, world 1 | Offline | 512x512 | 4 (1.0) | 0.28 s | 0.071 | 15.58 GiB | 47.3 s |
| `FLUX.2-klein-4B` | `v6e-1`, world 1 | Offline | 1024x1024 | 4 (1.0) | 0.70 s | 0.175 | 15.86 GiB | 183.2 s |
| `FLUX.2-klein-4B` | `v6e-1`, world 1 ¹ | `vllm serve` | 512x512 | 4 (default) | 0.30 s | 0.076 | 16.32 GiB | 37.6 s |
| `FLUX.2-klein-4B` | `v6e-1`, world 1 ¹ | `vllm serve` | 1024x1024 | 4 (default) | 0.78 s | 0.195 | 16.26 GiB | 175.8 s |
| `FLUX.2-klein-4B` | `v6e-4`, Ulysses `SP=4` | Offline | 1024x1024 | 4 (1.0) | 0.49 s | 0.123 | 19.17 GiB | 10.61 s |
| `FLUX.1-schnell` | `v6e-4`, `TP=4` | Offline | 512x512 | 4 (0.0) | 0.49 s | 0.122 | 17.65 GiB | 1.02 s |
| `FLUX.1-dev` | `v6e-4`, `TP=4` | Offline | 512x512 | 10 (3.5) | 1.06 s | 0.106 | 18.95 GiB | 1.09 s |
| `FLUX.1-dev` | `v6e-4`, `TP=4` | Offline | 1024x1024 | 28 (3.5) | 3.60 s | 0.129 | 22.55 GiB | 131.11 s |
| `FLUX.1-dev` | `v6e-4`, `TP=4` | `vllm serve` | 1024x1024 | 28 (3.5) | 3.70 s | 0.131 | 23.90 GiB | 4.50 s ² |
| `FLUX.1-dev` | `v6e-8`, `TP=8` | Offline | 1024x1024 | 28 (3.5) | 3.08 s | 0.110 | 21.02 GiB | 115.8 s |
| `Qwen-Image` | `v6e-8`, `TP=8` ³ | Offline | 512x512 | 20 (true CFG 4.0) | 8.60 s | 0.43 | 30.86 GiB | 40.8 s |
| `Qwen-Image` | `v6e-8`, `TP=8` ³ | Offline | 1024x1024 | 20 (true CFG 4.0) | 10.61 s | 0.53 | 31.11 GiB | 43.0 s |

¹ Measured on a native `v6e-1` VM with the PR-C fixes (#1944, #1945) applied in the container, without passing `guidance_scale` (pipeline default). Latency is end-to-end HTTP (`POST /v1/images/generations`), including PNG encoding and base64 in the response; warm latency is the faster of two warm requests at the same seed. Server startup to ready took `17.8 s`.

² Ran after the offline `1024x1024` `TP=4` run in the same container, so this is not a cold-compile number.

³ Measured with `vae_use_tiling=True` and `true_cfg_scale=4.0` (Qwen-Image uses true classifier-free guidance, so each step runs two transformer passes). Peak HBM is within 0.15 GiB of the 31.24 GiB usable per chip; engine initialization took 129.6 s (cold) and 65.6 s (warm Hugging Face cache). Running without VAE tiling, with larger resolutions, or with batch size > 1 was not tested and is expected to run out of HBM.

`FLUX.1-schnell` with `TP=4` on `v6e-4` also matches the CPU `diffusers` golden image within the PR CI thresholds (`MAE 0.0182`, `SSIM 0.9649`).

### Qwen-Image / Qwen-Image-Edit

`Qwen/Qwen-Image` runs on `v6e-8` with `TP=8`, but with almost no HBM headroom: the Qwen2.5-VL text encoder is replicated on every chip and the transformer's `img_mod`/`txt_mod` modulation projections are not tensor-parallel sharded, so peak HBM reaches `31.11 GiB` of the `31.24 GiB` usable per chip at 1024x1024. Treat it as experimental and send a warm-up request before serving traffic. Only the offline configuration below was measured; there is no `vllm serve` recipe for Qwen-Image yet:

```python
omni = Omni(
    "Qwen/Qwen-Image",
    tensor_parallel_size=8,
    vae_use_tiling=True,
    init_timeout=1200,
    stage_init_timeout=1200,
)
params = OmniDiffusionSamplingParams(
    height=1024, width=1024, num_inference_steps=20, true_cfg_scale=4.0, seed=42
)
```

`Qwen/Qwen-Image-Edit` does **not** fit on `v6e-8` at `TP=8`: weight loading fails with `RuntimeBufferAllocationFailure`. The HSDP path (`use_hsdp=True`) fails inside PyTorch FSDP2 initialization (`variable.set_data(tensor)` with incompatible tensor types in `torch.distributed.fsdp._fully_shard._fsdp_init._move_states_to_device`), an FSDP2 / `torch_tpu` issue rather than a `vllm-torchtpu` one, and CPU offload is not supported on TPU, so there is no supported `v6e` configuration for Qwen-Image-Edit today.

### Limitations on v6e

- **`FLUX.1-*` does not fit on `v6e-1`**: `FLUX.1-schnell` runs out of HBM while loading weights on a single chip (`30.66 GiB` peak, `RESOURCE_EXHAUSTED`). `FLUX.1-dev` uses the same transformer size. Use `TP=4` on `v6e-4` or `TP=8` on `v6e-8`.
- **2-chip world sizes are not supported**: world size 2 (for example `--tensor-parallel-size 2`) fails with `RuntimeError: No TPU topology found for count: 2`, and forcing `TORCH_TPU_TOPOLOGY=1,2,1` fails in libtpu with `START_SESSION failed`. Use a world size of 1, 4, or 8. (#1943 replaces this with a `ValueError` that lists the supported sizes.)
- **Qwen-Image-Edit does not fit on `v6e-8`**: `TP=8` runs out of HBM while loading weights (`RuntimeBufferAllocationFailure`). `Qwen-Image` itself fits only with `vae_use_tiling=True` and no batching.
- **HSDP is not available on TPU yet**: `use_hsdp=True` fails in PyTorch FSDP2 initialization (`variable.set_data(tensor)` incompatible tensor type), so it cannot be used to shard the replicated Qwen-Image layers.
- **First-request compilation cost**: the first request compiles the pipeline and can take minutes (up to `183.2 s` measured above). Send a warm-up request before serving traffic. The measured runs used `VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT=1800` and `1200 s` init timeouts.
