# TPU v7x (Ironwood) Serving Recipes

**TPU v7x (Ironwood)** is Google Cloud's 7th-generation TPU architecture. Each physical TPU v7x chip exposes **2 logical TensorCore devices** to PyTorch and JAX (`192 GB` HBM per chip / `96 GB` per logical device), so a single 4-chip TPU v7x host (`tpu7x-8`) provides **8 logical TPU devices** and `768 GB` of total HBM.

The recipes below are extracted directly from the repository's active PR and nightly CI performance benchmark configurations in [`scripts/vllm/benchmarking/configs/`](https://github.com/vllm-project/vllm-torchtpu/tree/main/scripts/vllm/benchmarking/configs) (scheduled via [`.buildkite/pipeline_perf.yml`](https://github.com/vllm-project/vllm-torchtpu/blob/main/.buildkite/pipeline_perf.yml)).

---

## Summary of Validated TPU v7x Topologies

| Model Checkpoint | Slice / Devices | Parallelism Layout | Quantization & KV Cache | CI Benchmark Config |
| :--- | :--- | :--- | :--- | :--- |
| `Qwen/Qwen3.5-397B-A17B-FP8` | `v7x-8` (1 host, 8 devices) | `TP=1, DP=8` + EP | `fp8` weights, `fp8` KV | [`qwen3.5-397b-fp8-tp1-dp8-ep.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/scripts/vllm/benchmarking/configs/qwen3.5-397b-fp8-tp1-dp8-ep.sh) |
| `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8` | `v7x-8` (1 host, 8 devices) | `TP=8, DP=1` + EP | `fp8` weights, `fp8` KV | [`qwen3-coder-480b-fp8-tp8-ep.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/scripts/vllm/benchmarking/configs/qwen3-coder-480b-fp8-tp8-ep.sh) |
| `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` | `v7x-8` (1 host, 8 devices) | `TP=8, DP=1` + EP | `fp8` weights, `fp8` KV | [`qwen3-coder-30b-fp8-tp8-ep.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/scripts/vllm/benchmarking/configs/qwen3-coder-30b-fp8-tp8-ep.sh) |
| `Qwen/Qwen3.5-35B-A3B-FP8` | `v7x-8` (1 host, 8 devices) | `TP=2, DP=4` + EP | `fp8` weights, `fp8` KV | [`qwen3.5-35b-fp8-dp4-tp2-ep.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/scripts/vllm/benchmarking/configs/qwen3.5-35b-fp8-dp4-tp2-ep.sh) |
| `RedHatAI/gemma-4-31B-it-FP8-Dynamic` | `v7x-8` (1 host, 8 devices) | `TP=2, DP=4` | `compressed-tensors` FP8 | [`gemma-4-31b-fp8-dp4-tp2-multimodal.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/scripts/vllm/benchmarking/configs/gemma-4-31b-fp8-dp4-tp2-multimodal.sh) |
| `zai-org/GLM-5.2-FP8` | `v7x-16` (2 hosts, 16 devices) | `TP=1, DP=16` + EP | `fp8` weights, `fp8` KV | [`glm-5.2-fp8-dp16-ep.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/scripts/vllm/benchmarking/configs/glm-5.2-fp8-dp16-ep.sh) |
| `deepseek-ai/DeepSeek-V4-Pro-0813` | `v7x-16` (2 hosts, 16 devices) | `TP=1, DP=16` + EP | `deepseek_v4_fp8` (W4A8 MoE), `fp8_e4m3` KV | [`deepseek-v4-pro-dp16-ep.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/scripts/vllm/benchmarking/configs/deepseek-v4-pro-dp16-ep.sh) |
| `moonshotai/Kimi-K3` | `v7x-32` (4 hosts, 32 devices) | `TP=32, DP=1` + EP | `compressed-tensors` FP8, `fp8` KV | [`kimi-k3-tp32-ep.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/scripts/vllm/benchmarking/configs/kimi-k3-tp32-ep.sh) |

Diffusion (image and video generation) recipes served with `vllm-omni` are listed separately in [3. Diffusion Recipes (vllm-omni)](#3-diffusion-recipes-vllm-omni); they are validated by the `tests_omni/` suites rather than the CI benchmark configs in this table.

---

## 1. Single-Host Recipes (`v7x-8` — 8 Logical Devices)

### `Qwen/Qwen3.5-397B-A17B-FP8` (`TP=1, DP=8 + EP`)

Golden throughput recipe on a single TPU v7x host ([`scripts/vllm/benchmarking/configs/qwen3.5-397b-fp8-tp1-dp8-ep.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/scripts/vllm/benchmarking/configs/qwen3.5-397b-fp8-tp1-dp8-ep.sh)):

```bash
export USE_MOE_SPARSE_CORE=1
export ONEHOT_MOE_PERMUTE_THRESHOLD=32768

vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --attention-backend CUSTOM \
  --max-model-len 10240 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.92 \
  --block-size 256 \
  --async-scheduling \
  --no-enable-prefix-caching \
  --language-model-only \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --port 8000
```

> [!TIP]
> **Alternative Parallelism Layouts for `Qwen3.5-397B-A17B-FP8`**: For lower per-request latency at smaller concurrencies, use `--tensor-parallel-size 2 --data-parallel-size 4` ([`qwen3.5-397b-fp8-tp2-dp4-ep.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/scripts/vllm/benchmarking/configs/qwen3.5-397b-fp8-tp2-dp4-ep.sh)) or `--tensor-parallel-size 8 --data-parallel-size 1` ([`qwen3.5-397b-fp8-tp8-dp1-ep.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/scripts/vllm/benchmarking/configs/qwen3.5-397b-fp8-tp8-dp1-ep.sh)).

### `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8` (`TP=8 + EP`)

Validated nightly in [`scripts/vllm/benchmarking/configs/qwen3-coder-480b-fp8-tp8-ep.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/scripts/vllm/benchmarking/configs/qwen3-coder-480b-fp8-tp8-ep.sh):

```bash
vllm serve Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
  --tensor-parallel-size 8 \
  --data-parallel-size 1 \
  --enable-expert-parallel \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --attention-backend CUSTOM \
  --max-model-len 16384 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 512 \
  --gpu-memory-utilization 0.95 \
  --async-scheduling \
  --no-enable-prefix-caching \
  --port 8000
```

### `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` (`TP=8 + EP`)

Used as the PR-time performance guard gate ([`scripts/vllm/benchmarking/configs/qwen3-coder-30b-fp8-tp8-ep.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/scripts/vllm/benchmarking/configs/qwen3-coder-30b-fp8-tp8-ep.sh)):

```bash
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --attention-backend CUSTOM \
  --max-model-len 16384 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 512 \
  --gpu-memory-utilization 0.95 \
  --async-scheduling \
  --no-enable-prefix-caching \
  --port 8000
```

---

## 2. Multi-Host Recipes (`v7x-16` & `v7x-32`)

On multi-host TPU v7x slices without large local disks, stream checkpoints directly from Google Cloud Storage (`gs://`) using `--load-format runai_streamer` and `--enable-ep-weight-filter` so each rank downloads only its own MoE expert shard.

### `zai-org/GLM-5.2-FP8` on `v7x-16` (`2 Hosts, TP=1, DP=16 + EP`)

Validated nightly in [`scripts/vllm/benchmarking/configs/glm-5.2-fp8-dp16-ep.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/scripts/vllm/benchmarking/configs/glm-5.2-fp8-dp16-ep.sh):

```bash
vllm serve zai-org/GLM-5.2-FP8 \
  --tensor-parallel-size 1 \
  --data-parallel-size 16 \
  --enable-expert-parallel \
  --enable-ep-weight-filter \
  --safetensors-load-strategy prefetch \
  --kv-cache-dtype fp8 \
  --max-model-len 9216 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.90 \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --reasoning-parser glm45 \
  --port 8000
```

### `moonshotai/Kimi-K3` on `v7x-32` (`4 Hosts, TP=32 + EP`)

Validated nightly in [`scripts/vllm/benchmarking/configs/kimi-k3-tp32-ep.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/scripts/vllm/benchmarking/configs/kimi-k3-tp32-ep.sh):

```bash
vllm serve moonshotai/Kimi-K3 \
  --tensor-parallel-size 32 \
  --enable-expert-parallel \
  --enable-ep-weight-filter \
  --trust-remote-code \
  --language-model-only \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --model-loader-extra-config '{"memory_limit":17179869184}' \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --max-model-len 10240 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.82 \
  --compilation-config '{"compile_sizes":[1,8,8192]}' \
  --async-scheduling \
  --port 8000
```

---

## 3. Diffusion Recipes (vllm-omni)

Image and video generation models are served through [vLLM-Omni](https://github.com/vllm-project/vllm-omni) with `vllm serve <model> --omni`. Install the `omni` extra (`uv sync --group vllm --extra omni`) or build the `omni` Docker target (`./docker/build_image.sh --target omni`); the `OmniTpuPlatform` plugin then registers itself automatically through the `vllm_omni.platform_plugins` entry point when a TPU is present. Each diffusion rank binds one logical TPU device, so a recipe uses as many devices as the product of its parallel degrees (for example `--cfg-parallel-size 2 --usp 4` uses 8), or one device when none is set.

> [!NOTE]
> Unlike the LLM recipes above, these are validated by the [`tests_omni/`](https://github.com/vllm-project/vllm-torchtpu/tree/main/tests_omni) suites (offline `vllm_omni.Omni` engine on a single device, `-omni` CI image) and by the multi-device PR benchmarks linked per recipe, not by `scripts/vllm/benchmarking/configs/`. The `vllm serve --omni` command lines map those validated settings onto the vLLM-Omni CLI; CI does not run them directly.

Defaults that apply to every recipe below:

- **Attention**: the Pallas flash-attention backend (`TPU_FLASH_ATTN`) is the default ([#1432](https://github.com/vllm-project/vllm-torchtpu/pull/1432)); `--diffusion-attention-backend TORCH_SDPA` selects the SDPA fallback (`SDPA` and `TPU_SDPA` are accepted aliases).
- **Multi-device runs**: dense all-to-all collectives are offloaded to SparseCore automatically ([#1450](https://github.com/vllm-project/vllm-torchtpu/pull/1450)); setting `--xla_tpu_enable_sparse_core_collective_offload_dense_all_to_all` yourself in `LIBTPU_INIT_ARGS` takes precedence.
- **Warm-up**: the first request at a new resolution, frame count, or step count JIT-compiles on TPU, so send one warm-up request before measuring latency.
- **Init timeouts**: each recipe passes `--stage-init-timeout 600 --init-timeout 1800` (vLLM-Omni defaults: `300` and `600` seconds), the values the `OmniRunner` test fixture uses ([#1356](https://github.com/vllm-project/vllm-torchtpu/pull/1356)).

| Model Checkpoint | Task | Slice / Devices | Parallelism Layout | Validated Resolution / Steps | Validation Test |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `black-forest-labs/FLUX.1-dev` | Text-to-image | Any `v7x` slice (1 device) | Single device | `512×512` / 10 steps | [`test_flux_nightly.py`](https://github.com/vllm-project/vllm-torchtpu/blob/main/tests_omni/models/flux/test_flux_nightly.py) |
| `black-forest-labs/FLUX.2-klein-4B` | Text-to-image | Any `v7x` slice (1 device) | Single device | `512×512` / 4 steps | [`test_flux_nightly.py`](https://github.com/vllm-project/vllm-torchtpu/blob/main/tests_omni/models/flux/test_flux_nightly.py) |
| `Qwen/Qwen-Image` | Text-to-image | Any `v7x` slice (1 device) | Single device | `512×512` / 10 steps | [`test_qwen_image_nightly.py`](https://github.com/vllm-project/vllm-torchtpu/blob/main/tests_omni/models/qwen_image/test_qwen_image_nightly.py) |
| `Qwen/Qwen-Image-Edit-2511` | Image editing | `v7x-8` (1 host, 8 devices) | `CFG=2, Ulysses=4` | `1024×1024` / 20 steps ([#1432](https://github.com/vllm-project/vllm-torchtpu/pull/1432), [#1748](https://github.com/vllm-project/vllm-torchtpu/pull/1748)); `512×512` / 10 steps (nightly, 1 device) | [`test_qwen_image_nightly.py`](https://github.com/vllm-project/vllm-torchtpu/blob/main/tests_omni/models/qwen_image/test_qwen_image_nightly.py) |
| `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | Image-to-video | `v7x-8` (1 host, 8 devices) | `CFG=2, Ulysses=4` | `480×832`, 81 frames / 40 steps ([#1432](https://github.com/vllm-project/vllm-torchtpu/pull/1432) benchmark); `480×832`, 17 frames / 2 steps (nightly, 1 device) | [`test_wan_nightly.py`](https://github.com/vllm-project/vllm-torchtpu/blob/main/tests_omni/models/wan/test_wan_nightly.py) |
| `Lightricks/LTX-2` | Text-to-video (with audio) | `v7x-8` (1 host, 8 devices) | `CFG=2, Ulysses=4` | `704×1280`, 121 frames / 40 steps | [#2297](https://github.com/vllm-project/vllm-torchtpu/pull/2297), [#2298](https://github.com/vllm-project/vllm-torchtpu/pull/2298) |

### `black-forest-labs/FLUX.1-dev` (`1 device`)

Validated nightly in [`tests_omni/models/flux/test_flux_nightly.py`](https://github.com/vllm-project/vllm-torchtpu/blob/main/tests_omni/models/flux/test_flux_nightly.py) (`512×512`, 10 steps, `guidance_scale` 3.5, seed 42) and benchmarked at `1024×1024` / 28 steps in [#1432](https://github.com/vllm-project/vllm-torchtpu/pull/1432):

```bash
vllm serve black-forest-labs/FLUX.1-dev \
  --omni \
  --stage-init-timeout 600 \
  --init-timeout 1800 \
  --port 8000
```

> [!NOTE]
> `black-forest-labs/FLUX.1-dev` is a gated Hugging Face repository: accept the license on the model page and export `HF_TOKEN` (or run `hf auth login`) before starting the server.

Request, using the nightly test parameters:

```bash
curl -s http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A futuristic glowing crystal monolith in a misty desert at night, neon reflections",
    "size": "512x512",
    "num_inference_steps": 10,
    "guidance_scale": 3.5,
    "seed": 42
  }' | python3 -c "import sys, json, base64; sys.stdout.buffer.write(base64.b64decode(json.load(sys.stdin)['data'][0]['b64_json']))" > flux1_dev.png
```

For `CFG=2, TP=4` on `v7x-8` (`1024×1024`, 28 steps), pass `--cfg-parallel-size 2 --tensor-parallel-size 4` and include `"true_cfg_scale": 2.0` with a `"negative_prompt"` in the request body so both CFG branches execute in parallel:

```bash
vllm serve black-forest-labs/FLUX.1-dev --omni --port 8000 --cfg-parallel-size 2 --tensor-parallel-size 4
```

```bash
curl -s http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "black-forest-labs/FLUX.1-dev",
    "prompt": "a cup of coffee on a wooden table, morning light",
    "negative_prompt": "blurry, low quality, distorted",
    "size": "1024x1024",
    "num_inference_steps": 28,
    "guidance_scale": 3.5,
    "true_cfg_scale": 2.0,
    "seed": 42,
    "response_format": "b64_json"
  }' | python3 -c "import sys, json, base64; sys.stdout.buffer.write(base64.b64decode(json.load(sys.stdin)['data'][0]['b64_json']))" > flux1_dev_cfg2_tp4.png
```

> [!TIP]
> **Lower latency on `v7x-8`**: add `--tensor-parallel-size 8`, benchmarked at `1024×1024` / 28 steps in [#1432](https://github.com/vllm-project/vllm-torchtpu/pull/1432) with a modest latency gain over a single device. `--cfg-parallel-size 2 --tensor-parallel-size 4` performed alike in the same benchmark when the request sets `"true_cfg_scale": 2.0` (above 1) together with a `negative_prompt`. The FLUX.1 transformer defines no Ulysses sequence-parallel plan, so scale it with `--tensor-parallel-size` rather than `--usp`.
>
> **`FLUX.1-schnell`**: serve `black-forest-labs/FLUX.1-schnell` with the same flags; the nightly test uses 4 steps and `guidance_scale` 0.0 (the PR-lane [`test_flux_pr.py`](https://github.com/vllm-project/vllm-torchtpu/blob/main/tests_omni/models/flux/test_flux_pr.py) uses 2 steps).

### `black-forest-labs/FLUX.2-klein-4B` (`1 device`)

Validated nightly in [`tests_omni/models/flux/test_flux_nightly.py`](https://github.com/vllm-project/vllm-torchtpu/blob/main/tests_omni/models/flux/test_flux_nightly.py) (`512×512`, 4 steps, `guidance_scale` 1.0, seed 42):

```bash
vllm serve black-forest-labs/FLUX.2-klein-4B \
  --omni \
  --stage-init-timeout 600 \
  --init-timeout 1800 \
  --port 8000
```

Request, using the nightly test parameters:

```bash
curl -s http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A futuristic glowing crystal monolith in a misty desert at night, neon reflections",
    "size": "512x512",
    "num_inference_steps": 4,
    "guidance_scale": 1.0,
    "seed": 42
  }' | python3 -c "import sys, json, base64; sys.stdout.buffer.write(base64.b64decode(json.load(sys.stdin)['data'][0]['b64_json']))" > flux2_klein.png
```

### `Qwen/Qwen-Image` (`1 device`)

Validated nightly in [`tests_omni/models/qwen_image/test_qwen_image_nightly.py`](https://github.com/vllm-project/vllm-torchtpu/blob/main/tests_omni/models/qwen_image/test_qwen_image_nightly.py) ([#1749](https://github.com/vllm-project/vllm-torchtpu/pull/1749); `512×512`, 10 steps, `true_cfg_scale` 4.0, seed 42, single-space `negative_prompt`):

```bash
vllm serve Qwen/Qwen-Image \
  --omni \
  --stage-init-timeout 600 \
  --init-timeout 1800 \
  --port 8000
```

Request, using the nightly test parameters:

```bash
curl -s http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A futuristic glowing crystal monolith in a misty desert at night, neon reflections",
    "negative_prompt": " ",
    "size": "512x512",
    "num_inference_steps": 10,
    "true_cfg_scale": 4.0,
    "seed": 42
  }' | python3 -c "import sys, json, base64; sys.stdout.buffer.write(base64.b64decode(json.load(sys.stdin)['data'][0]['b64_json']))" > qwen_image.png
```

> [!TIP]
> **8 devices**: add `--cfg-parallel-size 2 --usp 4` (`CFG 2 × Ulysses 4`), benchmarked on `Qwen-Image-Edit` at `1024×1024` / 20 steps in [#1432](https://github.com/vllm-project/vllm-torchtpu/pull/1432) (together with [#1748](https://github.com/vllm-project/vllm-torchtpu/pull/1748)); the text-to-image `Qwen/Qwen-Image` checkpoint has not been benchmarked with this layout on TPU. CFG parallelism only helps when requests set `true_cfg_scale` above 1 together with a `negative_prompt`, as in the request above.
>
> **Image editing**: `Qwen/Qwen-Image-Edit` and `Qwen/Qwen-Image-Edit-2511` use the same `vllm serve` flags and the same nightly test; send multipart requests to `/v1/images/edits` (`image`, `prompt`, `negative_prompt`, `size` (`auto` by default; the nightly test pins `512x512`), `num_inference_steps`, `true_cfg_scale`, `seed`).
>
> **First start**: checkpoints are downloaded on first use (the Qwen-Image repositories are several tens of GB), so on a cold cache pre-download them (`hf download Qwen/Qwen-Image`) or raise `--init-timeout` and `--stage-init-timeout`.

### `Qwen/Qwen-Image-Edit-2511` (`CFG=2, Ulysses=4`)

Validated nightly in [`tests_omni/models/qwen_image/test_qwen_image_nightly.py`](https://github.com/vllm-project/vllm-torchtpu/blob/main/tests_omni/models/qwen_image/test_qwen_image_nightly.py) ([#1749](https://github.com/vllm-project/vllm-torchtpu/pull/1749)) and verified on `v7x-8` (8 logical devices) with `CFG 2 × Ulysses 4` at `1024×1024` / 20 steps ([#1432](https://github.com/vllm-project/vllm-torchtpu/pull/1432), [#1748](https://github.com/vllm-project/vllm-torchtpu/pull/1748)):

```bash
vllm serve Qwen/Qwen-Image-Edit-2511 --omni --port 8000 --ulysses-degree 4 --cfg-parallel-size 2
```

Send a multipart image-edit request to `/v1/images/edits` (`input.png` is the source image):

```bash
curl -s -X POST http://localhost:8000/v1/images/edits \
  -F "model=Qwen/Qwen-Image-Edit-2511" \
  -F "image=@input.png" \
  -F "prompt=Turn the scene into a vintage watercolor painting" \
  -F "size=1024x1024" \
  -F "num_inference_steps=20" \
  -F "true_cfg_scale=4.0" \
  -F "seed=42" \
  -F "response_format=b64_json" \
  | python3 -c "import sys, json, base64; sys.stdout.buffer.write(base64.b64decode(json.load(sys.stdin)['data'][0]['b64_json']))" > qwen_image_edit.png
```

### `Wan-AI/Wan2.2-I2V-A14B-Diffusers` (`CFG=2, Ulysses=4`)

Validated nightly against a CPU fp32 golden video in [`tests_omni/models/wan/test_wan_nightly.py`](https://github.com/vllm-project/vllm-torchtpu/blob/main/tests_omni/models/wan/test_wan_nightly.py) ([#1188](https://github.com/vllm-project/vllm-torchtpu/pull/1188); `480×832`, 17 frames, 2 steps, `guidance_scale` 5.0, seed 42, single device), and benchmarked end to end on 8 devices with `CFG 2 × Ulysses 4` at `480×832` and `720×1280`, 81 frames, 40 steps in [#1432](https://github.com/vllm-project/vllm-torchtpu/pull/1432) (and, before flash attention, [#1427](https://github.com/vllm-project/vllm-torchtpu/pull/1427)):

```bash
vllm serve Wan-AI/Wan2.2-I2V-A14B-Diffusers \
  --omni \
  --cfg-parallel-size 2 \
  --usp 4 \
  --stage-init-timeout 600 \
  --init-timeout 1800 \
  --port 8000
```

Request: the reference image goes in `input_reference` (`/v1/videos/sync` returns the MP4; `/v1/videos` is the asynchronous job API):

```bash
curl -s http://localhost:8000/v1/videos/sync \
  -F "prompt=A serene cinematic drone shot gliding smoothly over ancient snow-covered pagodas in the mist, golden sunrise reflections on river ice, photorealistic, 4k" \
  -F "input_reference=@input.png" \
  -F "width=832" \
  -F "height=480" \
  -F "num_frames=81" \
  -F "fps=16" \
  -F "num_inference_steps=40" \
  -F "guidance_scale=5.0" \
  -F "seed=42" \
  -o output.mp4
```

> [!TIP]
> **Single device**: omit `--cfg-parallel-size` and `--usp` to run on one device (the nightly test configuration). `--usp 8` (`Ulysses 8`) was also run end to end in [#1427](https://github.com/vllm-project/vllm-torchtpu/pull/1427).
>
> **Other Wan checkpoints**: `Wan-AI/Wan2.2-TI2V-5B-Diffusers` (Text-Image-to-Video, 5B) and the Wan2.1 T2V, I2V, and VACE checkpoints are covered by the [`tests_omni/models/wan/`](https://github.com/vllm-project/vllm-torchtpu/tree/main/tests_omni/models/wan) tests on a single device.

### `Lightricks/LTX-2` (`CFG=2, Ulysses=4`)

Joint audio-video generation on `v7x-8` (8 logical devices) with `CFG 2 × Ulysses 4` at `704×1280`, 121 frames, 40 steps ([#2297](https://github.com/vllm-project/vllm-torchtpu/pull/2297), [#2298](https://github.com/vllm-project/vllm-torchtpu/pull/2298)):

```bash
vllm serve Lightricks/LTX-2 --omni --ulysses-degree 4 --cfg-parallel-size 2 --enforce-eager --port 8000
```

Submit an asynchronous video generation job to `/v1/videos`, poll `/v1/videos/<video_id>` until `"status": "completed"`, and download the rendered MP4 from `/v1/videos/<video_id>/content`:

```bash
curl -s http://localhost:8000/v1/videos \
  -F "prompt=A cinematic drone shot over a misty pine forest at sunrise, golden rays breaking through the trees." \
  -F "height=704" \
  -F "width=1280" \
  -F "num_frames=121" \
  -F "num_inference_steps=40" \
  -F "guidance_scale=4.0" \
  -F "seed=42" \
  -F "frame_rate=24"
```

```bash
curl -s http://localhost:8000/v1/videos/<video_id>
curl -s http://localhost:8000/v1/videos/<video_id>/content -o ltx2.mp4
```
