# MiniMax-H3 on TPU v7x with vLLM-Omni (Synthetic Random Weights)

MiniMax-H3 is a joint video and audio diffusion pipeline served by [vLLM-Omni](https://github.com/vllm-project/vllm-omni). This recipe runs every MiniMax-H3 pipeline (text-to-video+audio, first/last-frame-to-video+audio and reference-to-video+audio, offline and online) on one TPU v7x host (`tpu7x-8`, 8 logical TPU devices) with `vllm-torchtpu` and `vllm-omni`.

> [!WARNING]
> This recipe uses a synthetic checkpoint with random weights. It uses no file from the `MiniMaxAI/MiniMax-H3` Hugging Face repository (configs, parameters, weights or remote code), and copies no config from `diffusers`, `transformers` or the Hugging Face Hub. Every config value comes from the vLLM-Omni source tree (commit `e1d77a3`) or is labelled as a synthetic choice. Because the weights are random, every generated video and audio track is noise. The recipe checks that the TPU code paths run end to end and produce well-formed media. It says nothing about output quality.

---

## Summary of Validated Pipelines

| Pipeline | Task | Conditioning | Entry Point |
| :--- | :--- | :--- | :--- |
| `1_T2VA` | `t2va` | text only | `run_all_pipelines.py` (offline, `AsyncOmni`) |
| `2_FL2VA_0` | `fl2va` | 1 image at frame `0` | `run_all_pipelines.py` |
| `3_FL2VA_last` | `fl2va` | 1 image at frame `-1` | `run_all_pipelines.py` |
| `4_FL2VA_0_last` | `fl2va` | 2 images at frames `0, -1` | `run_all_pipelines.py` |
| `5_Ref2VA_image` | `ref2va` | reference image | `run_all_pipelines.py` |
| `6_Ref2VA_image_audio` | `ref2va` | reference image + audio | `run_all_pipelines.py` |
| `7_Ref2VA_video` | `ref2va` | reference video | `run_all_pipelines.py` |
| `8_online` | `t2va` | text only, `audio_mode=native` | `server_and_curl.py` (`vllm serve --omni`, `POST /v1/videos/sync`) |

All pipelines run with `TP=8`, text-encoder TP `8`, VAE patch parallel `8` (`tile` mode, VAE tiling on), `DP=1`, eager mode.

---

## 1. Prerequisites

- One TPU v7x host (`tpu7x-8`, 4 chips / 8 logical devices).
- A Python virtual environment with `vllm`, `vllm-torchtpu` and `torch_tpu` installed (see [Installation](../getting_started/installation.md)). The validated runs used the venv at `/mnt/workspace/vllm/repo_clean_env`.
- A `vllm-omni` checkout at commit `e1d77a3` on `PYTHONPATH`.
- `imageio-ffmpeg`, `Pillow`, `numpy` and `requests` in the venv. The example scripts use them to write and inspect media.
- `soundfile` in the venv. The `torchaudio` shim's `torchaudio.load` uses it to read the Ref2VA audio reference.
- `ffmpeg` and `ffprobe` on `PATH`. vLLM-Omni's Ref2VA `reference_video.py` calls both. The validated runs used a static ffmpeg 7.0.2 build in `/mnt/workspace/h3_tpu/bin`, which the example scripts and `launch_env.sh` prepend to `PATH`.
- About `140 GB` of free disk for the synthetic checkpoint (`136G` on disk).

`torchaudio` is not required. `vllm-torchtpu` installs a minimal `torchaudio` shim when `torchaudio` cannot be imported.

---

## 2. Environment

The checkpoint exists only in the local Hugging Face cache, so turn off all Hub access:

```bash
source /mnt/workspace/vllm/repo_clean_env/bin/activate

export PYTHONPATH=/path/to/vllm-torchtpu/src:/path/to/vllm-omni
export HF_HOME=/mnt/workspace/hf_cache

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1

export VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=3600
```

> [!IMPORTANT]
> `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and `DIFFUSERS_OFFLINE=1` are required. Without them the loaders may try to fetch `MiniMaxAI/MiniMax-H3` from the Hub instead of using the synthetic cache.

---

## 3. Generate the Synthetic Random-Weight Checkpoint

The generator writes a Hugging Face hub-layout cache folder, `models--MiniMaxAI--MiniMax-H3` (`refs/main`, `blobs/<sha256>`, `snapshots/<rev>/...`), with `FL2VA/` and `Ref2VA/` sub-pipelines. It imports only `torch`, `safetensors` and the standard library. All tensors are random with fixed seeds: `normal(0, 0.02)` for matrices, ones for norm gains, zeros for biases. `SYNTHETIC_PROVENANCE.json` maps every config value to a vLLM-Omni source location.

```bash
python examples/omni/minimax_h3/synthetic/gen_minimax_h3_cache.py \
  --out-hub-dir "$HF_HOME/hub" \
  --workers 4
```

Then verify the cache:

```bash
python examples/omni/minimax_h3/synthetic/verify_synthetic_cache.py \
  --hub-dir "$HF_HOME/hub"
```

Expected result: `SUMMARY: 13 passed, 0 failed`. The validated build took `562s` and produced 110 files (`136G`), revision `a08195adbc6aee6e57b90257f8b0b4dead361260`. The generator makes the blobs read-only at the end of the build.

> [!NOTE]
> The synthetic audio and video VAEs ship as remote code inside the synthetic checkpoint. That is why the scripts pass `trust_remote_code=True` / `--trust-remote-code`. No remote code from the upstream Hugging Face repository is used.

---

## 4. Generate Synthetic Inputs

The conditioning pipelines use small synthetic media (256x256, 24 fps, 4 s):

```bash
python examples/omni/minimax_h3/generate_inputs.py
```

This writes `image_color_bars.png`, `image_gradient.png`, `image_checkerboard.png`, `video_moving_gradient.mp4` and `audio_sine_sweep.wav`.

---

## 5. Run the Pipelines

Every pipeline uses 2 inference steps, `256x256` and 4 seconds of output (24 fps, seed `42`). Source the environment from Section 2 first. The validated runs launched each command as a TPU job with `tpu_python` on a CDK `tpu7x-8` job. On a TPU VM, run the scripts with `python` from the activated venv instead.

The scripts read inputs from `/mnt/workspace/h3_tpu/inputs` and write outputs to `/mnt/workspace/h3_tpu/outputs/random_weights`. Both paths are hard-coded; edit the scripts to change them. `6_Ref2VA_image_audio` and `7_Ref2VA_video` pass file paths to the pipeline (`audio_path` and `[video_path]`), not raw bytes.

### Offline pipelines (`AsyncOmni`)

```bash
# 1. Text to video + audio
tpu_python examples/omni/minimax_h3/run_all_pipelines.py --only 1_T2VA --steps 2 --size 256 --duration 4

# 2. First frame to video + audio
tpu_python examples/omni/minimax_h3/run_all_pipelines.py --only 2_FL2VA_0 --steps 2 --size 256 --duration 4

# 3. Last frame to video + audio
tpu_python examples/omni/minimax_h3/run_all_pipelines.py --only 3_FL2VA_last --steps 2 --size 256 --duration 4

# 4. First + last frame to video + audio
tpu_python examples/omni/minimax_h3/run_all_pipelines.py --only 4_FL2VA_0_last --steps 2 --size 256 --duration 4

# 5. Reference image to video + audio
tpu_python examples/omni/minimax_h3/run_all_pipelines.py --only 5_Ref2VA_image --steps 2 --size 256 --duration 4

# 6. Reference image + audio to video + audio
tpu_python examples/omni/minimax_h3/run_all_pipelines.py --only 6_Ref2VA_image_audio --steps 2 --size 256 --duration 4

# 7. Reference video to video + audio
tpu_python examples/omni/minimax_h3/run_all_pipelines.py --only 7_Ref2VA_video --steps 2 --size 256 --duration 4
```

The script builds the engine with:

```python
AsyncOmni(
    model="MiniMaxAI/MiniMax-H3",
    trust_remote_code=True,
    tensor_parallel_size=8,
    text_encoder_tp_size=8,
    vae_patch_parallel_size=8,
    data_parallel_size=1,
    num_gpus=8,
    enforce_eager=True,
    vae_parallel_mode="tile",
    vae_use_tiling=True,
)
```

### Online pipeline (`8_online`)

```bash
tpu_python examples/omni/minimax_h3/server_and_curl.py
```

The script starts the server, waits until `GET /v1/models` answers, sends one request and stops the server:

```bash
vllm serve MiniMaxAI/MiniMax-H3 --omni --trust-remote-code \
  --num-gpus 8 \
  --tensor-parallel-size 8 \
  --text-encoder-tp-size 8 \
  --vae-patch-parallel-size 8 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --enforce-eager \
  --stage-init-timeout 1800 \
  --init-timeout 1800 \
  --port 8000
```

```bash
curl -X POST http://localhost:8000/v1/videos/sync \
  --data-urlencode prompt="A test prompt" \
  --data-urlencode size=256x256 \
  --data-urlencode seconds=4 \
  --data-urlencode num_inference_steps=2 \
  --data-urlencode extra_params='{"audio_mode": "native", "task": "t2va", "aspect_ratio": "1:1"}' \
  -o 8_online.mp4
```

`t2va` needs an explicit `aspect_ratio`. This `curl` is the shell form of the script's form-encoded request; it was not run on its own.

---

## 6. Expected Outputs

For each pipeline `<name>` the scripts write to the output directory:

- `<name>.mp4`: H.264 video (`yuv420p`, 24 fps, requested size). For the offline pipelines it carries an AAC audio track (32 kHz, stereo). For `8_online` it is the body of the `/v1/videos/sync` response; if the server returns JSON, the script decodes the base64 video from it. Either way the script checks for the MP4 `ftyp` box before saving.
- `<name>_frame0.png` and `<name>_mid.png`: first and middle decoded frames (offline pipelines). `8_online` writes only `8_online_frame0.png`.
- `run_status.json`: per-pipeline `status`, `seconds`, `error`, the `ffmpeg` stream lines of the mp4 and the PNG names.

All frames and audio are noise because the weights are random.

---

## 7. Results (TPU v7x, `tpu7x-8`, synthetic random weights)

| Pipeline | Status | Wall Time | Output |
| :--- | :--- | :--- | :--- |
| `1_T2VA` | PASS | 59.57s | 256x256 h264 + 32 kHz stereo AAC, 4.46s |
| `2_FL2VA_0` | PASS | 76.57s | 256x256 h264 + 32 kHz stereo AAC, 4.46s |
| `3_FL2VA_last` | PASS | 77.90s | 256x256 h264 + 32 kHz stereo AAC, 4.46s |
| `4_FL2VA_0_last` | PASS | 82.97s | 256x256 h264 + 32 kHz stereo AAC, 4.46s |
| `5_Ref2VA_image` | PASS | 80.69s | 256x256 h264 + 32 kHz stereo AAC, 4.46s |
| `6_Ref2VA_image_audio` | PASS | 85.04s | 256x256 h264 + 32 kHz stereo AAC, 4.46s |
| `7_Ref2VA_video` | PASS | 83.32s | 256x256 h264 + 32 kHz stereo AAC, 4.46s |
| `8_online` | PASS | 54.61s | 256x256 h264 + 32 kHz stereo AAC, 4.46s |

Validated with vllm-omni e1d77a3 on the enablement commits of this series, before the performance commits.

Wall time is the per-request time measured by the script, without engine startup.

---

## 8. TPU Patches Applied

`vllm-torchtpu` applies these patches automatically; no user action is needed. Each one is a no-op when vLLM-Omni or the MiniMax-H3 pipeline is not loaded.

| Patch | Location | Why |
| :--- | :--- | :--- |
| TPU-safe tensor broadcasts | `src/vllm_torchtpu/omni/minimax_h3.py` (`patch_pipeline_tensor_broadcasts`) | The pipeline sends encoder inputs/outputs and media conditioning from rank 0 to the other ranks with its own collectives. The patch broadcasts shape headers and payloads as whole host-staged tensors, which avoids a `tpu_dist` broadcast hang. |
| TPU-safe object collectives | `src/vllm_torchtpu/omni/minimax_h3.py` (`patch_pipeline_object_collectives`) | `broadcast_object_list` / `all_gather_object` use the gloo group when one exists, otherwise a host-built length header plus an int32 payload. |
| Indexed modulation reference | `src/vllm_torchtpu/omni/minimax_h3.py` (`patch_indexed_modulation`) | The DiT AdaLN indexed-modulation kernels exist only in Triton; on TPU they go to the PyTorch reference path. |
| Video VAE device context | `src/vllm_torchtpu/omni/minimax_h3.py` (`patch_video_vae_device_module`) | `TpuDeviceModule.device` is internal to `torch_tpu`, so on TPU the video VAE's `device_module.device(d)` uses `torch.accelerator.device_index` instead. |
| Diffusion async output stream device | `src/vllm_torchtpu/omni/patches.py` (`_patch_worker_proc_async_output_device`) | `WorkerProc._async_output_loop` creates its D2H stream on `gpu_id`, but on TPU each worker process sees one device, at index `0`. |
| Diffusion IPC tensor packing | `src/vllm_torchtpu/omni/patches.py` (`_safe_pack_tensor_if_large`) | Moves the tensor to host, then forwards the new `ipc._pack_tensor_if_large` arguments (`created=`, `enable_registered_shm=`). |
| `torchaudio` shim | `src/vllm_torchtpu/torchaudio_shim.py` (registered at the `import` stage in `patch_registry.py`) | vLLM-Omni imports `torchaudio` in its API server and in the MiniMax-H3 audio VAE. The shim provides real resampling and a `soundfile`-backed `torchaudio.load` (used for Ref2VA audio references). `Spectrogram`, `TimeStretch`, `InverseSpectrogram` and `melscale_fbanks` can be imported but raise `NotImplementedError`. |

---

## 9. Known Limitations

- Random weights only. Outputs are noise; this recipe checks execution, not quality.
- Only the small configuration above (2 steps, `256x256`, 4 s) has been validated.
- The run is eager (`--enforce-eager`).
- The disaggregated-decode deployment (`minimax_h3_disaggregated_decode`, VAE decode in a separate stage) was not validated. Its decode stage builds no diffusion attention layer, so the model-specific TPU patches are not installed there.
- With the `torchaudio` shim, API-server speed adjustment (`Spectrogram` / `TimeStretch` / `InverseSpectrogram`) raises `NotImplementedError`. Install real `torchaudio` if you need it.
