"""Run the online MiniMax-H3 T2VA pipeline (8_online) against `vllm serve --omni`.

Starts the server, waits for GET /v1/models, sends one POST /v1/videos/sync
request, saves the mp4 and its first frame, records the result in
run_status.json and stops the server.
"""

import base64
import json
import os
import subprocess
import sys
import time

import imageio_ffmpeg
import requests

# Ref2VA calls ffmpeg and ffprobe (vllm-omni reference_video.py). The TPU image
# has neither on PATH, so prepend the static build if it is there.
_FFMPEG_BIN = "/mnt/workspace/h3_tpu/bin"
if os.path.isdir(_FFMPEG_BIN):
    os.environ["PATH"] = _FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")

out_dir = "/mnt/workspace/h3_tpu/outputs/random_weights"

proc = subprocess.Popen(
    [
        "vllm",
        "serve",
        "MiniMaxAI/MiniMax-H3",
        "--omni",
        "--trust-remote-code",
        "--num-gpus",
        "8",
        "--tensor-parallel-size",
        "8",
        "--text-encoder-tp-size",
        "8",
        "--vae-patch-parallel-size",
        "8",
        "--vae-parallel-mode",
        "tile",
        "--vae-use-tiling",
        "--enforce-eager",
        "--stage-init-timeout",
        "1800",
        "--init-timeout",
        "1800",
        "--port",
        "8000",
    ],
    stdout=sys.stdout,
    stderr=sys.stderr,
    text=True,
)

ready = False
print("Starting server...", flush=True)
print("Waiting for server...", flush=True)

start = time.time()
while time.time() - start < 1800:
    try:
        res = requests.get("http://localhost:8000/v1/models")
        if res.status_code == 200:
            ready = True
            break
    except Exception:
        # Connection refused until the server is up.
        time.sleep(0.1)
    time.sleep(5)

if not ready:
    print("Server failed to start in time.")
    proc.kill()
    sys.exit(1)

print("Server ready! Sending POST /v1/videos/sync...", flush=True)
payload = {
    "prompt": "A test prompt",
    "size": "256x256",
    "seconds": "4",
    "num_inference_steps": 2,
    "extra_params": json.dumps(
        {
            "audio_mode": "native",
            "task": "t2va",
            "aspect_ratio": "1:1",
        }
    ),
}

status_entry = {
    "status": "success",
    "seconds": 0.0,
    "error": None,
    "mp4": None,
    "png": [],
}
t0 = time.time()
try:
    resp = requests.post("http://localhost:8000/v1/videos/sync", data=payload)
    print("HTTP Response code:", resp.status_code)
    if resp.status_code != 200:
        print("HTTP Response text:", resp.text)
    resp.raise_for_status()

    mp4_path = f"{out_dir}/8_online.mp4"
    content = resp.content
    ctype = resp.headers.get("content-type", "")
    if "json" in ctype:
        body = resp.json()
        b64 = None
        items = body.get("data") if isinstance(body, dict) else None
        if isinstance(items, list) and items and isinstance(items[0], dict):
            b64 = items[0].get("b64_json")
        if b64 is None and isinstance(body, dict):
            b64 = body.get("b64_json") or body.get("video")
        if not b64:
            raise RuntimeError(f"Unexpected JSON response: {str(body)[:500]}")
        content = base64.b64decode(b64)
    if len(content) < 12 or content[4:8] != b"ftyp":
        raise RuntimeError(
            f"Response is not an MP4 (content-type={ctype!r}, {len(content)} bytes)"
        )
    with open(mp4_path, "wb") as out:
        out.write(content)
    print("Saved response.", flush=True)

    # Save the first frame as a PNG.
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    png_path = f"{out_dir}/8_online_frame0.png"
    subprocess.run(
        [exe, "-y", "-i", mp4_path, "-vframes", "1", png_path], capture_output=True
    )

    # Record ffmpeg's stream lines as a check of the mp4.
    proc2 = subprocess.run(
        [exe, "-hide_banner", "-i", mp4_path], capture_output=True, text=True
    )
    stream_lines = [
        line.strip() for line in proc2.stderr.split("\n") if "Stream" in line
    ]

    status_entry["mp4"] = stream_lines
    status_entry["png"] = ["8_online_frame0.png"]

except Exception as e:
    print("Failed to get response:", e)
    status_entry["status"] = "error"
    status_entry["error"] = str(e)

status_entry["seconds"] = time.time() - t0

try:
    status_file = f"{out_dir}/run_status.json"
    run_status = {}
    if os.path.exists(status_file):
        with open(status_file) as f:
            run_status = json.load(f)
    run_status["8_online"] = status_entry
    with open(status_file, "w") as f:
        json.dump(run_status, f, indent=2)
except Exception as e:
    print("Failed to save status:", e)

print("Killing server...", flush=True)
proc.terminate()
proc.wait()
