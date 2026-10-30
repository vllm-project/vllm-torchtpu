# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Real-checkpoint smoke of the Qwen3.5 megakernel: decode, then MTP.

For each mode it starts `serve.sh` (the template's serving flags), waits for
`/health`, sends greedy single and batched completions, and checks the
server's log lines:

- `Megakernel plugin qwen35 ... loaded from ...` (the plugin loaded),
- `... megakernel bound to N unified pools (profile=..., numerics=...)`,
- `... megakernel step mix over ...: n/m MK (...) {...}` with megakernel
  steps (`mk_t1_*` for decode, `mk_t4_*` for MTP verify),

and fails on a megakernel decline. It prints a JSON summary between
`=== MK_REAL_CKPT_SMOKE_SUMMARY_BEGIN/END ===`.

Environment: `MK_SMOKE_MODE` (`decode`, `mtp` or `both`, default `both`),
`MK_SMOKE_CKPT` (a checkpoint directory; default: `Qwen/Qwen3.5-397B-A17B-FP8`
from the local Hugging Face cache, `HF_HUB_CACHE`), `MK_SMOKE_MEGAKERNEL`
(extra `VLLM_TPU_MEGAKERNEL` settings, e.g. `profile=qwen35-baseline`),
`MK_SMOKE_PORT` (default 8123).

Run it from a checkout of the repository (e.g. the recipe's NFS checkout, any
working directory): it serves that checkout's `src/` (first on `PYTHONPATH`),
checks first that `vllm_torchtpu` resolves there and not to the image's
installed package, and writes nothing outside `/tmp`.
"""

from __future__ import annotations

import ast
import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SRC_ROOT = str(HERE.parents[4])  # the directory holding `vllm_torchtpu`
CKPT_ID = "Qwen/Qwen3.5-397B-A17B-FP8"
STATS_PERIOD_S = 5
PROMPTS = [
    "The capital of France is",
    "In computer science, a hash table is a data structure that",
    "def fibonacci(n: int) -> int:\n",
    "The first five prime numbers are 2, 3, 5,",
]

LOADED_RE = re.compile(r"Megakernel plugin (\S+) (\S+) loaded from (\S+) \((.*)\)")
BOUND_RE = re.compile(
    r"(\S+) megakernel bound to (\d+) unified pools \(profile=([^,]+), numerics=(\w+)"
)
STEP_MIX_RE = re.compile(
    r"megakernel step mix over \d+s: (\d+)/(\d+) MK \([\d.]+%\) (\{.*\})"
)
DECLINE_RE = re.compile(r"megakernel declined \((\S+)\)|Megakernel .*; skipping\.")


def _log(msg: str) -> None:
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def _checkpoint() -> str:
    if ckpt := os.environ.get("MK_SMOKE_CKPT"):
        return ckpt
    hub = os.environ.setdefault("HF_HUB_CACHE", "/vllm-cache/hub")
    repo_dir = f"models--{CKPT_ID.replace('/', '--')}"
    snaps = sorted(glob.glob(os.path.join(hub, repo_dir, "snapshots", "*")))
    if not snaps:
        raise RuntimeError(f"{CKPT_ID} is not in {hub}; set MK_SMOKE_CKPT")
    return snaps[-1]


def _complete(port: int, prompt: Any, max_tokens: int = 24) -> list[str]:
    body = {
        "model": "qwen35",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 42,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        choices = json.loads(resp.read().decode("utf-8"))["choices"]
    return [c["text"] for c in sorted(choices, key=lambda c: c.get("index", 0))]


def _wait_healthy(proc: subprocess.Popen[str], port: int, timeout_s: float) -> float:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"vllm serve exited with code {proc.returncode}")
        try:
            url = f"http://127.0.0.1:{port}/health"
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if resp.status == 200:
                    return time.monotonic() - t0
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(2.0)
    raise TimeoutError("timed out waiting for vllm serve /health")


def _stop(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=60)


def tree_env() -> dict[str, str]:
    """The environment that serves this tree's `vllm_torchtpu`: its `src/`
    first on `PYTHONPATH` (ahead of the image's installed package)."""
    py_path = os.pathsep.join(p for p in (SRC_ROOT, os.environ.get("PYTHONPATH")) if p)
    return dict(os.environ, PYTHONPATH=py_path)


def check_tree_is_served() -> str:
    """Fail unless `vllm_torchtpu` resolves under this tree for the interpreter
    `vllm serve` runs (the image's installed package must not shadow it)."""
    python = sys.executable
    vllm = shutil.which("vllm")
    if vllm is not None:
        with open(vllm, encoding="utf-8", errors="replace") as f:
            first = f.readline().strip()
        if first.startswith("#!") and os.path.isfile(first[2:].split()[0]):
            python = first[2:].split()[0]
    out = subprocess.run(
        [python, "-c", "import vllm_torchtpu; print(vllm_torchtpu.__file__)"],
        env=tree_env(),
        cwd="/tmp",
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if not Path(out).resolve().is_relative_to(Path(SRC_ROOT).resolve()):
        raise RuntimeError(
            f"{python} imports vllm_torchtpu from {out}, not from this tree "
            f"({SRC_ROOT}): the image's installed package shadows it"
        )
    return out


def run_mode(mode: str, ckpt: str, port: int) -> dict[str, Any]:
    """Serve `ckpt` in `mode` and check the megakernel's log lines."""
    mk_cfg = f"stats_period_s={STATS_PERIOD_S}"
    if extra := os.environ.get("MK_SMOKE_MEGAKERNEL"):
        mk_cfg = f"{extra},{mk_cfg}"
    env = dict(
        tree_env(),
        MODEL=ckpt,
        MODE=mode,
        PORT=str(port),
        VLLM_TPU_MEGAKERNEL=mk_cfg,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        VLLM_ENGINE_READY_TIMEOUT_S="3600",
    )
    lines: list[str] = []
    proc = subprocess.Popen(
        ["bash", str(HERE / "serve.sh")],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )

    def pump() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            lines.append(line)

    threading.Thread(target=pump, daemon=True).start()
    try:
        startup_s = _wait_healthy(proc, port, timeout_s=3600.0)
        single = [_complete(port, p)[0] for p in PROMPTS]
        batch = _complete(port, PROMPTS)
        # One more request after a full stats period, so the step mix logs.
        time.sleep(STATS_PERIOD_S + 1)
        _complete(port, PROMPTS[0], max_tokens=8)
        time.sleep(2.0)
    finally:
        _stop(proc)

    text = "".join(lines)
    loaded = LOADED_RE.search(text)
    bound = BOUND_RE.search(text)
    mix: dict[str, int] = {}
    mk_steps = total_steps = 0
    for m in STEP_MIX_RE.finditer(text):
        mk_steps += int(m.group(1))
        total_steps += int(m.group(2))
        for k, v in ast.literal_eval(m.group(3)).items():
            mix[k] = mix.get(k, 0) + int(v)
    prefix = "mk_t4_" if mode == "mtp" else "mk_t1_"
    declines = sorted({m.group(0) for m in DECLINE_RE.finditer(text)})
    errors = []
    if loaded is None:
        errors.append("no 'Megakernel plugin ... loaded' line")
    if bound is None:
        errors.append("no 'megakernel bound to' line")
    if not any(k.startswith(prefix) and v > 0 for k, v in mix.items()):
        errors.append(f"no {prefix}* megakernel steps in the step mix {mix}")
    if declines:
        errors.append(f"megakernel declined: {declines}")
    return {
        "mode": mode,
        "startup_s": round(startup_s, 1),
        "loaded": loaded.group(0) if loaded else None,
        "bound": bound.group(0) if bound else None,
        "profile": bound.group(3) if bound else None,
        "numerics": bound.group(4) if bound else None,
        "step_mix": mix,
        "mk_steps": mk_steps,
        "total_steps": total_steps,
        "single_completions": single,
        "batch_completions": batch,
        "errors": errors,
    }


def main() -> int:
    mode = os.environ.get("MK_SMOKE_MODE", "both").strip().lower()
    modes = ["decode", "mtp"] if mode == "both" else [mode]
    if any(m not in ("decode", "mtp") for m in modes):
        raise SystemExit(f"MK_SMOKE_MODE must be decode, mtp or both, got {mode!r}")
    served = check_tree_is_served()
    ckpt = _checkpoint()
    port = int(os.environ.get("MK_SMOKE_PORT", "8123"))
    _log(f"[MK_SMOKE] modes={modes} ckpt={ckpt} vllm_torchtpu={served}")
    results = [run_mode(m, ckpt, port) for m in modes]
    _log("=== MK_REAL_CKPT_SMOKE_SUMMARY_BEGIN ===")
    _log(json.dumps(results, indent=1, sort_keys=True))
    _log("=== MK_REAL_CKPT_SMOKE_SUMMARY_END ===")
    failed = [(r["mode"], e) for r in results for e in r["errors"]]
    for m, e in failed:
        _log(f"[MK_SMOKE] FAIL {m}: {e}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
