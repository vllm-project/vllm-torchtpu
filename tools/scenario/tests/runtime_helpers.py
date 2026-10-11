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
"""Helpers for runtime tests: stub programs on PATH, their records, a small scenario.

Imported by name, like plan_fixtures.py. The stubs (stubs/stubs.py) need no
network, TPU or vLLM.
"""

import json
import os
import pathlib
import socket
import sys
import threading
import time
from typing import Any

STUBS = pathlib.Path(__file__).resolve().parent / "stubs" / "stubs.py"
PROGRAMS = ("vllm", "python3")
# A stand-in vllm package: just runai_utils.ObjectStorageModel.
SITE = pathlib.Path(__file__).resolve().parent / "stubs" / "site"
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def stub_env(tmp_path: pathlib.Path, **knobs: str) -> dict[str, str]:
    """The env a run gets: stub programs first on PATH, records under tmp_path."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for program in PROGRAMS:
        shim = bin_dir / program
        # -S: the stubs need only the standard library, and start faster.
        shim.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" -S "{STUBS}" {program} "$@"\n'
        )
        shim.chmod(0o755)
    records = tmp_path / "records"
    records.mkdir(exist_ok=True)
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("SCENARIO_", "STUB_"))
    }
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    env["STUB_RECORDS"] = str(records)
    env["PYTHONPATH"] = str(SITE)
    env["STUB_PULLED_DIR"] = str(tmp_path / "pulled")
    env.update(knobs)
    return env


def records(env: dict[str, str], program: str | None = None) -> list[dict[str, Any]]:
    """Every recorded call in order, or those of one program."""
    found = [
        json.loads(path.read_text())
        for path in sorted(pathlib.Path(env["STUB_RECORDS"]).glob("*.json"))
    ]
    return [r for r in found if program is None or r["program"] == program]


def gone(pid: int, within: float = 5.0) -> bool:
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def no_stray_processes(env: dict[str, str]) -> bool:
    """Whether every recorded process and engine child has gone."""
    pids = []
    for found in records(env):
        pids.append(found["pid"])
        if "engine_pid" in found:
            pids.append(found["engine_pid"])
    return all(gone(pid) for pid in pids)


PERCENTILES = ["ttft", "tpot", "itl", "e2el"]


def tiny() -> dict[str, Any]:
    """One chip, perf and eval, with the values defaults.yaml adds."""
    return {
        "schema": 1,
        "name": "tiny",
        "server": {
            "model": {"name": "test/tiny"},
            "accelerator": "tpu7x",
            "topology": "1x1x1",
            "hosts": 1,
            "chips_per_host": 1,
            "startup_timeout": "10m",
            "args": {
                "tensor-parallel-size": 1,
                "data-parallel-size": 1,
                "max-model-len": 4096,
                "max-num-batched-tokens": 1024,
                "max-num-seqs": 64,
                "async-scheduling": True,
                "kv-cache-dtype": "fp8",
            },
            "env": {"TINY_KNOB": "1"},
        },
        "tasks": {
            "perf": {
                "dataset": {"name": "random"},
                "isl_osl": [[128, 128]],
                "concurrency": [1, 4],
                "num_prompts": {"per_concurrency": 2},
                "random_range_ratio": 0.5,
                "range_ratio_style": "min",
                "warmup_runs": 1,
                "temperature": 0,
                "percentile_metrics": PERCENTILES,
                "client_args": {"backend": "vllm", "ignore-eos": True, "seed": 42},
                "tolerance": 0.05,
                "baseline": {
                    "isl128_osl128_c1": {
                        "completed": 2,
                        "median_tpot_ms": 10.0,
                        "output_token_throughput": 500.0,
                    },
                    "isl128_osl128_c4": {"completed": 8, "median_tpot_ms": 10.0},
                },
            },
            "eval": {
                "lm_eval": {
                    "tasks": [
                        "mmlu_pro",
                        {
                            "name": "gsm8k",
                            "limit": 5,
                            "env": {"HF_ALLOW_CODE_EVAL": "1"},
                        },
                    ],
                    "model": "local-chat-completions",
                    "model_args": {"num_concurrent": 8},
                    "seed": "0,1234,None,1234",
                    "limit": 10,
                    "apply_chat_template": True,
                    "fewshot_as_multiturn": True,
                    "gen_kwargs": {"chat_template_kwargs": {"enable_thinking": False}},
                },
                "tolerance": 0.02,
                "baseline": {
                    "mmlu_pro": {"exact_match,custom-extract": 0.8},
                    "gsm8k": {"exact_match,custom-extract": 0.8},
                },
            },
        },
    }


def signal_when_serving(summary: pathlib.Path, signum: int) -> threading.Thread:
    """Send `signum` to this process once the run's summary says it serves."""

    def watch() -> None:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                if json.loads(summary.read_text())["state"] == "serving":
                    break
            except (OSError, ValueError):
                pass
            time.sleep(0.02)
        os.kill(os.getpid(), signum)

    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    return thread


def serve_only(data: dict[str, Any]) -> dict[str, Any]:
    return {**data, "tasks": {}}
