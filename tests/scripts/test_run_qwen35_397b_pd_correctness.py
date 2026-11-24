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
"""scripts/vllm/integration/run_qwen35_397b_pd_correctness.sh against a fake
prefill/decode pair: every stage fails when it measures a failure or nothing,
and the script's exit status carries it."""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import shutil
import stat
import subprocess
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]
_CONFIG = "qwen3.5-397b-fp8-pd-1p1d"
_BASELINE = pathlib.Path(
    f"scripts/vllm/benchmarking/baselines/eval/{_CONFIG}.gsm8k.baseline.json"
)
_COPIED = [
    "scripts/vllm/integration/run_qwen35_397b_pd_correctness.sh",
    "scripts/vllm/integration/qwen35_pd_checks.py",
    ".buildkite/kubernetes/check_completions.py",
    "scripts/vllm/benchmarking/run_eval_flow.sh",
    "scripts/vllm/benchmarking/run_benchmarks.sh",
    "scripts/vllm/benchmarking/check_regression.py",
    "scripts/vllm/benchmarking/cleanup_server.sh",
    "scripts/vllm/benchmarking/configs/qwen3.5-397b-fp8-pd.sh",
    f"scripts/vllm/benchmarking/configs/{_CONFIG}.sh",
    str(_BASELINE),
]
_MEASURED = {
    "_source": {"url": "https://example.invalid/run", "description": "a test"},
    "gsm8k": {
        "exact_match,strict-match": 0.97,
        "exact_match_stderr,strict-match": 0.0047,
    },
}

_spec = importlib.util.spec_from_file_location(
    "pd_checks_for_stages", _REPO / "scripts/vllm/integration/qwen35_pd_checks.py"
)
assert _spec is not None and _spec.loader is not None
checks = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = checks
_spec.loader.exec_module(checks)

# Stands in for lm_eval: records its arguments and writes a results file the
# way lm-eval nests it under --output_path, unless told to fail.
_FAKE_LM_EVAL = """#!{python}
import json, os, pathlib, sys
args = sys.argv[1:]
pathlib.Path(os.environ["FAKE_LM_EVAL_ARGS"]).write_text(json.dumps(args))
mode = os.environ.get("FAKE_LM_EVAL", "good")
if mode == "noresults":
    sys.exit(0)
if mode == "retried":
    print("WARNING API request failed with error message: 503. Retrying...")
out = pathlib.Path(args[args.index("--output_path") + 1]) / "model"
out.mkdir(parents=True, exist_ok=True)
task = args[args.index("--tasks") + 1]
effective = 100 if mode == "partial" else 1319
(out / "results_2026.json").write_text(json.dumps({{
    "results": {{task: {{"exact_match,strict-match": float(os.environ["FAKE_SCORE"]),
                       "exact_match_stderr,strict-match": 0.0047}}}},
    "n-samples": {{task: {{"original": 1319, "effective": effective}}}}}}))
"""

_PAGE = (
    "INFO derivation: backend_supported_kernel_block_sizes={} -> input_block_size=1280"
)
_MISMATCH = (
    "ERROR Remote Controller Server execution failed: Layout fingerprint "
    "mismatch between source and destination"
)


def _executable(path: pathlib.Path, text: str) -> None:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


class _Pair:
    """A proxy (/v1/completions), a prefill and two decode /metrics servers."""

    def __init__(self, knobs: set[str]):
        self.knobs = knobs
        self.servers = []
        for role in ("proxy", "prefill", "decode", "decode"):
            server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler(role))
            threading.Thread(target=server.serve_forever, daemon=True).start()
            self.servers.append(server)

    def url(self, i: int) -> str:
        return f"http://127.0.0.1:{self.servers[i].server_address[1]}"

    def close(self) -> None:
        for server in self.servers:
            server.shutdown()
            server.server_close()

    def _metrics(self, role: str) -> str:
        knobs = self.knobs
        if role == "prefill":
            return 'vllm:prefix_cache_hits_total{engine="0"} 6912.0\n'
        accepted = 1500.0 if "low_mtp" in knobs else 2250.0
        transferred = 0.0 if "no_transfer" in knobs else 90000.0
        spec = "vllm:spec_decode_num_"
        source = "vllm:prompt_tokens_by_source_total"
        return "\n".join(
            [
                f'{spec}drafts_total{{engine="0"}} 1000.0',
                f'{spec}draft_tokens_total{{engine="0"}} 3000.0',
                f'{spec}accepted_tokens_total{{engine="0"}} {accepted}',
                f'{source}{{source="external_kv_transfer"}} {transferred}',
                f'{source}{{source="local_compute"}} 100.0',
                f'{source}{{source="local_cache_hit"}} 7680.0',
            ]
        )

    def _handler(self, role: str):
        pair = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, body: str) -> None:
                data = body.encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._send(pair._metrics(role))

            def do_POST(self):
                request = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"]))
                )
                answers = {
                    "The capital of France is": " Paris.",
                    "1, 2, 3, 4, 5, 6,": " 7, 8",
                    "The chemical symbol for gold is": " Au.",
                }
                text = answers.get(request.get("prompt", ""), " 4242")
                if "wrong_answer" in pair.knobs:
                    text = " banana"
                usage = {
                    "prompt_tokens": 8000,
                    "prompt_tokens_details": {"cached_tokens": 7680},
                }
                self._send(json.dumps({"choices": [{"text": text}], "usage": usage}))

        return Handler


@pytest.fixture
def sandbox(tmp_path: pathlib.Path) -> Iterator[pathlib.Path]:
    root = tmp_path / "repo"
    for rel in _COPIED:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_REPO / rel, root / rel)
    shutil.copytree(
        _REPO / "scripts/vllm/benchmarking/lm_eval_tasks",
        root / "scripts/vllm/benchmarking/lm_eval_tasks",
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _executable(bin_dir / "lm_eval", _FAKE_LM_EVAL.format(python=sys.executable))
    _executable(bin_dir / "stdbuf", '#!/bin/bash\nshift\nexec "$@"\n')
    (tmp_path / "stub" / "vllm").mkdir(parents=True)
    (tmp_path / "stub" / "vllm" / "__init__.py").write_text("")
    yield root


def _leg_status(path: pathlib.Path, *lines: str) -> str:
    path.write_text(json.dumps(checks.leg_status("\n".join(lines))))
    return str(path)


def _run(root: pathlib.Path, knobs: str = "", baseline: dict | None = None, **env: str):
    tmp = root.parent
    if baseline is not None:
        (root / _BASELINE).write_text(json.dumps(baseline))
    pair = _Pair(set(knobs.split(",")))
    decode_page = [_PAGE.format("[128]")] + ([_MISMATCH] if "mismatch" in knobs else [])
    run_env = {
        "PATH": f"{tmp / 'bin'}:{os.environ['PATH']}",
        "PYTHONPATH": str(tmp / "stub"),
        "HOME": str(tmp),
        "FAKE_LM_EVAL_ARGS": str(tmp / "lm_eval_args.json"),
        "FAKE_SCORE": "0.97",
        "LM_EVAL_REQUEST_CACHE_ROOT": str(tmp / "request_cache"),
        "QWEN35_PD_CONFIG": _CONFIG,
        "PROXY_HOST": "127.0.0.1",
        "PROXY_PORT": str(pair.servers[0].server_address[1]),
        "PREFILL_URL": pair.url(1),
        "DECODE_URLS": f"{pair.url(2)} {pair.url(3)}",
        "PREFILL_LEG_STATUS": _leg_status(tmp / "p.json", _PAGE.format("[128]")),
        "DECODE_LEG_STATUS": _leg_status(tmp / "d.json", *decode_page),
        "LOG_DIR": str(tmp / "logs"),
        "LEG_STATUS_SETTLE_S": "0",
        **env,
    }
    shutil.rmtree(tmp / "logs", ignore_errors=True)
    try:
        proc = subprocess.run(
            ["bash", "scripts/vllm/integration/run_qwen35_397b_pd_correctness.sh"],
            cwd=root,
            env=run_env,
            capture_output=True,
            text=True,
            timeout=300,
        )
    finally:
        pair.close()
    stages = {}
    for line in proc.stdout.splitlines():
        if line.startswith("STAGE_RESULT "):
            _, name, rc = line.split()
            stages[name] = int(rc)
    return proc, stages


def test_a_good_pair_passes_every_stage_on_the_whole_split(sandbox):
    proc, stages = _run(sandbox, baseline=_MEASURED)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-2000:]
    assert stages == dict.fromkeys(
        ["completions", "prefix_hit", "gsm8k", "transfer", "mtp", "legs"], 0
    )
    args = json.loads((sandbox.parent / "lm_eval_args.json").read_text())
    assert args[args.index("--tasks") + 1] == "gsm8k"
    assert "--limit" not in args
    gen_kwargs = json.loads(args[args.index("--gen_kwargs") + 1])
    assert gen_kwargs["chat_template_kwargs"] == {"enable_thinking": True}


@pytest.mark.parametrize(
    "knobs,baseline,env,failed",
    [
        # A pending baseline fails the gate unless the step asks it to warn.
        ("", None, {}, "gsm8k"),
        ("", _MEASURED, {"FAKE_SCORE": "0.90"}, "gsm8k"),
        ("", _MEASURED, {"FAKE_LM_EVAL": "noresults"}, "gsm8k"),
        ("", _MEASURED, {"FAKE_LM_EVAL": "partial"}, "gsm8k"),
        ("", _MEASURED, {"FAKE_LM_EVAL": "retried"}, "gsm8k"),
        ("wrong_answer", _MEASURED, {}, "completions"),
        ("low_mtp", _MEASURED, {}, "mtp"),
        ("no_transfer", _MEASURED, {}, "transfer"),
        ("mismatch", _MEASURED, {}, "legs"),
    ],
    ids=[
        "pending",
        "low_score",
        "no_results",
        "partial_split",
        "retried_request",
        "wrong_answer",
        "low_mtp",
        "no_transfer",
        "mismatch",
    ],
)
def test_a_failing_stage_fails_the_script(sandbox, knobs, baseline, env, failed):
    proc, stages = _run(sandbox, knobs, baseline, **env)
    assert proc.returncode == 1, proc.stdout[-3000:]
    assert stages[failed] != 0, stages
    assert {k for k, v in stages.items() if v} == {failed}


def test_a_pending_baseline_only_warns_under_eval_soft_fail(sandbox):
    proc, stages = _run(sandbox, EVAL_SOFT_FAIL="1")
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert set(stages.values()) == {0}
    assert "baseline is pending" in proc.stdout
