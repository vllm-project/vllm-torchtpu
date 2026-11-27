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
"""Tests for 1P1D disaggregated evaluation flows on Qwen3.5-397B and Kimi-K3."""

import http.server
import json
import pathlib
import socket
import stat
import subprocess
import threading
from typing import Any

import pytest
import yaml  # pyrefly: ignore[missing-import]

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DISAGG_BENCHMARK = _REPO_ROOT / ".buildkite" / "kubernetes" / "disagg_benchmark.sh"
_K3_CLIENT_POD = _REPO_ROOT / ".buildkite" / "kubernetes" / "k3_agentx_client_pod.sh"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _valid_schema_reply(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a schema-valid JSON object for any json_schema_default.yaml case."""
    rf = payload.get("response_format") or {}
    schema = (rf.get("json_schema") or {}).get("schema") or {}
    props = schema.get("properties", {})
    if "users" in props:
        obj: Any = {
            "users": [
                {
                    "name": "a",
                    "age": 1,
                    "email": "a@b.c",
                    "is_active": True,
                    "role": "user",
                }
                for _ in range(10)
            ]
        }
    elif "expressions" in props:
        obj = {
            "expressions": [
                {"ast": {"op": "+", "left": 1, "right": 2}, "description": "d"}
                for _ in range(10)
            ]
        }
    elif "house" in props:
        obj = {
            "name": "Harry Potter",
            "house": "Gryffindor",
            "blood status": "Half-blood",
            "occupation": "auror",
            "wand": {"wood": "Holly", "core": "Phoenix", "length": 11},
            "alive": "Alive",
            "patronus": "Stag",
            "bogart": "Dementor",
        }
    else:
        obj = {
            "tool_name": "query_logs",
            "arguments": {"service": "gateway", "limit": 10, "severity": "ERROR"},
        }
    return {
        "choices": [{"message": {"content": json.dumps(obj)}, "finish_reason": "stop"}]
    }


class _MockDisaggEndpoint:
    """Local HTTP endpoint serving health, metrics, models, and chat completions."""

    def __init__(self, prefix: str = "", model_id: str = "moonshotai/Kimi-K3"):
        self.prefix = prefix
        self.model_id = model_id
        self.requests: list[tuple[str, str]] = []
        self.chat_payloads: list[dict[str, Any]] = []
        endpoint = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def _send(self, code: int, body: bytes, content_type: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                endpoint.requests.append(("GET", self.path))
                if self.path in (
                    f"{endpoint.prefix}/health",
                    "/health",
                    "/healthcheck",
                ):
                    self._send(200, b'{"status":"ok"}', "application/json")
                elif self.path in (f"{endpoint.prefix}/metrics", "/metrics"):
                    self._send(
                        200,
                        b'vllm:cache_config_info{block_size="1296"} 1\n'
                        b"vllm:num_requests_running 0\n"
                        b"vllm:num_requests_waiting 0\n",
                        "text/plain",
                    )
                elif self.path == f"{endpoint.prefix}/v1/models":
                    card = {
                        "data": [
                            {
                                "id": endpoint.model_id,
                                "max_model_len": 65536,
                            }
                        ]
                    }
                    self._send(
                        200, json.dumps(card).encode("utf-8"), "application/json"
                    )
                else:
                    self._send(404, b"{}", "application/json")

            def do_POST(self) -> None:
                endpoint.requests.append(("POST", self.path))
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                if self.path == f"{endpoint.prefix}/v1/chat/completions":
                    endpoint.chat_payloads.append(payload)
                    reply = _valid_schema_reply(payload)
                    self._send(
                        200, json.dumps(reply).encode("utf-8"), "application/json"
                    )
                else:
                    self._send(404, b"{}", "application/json")

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = int(self._server.server_address[1])
        self.url = f"http://127.0.0.1:{self.port}{prefix}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def make_endpoint():
    endpoints: list[_MockDisaggEndpoint] = []

    def _make(
        prefix: str = "", model_id: str = "moonshotai/Kimi-K3"
    ) -> _MockDisaggEndpoint:
        ep = _MockDisaggEndpoint(prefix=prefix, model_id=model_id)
        endpoints.append(ep)
        return ep

    yield _make
    for ep in endpoints:
        ep.close()


def _install_disagg_stubs(bin_dir: pathlib.Path, endpoint_port: int) -> None:
    """Install stub vllm-router, vllm, lm_eval, pip, and buildkite-agent."""
    router_stub = bin_dir / "vllm-router"
    router_stub.write_text(
        f"""#!/usr/bin/env python3
import http.server, json, pathlib, sys, urllib.request

log_dir = pathlib.Path({str(bin_dir.parent)!r})
(log_dir / "vllm-router.args").write_text("\\n".join(sys.argv[1:]) + "\\n")
args = sys.argv[1:]
port = int(args[args.index("--port") + 1])
upstream = "http://127.0.0.1:{endpoint_port}"

class Proxy(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def do_GET(self):
        with urllib.request.urlopen(upstream + self.path) as r:
            body = r.read()
            self.send_response(r.status)
            ctype = r.headers.get("Content-Type", "application/json")
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        req = urllib.request.Request(
            upstream + self.path,
            data=self.rfile.read(n),
            headers={{"Content-Type": "application/json"}},
            method="POST",
        )
        with urllib.request.urlopen(req) as r:
            body = r.read()
            self.send_response(r.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

http.server.ThreadingHTTPServer(("127.0.0.1", port), Proxy).serve_forever()
"""
    )
    router_stub.chmod(router_stub.stat().st_mode | stat.S_IXUSR)

    vllm_stub = bin_dir / "vllm"
    vllm_stub.write_text(
        """#!/bin/bash
printf '%s\\n' "$@" >> "$STUB_LOG_DIR/vllm.args"
out=""
for arg in "$@"; do
    case "$arg" in
        --result-file=*) out="${arg#--result-file=}" ;;
    esac
done
if [ -z "$out" ]; then
    while [[ $# -gt 0 ]]; do
        case $1 in
            --result-filename) out="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
fi
if [ -n "$out" ]; then
    mkdir -p "$(dirname "$out")"
    printf '%s\n' '{"completed": 8, "failed": 0, '\
        '"output_throughput": 100.0, "total_token_throughput": 800.0, '\
        '"median_tpot_ms": 20.0}' > "$out"
fi
"""
    )
    vllm_stub.chmod(vllm_stub.stat().st_mode | stat.S_IXUSR)

    lm_eval_stub = bin_dir / "lm_eval"
    lm_eval_stub.write_text(
        """#!/bin/bash
printf '%s\\n' "$@" >> "$STUB_LOG_DIR/lm_eval.args"
task=""
out=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --tasks) task="$2"; shift 2 ;;
        --output_path) out="$2"; shift 2 ;;
        *) shift ;;
    esac
done
cat > "$out/results_${task}.json" << JSON
{
  "results": {
    "${task}": {
      "exact_match,custom-extract": 0.84,
      "exact_match_stderr,custom-extract": 0.01,
      "exact_match,flexible-extract": 0.92,
      "exact_match_stderr,flexible-extract": 0.01
    }
  }
}
JSON
"""
    )
    lm_eval_stub.chmod(lm_eval_stub.stat().st_mode | stat.S_IXUSR)

    for noop in ("pip", "buildkite-agent", "sleep"):
        p = bin_dir / noop
        p.write_text(
            '#!/bin/bash\nprintf "%s\\n" "$@" >> "$STUB_LOG_DIR/'
            + noop
            + '.args"\nexit 0\n'
        )
        p.chmod(p.stat().st_mode | stat.S_IXUSR)


def test_qwen35_disagg_benchmark_starts_vllm_router_and_runs_eval_flow(
    tmp_path, make_endpoint
):
    endpoint = make_endpoint(model_id="Qwen/Qwen3.5-397B-A17B-FP8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _install_disagg_stubs(bin_dir, endpoint.port)

    fake_site = tmp_path / "site"
    (fake_site / "vllm").mkdir(parents=True)
    (fake_site / "vllm" / "__init__.py").write_text("")
    proxy_port = _free_port()

    proc = subprocess.run(
        ["bash", str(_DISAGG_BENCHMARK)],
        cwd=str(_REPO_ROOT),
        env={
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "PYTHONPATH": str(fake_site),
            "STUB_LOG_DIR": str(tmp_path),
            "WORKLOAD_NAME": "test-qwen",
            "PREFILL_HOST": "127.0.0.1",
            "DECODE_HOST": "127.0.0.1",
            "ENGINE_PORT": str(endpoint.port),
            "PROXY_PORT": str(proxy_port),
            "REPO_DIR": str(_REPO_ROOT),
            "ARTIFACTS_DIR": str(tmp_path / "artifacts"),
            "BENCHMARK_RESULTS_DIR": str(tmp_path / "bench_cells"),
            "LM_EVAL_REQUEST_CACHE_ROOT": str(tmp_path / "cache"),
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

    router_args = (tmp_path / "vllm-router.args").read_text().splitlines()
    assert "--vllm-pd-disaggregation" in router_args
    assert f"http://127.0.0.1:{endpoint.port}" in router_args

    # Accuracy suites (mmlu_pro, gsm8k, json_schema) ran through run_eval_flow.sh
    # against the router's --base-url and passed their regression checks.
    eval_dir = tmp_path / "artifacts" / "qwen3.5-397b-fp8-1p1d"
    assert (eval_dir / "results_mmlu_pro.json").exists()
    assert (eval_dir / "results_gsm8k.json").exists()
    assert (eval_dir / "results_json_schema.json").exists()
    assert (eval_dir / "eval_check_mmlu_pro.md").exists()
    assert (eval_dir / "eval_check_json_schema.md").exists()
    bk_args = (tmp_path / "buildkite-agent.args").read_text().splitlines()
    assert "artifacts/**/*" in bk_args
    assert f"{tmp_path / 'artifacts'}/**/*" not in bk_args


def test_k3_agentx_client_pod_uses_vllm_router_and_runs_eval_flow(
    tmp_path, make_endpoint
):
    endpoint = make_endpoint(model_id="moonshotai/Kimi-K3")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _install_disagg_stubs(bin_dir, endpoint.port)

    fake_site = tmp_path / "site"
    (fake_site / "vllm").mkdir(parents=True)
    (fake_site / "vllm" / "__init__.py").write_text("")
    proxy_port = _free_port()

    proc = subprocess.run(
        ["bash", str(_K3_CLIENT_POD)],
        cwd=str(_REPO_ROOT),
        env={
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "PYTHONPATH": str(fake_site),
            "STUB_LOG_DIR": str(tmp_path),
            "WORKLOAD_NAME": "test-k3",
            "PREFILL_HOST": "127.0.0.1",
            "DECODE_HOST": "127.0.0.1",
            "PORT": str(endpoint.port),
            "PROXY_PORT": str(proxy_port),
            "REPO_DIR": str(_REPO_ROOT),
            "ARTIFACTS_DIR": str(tmp_path / "artifacts"),
            "AGENTX_CHIPS": "32",
            "AGENTX_SKIP_REPLAY": "1",
            "AGENTX_SKIP_POST_EVAL": "1",
            "LM_EVAL_REQUEST_CACHE_ROOT": str(tmp_path / "cache"),
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

    # Default proxy in kimi-k3-tp8-dp4-ep-agentx.sh is vllm-router.
    router_args = (tmp_path / "vllm-router.args").read_text().splitlines()
    assert "--vllm-pd-disaggregation" in router_args

    eval_dir = tmp_path / "artifacts" / "kimi-k3-tp8-dp4-ep-agentx"
    assert (eval_dir / "results_mmlu_pro.json").exists()
    assert (eval_dir / "results_json_schema.json").exists()
    assert (eval_dir / "eval_check_mmlu_pro.md").exists()
    assert (eval_dir / "eval_check_json_schema.md").exists()
    # Kimi-K3 tokenizer reads thinking=False alongside enable_thinking=False.
    assert endpoint.chat_payloads
    assert all(
        pl.get("chat_template_kwargs") == {"enable_thinking": False, "thinking": False}
        for pl in endpoint.chat_payloads
    )


def test_pipeline_perf_has_qwen35_and_kimi_k3_1p1d_steps():
    pipeline_path = _REPO_ROOT / ".buildkite" / "pipeline_perf.yml"
    doc = yaml.safe_load(pipeline_path.read_text())
    steps = [s for s in doc["steps"][0]["steps"] if isinstance(s, dict)]
    qwen_matches = [
        s
        for s in steps
        if any("qwen3.5-397b-1p1d.yaml" in cmd for cmd in s.get("commands", []))
    ]
    k3_matches = [
        s
        for s in steps
        if any("kimi-k3-agentx-1p1d.yaml" in cmd for cmd in s.get("commands", []))
    ]
    assert len(qwen_matches) == 1
    assert len(k3_matches) == 1
    for step in (*qwen_matches, *k3_matches):
        assert step.get("if") == 'build.env("NIGHTLY") == "1"'
        deps = step.get("depends_on")
        if isinstance(deps, str):
            deps = [deps]
        assert deps == ["build_docker"]


def test_qwen35_disagg_benchmark_perf_suites_delegates_to_run_eval_flow(
    tmp_path, make_endpoint
):
    endpoint = make_endpoint(model_id="Qwen/Qwen3.5-397B-A17B-FP8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _install_disagg_stubs(bin_dir, endpoint.port)

    fake_site = tmp_path / "site"
    (fake_site / "vllm").mkdir(parents=True)
    (fake_site / "vllm" / "__init__.py").write_text("")
    proxy_port = _free_port()
    bench_cells = tmp_path / "bench_cells"

    proc = subprocess.run(
        ["bash", str(_DISAGG_BENCHMARK)],
        cwd=str(_REPO_ROOT),
        env={
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "PYTHONPATH": str(fake_site),
            "STUB_LOG_DIR": str(tmp_path),
            "WORKLOAD_NAME": "test-qwen",
            "PREFILL_HOST": "127.0.0.1",
            "DECODE_HOST": "127.0.0.1",
            "ENGINE_PORT": str(endpoint.port),
            "PROXY_PORT": str(proxy_port),
            "REPO_DIR": str(_REPO_ROOT),
            "ARTIFACTS_DIR": str(tmp_path / "artifacts"),
            "BENCHMARK_RESULTS_DIR": str(bench_cells),
            "DISAGG_PERF_SUITES": "bench_serve",
            "DISAGG_EVAL_SUITES": "none",
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    # The only baseline is the single-host config's, which a 1P1D pair is not
    # comparable to, so the speed run is recorded but not gated: the stub's
    # 100 tok/s, far below that baseline, still exits 0.
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "=== Checking Perf Regression ===" not in proc.stdout
    assert list(bench_cells.glob("pd_disagg_*.json")) == []
    eval_dir = tmp_path / "artifacts" / "qwen3.5-397b-fp8-1p1d"
    assert sorted(eval_dir.glob("isl*_osl*_c*.json")), (
        "expected speed result JSON(s) under artifacts dir"
    )
    vllm_args = (tmp_path / "vllm.args").read_text().splitlines()
    assert "--base-url" in vllm_args
    assert f"http://127.0.0.1:{proxy_port}" in vllm_args


def test_k3_client_pod_dedupes_gsm8k_and_skips_flow_on_eval_smoke(
    tmp_path, make_endpoint
):
    endpoint = make_endpoint(model_id="moonshotai/Kimi-K3")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _install_disagg_stubs(bin_dir, endpoint.port)

    ix_dir = tmp_path / "ix"
    ix_dir.mkdir()
    (ix_dir / "sweep_lib.sh").symlink_to(
        _REPO_ROOT / "scripts" / "vllm" / "benchmark" / "inferencex" / "sweep_lib.sh"
    )
    agentx_client = ix_dir / "agentx_client.sh"
    agentx_client.write_text(
        """#!/bin/bash
printf '%s\\n' "$@" >> "$STUB_LOG_DIR/agentx_client.args"
res=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --results-dir) res="$2"; shift 2 ;;
        *) shift ;;
    esac
done
mkdir -p "$res/gsm8k_c4/model"
cat > "$res/gsm8k_c4/model/results_1.json" << 'JSON'
{"results": {"gsm8k": {"exact_match,flexible-extract": 0.94}}}
JSON
"""
    )
    agentx_client.chmod(agentx_client.stat().st_mode | stat.S_IXUSR)

    agentx_eval_client = ix_dir / "agentx_eval_client.sh"
    agentx_eval_client.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$@" >> "$STUB_LOG_DIR/agentx_eval_client.args"\n'
    )
    agentx_eval_client.chmod(agentx_eval_client.stat().st_mode | stat.S_IXUSR)

    fake_site = tmp_path / "site"
    (fake_site / "vllm").mkdir(parents=True)
    (fake_site / "vllm" / "__init__.py").write_text("")
    proxy_port = _free_port()

    # 1. EVAL_ONLY=0: agentx_client.sh runs GSM8K once (copied to results_gsm8k.json)
    #    and run_eval_flow.sh runs only mmlu_pro + json_schema (never gsm8k twice).
    proc = subprocess.run(
        ["bash", str(_K3_CLIENT_POD)],
        cwd=str(_REPO_ROOT),
        env={
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "PYTHONPATH": str(fake_site),
            "STUB_LOG_DIR": str(tmp_path),
            "WORKLOAD_NAME": "test-k3",
            "PREFILL_HOST": "127.0.0.1",
            "DECODE_HOST": "127.0.0.1",
            "PORT": str(endpoint.port),
            "PROXY_PORT": str(proxy_port),
            "REPO_DIR": str(_REPO_ROOT),
            "IX_DIR": str(ix_dir),
            "ARTIFACTS_DIR": str(tmp_path / "artifacts"),
            "AGENTX_CHIPS": "32",
            "AGENTX_SKIP_POST_EVAL": "1",
            "LM_EVAL_REQUEST_CACHE_ROOT": str(tmp_path / "cache"),
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    lm_args = (tmp_path / "lm_eval.args").read_text().splitlines()
    tasks = [lm_args[i + 1] for i, a in enumerate(lm_args) if a == "--tasks"]
    assert tasks == ["mmlu_pro"]
    eval_dir = tmp_path / "artifacts" / "kimi-k3-tp8-dp4-ep-agentx"
    assert (eval_dir / "results_gsm8k.json").exists()

    # 2. EVAL_ONLY=1 + AGENTX_EVAL_SMOKE=1: runs agentx_eval_client.sh smoke only,
    #    without launching a second full run_eval_flow.sh sweep.
    smoke_logs = tmp_path / "smoke_logs"
    smoke_logs.mkdir()
    proc_smoke = subprocess.run(
        ["bash", str(_K3_CLIENT_POD)],
        cwd=str(_REPO_ROOT),
        env={
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "PYTHONPATH": str(fake_site),
            "STUB_LOG_DIR": str(smoke_logs),
            "WORKLOAD_NAME": "test-k3-smoke",
            "PREFILL_HOST": "127.0.0.1",
            "DECODE_HOST": "127.0.0.1",
            "PORT": str(endpoint.port),
            "PROXY_PORT": str(_free_port()),
            "REPO_DIR": str(_REPO_ROOT),
            "IX_DIR": str(ix_dir),
            "ARTIFACTS_DIR": str(tmp_path / "smoke_artifacts"),
            "AGENTX_CHIPS": "32",
            "AGENTX_EVAL_JOB": "1",
            "AGENTX_EVAL_SMOKE": "1",
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc_smoke.returncode == 0
    assert (smoke_logs / "agentx_eval_client.args").exists()
    assert not (smoke_logs / "lm_eval.args").exists()


def test_k3_client_pod_runs_post_eval_when_replay_is_skipped(tmp_path, make_endpoint):
    """Skipping the AgentX replay still runs post-eval when AGENTX_SKIP_POST_EVAL!=1."""
    endpoint = make_endpoint(model_id="moonshotai/Kimi-K3")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _install_disagg_stubs(bin_dir, endpoint.port)

    ix_dir = tmp_path / "ix"
    ix_dir.mkdir()
    (ix_dir / "sweep_lib.sh").symlink_to(
        _REPO_ROOT / "scripts" / "vllm" / "benchmark" / "inferencex" / "sweep_lib.sh"
    )
    agentx_client = ix_dir / "agentx_client.sh"
    agentx_client.write_text("#!/bin/bash\nexit 99\n")
    agentx_client.chmod(agentx_client.stat().st_mode | stat.S_IXUSR)

    agentx_eval_client = ix_dir / "agentx_eval_client.sh"
    agentx_eval_client.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$@" >> "$STUB_LOG_DIR/agentx_eval_client.args"\n'
    )
    agentx_eval_client.chmod(agentx_eval_client.stat().st_mode | stat.S_IXUSR)

    fake_site = tmp_path / "site"
    (fake_site / "vllm").mkdir(parents=True)
    (fake_site / "vllm" / "__init__.py").write_text("")
    proxy_port = _free_port()

    proc = subprocess.run(
        ["bash", str(_K3_CLIENT_POD)],
        cwd=str(_REPO_ROOT),
        env={
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "PYTHONPATH": str(fake_site),
            "STUB_LOG_DIR": str(tmp_path),
            "WORKLOAD_NAME": "test-k3-post",
            "PREFILL_HOST": "127.0.0.1",
            "DECODE_HOST": "127.0.0.1",
            "PORT": str(endpoint.port),
            "PROXY_PORT": str(proxy_port),
            "REPO_DIR": str(_REPO_ROOT),
            "IX_DIR": str(ix_dir),
            "ARTIFACTS_DIR": str(tmp_path / "artifacts"),
            "AGENTX_CHIPS": "32",
            "AGENTX_SKIP_REPLAY": "1",
            "K3_EVAL_SUITES": "none",
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    post_args = (tmp_path / "agentx_eval_client.args").read_text().splitlines()
    assert "--suites" in post_args
    assert "kimi_tool_call_schema_full" in post_args


def test_qwen35_disagg_benchmark_resolves_relative_artifacts_dir(
    tmp_path, make_endpoint
):
    """Relative ARTIFACTS_DIR resolves against caller cwd before cd REPO_DIR."""
    endpoint = make_endpoint(model_id="Qwen/Qwen3.5-397B-A17B-FP8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _install_disagg_stubs(bin_dir, endpoint.port)

    fake_site = tmp_path / "site"
    (fake_site / "vllm").mkdir(parents=True)
    (fake_site / "vllm" / "__init__.py").write_text("")
    proxy_port = _free_port()
    caller_cwd = tmp_path / "caller_dir"
    caller_cwd.mkdir()

    proc = subprocess.run(
        ["bash", str(_DISAGG_BENCHMARK)],
        cwd=str(caller_cwd),
        env={
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "PYTHONPATH": str(fake_site),
            "STUB_LOG_DIR": str(tmp_path),
            "WORKLOAD_NAME": "test-qwen-rel",
            "PREFILL_HOST": "127.0.0.1",
            "DECODE_HOST": "127.0.0.1",
            "ENGINE_PORT": str(endpoint.port),
            "PROXY_PORT": str(proxy_port),
            "REPO_DIR": str(_REPO_ROOT),
            "ARTIFACTS_DIR": "rel_artifacts",
            "BENCHMARK_RESULTS_DIR": str(tmp_path / "bench_cells"),
            "DISAGG_PERF_SUITES": "bench_serve",
            "DISAGG_EVAL_SUITES": "mmlu_pro",
            "LM_EVAL_REQUEST_CACHE_ROOT": str(tmp_path / "cache"),
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    resolved = caller_cwd / "rel_artifacts"
    assert (resolved / "proxy.log").exists()
    assert (resolved / "qwen3.5-397b-fp8-1p1d" / "results_mmlu_pro.json").exists()
    assert not (_REPO_ROOT / "rel_artifacts").exists()


def test_common_sh_forwards_1p1d_env_vars():
    common_sh = _REPO_ROOT / ".buildkite" / "kubernetes" / "common.sh"
    expected = {
        "AGENTX_PROXY",
        "AGENTX_SKIP_REPLAY",
        "DISAGG_PROXY",
        "DISAGG_CONFIG",
        "DISAGG_BASELINE_CONFIG",
        "DISAGG_EVAL_SUITES",
        "DISAGG_PERF_SUITES",
        "K3_EVAL_SUITES",
        "K3_PERF_SUITES",
        "K3_BASELINE_CONFIG",
    }
    env = {"PATH": "/usr/bin:/bin", "WORKLOAD_IMAGE": "example/img:test"}
    for name in expected:
        env[name] = f"val_{name}"
    proc = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{common_sh}" && printf "%s\\n" "${{env_args[@]}}"',
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    args = proc.stdout.splitlines()
    forwarded = {args[i + 1] for i in range(len(args) - 1) if args[i] == "--env"}
    assert expected <= forwarded


def test_kimi_k3_agentx_config_perf_suites_accepted_by_run_eval_flow(
    tmp_path, make_endpoint
):
    """kimi-k3-tp8-dp4-ep-agentx.sh declares PERF_SUITES valid for run_eval_flow.sh."""
    endpoint = make_endpoint(model_id="moonshotai/Kimi-K3")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _install_disagg_stubs(bin_dir, endpoint.port)

    fake_site = tmp_path / "site"
    (fake_site / "vllm").mkdir(parents=True)
    (fake_site / "vllm" / "__init__.py").write_text("")
    results_dir = tmp_path / "flow_results"

    # Omit --perf-suites so run_eval_flow.sh reads PERF_SUITES from the config.
    proc = subprocess.run(
        [
            "bash",
            str(_REPO_ROOT / "scripts" / "vllm" / "benchmarking" / "run_eval_flow.sh"),
            "--config",
            "kimi-k3-tp8-dp4-ep-agentx",
            "--base-url",
            endpoint.url,
            "--eval-suites",
            "none",
            "--results-dir",
            str(results_dir),
            "--skip-db-upload",
        ],
        cwd=str(_REPO_ROOT),
        env={
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "PYTHONPATH": str(fake_site),
            "STUB_LOG_DIR": str(tmp_path),
            "K3_1P1D_ROLE": "decode",
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert not (tmp_path / "vllm.args").exists()


def test_qwen35_1p1d_config_does_not_set_tpu_kv_transfer_namespace():
    cfg_path = (
        _REPO_ROOT
        / "scripts"
        / "vllm"
        / "benchmarking"
        / "configs"
        / "qwen3.5-397b-fp8-1p1d.sh"
    )
    assert "TPU_KV_TRANSFER_NAMESPACE" not in cfg_path.read_text()
    cmd = (
        f'source "{cfg_path}" >/dev/null '
        '&& printf "%s" "${TPU_KV_TRANSFER_NAMESPACE:-}"'
    )
    proc = subprocess.run(
        ["bash", "-c", cmd],
        cwd=str(_REPO_ROOT),
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout == ""


@pytest.mark.parametrize("eval_suites", ["all", "mmlu_pro, gsm8k"])
def test_k3_client_pod_dedupes_gsm8k_for_all_and_whitespace_eval_suites(
    tmp_path, make_endpoint, eval_suites
):
    endpoint = make_endpoint(model_id="moonshotai/Kimi-K3")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _install_disagg_stubs(bin_dir, endpoint.port)

    ix_dir = tmp_path / "ix"
    ix_dir.mkdir()
    (ix_dir / "sweep_lib.sh").symlink_to(
        _REPO_ROOT / "scripts" / "vllm" / "benchmark" / "inferencex" / "sweep_lib.sh"
    )
    agentx_client = ix_dir / "agentx_client.sh"
    agentx_client.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$@" >> "$STUB_LOG_DIR/agentx_client.args"\n'
    )
    agentx_client.chmod(agentx_client.stat().st_mode | stat.S_IXUSR)

    fake_site = tmp_path / "site"
    (fake_site / "vllm").mkdir(parents=True)
    (fake_site / "vllm" / "__init__.py").write_text("")
    proxy_port = _free_port()

    proc = subprocess.run(
        ["bash", str(_K3_CLIENT_POD)],
        cwd=str(_REPO_ROOT),
        env={
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "PYTHONPATH": str(fake_site),
            "STUB_LOG_DIR": str(tmp_path),
            "WORKLOAD_NAME": "test-k3-dedupe",
            "PREFILL_HOST": "127.0.0.1",
            "DECODE_HOST": "127.0.0.1",
            "PORT": str(endpoint.port),
            "PROXY_PORT": str(proxy_port),
            "REPO_DIR": str(_REPO_ROOT),
            "IX_DIR": str(ix_dir),
            "ARTIFACTS_DIR": str(tmp_path / "artifacts"),
            "AGENTX_CHIPS": "32",
            "AGENTX_SKIP_POST_EVAL": "1",
            "K3_EVAL_SUITES": eval_suites,
            "LM_EVAL_REQUEST_CACHE_ROOT": str(tmp_path / "cache"),
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    client_args = (tmp_path / "agentx_client.args").read_text().splitlines()
    idx = client_args.index("--gsm8k-limit")
    assert client_args[idx + 1] == "0"


def test_k3_client_pod_rejects_invalid_k3_perf_suites(tmp_path):
    proc = subprocess.run(
        ["bash", str(_K3_CLIENT_POD)],
        cwd=str(_REPO_ROOT),
        env={
            "PATH": "/usr/bin:/bin",
            "WORKLOAD_NAME": "test-k3-bad-perf",
            "REPO_DIR": str(_REPO_ROOT),
            "ARTIFACTS_DIR": str(tmp_path / "artifacts"),
            "AGENTX_CHIPS": "32",
            "K3_PERF_SUITES": "agentx",
        },
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 2
    assert "unknown K3_PERF_SUITES item 'agentx'" in proc.stdout


def test_disagg_benchmark_uses_config_eval_suites_when_env_unset(
    tmp_path, make_endpoint
):
    """When DISAGG_EVAL_SUITES is unset, disagg_benchmark.sh omits --eval-suites."""
    endpoint = make_endpoint(model_id="Qwen/Qwen3.5-397B-A17B-FP8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _install_disagg_stubs(bin_dir, endpoint.port)

    # Custom config declaring only mmlu_pro proves disagg_benchmark.sh does not
    # hardcode qwen3.5's suite list when DISAGG_EVAL_SUITES is unset.
    cfg_dir = tmp_path / "repo" / "scripts" / "vllm" / "benchmarking" / "configs"
    cfg_dir.mkdir(parents=True)
    run_flow = (
        tmp_path / "repo" / "scripts" / "vllm" / "benchmarking" / "run_eval_flow.sh"
    )
    run_flow.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$@" > "$STUB_LOG_DIR/run_eval_flow.args"\n'
    )
    run_flow.chmod(run_flow.stat().st_mode | stat.S_IXUSR)

    proxy_port = _free_port()
    proc = subprocess.run(
        ["bash", str(_DISAGG_BENCHMARK)],
        cwd=str(tmp_path),
        env={
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "STUB_LOG_DIR": str(tmp_path),
            "WORKLOAD_NAME": "test-qwen-cfg",
            "PREFILL_HOST": "127.0.0.1",
            "DECODE_HOST": "127.0.0.1",
            "ENGINE_PORT": str(endpoint.port),
            "PROXY_PORT": str(proxy_port),
            "REPO_DIR": str(tmp_path / "repo"),
            "ARTIFACTS_DIR": str(tmp_path / "artifacts"),
            "BENCHMARK_RESULTS_DIR": str(tmp_path / "bench_cells"),
            "DISAGG_CONFIG": "custom-1p1d",
            "DISAGG_PERF_SUITES": "bench_serve",
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    flow_args = (tmp_path / "run_eval_flow.args").read_text().splitlines()
    assert "--eval-suites" not in flow_args
    assert "--config" in flow_args
    assert "custom-1p1d" in flow_args
