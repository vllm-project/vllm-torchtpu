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
"""The programs a run starts, as stubs: each records its call and fakes its work.

Run as `python stubs.py <program> <args...>` from a shell shim per program
name (runtime_helpers.stub_env writes them). Every call is recorded as one
JSON file under $STUB_RECORDS: the program, argv, env, cwd and pid.

`vllm serve` is a small OpenAI-compatible HTTP server: /health,
/v1/models, /v1/completions, /v1/chat/completions and /metrics, plus
/stub/die, which ends it as a crash would. It forks one child that stands in
for an engine core. Knobs, all env:

    STUB_SERVE_EXIT=N     exit N at once, before serving
    STUB_READY_AFTER=S    /health answers 503 for S seconds
    STUB_NEVER_READY=1    /health never answers 200
    STUB_DIE_AFTER=S      crash S seconds after start
    STUB_PULL=1           fill $STUB_PULLED_DIR as runai pulls a model

`vllm bench serve` and InferenceX's benchmark_serving (`python3 -m
infx.bench_serving.benchmark_serving`) ask the server for /v1/models and
write a result file with the fields the gate and the uploaders read: every
prompt completed when the server answered, every prompt failed when it did
not, as the real client does. Knobs:

    STUB_BENCH_FAIL=NAME     exit 1 without a result when writing file NAME
    STUB_BENCH_KILL=NAME     crash the server after writing file NAME
    STUB_BENCH_FAILED=N      report N failed requests
    STUB_BENCH_METRICS=JSON  metric values to report

`python3` runs the real interpreter for anything but `-m infx...`.
"""

import http.client
import http.server
import json
import os
import pathlib
import signal
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

RECORDS = os.environ.get("STUB_RECORDS")


def record(program: str, argv: list[str], **extra: Any) -> None:
    if not RECORDS:
        return
    data = {
        "program": program,
        "argv": argv,
        "env": dict(os.environ),
        "cwd": os.getcwd(),
        "pid": os.getpid(),
        **extra,
    }
    path = pathlib.Path(RECORDS) / f"{time.time_ns()}-{os.getpid()}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.rename(path)


def flag(args: list[str], name: str, default: str | None = None) -> str | None:
    """The value of --name, as `--name v` or `--name=v`; the last one wins."""
    value = default
    for i, arg in enumerate(args):
        if arg == name and i + 1 < len(args):
            value = args[i + 1]
        elif arg.startswith(name + "="):
            value = arg.split("=", 1)[1]
    return value


def number(name: str, default: float = 0.0) -> float:
    return float(os.environ.get(name, default))


# vllm serve -----------------------------------------------------------------


def serve(args: list[str]) -> int:
    if "STUB_SERVE_EXIT" in os.environ:
        print("ERROR stub: engine failed to start", flush=True)
        return int(os.environ["STUB_SERVE_EXIT"])
    port = int(flag(args, "--port", "8000") or 8000)
    served = flag(args, "--served-model-name") or args[0]
    max_len = int(flag(args, "--max-model-len", "4096") or 4096)
    started = time.monotonic()
    ready_after = number("STUB_READY_AFTER")
    never_ready = os.environ.get("STUB_NEVER_READY") == "1"
    engine = os.fork()
    if engine == 0:
        # The engine core: it lives as long as the group does.
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        while True:
            time.sleep(60)
    record("vllm serve", args, engine_pid=engine)
    if os.environ.get("STUB_PULL") == "1":
        pulled = pathlib.Path(os.environ["STUB_PULLED_DIR"])
        pulled.mkdir(parents=True, exist_ok=True)
        (pulled / "config.json").write_text("{}")

    class Handler(http.server.BaseHTTPRequestHandler):
        def reply(self, code: int, body: Any, kind: str = "application/json") -> None:
            data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/health":
                ready = not never_ready and time.monotonic() - started >= ready_after
                self.reply(200 if ready else 503, {})
            elif self.path == "/v1/models":
                card = {"id": served, "object": "model", "max_model_len": max_len}
                self.reply(200, {"object": "list", "data": [card]})
            elif self.path == "/metrics":
                self.reply(200, "vllm:num_requests_running 0\n", "text/plain")
            elif self.path == "/stub/die":
                self.reply(200, {})
                print("ERROR stub: engine core died", flush=True)
                os.kill(engine, signal.SIGKILL)
                os._exit(1)
            else:
                self.reply(404, {"error": self.path})

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(length)
            if self.path == "/v1/completions":
                self.reply(200, {"choices": [{"text": " 5", "finish_reason": "stop"}]})
            elif self.path == "/v1/chat/completions":
                message = {"role": "assistant", "content": "5"}
                self.reply(200, {"choices": [{"message": message}]})
            else:
                self.reply(404, {"error": self.path})

        def log_message(self, fmt: str, *values: Any) -> None:
            print(f"INFO stub: {fmt % values}", flush=True)

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"INFO stub: serving {served} on port {port}", flush=True)
    die_after = number("STUB_DIE_AFTER")
    if die_after:

        def crash() -> None:
            time.sleep(die_after)
            print("ERROR stub: engine core died", flush=True)
            os.kill(engine, signal.SIGKILL)
            os._exit(1)

        threading.Thread(target=crash, daemon=True).start()

    def stop(signum: int, frame: Any) -> None:
        print("INFO stub: shutting down", flush=True)
        os.kill(engine, signal.SIGTERM)
        os._exit(0)

    signal.signal(signal.SIGTERM, stop)
    httpd.serve_forever()
    return 0


# Perf clients -----------------------------------------------------------------

METRICS = {
    "median_ttft_ms": 100.0,
    "median_tpot_ms": 10.0,
    "total_token_throughput": 1000.0,
    "output_throughput": 500.0,
}


def _get(host: str, port: int, path: str) -> int | None:
    conn = http.client.HTTPConnection(host, port, timeout=5)
    try:
        conn.request("GET", path)
        response = conn.getresponse()
        response.read()
        return response.status
    except OSError:
        return None
    finally:
        conn.close()


def _endpoint(args: list[str]) -> tuple[str, int]:
    base = flag(args, "--base-url")
    if base is not None:
        host, _, port = base.removeprefix("http://").rstrip("/").partition(":")
        return host, int(port or 80)
    return flag(args, "--host", "127.0.0.1") or "", int(
        flag(args, "--port", "8000") or 0
    )


def bench(program: str, args: list[str]) -> int:
    record(program, args)
    name = flag(args, "--result-filename") or "result.json"
    directory = flag(args, "--result-dir")
    path = pathlib.Path(directory, name) if directory else pathlib.Path(name)
    if os.environ.get("STUB_BENCH_FAIL") == path.name:
        print(f"stub bench: failed before writing {path.name}", flush=True)
        return 1
    host, port = _endpoint(args)
    prompts = int(flag(args, "--num-prompts", "10") or 10)
    reachable = _get(host, port, "/v1/models") == 200
    for percent in (0, 50, 100):
        sys.stdout.write(f"{percent:3d}%|{'#' * (percent // 10)}|\r")
    sys.stdout.write("\n")
    failed = int(os.environ.get("STUB_BENCH_FAILED", 0)) if reachable else prompts
    result = {
        "date": time.strftime("%Y%m%d-%H%M%S"),
        "backend": flag(args, "--backend", "vllm"),
        "model_id": flag(args, "--model"),
        "tokenizer_id": flag(args, "--tokenizer") or flag(args, "--model"),
        "num_prompts": prompts,
        "completed": prompts - failed,
        "failed": failed,
        "request_rate": "inf",
        "max_concurrency": int(flag(args, "--max-concurrency", "1") or 1),
        **METRICS,
        **json.loads(os.environ.get("STUB_BENCH_METRICS", "{}")),
    }
    label = flag(args, "--label")
    if label is not None:
        result["label"] = label
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result) + "\n")
    print(f"stub bench: wrote {path.name}", flush=True)
    if os.environ.get("STUB_BENCH_KILL") == path.name:
        _get(host, port, "/stub/die")
    return 0


def vllm(args: list[str]) -> int:
    if args[:1] == ["serve"]:
        return serve(args[1:])
    if args[:2] == ["bench", "serve"]:
        return bench("vllm bench serve", args[2:])
    record("vllm", args)
    print(f"stub vllm: unknown command {args}", file=sys.stderr)
    return 2


INFX: dict[str, Callable[[list[str]], int]] = {
    "infx.bench_serving.benchmark_serving": lambda args: bench(
        "infx benchmark_serving", args
    ),
}


def python3(args: list[str]) -> int:
    if len(args) >= 2 and args[0] == "-m" and args[1] in INFX:
        return INFX[args[1]](args[2:])
    os.execv(sys.executable, [sys.executable, *args])
    return 1


PROGRAMS: dict[str, Callable[[list[str]], int]] = {"vllm": vllm, "python3": python3}


def main() -> int:
    program, args = sys.argv[1], sys.argv[2:]
    return PROGRAMS[program](args)


if __name__ == "__main__":
    sys.exit(main())
