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
"""The programs today's scripts call, as stubs: each records its call and fakes it.

Run as `python -I -S stubs.py <program> <args...>` from a small shell shim per
program name. A record is one JSON file under $GOLDEN_STATE/records holding
the argv, the full environment and the host ($GOLDEN_HOST) of the call.

Servers fake readiness with marker files: `vllm serve` and the toy proxy
write $GOLDEN_STATE/http/<host>:<port>, `ray start --head` writes
$GOLDEN_STATE/tcp/<host>:<port>, and curl and TCP probes answer from them.
$GOLDEN_STATE/hosts.json maps the pods' DNS names and IPs to hosts.
"""

import json
import os
import pathlib
import re
import subprocess
import sys
import time
import urllib.parse

STATE = pathlib.Path(os.environ["GOLDEN_STATE"])
HOST = os.environ.get("GOLDEN_HOST", "agent")
FAKES = pathlib.Path(__file__).resolve().parent / "fakes"
LOOPBACK = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}
# What a pulled object-storage model directory holds; the bench client
# tokenizes from it when config.json is there.
PULLED_MODEL_DIR = STATE / "pulled-model"
IMAGE = "registry.example/vllm-torchtpu:golden"
# How long a command under `timeout` may run here, whatever it asked for.
TIMEOUT_CAP_S = 30


def record(tool: str, argv: list[str]) -> None:
    stamp = time.time_ns()
    data = {
        "tool": tool,
        "argv": argv,
        "env": dict(os.environ),
        "cwd": os.getcwd(),
        "host": HOST,
        "time": stamp,
    }
    path = STATE / "records" / f"{stamp}-{os.getpid()}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.rename(path)


def flag(args: list[str], name: str) -> str | None:
    """The value of --name, as `--name v` or `--name=v`; the last one wins."""
    value = None
    for i, arg in enumerate(args):
        if arg == name and i + 1 < len(args):
            value = args[i + 1]
        elif arg.startswith(name + "="):
            value = arg.split("=", 1)[1]
    return value


def mark(kind: str, port: str) -> None:
    path = STATE / kind / f"{HOST}:{port}"
    path.parent.mkdir(exist_ok=True)
    path.touch()


def host_of(name: str) -> str | None:
    if name in LOOPBACK:
        return HOST
    table = json.loads((STATE / "hosts.json").read_text())
    return table["names"].get(name) or table["ips"].get(name)


def up(kind: str, name: str, port: str) -> bool:
    host = host_of(name)
    return host is not None and (STATE / kind / f"{host}:{port}").exists()


def serve_forever() -> int:
    while True:
        time.sleep(3600)


def vllm(args: list[str]) -> int:
    record("vllm", args)
    if args[:1] == ["serve"]:
        # A headless host serves no API, so it marks no port.
        if "--headless" not in args:
            mark("http", flag(args, "--port") or "8000")
        return serve_forever()
    if args[:2] == ["bench", "serve"] and (
        "--save-result" in args or "--append-result" in args
    ):
        name = flag(args, "--result-filename") or flag(args, "--result-file")
        if name:
            path = pathlib.Path(flag(args, "--result-dir") or ".") / name
            path.parent.mkdir(parents=True, exist_ok=True)
            prompts = int(flag(args, "--num-prompts") or 0)
            path.write_text(json.dumps({"completed": prompts, "failed": 0}))
    return 0


def curl(args: list[str]) -> int:
    url = next((a for a in args if a.startswith(("http://", "https://"))), "")
    parsed = urllib.parse.urlsplit(url)
    write_out = flag(args, "-w") or flag(args, "--write-out")
    if not up("http", parsed.hostname or "", str(parsed.port or 80)):
        if write_out:
            sys.stdout.write("000")
        return 7
    body = ""
    if parsed.path.startswith("/metrics"):
        body = 'vllm:cache_config_info{block_size="1296",mamba_block_size="1296"} 1\n'
    elif parsed.path.startswith("/v1/models"):
        body = json.dumps({"data": [{"id": "model", "max_model_len": 0}]}) + "\n"
    if flag(args, "-o") is None:
        sys.stdout.write(body)
    if write_out:
        sys.stdout.write("200")
    return 0


def getent(args: list[str]) -> int:
    table = json.loads((STATE / "hosts.json").read_text())
    name = args[-1] if args else ""
    host = table["names"].get(name)
    if args[:1] != ["ahostsv4"] or host is None:
        return 2
    ip = table["host_ips"][host]
    print(f"{ip}       STREAM {name}\n{ip}       DGRAM\n{ip}       RAW")
    return 0


def _seconds(text: str) -> float:
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    if text and text[-1] in units:
        return float(text[:-1]) * units[text[-1]]
    return float(text)


_TCP_PROBE = re.compile(r"exec 3<>/dev/tcp/([^/]+)/(\d+)")


def timeout(args: list[str]) -> int:
    i = 0
    while i < len(args) and args[i].startswith("-"):
        i += 2 if args[i] in ("-s", "-k", "--signal", "--kill-after") else 1
    if i >= len(args) - 1:
        return 125
    limit, command = _seconds(args[i]), args[i + 1 :]
    probe = _TCP_PROBE.fullmatch(command[-1]) if command[:2] == ["bash", "-c"] else None
    if probe is not None:
        return 0 if up("tcp", probe.group(1), probe.group(2)) else 1
    try:
        return subprocess.run(command, timeout=min(limit, TIMEOUT_CAP_S)).returncode
    except subprocess.TimeoutExpired:
        return 124


def ray(args: list[str]) -> int:
    record("ray", args)
    if args[:1] == ["start"] and "--head" in args:
        mark("tcp", flag(args, "--port") or "6379")
    return 0


def _proxy(args: list[str]) -> int:
    mark("http", flag(args, "--port") or "8000")
    return serve_forever()


# Scripts the python3 stub records and fakes instead of running.
FAKED_SCRIPTS = {
    "toy_proxy_server.py": _proxy,
    "check_completions.py": lambda args: 0,
    "warm_remote_code.py": lambda args: 0,
    "check_regression.py": lambda args: 0,
    "upload_results.py": lambda args: 0,
    "parse_gke_results.py": lambda args: 0,
}


def python3(args: list[str]) -> int:
    i = 0
    while i < len(args) and args[i] in ("-u", "-B", "-E", "-s", "-S", "-I"):
        i += 1
    rest = args[i:]
    if rest[:1] == ["-c"] and "connect_ex" in (rest[1:2] or [""])[0]:
        return 1  # a port probe: nothing listens on this host
    if rest and not rest[0].startswith("-"):
        fake = FAKED_SCRIPTS.get(pathlib.Path(rest[0]).name)
        if fake is not None:
            record("python3", args)
            return fake(rest[1:])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(FAKES), env.get("PYTHONPATH")) if part
    )
    # The fakes live in the checkout; leave no bytecode beside them.
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    os.execve(sys.executable, [sys.executable, *args], env)
    return 1


def buildkite_agent(args: list[str]) -> int:
    if args[:2] == ["meta-data", "get"]:
        print(IMAGE)
        return 0
    record("buildkite-agent", args)
    return 0


def stdbuf(args: list[str]) -> int:
    i = 0
    while i < len(args) and args[i].startswith("-"):
        i += 2 if args[i] in ("-i", "-o", "-e") else 1
    os.execvp(args[i], args[i:])
    return 1


def recorded(name: str) -> object:
    def run(args: list[str]) -> int:
        record(name, args)
        return 0

    return run


def script(args: list[str]) -> int:
    """A script replaced by a recorder in the scratch copy of the repository."""
    path, rest = pathlib.Path(args[0]).resolve(), args[1:]
    root = pathlib.Path(os.environ["GOLDEN_ROOT"]).resolve()
    record("script", [str(path.relative_to(root)), *rest])
    return 0


PROGRAMS = {
    "vllm": vllm,
    "curl": curl,
    "getent": getent,
    "timeout": timeout,
    "ray": ray,
    "python3": python3,
    "buildkite-agent": buildkite_agent,
    "stdbuf": stdbuf,
    "lm_eval": recorded("lm_eval"),
    "pip": recorded("pip"),
    "gsutil": recorded("gsutil"),
    "launch": recorded("launch"),
    "script": script,
}


if __name__ == "__main__":
    sys.exit(PROGRAMS[sys.argv[1]](sys.argv[2:]))
