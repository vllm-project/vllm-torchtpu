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
"""Fails a TPU pod fast when its node cannot bring up the TPU runtime.

run.sh runs this in the pod before the step's command, on TPU shapes only.

A pod scheduled onto a broken TPU node (a dead TPU device plugin, so
/var/run/tpu-plugin/vbar.sock refuses connections) does not fail on its own:
every process that touches the chip waits out the PJRT init deadline and then
fails, so a pytest shard pays that wait once per test. One such shard spent
four and a half hours failing all 618 of its tests a minute at a time before it
was cancelled by hand.

The probe builds a PJRT TPU client in a child process, never in this one, and
waits for that child to exit before returning. libtpu is process-exclusive and
has no release API: a client built in a process that outlives the probe would
hold the chips and lock out the step's own pytest and engine processes.
"""

import argparse
import contextlib
import importlib.util
import os
import signal
import socket
import stat
import subprocess
import sys
import time
import urllib.request

# The socket the GKE TPU device plugin serves and libtpu dials on tpu7x. Absent
# on shapes whose runtime does not use it, which is not by itself a failure.
VBAR_SOCKET = "/var/run/tpu-plugin/vbar.sock"

# A healthy chip comes up in well under a minute; JAX starts warning at 60 s.
# Generous, because a false failure here fails a step on a healthy node, while
# waiting longer on a broken one costs minutes against hours.
DEFAULT_TIMEOUT_S = 300

# What the probe prints in front of its device summary, so the summary can be
# told apart from whatever JAX and libtpu log around it.
_DEVICES_MARKER = "TPU_PREFLIGHT_DEVICES "

# Built in the child. `jax.devices("tpu")` raises rather than falling back to
# CPU, so a printed device list means the TPU backend really came up.
DEFAULT_PROBE = f"""
import jax
devices = jax.devices("tpu")
print("{_DEVICES_MARKER}" f"{{len(devices)}} x {{devices[0].device_kind}}", flush=True)
"""

# Where an operator would look for the node. The pod name is always there; the
# others only when the pod spec or the launcher passes them in.
_IDENTITY_ENV = (
    "HOSTNAME",
    "NODE_NAME",
    "K8S_NODE_NAME",
    "KUBERNETES_NODE_NAME",
    "BUILDKITE_K8S_NODE",
    "BUILDKITE_AGENT_NAME",
    "BUILDKITE_BUILD_URL",
    "BUILDKITE_JOB_ID",
)

_METADATA_URL = "http://metadata.google.internal/computeMetadata/v1/instance/"

_OUTPUT_TAIL_LINES = 40


def _skip_reason() -> str | None:
    """Why the probe should not run in this pod, or None if it should."""
    platforms = os.environ.get("JAX_PLATFORMS", "")
    if platforms and "tpu" not in platforms.split(","):
        return f"JAX_PLATFORMS={platforms} asks for no TPU"
    if importlib.util.find_spec("jax") is None:
        return "jax is not installed in this image"
    return None


def run_probe(probe: str, timeout_s: float) -> tuple[bool, str, str]:
    """Runs `probe` in a child Python and waits for it to exit.

    Returns:
        (ok, reason, output): whether the child exited 0 within the deadline,
        a one-line verdict, and the child's combined stdout and stderr.
    """
    start = time.monotonic()
    # Own session, so a timeout kills anything the child started as well and
    # nothing is left holding the chips.
    proc = subprocess.Popen(
        [sys.executable, "-c", probe],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        output, _ = proc.communicate()
        return False, f"PJRT TPU init did not finish within {timeout_s:g}s", output
    elapsed = time.monotonic() - start
    if proc.returncode != 0:
        return (
            False,
            f"PJRT TPU init failed (exit {proc.returncode}) after {elapsed:.0f}s",
            output,
        )
    return True, f"TPU runtime came up in {elapsed:.0f}s", output


def _vbar_socket_state(path: str = VBAR_SOCKET) -> str:
    """Describes the TPU device plugin socket without touching libtpu."""
    try:
        mode = os.stat(path).st_mode
    except FileNotFoundError:
        return "absent"
    except OSError as e:
        return f"cannot stat: {e}"
    if not stat.S_ISSOCK(mode):
        return "present but not a socket"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(3)
        try:
            sock.connect(path)
        except OSError as e:
            return f"connect failed: {e}"
    return "accepting connections"


def _metadata(key: str) -> str:
    """Best-effort lookup of the VM behind this pod; empty when unavailable."""
    request = urllib.request.Request(
        _METADATA_URL + key, headers={"Metadata-Flavor": "Google"}
    )
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            return response.read().decode().strip()
    except Exception:
        return ""


def _diagnostics() -> list[str]:
    lines = [
        f"{name}={os.environ[name]}" for name in _IDENTITY_ENV if os.environ.get(name)
    ]
    for key in ("hostname", "zone"):
        value = _metadata(key)
        if value:
            lines.append(f"metadata instance/{key}={value}")
    lines.append(f"{VBAR_SOCKET}: {_vbar_socket_state()}")
    for path in ("/dev/vfio", "/dev/accel0"):
        lines.append(f"{path}: {'present' if os.path.exists(path) else 'absent'}")
    return lines


def _report_failure(reason: str, output: str) -> None:
    tail = output.rstrip().splitlines()[-_OUTPUT_TAIL_LINES:]
    rule = "=" * 78
    # `^^^ +++` expands the enclosing Buildkite log group, so the verdict is
    # what the log opens on rather than something to dig for.
    print("^^^ +++", flush=True)
    print(
        "\n".join(
            [
                rule,
                "TPU PREFLIGHT FAILED: this pod's node cannot initialize its TPU.",
                f"Reason: {reason}",
                "The step's command was NOT run: every TPU test would wait out the",
                "same init deadline and fail one at a time. This is a node problem,",
                "not a code problem - retry the job so it lands on another node,",
                "and report the node below to the fleet owners.",
                "",
                *_diagnostics(),
                "",
                f"Last {len(tail)} lines of the probe's output:",
                *(f"  {line}" for line in tail),
                rule,
            ]
        ),
        file=sys.stderr,
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help="seconds to wait for the PJRT TPU client (default: %(default)s)",
    )
    # A test seam: replaces the JAX probe with any Python program.
    parser.add_argument("--probe", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    probe = args.probe
    if probe is None:
        skip = _skip_reason()
        if skip:
            print(f"TPU preflight skipped: {skip}.", flush=True)
            return 0
        probe = DEFAULT_PROBE

    print(f"TPU preflight: probing PJRT (timeout {args.timeout:g}s)...", flush=True)
    ok, reason, output = run_probe(probe, args.timeout)
    if not ok:
        _report_failure(reason, output)
        return 1
    summary = [
        line.removeprefix(_DEVICES_MARKER)
        for line in output.splitlines()
        if line.startswith(_DEVICES_MARKER)
    ]
    print(f"TPU preflight passed: {reason}. {' '.join(summary)}".rstrip(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
