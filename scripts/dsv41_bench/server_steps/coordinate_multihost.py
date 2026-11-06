# SPDX-License-Identifier: Apache-2.0
"""Coordinate the two pods of the DP16 decode benchmark over one HTTP port.

Each pod of ``.buildkite/kubernetes/manifests/workloads/dsv41-dp16-decode.yaml``
wraps its own ``run_dsv41_benchmark.sh decode_dp16`` in this script, which
turns that benchmark's handshake files into requests to a small HTTP service
on the head (index 0). It replaces the SSH coordinator the VM lane ran on the
head host.

- Launch: neither host starts serving until both have written
  ``ready-to-launch`` (the head after preparing inputs, the worker after
  fetching the model metadata); then both get ``launch-permitted``. Input
  preparation would otherwise eat the MP rendezvous timeout.
- Shutdown: after the head's ``benchmark/formal-complete`` the worker sends
  SIGTERM to its ``benchmark/launcher.pid`` (run.py's orderly stop, which the
  audit requires), waits for its benchmark to exit and posts its results. Only
  then does the head get ``benchmark/shutdown-permitted``: stopping the DP
  master first would look like a peer crash to the worker.
- Collection: the worker's ``worker/`` tree arrives as a tar and is unpacked
  next to the head's results, where the head runs ``audit_multihost.py``.

Fail-closed. The head fails the run if the worker exits early, reports a
failure, goes silent, or a second worker appears after launch (a recreated pod
cannot rejoin a live collective world). The worker stops its benchmark if the
head aborts, stops answering, or is replaced by a new head. The head's exit
status is the step's verdict, so the worker always exits 0: a failing worker
pod would fail the Job and kill the head before it publishes the check and
uploads the logs. A worker that could not deliver its results leaves the head
to fail the run for want of them.

    coordinate_multihost.py head --results DIR [--port N] -- <benchmark command>
    coordinate_multihost.py worker --results DIR --head ADDR [--port N] -- <command>
"""

from __future__ import annotations

import argparse
import io
import json
import os
import signal
import subprocess
import sys
import tarfile
import threading
import time
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Outside the ports run.py binds: 8000 (API), 19000-19709 (bootstrap, DP RPC
# and master) and the kernel's ephemeral range vLLM draws its ZMQ ports from.
DEFAULT_PORT = 18000


def log(message: str) -> None:
    print(f"[dp16-coordinator] {message}", file=sys.stderr, flush=True)


def terminate(process: subprocess.Popen, grace: float = 60) -> None:
    """Stop a benchmark and everything it started in its session."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def wait_for(predicate: Callable[[], bool], timeout: float, poll: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll)
    return predicate()


class HeadState:
    """What the head knows about the worker, shared with the HTTP handlers."""

    def __init__(self, results: Path):
        self.results = results
        self.lock = threading.Lock()
        # Lets the worker tell a recreated head from the one it launched with.
        self.run_id = uuid.uuid4().hex
        self.worker_id: str | None = None
        self.worker_ready = False
        self.launch = False
        self.stop = False
        self.abort = False
        self.error: str | None = None
        self.worker_rc: int | None = None
        # Monotonic time of the registered worker's last poll.
        self.worker_seen: float | None = None

    def fail(self, message: str) -> None:
        with self.lock:
            if not self.abort:
                self.abort = True
                self.error = message

    def check_worker(self, worker: str) -> None:
        """Record a request from ``worker``; a stranger after launch fails."""
        with self.lock:
            if self.worker_id != worker and not self.launch:
                # A recreated worker pod before launch: it has to prepare and
                # register again before the head may launch.
                self.worker_id = worker
                self.worker_ready = False
            stranger = worker != self.worker_id
            if not stranger:
                self.worker_seen = time.monotonic()
        if stranger:
            self.fail(f"a second worker ({worker}) appeared after launch")

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "run": self.run_id,
                "launch": self.launch,
                "stop": self.stop,
                "abort": self.abort,
            }


def make_handler(state: HeadState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A002
            pass

        def reply(self, body: dict, code: int = 200) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def worker(self) -> str | None:
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            return query.get("worker", [None])[0]

        def do_GET(self):
            worker = self.worker()
            if urllib.parse.urlsplit(self.path).path != "/state" or not worker:
                self.reply({"error": "not found"}, 404)
                return
            state.check_worker(worker)
            self.reply(state.snapshot())

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            path = urllib.parse.urlsplit(self.path).path
            worker = self.worker()
            if not worker or path not in ("/ready", "/result"):
                self.reply({"error": "not found"}, 404)
                return
            state.check_worker(worker)
            if path == "/ready":
                with state.lock:
                    state.worker_ready = state.worker_id == worker
                self.reply(state.snapshot())
                return
            rc = int(self.headers.get("X-Exit-Status", "1"))
            try:
                with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as tar:
                    if any(
                        name != "worker" and not name.startswith("worker/")
                        for name in tar.getnames()
                    ):
                        raise ValueError("worker archive must contain only worker/")
                    tar.extractall(state.results, filter="data")
            except (tarfile.TarError, ValueError, OSError) as error:
                log(f"head: cannot unpack worker results: {error}")
                rc = rc or 1
            with state.lock:
                state.worker_rc = rc
            self.reply({"received": True})

    return Handler


def supervise_head(args, state: HeadState, bench: subprocess.Popen) -> int:
    results = state.results
    benchmark = results / "benchmark"
    while bench.poll() is None:
        with state.lock:
            aborted, worker_rc = state.abort, state.worker_rc
            ready, launch, seen = state.worker_ready, state.launch, state.worker_seen
        if aborted:
            break
        if worker_rc is not None:
            state.fail(f"worker exited ({worker_rc}) before the head completed")
            break
        # A worker that was polling and went silent was killed or lost its
        # node; one that never called yet is still pulling or preparing, which
        # the benchmark's own launch deadline bounds.
        if seen is not None and time.monotonic() - seen >= args.worker_lost_after:
            state.fail(f"worker silent for {args.worker_lost_after:.0f} seconds")
            break
        if not launch and ready and (results / "ready-to-launch").exists():
            # Both hosts are prepared: the worker registers only after its own
            # ready-to-launch, and this is the head's.
            (results / "launch-permitted").touch()
            with state.lock:
                state.launch = True
            log("head: both hosts prepared; launch permitted")
        if (benchmark / "formal-complete").exists():
            break
        time.sleep(args.poll)

    if not state.abort:
        if not (benchmark / "formal-complete").exists():
            state.fail(f"head benchmark exited ({bench.poll()}) before completing")
        else:
            log("head: formal round complete; stopping the worker")
            with state.lock:
                state.stop = True
            if not wait_for(
                lambda: state.worker_rc is not None, args.worker_stop_timeout, args.poll
            ):
                state.fail(
                    f"worker did not stop within {args.worker_stop_timeout:.0f} seconds"
                )
            elif state.worker_rc != 0:
                state.fail(f"worker exited {state.worker_rc} on shutdown")
            else:
                (benchmark / "shutdown-permitted").touch()
                log("head: worker stopped; shutdown permitted")

    if state.abort:
        log(f"head: aborting: {state.error}")
        terminate(bench)
    rc = bench.wait()
    status = 1 if state.abort else rc
    # The endpoint stays up meanwhile, so a worker still running sees the
    # abort, stops and sends its logs.
    if not wait_for(
        lambda: state.worker_rc is not None, args.collect_timeout, args.poll
    ):
        log("head: worker results never arrived")
        status = status or 1
    if not (results / "worker").is_dir():
        log("head: missing worker/ results")
        status = status or 1
    if status == 0:
        status = subprocess.run(
            [sys.executable, str(args.audit), str(results)]
        ).returncode
    log(f"head: exiting {status}")
    return status


def run_head(args) -> int:
    results = Path(args.results)
    results.mkdir(parents=True, exist_ok=True)
    state = HeadState(results)
    server = ThreadingHTTPServer((args.bind, args.port), make_handler(state))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log(f"head: listening on {args.bind}:{args.port}")
    bench = subprocess.Popen(args.command, start_new_session=True)
    try:
        return supervise_head(args, state, bench)
    finally:
        terminate(bench)
        server.shutdown()
        server.server_close()


def request(url: str, data: bytes | None = None, headers=None) -> dict:
    req = urllib.request.Request(url, data=data, headers=headers or {})
    # Long enough to send the worker's logs; short for the state polls.
    timeout = 600 if data and len(data) > 1 << 20 else 10
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read() or b"{}")


def supervise_worker(args, worker_id: str, bench: subprocess.Popen) -> str | None:
    """Run the worker's side of the protocol; return why it aborted, if it did."""
    results = Path(args.results) / "worker"
    base = f"http://{args.head}:{args.port}"
    query = f"?worker={worker_id}"
    head_run: str | None = None
    ready_sent = launched = False
    stop_deadline: float | None = None
    lost_since: float | None = None
    connect_deadline = time.monotonic() + args.connect_timeout
    while bench.poll() is None:
        now = time.monotonic()
        try:
            state = request(f"{base}/state{query}")
        except (OSError, ValueError):
            state = None
        if state is None:
            if head_run is None and now >= connect_deadline:
                return f"head never answered within {args.connect_timeout:.0f} seconds"
            if head_run is not None:
                lost_since = lost_since or now
                if now - lost_since >= args.head_lost_after:
                    return f"head gone for {args.head_lost_after:.0f} seconds"
            time.sleep(args.poll)
            continue
        lost_since = None
        head_run = head_run or state["run"]
        if state["run"] != head_run:
            return "the head was replaced by a new one"
        if state["abort"]:
            return "the head aborted the run"
        if not ready_sent and (results / "ready-to-launch").exists():
            try:
                request(f"{base}/ready{query}", data=b"")
                ready_sent = True
            except (OSError, ValueError) as error:
                log(f"worker: cannot register: {error}")
        # Launch is granted only to a worker that registered, so a pod
        # recreated after launch never touches the file.
        if ready_sent and state["launch"] and not launched:
            (results / "launch-permitted").touch()
            launched = True
        if state["stop"] and stop_deadline is None:
            # run.py's worker treats SIGTERM as the orderly shutdown request
            # and records shutdown_requested, which the audit requires.
            try:
                pid = int((results / "benchmark/launcher.pid").read_text())
                os.kill(pid, signal.SIGTERM)
            except (OSError, ValueError) as error:
                return f"cannot stop the launcher: {error}"
            stop_deadline = now + args.stop_timeout
        if stop_deadline is not None and now >= stop_deadline:
            return f"benchmark did not stop within {args.stop_timeout:.0f} seconds"
        time.sleep(args.poll)
    if stop_deadline is None:
        return f"benchmark exited ({bench.returncode}) before the head asked it to"
    return None


def run_worker(args) -> int:
    results = Path(args.results) / "worker"
    base = f"http://{args.head}:{args.port}"
    worker_id = uuid.uuid4().hex
    bench = subprocess.Popen(args.command, start_new_session=True)
    try:
        reason = supervise_worker(args, worker_id, bench)
        if reason:
            log(f"worker: aborting: {reason}")
            terminate(bench)
        rc = bench.wait()
        if reason:
            rc = rc or 1
        log(f"worker: benchmark exited {rc}")
    finally:
        terminate(bench)

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        if results.is_dir():
            tar.add(results, arcname="worker")
    for attempt in range(1, 6):
        try:
            request(
                f"{base}/result?worker={worker_id}",
                data=buffer.getvalue(),
                headers={"X-Exit-Status": str(rc)},
            )
            break
        except (OSError, ValueError) as error:
            log(f"worker: result upload attempt {attempt}/5 failed: {error}")
            time.sleep(args.poll)
    else:
        log("worker: the head never received the results; it fails the run")
    # The head is the verdict either way: it has this exit status, or it fails
    # for want of the worker's results. A failing worker pod would instead fail
    # the Job (backoffLimit 0) and kill the head before it publishes the check
    # and uploads the logs.
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="role", required=True)
    for role in ("head", "worker"):
        p = sub.add_parser(role)
        p.add_argument("--results", required=True)
        p.add_argument("--port", type=int, default=DEFAULT_PORT)
        p.add_argument("--poll", type=float, default=2)
        if role == "head":
            p.add_argument("--bind", default="0.0.0.0")
            # Under run.py's 240 s wait for shutdown-permitted.
            p.add_argument("--worker-stop-timeout", type=float, default=120)
            p.add_argument("--collect-timeout", type=float, default=180)
            p.add_argument("--worker-lost-after", type=float, default=120)
            p.add_argument("--audit", type=Path, default=HERE / "audit_multihost.py")
        else:
            p.add_argument("--head", required=True)
            p.add_argument("--head-lost-after", type=float, default=60)
            # The head pod pulls the same image at the same time; two hours
            # matches the benchmark's own wait for launch.
            p.add_argument("--connect-timeout", type=float, default=7200)
            p.add_argument("--stop-timeout", type=float, default=90)
        p.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    # Kubernetes stops a pod with SIGTERM (deadline, eviction, the Job ending).
    # Exit through the `finally` blocks, which stop the benchmark's whole
    # session, rather than leave it holding the chips.
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("a benchmark command is required after --")
    return run_head(args) if args.role == "head" else run_worker(args)


if __name__ == "__main__":
    sys.exit(main())
