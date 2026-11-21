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
"""Lifecycle manager for local or in-pod vLLM server instances."""

from __future__ import annotations

import atexit
import logging
import os
import signal
import socket
import subprocess
import threading
import time
import urllib.request

from .config import ModelConfig

logger = logging.getLogger(__name__)


class ServerManager:
    """Controls starting, health polling, and stopping of `vllm serve`."""

    def __init__(self, model_cfg: ModelConfig, dry_run: bool = False):
        self.model_cfg = model_cfg
        self.dry_run = dry_run
        self._server_proc: subprocess.Popen | None = None
        atexit.register(self.stop_server)

    def build_env(self, extra_libtpu_init_args: list[str]) -> dict[str, str]:
        """Construct the environment dictionary with candidate XLA flags."""
        env = os.environ.copy()

        # Combine base flags and trial flags
        flag_parts = []
        if self.model_cfg.base_libtpu_init_args.strip():
            flag_parts.append(self.model_cfg.base_libtpu_init_args.strip())
        if extra_libtpu_init_args:
            flag_parts.extend([f.strip() for f in extra_libtpu_init_args if f.strip()])

        env["LIBTPU_INIT_ARGS"] = " ".join(flag_parts)
        # Avoid premature recompilation assertion failures during flag sweeps
        env["VLLM_XLA_CHECK_RECOMPILATION"] = "0"

        # Apply model-specific environment variables
        for k, v in self.model_cfg.server_env.items():
            env[k] = str(v)

        return env

    def start_server(
        self, extra_libtpu_init_args: list[str], trial_dir: str
    ) -> dict[str, str]:
        """Launch the server process and wait for readiness."""
        env = self.build_env(extra_libtpu_init_args)
        os.makedirs(trial_dir, exist_ok=True)

        cmd = [
            "vllm",
            "serve",
            self.model_cfg.get_effective_model_path(),
            "--port",
            str(self.model_cfg.port),
            *self.model_cfg.serve_args,
        ]

        cmd_path = os.path.join(trial_dir, "vllm_server.cmd")
        with open(cmd_path, "w", encoding="utf-8") as f:
            f.write(" ".join(cmd) + "\n")

        env_path = os.path.join(trial_dir, "vllm_server_env.txt")
        with open(env_path, "w", encoding="utf-8") as f:
            for k in sorted(env):
                f.write(f"{k}={env[k]}\n")

        if self.dry_run:
            logger.info("[dry-run] Would launch: %s", " ".join(cmd))
            return env

        log_path = os.path.join(trial_dir, "vllm_server.log")
        logger.info("Starting vllm serve on port %d...", self.model_cfg.port)

        self._server_proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
            text=True,
            errors="replace",
        )

        def _stream_logs() -> None:
            with open(log_path, "w", encoding="utf-8") as lf:
                for line in iter(self._server_proc.stdout.readline, ""):
                    lf.write(line)
                    lf.flush()

        threading.Thread(target=_stream_logs, daemon=True).start()
        self._wait_for_ready()
        return env

    def _wait_for_ready(self) -> None:
        """Poll the server port and HTTP health endpoint until healthy."""
        t0 = time.time()
        timeout = self.model_cfg.startup_timeout_s
        port = self.model_cfg.port
        health_url = f"http://127.0.0.1:{port}{self.model_cfg.health_endpoint}"

        logger.info("Waiting for server readiness (timeout: %ds)...", timeout)

        while True:
            if self._server_proc is not None and self._server_proc.poll() is not None:
                raise RuntimeError(
                    f"vllm serve process exited prematurely with code {self._server_proc.poll()}"
                )

            if time.time() - t0 > timeout:
                raise TimeoutError(
                    f"vllm serve failed to become ready on port {port} within {timeout}s"
                )

            # 1. Quick TCP socket probe
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    # 2. HTTP health check probe
                    try:
                        req = urllib.request.Request(health_url)
                        with urllib.request.urlopen(req, timeout=2) as resp:
                            if resp.status == 200:
                                elapsed = int(time.time() - t0)
                                logger.info("vllm serve is ready! (took %ds)", elapsed)
                                return
                    except Exception:
                        pass
            except OSError:
                pass

            time.sleep(5)

    def stop_server(self) -> None:
        """Terminate the server process group gracefully."""
        if not self._server_proc:
            return

        logger.info("Stopping vllm serve...")
        try:
            pgid = os.getpgid(self._server_proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                self._server_proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "vllm serve did not terminate within 20s, sending SIGKILL..."
                )
                os.killpg(pgid, signal.SIGKILL)
                self._server_proc.wait()
        except ProcessLookupError:
            pass
        finally:
            self._server_proc = None
