"""eval_runner.py.

Quality and accuracy evaluation runner using lm-evaluation-harness on Cloud TPUs.
Executes standard benchmarks (e.g., MMLU-Pro) via the vLLM OpenAI-compatible endpoint.
"""

import glob
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from core.env import HF_HOME
from core.models import BenchmarkResult, ExecutionContext
from runners.base_runner import BaseVllmRunner


class TpuEvalRunner(BaseVllmRunner):
    """Executes accuracy evaluation via lm-evaluation-harness against an OpenAI-compatible vLLM server."""

    def execute(self, context: ExecutionContext) -> BenchmarkResult:
        """Launches vLLM server, polls health, runs lm_eval, parses metrics, and cleans up."""
        print(
            f"\n=======================================================\n"
            f">>> [EVALUATION] Model: {context.model_path}\n"
            f">>> Tasks: {context.tasks} | Limit: {context.limit} | Batch Size: {context.batch_size}\n"
            f"=======================================================")

        eval_logs_dir = os.path.join(context.output_dir, "eval_logs")
        os.makedirs(eval_logs_dir, exist_ok=True)
        port = 8000
        health_url = f"http://127.0.0.1:{port}/health"

        # 1. Prepare environment and launch vLLM OpenAI server
        server_env = os.environ.copy()

        kv_dtype = str(context.engine_args.get("kv_cache_dtype", "bfloat16"))
        gpu_util = str(context.engine_args.get("gpu_memory_utilization", 0.85))
        server_cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            context.model_path,
            "--tensor-parallel-size",
            str(context.tensor_parallel_size),
            "--max-model-len",
            str(context.max_model_len or 8192),
            "--port",
            str(port),
            "--async-scheduling",
            "--no-enable-prefix-caching",
            "--gpu-memory-utilization",
            gpu_util,
            "--download-dir",
            HF_HOME,
            "--kv-cache-dtype",
            kv_dtype,
            "--disable-chunked-mm-input",
            "--generation-config",
            "vllm",
        ]

        print(f">>> [SERVER] Launching vLLM OpenAI Server on port {port}...")
        server_proc = subprocess.Popen(server_cmd,
                                       env=server_env,
                                       start_new_session=True)
        start_time = datetime.now()

        try:
            # 2. Poll /health endpoint until server is ready
            poll_start = time.time()
            while time.time() - poll_start < 600:
                if server_proc.poll() is not None:
                    raise RuntimeError(
                        f"vLLM server exited prematurely with code {server_proc.returncode}"
                    )
                try:
                    with urllib.request.urlopen(health_url, timeout=2) as resp:
                        if resp.status == 200:
                            print(
                                ">>> [SERVER] vLLM Server is healthy and ready to receive requests!"
                            )
                            break
                except Exception:
                    time.sleep(5)
            else:
                raise TimeoutError(
                    "vLLM server failed to become healthy within 600s")

            # 3. Assemble and execute lm_eval CLI (CPU client with chat templates)
            eval_env = {
                **os.environ,
                "TPU_VISIBLE_DEVICES": "",
                "TPU_VISIBLE_CHIPS": "",
                "HF_HOME": HF_HOME,
            }

            # Set concurrency to 4 to avoid queueing requests which will cause corrupted results in lm_eval
            concurrency = 4
            eval_cmd = [
                sys.executable,
                "-m",
                "lm_eval",
                "--model",
                "local-chat-completions",
                "--model_args",
                f"model={context.model_path},base_url=http://127.0.0.1:{port}/v1/chat/completions,num_concurrent={concurrency}",
                "--tasks",
                str(context.tasks),
                "--apply_chat_template",
                "--fewshot_as_multiturn",
                "--log_samples",
                "--output_path",
                eval_logs_dir,
                "--seed",
                str(context.seed),
            ]
            if context.limit is not None:
                eval_cmd.extend(["--limit", str(context.limit)])
            if "humaneval_instruct" in context.tasks or "mbpp_plus_instruct" in context.tasks:
                eval_env["HF_ALLOW_CODE_EVAL"] = "1"
                eval_cmd.append("--confirm_run_unsafe_code")

            print(
                f">>> [EVALUATION] Executing lm_eval command: {' '.join(eval_cmd)}"
            )
            subprocess.run(eval_cmd, env=eval_env, check=True)

        finally:
            # 4. Clean shutdown of vLLM server and reclaim TPU locks
            print(">>> [SERVER] Shutting down vLLM server...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server_proc.kill()

            # Reclaim TPU locks to avoid "libtpu.so already loaded" errors in subsequent runs
            subprocess.run("fuser -k -9 /dev/vfio/* 2>/dev/null || true",
                           shell=True,
                           check=False)
            Path("/tmp/libtpu_lockfile").unlink(missing_ok=True)

        duration_s = (datetime.now() - start_time).total_seconds()

        # 5. Parse latest evaluation results JSON
        json_files = glob.glob(os.path.join(eval_logs_dir, "**",
                                            "*results*.json"),
                               recursive=True)
        latest_json = max(json_files, key=os.path.getmtime, default=None)

        parsed_scores: Dict[str, Any] = {}
        if latest_json:
            with open(latest_json, "r") as jf:
                results_dict = json.load(jf).get("results", {})
                parsed_scores = {
                    f"{task}_{metric}": f"{val:.4f}"
                    for task, metrics in results_dict.items()
                    for metric, val in metrics.items()
                    if isinstance(val, (int, float))
                }

        metrics = {"Eval_Duration_s": f"{duration_s:.2f}", **parsed_scores}
        print(f">>> [EVALUATION] Complete: Duration {duration_s:.2f}s")
        return BenchmarkResult(duration_s=duration_s,
                               metrics=metrics,
                               log_path=latest_json)
