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
"""CDK (Cloud DevKit) job orchestrator for remote TPU autotuning."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from typing import Any

from .config import CDKConfig, ModelConfig, Trial
from .utils import get_cdk_env, parse_cdk_job_id, sync_and_parse_cdk_job

logger = logging.getLogger(__name__)


class CDKOrchestrator:
    """Manages submitting, monitoring, and retrieving results for CDK jobs."""

    def __init__(
        self, cdk_cfg: CDKConfig, model_cfg: ModelConfig, dry_run: bool = False
    ):
        self.cdk_cfg = cdk_cfg
        self.model_cfg = model_cfg
        self.dry_run = dry_run
        self._ack_code: str | None = self._resolve_ack_code()

    def _resolve_ack_code(self) -> str | None:
        """Resolve the agent ACK code required by CDK when running under AI agents."""
        if self.cdk_cfg.agent_code:
            return self.cdk_cfg.agent_code

        if "CDK_ACK" in os.environ and os.environ["CDK_ACK"].strip():
            return os.environ["CDK_ACK"].strip()

        # Attempt to auto-fetch from cdk agent-letter
        try:
            res = subprocess.run(
                ["cdk", "agent-letter"],
                capture_output=True,
                text=True,
                check=False,
                timeout=2,
            )
            if res.returncode in (0, 1) and res.stdout:
                codes = re.findall(r"\[ack \d+/\d+: ([0-9A-F]{4})\]", res.stdout)
                if codes:
                    ack = "-".join(codes)
                    logger.info("Resolved CDK agent code: %s", ack)
                    return ack
        except Exception as e:
            logger.debug("Could not resolve CDK agent code: %s", e)

        return None

    def _get_env(self) -> dict[str, str]:
        """Prepare environment for CDK CLI calls, ensuring CDK_ROOT_DIR is set."""
        return get_cdk_env(cdk_root_dir=self.cdk_cfg.cdk_root_dir)

    def _build_cdk_cmd(self, subcmd_and_args: list[str]) -> list[str]:
        """Build a cdk CLI command line with agent code and profile flags."""
        cmd = ["cdk"]
        if self._ack_code:
            cmd.append(f"--agent-code={self._ack_code}")
        if self.cdk_cfg.profile:
            cmd.append(f"--profile={self.cdk_cfg.profile}")
        cmd.extend(subcmd_and_args)
        return cmd

    def run_trial(
        self,
        trial: Trial,
        trial_dir: str,
        target_metric: str,
    ) -> tuple[bool, dict[str, Any], str | None, str]:
        """Submit a CDK job for this trial, wait for it to finish, and extract metrics.

        Returns: (success, metrics_dict, cdk_job_id, error_message)
        """
        os.makedirs(trial_dir, exist_ok=True)
        flag_val = trial.flag or ""

        # Construct template parameters for the recipe
        recipe_params = dict(self.cdk_cfg.extra_recipe_params)
        if flag_val:
            recipe_params["EXTRA_LIBTPU_INIT_ARGS"] = flag_val

        tags = list(self.cdk_cfg.tags)
        tags.append(f"trial-{trial.trial_id}")

        create_args = [
            "job",
            "create",
            self.cdk_cfg.recipe,
        ]

        for k, v in recipe_params.items():
            create_args.append(f"{k}={v}")

        if tags:
            create_args.extend(["-t", ",".join(tags)])

        full_cmd = self._build_cdk_cmd(create_args)
        logger.info(
            "[cdk] Submitting job for %s: %s", trial.trial_id, " ".join(full_cmd)
        )

        with open(
            os.path.join(trial_dir, "cdk_submit.cmd"), "w", encoding="utf-8"
        ) as f:
            f.write(" ".join(full_cmd) + "\n")

        if self.dry_run:
            logger.info(
                "[dry-run] Simulating CDK job submission for %s", trial.trial_id
            )
            mock_id = f"j-simulated-{trial.trial_id}"
            mock_metrics = {
                "request_throughput": 15.2,
                "total_token_throughput": 1520.0,
                "mean_ttft_ms": 42.0,
                "mean_tpot_ms": 3.9,
            }
            return True, mock_metrics, mock_id, ""

        existing_submit_log = os.path.join(trial_dir, "cdk_submit.log")
        job_id = None
        if os.path.exists(existing_submit_log):
            with open(existing_submit_log, encoding="utf-8", errors="replace") as f:
                prev_text = f.read()
            candidate_id = parse_cdk_job_id(prev_text)
            if candidate_id:
                cmd = self._build_cdk_cmd(
                    ["job", "list", "-a", "-n", "50", "-o", "json"]
                )
                try:
                    res = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=15,
                        env=self._get_env(),
                    )
                    if res.returncode == 0 and res.stdout.strip():
                        jobs = json.loads(res.stdout)
                        target = next(
                            (j for j in jobs if j.get("id") == candidate_id), None
                        )
                        if target and target.get("job_status") in (
                            "Pending",
                            "Running",
                            "InQueue",
                            "Succeeded",
                        ):
                            logger.info(
                                "[cdk] Attaching to existing CDK job %s for %s (status: %s)",
                                candidate_id,
                                trial.trial_id,
                                target.get("job_status"),
                            )
                            job_id = candidate_id
                except Exception as e:
                    logger.debug(
                        "Could not verify existing job %s: %s", candidate_id, e
                    )

        if job_id is None:
            # Execute submission
            proc = subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                check=False,
                env=self._get_env(),
            )

            with open(
                os.path.join(trial_dir, "cdk_submit.log"), "w", encoding="utf-8"
            ) as f:
                f.write(proc.stdout)
                if proc.stderr:
                    f.write("\n--- STDERR ---\n" + proc.stderr)

            if proc.returncode != 0:
                err = f"cdk job create failed (rc={proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
                logger.error("[cdk] %s", err)
                return False, {}, None, err

            # Extract full Job ID (e.g. j-7fb420a0-198f-42df-9d73)
            job_id = parse_cdk_job_id(proc.stdout)
            if not job_id:
                err = f"Could not parse job ID from cdk output: {proc.stdout}"
                logger.error("[cdk] %s", err)
                return False, {}, None, err
            logger.info("[cdk] Job created successfully: %s", job_id)

        # Wait for completion
        success, err = self._poll_job(job_id, trial_dir)
        if not success:
            return False, {}, job_id, err

        # Fetch logs & parse metrics
        metrics, log_err = sync_and_parse_cdk_job(
            job_id, trial_dir, env=self._get_env()
        )
        if log_err:
            return False, metrics, job_id, log_err

        return True, metrics, job_id, ""

    def _poll_job(self, job_id: str, trial_dir: str) -> tuple[bool, str]:
        """Poll `cdk job list -a -o json` until terminal state is reached."""
        t0 = time.time()
        timeout = self.cdk_cfg.job_timeout_s
        interval = self.cdk_cfg.poll_interval_s

        logger.info("[cdk] Monitoring job %s (timeout: %ds)...", job_id, timeout)

        while time.time() - t0 < timeout:
            cmd = self._build_cdk_cmd(["job", "list", "-a", "-n", "100", "-o", "json"])
            try:
                res = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=30,
                    env=self._get_env(),
                )
            except subprocess.TimeoutExpired:
                logger.warning("[cdk] cdk job list timed out, retrying...")
                time.sleep(interval)
                continue

            if res.returncode == 0 and res.stdout.strip():
                try:
                    jobs = json.loads(res.stdout)
                    target = next((j for j in jobs if j.get("id") == job_id), None)
                    if target:
                        status = target.get("job_status")
                        state = target.get("state")
                        logger.info(
                            "[cdk] Job %s status: %s, state: %s (elapsed %ds)",
                            job_id,
                            status,
                            state,
                            int(time.time() - t0),
                        )

                        if status == "Succeeded":
                            return True, ""
                        elif status == "Failed":
                            return False, f"Job {job_id} failed in state {state}"
                except json.JSONDecodeError:
                    pass

            time.sleep(interval)

        return False, f"Job {job_id} timed out after {timeout}s"
