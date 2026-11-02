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
"""Unit tests for `.buildkite/scripts/bootstrap.py` and its shell wrappers."""

from __future__ import annotations

import http.client
import http.server
import importlib
import json
import os
import pathlib
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
from collections.abc import Mapping
from typing import Any
from unittest import mock

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _REPO_ROOT / ".buildkite" / "scripts"

_FAKE_AGENT_BODY = """
import json
import os
import pathlib
import sys
import yaml

log_path = pathlib.Path(os.environ["TEST_UPLOAD_LOG"])
args = sys.argv[1:]
if args[:2] == ["meta-data", "set"]:
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"meta": [args[2], args[3]]}) + "\\n")
elif args[:2] == ["annotate", "--style"]:
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"annotate": [args[2], args[4], args[5]]}) + "\\n")
elif args[:2] == ["pipeline", "upload"]:
    content = sys.stdin.read()
    if "--dry-run" in args:
        if os.environ.get("TEST_DRY_RUN_FAIL") == "1":
            sys.exit(1)
        if "TEST_DRY_RUN_RAW" in os.environ:
            sys.stdout.write(os.environ["TEST_DRY_RUN_RAW"])
            sys.exit(0)
        doc = yaml.safe_load(content) or {}
        clean = {k: v for k, v in doc.items() if not k.startswith("x-")}
        json.dump(clean, sys.stdout)
    else:
        print("agent: pipeline upload", flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"args": args[2:], "content": content}) + "\\n")
"""


def _install_upload_stubs(bin_dir: pathlib.Path) -> None:
    agent = bin_dir / "buildkite-agent"
    agent.write_text(
        f"#!{sys.executable}\n{_FAKE_AGENT_BODY.lstrip()}", encoding="utf-8"
    )
    agent.chmod(agent.stat().st_mode | stat.S_IEXEC)
    py3_wrap = bin_dir / "python3"
    py3_wrap.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
    py3_wrap.chmod(py3_wrap.stat().st_mode | stat.S_IEXEC)


class TestBootstrapUnits(unittest.TestCase):
    """Focused unit tests for planning helpers, rendering, effects, and CLI."""

    bp: Any

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        had_bp = "bootstrap" in sys.modules
        had_cap = "cap_pr_tpu_concurrency" in sys.modules
        with mock.patch.object(sys, "path", [str(_SCRIPTS_DIR), *sys.path]):
            cls.bp = importlib.import_module("bootstrap")

        def _cleanup_modules() -> None:
            if not had_bp:
                sys.modules.pop("bootstrap", None)
            if not had_cap:
                sys.modules.pop("cap_pr_tpu_concurrency", None)

        cls.addClassCleanup(_cleanup_modules)

    def test_pipelines_yaml_is_plain_block_style(self) -> None:
        raw = (_REPO_ROOT / ".buildkite" / "pipelines.yaml").read_text(encoding="utf-8")
        for idx, line in enumerate(raw.splitlines(), start=1):
            code = line.split("#", 1)[0]
            self.assertNotIn("{", code, f"line {idx} uses flow mapping")
            self.assertNotIn("[", code, f"line {idx} uses flow sequence")
            self.assertIsNone(
                re.search(r"(^|\s)(&|\*|<<:)", code), f"line {idx} uses YAML anchor"
            )
        for slug in (
            "vllm-torchtpu-ci",
            "vllm-torchtpu-torchtpu-nightly",
            "vllm-torchtpu-integration",
        ):
            spec = self.bp.load_pipeline_spec(slug)
            self.assertEqual(spec["slug"], slug)

    def test_plan_priority_and_concurrency_group(self) -> None:
        bp = self.bp
        nightly = bp.plan_priority(bp.BuildContext.from_env({"NIGHTLY": "1"}))
        self.assertEqual(nightly.job_priority, bp.PRIORITY_NIGHTLY)
        self.assertIsNone(nightly.workload_priority)
        self.assertIsNone(nightly.pr_tpu_concurrency_group)

        pr = bp.plan_priority(
            bp.BuildContext.from_env(
                {
                    "BUILDKITE_PULL_REQUEST": "123",
                    "BUILDKITE_BUILD_AUTHOR_EMAIL": "Alice@Example.com",
                }
            )
        )
        self.assertEqual(pr.job_priority, bp.PRIORITY_PRE_MERGE)
        self.assertEqual(pr.workload_priority, "pre-merge")
        self.assertEqual(pr.pr_tpu_concurrency_group, "tpu-pr/alice@example.com")

        oncall = bp.plan_priority(
            bp.BuildContext.from_env(
                {
                    "BUILDKITE_PULL_REQUEST": "123",
                    "BUILDKITE_PULL_REQUEST_LABELS": "ready,oncall-fix",
                    "BUILDKITE_BUILD_CREATOR_EMAIL": "Oncall@Example.com",
                }
            )
        )
        self.assertEqual(oncall.job_priority, bp.PRIORITY_ONCALL_FIX)
        self.assertEqual(oncall.workload_priority, "oncall-fix")
        self.assertEqual(oncall.annotation.context, "oncall-fix")
        self.assertEqual(oncall.pr_tpu_concurrency_group, "tpu-pr/oncall@example.com")

        post_merge = bp.plan_priority(
            bp.BuildContext.from_env({"BUILDKITE_BRANCH": "main"})
        )
        self.assertEqual(post_merge.job_priority, bp.PRIORITY_POST_MERGE)
        self.assertEqual(post_merge.workload_priority, "post-merge")

        general = bp.plan_priority(
            bp.BuildContext.from_env({"BUILDKITE_BRANCH": "release/v0.1"})
        )
        self.assertEqual(general.job_priority, bp.PRIORITY_DEFAULT)
        self.assertIsNone(general.workload_priority)

    def test_skip_docs_vllm_pin_and_merged_pr_parsing(self) -> None:
        bp = self.bp
        self.assertTrue(bp.has_skip_ci_marker("chore: note [skip ci]"))
        self.assertTrue(bp.has_skip_ci_marker("chore: note [ci skip]"))
        self.assertFalse(bp.has_skip_ci_marker("feat: normal change"))

        self.assertFalse(bp.only_docs_changed([]))
        self.assertTrue(
            bp.only_docs_changed(
                ["README.md", "docs/guide.md", ".pyrefly-baseline.json"]
            )
        )
        self.assertFalse(
            bp.only_docs_changed([".pyrefly-baseline-full.json", "README.md"])
        )
        self.assertFalse(bp.only_docs_changed(["src/vllm_torchtpu/envs.py"]))

        pyproject = (
            "[dependency-groups]\nvllm = [\n"
            '  "vllm @ git+https://github.com/vllm-project/vllm.git@abc123def",\n]\n'
        )
        self.assertEqual(bp.parse_pinned_vllm_commit(pyproject), "abc123def")
        self.assertEqual(
            bp.resolve_vllm_commit_hash("abc123def", "override99"), "override99"
        )
        with self.assertRaises(bp.BootstrapError):
            bp.parse_pinned_vllm_commit("[project]\nname = 'vllm-torchtpu'\n")

        self.assertEqual(
            bp.parse_merged_pr_number("feat: add kernel (#789)\nbody (#999)"), "789"
        )
        self.assertIsNone(bp.parse_merged_pr_number("chore: direct commit"))

    def test_render_pipeline_conditions_variants_extends_and_errors(self) -> None:
        bp = self.bp
        ci = {"BUILDKITE_PIPELINE_SLUG": "vllm-torchtpu-ci"}

        pr = bp.render_pipeline({**ci, "BUILDKITE_PULL_REQUEST": "10"})
        self.assertEqual(
            pr.files,
            ("pipeline_build.yml", "pipeline_tests.yml", "pipeline_integration.yml"),
        )
        self.assertEqual(pr.omni_image_skip, "")

        # Release tag runs models + verify alongside build, tests, integration.
        tag = bp.render_pipeline({**ci, "BUILDKITE_TAG": "v0.8.0"})
        self.assertIn("pipeline_models.yml", tag.files)
        self.assertIn("pipeline_verify.yml", tag.files)
        self.assertNotIn("pipeline_perf.yml", tag.files)

        # skip-tpu-ci overrides run-full-ci and run-perf-eval; skip reason <= 70 chars.
        skip = bp.render_pipeline(
            {
                **ci,
                "BUILDKITE_PULL_REQUEST": "10",
                "BUILDKITE_PULL_REQUEST_LABELS": (
                    "run-full-ci,run-perf-eval,skip-tpu-ci"
                ),
            }
        )
        self.assertTrue(skip.skip_tpu_ci)
        self.assertEqual(skip.files, ("pipeline_build.yml", "pipeline_tests.yml"))
        self.assertLessEqual(len(skip.omni_image_skip), 70)

        # CI_VARIANT=perf-only uploads only build + perf even without NIGHTLY=1.
        perf_only = bp.render_pipeline(
            {**ci, "CI_VARIANT": "perf-only", "BUILDKITE_SOURCE": "schedule"}
        )
        self.assertEqual(perf_only.files, ("pipeline_build.yml", "pipeline_perf.yml"))
        self.assertLessEqual(len(perf_only.omni_image_skip), 70)

        # Validation errors on missing/unknown slug, variant, condition, or key.
        with self.assertRaisesRegex(bp.BootstrapError, "BUILDKITE_PIPELINE_SLUG"):
            bp.render_pipeline({})
        with self.assertRaisesRegex(bp.BootstrapError, "unknown pipeline"):
            bp.render_pipeline({"BUILDKITE_PIPELINE_SLUG": "bogus"})
        with self.assertRaisesRegex(bp.BootstrapError, "unknown CI_VARIANT"):
            bp.render_pipeline({**ci, "CI_VARIANT": "bogus"})

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            bk = root / ".buildkite"
            bk.mkdir()
            (bk / "a.yml").write_text("steps:\n  - key: a\n    command: 'true'\n")
            (bk / "b.yml").write_text("steps:\n  - key: b\n    command: 'true'\n")
            cfg = bk / "pipelines.yaml"
            cfg.write_text(
                "p:\n"
                "  files:\n"
                "    a.yml:\n"
                "      run_if:\n"
                "        - pr\n"
                "    b.yml:\n"
                "      depends_on:\n"
                "        - file: a.yml\n"
                "          skip_if:\n"
                "            - scheduled\n"
            )
            # Conditional depends_on dropped when skipped or when target not uploaded.
            non_pr = bp.render_pipeline(
                {"BUILDKITE_PIPELINE_SLUG": "p"}, repo_root=root, pipelines_path=cfg
            )
            self.assertEqual(non_pr.files, ("b.yml",))
            self.assertNotIn("depends_on", non_pr.document["steps"][0])

            sched_pr = bp.render_pipeline(
                {
                    "BUILDKITE_PIPELINE_SLUG": "p",
                    "BUILDKITE_PULL_REQUEST": "1",
                    "BUILDKITE_SOURCE": "schedule",
                },
                repo_root=root,
                pipelines_path=cfg,
            )
            self.assertNotIn("depends_on", sched_pr.document["steps"][1])

            gated_pr = bp.render_pipeline(
                {"BUILDKITE_PIPELINE_SLUG": "p", "BUILDKITE_PULL_REQUEST": "1"},
                repo_root=root,
                pipelines_path=cfg,
            )
            self.assertEqual(gated_pr.document["steps"][1]["depends_on"], ["a"])

            # Bad condition and keyless top-level step raise BootstrapError.
            cfg.write_text("p:\n  files:\n    a.yml:\n      run_if:\n        - nope\n")
            with self.assertRaisesRegex(bp.BootstrapError, "unknown condition"):
                bp.render_pipeline(
                    {"BUILDKITE_PIPELINE_SLUG": "p"}, repo_root=root, pipelines_path=cfg
                )
            (bk / "a.yml").write_text("steps:\n  - label: keyless\n    command: ok\n")
            cfg.write_text("p:\n  files:\n    a.yml:\n")
            with self.assertRaisesRegex(bp.BootstrapError, "lacks a key"):
                bp.render_pipeline(
                    {"BUILDKITE_PIPELINE_SLUG": "p"}, repo_root=root, pipelines_path=cfg
                )

    def test_effects_github_error_resilience(self) -> None:
        bp = self.bp
        seen_auth: list[str | None] = []

        class AuthHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                seen_auth.append(self.headers.get("Authorization"))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'[{"name": "skip-tpu-ci"}]')

            def log_message(self, format: str, *args: object) -> None:
                return

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            helper = tmp_path / "git-credential-github-app"
            helper.write_text(
                "#!/usr/bin/env bash\necho password=ghs_tok\n", encoding="utf-8"
            )
            helper.chmod(helper.stat().st_mode | stat.S_IEXEC)

            slow_helper = tmp_path / "slow-credential-helper"
            slow_helper.write_text("#!/usr/bin/env bash\nsleep 0.5\n", encoding="utf-8")
            slow_helper.chmod(slow_helper.stat().st_mode | stat.S_IEXEC)

            server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), AuthHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                api_url = f"http://127.0.0.1:{server.server_address[1]}"
                ok_effects = bp.BootstrapEffects(
                    repo_root=_REPO_ROOT,
                    credential_helper=helper,
                    github_api_url=api_url,
                )
                timeout_effects = bp.BootstrapEffects(
                    repo_root=_REPO_ROOT,
                    credential_helper=slow_helper,
                    credential_timeout=0.05,
                    github_api_url=api_url,
                )
                self.assertEqual(
                    ok_effects.merged_pr_labels("42", env={}), ("skip-tpu-ci",)
                )
                self.assertEqual(
                    timeout_effects.merged_pr_labels("42", env={}), ("skip-tpu-ci",)
                )
            finally:
                server.shutdown()
                server.server_close()

            self.assertEqual(seen_auth, ["token ghs_tok", None])

            with mock.patch(
                "urllib.request.urlopen",
                side_effect=http.client.IncompleteRead(b"partial", 10),
            ):
                err_effects = bp.BootstrapEffects(
                    repo_root=_REPO_ROOT,
                    credential_helper=tmp_path / "missing",
                )
                self.assertEqual(err_effects.merged_pr_labels("42", env={}), ())

    def test_dev_mode_vllm_pin_and_shell_wrappers(self) -> None:
        bp = self.bp
        dev_default = bp.plan_dev_bootstrap({})
        self.assertEqual(dev_default.raw_job_priority, str(bp.PRIORITY_DEFAULT))
        self.assertEqual(dev_default.pipeline_file, ".buildkite/pipeline_dev.yml")

        dev_custom = bp.plan_dev_bootstrap(
            {
                "DEV_JOB_PRIORITY": "4",
                "DEV_PIPELINE_FILE": ".buildkite/exp.yml",
                "BUILDKITE_BRANCH": "exp/test",
                "BUILDKITE_COMMIT": "abc1234",
                "NIGHTLY": "1",
            }
        )
        self.assertEqual(bp.parse_dev_job_priority(dev_custom.raw_job_priority), 4)
        self.assertTrue(dev_custom.nightly)
        with self.assertRaises(bp.BootstrapError):
            bp.parse_dev_job_priority("not-an-int")

        good_cli = subprocess.run(
            [sys.executable, str(_SCRIPTS_DIR / "bootstrap.py"), "vllm-pin"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(
            good_cli.stdout.strip(), bp.read_pinned_vllm_commit(_REPO_ROOT)
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_repo = pathlib.Path(tmp) / "repo"
            shutil.copytree(_SCRIPTS_DIR, tmp_repo / ".buildkite" / "scripts")
            with self.assertRaisesRegex(bp.BootstrapError, "No such file"):
                bp.read_pinned_vllm_commit(tmp_repo)
            (tmp_repo / "pyproject.toml").write_text(
                "[project]\nname = 'empty'\n", encoding="utf-8"
            )
            bad_cli = subprocess.run(
                [
                    sys.executable,
                    str(tmp_repo / ".buildkite" / "scripts" / "bootstrap.py"),
                    "vllm-pin",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(bad_cli.returncode, 1)
            self.assertIn(
                "ERROR: vLLM commit hash is missing from pyproject.toml.",
                bad_cli.stderr,
            )

            tmp_path = pathlib.Path(tmp)
            bin_dir = tmp_path / "bin"
            bin_dir.mkdir()
            _install_upload_stubs(bin_dir)
            log_path = tmp_path / "uploads.jsonl"

            sample_yaml = tmp_path / "sample.yml"
            sample_yaml.write_text(
                "steps:\n"
                '  - label: "TPU step"\n'
                '    if: \'build.env("NIGHTLY") == "1"\'\n'
                "    agents:\n"
                '      queue: "kube"\n'
                "    env:\n"
                '      SHAPE: "tpu7x-standard-1t/1x1x1"\n'
                "    commands:\n"
                "      - echo hi\n",
                encoding="utf-8",
            )
            env = {
                "PATH": f"{bin_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
                "HOME": str(tmp_path),
                "TEST_UPLOAD_LOG": str(log_path),
                "DEV_PIPELINE_FILE": str(sample_yaml),
                "NIGHTLY": "1",
                "PR_TPU_CONCURRENCY_GROUP": "tpu-pr/dev@example.com",
            }
            ci_proc = subprocess.run(
                [
                    "bash",
                    str(_SCRIPTS_DIR / "bootstrap.sh"),
                    "--pipeline",
                    "vllm-torchtpu-ci",
                ],
                cwd=_REPO_ROOT,
                env={**env, "BUILDKITE_MESSAGE": "chore [skip ci]"},
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("Commit message contains [skip ci].", ci_proc.stdout)

            capped = subprocess.run(
                ["bash", str(_SCRIPTS_DIR / "bootstrap_dev.sh")],
                cwd=_REPO_ROOT,
                env={**env, "DEV_JOB_PRIORITY": "3"},
                capture_output=True,
                text=True,
                check=True,
            )
            lines = capped.stdout.splitlines()
            hdr = lines.index(f"--- :pipeline: Uploading {sample_yaml} with priority 3")
            self.assertEqual(lines[hdr + 1], "agent: pipeline upload")

            for extra_env in (
                {"TEST_DRY_RUN_FAIL": "1"},
                {"TEST_DRY_RUN_RAW": "{not-json"},
                {"TEST_DRY_RUN_RAW": "[1, 2]"},
            ):
                uncapped = subprocess.run(
                    ["bash", str(_SCRIPTS_DIR / "bootstrap_dev.sh")],
                    cwd=_REPO_ROOT,
                    env={**env, "DEV_JOB_PRIORITY": "9", **extra_env},
                    capture_output=True,
                    text=True,
                    check=True,
                )
                self.assertIn("WARNING: could not put", uncapped.stderr)

            events = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
            ]
            uploads = [e for e in events if "args" in e]
            self.assertEqual(uploads[0]["args"], ["--no-interpolation"])
            parsed = json.loads(uploads[0]["content"])
            self.assertEqual(parsed["priority"], 3)
            self.assertEqual(
                parsed["steps"][0]["concurrency_group"], "tpu-pr/dev@example.com"
            )

            missing = subprocess.run(
                ["bash", str(_SCRIPTS_DIR / "bootstrap_dev.sh")],
                cwd=_REPO_ROOT,
                env={
                    **env,
                    "DEV_PIPELINE_FILE": ".buildkite/missing_dev.yml",
                    "BUILDKITE_BRANCH": "exp/missing",
                    "BUILDKITE_COMMIT": "deadbeef",
                },
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(missing.returncode, 1)
            self.assertIn("missing_dev.yml' not found", missing.stdout)

    def test_run_bootstrap_orchestration(self) -> None:
        import yaml  # pyrefly: ignore[missing-import]

        bp = self.bp
        pinned = bp.parse_pinned_vllm_commit(
            (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )

        class FakeEffects(bp.BootstrapEffects):
            def __init__(
                self,
                files: list[str] | None = None,
                labels: tuple[str, ...] = (),
            ) -> None:
                super().__init__(repo_root=_REPO_ROOT)
                self.files = ["src/vllm_torchtpu/envs.py"] if files is None else files
                self.labels = labels
                self.meta: list[tuple[str, str]] = []
                self.annotations: list[str] = []
                self.uploads: list[tuple[dict[str, Any], dict[str, str]]] = []

            def set_metadata(
                self, key: str, value: str, *, env: Mapping[str, str]
            ) -> None:
                self.meta.append((key, value))

            def annotate(self, annotation: Any, *, env: Mapping[str, str]) -> None:
                self.annotations.append(annotation.context)

            def upload_pipeline(
                self,
                content: str,
                *,
                no_interpolation: bool,
                env: Mapping[str, str],
            ) -> None:
                self.uploads.append((yaml.safe_load(content), dict(env)))

            def pr_changed_files(
                self, base_branch: str, commit: str, *, env: Mapping[str, str]
            ) -> list[str]:
                return list(self.files)

            def commit_changed_files(
                self, commit: str, *, env: Mapping[str, str]
            ) -> list[str]:
                return list(self.files)

            def merged_pr_labels(
                self, pr_number: str, *, env: Mapping[str, str]
            ) -> tuple[str, ...]:
                return self.labels

            def vllm_upstream_head(self, *, env: Mapping[str, str]) -> str:
                return "upstream999"

        def run(
            env: dict[str, str],
            *,
            files: list[str] | None = None,
            labels: tuple[str, ...] = (),
        ) -> FakeEffects:
            merged = {"BUILDKITE_PIPELINE_SLUG": "vllm-torchtpu-ci", **env}
            fx = FakeEffects(files=files, labels=labels)
            self.assertEqual(
                bp.run_bootstrap(merged, repo_root=_REPO_ROOT, effects=fx), 0
            )
            return fx

        # PR default: 1 upload, build -> tests + integration, per-author cap applied.
        pr = run(
            {
                "BUILDKITE_PULL_REQUEST": "42",
                "BUILDKITE_BUILD_AUTHOR_EMAIL": "Alice@Example.com",
            }
        )
        self.assertEqual(
            pr.meta,
            [
                ("JOB_PRIORITY", "6"),
                ("WORKLOAD_PRIORITY", "pre-merge"),
                ("VLLM_COMMIT_HASH", pinned),
            ],
        )
        self.assertEqual(pr.annotations, ["pipeline-graph"])
        self.assertEqual(len(pr.uploads), 1)
        pr_steps = {s["key"]: s for s in pr.uploads[0][0]["steps"]}
        self.assertIn("build_docker_group", pr_steps["unit_tests_tpu"]["depends_on"])
        self.assertIn("build_docker_group", pr_steps["integration_tpu"]["depends_on"])
        self.assertIn("tpu-pr/alice@example.com", yaml.safe_dump(pr.uploads[0][0]))

        # PR skip-tpu-ci & post-merge main webhook honoring merged PR skip-tpu-ci.
        for fx in (
            run(
                {
                    "BUILDKITE_PULL_REQUEST": "42",
                    "BUILDKITE_PULL_REQUEST_LABELS": "ready,skip-tpu-ci",
                }
            ),
            run(
                {
                    "BUILDKITE_BRANCH": "main",
                    "BUILDKITE_SOURCE": "webhook",
                    "BUILDKITE_MESSAGE": "fix: tweak (#99)",
                },
                labels=("skip-tpu-ci",),
            ),
        ):
            self.assertEqual(fx.annotations, ["skip-tpu-ci", "pipeline-graph"])
            self.assertEqual(fx.uploads[0][1]["SKIP_TPU_CI"], "1")
            self.assertTrue(bool(fx.uploads[0][1]["OMNI_IMAGE_SKIP"]))
            keys = {s["key"] for s in fx.uploads[0][0]["steps"]}
            self.assertNotIn("integration_tpu", keys)

        # Nightly on both ci and torchtpu-nightly slugs.
        for slug in ("vllm-torchtpu-ci", "vllm-torchtpu-torchtpu-nightly"):
            nightly = run(
                {
                    "BUILDKITE_PIPELINE_SLUG": slug,
                    "NIGHTLY": "1",
                    "BUILDKITE_SOURCE": "schedule",
                    "BUILDKITE_BRANCH": "main",
                }
            )
            n_steps = {s["key"]: s for s in nightly.uploads[0][0]["steps"]}
            self.assertIn("upload_models_and_features", n_steps)
            self.assertIn(
                "upload_models_and_features", n_steps["nightly_report"]["depends_on"]
            )
            self.assertIn("v6e-group", n_steps["nightly_report"]["depends_on"])

        # Integration pipeline (upstream head + promote_lkg) and early exits.
        integ = run({"BUILDKITE_PIPELINE_SLUG": "vllm-torchtpu-integration"})
        self.assertIn(("VLLM_COMMIT_HASH", "upstream999"), integ.meta)
        i_steps = {s["key"]: s for s in integ.uploads[0][0]["steps"]}
        self.assertEqual(
            set(i_steps["promote_lkg"]["depends_on"]),
            {
                "build_docker_group",
                "typecheck_full",
                "unit_tests_cpu_pr",
                "unit_tests_cpu_nightly",
                "unit_tests_tpu",
                "integration_tpu",
                "perf_tpu",
            },
        )
        self.assertEqual(run({"BUILDKITE_MESSAGE": "doc [skip ci]"}).uploads, [])
        self.assertEqual(
            run({"BUILDKITE_PULL_REQUEST": "42"}, files=["README.md"]).uploads, []
        )


if __name__ == "__main__":
    unittest.main()
