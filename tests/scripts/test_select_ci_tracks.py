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
"""Unit tests for two-track CI selection, pipeline YAML skip wiring, and Dockerfile."""

from __future__ import annotations

import importlib.util
import os
import pathlib
import re
import stat
import subprocess
import sys
import tempfile
import unittest

import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SELECT_SCRIPT = _REPO_ROOT / ".buildkite" / "scripts" / "select_ci_tracks.py"
_BOOTSTRAP = _REPO_ROOT / ".buildkite" / "scripts" / "bootstrap.sh"

_spec = importlib.util.spec_from_file_location("select_ci_tracks", _SELECT_SCRIPT)
assert _spec is not None and _spec.loader is not None
select_ci_tracks = importlib.util.module_from_spec(_spec)
sys.modules["select_ci_tracks"] = select_ci_tracks
_spec.loader.exec_module(select_ci_tracks)


def _interpolate_buildkite_vars(yaml_text: str, env_vars: dict[str, str]) -> str:
    """Substitute `${VAR:-default}` expressions as `buildkite-agent pipeline upload`."""

    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2) or ""
        val = env_vars.get(name, "")
        return val if val != "" else default

    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]*)\}", repl, yaml_text)


def _iter_command_steps(steps: list[dict]) -> list[dict]:
    """Flatten top-level and group command steps from a parsed Buildkite YAML."""
    flat: list[dict] = []
    for entry in steps:
        if "steps" in entry and isinstance(entry["steps"], list):
            flat.extend(_iter_command_steps(entry["steps"]))
        else:
            flat.append(entry)
    return flat


class TestOmniSharedClosure(unittest.TestCase):
    """Verify AST-derived Omni shared-file boundary on the real repository."""

    omni_shared: frozenset[str] = frozenset()

    @classmethod
    def setUpClass(cls) -> None:
        cls.omni_shared = select_ci_tracks.compute_omni_shared_closure(_REPO_ROOT)

    def test_known_shared_modules_are_included(self) -> None:
        expected_shared = {
            "src/vllm_torchtpu/__init__.py",
            "src/vllm_torchtpu/env_override.py",
            "src/vllm_torchtpu/envs.py",
            "src/vllm_torchtpu/logger.py",
            "src/vllm_torchtpu/profiler_trace.py",
            "src/vllm_torchtpu/tpu_info.py",
            "src/vllm_torchtpu/utils.py",
            "src/vllm_torchtpu/tracing/options.py",
            "src/vllm_torchtpu/platforms/tpu_platform.py",
            "src/vllm_torchtpu/kernels/diffusion_attention/kernel.py",
            "src/vllm_torchtpu/kernels/diffusion_attention/qk_norm_rope.py",
            "src/vllm_torchtpu/layers/adapter/custom_ops/optimization_barrier_op.py",
            "tests/tpu_release.py",
        }
        for rel_path in expected_shared:
            self.assertIn(rel_path, self.omni_shared)

    def test_known_core_only_modules_are_excluded(self) -> None:
        expected_core_only = {
            "src/vllm_torchtpu/runner/tpu_runner.py",
            "src/vllm_torchtpu/worker/tpu_worker.py",
            "src/vllm_torchtpu/offload/raiden_connector.py",
            "src/vllm_torchtpu/spec_decode/eagle3.py",
            "src/vllm_torchtpu/lora/torch_punica_tpu.py",
            "src/vllm_torchtpu/megakernel/planner.py",
            "tests/conftest.py",
        }
        for rel_path in expected_core_only:
            self.assertNotIn(rel_path, self.omni_shared)

    def test_synthetic_repo_discovers_new_imports_automatically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            pkg = root / "src" / "vllm_torchtpu"
            (pkg / "omni").mkdir(parents=True)
            (root / "tests").mkdir()
            (root / "tests_omni").mkdir()

            (pkg / "__init__.py").write_text(
                "import vllm_torchtpu.boot\ndef lazy():\n"
                "    import vllm_torchtpu.core_only\n",
                encoding="utf-8",
            )
            (pkg / "boot.py").write_text("X = 1\n", encoding="utf-8")
            (pkg / "core_only.py").write_text("Y = 2\n", encoding="utf-8")
            (pkg / "shared_util.py").write_text("Z = 3\n", encoding="utf-8")
            (pkg / "omni" / "__init__.py").write_text("", encoding="utf-8")
            (pkg / "omni" / "impl.py").write_text(
                "from vllm_torchtpu.shared_util import Z\n",
                encoding="utf-8",
            )
            (root / "tests" / "helper.py").write_text("H = 1\n", encoding="utf-8")
            (root / "tests_omni" / "test_impl.py").write_text(
                "from tests.helper import H\n",
                encoding="utf-8",
            )

            closure = select_ci_tracks.compute_omni_shared_closure(root)
            self.assertEqual(
                closure,
                frozenset(
                    {
                        "src/vllm_torchtpu/__init__.py",
                        "src/vllm_torchtpu/boot.py",
                        "src/vllm_torchtpu/shared_util.py",
                        "tests/helper.py",
                    }
                ),
            )


class TestPyprojectDiff(unittest.TestCase):
    """Verify structured `tomllib` diffing of `pyproject.toml`."""

    BASE_TOML = """
[project]
name = "vllm-torchtpu-core"
dependencies = ["torch==2.13.0", "jax==0.11.2"]

[project.optional-dependencies]
benchmarking = ["lm-eval==0.5.0.dev1"]
omni = ["vllm-omni==0.32.0rc1", "torchaudio==2.11.0"]

[dependency-groups]
vllm = ["vllm @ git+https://github.com/vllm-project/vllm.git@v0.32.0"]
omni-git = ["vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@v0.32.0rc1"]
test = ["pytest"]

[tool.ruff]
line-length = 88
"""

    def test_omni_pin_bump_is_omni_only(self) -> None:
        head = self.BASE_TOML.replace("0.32.0rc1", "0.32.0")
        self.assertEqual(
            select_ci_tracks.classify_pyproject_diff(self.BASE_TOML, head),
            "omni",
        )

    def test_benchmarking_bump_is_core_only(self) -> None:
        head = self.BASE_TOML.replace("lm-eval==0.5.0.dev1", "lm-eval==0.5.0")
        self.assertEqual(
            select_ci_tracks.classify_pyproject_diff(self.BASE_TOML, head),
            "core",
        )

    def test_core_runtime_or_vllm_pin_bump_runs_both(self) -> None:
        for old, new in (
            ("jax==0.11.2", "jax==0.11.3"),
            ("vllm.git@v0.32.0", "vllm.git@abcdef123456"),
            ("pytest", "pytest>=8.0"),
            ("line-length = 88", "line-length = 100"),
        ):
            head = self.BASE_TOML.replace(old, new)
            self.assertEqual(
                select_ci_tracks.classify_pyproject_diff(self.BASE_TOML, head),
                "both",
            )

    def test_comment_only_change_is_docs(self) -> None:
        head = "# Extra comment\n" + self.BASE_TOML
        self.assertEqual(
            select_ci_tracks.classify_pyproject_diff(self.BASE_TOML, head),
            "docs",
        )

    def test_missing_or_invalid_toml_falls_back_to_both(self) -> None:
        self.assertEqual(
            select_ci_tracks.classify_pyproject_diff(None, self.BASE_TOML),
            "both",
        )
        self.assertEqual(
            select_ci_tracks.classify_pyproject_diff(self.BASE_TOML, "not = [valid"),
            "both",
        )


class TestDecideCiPlan(unittest.TestCase):
    """Verify track selection, label overrides, and Buildkite 70-char skip bounds."""

    omni_shared: frozenset[str] = frozenset()

    @classmethod
    def setUpClass(cls) -> None:
        cls.omni_shared = select_ci_tracks.compute_omni_shared_closure(_REPO_ROOT)

    def test_skip_reason_length_guard(self) -> None:
        with self.assertRaises(ValueError):
            select_ci_tracks.TrackPlan(
                selection="core-only",
                default_lanes="tests",
                omni_image_skip="x" * 71,
            )

    def test_docs_only_skips_unless_override_label(self) -> None:
        plan = select_ci_tracks.decide_ci_plan(
            changed_files=["docs/README.md", ".pyrefly-baseline.json"],
            omni_shared_files=self.omni_shared,
            is_pull_request=True,
        )
        self.assertEqual(plan.selection, "skip-all")

        full_omni = select_ci_tracks.decide_ci_plan(
            changed_files=["docs/README.md"],
            omni_shared_files=self.omni_shared,
            labels={"ready", "run-omni-full-ci"},
            is_pull_request=True,
        )
        self.assertEqual(full_omni.selection, "omni-only")
        self.assertEqual(full_omni.omni_full_ci, "1")
        self.assertEqual(full_omni.default_lanes, "tests")

    def test_core_only_diff_skips_omni_image_and_tests(self) -> None:
        plan = select_ci_tracks.decide_ci_plan(
            changed_files=[
                "src/vllm_torchtpu/runner/tpu_runner.py",
                "tests/runner/test_tpu_runner.py",
            ],
            omni_shared_files=self.omni_shared,
            is_pull_request=True,
        )
        self.assertEqual(plan.selection, "core-only")
        self.assertEqual(plan.default_lanes, "verify integration tests")
        self.assertEqual(plan.core_image_skip, "")
        self.assertEqual(plan.core_tests_skip, "")
        self.assertTrue(plan.omni_image_skip)
        self.assertTrue(plan.omni_tests_skip)
        self.assertEqual(plan.annotation, "")

    def test_omni_only_diff_skips_core_image_and_core_lanes(self) -> None:
        plan = select_ci_tracks.decide_ci_plan(
            changed_files=[
                "src/vllm_torchtpu/omni/patches.py",
                "tests_omni/models/wan/test_wan_pr.py",
            ],
            omni_shared_files=self.omni_shared,
            is_pull_request=True,
        )
        self.assertEqual(plan.selection, "omni-only")
        self.assertEqual(plan.default_lanes, "tests")
        self.assertTrue(plan.core_image_skip)
        self.assertTrue(plan.core_tests_skip)
        self.assertEqual(plan.omni_image_skip, "")
        self.assertEqual(plan.omni_tests_skip, "")
        self.assertEqual(plan.annotation, "")

    def test_shared_or_hybrid_diff_runs_both_tracks(self) -> None:
        for files in (
            ["src/vllm_torchtpu/kernels/diffusion_attention/kernel.py"],
            ["src/vllm_torchtpu/platforms/tpu_platform.py"],
            ["docker/Dockerfile"],
            [
                "src/vllm_torchtpu/runner/tpu_runner.py",
                "tests_omni/test_omni_platform.py",
            ],
        ):
            plan = select_ci_tracks.decide_ci_plan(
                changed_files=files,
                omni_shared_files=self.omni_shared,
                is_pull_request=True,
            )
            self.assertEqual(plan.selection, "both", msg=f"Failed for {files}")
            self.assertEqual(plan.core_image_skip, "")
            self.assertEqual(plan.omni_image_skip, "")

    def test_integration_pipeline_runs_core_only_across_all_lanes(self) -> None:
        plan = select_ci_tracks.decide_ci_plan(
            changed_files=[],
            omni_shared_files=self.omni_shared,
            pipeline_slug="vllm-torchtpu-integration",
        )
        self.assertEqual(plan.selection, "core-only")
        self.assertEqual(plan.default_lanes, "verify perf integration tests")
        self.assertEqual(plan.core_image_skip, "")
        self.assertTrue(plan.omni_image_skip)
        self.assertTrue(plan.omni_tests_skip)

    def test_nightly_and_schedule_always_run_both_tracks(self) -> None:
        plan = select_ci_tracks.decide_ci_plan(
            changed_files=["tests_omni/test_omni_platform.py"],
            omni_shared_files=self.omni_shared,
            nightly=True,
            is_schedule=True,
        )
        self.assertEqual(plan.selection, "both")
        self.assertEqual(plan.default_lanes, "verify perf integration tests")
        self.assertEqual(plan.core_image_skip, "")
        self.assertEqual(plan.omni_image_skip, "")

    def test_skip_tpu_ci_overrides_run_omni_full_ci(self) -> None:
        plan = select_ci_tracks.decide_ci_plan(
            changed_files=["tests_omni/test_omni_platform.py"],
            omni_shared_files=self.omni_shared,
            labels={"ready", "run-omni-full-ci", "skip-tpu-ci"},
            is_pull_request=True,
        )
        self.assertEqual(plan.selection, "core-only")
        self.assertEqual(plan.skip_tpu_ci, "1")
        self.assertEqual(plan.default_lanes, "tests")
        self.assertEqual(plan.core_image_skip, "")
        self.assertTrue(plan.omni_image_skip)
        self.assertTrue(plan.omni_tests_skip)


class TestPipelineAndBootstrapSplit(unittest.TestCase):
    """Verify `bootstrap.sh`, `pipeline_build.yml`, and `pipeline_tests.yml`."""

    def _run_bootstrap_with_diff(
        self,
        diff_files: list[str],
        **extra_env: str,
    ) -> tuple[list[str], list[str], str]:
        fake_agent = """#!/usr/bin/env bash
set -euo pipefail
echo "$*" >> "${TEST_AGENT_LOG}/calls.log"
if [[ "$1" == "pipeline" && "$2" == "upload" ]]; then
  shopt -s nullglob
  ufiles=("${TEST_AGENT_LOG}"/upload_*.yml)
  env > "${TEST_AGENT_LOG}/env_${#ufiles[@]}.txt"
  cat > "${TEST_AGENT_LOG}/upload_${#ufiles[@]}.yml"
fi
"""
        diff_body = "\n".join(diff_files)
        fake_git = f"""#!/usr/bin/env bash
if [[ "${{1:-}}" == "fetch" ]]; then
  exit 0
fi
if [[ "${{1:-}}" == "diff" || "${{1:-}}" == "diff-tree" ]]; then
  cat <<'DIFF_EOF'
{diff_body}
DIFF_EOF
  exit 0
fi
exec /usr/bin/git "$@"
"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            bin_dir = tmp_path / "bin"
            bin_dir.mkdir()
            for name, content in (("buildkite-agent", fake_agent), ("git", fake_git)):
                script_path = bin_dir / name
                script_path.write_text(content, encoding="utf-8")
                script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)

            env = {
                "PATH": f"{bin_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
                "HOME": str(tmp_path),
                "TEST_AGENT_LOG": str(tmp_path),
                "BUILDKITE_COMMIT": "HEAD",
                "BUILDKITE_BRANCH": "feature-branch",
                "BUILDKITE_PULL_REQUEST": "false",
                "BUILDKITE_PULL_REQUEST_LABELS": "",
                "BUILDKITE_PIPELINE_SLUG": "vllm-torchtpu-ci",
                **extra_env,
            }
            proc = subprocess.run(
                ["bash", str(_BOOTSTRAP)],
                cwd=_REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            calls_path = tmp_path / "calls.log"
            calls = calls_path.read_text().splitlines() if calls_path.exists() else []
            uploads: list[str] = []
            for idx, up_path in enumerate(sorted(tmp_path.glob("upload_*.yml"))):
                env_map: dict[str, str] = {}
                for line in (tmp_path / f"env_{idx}.txt").read_text().splitlines():
                    if "=" in line:
                        k, v = line.split("=", 1)
                        env_map[k] = v
                uploads.append(
                    _interpolate_buildkite_vars(up_path.read_text(), env_map)
                )
            return calls, uploads, proc.stdout

    def test_omni_only_pr_uploads_only_tests_and_skips_all_build_docker_dependents(
        self,
    ) -> None:
        calls, uploads, stdout = self._run_bootstrap_with_diff(
            ["src/vllm_torchtpu/omni/patches.py", "tests_omni/test_omni_platform.py"],
            BUILDKITE_PULL_REQUEST="456",
            BUILDKITE_PULL_REQUEST_LABELS="ready",
        )
        # Only pipeline_tests.yml + pipeline_build.yml uploaded!
        self.assertEqual(len(uploads), 2)
        self.assertNotIn("Uploading .buildkite/pipeline_integration.yml", stdout)
        self.assertNotIn("Uploading .buildkite/pipeline_verify.yml", stdout)
        self.assertFalse(any("annotate" in c for c in calls))

        tests_doc = yaml.safe_load(uploads[0])
        build_doc = yaml.safe_load(uploads[1])
        build_steps = {s["key"]: s for s in _iter_command_steps(build_doc["steps"])}
        self.assertTrue(build_steps["build_docker"]["skip"])
        self.assertLessEqual(len(build_steps["build_docker"]["skip"]), 70)
        self.assertEqual(build_steps["build_docker_omni"]["skip"], "")

        for step in _iter_command_steps(tests_doc["steps"]):
            dep = step.get("depends_on")
            if dep == "build_docker":
                self.assertTrue(
                    step.get("skip"),
                    msg=(
                        f"Core step {step.get('key')} was not skipped when "
                        "build_docker was skipped!"
                    ),
                )
                self.assertLessEqual(len(step["skip"]), 70)
            elif dep == "build_docker_omni":
                self.assertEqual(step.get("skip"), "")

    def test_core_only_pr_skips_omni_build_and_all_omni_dependents(self) -> None:
        _, uploads, _ = self._run_bootstrap_with_diff(
            ["src/vllm_torchtpu/runner/tpu_runner.py"],
            BUILDKITE_PULL_REQUEST="457",
            BUILDKITE_PULL_REQUEST_LABELS="ready",
        )
        # verify, integration, tests + build = 4 uploads
        self.assertEqual(len(uploads), 4)
        tests_doc = yaml.safe_load(uploads[2])
        build_doc = yaml.safe_load(uploads[3])
        build_steps = {s["key"]: s for s in _iter_command_steps(build_doc["steps"])}
        self.assertEqual(build_steps["build_docker"]["skip"], "")
        self.assertTrue(build_steps["build_docker_omni"]["skip"])

        for step in _iter_command_steps(tests_doc["steps"]):
            dep = step.get("depends_on")
            if dep == "build_docker":
                self.assertEqual(step.get("skip"), "")
            elif dep == "build_docker_omni":
                self.assertTrue(
                    step.get("skip"),
                    msg=(
                        f"Omni step {step.get('key')} was not skipped when "
                        "build_docker_omni was skipped!"
                    ),
                )
                self.assertLessEqual(len(step["skip"]), 70)

    def test_run_omni_full_ci_enables_omni_nightly_steps_without_core_lanes(
        self,
    ) -> None:
        calls, uploads, _ = self._run_bootstrap_with_diff(
            ["tests_omni/models/wan/test_wan_multirank.py"],
            BUILDKITE_PULL_REQUEST="458",
            BUILDKITE_PULL_REQUEST_LABELS="ready,run-omni-full-ci",
        )
        self.assertEqual(len(uploads), 2)
        self.assertTrue(any("--context ci-track-selection" in c for c in calls))
        tests_doc = yaml.safe_load(uploads[0])
        by_key = {s["key"]: s for s in _iter_command_steps(tests_doc["steps"])}
        self.assertIn('"1" != "1"', by_key["unit_tests_omni_pr"]["if"])
        self.assertIn('"1" == "1"', by_key["unit_tests_omni_nightly"]["if"])
        self.assertIn('"1" == "1"', by_key["unit_tests_omni_multichip_nightly"]["if"])


class TestDockerfileSplit(unittest.TestCase):
    """Verify multi-stage independence and layer ordering in `docker/Dockerfile`."""

    def test_omni_branches_from_ci_and_installs_deps_before_copy_repo(self) -> None:
        dockerfile = (_REPO_ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
        from_re = re.compile(r"^FROM\s+(\S+)\s+AS\s+(\S+)", re.MULTILINE)
        parents = {stage: parent for parent, stage in from_re.findall(dockerfile)}

        for required in ("tpu-runtime", "base", "ci", "dev", "omni", "prod"):
            self.assertIn(required, parents)

        self.assertEqual(parents["base"], "tpu-runtime")
        self.assertEqual(parents["ci"], "base")
        self.assertEqual(parents["dev"], "ci")
        self.assertEqual(parents["omni"], "ci")
        self.assertEqual(parents["prod"], "base")

        # Verify dev and omni both install stage dependencies BEFORE `COPY . .`
        omni_start = dockerfile.index("FROM ci AS omni")
        prod_start = dockerfile.index("FROM base AS prod")
        omni_block = dockerfile[omni_start:prod_start]
        self.assertLess(
            omni_block.index("--extra omni"),
            omni_block.index("COPY . ."),
        )
        self.assertNotIn("--extra benchmarking", omni_block)


if __name__ == "__main__":
    unittest.main()
