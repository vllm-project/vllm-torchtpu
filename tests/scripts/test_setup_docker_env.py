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
"""Unit tests for .buildkite/scripts/setup_docker_env.sh and docker/build_image.sh."""

from __future__ import annotations

import os
import pathlib
import shutil
import stat
import subprocess
import tempfile
import unittest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_IMAGE_REPO = (
    "us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/"
    "vllm-torchtpu-ci/vllm-torchtpu"
)
_VLLM_SHA = "v0.32.0"

_FAKE_AGENT = """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "meta-data" && "${2:-}" == "get" ]]; then
  printf '%s' "${TEST_VLLM_COMMIT_HASH:-}"
  exit 0
fi
if [[ "${1:-}" == "meta-data" && "${2:-}" == "set" ]]; then
  printf '%s=%s\\n' "$3" "$4" >> "${TEST_LOG_DIR}/metadata.log"
  exit 0
fi
exit 0
"""

_FAKE_DOCKER = """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "${TEST_LOG_DIR}/docker.log"
if [[ "${1:-}" == "manifest" && "${2:-}" == "inspect" ]]; then
  tag="${3:-}"
  if grep -Fqx "${tag}" "${TEST_LOG_DIR}/existing_images.txt" 2>/dev/null; then
    exit 0
  fi
  exit 1
fi
exit 0
"""


def _git(cwd: pathlib.Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=cwd,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
        },
        text=True,
    ).strip()


class SetupDockerEnvTest(unittest.TestCase):
    """Tests parent-image --cache-from lookup and inline cache export."""

    def _run_setup(
        self,
        *,
        target: str = "dev",
        existing_images: list[str] | None = None,
        advance_origin_main: bool = False,
        on_pr_branch: bool = False,
        env: dict[str, str] | None = None,
    ) -> tuple[str, list[str], dict[str, str], dict[str, str]]:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            origin_dir = tmp_path / "origin.git"
            work_dir = tmp_path / "work"
            log_dir = tmp_path / "logs"
            bin_dir = tmp_path / "bin"
            origin_dir.mkdir()
            log_dir.mkdir()
            bin_dir.mkdir()

            for name, content in (
                ("buildkite-agent", _FAKE_AGENT),
                ("docker", _FAKE_DOCKER),
            ):
                script = bin_dir / name
                script.write_text(content)
                script.chmod(script.stat().st_mode | stat.S_IEXEC)

            (log_dir / "existing_images.txt").write_text(
                "\n".join(existing_images or ()) + "\n"
            )

            # Build a tiny git history with a bare origin remote and real scripts.
            _git(origin_dir, "init", "--bare", "-b", "main")
            _git(tmp_path, "clone", "--quiet", str(origin_dir), str(work_dir))

            (work_dir / ".buildkite" / "scripts").mkdir(parents=True)
            (work_dir / "docker").mkdir(parents=True)
            for rel in (
                ".buildkite/scripts/setup_docker_env.sh",
                ".buildkite/scripts/cleanup_docker.sh",
                "docker/build_image.sh",
                "docker/Dockerfile",
                "pyproject.toml",
            ):
                shutil.copy2(_REPO_ROOT / rel, work_dir / rel)

            _git(work_dir, "add", ".")
            _git(work_dir, "commit", "-m", "main-1")
            main1 = _git(work_dir, "rev-parse", "HEAD")
            _git(work_dir, "push", "--quiet", "origin", "main")

            (work_dir / "README_TEST.txt").write_text("main-2\n")
            _git(work_dir, "add", "README_TEST.txt")
            _git(work_dir, "commit", "-m", "main-2")
            main2 = _git(work_dir, "rev-parse", "HEAD")
            _git(work_dir, "push", "--quiet", "origin", "main")

            shas = {"main1": main1, "main2": main2}

            if on_pr_branch:
                _git(work_dir, "checkout", "-q", "-b", "pr-branch", main2)
                (work_dir / "PR_CHANGE.txt").write_text("pr\n")
                _git(work_dir, "add", "PR_CHANGE.txt")
                _git(work_dir, "commit", "-m", "pr-commit")
                shas["head"] = _git(work_dir, "rev-parse", "HEAD")
                if advance_origin_main:
                    # Simulate Buildkite PR checkout where local origin/main
                    # is stale until setup_docker_env.sh fetches origin/main.
                    _git(
                        work_dir,
                        "update-ref",
                        "refs/remotes/origin/main",
                        main1,
                    )
            else:
                shas["head"] = main2

            # Expand `{main1}`, `{main2}`, and `{head}` placeholders.
            resolved_existing = [s.format(**shas) for s in (existing_images or ())]
            (log_dir / "existing_images.txt").write_text(
                "\n".join(resolved_existing) + "\n"
            )

            resolved_env = {k: v.format(**shas) for k, v in (env or {}).items()}
            run_env = {
                "PATH": f"{bin_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
                "HOME": str(tmp_path),
                "TEST_LOG_DIR": str(log_dir),
                "TEST_VLLM_COMMIT_HASH": _VLLM_SHA,
                "BUILDKITE_COMMIT": shas["head"],
                "BUILDKITE_BRANCH": "feature-branch",
                "BUILDKITE_PULL_REQUEST": "false",
                **resolved_env,
            }
            proc = subprocess.run(
                [
                    "bash",
                    str(work_dir / ".buildkite" / "scripts" / "setup_docker_env.sh"),
                    target,
                ],
                cwd=work_dir,
                env=run_env,
                capture_output=True,
                text=True,
                check=True,
            )

            docker_calls = (
                (log_dir / "docker.log").read_text().splitlines()
                if (log_dir / "docker.log").exists()
                else []
            )
            metadata: dict[str, str] = {}
            if (log_dir / "metadata.log").exists():
                for line in (log_dir / "metadata.log").read_text().splitlines():
                    key, _, val = line.partition("=")
                    metadata[key] = val
            return proc.stdout, docker_calls, metadata, shas

    def test_merge_queue_uses_base_commit_cache_from(self) -> None:
        parent_img = f"{_IMAGE_REPO}:{{main2}}-{_VLLM_SHA}"
        stdout, docker_calls, metadata, shas = self._run_setup(
            on_pr_branch=True,
            existing_images=[parent_img],
            env={"BUILDKITE_MERGE_QUEUE_BASE_COMMIT": "{main2}"},
        )
        expected_cache = f"{_IMAGE_REPO}:{shas['main2']}-{_VLLM_SHA}"
        expected_tag = f"{_IMAGE_REPO}:{shas['head']}-{_VLLM_SHA}"
        self.assertIn(f"cache-from {expected_cache}: found", stdout)
        build_calls = [c for c in docker_calls if c.startswith("build ")]
        self.assertEqual(len(build_calls), 1)
        self.assertIn("--build-arg BUILDKIT_INLINE_CACHE=1", build_calls[0])
        self.assertIn(f"--cache-from {expected_cache}", build_calls[0])
        self.assertEqual(metadata["CI_IMAGE_TAG"], expected_tag)

    def test_pr_build_fetches_origin_main_and_uses_merge_base(self) -> None:
        parent_img = f"{_IMAGE_REPO}:{{main2}}-{_VLLM_SHA}"
        stdout, docker_calls, _, shas = self._run_setup(
            on_pr_branch=True,
            advance_origin_main=True,
            existing_images=[parent_img],
            env={
                "BUILDKITE_PULL_REQUEST": "2560",
                "BUILDKITE_BRANCH": "pr-branch",
            },
        )
        expected_cache = f"{_IMAGE_REPO}:{shas['main2']}-{_VLLM_SHA}"
        self.assertIn(f"cache-from {expected_cache}: found", stdout)
        build_calls = [c for c in docker_calls if c.startswith("build ")]
        self.assertEqual(len(build_calls), 1)
        self.assertIn(f"--cache-from {expected_cache}", build_calls[0])

    def test_post_merge_main_uses_first_parent(self) -> None:
        parent_img = f"{_IMAGE_REPO}:{{main1}}-{_VLLM_SHA}"
        stdout, docker_calls, _, shas = self._run_setup(
            existing_images=[parent_img],
            env={"BUILDKITE_BRANCH": "main", "BUILDKITE_PULL_REQUEST": "false"},
        )
        expected_cache = f"{_IMAGE_REPO}:{shas['main1']}-{_VLLM_SHA}"
        self.assertIn(f"cache-from {expected_cache}: found", stdout)
        build_calls = [c for c in docker_calls if c.startswith("build ")]
        self.assertEqual(len(build_calls), 1)
        self.assertIn(f"--cache-from {expected_cache}", build_calls[0])

    def test_missing_parent_image_builds_normally(self) -> None:
        stdout, docker_calls, _, shas = self._run_setup(
            existing_images=[],
            env={"BUILDKITE_BRANCH": "main", "BUILDKITE_PULL_REQUEST": "false"},
        )
        expected_cache = f"{_IMAGE_REPO}:{shas['main1']}-{_VLLM_SHA}"
        self.assertIn(f"cache-from {expected_cache}: missing", stdout)
        build_calls = [c for c in docker_calls if c.startswith("build ")]
        self.assertEqual(len(build_calls), 1)
        self.assertIn("--build-arg BUILDKIT_INLINE_CACHE=1", build_calls[0])
        self.assertNotIn("--cache-from", build_calls[0])

    def test_omni_target_uses_parent_main_image_and_sets_omni_metadata(self) -> None:
        parent_img = f"{_IMAGE_REPO}:{{main2}}-{_VLLM_SHA}"
        stdout, docker_calls, metadata, shas = self._run_setup(
            target="omni",
            on_pr_branch=True,
            existing_images=[parent_img],
            env={"BUILDKITE_MERGE_QUEUE_BASE_COMMIT": "{main2}"},
        )
        expected_cache = f"{_IMAGE_REPO}:{shas['main2']}-{_VLLM_SHA}"
        expected_tag = f"{_IMAGE_REPO}:{shas['head']}-{_VLLM_SHA}-omni"
        self.assertIn(f"cache-from {expected_cache}: found", stdout)
        build_calls = [c for c in docker_calls if c.startswith("build ")]
        self.assertEqual(len(build_calls), 1)
        self.assertIn("--target omni", build_calls[0])
        self.assertIn(f"--cache-from {expected_cache}", build_calls[0])
        self.assertEqual(metadata["CI_IMAGE_TAG-omni"], expected_tag)

    def test_unmerged_non_pr_branch_does_not_use_branch_cache(self) -> None:
        stdout, docker_calls, _, _ = self._run_setup(
            on_pr_branch=True,
            existing_images=[f"{_IMAGE_REPO}:{{main2}}-{_VLLM_SHA}"],
            env={
                "BUILDKITE_BRANCH": "random-branch",
                "BUILDKITE_PULL_REQUEST": "false",
            },
        )
        self.assertNotIn("cache-from", stdout)
        build_calls = [c for c in docker_calls if c.startswith("build ")]
        self.assertEqual(len(build_calls), 1)
        self.assertNotIn("--cache-from", build_calls[0])

    def test_special_paths_preserved(self) -> None:
        # 1) Existing per-commit image is reused without rebuilding.
        _, docker_calls, metadata, shas = self._run_setup(
            existing_images=[f"{_IMAGE_REPO}:{{main2}}-{_VLLM_SHA}"],
            env={"BUILDKITE_BRANCH": "main", "BUILDKITE_PULL_REQUEST": "false"},
        )
        self.assertEqual([c for c in docker_calls if c.startswith("build ")], [])
        self.assertEqual(
            metadata["CI_IMAGE_TAG"], f"{_IMAGE_REPO}:{shas['main2']}-{_VLLM_SHA}"
        )

        # 2) Explicit CI_IMAGE skips docker build completely.
        explicit = f"{_IMAGE_REPO}:custom-tag"
        _, docker_calls, metadata, _ = self._run_setup(
            existing_images=[explicit],
            env={"CI_IMAGE": explicit},
        )
        self.assertEqual([c for c in docker_calls if c.startswith("build ")], [])
        self.assertEqual(metadata["CI_IMAGE_TAG"], explicit)

        # 2) CI_IMAGE_REBUILD=1 bypasses both per-commit reuse and --cache-from.
        _, docker_calls, _, _ = self._run_setup(
            existing_images=[
                f"{_IMAGE_REPO}:{{main2}}-{_VLLM_SHA}",
                f"{_IMAGE_REPO}:{{main1}}-{_VLLM_SHA}",
            ],
            env={
                "BUILDKITE_BRANCH": "main",
                "BUILDKITE_PULL_REQUEST": "false",
                "CI_IMAGE_REBUILD": "1",
            },
        )
        build_calls = [c for c in docker_calls if c.startswith("build ")]
        self.assertEqual(len(build_calls), 1)
        self.assertNotIn("--cache-from", build_calls[0])

        # 3) TORCHTPU_NIGHTLY=1 keeps its Dockerfile.override delta build path.
        stdout, docker_calls, metadata, _ = self._run_setup(
            env={
                "BUILDKITE_BRANCH": "main",
                "BUILDKITE_PULL_REQUEST": "false",
                "TORCHTPU_NIGHTLY": "1",
                "ACCESS_TOKEN": "stub-token",
                "TORCH_TPU_VERSION_OVERRIDE": "0.1.0+nightly",
                "TPU_SYNC_TORCH_VERSION_OVERRIDE": "0.2.0+nightly",
            },
        )
        self.assertNotIn("cache-from", stdout)
        override_calls = [c for c in docker_calls if "Dockerfile.override" in c]
        self.assertEqual(len(override_calls), 2)
        self.assertIn("TORCH_TPU_VERSION", metadata)


if __name__ == "__main__":
    unittest.main()
