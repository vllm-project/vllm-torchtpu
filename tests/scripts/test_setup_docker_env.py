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
"""Unit tests for `.buildkite/scripts/setup_docker_env.sh`."""

from __future__ import annotations

import json
import os
import pathlib
import stat
import subprocess
import sys
import tempfile
import unittest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / ".buildkite" / "scripts" / "setup_docker_env.sh"
_IMAGE_REPO = (
    "us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/"
    "vllm-torchtpu-ci/vllm-torchtpu"
)

_STUB_TEMPLATE = """#!{python}
import json
import os
import pathlib
import sys

log_path = pathlib.Path(os.environ["TEST_CALL_LOG"])
args = sys.argv[1:]
with log_path.open("a", encoding="utf-8") as fh:
    fh.write(json.dumps({{"cmd": {cmd!r}, "args": args}}) + "\\n")
{body}
"""

_AGENT_BODY = """
if args[:3] == ["meta-data", "get", "VLLM_COMMIT_HASH"]:
    print("vllmsha123")
elif args[:3] == ["meta-data", "get", "TORCHTPU_NIGHTLY_LEG"]:
    meta_leg = {meta_leg!r}
    if meta_leg:
        print(meta_leg)
"""

_DOCKER_BODY = """
# Simulate cleanup_docker.sh / manifest inspect / pull / build / push / rmi.
if args[:1] == ["manifest"]:
    sys.exit({manifest_rc})
sys.exit(0)
"""

_CURL_BODY = """
url = args[-1]
if url.endswith("/torch-tpu/"):
    print(
        '<a href="torch_tpu-0.2.0.dev20261122+abc-cp312-cp312-linux_x86_64.whl">'
        "torch_tpu-0.2.0.dev20261122+abc-cp312-cp312-linux_x86_64.whl</a>"
    )
elif url.endswith("/tpu-sync-torch/"):
    print(
        '<a href="tpu_sync_torch-0.0.1.dev20261122-cp312-cp312-linux_x86_64.whl">'
        "tpu_sync_torch-0.0.1.dev20261122-cp312-cp312-linux_x86_64.whl</a>"
    )
else:
    sys.exit(22)
"""


def _write_stub(bin_dir: pathlib.Path, cmd: str, body: str) -> None:
    path = bin_dir / cmd
    path.write_text(
        _STUB_TEMPLATE.format(python=sys.executable, cmd=cmd, body=body.lstrip()),
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _docker_calls(calls: list[dict[str, object]], subcmd: str) -> list[list[str]]:
    return [
        c["args"]
        for c in calls
        if c["cmd"] == "docker"
        and isinstance(c["args"], list)
        and c["args"][:1] == [subcmd]
    ]


def _meta_sets(calls: list[dict[str, object]]) -> list[tuple[str, str]]:
    return [
        (c["args"][2], c["args"][3])
        for c in calls
        if c["cmd"] == "buildkite-agent"
        and isinstance(c["args"], list)
        and c["args"][:2] == ["meta-data", "set"]
    ]


def _curl_urls(calls: list[dict[str, object]]) -> list[str]:
    return [
        str(c["args"][-1])
        for c in calls
        if c["cmd"] == "curl" and isinstance(c["args"], list)
    ]


class TestSetupDockerEnv(unittest.TestCase):
    """Verify non-canary reuse and single-leg canary overrides."""

    def _run(
        self,
        *,
        target: str = "dev",
        meta_leg: str = "",
        manifest_rc: int = 1,
        extra_env: dict[str, str] | None = None,
    ) -> tuple[subprocess.CompletedProcess[str], list[dict[str, object]]]:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            bin_dir = tmp_path / "bin"
            bin_dir.mkdir()
            _write_stub(
                bin_dir, "buildkite-agent", _AGENT_BODY.format(meta_leg=meta_leg)
            )
            _write_stub(bin_dir, "docker", _DOCKER_BODY.format(manifest_rc=manifest_rc))
            _write_stub(bin_dir, "curl", _CURL_BODY)

            log_path = tmp_path / "calls.jsonl"
            env = {
                "PATH": f"{bin_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
                "HOME": str(tmp_path),
                "TEST_CALL_LOG": str(log_path),
                "BUILDKITE_COMMIT": "deadbeef",
                "ACCESS_TOKEN": "fake-token",
                **(extra_env or {}),
            }
            proc = subprocess.run(
                ["bash", str(_SCRIPT), target],
                cwd=_REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            calls = (
                [
                    json.loads(line)
                    for line in log_path.read_text(encoding="utf-8").splitlines()
                ]
                if log_path.exists()
                else []
            )
            return proc, calls

    def test_non_canary_reuses_registry_image_without_building(self) -> None:
        for target, expected_key, expected_tag in (
            ("dev", "CI_IMAGE_TAG", f"{_IMAGE_REPO}:deadbeef-vllmsha123"),
            ("omni", "CI_IMAGE_TAG-omni", f"{_IMAGE_REPO}:deadbeef-vllmsha123-omni"),
        ):
            with self.subTest(target=target):
                proc, calls = self._run(target=target, meta_leg="", manifest_rc=0)
                self.assertEqual(proc.returncode, 0, msg=proc.stderr)
                self.assertEqual(
                    _docker_calls(calls, "manifest"),
                    [["manifest", "inspect", expected_tag]],
                )
                self.assertEqual(_docker_calls(calls, "build"), [])
                self.assertEqual(_docker_calls(calls, "push"), [])
                self.assertEqual(_curl_urls(calls), [])
                self.assertEqual(_meta_sets(calls), [(expected_key, expected_tag)])

    def test_single_leg_selection_override_and_metadata(self) -> None:
        cases = [
            # Standard canary pipeline: bootstrap meta-data selects leg.
            {
                "name": "meta_torch_tpu_dev_index",
                "target": "dev",
                "meta_leg": "torch-tpu",
                "env": {},
                "pkg": "torch-tpu",
                "ver": "0.2.0.dev20261122+abc",
                "curl_suffix": "/torch-tpu/",
                "pushes": [
                    f"{_IMAGE_REPO}:deadbeef-vllmsha123-torch-tpu_0.2.0.dev20261122_abc"
                ],
                "meta_key": "CI_IMAGE_TAG",
                "ver_key": "TORCH_TPU_VERSION",
            },
            # tpu-sync without TPU_SYNC_TORCH_VERSION_OVERRIDE queries the
            # torch-tpu Artifact Registry /tpu-sync-torch/ index.
            {
                "name": "meta_tpu_sync_dev_index",
                "target": "dev",
                "meta_leg": "tpu-sync",
                "env": {},
                "pkg": "tpu-sync-torch",
                "ver": "0.0.1.dev20261122",
                "curl_suffix": "/tpu-sync-torch/",
                "pushes": [
                    f"{_IMAGE_REPO}:deadbeef-vllmsha123-tpu-sync_0.0.1.dev20261122",
                    f"{_IMAGE_REPO}:deadbeef-vllmsha123-tpu-sync-torch_0.0.1.dev20261122",
                ],
                "meta_key": "CI_IMAGE_TAG",
                "ver_key": "TPU_SYNC_TORCH_VERSION",
            },
            # Explicit version override skips querying the Artifact Registry index.
            {
                "name": "meta_tpu_sync_with_override",
                "target": "dev",
                "meta_leg": "tpu-sync",
                "env": {"TPU_SYNC_TORCH_VERSION_OVERRIDE": "0.0.1.dev20261123"},
                "pkg": "tpu-sync-torch",
                "ver": "0.0.1.dev20261123",
                "curl_suffix": None,
                "pushes": [
                    f"{_IMAGE_REPO}:deadbeef-vllmsha123-tpu-sync_0.0.1.dev20261123",
                    f"{_IMAGE_REPO}:deadbeef-vllmsha123-tpu-sync-torch_0.0.1.dev20261123",
                ],
                "meta_key": "CI_IMAGE_TAG",
                "ver_key": "TPU_SYNC_TORCH_VERSION",
            },
            {
                "name": "meta_torch_tpu_with_override",
                "target": "dev",
                "meta_leg": "torch-tpu",
                "env": {"TORCH_TPU_VERSION_OVERRIDE": "0.2.0.dev9"},
                "pkg": "torch-tpu",
                "ver": "0.2.0.dev9",
                "curl_suffix": None,
                "pushes": [f"{_IMAGE_REPO}:deadbeef-vllmsha123-torch-tpu_0.2.0.dev9"],
                "meta_key": "CI_IMAGE_TAG",
                "ver_key": "TORCH_TPU_VERSION",
            },
            # Canary builds always build the delta image even when the base tag
            # is already in the registry.
            {
                "name": "canary_ignores_manifest_reuse",
                "target": "dev",
                "meta_leg": "torch-tpu",
                "manifest_rc": 0,
                "env": {"TORCH_TPU_VERSION_OVERRIDE": "0.2.0.dev9"},
                "pkg": "torch-tpu",
                "ver": "0.2.0.dev9",
                "curl_suffix": None,
                "pushes": [f"{_IMAGE_REPO}:deadbeef-vllmsha123-torch-tpu_0.2.0.dev9"],
                "meta_key": "CI_IMAGE_TAG",
                "ver_key": "TORCH_TPU_VERSION",
            },
            # Omni target publishes CI_IMAGE_TAG-omni without overwriting
            # CI_IMAGE_TAG for both legs.
            {
                "name": "omni_torch_tpu",
                "target": "omni",
                "meta_leg": "torch-tpu",
                "env": {"TORCH_TPU_VERSION_OVERRIDE": "0.2.0.dev1"},
                "pkg": "torch-tpu",
                "ver": "0.2.0.dev1",
                "curl_suffix": None,
                "pushes": [
                    f"{_IMAGE_REPO}:deadbeef-vllmsha123-omni-torch-tpu_0.2.0.dev1"
                ],
                "meta_key": "CI_IMAGE_TAG-omni",
                "ver_key": "TORCH_TPU_VERSION",
            },
            {
                "name": "omni_tpu_sync",
                "target": "omni",
                "meta_leg": "tpu-sync",
                "env": {"TPU_SYNC_TORCH_VERSION_OVERRIDE": "0.0.1.dev2"},
                "pkg": "tpu-sync-torch",
                "ver": "0.0.1.dev2",
                "curl_suffix": None,
                "pushes": [
                    f"{_IMAGE_REPO}:deadbeef-vllmsha123-omni-tpu-sync_0.0.1.dev2",
                    f"{_IMAGE_REPO}:deadbeef-vllmsha123-omni-tpu-sync-torch_0.0.1.dev2",
                ],
                "meta_key": "CI_IMAGE_TAG-omni",
                "ver_key": "TPU_SYNC_TORCH_VERSION",
            },
        ]

        for case in cases:
            with self.subTest(case=case["name"]):
                proc, calls = self._run(
                    target=str(case["target"]),
                    meta_leg=str(case["meta_leg"]),
                    manifest_rc=int(case.get("manifest_rc", 1)),
                    extra_env=dict(case["env"]),
                )
                self.assertEqual(proc.returncode, 0, msg=proc.stderr)

                urls = _curl_urls(calls)
                if case["curl_suffix"] is None:
                    self.assertEqual(urls, [])
                else:
                    self.assertEqual(len(urls), 1)
                    self.assertTrue(urls[0].endswith(str(case["curl_suffix"])))

                builds = _docker_calls(calls, "build")
                self.assertEqual(len(builds), 1)
                self.assertIn(f"OVERRIDE_PACKAGE={case['pkg']}", builds[0])
                self.assertIn(f"OVERRIDE_VERSION={case['ver']}", builds[0])

                pushes = [args[1] for args in _docker_calls(calls, "push")]
                self.assertEqual(pushes, case["pushes"])

                self.assertEqual(
                    _meta_sets(calls),
                    [
                        (case["meta_key"], case["pushes"][0]),
                        (case["ver_key"], case["ver"]),
                    ],
                )

    def test_invalid_leg_and_mismatched_version_overrides_fail_fast(self) -> None:
        cases = [
            (
                "invalid_meta_leg",
                "bogus",
                {},
                "Unsupported TORCHTPU_NIGHTLY_LEG 'bogus'",
            ),
            (
                "torch_tpu_with_tpu_sync_override",
                "torch-tpu",
                {"TPU_SYNC_TORCH_VERSION_OVERRIDE": "0.0.1.dev1"},
                "TPU_SYNC_TORCH_VERSION_OVERRIDE is set",
            ),
            (
                "tpu_sync_with_torch_tpu_override",
                "tpu-sync",
                {"TORCH_TPU_VERSION_OVERRIDE": "0.2.0.dev1"},
                "TORCH_TPU_VERSION_OVERRIDE is set",
            ),
            (
                "both_overrides_with_leg",
                "torch-tpu",
                {
                    "TORCH_TPU_VERSION_OVERRIDE": "0.2.0.dev1",
                    "TPU_SYNC_TORCH_VERSION_OVERRIDE": "0.0.1.dev1",
                },
                "set only one version override per build",
            ),
            (
                "override_without_meta_leg",
                "",
                {"TORCH_TPU_VERSION_OVERRIDE": "0.2.0.dev1"},
                "without TORCHTPU_NIGHTLY_LEG metadata",
            ),
        ]
        for label, meta_leg, extra_env, err_substr in cases:
            with self.subTest(case=label):
                proc, calls = self._run(
                    meta_leg=meta_leg, manifest_rc=0, extra_env=extra_env
                )
                self.assertEqual(proc.returncode, 1)
                self.assertIn(err_substr, proc.stderr)
                self.assertEqual(_docker_calls(calls, "build"), [])
                self.assertEqual(_meta_sets(calls), [])


if __name__ == "__main__":
    unittest.main()
