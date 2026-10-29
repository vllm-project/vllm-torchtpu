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
"""Unit tests for soft_fail handling in support matrix scripts."""

import csv
import importlib.util
import json
import os
import pathlib
import stat
import subprocess
import tempfile
import unittest
from unittest import mock

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_RECORD_SCRIPT = _REPO_ROOT / ".buildkite" / "scripts" / "record_step_result.sh"
_MATRIX_SCRIPT = _REPO_ROOT / ".buildkite" / "scripts" / "generate_support_matrices.py"

_spec = importlib.util.spec_from_file_location(
    "generate_support_matrices", _MATRIX_SCRIPT
)
assert _spec is not None and _spec.loader is not None
generate_support_matrices = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(generate_support_matrices)

_FAKE_BK_AGENT = """#!/usr/bin/env python3
import json
import os
import pathlib
import sys

state_file = pathlib.Path(os.environ["FAKE_BK_STATE"])
state = (
    json.loads(state_file.read_text())
    if state_file.exists()
    else {"meta": {}, "outcomes": {}}
)
args = sys.argv[1:]

if args[:2] == ["meta-data", "get"]:
    key = args[2]
    default = args[4] if len(args) >= 5 and args[3] == "--default" else ""
    print(state["meta"].get(key, default))
elif args[:2] == ["meta-data", "set"]:
    key = args[2]
    val = args[3] if len(args) >= 4 else sys.stdin.read().rstrip("\\n")
    state["meta"][key] = val
    state_file.write_text(json.dumps(state))
elif args[:3] == ["step", "get", "outcome"]:
    step_key = args[4]
    if step_key not in state["outcomes"]:
        sys.exit(1)
    print(state["outcomes"][step_key])
elif args[:2] == ["artifact", "upload"]:
    state.setdefault("artifacts", []).append(args[2])
    state_file.write_text(json.dumps(state))
else:
    sys.exit(f"unexpected args: {args}")
"""


class TestSoftFailSupportMatrix(unittest.TestCase):
    """Verify soft_fail steps record failing cells without failing the build."""

    def _make_fake_agent(self, tmpdir: str, outcomes: dict[str, str]) -> pathlib.Path:
        bin_dir = pathlib.Path(tmpdir) / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        agent_path = bin_dir / "buildkite-agent"
        agent_path.write_text(_FAKE_BK_AGENT, encoding="utf-8")
        agent_path.chmod(agent_path.stat().st_mode | stat.S_IXUSR)
        state_path = pathlib.Path(tmpdir) / "bk_state.json"
        state_path.write_text(
            json.dumps({"meta": {}, "outcomes": outcomes}), encoding="utf-8"
        )
        return state_path

    def test_record_step_result_soft_failed_exits_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = self._make_fake_agent(
                tmp, {"tpu7x_Qwen_Accuracy": "soft_failed"}
            )
            env = {
                **os.environ,
                "PATH": f"{tmp}/bin:{os.environ.get('PATH', '')}",
                "FAKE_BK_STATE": str(state_path),
                "CI_TPU_VERSION": "tpu7x",
                "CI_TARGET": "Qwen/Qwen3.5-397B-A17B",
                "CI_STAGE": "Accuracy/Correctness",
                "CI_CATEGORY": "text-only",
            }
            proc = subprocess.run(
                ["sh", str(_RECORD_SCRIPT), "tpu7x_Qwen_Accuracy"],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr + proc.stdout)
            meta = json.loads(state_path.read_text(encoding="utf-8"))["meta"]
            self.assertEqual(
                meta["v7Qwen/Qwen3.5-397B-A17B:Accuracy/Correctness"], "❌ Failing"
            )
            self.assertEqual(
                meta["v7Qwen/Qwen3.5-397B-A17B:Accuracy/Correctness:soft_failed"],
                "true",
            )

    def test_record_step_result_hard_failed_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = self._make_fake_agent(tmp, {"tpu7x_Qwen_Accuracy": "failed"})
            env = {
                **os.environ,
                "PATH": f"{tmp}/bin:{os.environ.get('PATH', '')}",
                "FAKE_BK_STATE": str(state_path),
                "CI_TPU_VERSION": "tpu7x",
                "CI_TARGET": "Qwen/Qwen3.5-397B-A17B",
                "CI_STAGE": "Accuracy/Correctness",
                "CI_CATEGORY": "text-only",
            }
            proc = subprocess.run(
                ["sh", str(_RECORD_SCRIPT), "tpu7x_Qwen_Accuracy"],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 1)
            meta = json.loads(state_path.read_text(encoding="utf-8"))["meta"]
            self.assertEqual(
                meta["v7Qwen/Qwen3.5-397B-A17B:Accuracy/Correctness"], "❌ Failing"
            )
            self.assertEqual(
                meta["v7Qwen/Qwen3.5-397B-A17B:Accuracy/Correctness:soft_failed"],
                "false",
            )

    def test_generate_support_matrices_soft_fail_vs_hard_fail(self) -> None:
        meta_store = {
            "model-list": "Qwen/Qwen3.5-397B-A17B",
            "v7Qwen/Qwen3.5-397B-A17B_category": "text-only",
            "v7Qwen/Qwen3.5-397B-A17B:UnitTest": "✅ Passing",
            "v7Qwen/Qwen3.5-397B-A17B:Accuracy/Correctness": "❌ Failing",
            "v7Qwen/Qwen3.5-397B-A17B:Accuracy/Correctness:soft_failed": "true",
            "v7Qwen/Qwen3.5-397B-A17B:Benchmark": "✅ Passing",
        }
        captured_csv: list[dict[str, str]] = []

        def fake_get(key: str, default: str = "") -> str:
            return meta_store.get(key, default)

        def fake_set(key: str, value: str) -> None:
            meta_store[key] = value

        def fake_upload(file_path: str) -> None:
            if file_path.endswith("model_support_matrix.csv"):
                with open(file_path, encoding="utf-8") as f:
                    captured_csv.extend(csv.DictReader(f))

        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.dict(os.environ, {"TPU_VERSION": "v7x"}, clear=False),
            mock.patch.object(
                generate_support_matrices,
                "get_buildkite_metadata",
                side_effect=fake_get,
            ),
            mock.patch.object(
                generate_support_matrices,
                "set_buildkite_metadata",
                side_effect=fake_set,
            ),
            mock.patch.object(
                generate_support_matrices,
                "upload_buildkite_artifact",
                side_effect=fake_upload,
            ),
        ):
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                generate_support_matrices.main()
            finally:
                os.chdir(old_cwd)

            self.assertEqual(meta_store["v7_CI_TESTS_FAILED"], "false")
            self.assertEqual(len(captured_csv), 1)
            self.assertEqual(
                captured_csv[0]["Accuracy/Correctness"],
                "❌ Failing",
            )

            # A hard failure must still fail the report.
            meta_store["v7Qwen/Qwen3.5-397B-A17B:Accuracy/Correctness:soft_failed"] = (
                "false"
            )
            os.chdir(tmp)
            try:
                generate_support_matrices.main()
            finally:
                os.chdir(old_cwd)
            self.assertEqual(meta_store["v7_CI_TESTS_FAILED"], "true")


if __name__ == "__main__":
    unittest.main()
