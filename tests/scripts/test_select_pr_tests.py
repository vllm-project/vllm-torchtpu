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
"""Unit tests for .buildkite/scripts/select_pr_tests.py."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / ".buildkite" / "scripts" / "select_pr_tests.py"

spec = importlib.util.spec_from_file_location("select_pr_tests", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
select_pr_tests = importlib.util.module_from_spec(spec)
sys.modules["select_pr_tests"] = select_pr_tests
spec.loader.exec_module(select_pr_tests)


_SAMPLE_TESTS_YAML = """\
steps:
  - group: "Unit Tests on TPU"
    steps:
      - label: "Unit Tests - General (PR)"
        key: "unit_tests_general_pr"
        <<: *tpu_step
        if: 'build.env("NIGHTLY") != "1"'
        parallelism: 3
        env:
          SHAPE: *shape_single
        plugins:
          - *kube_plugin
          - tests#v1.0.0:
              <<: *tests_plugin_defaults
              test-cmd: "pytest -sv"

      - label: "Unit Tests - Speculative Decoding (PR)"
        key: "unit_tests_spec_decode_pr"
        <<: *tpu_step
        if: 'build.env("NIGHTLY") != "1"'
        parallelism: 3
"""


class SelectPrTestsTest(unittest.TestCase):
    """Tests diff-aware PR test selection and YAML step filtering."""

    def test_skip_all_ci_for_non_pr_files(self) -> None:
        plan = select_pr_tests.select_for_files(
            [
                "docs/README.md",
                ".github/workflows/pre-commit.yml",
                "support_matrices/nightly.csv",
                ".buildkite/features/moe.yml",
                ".buildkite/pipeline_dynamic.yml",
                "scripts/vllm/benchmarking/configs/qwen3.5-35b-fp8-tp4-ep.sh",
                "scripts/vllm/integration/run_dsv4_offload_correctness.sh",
            ]
        )
        self.assertTrue(plan.skip_all_ci)
        self.assertEqual(plan.lanes, ())

    def test_cpu_only_tooling_skips_integration_and_tpu_unit_tests(self) -> None:
        plan = select_pr_tests.select_for_files(
            [
                "tools/check_env_reads.py",
                "tests/scripts/test_check_regression.py",
                "scripts/vllm/benchmarking/check_regression.py",
            ]
        )
        self.assertFalse(plan.skip_all_ci)
        self.assertEqual(plan.lanes, ("verify", "tests"))
        self.assertTrue(plan.omni_image_skip)
        self.assertNotIn("typecheck_full", plan.skip_steps)
        self.assertNotIn("unit_tests_cpu_pr", plan.skip_steps)
        self.assertIn("unit_tests_general_pr", plan.skip_steps)
        self.assertIn("unit_tests_multichip_pr", plan.skip_steps)
        self.assertIn("unit_tests_spec_decode_pr", plan.skip_steps)
        self.assertIn("unit_tests_spec_decode_multichip_pr", plan.skip_steps)
        self.assertIn("unit_tests_omni_pr", plan.skip_steps)

    def test_omni_only_runs_omni_and_cpu_tests(self) -> None:
        plan = select_pr_tests.select_for_files(
            [
                "src/vllm_torchtpu/omni/runner.py",
                "tests_omni/test_omni_smoke.py",
            ]
        )
        self.assertFalse(plan.skip_all_ci)
        self.assertEqual(plan.lanes, ("verify", "tests"))
        self.assertEqual(plan.omni_image_skip, "")
        self.assertNotIn("unit_tests_omni_pr", plan.skip_steps)
        self.assertIn("unit_tests_general_pr", plan.skip_steps)
        self.assertIn("unit_tests_multichip_pr", plan.skip_steps)
        self.assertIn("unit_tests_spec_decode_pr", plan.skip_steps)

    def test_integration_script_only_runs_matching_step(self) -> None:
        plan = select_pr_tests.select_for_files(
            ["scripts/vllm/integration/run_qwen35_apc_correctness.sh"]
        )
        self.assertFalse(plan.skip_all_ci)
        self.assertEqual(plan.lanes, ("verify", "integration"))
        self.assertTrue(plan.omni_image_skip)
        self.assertNotIn("integration_qwen35_apc_correctness", plan.skip_steps)
        self.assertIn(
            "integration_k3_pruned75_apc_offload_correctness",
            plan.skip_steps,
        )
        self.assertIn("vllm_torchtpu_multihost_presubmit", plan.skip_steps)

    def test_spec_decode_unit_test_only(self) -> None:
        plan = select_pr_tests.select_for_files(
            ["tests/entrypoints/llm/test_speculative_decoding.py"]
        )
        self.assertFalse(plan.skip_all_ci)
        self.assertEqual(plan.lanes, ("verify", "tests"))
        self.assertNotIn("unit_tests_spec_decode_pr", plan.skip_steps)
        self.assertNotIn("unit_tests_spec_decode_multichip_pr", plan.skip_steps)
        self.assertIn("unit_tests_general_pr", plan.skip_steps)
        self.assertIn("unit_tests_multichip_pr", plan.skip_steps)
        self.assertIn("unit_tests_omni_pr", plan.skip_steps)

    def test_single_general_unit_test_narrows_pattern_and_parallelism(
        self,
    ) -> None:
        test_file = "tests/entrypoints/llm/test_structured_output.py"
        plan = select_pr_tests.select_for_files([test_file])
        self.assertFalse(plan.skip_all_ci)
        self.assertEqual(plan.lanes, ("verify", "tests"))
        self.assertEqual(plan.general_test_pattern, test_file)
        self.assertEqual(plan.general_test_parallelism, 1)
        self.assertNotIn("unit_tests_general_pr", plan.skip_steps)
        self.assertIn("unit_tests_multichip_pr", plan.skip_steps)
        self.assertIn("unit_tests_spec_decode_pr", plan.skip_steps)

    def test_deleted_unit_test_file_does_not_narrow_to_missing_file(
        self,
    ) -> None:
        plan = select_pr_tests.select_for_files(
            ["tests/entrypoints/llm/test_deleted_nonexistent.py"]
        )
        self.assertFalse(plan.skip_all_ci)
        self.assertEqual(plan.lanes, ("verify", "tests"))
        self.assertEqual(
            plan.general_test_pattern, select_pr_tests.DEFAULT_TEST_PATTERN
        )
        self.assertIn("unit_tests_general_pr", plan.skip_steps)
        self.assertIn("unit_tests_multichip_pr", plan.skip_steps)
        self.assertNotIn("typecheck_full", plan.skip_steps)
        self.assertNotIn("unit_tests_cpu_pr", plan.skip_steps)

    def test_kv_transfer_and_offload_run_disagg_and_offload_steps(self) -> None:
        plan = select_pr_tests.select_for_files(
            ["src/vllm_torchtpu/offload/raiden_connector.py"]
        )
        self.assertFalse(plan.skip_all_ci)
        self.assertEqual(plan.lanes, ("verify", "integration", "tests"))
        self.assertNotIn("unit_tests_general_pr", plan.skip_steps)
        self.assertNotIn("unit_tests_multichip_pr", plan.skip_steps)
        self.assertNotIn(
            "integration_k3_pruned75_apc_offload_correctness",
            plan.skip_steps,
        )
        self.assertNotIn("integration_qwen35_gpc_correctness", plan.skip_steps)
        self.assertIn("integration_compilation_cache_e2e", plan.skip_steps)
        self.assertIn("unit_tests_spec_decode_pr", plan.skip_steps)
        self.assertIn("unit_tests_omni_pr", plan.skip_steps)

    def test_kimi_k3_src_change_runs_only_k3_integration(self) -> None:
        plan = select_pr_tests.select_for_files(
            ["src/vllm_torchtpu/models/vllm/kimi_k3/kimi_k3.py"]
        )
        self.assertFalse(plan.skip_all_ci)
        self.assertEqual(plan.lanes, ("verify", "integration", "tests"))
        self.assertNotIn(
            "integration_k3_pruned75_apc_offload_correctness",
            plan.skip_steps,
        )
        self.assertIn("integration_qwen35_apc_correctness", plan.skip_steps)
        self.assertIn("unit_tests_spec_decode_pr", plan.skip_steps)
        self.assertIn("unit_tests_omni_pr", plan.skip_steps)

    def test_core_src_change_runs_full_non_omni_suite(self) -> None:
        plan = select_pr_tests.select_for_files(
            ["src/vllm_torchtpu/worker/model_runner.py"]
        )
        self.assertFalse(plan.skip_all_ci)
        self.assertEqual(plan.lanes, ("verify", "integration", "tests"))
        self.assertEqual(list(plan.skip_steps.keys()), ["unit_tests_omni_pr"])
        self.assertTrue(plan.omni_image_skip)

    def test_dockerfile_change_runs_everything_including_omni(self) -> None:
        plan = select_pr_tests.select_for_files(["docker/Dockerfile"])
        self.assertFalse(plan.skip_all_ci)
        self.assertEqual(plan.lanes, ("verify", "integration", "tests"))
        self.assertEqual(plan.skip_steps, {})
        self.assertEqual(plan.omni_image_skip, "")

    def test_filter_yaml_injects_skip_and_test_pattern(self) -> None:
        yaml_inputs = [_SAMPLE_TESTS_YAML]
        tests_yaml_path = REPO_ROOT / ".buildkite" / "pipeline_tests.yml"
        if tests_yaml_path.exists():
            yaml_inputs.append(tests_yaml_path.read_text(encoding="utf-8"))

        test_file = "tests/entrypoints/llm/test_structured_output.py"
        plan = select_pr_tests.select_for_files([test_file])
        for tests_yaml in yaml_inputs:
            filtered = select_pr_tests.filter_yaml_content(
                yaml_text=tests_yaml,
                skip_steps=plan.skip_steps,
                general_test_pattern=plan.general_test_pattern,
                general_test_parallelism=plan.general_test_parallelism,
                multichip_test_pattern=plan.multichip_test_pattern,
            )
            self.assertIn(
                f'              test-file-pattern: "{test_file}"\n',
                filtered,
            )
            self.assertIn("        parallelism: 1\n", filtered)
            self.assertIn(
                '        key: "unit_tests_spec_decode_pr"\n        skip: "',
                filtered,
            )
        for reason in plan.skip_steps.values():
            self.assertLessEqual(len(reason), 70)

    def test_mapped_repo_paths_and_step_keys_exist(self) -> None:
        for rel_path, steps in select_pr_tests.INTEGRATION_FILE_TO_STEPS.items():
            self.assertTrue(
                (REPO_ROOT / rel_path).is_file(),
                f"Mapped integration file does not exist: {rel_path}",
            )
            self.assertTrue(
                steps.issubset(select_pr_tests.ALL_PR_INTEGRATION_STEPS),
                f"Unknown integration steps for {rel_path}: {steps}",
            )
        for rel_path in select_pr_tests.CPU_TESTED_BENCHMARK_FILES:
            self.assertTrue(
                (REPO_ROOT / rel_path).is_file(),
                f"CPU-tested benchmark file does not exist: {rel_path}",
            )
        for rel_path in select_pr_tests.NIGHTLY_ONLY_INTEGRATION_EXACT:
            self.assertTrue(
                (REPO_ROOT / rel_path).is_file(),
                f"Nightly integration file does not exist: {rel_path}",
            )
        for rel_path in select_pr_tests.NON_PR_BUILDKITE_EXACT:
            if rel_path.startswith(".buildkite/pipeline_"):
                continue
            self.assertTrue(
                (REPO_ROOT / rel_path).is_file(),
                f"Non-PR Buildkite script does not exist: {rel_path}",
            )

        tests_yaml_path = REPO_ROOT / ".buildkite" / "pipeline_tests.yml"
        if tests_yaml_path.is_file():
            tests_yaml = tests_yaml_path.read_text(encoding="utf-8")
            for step_key in select_pr_tests.ALL_PR_TEST_STEPS:
                self.assertIn(f'key: "{step_key}"', tests_yaml)

        integ_yaml_path = REPO_ROOT / ".buildkite" / "pipeline_integration.yml"
        if integ_yaml_path.is_file():
            integ_yaml = integ_yaml_path.read_text(encoding="utf-8")
            for step_key in select_pr_tests.ALL_PR_INTEGRATION_STEPS:
                self.assertIn(f'key: "{step_key}"', integ_yaml)


if __name__ == "__main__":
    unittest.main()
