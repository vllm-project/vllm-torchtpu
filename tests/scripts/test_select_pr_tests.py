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
"""Unit tests and AST import checks for .buildkite/scripts/select_pr_tests.py."""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / ".buildkite" / "scripts" / "select_pr_tests.py"
FIND_OWNERS_PATH = REPO_ROOT / "scripts" / "find_owners.py"

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


def _parse_python_ast(source: str, filename: str) -> ast.AST:
    """Parses Python source into an AST, normalizing PEP 646/695 on Python <3.12."""
    try:
        return ast.parse(source, filename=filename)
    except SyntaxError:
        normalized = re.sub(r"\[\*([A-Za-z_][A-Za-z0-9_]*)", r"[\1", source)
        normalized = re.sub(
            r"^(\s*)type\s+([A-Za-z_][A-Za-z0-9_]*)\s*=",
            r"\1\2 =",
            normalized,
            flags=re.MULTILINE,
        )
        normalized = re.sub(
            r"(\b(?:def|class)\s+[A-Za-z_][A-Za-z0-9_]*)\[[^\]]+\]",
            r"\1",
            normalized,
        )
        return ast.parse(normalized, filename=filename)


def _resolve_relative_module(rel_path: str, level: int, module: str | None) -> str:
    """Resolves a relative `from ..foo import bar` inside `src/vllm_torchtpu/`."""
    posix = Path(rel_path).as_posix()
    if not posix.startswith("src/"):
        return module or ""
    pkg_parts = list(Path(posix[len("src/") :]).parent.parts)
    # A level-1 relative import resolves against the file's own package, so
    # each extra leading dot walks one package up from the file's directory.
    up = level - 1
    if up > 0 and up <= len(pkg_parts):
        base_parts = pkg_parts[: len(pkg_parts) - up]
    elif up == 0:
        base_parts = pkg_parts
    else:
        base_parts = []
    base = ".".join(base_parts)
    if module:
        return f"{base}.{module}" if base else module
    return base


def extract_imported_modules(source: str, rel_path: str) -> set[str]:
    """Extracts all imported module paths from Python source using `ast`."""
    tree = _parse_python_ast(source, filename=rel_path)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            base_mod = (
                node.module or ""
                if node.level == 0
                else _resolve_relative_module(rel_path, node.level, node.module)
            )
            if base_mod:
                imported.add(base_mod)
                for alias in node.names:
                    imported.add(f"{base_mod}.{alias.name}")
    return imported


def find_import_boundary_violations(
    file_imports: dict[str, set[str]],
    rules: tuple[select_pr_tests.ComponentRule, ...] = select_pr_tests.COMPONENT_RULES,
) -> list[str]:
    """Checks that no unmapped file imports an isolated component's modules."""
    violations: list[str] = []
    for rel_path, imported_mods in sorted(file_imports.items()):
        for rule in rules:
            if not rule.module_prefixes:
                continue
            allowed_prefixes = (
                rule.path_prefixes
                + rule.allowed_test_importers
                + rule.allowed_external_importers
            )
            for imp in sorted(imported_mods):
                imports_component = any(
                    imp == mp or imp.startswith(f"{mp}.") for mp in rule.module_prefixes
                )
                is_allowed = any(
                    rel_path == ap or rel_path.startswith(ap) for ap in allowed_prefixes
                )
                if imports_component and not is_allowed:
                    violations.append(
                        f"{rel_path} imports '{imp}' (isolated component "
                        f"'{rule.component}' defined in {rule.config_path})"
                    )
                    break
    return violations


class SelectPrTestsTest(unittest.TestCase):
    """Tests diff-aware PR test selection, YAML filtering, and AST import boundaries."""

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

    def test_diffusion_attention_runs_omni_and_narrowed_kernel_tests(self) -> None:
        plan = select_pr_tests.select_for_files(
            ["src/vllm_torchtpu/kernels/diffusion_attention/kernel.py"]
        )
        self.assertFalse(plan.skip_all_ci)
        self.assertEqual(plan.lanes, ("verify", "tests"))
        self.assertEqual(plan.omni_image_skip, "")
        self.assertNotIn("unit_tests_omni_pr", plan.skip_steps)
        self.assertNotIn("unit_tests_general_pr", plan.skip_steps)
        self.assertIn("unit_tests_multichip_pr", plan.skip_steps)
        self.assertEqual(plan.general_test_parallelism, 1)
        self.assertIn("test_diffusion_flash_attention.py", plan.general_test_pattern)

    def test_megakernel_src_change_narrows_to_megakernel_tests(self) -> None:
        plan = select_pr_tests.select_for_files(["src/vllm_torchtpu/megakernel/ops.py"])
        self.assertFalse(plan.skip_all_ci)
        self.assertEqual(plan.lanes, ("verify", "tests"))
        self.assertTrue(plan.omni_image_skip)
        self.assertNotIn("unit_tests_general_pr", plan.skip_steps)
        self.assertNotIn("unit_tests_multichip_pr", plan.skip_steps)
        self.assertIn("unit_tests_spec_decode_pr", plan.skip_steps)
        self.assertIn("unit_tests_omni_pr", plan.skip_steps)
        self.assertEqual(
            plan.general_test_pattern, "tests/megakernel/**/{test_*,*_test}.py"
        )
        self.assertEqual(plan.general_test_parallelism, 1)

    def test_experimental_src_change_narrows_to_experimental_tests(self) -> None:
        plan = select_pr_tests.select_for_files(
            ["src/vllm_torchtpu/experimental/export/exporter.py"]
        )
        self.assertFalse(plan.skip_all_ci)
        self.assertEqual(plan.lanes, ("verify", "tests"))
        self.assertTrue(plan.omni_image_skip)
        self.assertNotIn("unit_tests_general_pr", plan.skip_steps)
        self.assertIn("unit_tests_multichip_pr", plan.skip_steps)
        self.assertIn("unit_tests_spec_decode_pr", plan.skip_steps)
        self.assertEqual(
            plan.general_test_pattern, "tests/experimental/**/{test_*,*_test}.py"
        )
        self.assertEqual(plan.general_test_parallelism, 1)

    def test_spec_decode_src_change_narrows_general_and_runs_spec_decode(self) -> None:
        plan = select_pr_tests.select_for_files(
            ["src/vllm_torchtpu/spec_decode/eagle3.py"]
        )
        self.assertFalse(plan.skip_all_ci)
        self.assertEqual(plan.lanes, ("verify", "tests"))
        self.assertNotIn("unit_tests_spec_decode_pr", plan.skip_steps)
        self.assertNotIn("unit_tests_spec_decode_multichip_pr", plan.skip_steps)
        self.assertNotIn("unit_tests_general_pr", plan.skip_steps)
        self.assertEqual(
            plan.general_test_pattern, "tests/spec_decode/**/{test_*,*_test}.py"
        )
        self.assertEqual(plan.general_test_parallelism, 1)

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

        sub_plan = select_pr_tests.select_for_files(
            ["tests/spec_decode/test_eagle3.py"]
        )
        self.assertNotIn("unit_tests_spec_decode_pr", sub_plan.skip_steps)
        self.assertNotIn("unit_tests_spec_decode_multichip_pr", sub_plan.skip_steps)
        self.assertNotIn("unit_tests_general_pr", sub_plan.skip_steps)
        self.assertEqual(
            sub_plan.general_test_pattern, "tests/spec_decode/test_eagle3.py"
        )
        self.assertEqual(sub_plan.general_test_parallelism, 1)

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

    def test_phase1_mode_keeps_phase2_src_components_on_full_suite(self) -> None:
        plan = select_pr_tests.select_for_files(
            ["src/vllm_torchtpu/megakernel/ops.py"],
            mode="phase1",
        )
        self.assertEqual(plan.lanes, ("verify", "integration", "tests"))
        self.assertEqual(list(plan.skip_steps.keys()), ["unit_tests_omni_pr"])

        omni_plan = select_pr_tests.select_for_files(
            ["src/vllm_torchtpu/omni/runner.py"],
            mode="phase1",
        )
        self.assertEqual(omni_plan.lanes, ("verify", "tests"))
        self.assertNotIn("unit_tests_omni_pr", omni_plan.skip_steps)

    def test_shadow_mode_runs_full_suite_with_would_skip_annotation(self) -> None:
        plan = select_pr_tests.select_for_files(
            ["src/vllm_torchtpu/megakernel/ops.py"],
            mode="shadow",
        )
        self.assertFalse(plan.skip_all_ci)
        self.assertEqual(plan.lanes, ("verify", "integration", "tests"))
        self.assertEqual(plan.skip_steps, {})
        self.assertEqual(plan.omni_image_skip, "")
        self.assertIn("would-skip; running full CI", plan.annotation)

    def test_off_mode_and_ci_full_override_run_unfiltered(self) -> None:
        for kwargs in (
            {"mode": "off"},
            {"commit_message": "Fix megakernel [ci full]"},
            {"ci_extras": "full"},
        ):
            plan = select_pr_tests.select_for_files(
                ["src/vllm_torchtpu/megakernel/ops.py"],
                **kwargs,
            )
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

        sample_json = {
            "steps": [
                {
                    "group": "Unit Tests on TPU",
                    "steps": [
                        {
                            "key": "unit_tests_general_pr",
                            "parallelism": 3,
                            "plugins": [
                                {
                                    "tests#v1.0.0": {
                                        "test-file-pattern": (
                                            "tests/**/{test_*,*_test}.py"
                                        )
                                    }
                                }
                            ],
                        },
                        {"key": "unit_tests_spec_decode_pr", "parallelism": 3},
                    ],
                }
            ]
        }
        filtered_json = select_pr_tests.filter_json_content(
            doc=sample_json,
            skip_steps=plan.skip_steps,
            general_test_pattern=plan.general_test_pattern,
            general_test_parallelism=plan.general_test_parallelism,
            multichip_test_pattern=plan.multichip_test_pattern,
        )
        gen_step = filtered_json["steps"][0]["steps"][0]
        spec_step = filtered_json["steps"][0]["steps"][1]
        self.assertEqual(gen_step["parallelism"], 1)
        self.assertEqual(
            gen_step["plugins"][0]["tests#v1.0.0"]["test-file-pattern"],
            test_file,
        )
        self.assertIn("skip", spec_step)

    def test_mapped_repo_paths_and_step_keys_exist(self) -> None:
        config = select_pr_tests.load_ci_paths_config(REPO_ROOT)
        for comp_rel in config.component_config_paths:
            self.assertTrue(
                (REPO_ROOT / comp_rel).is_file(),
                f"Component ci_paths.yaml does not exist: {comp_rel}",
            )
        for rule in config.component_rules:
            self.assertTrue(
                set(rule.test_steps).issubset(select_pr_tests.ALL_PR_TEST_STEPS),
                f"Unknown test steps in {rule.component}: {rule.test_steps}",
            )
            self.assertTrue(
                set(rule.integration_steps).issubset(
                    select_pr_tests.ALL_PR_INTEGRATION_STEPS
                ),
                (
                    f"Unknown integration steps in {rule.component}: "
                    f"{rule.integration_steps}"
                ),
            )
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

    def test_ast_import_boundaries_for_isolated_components(self) -> None:
        """Parses repo Python files with `ast` to verify component boundaries."""
        scan_roots = ("src/vllm_torchtpu", "tests", "tests_omni")
        file_imports: dict[str, set[str]] = {}
        for root_rel in scan_roots:
            root_dir = REPO_ROOT / root_rel
            if not root_dir.is_dir():
                continue
            for py_file in sorted(root_dir.rglob("*.py")):
                rel_path = py_file.relative_to(REPO_ROOT).as_posix()
                source = py_file.read_text(encoding="utf-8")
                file_imports[rel_path] = extract_imported_modules(source, rel_path)

        violations = find_import_boundary_violations(file_imports)
        self.assertEqual(
            violations,
            [],
            "Isolated component AST import boundary violated! Update the "
            "component's ci_paths.yaml (allowed_test_importers / "
            "allowed_external_importers) or widen its test_steps:\n  "
            + "\n  ".join(violations),
        )

    def test_ast_boundary_check_catches_simulated_cross_import(self) -> None:
        """Verifies that a new cross-import from shared src/ is caught by AST."""
        simulated_imports = {
            "src/vllm_torchtpu/worker/model_runner.py": extract_imported_modules(
                "from vllm_torchtpu.megakernel.ops import rmsnorm\n",
                "src/vllm_torchtpu/worker/model_runner.py",
            ),
            "src/vllm_torchtpu/omni/runner.py": extract_imported_modules(
                "from ..experimental.export import exporter\n",
                "src/vllm_torchtpu/omni/runner.py",
            ),
        }
        violations = find_import_boundary_violations(simulated_imports)
        self.assertEqual(len(violations), 2)
        self.assertIn("experimental", violations[0])
        self.assertIn("megakernel", violations[1])

    def test_component_ci_paths_codeowners_routing(self) -> None:
        """Verifies .github/CODEOWNERS routes each ci_paths.yaml to right owners."""
        owners_spec = importlib.util.spec_from_file_location(
            "find_owners", FIND_OWNERS_PATH
        )
        assert owners_spec is not None and owners_spec.loader is not None
        find_owners = importlib.util.module_from_spec(owners_spec)
        owners_spec.loader.exec_module(find_owners)

        rules, _ = find_owners.load_rules(str(REPO_ROOT))
        expected_owners = {
            ".buildkite/ci_paths.yaml": {"@QiliangCui", "@theminghuang"},
            "src/vllm_torchtpu/omni/ci_paths.yaml": {
                "@mbohlool",
                "@QiliangCui",
                "@theminghuang",
            },
            "src/vllm_torchtpu/megakernel/ci_paths.yaml": {
                "@caojx-google",
                "@weinan1997",
                "@hhb",
                "@QiliangCui",
                "@theminghuang",
            },
            "src/vllm_torchtpu/experimental/ci_paths.yaml": {
                "@QiliangCui",
                "@theminghuang",
            },
            "src/vllm_torchtpu/spec_decode/ci_paths.yaml": {
                "@Lumosis",
                "@gxd3",
                "@QiliangCui",
                "@theminghuang",
            },
            "src/vllm_torchtpu/models/vllm/kimi_k3/ci_paths.yaml": {
                "@kyuyeunk",
                "@qizzzh",
                "@QiliangCui",
                "@theminghuang",
            },
        }
        for path, required in expected_owners.items():
            matched = find_owners.owners_for(path, rules)
            self.assertIsNotNone(matched, f"No CODEOWNERS rule matched {path}")
            self.assertTrue(
                required.issubset(set(matched.owners)),
                f"Expected {required} in CODEOWNERS for {path}, got {matched.owners}",
            )

    def test_bootstrap_merge_queue_and_release_queue_behavior(self) -> None:
        """Verifies bootstrap.sh scopes main Merge Queue and not release queues."""
        bootstrap_sh = REPO_ROOT / ".buildkite" / "scripts" / "bootstrap.sh"
        fake_agent = (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            'echo "$*" >> "${TEST_AGENT_LOG}/calls.log"\n'
            'if [[ "$1" == "pipeline" && "$2" == "upload" ]]; then\n'
            "  shopt -s nullglob\n"
            '  ufiles=("${TEST_AGENT_LOG}"/upload_*.yml)\n'
            '  cat > "${TEST_AGENT_LOG}/upload_${#ufiles[@]}.yml"\n'
            "fi\n"
        )
        fake_git = (
            "#!/usr/bin/env bash\n"
            'if [[ "${1:-}" == "fetch" || "${1:-}" == "merge-base" ]]; then\n'
            "  exit 0\n"
            "fi\n"
            'if [[ "${1:-}" == "diff" || "${1:-}" == "diff-tree" ]]; then\n'
            '  echo "src/vllm_torchtpu/megakernel/ops.py"\n'
            "  exit 0\n"
            "fi\n"
            'exec /usr/bin/git "$@"\n'
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bin_dir = tmp_path / "bin"
            bin_dir.mkdir()
            for name, content in (
                ("buildkite-agent", fake_agent),
                ("git", fake_git),
            ):
                p = bin_dir / name
                p.write_text(content, encoding="utf-8")
                p.chmod(p.stat().st_mode | stat.S_IEXEC)

            # 1. Merge Queue build on main (gh-readonly-queue/main/pr-123)
            mq_log = tmp_path / "mq"
            mq_log.mkdir()
            subprocess.run(
                ["bash", str(bootstrap_sh)],
                cwd=REPO_ROOT,
                env={
                    "PATH": f"{bin_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
                    "HOME": str(tmp_path),
                    "TEST_AGENT_LOG": str(mq_log),
                    "BUILDKITE_COMMIT": "HEAD",
                    "BUILDKITE_BRANCH": "gh-readonly-queue/main/pr-123-abc",
                    "BUILDKITE_PULL_REQUEST": "false",
                    "BUILDKITE_PULL_REQUEST_LABELS": "",
                    "BUILDKITE_PIPELINE_SLUG": "vllm-torchtpu-ci",
                },
                capture_output=True,
                text=True,
                check=True,
            )
            mq_uploads = [
                p.read_text(encoding="utf-8")
                for p in sorted(mq_log.glob("upload_*.yml"))
            ]
            # Only verify + tests + build (integration lane skipped for megakernel-only)
            self.assertEqual(len(mq_uploads), 3)
            self.assertIn(
                'test-file-pattern: "tests/megakernel/**/{test_*,*_test}.py"',
                mq_uploads[1],
            )

            # 2. Release branch Merge Queue (gh-readonly-queue/releases/v0.1.0/pr-123)
            rel_log = tmp_path / "rel"
            rel_log.mkdir()
            subprocess.run(
                ["bash", str(bootstrap_sh)],
                cwd=REPO_ROOT,
                env={
                    "PATH": f"{bin_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
                    "HOME": str(tmp_path),
                    "TEST_AGENT_LOG": str(rel_log),
                    "BUILDKITE_COMMIT": "HEAD",
                    "BUILDKITE_BRANCH": "gh-readonly-queue/releases/v0.1.0/pr-123",
                    "BUILDKITE_PULL_REQUEST": "false",
                    "BUILDKITE_PULL_REQUEST_LABELS": "",
                    "BUILDKITE_PIPELINE_SLUG": "vllm-torchtpu-ci",
                },
                capture_output=True,
                text=True,
                check=True,
            )
            rel_uploads = [
                p.read_text(encoding="utf-8")
                for p in sorted(rel_log.glob("upload_*.yml"))
            ]
            # Unfiltered: verify + integration + tests + build = 4 uploads
            self.assertEqual(len(rel_uploads), 4)


if __name__ == "__main__":
    unittest.main()
