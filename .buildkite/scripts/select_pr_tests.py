#!/usr/bin/env python3
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
"""Diff-aware PR test selection and YAML step filter for Buildkite CI."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEST_PATTERN = "tests/**/{test_*,*_test}.py"

ALL_PR_TEST_STEPS: tuple[str, ...] = (
    "typecheck_full",
    "unit_tests_cpu_pr",
    "unit_tests_general_pr",
    "unit_tests_multichip_pr",
    "unit_tests_spec_decode_pr",
    "unit_tests_spec_decode_multichip_pr",
    "unit_tests_omni_pr",
)

ALL_PR_INTEGRATION_STEPS: tuple[str, ...] = (
    "integration_qwen35_pcp4_dp4_disagg_correctness",
    "integration_qwen35_pcp4_2tp2dp_disagg_correctness",
    "integration_qwen35_pcp4tp1_tp4dcp2_disagg_correctness",
    "presubmit_disagg_single_host",
    "integration_qwen35_apc_correctness",
    "integration_qwen35_gpc_global_dram_pool",
    "integration_compilation_cache_e2e",
    "integration_k3_pruned75_apc_offload_correctness",
    "vllm_torchtpu_multihost_presubmit",
    "integration_qwen35_pcp8_2tp4dp_disagg_correctness",
    "integration_qwen35_gpc_correctness",
)

INTEGRATION_FILE_TO_STEPS: dict[str, set[str]] = {
    "scripts/vllm/integration/run_qwen35_p4d2_disagg_correctness.sh": {
        "integration_qwen35_pcp4_dp4_disagg_correctness",
        "integration_qwen35_pcp4_2tp2dp_disagg_correctness",
    },
    "examples/disagg/launch_qwen35_p4d2_v2_baseline.sh": {
        "integration_qwen35_pcp4_dp4_disagg_correctness",
        "integration_qwen35_pcp4_2tp2dp_disagg_correctness",
    },
    "scripts/vllm/integration/run_qwen35_pcp4_2tp2dp_disagg_correctness.sh": {
        "integration_qwen35_pcp4_2tp2dp_disagg_correctness"
    },
    "scripts/vllm/integration/qwen35_pd_disagg_common_env.sh": {
        "integration_qwen35_pcp4_2tp2dp_disagg_correctness",
        "integration_qwen35_pcp4tp1_tp4dcp2_disagg_correctness",
        "integration_qwen35_pcp8_2tp4dp_disagg_correctness",
    },
    "scripts/vllm/integration/run_qwen35_pcp_tp_dcp_disagg_correctness.sh": {
        "integration_qwen35_pcp4tp1_tp4dcp2_disagg_correctness"
    },
    "examples/disagg/run_disagg_single_host.sh": {"presubmit_disagg_single_host"},
    "examples/disagg/toy_proxy_server.py": {
        "presubmit_disagg_single_host",
        "integration_qwen35_pcp4_dp4_disagg_correctness",
        "integration_qwen35_pcp4_2tp2dp_disagg_correctness",
        "integration_qwen35_pcp4tp1_tp4dcp2_disagg_correctness",
        "integration_qwen35_pcp8_2tp4dp_disagg_correctness",
    },
    "scripts/vllm/integration/run_qwen35_apc_correctness.sh": {
        "integration_qwen35_apc_correctness"
    },
    "scripts/vllm/integration/run_qwen35_gpc_global_dram_pool.sh": {
        "integration_qwen35_gpc_global_dram_pool"
    },
    "scripts/vllm/integration/smoke_gpc_global_dram_pool.py": {
        "integration_qwen35_gpc_global_dram_pool"
    },
    "scripts/vllm/integration/run_compilation_cache_e2e.sh": {
        "integration_compilation_cache_e2e"
    },
    "scripts/vllm/integration/run_k3_pruned75_apc_offload_correctness.sh": {
        "integration_k3_pruned75_apc_offload_correctness"
    },
    "scripts/vllm/integration/smoke_prefix_multiturn_and_eviction_recall.py": {
        "integration_k3_pruned75_apc_offload_correctness"
    },
    "scripts/vllm/integration/smoke_basic_offload_correctness.py": {
        "integration_k3_pruned75_apc_offload_correctness"
    },
    "scripts/vllm/integration/smoke_mamba_state_contamination.py": {
        "integration_qwen35_apc_correctness",
        "integration_k3_pruned75_apc_offload_correctness",
    },
    "scripts/vllm/integration/smoke_prefix_cache_correctness.py": {
        "integration_qwen35_pcp4_dp4_disagg_correctness",
        "integration_qwen35_pcp4_2tp2dp_disagg_correctness",
        "integration_qwen35_pcp4tp1_tp4dcp2_disagg_correctness",
        "integration_qwen35_apc_correctness",
        "integration_qwen35_pcp8_2tp4dp_disagg_correctness",
    },
    "scripts/vllm/integration/smoke_prefix_cache_e2e_divergence.py": {
        "integration_qwen35_pcp4_dp4_disagg_correctness",
        "integration_qwen35_pcp4_2tp2dp_disagg_correctness",
        "integration_qwen35_pcp4tp1_tp4dcp2_disagg_correctness",
        "integration_qwen35_apc_correctness",
        "integration_k3_pruned75_apc_offload_correctness",
        "integration_qwen35_pcp8_2tp4dp_disagg_correctness",
    },
    "scripts/vllm/integration/run_qwen35_p8d8_disagg_correctness.sh": {
        "integration_qwen35_pcp8_2tp4dp_disagg_correctness",
        "integration_qwen35_pcp4tp1_tp4dcp2_disagg_correctness",
    },
    "examples/disagg/launch_qwen35_disagg_multihost.sh": {
        "integration_qwen35_pcp8_2tp4dp_disagg_correctness",
        "integration_qwen35_pcp4tp1_tp4dcp2_disagg_correctness",
    },
    ".buildkite/kubernetes/manifests/workloads/qwen3.5-p8d8-disagg.yaml": {
        "integration_qwen35_pcp8_2tp4dp_disagg_correctness"
    },
    "scripts/vllm/integration/run_qwen35_gpc_correctness.sh": {
        "integration_qwen35_gpc_correctness"
    },
    "scripts/vllm/integration/smoke_gpc_global_offloading.py": {
        "integration_qwen35_gpc_correctness"
    },
    ".buildkite/kubernetes/manifests/workloads/qwen3.5-gpc-two-replica.yaml": {
        "integration_qwen35_gpc_correctness"
    },
    ".buildkite/kubernetes/single_host_engine.sh": {
        "integration_qwen35_pcp8_2tp4dp_disagg_correctness",
        "integration_qwen35_gpc_correctness",
    },
    "examples/offline_inference.py": {
        "vllm_torchtpu_multihost_presubmit",
        "integration_compilation_cache_e2e",
    },
    ".buildkite/kubernetes/manifests/workloads/ray-multihost-slice.yaml": {
        "vllm_torchtpu_multihost_presubmit"
    },
    ".buildkite/kubernetes/multihost_entry.sh": {"vllm_torchtpu_multihost_presubmit"},
    ".buildkite/kubernetes/manifests/workloads/mp-multihost-slice.yaml": {
        "vllm_torchtpu_multihost_presubmit"
    },
    ".buildkite/kubernetes/mp_multihost_entry.sh": {
        "vllm_torchtpu_multihost_presubmit"
    },
    ".buildkite/kubernetes/multihost_lib.sh": {
        "vllm_torchtpu_multihost_presubmit",
        "integration_qwen35_pcp8_2tp4dp_disagg_correctness",
        "integration_qwen35_gpc_correctness",
    },
    ".buildkite/kubernetes/run_multihost.sh": {
        "vllm_torchtpu_multihost_presubmit",
        "integration_qwen35_pcp8_2tp4dp_disagg_correctness",
        "integration_qwen35_gpc_correctness",
    },
}

CPU_TESTED_BENCHMARK_FILES: set[str] = {
    "scripts/vllm/benchmarking/check_regression.py",
    "scripts/vllm/benchmarking/configs/qwen3-0.6b-structured-flat-json.sh",
    "scripts/vllm/benchmarking/configs/qwen3-0.6b-structured-complex-ast.sh",
    "scripts/vllm/benchmarking/configs/qwen3-0.6b-structured-character-dataset.sh",
    "scripts/vllm/benchmarking/structured_schemas/flat_json.json",
    "scripts/vllm/benchmarking/structured_schemas/complex_ast.json",
    "scripts/vllm/benchmarking/structured_schemas/character_schema.json",
    "scripts/vllm/benchmarking/datasets/character_benchmark.json",
    "scripts/vllm/benchmark/inferencex/sweep_summary.py",
    "scripts/vllm/benchmark/inferencex/sweep_lib.sh",
    "scripts/vllm/integration/lm_eval_accuracy.py",
    "scripts/ci_failure_metrics.py",
    "scripts/find_owners.py",
}

NIGHTLY_ONLY_INTEGRATION_EXACT: set[str] = {
    "scripts/vllm/integration/test_accuracy.py",
    "scripts/vllm/integration/test_safety_model_accuracy.py",
    "scripts/vllm/integration/run_dsv4_offload_correctness.sh",
    "scripts/vllm/integration/run_dsv4_pd_correctness.sh",
    "scripts/vllm/integration/check_dsv4_stage3_log.py",
    "scripts/vllm/integration/run_glm52_fp4_offload_correctness.sh",
    "scripts/vllm/integration/run_k3_1p1d_disagg_correctness.sh",
    "scripts/vllm/integration/run_pcp_correctness.sh",
    "scripts/vllm/integration/run_qwen35_block_major_disagg_offload.sh",
    "scripts/vllm/integration/run_qwen35_pp_disagg_offload.sh",
    "scripts/vllm/integration/smoke_block_major_offload.py",
    ".buildkite/kubernetes/manifests/workloads/dsv4-pd-disagg.yaml",
    ".buildkite/kubernetes/manifests/workloads/kimi-k3-1p1d-4dp8tp.yaml",
    ".buildkite/kubernetes/manifests/workloads/kimi-k3-agentx-1p1d.yaml",
    ".buildkite/kubernetes/manifests/workloads/qwen3.5-397b-1p1d.yaml",
    ".buildkite/kubernetes/disagg_benchmark.sh",
    ".buildkite/kubernetes/disagg_engine.sh",
    ".buildkite/kubernetes/k3_1p1d.sh",
    ".buildkite/kubernetes/k3_agentx_client_pod.sh",
    ".buildkite/kubernetes/benchmark_with_profile.sh",
}

NON_PR_BUILDKITE_EXACT: set[str] = {
    ".buildkite/pipeline_perf.yml",
    ".buildkite/pipeline_promote.yml",
    ".buildkite/pipeline_nightly_image.yml",
    ".buildkite/pipeline_dynamic.yml",
    ".buildkite/pipeline_dev.yml",
    ".buildkite/pipeline_coverage.yml",
    ".buildkite/scripts/bootstrap_dev.sh",
    ".buildkite/scripts/upload_models_and_features.sh",
    ".buildkite/scripts/record_step_result.sh",
    ".buildkite/scripts/generate_support_matrices.sh",
    ".buildkite/scripts/commit_support_matrices.sh",
    ".buildkite/scripts/commit_verified_commit_hashes.sh",
    ".buildkite/scripts/nightly_report.sh",
    ".buildkite/scripts/notify_test_results.sh",
    ".buildkite/scripts/update_lkg_version.sh",
    ".buildkite/scripts/publish_nightly_images.sh",
    ".buildkite/scripts/build_k3_engine_image.sh",
    ".buildkite/scripts/build_core_wheel.sh",
    ".buildkite/scripts/build_vllm_torchtpu.sh",
    ".buildkite/scripts/publish_wheel.sh",
    ".buildkite/scripts/upload_profile.sh",
}


@dataclass
class SelectionPlan:
    """Computed lane and step selection for a PR diff."""

    skip_all_ci: bool = False
    skip_all_reason: str = ""
    lanes: tuple[str, ...] = ("verify", "integration", "tests")
    omni_image_skip: str = ""
    skip_steps: dict[str, str] = field(default_factory=dict)
    general_test_pattern: str = DEFAULT_TEST_PATTERN
    general_test_parallelism: int = 3
    multichip_test_pattern: str = DEFAULT_TEST_PATTERN
    annotation: str = ""


def _is_doc_or_non_pr_file(path: str) -> bool:
    if path.endswith((".md", ".ico", ".png")):
        return True
    if path in {
        "README",
        "LICENSE",
        "mkdocs.yml",
        ".github/CODEOWNERS",
        ".git-blame-ignore-revs",
        ".pyrefly-baseline.json",
        "verified_commit_hashes.csv",
    }:
        return True
    if path.startswith(
        (
            "docs/",
            ".github/",
            "support_matrices/",
            ".buildkite/features/",
            ".buildkite/models/",
            ".buildkite/parallelism/",
            ".buildkite/quantization/",
            ".buildkite/rl/",
            ".buildkite/kernel_microbenchmarks/",
        )
    ):
        return True
    if path in NON_PR_BUILDKITE_EXACT:
        return True
    if path in NIGHTLY_ONLY_INTEGRATION_EXACT:
        return True
    return (
        path.startswith(
            (
                "scripts/vllm/benchmarking/",
                "scripts/vllm/benchmark/",
                "scripts/vllm/benchmarks/",
            )
        )
        and path not in CPU_TESTED_BENCHMARK_FILES
    )


def _is_cpu_only_file(path: str) -> bool:
    if path in {
        ".pyrefly-baseline-full.json",
        ".pre-commit-config.yaml",
    }:
        return True
    if path.startswith(".buildkite/scripts/") and path.endswith(".py"):
        return True
    if path in CPU_TESTED_BENCHMARK_FILES:
        return True
    return path.startswith(
        ("tools/", "tests/scripts/", "tests/tools/", ".buildkite/release/")
    )


def _is_omni_file(path: str) -> bool:
    return path.startswith(
        (
            "src/vllm_torchtpu/omni/",
            "src/vllm_torchtpu/kernels/diffusion_attention/",
            "tests_omni/",
            "tests/omni/",
        )
    )


def _classify_unit_test_file(
    rel_path: str, repo_root: Path = REPO_ROOT
) -> tuple[bool, bool]:
    """Returns (has_tpu_general, has_multichip) for a unit test file."""
    full_path = repo_root / rel_path
    if not full_path.is_file():
        return (False, False)
    try:
        content = full_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return (True, True)
    has_tpu = bool(re.search(r"\btpu_test\b", content))
    has_multichip = bool(re.search(r"\bmultichip\b", content))
    return (has_tpu, has_multichip)


def _format_brace_pattern(files: list[str]) -> str:
    unique = sorted(set(files))
    if not unique:
        return DEFAULT_TEST_PATTERN
    if len(unique) == 1:
        return unique[0]
    return "{" + ",".join(unique) + "}"


def _finalize_plan(
    run_test_steps: set[str],
    run_integration_steps: set[str],
    reason: str,
    general_test_pattern: str = DEFAULT_TEST_PATTERN,
    general_test_parallelism: int = 3,
    multichip_test_pattern: str = DEFAULT_TEST_PATTERN,
) -> SelectionPlan:
    lanes = ["verify"]
    if run_integration_steps:
        lanes.append("integration")
    if run_test_steps:
        lanes.append("tests")

    skip_steps: dict[str, str] = {}
    if run_test_steps:
        for step in ALL_PR_TEST_STEPS:
            if step not in run_test_steps:
                skip_steps[step] = reason[:70]
    if run_integration_steps:
        for step in ALL_PR_INTEGRATION_STEPS:
            if step not in run_integration_steps:
                skip_steps[step] = reason[:70]

    omni_image_skip = ""
    if "unit_tests_omni_pr" not in run_test_steps:
        omni_image_skip = "No Omni changes in this PR."

    skipped_count = len(skip_steps)
    dropped_lanes = [lane for lane in ("integration", "tests") if lane not in lanes]
    summary_parts = [f"lanes: `{' '.join(lanes)}`"]
    if dropped_lanes:
        summary_parts.append(f"skipped lanes: `{', '.join(dropped_lanes)}`")
    if skipped_count:
        summary_parts.append(f"skipped {skipped_count} step(s)")
    if general_test_pattern != DEFAULT_TEST_PATTERN:
        summary_parts.append(
            f"general pattern: `{general_test_pattern}` "
            f"(parallelism={general_test_parallelism})"
        )
    if multichip_test_pattern != DEFAULT_TEST_PATTERN:
        summary_parts.append(f"multichip pattern: `{multichip_test_pattern}`")

    annotation = ""
    if (
        dropped_lanes
        or set(skip_steps) - {"unit_tests_omni_pr"}
        or general_test_pattern != DEFAULT_TEST_PATTERN
        or multichip_test_pattern != DEFAULT_TEST_PATTERN
    ):
        annotation = (
            f"Diff-aware PR test selection ({reason}): "
            + "; ".join(summary_parts)
            + "."
        )
    return SelectionPlan(
        skip_all_ci=False,
        lanes=tuple(lanes),
        omni_image_skip=omni_image_skip,
        skip_steps=skip_steps,
        general_test_pattern=general_test_pattern,
        general_test_parallelism=general_test_parallelism,
        multichip_test_pattern=multichip_test_pattern,
        annotation=annotation,
    )


def select_for_files(
    changed_files: list[str], repo_root: Path = REPO_ROOT
) -> SelectionPlan:
    """Computes the PR test selection plan for a list of changed file paths."""
    files = [f.strip() for f in changed_files if f.strip()]
    if not files:
        return SelectionPlan()

    active_files = [f for f in files if not _is_doc_or_non_pr_file(f)]
    if not active_files:
        return SelectionPlan(
            skip_all_ci=True,
            skip_all_reason=(
                "Only docs, benchmarks, or non-PR workflow/config files changed."
            ),
            lanes=(),
        )

    for f in active_files:
        if (
            f == "pyproject.toml"
            or f.startswith("docker/")
            or f
            in {
                ".buildkite/pipeline_build.yml",
                ".buildkite/pipeline_verify.yml",
                ".buildkite/scripts/bootstrap.sh",
                ".buildkite/scripts/cap_pr_tpu_concurrency.py",
                ".buildkite/scripts/configs/pipeline_config.sh",
                ".buildkite/scripts/setup_docker_env.sh",
                ".buildkite/kubernetes/run.sh",
                ".buildkite/kubernetes/common.sh",
            }
        ):
            return SelectionPlan()

    run_test_steps: set[str] = set()
    run_integration_steps: set[str] = set()
    general_test_files: list[str] = []
    multichip_test_files: list[str] = []
    full_general_tests = False
    full_multichip_tests = False
    reasons: list[str] = []

    for f in active_files:
        if _is_cpu_only_file(f):
            run_test_steps.update({"typecheck_full", "unit_tests_cpu_pr"})
            reasons.append("CPU/tooling changes")
            continue

        if _is_omni_file(f):
            run_test_steps.update(
                {"typecheck_full", "unit_tests_cpu_pr", "unit_tests_omni_pr"}
            )
            reasons.append("Omni changes")
            continue

        if f in INTEGRATION_FILE_TO_STEPS:
            run_integration_steps.update(INTEGRATION_FILE_TO_STEPS[f])
            if f.endswith(".py"):
                run_test_steps.update({"typecheck_full", "unit_tests_cpu_pr"})
            reasons.append("integration script/manifest changes")
            continue

        if f == ".buildkite/pipeline_integration.yml":
            run_integration_steps.update(ALL_PR_INTEGRATION_STEPS)
            reasons.append("integration pipeline YAML changes")
            continue

        if f == ".buildkite/pipeline_tests.yml":
            run_test_steps.update(ALL_PR_TEST_STEPS)
            full_general_tests = True
            full_multichip_tests = True
            reasons.append("tests pipeline YAML changes")
            continue

        if f.startswith("tests/"):
            run_test_steps.update({"typecheck_full", "unit_tests_cpu_pr"})
            if f == "tests/entrypoints/llm/test_speculative_decoding.py":
                run_test_steps.update(
                    {
                        "unit_tests_spec_decode_pr",
                        "unit_tests_spec_decode_multichip_pr",
                    }
                )
                reasons.append("spec-decode unit test changes")
            elif not (
                Path(f).name.startswith("test_") or Path(f).name.endswith("_test.py")
            ):
                run_test_steps.update(
                    {
                        "unit_tests_general_pr",
                        "unit_tests_multichip_pr",
                        "unit_tests_spec_decode_pr",
                        "unit_tests_spec_decode_multichip_pr",
                    }
                )
                full_general_tests = True
                full_multichip_tests = True
                reasons.append("shared test fixture changes")
            elif not (repo_root / f).is_file():
                reasons.append("deleted unit test file")
            else:
                has_gen, has_multi = _classify_unit_test_file(f, repo_root)
                if not has_gen and not has_multi:
                    reasons.append("CPU unit test changes")
                else:
                    if has_gen:
                        run_test_steps.add("unit_tests_general_pr")
                        general_test_files.append(f)
                    if has_multi:
                        run_test_steps.add("unit_tests_multichip_pr")
                        multichip_test_files.append(f)
                    reasons.append("TPU unit test changes")
            continue

        if f.startswith("src/vllm_torchtpu/spec_decode/"):
            run_test_steps.update(
                {
                    "typecheck_full",
                    "unit_tests_cpu_pr",
                    "unit_tests_general_pr",
                    "unit_tests_multichip_pr",
                    "unit_tests_spec_decode_pr",
                    "unit_tests_spec_decode_multichip_pr",
                }
            )
            full_general_tests = True
            full_multichip_tests = True
            reasons.append("spec_decode module changes")
            continue

        if f.startswith(
            (
                "src/vllm_torchtpu/models/vllm/kimi_k3/",
                "src/vllm_torchtpu/kernels/kimi_k3/",
            )
        ):
            run_test_steps.update(
                {
                    "typecheck_full",
                    "unit_tests_cpu_pr",
                    "unit_tests_general_pr",
                    "unit_tests_multichip_pr",
                }
            )
            run_integration_steps.add("integration_k3_pruned75_apc_offload_correctness")
            full_general_tests = True
            full_multichip_tests = True
            reasons.append("Kimi-K3 module changes")
            continue

        if f.startswith(
            (
                "src/vllm_torchtpu/distributed/kv_transfer/",
                "src/vllm_torchtpu/offload/",
            )
        ):
            run_test_steps.update(
                {
                    "typecheck_full",
                    "unit_tests_cpu_pr",
                    "unit_tests_general_pr",
                    "unit_tests_multichip_pr",
                }
            )
            run_integration_steps.update(
                {
                    "integration_qwen35_pcp4_dp4_disagg_correctness",
                    "integration_qwen35_pcp4_2tp2dp_disagg_correctness",
                    "integration_qwen35_pcp4tp1_tp4dcp2_disagg_correctness",
                    "presubmit_disagg_single_host",
                    "integration_qwen35_pcp8_2tp4dp_disagg_correctness",
                    "integration_qwen35_gpc_correctness",
                    "integration_qwen35_gpc_global_dram_pool",
                    "integration_k3_pruned75_apc_offload_correctness",
                }
            )
            full_general_tests = True
            full_multichip_tests = True
            reasons.append("KV-transfer/offload module changes")
            continue

        run_test_steps.update(
            {
                "typecheck_full",
                "unit_tests_cpu_pr",
                "unit_tests_general_pr",
                "unit_tests_multichip_pr",
                "unit_tests_spec_decode_pr",
                "unit_tests_spec_decode_multichip_pr",
            }
        )
        run_integration_steps.update(ALL_PR_INTEGRATION_STEPS)
        full_general_tests = True
        full_multichip_tests = True
        reasons.append("core runtime changes")

    general_pattern = DEFAULT_TEST_PATTERN
    general_parallelism = 3
    if (
        "unit_tests_general_pr" in run_test_steps
        and not full_general_tests
        and general_test_files
    ):
        general_pattern = _format_brace_pattern(general_test_files)
        general_parallelism = min(3, len(set(general_test_files)))

    multichip_pattern = DEFAULT_TEST_PATTERN
    if (
        "unit_tests_multichip_pr" in run_test_steps
        and not full_multichip_tests
        and multichip_test_files
    ):
        multichip_pattern = _format_brace_pattern(multichip_test_files)

    unique_reasons = list(dict.fromkeys(reasons))
    reason_str = ", ".join(unique_reasons[:2]) if unique_reasons else "PR diff"
    if len(unique_reasons) > 2:
        reason_str += f" (+{len(unique_reasons) - 2} more)"

    return _finalize_plan(
        run_test_steps=run_test_steps,
        run_integration_steps=run_integration_steps,
        reason=f"Skipped: only {reason_str}."[:70],
        general_test_pattern=general_pattern,
        general_test_parallelism=general_parallelism,
        multichip_test_pattern=multichip_pattern,
    )


def filter_yaml_content(
    yaml_text: str,
    skip_steps: dict[str, str],
    general_test_pattern: str = DEFAULT_TEST_PATTERN,
    general_test_parallelism: int = 3,
    multichip_test_pattern: str = DEFAULT_TEST_PATTERN,
) -> str:
    """Injects `skip:` and narrowed `test-file-pattern:` into a pipeline YAML."""
    if (
        not skip_steps
        and general_test_pattern == DEFAULT_TEST_PATTERN
        and general_test_parallelism == 3
        and multichip_test_pattern == DEFAULT_TEST_PATTERN
    ):
        return yaml_text

    step_start_re = re.compile(r"^[ \t]*-[ \t]+(?:label|group|wait|block):")
    key_re = re.compile(r'^(?P<indent>[ \t]*)key:[ \t]*"(?P<key>[^"]+)"[ \t]*$')
    parallelism_re = re.compile(r"^(?P<indent>[ \t]*)parallelism:[ \t]*\d+[ \t]*$")
    plugin_defaults_re = re.compile(
        r"^(?P<indent>[ \t]*)<<:[ \t]*\*tests_plugin_defaults[ \t]*$"
    )

    current_key: str | None = None
    out_lines: list[str] = []

    for line in yaml_text.splitlines(keepends=True):
        stripped = line.rstrip("\r\n")
        eol = line[len(stripped) :] or "\n"

        if step_start_re.match(stripped):
            current_key = None

        m_key = key_re.match(stripped)
        if m_key:
            current_key = m_key.group("key")
            out_lines.append(line)
            if current_key in skip_steps:
                indent = m_key.group("indent")
                reason = skip_steps[current_key][:70]
                out_lines.append(f"{indent}skip: {json.dumps(reason)}{eol}")
            continue

        if current_key == "unit_tests_general_pr" and general_test_parallelism != 3:
            m_par = parallelism_re.match(stripped)
            if m_par:
                indent = m_par.group("indent")
                out_lines.append(
                    f"{indent}parallelism: {general_test_parallelism}{eol}"
                )
                continue

        if current_key in {"unit_tests_general_pr", "unit_tests_multichip_pr"}:
            m_plug = plugin_defaults_re.match(stripped)
            if m_plug:
                out_lines.append(line)
                pattern = (
                    general_test_pattern
                    if current_key == "unit_tests_general_pr"
                    else multichip_test_pattern
                )
                if pattern and pattern != DEFAULT_TEST_PATTERN:
                    indent = m_plug.group("indent")
                    out_lines.append(
                        f"{indent}test-file-pattern: {json.dumps(pattern)}{eol}"
                    )
                continue

        out_lines.append(line)

    return "".join(out_lines)


def format_shell(plan: SelectionPlan) -> str:
    """Formats SelectionPlan as bash assignments for `eval` in bootstrap.sh."""
    lines = [
        f"PR_SKIP_ALL_CI={shlex.quote('1' if plan.skip_all_ci else '0')}",
        f"PR_SKIP_ALL_REASON={shlex.quote(plan.skip_all_reason)}",
        f"PR_SELECTED_LANES={shlex.quote(' '.join(plan.lanes))}",
        f"OMNI_IMAGE_SKIP={shlex.quote(plan.omni_image_skip)}",
        f"PR_TEST_SELECTION_ANNOTATION={shlex.quote(plan.annotation)}",
        (
            "export PR_SKIP_STEPS_JSON="
            f"{shlex.quote(json.dumps(plan.skip_steps, sort_keys=True))}"
        ),
        (f"export GENERAL_TEST_FILE_PATTERN={shlex.quote(plan.general_test_pattern)}"),
        (
            "export GENERAL_TEST_PARALLELISM="
            f"{shlex.quote(str(plan.general_test_parallelism))}"
        ),
        (
            "export MULTICHIP_TEST_FILE_PATTERN="
            f"{shlex.quote(plan.multichip_test_pattern)}"
        ),
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--shell",
        action="store_true",
        help="Emit shell variable assignments from changed files on stdin.",
    )
    group.add_argument(
        "--filter-yaml",
        metavar="YAML_FILE",
        help="Filter a Buildkite pipeline YAML file using exported env vars.",
    )
    args = parser.parse_args()

    if args.filter_yaml:
        yaml_path = Path(args.filter_yaml)
        yaml_text = yaml_path.read_text(encoding="utf-8")
        raw_skip = os.environ.get("PR_SKIP_STEPS_JSON", "")
        skip_steps = json.loads(raw_skip) if raw_skip else {}
        general_pattern = os.environ.get(
            "GENERAL_TEST_FILE_PATTERN", DEFAULT_TEST_PATTERN
        )
        general_parallelism = int(os.environ.get("GENERAL_TEST_PARALLELISM", "3"))
        multichip_pattern = os.environ.get(
            "MULTICHIP_TEST_FILE_PATTERN", DEFAULT_TEST_PATTERN
        )
        sys.stdout.write(
            filter_yaml_content(
                yaml_text=yaml_text,
                skip_steps=skip_steps,
                general_test_pattern=general_pattern,
                general_test_parallelism=general_parallelism,
                multichip_test_pattern=multichip_pattern,
            )
        )
        return 0

    changed_files = [line.strip() for line in sys.stdin if line.strip()]
    plan = select_for_files(changed_files)
    print(format_shell(plan))
    return 0


if __name__ == "__main__":
    sys.exit(main())
