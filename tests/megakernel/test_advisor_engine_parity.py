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
"""Differential parity suite between Python advisor physics and JS workbench engine."""

from __future__ import annotations

import dataclasses
import json
import math
import os
import shutil
import signal
import time
from pathlib import Path
from typing import TypedDict, cast

import pytest

from tests.megakernel.test_advisor_bounds import (
    compute_expected_kimi_k3_decode_comm_us,
)
from vllm_torchtpu.megakernel import advisor
from vllm_torchtpu.megakernel.advisor import (
    constants,
    models,
    support,
    workbench_engine,
)
from vllm_torchtpu.megakernel.advisor.roofline import _build_scope_context


class _ScopeParityEntry(TypedDict):
    """Typed schema for one scope parity entry."""

    feasible: bool
    t_vmem_max: int
    n_splits: int
    selected_dense_slots: int
    selected_expert_slots: int
    raw_step_us: float
    step_bound_us: float
    step_floor_strict_us: float
    step_achievable_us: float
    step_achievable_upper_us: float
    w_step_strict_gb: float
    act_hbm_mb: float
    comm_us: float
    chain_us: float
    support: str
    blockers: list[str]
    expert_mode: str
    host_head_us: float
    break_even_host_head_delta_us: float


class _EvalParityResult(TypedDict):
    """Typed schema for model evaluation parity result."""

    invalid_reasons: list[str]
    fastest_scope: str
    recommended_scope: str
    scopes: dict[str, _ScopeParityEntry]


class _BatchScenarioItem(TypedDict):
    """Typed schema for one input scenario in parity batch."""

    only_validate: bool
    model_name: str
    wl: advisor.WorkloadSpec


# 0.01 us 2-decimal display quantum + 0.005 us JS/Python rounding slack.
DISPLAY_ROUND_TOL_US = 0.015


def _select_shipped_models() -> tuple[str, ...]:
    """Return all 3 shipped model presets spanning GQA, GDN, KDA, MLA, DSA, and MoE."""
    return (
        "qwen3.5-397b",
        "kimi-k3",
        "glm-5.2",
    )


def _valid_workloads_for_model(
    model_profile: advisor.ModelProfile,
) -> list[advisor.WorkloadSpec]:
    """Build 15 diverse valid WorkloadSpec regimes for one model profile."""
    has_moe = model_profile.num_moe_layers > 0 and model_profile.num_experts > 0
    tp_base_wide = 8
    tp_base_mid = 4
    tp_base = (
        tp_base_wide
        if model_profile.num_attention_heads % tp_base_wide == 0
        else tp_base_mid
    )
    ep_valid = (
        tp_base_wide
        if (
            has_moe
            and model_profile.num_experts % tp_base_wide == 0
            and tp_base >= tp_base_wide
        )
        else 1
    )
    dcp_val = tp_base_mid if tp_base >= tp_base_mid else 1
    return [
        advisor.WorkloadSpec(
            phase="decode", batch=4, isl=3072, osl=1024, tp=tp_base, ep=ep_valid
        ),
        advisor.WorkloadSpec(
            phase="decode",
            batch=16,
            isl=4096,
            osl=512,
            tp=tp_base,
            ep=ep_valid,
            dispatch_regime="zero_floor",
        ),
        advisor.WorkloadSpec(
            phase="decode",
            batch=32,
            isl=8192,
            osl=512,
            tp=tp_base,
            ep=ep_valid,
            dispatch_regime="compiled_xla",
        ),
        advisor.WorkloadSpec(
            phase="decode",
            batch=64,
            isl=16384,
            osl=512,
            tp=tp_base,
            ep=ep_valid,
            dispatch_regime="unfused_eager",
        ),
        advisor.WorkloadSpec(
            phase="decode",
            batch=8,
            isl=2048,
            osl=256,
            tp=tp_base,
            ep=ep_valid,
            moe_mode="grouped_on_chip",
        ),
        advisor.WorkloadSpec(
            phase="decode",
            batch=16,
            isl=4096,
            osl=1024,
            tp=tp_base,
            ep=ep_valid,
            dispatch_regime="compiled_xla",
        ),
        advisor.WorkloadSpec(
            phase="verify",
            batch=24,
            spec_tokens=5,
            isl=8192,
            osl=512,
            tp=tp_base,
            ep=ep_valid,
            moe_mode="grouped_on_chip",
        ),
        advisor.WorkloadSpec(
            phase="prefill",
            batch=1,
            isl=4096,
            osl=512,
            chunk_size=512,
            prefix_hit_rate=0.5,
            tp=tp_base,
            ep=ep_valid,
        ),
        advisor.WorkloadSpec(
            phase="prefill",
            batch=2,
            isl=8192,
            osl=512,
            pcp=2,
            tp=tp_base,
            ep=ep_valid,
            moe_mode="grouped_on_chip",
        ),
        advisor.WorkloadSpec(
            phase="mixed",
            batch=16,
            isl=4096,
            osl=2048,
            chunk_size=256,
            prefix_hit_rate=0.90,
            tp=tp_base,
            ep=ep_valid,
            moe_mode="grouped_on_chip",
        ),
        advisor.WorkloadSpec(
            phase="mixed",
            batch=12,
            isl=8192,
            osl=1024,
            pcp=2,
            dcp=dcp_val,
            tp=1 if dcp_val > 1 else tp_base,
            ep=1,
            moe_mode="grouped_on_chip",
        ),
        advisor.WorkloadSpec(
            phase="decode",
            batch=16,
            isl=4096,
            osl=512,
            tp=tp_base,
            pp=2,
            ep=ep_valid,
            vmem_alias=False,
        ),
        advisor.WorkloadSpec(
            phase="decode",
            batch=20,
            isl=4096,
            osl=512,
            tp=tp_base,
            ep=ep_valid,
            vmem_budget_mib=48.0,
            dma_buffers=3,
        ),
        advisor.WorkloadSpec(
            phase="decode",
            batch=4,
            isl=1024,
            osl=128,
            tp=tp_base,
            ep=ep_valid,
            vmem_fit_headroom=0.95,
        ),
        advisor.WorkloadSpec(
            phase="decode",
            batch=8,
            isl=4096,
            osl=512,
            tp=tp_base,
            ep=ep_valid,
            moe_mode="masked_on_chip",
        ),
    ]


def _invalid_sharding_scenarios() -> list[tuple[str, advisor.WorkloadSpec]]:
    """Build 6 invalid scenarios: 5 divisibility checks, 1 row limit below a request."""
    return [
        ("qwen3.5-397b", advisor.WorkloadSpec(phase="decode", batch=8, tp=6, ep=1)),
        ("qwen3.5-397b", advisor.WorkloadSpec(phase="decode", batch=8, tp=8, ep=7)),
        ("kimi-k3", advisor.WorkloadSpec(phase="decode", batch=8, tp=2, dp=1, ep=8)),
        ("qwen3.5-397b", advisor.WorkloadSpec(phase="decode", batch=8, tp=8, dcp=3)),
        ("qwen3.5-397b", advisor.WorkloadSpec(phase="decode", batch=8, tp=8, pp=512)),
        (
            "qwen3.5-397b",
            advisor.WorkloadSpec(
                phase="verify", batch=8, spec_tokens=3, tp=8, ep=8, max_rows_per_call=3
            ),
        ),
    ]


def _python_reference_eval(
    model_name: str,
    workload: advisor.WorkloadSpec,
    chip_spec: advisor.ChipSpec,
) -> _EvalParityResult:
    """Compute authoritative Python scopes and reasons using public advisor APIs."""
    model_profile = models.resolve_model_profile(model_name)
    reasons = list(advisor.validate_workload(model_profile, chip_spec, workload))
    report = advisor.advise(model_profile, workload, chip_spec)
    raw_steps = advisor.compute_scope_raw_step_us(
        model_profile, workload, chip=chip_spec.name
    )
    scopes_out: dict[str, _ScopeParityEntry] = {}
    for scope in report.scopes:
        scopes_out[scope.scope] = {
            "feasible": scope.feasible,
            "t_vmem_max": scope.t_vmem_max,
            "n_splits": scope.n_splits,
            "selected_dense_slots": scope.selected_dense_slots,
            "selected_expert_slots": scope.selected_expert_slots,
            "raw_step_us": raw_steps.get(scope.scope, 0.0),
            "step_bound_us": scope.step_bound_us,
            "step_floor_strict_us": scope.step_floor_strict_us,
            "step_achievable_us": scope.step_achievable_us,
            "step_achievable_upper_us": scope.step_achievable_upper_us,
            "w_step_strict_gb": scope.w_step_strict_gb,
            "act_hbm_mb": scope.act_hbm_mb,
            "comm_us": scope.comm_us,
            "chain_us": scope.chain_us,
            "support": scope.support,
            "blockers": list(scope.blockers),
            "expert_mode": scope.expert_mode,
            "host_head_us": scope.host_head_us,
            "break_even_host_head_delta_us": (scope.break_even_host_head_delta_us),
        }
    return {
        "invalid_reasons": reasons,
        "fastest_scope": report.fastest_scope,
        "recommended_scope": report.recommended_scope,
        "scopes": scopes_out,
    }


def test_workbench_engine_asset_and_seed_contract() -> None:
    """Verify static JS engine asset carries cleanroom physics API and license."""
    engine_js = workbench_engine.render_workbench_engine_js()
    assert "// Copyright 2026 Google LLC" in engine_js
    assert "Licensed under the Apache License, Version 2.0" in engine_js
    assert "createAdvisorEngine" in engine_js
    assert "__MK_CREATE_ADVISOR_ENGINE" in engine_js

    for symbol in (
        "computePerTokenVmem",
        "evaluateWorkload",
        "allReduceCost",
        "expertsHit",
        "selectAdaptiveRing",
        "validateWorkload",
        "roundEven",
    ):
        assert symbol in engine_js, f"Missing symbol: {symbol}"

    # Verify no disallowed remote schemes or hardcoded constants in engine body
    # (excluding the standard Apache license URL in the header).
    body_lines = [
        line
        for line in engine_js.splitlines()
        if "http://www.apache.org/licenses/LICENSE-2.0" not in line
    ]
    body_js = "\n".join(body_lines)
    for banned in (
        "3.7e12",
        "3e11",
        "1e11",
        "2.3e15",
        "1.155e15",
        "103e9",
        "https://",
        "http://",
    ):
        assert banned not in body_js, f"Banned literal: {banned}"


def _build_parity_payloads(
    chip_spec: advisor.ChipSpec,
) -> tuple[
    list[_BatchScenarioItem],
    list[_EvalParityResult],
    list[dict[str, object]],
]:
    """Assemble reference evaluations and payload items for parity run."""
    chip_dict = dataclasses.asdict(chip_spec)
    models_sample = _select_shipped_models()
    batch_items: list[_BatchScenarioItem] = []
    expected_refs: list[_EvalParityResult] = []
    payload_items: list[dict[str, object]] = []

    for model_name in models_sample:
        model_profile = models.resolve_model_profile(model_name)
        prof_dict = dataclasses.asdict(model_profile)
        for workload in _valid_workloads_for_model(model_profile):
            batch_items.append(
                {"only_validate": False, "model_name": model_name, "wl": workload}
            )
            expected_refs.append(
                _python_reference_eval(model_name, workload, chip_spec)
            )
            payload_items.append(
                {
                    "params": {
                        "only_validate": False,
                        "profile": prof_dict,
                        "chip": chip_dict,
                        "wl": dataclasses.asdict(workload),
                    }
                }
            )

    for model_name, bad_wl in _invalid_sharding_scenarios():
        model_profile = models.resolve_model_profile(model_name)
        prof_dict = dataclasses.asdict(model_profile)
        inv_reasons = list(advisor.validate_workload(model_profile, chip_spec, bad_wl))
        assert len(inv_reasons) >= 1
        batch_items.append(
            {"only_validate": True, "model_name": model_name, "wl": bad_wl}
        )
        expected_refs.append(
            {
                "invalid_reasons": inv_reasons,
                "fastest_scope": "per_op",
                "recommended_scope": "per_op",
                "scopes": {},
            }
        )
        payload_items.append(
            {
                "params": {
                    "only_validate": True,
                    "profile": prof_dict,
                    "chip": chip_dict,
                    "wl": dataclasses.asdict(bad_wl),
                }
            }
        )
    return batch_items, expected_refs, payload_items


def _run_engine_runner(
    js_runtime: str,
    runner_path: Path,
    in_json: Path,
    out_json: Path,
    timeout_s: float = 60.0,
) -> None:
    """Execute advisor engine runner script using POSIX spawn without shell."""
    argv = [js_runtime, str(runner_path), str(in_json), str(out_json)]
    pid = os.posix_spawnp(js_runtime, argv, os.environ)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        wpid, status = os.waitpid(pid, os.WNOHANG)
        if wpid == pid:
            rc = os.waitstatus_to_exitcode(status)
            if rc != 0:
                msg = f"Engine runner failed with return code {rc}"
                raise RuntimeError(msg)
            return
        time.sleep(0.05)
    os.kill(pid, signal.SIGKILL)
    os.waitpid(pid, 0)
    msg = f"Engine runner timed out after {timeout_s}s"
    raise TimeoutError(msg)


def _verify_parity_records(
    batch_items: list[_BatchScenarioItem],
    expected_refs: list[_EvalParityResult],
    out_records: list[dict[str, object]],
) -> None:
    """Verify each output record against authoritative reference evaluation."""
    for idx, (item, exp, rec) in enumerate(
        zip(batch_items, expected_refs, out_records, strict=True)
    ):
        label = f"scenario[{idx}] model={item['model_name']}"
        got = rec["eval"]
        if item["only_validate"]:
            assert list(got["invalid_reasons"]) == list(exp["invalid_reasons"]), label
            continue

        assert list(got["invalid_reasons"]) == list(exp["invalid_reasons"]), label
        full_got = got["full"]
        assert full_got["fastestScope"] == exp["fastest_scope"], label
        assert full_got["recScope"] == exp["recommended_scope"], label
        for scope_name, exp_sc in exp["scopes"].items():
            got_sc = got["scopes"][scope_name]
            sc_lbl = f"{label} scope={scope_name}"
            assert got_sc["feasible"] is exp_sc["feasible"], sc_lbl
            assert got_sc["t_vmem_max"] == exp_sc["t_vmem_max"], sc_lbl
            assert got_sc["n_splits"] == exp_sc["n_splits"], sc_lbl
            assert got_sc["selected_dense_slots"] == exp_sc["selected_dense_slots"], (
                sc_lbl
            )
            assert got_sc["selected_expert_slots"] == exp_sc["selected_expert_slots"], (
                sc_lbl
            )
            assert got_sc["support"] == exp_sc["support"], sc_lbl
            assert list(got_sc["blockers"]) == list(exp_sc["blockers"]), sc_lbl
            assert got_sc["expert_mode"] == exp_sc["expert_mode"], sc_lbl
            assert got_sc["host_head_us"] == pytest.approx(
                exp_sc["host_head_us"], abs=0.015
            ), f"{sc_lbl} field=host_head_us"
            assert got_sc["break_even_host_head_delta_us"] == pytest.approx(
                exp_sc["break_even_host_head_delta_us"], abs=0.02
            ), f"{sc_lbl} field=break_even_host_head_delta_us"
            assert got_sc["rawStepUs"] == pytest.approx(
                exp_sc["raw_step_us"], rel=1e-6, abs=1e-9
            ), f"{sc_lbl} field=rawStepUs"
            assert got_sc["w_step_strict_gb"] == pytest.approx(
                exp_sc["w_step_strict_gb"], abs=1e-4
            ), f"{sc_lbl} field=w_step_strict_gb"
            for disp_key in (
                "step_bound_us",
                "step_floor_strict_us",
                "step_achievable_us",
                "step_achievable_upper_us",
                "act_hbm_mb",
                "comm_us",
                "chain_us",
            ):
                assert got_sc[disp_key] == pytest.approx(exp_sc[disp_key], abs=0.015), (
                    f"{sc_lbl} field={disp_key}"
                )
            assert got_sc["step_floor_strict_us"] <= got_sc["step_bound_us"]
            assert (
                got_sc["step_bound_us"]
                <= got_sc["step_achievable_us"]
                <= got_sc["step_achievable_upper_us"]
            )


def test_js_engine_runner_executes_static_asset_parity(tmp_path: Path) -> None:
    """Verify exact Python vs JS parity across 80 scenarios via static engine runner."""
    js_runtime = shutil.which("node") or shutil.which("gjs")
    if js_runtime is None:
        msg = "Neither node nor gjs runtime found; install nodejs"
        raise RuntimeError(msg)

    runner_path = Path(__file__).with_name("advisor_engine_runner.js")
    assert runner_path.is_file(), f"Missing runner at {runner_path}"

    engine_path = Path(models.__file__).parent / "assets" / "workbench_engine.js"
    assert engine_path.is_file(), f"Missing engine at {engine_path}"

    chip_spec = advisor.resolve_chip_spec("tpu7x")
    batch_items, expected_refs, payload_items = _build_parity_payloads(chip_spec)
    expected_scenario_count = 51  # 3 shipped models x 15 valid + 6 invalid
    assert len(batch_items) == expected_scenario_count

    in_json = tmp_path / "parity_in.json"
    out_json = tmp_path / "parity_out.json"
    in_json.write_text(
        json.dumps(
            {
                "engine_path": str(engine_path),
                "seed_data": {
                    "chip_spec": dataclasses.asdict(chip_spec),
                    **workbench_engine.build_engine_seed_defaults(),
                },
                "items": payload_items,
            }
        ),
        encoding="utf-8",
    )

    _run_engine_runner(js_runtime, runner_path, in_json, out_json)

    out_records = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(out_records) == len(expected_refs)
    _verify_parity_records(batch_items, expected_refs, out_records)


def test_js_engine_intra_host_dp_and_pp_prefill_parity(tmp_path: Path) -> None:
    """Verify JS engine parity on intra-host DPExchange, DCN EP, and PP prefill."""
    js_runtime = shutil.which("node") or shutil.which("gjs")
    if js_runtime is None:
        msg = "Neither node nor gjs runtime found; install nodejs"
        raise RuntimeError(msg)

    runner_path = Path(__file__).with_name("advisor_engine_runner.js")
    engine_path = Path(models.__file__).parent / "assets" / "workbench_engine.js"
    chip_spec = advisor.resolve_chip_spec("tpu7x")
    chip_dict = dataclasses.asdict(chip_spec)
    model_profile = models.resolve_model_profile("qwen3.5-397b")
    prof_dict = dataclasses.asdict(model_profile)

    workloads = [
        advisor.WorkloadSpec(
            phase="decode",
            batch=384,
            context=32768,
            tp=2,
            dp=4,
            ep=8,
            moe_mode="grouped_on_chip",
        ),
        advisor.WorkloadSpec(
            phase="decode",
            batch=384,
            context=32768,
            tp=2,
            dp=4,
            ep=8,
            moe_mode="masked_on_chip",
        ),
        advisor.WorkloadSpec(
            phase="decode",
            batch=768,
            context=32768,
            tp=2,
            dp=8,
            ep=16,
            moe_mode="grouped_on_chip",
        ),
        advisor.WorkloadSpec(
            phase="prefill",
            batch=1,
            isl=4096,
            tp=1,
            pp=8,
            ep=1,
            chunk_size=0,
            microbatches=0,
        ),
    ]
    items = [
        {
            "params": {
                "only_validate": False,
                "profile": prof_dict,
                "chip": chip_dict,
                "wl": dataclasses.asdict(workload),
            }
        }
        for workload in workloads
    ]
    in_json = tmp_path / "dp_pp_in.json"
    out_json = tmp_path / "dp_pp_out.json"
    in_json.write_text(
        json.dumps(
            {
                "engine_path": str(engine_path),
                "seed_data": {
                    "chip_spec": chip_dict,
                    **workbench_engine.build_engine_seed_defaults(),
                },
                "items": items,
            }
        ),
        encoding="utf-8",
    )
    _run_engine_runner(js_runtime, runner_path, in_json, out_json)
    records = json.loads(out_json.read_text(encoding="utf-8"))

    for workload, rec in zip(workloads, records, strict=True):
        report = advisor.advise(model_profile, workload, chip_spec)
        full = rec["eval"]["full"]
        assert full["activeTokens"] == report.active_tokens
        assert full["recScope"] == report.recommended_scope
        assert full["moeModeRec"] == report.recommended_moe_mode
        assert full["compilerProofGainPct"] == pytest.approx(
            float(report.metrics["compiler_proof_gain_pct"]["value"]),
            abs=0.015,
        )
        assert full["achievableGainPct"] == pytest.approx(
            float(report.metrics["achievable_gain_pct"]["value"]),
            abs=0.015,
        )


def test_js_engine_support_gating_parity(tmp_path: Path) -> None:
    """Verify JS support gating, blockers, strict floor, and recScope parity."""
    js_runtime = shutil.which("node") or shutil.which("gjs")
    if js_runtime is None:
        msg = "Neither node nor gjs runtime found; install nodejs"
        raise RuntimeError(msg)

    seed_defaults = workbench_engine.build_engine_seed_defaults()
    assert "core_limits" in seed_defaults
    assert "workload_defaults" in seed_defaults

    runner_path = Path(__file__).with_name("advisor_engine_runner.js")
    engine_path = Path(models.__file__).parent / "assets" / "workbench_engine.js"
    chip_spec = advisor.resolve_chip_spec("tpu7x")
    chip_dict = dataclasses.asdict(chip_spec)

    qwen_prof = models.resolve_model_profile("qwen3.5-397b")
    kimi_prof = models.resolve_model_profile("kimi-k3")
    glm_prof = models.resolve_model_profile("glm-5.2")

    cases = [
        (qwen_prof, advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=1)),
        (qwen_prof, advisor.WorkloadSpec(phase="decode", batch=256, tp=8, ep=1)),
        (kimi_prof, advisor.WorkloadSpec(phase="decode", batch=64, tp=4, dp=8, ep=32)),
        (glm_prof, advisor.WorkloadSpec(phase="decode", batch=16, tp=16, ep=1)),
        (qwen_prof, advisor.WorkloadSpec(phase="decode", batch=16, tp=6, ep=1)),
    ]
    items = [
        {
            "params": {
                "only_validate": False,
                "profile": dataclasses.asdict(model_profile),
                "chip": chip_dict,
                "wl": dataclasses.asdict(workload),
            }
        }
        for model_profile, workload in cases
    ]
    in_json = tmp_path / "support_in.json"
    out_json = tmp_path / "support_out.json"
    in_json.write_text(
        json.dumps(
            {
                "engine_path": str(engine_path),
                "seed_data": {"chip_spec": chip_dict, **seed_defaults},
                "items": items,
            }
        ),
        encoding="utf-8",
    )
    _run_engine_runner(js_runtime, runner_path, in_json, out_json)
    records = json.loads(out_json.read_text(encoding="utf-8"))

    for (model_profile, workload), rec in zip(cases, records, strict=True):
        report = advisor.advise(model_profile, workload, chip_spec)
        got = rec["eval"]
        full = got["full"]
        assert full["fastestScope"] == report.fastest_scope
        assert full["recScope"] == report.recommended_scope
        for scope in report.scopes:
            got_sc = got["scopes"][scope.scope]
            assert got_sc["support"] == scope.support
            assert list(got_sc["blockers"]) == list(scope.blockers)
            assert got_sc["step_floor_strict_us"] == pytest.approx(
                scope.step_floor_strict_us, abs=0.015
            )
            assert got_sc["step_achievable_upper_us"] == pytest.approx(
                scope.step_achievable_upper_us, abs=0.015
            )


def test_engine_parity_max_rows_per_call(tmp_path: Path) -> None:
    """Verify JS engine parity under a single-call row limit, one call and split."""
    js_runtime = shutil.which("node") or shutil.which("gjs")
    if js_runtime is None:
        msg = "Neither node nor gjs runtime found; install nodejs"
        raise RuntimeError(msg)

    runner_path = Path(__file__).with_name("advisor_engine_runner.js")
    engine_path = Path(models.__file__).parent / "assets" / "workbench_engine.js"
    chip_spec = advisor.resolve_chip_spec("tpu7x")
    chip_dict = dataclasses.asdict(chip_spec)
    model_name = "qwen3.5-397b"
    prof_dict = dataclasses.asdict(models.resolve_model_profile(model_name))

    # Verify requests hold 4 rows: one call, exactly full, full plus remainder, a
    # limit between request boundaries, and two full calls. Then a decode split.
    verify_cases = ((16, 80), (20, 80), (22, 80), (28, 81), (40, 80))
    workloads = [
        advisor.WorkloadSpec(
            phase="verify",
            batch=batch,
            spec_tokens=3,
            context=4096,
            tp=8,
            dcp=4,
            ep=8,
            max_rows_per_call=limit,
        )
        for batch, limit in verify_cases
    ]
    workloads.append(
        advisor.WorkloadSpec(
            phase="decode", batch=100, tp=8, ep=8, max_rows_per_call=32
        )
    )
    batch_items: list[_BatchScenarioItem] = [
        {"only_validate": False, "model_name": model_name, "wl": workload}
        for workload in workloads
    ]
    expected = [
        _python_reference_eval(model_name, workload, chip_spec)
        for workload in workloads
    ]
    calls = [exp["scopes"]["whole_step"]["n_splits"] for exp in expected]
    assert calls[:2] == [1, 1]
    assert all(n > 1 for n in calls[2:])

    in_json = tmp_path / "row_limit_in.json"
    out_json = tmp_path / "row_limit_out.json"
    in_json.write_text(
        json.dumps(
            {
                "engine_path": str(engine_path),
                "seed_data": {
                    "chip_spec": chip_dict,
                    **workbench_engine.build_engine_seed_defaults(),
                },
                "items": [
                    {
                        "params": {
                            "only_validate": False,
                            "profile": prof_dict,
                            "chip": chip_dict,
                            "wl": dataclasses.asdict(workload),
                        }
                    }
                    for workload in workloads
                ],
            }
        ),
        encoding="utf-8",
    )
    _run_engine_runner(js_runtime, runner_path, in_json, out_json)
    records = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(records) == len(expected)
    _verify_parity_records(batch_items, expected, records)


def test_engine_parity_declared_preset_name_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify JS finds a declared preset whose key differs in case and separators."""
    js_runtime = shutil.which("node") or shutil.which("gjs")
    if js_runtime is None:
        msg = "Neither node nor gjs runtime found; install nodejs"
        raise RuntimeError(msg)

    runner_path = Path(__file__).with_name("advisor_engine_runner.js")
    engine_path = Path(models.__file__).parent / "assets" / "workbench_engine.js"
    chip_spec = advisor.resolve_chip_spec("tpu7x")
    chip_dict = dataclasses.asdict(chip_spec)
    model_name = "qwen3.5-397b"
    prof_dict = dataclasses.asdict(models.resolve_model_profile(model_name))
    dep = models.DEPLOYMENT_PRESETS[model_name]
    declaration = {
        "source": "unit test",
        "scopes": ["whole_step"],
        "max_rows": constants.DEFAULT_ROW_LIMIT,
        "max_total_rows": 2 * constants.DEFAULT_ROW_LIMIT,
        "ranks": dep.decode.tp,
    }
    declared = dataclasses.replace(
        dep,
        decode=dataclasses.replace(dep.decode, megakernel=declaration),
        megakernel_layouts=(),
    )
    presets = {"Qwen3_5-397B": declared}
    monkeypatch.setattr(support, "DEPLOYMENT_PRESETS", presets)
    monkeypatch.setattr(workbench_engine, "DEPLOYMENT_PRESETS", presets)

    # Verify requests hold 4 rows: one call under the declared limit, then a split.
    workloads = [
        advisor.WorkloadSpec(
            phase="verify", batch=batch, spec_tokens=3, context=4096, tp=8, dcp=4, ep=8
        )
        for batch in (16, 28)
    ]
    batch_items: list[_BatchScenarioItem] = [
        {"only_validate": False, "model_name": model_name, "wl": workload}
        for workload in workloads
    ]
    expected = [
        _python_reference_eval(model_name, workload, chip_spec)
        for workload in workloads
    ]
    whole_steps = [exp["scopes"]["whole_step"] for exp in expected]
    assert [ws["support"] for ws in whole_steps] == ["declared", "declared"]
    assert [ws["t_vmem_max"] for ws in whole_steps] == [constants.DEFAULT_ROW_LIMIT] * 2

    in_json = tmp_path / "declared_in.json"
    out_json = tmp_path / "declared_out.json"
    in_json.write_text(
        json.dumps(
            {
                "engine_path": str(engine_path),
                "seed_data": {
                    "chip_spec": chip_dict,
                    **workbench_engine.build_engine_seed_defaults(),
                },
                "items": [
                    {
                        "params": {
                            "only_validate": False,
                            "profile": prof_dict,
                            "chip": chip_dict,
                            "wl": dataclasses.asdict(workload),
                        }
                    }
                    for workload in workloads
                ],
            }
        ),
        encoding="utf-8",
    )
    _run_engine_runner(js_runtime, runner_path, in_json, out_json)
    records = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(records) == len(expected)
    _verify_parity_records(batch_items, expected, records)


def test_js_engine_physical_validation_vmem_overflow_and_tps_parity(
    tmp_path: Path,
) -> None:
    """Verify physical validation rules, per_op VMEM overflow, and recWinner TPS."""
    js_runtime = shutil.which("node") or shutil.which("gjs")
    if js_runtime is None:
        msg = "Neither node nor gjs runtime found; install nodejs"
        raise RuntimeError(msg)

    runner_path = Path(__file__).with_name("advisor_engine_runner.js")
    engine_path = Path(models.__file__).parent / "assets" / "workbench_engine.js"
    chip_spec = advisor.resolve_chip_spec("tpu7x")
    chip_dict = dataclasses.asdict(chip_spec)
    qwen_prof = models.resolve_model_profile("Qwen/Qwen3.5-397B-A17B-FP8")
    kv_less_prof = dataclasses.replace(qwen_prof, num_kv_heads=0)

    invalid_cases: list[tuple[advisor.ModelProfile, advisor.WorkloadSpec]] = [
        (qwen_prof, advisor.WorkloadSpec(phase="decode", batch=8, tp=2, ep=1)),
        (qwen_prof, advisor.WorkloadSpec(phase="decode", batch=8, tp=8, dp=3)),
        (qwen_prof, advisor.WorkloadSpec(phase="decode", batch=8, tp=8, pcp=3)),
        (qwen_prof, advisor.WorkloadSpec(phase="decode", batch=4, tp=8, dp=8)),
        (qwen_prof, advisor.WorkloadSpec(phase="decode", batch=8, tp=8, pp=2, pcp=2)),
        (qwen_prof, advisor.WorkloadSpec(phase="verify", batch=8, spec_tokens=0, tp=8)),
        (kv_less_prof, advisor.WorkloadSpec(phase="decode", batch=8, tp=8, kv_split=2)),
    ]
    eval_cases: list[tuple[advisor.ModelProfile, advisor.WorkloadSpec]] = [
        (
            qwen_prof,
            advisor.WorkloadSpec(phase="decode", batch=16, tp=2),
        ),
        (
            models.resolve_model_profile("kimi-k3"),
            advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8),
        ),
        (
            models.resolve_model_profile("qwen3.5-397b"),
            advisor.WorkloadSpec(phase="decode", batch=16, tp=8, vmem_budget_mib=2.0),
        ),
        (
            models.resolve_model_profile("qwen3.5-397b"),
            advisor.WorkloadSpec(
                phase="mixed",
                batch=8,
                spec_tokens=3,
                chunk_size=64,
                tp=8,
                ep=8,
                dispatch_regime="compiled_xla",
            ),
        ),
    ]

    items = [
        {
            "params": {
                "only_validate": True,
                "profile": dataclasses.asdict(p),
                "chip": chip_dict,
                "wl": dataclasses.asdict(workload),
            }
        }
        for p, workload in invalid_cases
    ] + [
        {
            "params": {
                "only_validate": False,
                "profile": dataclasses.asdict(p),
                "chip": chip_dict,
                "wl": dataclasses.asdict(workload),
            }
        }
        for p, workload in eval_cases
    ]

    in_json = tmp_path / "edge_parity_in.json"
    out_json = tmp_path / "edge_parity_out.json"
    in_json.write_text(
        json.dumps(
            {
                "engine_path": str(engine_path),
                "seed_data": {
                    "chip_spec": chip_dict,
                    **workbench_engine.build_engine_seed_defaults(),
                },
                "items": items,
            }
        ),
        encoding="utf-8",
    )
    _run_engine_runner(js_runtime, runner_path, in_json, out_json)
    records = json.loads(out_json.read_text(encoding="utf-8"))

    for idx, (p, workload) in enumerate(invalid_cases):
        exp_reasons = list(advisor.validate_workload(p, chip_spec, workload))
        got_reasons = list(records[idx]["eval"]["invalid_reasons"])
        assert got_reasons == exp_reasons, f"invalid_case[{idx}] {workload}"

    offset = len(invalid_cases)
    for idx, (p, workload) in enumerate(eval_cases):
        report = advisor.advise(p, workload, chip_spec)
        got = records[offset + idx]["eval"]
        full = got["full"]
        assert full["recScope"] == report.recommended_scope
        assert full["tpsChip"] == pytest.approx(
            float(report.metrics["throughput_tok_s_per_chip"]["value"]),
            abs=0.015,
        )
        per_op_py = next(s for s in report.scopes if s.scope == "per_op")
        assert got["scopes"]["per_op"]["feasible"] is per_op_py.feasible
        if workload.phase == "mixed":
            assert full["activeTokens"] == report.active_tokens
            assert got["scopes"]["whole_step"]["step_bound_us"] == pytest.approx(
                report.scopes[-1].step_bound_us, abs=0.015
            )


def _assert_measured_collective_js_parity(
    py_reports: list[advisor.AdvisorReport],
    records: list[dict[str, object]],
    tolerance_abs: float,
) -> None:
    """Assert Python/JS scope parity across measured-collective arms."""
    for py_report, record in zip(py_reports, records, strict=True):
        eval_payload = cast("dict[str, object]", record["eval"])
        js_scopes = cast("dict[str, dict[str, float]]", eval_payload["scopes"])
        for scope_name in ("per_op", "whole_step"):
            py_scope = next(s for s in py_report.scopes if s.scope == scope_name)
            js_scope = js_scopes[scope_name]
            assert js_scope["comm_us"] == pytest.approx(
                py_scope.comm_us, abs=tolerance_abs
            )
            assert js_scope["step_floor_strict_us"] == pytest.approx(
                py_scope.step_floor_strict_us, abs=tolerance_abs
            )
            assert js_scope["step_bound_us"] == pytest.approx(
                py_scope.step_bound_us, abs=tolerance_abs
            )
            assert js_scope["step_upper_us"] == pytest.approx(
                py_scope.step_upper_us, abs=tolerance_abs
            )
            assert js_scope["step_achievable_us"] == pytest.approx(
                py_scope.step_achievable_us, abs=tolerance_abs
            )
            assert js_scope["step_achievable_upper_us"] == pytest.approx(
                py_scope.step_achievable_upper_us, abs=tolerance_abs
            )


def _assert_measured_collective_invariants(
    py_reports: list[advisor.AdvisorReport],
    model_profile: advisor.ModelProfile,
    workload: advisor.WorkloadSpec,
    chip_spec: advisor.ChipSpec,
    tolerance_abs: float,
) -> None:
    """Assert floors stay analytical while achievable steps apply measured AR calls."""
    num_all_reduce_calls = model_profile.ar_slots_per_layer * model_profile.num_layers
    eta_ici = workload.efficiency.eta_ici
    ar_call_factor_extra_us = (
        num_all_reduce_calls
        * (advisor.constants.PER_OP_COLLECTIVE_CALL_FACTOR - 1)
        * round(math.log2(workload.tp))
        * chip_spec.collective_latency_s
        * 1e6
    )
    baseline_per_op = next(s for s in py_reports[0].scopes if s.scope == "per_op")
    baseline_whole_step = next(
        s for s in py_reports[0].scopes if s.scope == "whole_step"
    )

    xla_mean_rate_us = 18.0
    arm1_per_op = next(s for s in py_reports[1].scopes if s.scope == "per_op")
    arm1_whole = next(s for s in py_reports[1].scopes if s.scope == "whole_step")
    assert arm1_per_op.comm_us == baseline_per_op.comm_us
    assert arm1_per_op.step_bound_us == baseline_per_op.step_bound_us
    assert arm1_per_op.step_floor_strict_us == baseline_per_op.step_floor_strict_us
    assert arm1_per_op.step_upper_us == baseline_per_op.step_upper_us
    assert arm1_per_op.step_achievable_us == pytest.approx(
        baseline_per_op.step_achievable_us
        - (baseline_per_op.comm_us + ar_call_factor_extra_us) / eta_ici
        + num_all_reduce_calls * xla_mean_rate_us,
        abs=tolerance_abs,
    )
    assert arm1_whole.comm_us == baseline_whole_step.comm_us
    assert arm1_whole.step_achievable_us == baseline_whole_step.step_achievable_us

    in_kernel_mean_rate_us = 11.0
    arm2_per_op = next(s for s in py_reports[2].scopes if s.scope == "per_op")
    arm2_whole = next(s for s in py_reports[2].scopes if s.scope == "whole_step")
    assert arm2_whole.comm_us == baseline_whole_step.comm_us
    assert arm2_whole.step_bound_us == baseline_whole_step.step_bound_us
    assert arm2_whole.step_floor_strict_us == baseline_whole_step.step_floor_strict_us
    assert arm2_whole.step_upper_us == baseline_whole_step.step_upper_us
    assert arm2_whole.step_achievable_us == pytest.approx(
        baseline_whole_step.step_achievable_us
        - baseline_whole_step.comm_us / eta_ici
        + num_all_reduce_calls * in_kernel_mean_rate_us,
        abs=tolerance_abs,
    )
    assert arm2_per_op.comm_us == baseline_per_op.comm_us
    assert arm2_per_op.step_achievable_us == baseline_per_op.step_achievable_us

    arm3_per_op = next(s for s in py_reports[3].scopes if s.scope == "per_op")
    arm3_whole = next(s for s in py_reports[3].scopes if s.scope == "whole_step")
    assert arm3_per_op.comm_us == baseline_per_op.comm_us
    assert arm3_whole.comm_us == baseline_whole_step.comm_us
    assert arm3_per_op.step_achievable_us == baseline_per_op.step_achievable_us
    assert arm3_whole.step_achievable_us == baseline_whole_step.step_achievable_us


def test_engine_parity_measured_collective(tmp_path: Path) -> None:
    """Verify parity between Python and JS engines for calibrated collective calls."""
    js_runtime = shutil.which("node") or shutil.which("gjs")
    if js_runtime is None:
        msg = "Neither node nor gjs runtime found; install nodejs"
        raise RuntimeError(msg)

    runner_path = Path(__file__).with_name("advisor_engine_runner.js")
    engine_path = Path(models.__file__).parent / "assets" / "workbench_engine.js"
    chip_spec = advisor.resolve_chip_spec("tpu7x")
    chip_dict = dataclasses.asdict(chip_spec)
    model_profile = advisor.resolve_model_profile("qwen3.5-397b")
    prof_dict = dataclasses.asdict(model_profile)

    workload = advisor.WorkloadSpec(phase="decode", batch=16, context=4096, tp=8, ep=1)

    measurement_cases = [
        None,
        advisor.MeasurementsData(
            source_desc="xla_coll_trace",
            source_kind="trace",
            collective_calls=(
                advisor.CollectiveCall(
                    path="xla",
                    op="all-reduce",
                    group_size=8,
                    latency_us=15.0,
                    mean_latency_us=18.0,
                    n=10,
                ),
            ),
        ),
        advisor.MeasurementsData(
            source_desc="in_kernel_coll_trace",
            source_kind="trace",
            collective_calls=(
                advisor.CollectiveCall(
                    path="in_kernel",
                    op="all-reduce",
                    group_size=8,
                    latency_us=9.0,
                    mean_latency_us=11.0,
                    n=10,
                ),
            ),
        ),
        advisor.MeasurementsData(
            source_desc="mismatched_coll_trace",
            source_kind="trace",
            collective_calls=(
                advisor.CollectiveCall(
                    path="xla",
                    op="all-reduce",
                    group_size=4,
                    latency_us=25.0,
                    n=10,
                ),
            ),
        ),
    ]

    py_reports = [
        advisor.advise(model_profile, workload, chip_spec, measurements=meas_case)
        for meas_case in measurement_cases
    ]

    js_items = []
    for meas_case in measurement_cases:
        wl_dict = dataclasses.asdict(workload)
        if meas_case is not None:
            wl_dict["collective_calls"] = [
                dataclasses.asdict(c) for c in meas_case.collective_calls
            ]
        js_items.append(
            {
                "params": {
                    "only_validate": False,
                    "profile": prof_dict,
                    "chip": chip_dict,
                    "wl": wl_dict,
                }
            }
        )

    in_json = tmp_path / "measured_coll_parity_in.json"
    out_json = tmp_path / "measured_coll_parity_out.json"
    in_json.write_text(
        json.dumps(
            {
                "engine_path": str(engine_path),
                "seed_data": {
                    "chip_spec": chip_dict,
                    **workbench_engine.build_engine_seed_defaults(),
                },
                "items": js_items,
            }
        ),
        encoding="utf-8",
    )
    _run_engine_runner(js_runtime, runner_path, in_json, out_json)
    records = json.loads(out_json.read_text(encoding="utf-8"))

    tolerance_abs = 0.05
    _assert_measured_collective_js_parity(py_reports, records, tolerance_abs)
    _assert_measured_collective_invariants(
        py_reports, model_profile, workload, chip_spec, tolerance_abs
    )


def test_engine_parity_layout_declaration(tmp_path: Path) -> None:
    """Verify JS and Python parity on layout declarations across serving meshes."""
    js_runtime = shutil.which("node") or shutil.which("gjs")
    if js_runtime is None:
        msg = "Neither node nor gjs runtime found; install nodejs"
        raise RuntimeError(msg)

    runner_path = Path(__file__).with_name("advisor_engine_runner.js")
    engine_path = Path(models.__file__).parent / "assets" / "workbench_engine.js"
    chip_spec = advisor.resolve_chip_spec("tpu7x")
    chip_dict = dataclasses.asdict(chip_spec)
    model_name = "qwen3.5-397b"
    prof_dict = dataclasses.asdict(models.resolve_model_profile(model_name))

    workloads = [
        advisor.WorkloadSpec(phase="decode", batch=16, context=4096, tp=8, ep=8, dcp=1),
        advisor.WorkloadSpec(phase="decode", batch=16, context=4096, tp=8, ep=8, dcp=4),
        advisor.WorkloadSpec(phase="decode", batch=16, context=4096, tp=8, ep=1, dcp=1),
    ]
    batch_items: list[_BatchScenarioItem] = [
        {"only_validate": False, "model_name": model_name, "wl": wl} for wl in workloads
    ]
    expected = [_python_reference_eval(model_name, wl, chip_spec) for wl in workloads]

    in_json = tmp_path / "layout_parity_in.json"
    out_json = tmp_path / "layout_parity_out.json"
    in_json.write_text(
        json.dumps(
            {
                "engine_path": str(engine_path),
                "seed_data": {
                    "chip_spec": chip_dict,
                    **workbench_engine.build_engine_seed_defaults(),
                },
                "items": [
                    {
                        "params": {
                            "only_validate": False,
                            "profile": prof_dict,
                            "chip": chip_dict,
                            "wl": dataclasses.asdict(wl),
                        }
                    }
                    for wl in workloads
                ],
            }
        ),
        encoding="utf-8",
    )
    _run_engine_runner(js_runtime, runner_path, in_json, out_json)
    records = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(records) == len(expected)

    expected_coloc_rows = 64
    expected_dcp4_rows = 112

    # Layout 1 (TP8 x EP8)
    exp_sc_1 = expected[0]["scopes"]["whole_step"]
    rec_sc_1 = records[0]["eval"]["scopes"]["whole_step"]
    assert exp_sc_1["support"] == "declared"
    assert exp_sc_1["t_vmem_max"] == expected_coloc_rows
    assert rec_sc_1["support"] == "declared"
    assert rec_sc_1["t_vmem_max"] == expected_coloc_rows

    # Layout 2 (TP8 x EP8 x DCP4)
    exp_sc_2 = expected[1]["scopes"]["whole_step"]
    rec_sc_2 = records[1]["eval"]["scopes"]["whole_step"]
    assert exp_sc_2["support"] == "declared"
    assert exp_sc_2["t_vmem_max"] == expected_dcp4_rows
    assert rec_sc_2["support"] == "declared"
    assert rec_sc_2["t_vmem_max"] == expected_dcp4_rows

    # Layout 3 (TP8)
    exp_sc_3 = expected[2]["scopes"]["whole_step"]
    rec_sc_3 = records[2]["eval"]["scopes"]["whole_step"]
    assert exp_sc_3["support"] == "hypothetical"
    assert "layout_not_declared" in exp_sc_3["blockers"]
    assert "rank_count_outside_core_grid" not in exp_sc_3["blockers"]
    assert rec_sc_3["support"] == "hypothetical"
    assert "layout_not_declared" in rec_sc_3["blockers"]
    assert "rank_count_outside_core_grid" not in rec_sc_3["blockers"]

    _verify_parity_records(batch_items, expected, records)


def test_engine_parity_row_limit_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify row limit gating parity between Python advisor and JS engine."""
    monkeypatch.setattr(support, "DEPLOYMENT_PRESETS", {})
    monkeypatch.setattr(workbench_engine, "DEPLOYMENT_PRESETS", {})

    js_runtime = shutil.which("node") or shutil.which("gjs")
    if js_runtime is None:
        msg = "Neither node nor gjs runtime found; install nodejs"
        raise RuntimeError(msg)

    runner_path = Path(__file__).with_name("advisor_engine_runner.js")
    engine_path = Path(models.__file__).parent / "assets" / "workbench_engine.js"
    chip_spec = advisor.resolve_chip_spec("tpu7x")
    chip_dict = dataclasses.asdict(chip_spec)
    model_name = "qwen3.5-397b"
    prof_dict = dataclasses.asdict(models.resolve_model_profile(model_name))

    workloads = [
        advisor.WorkloadSpec(phase="decode", batch=32, tp=8, ep=8, context=4096),
        advisor.WorkloadSpec(phase="decode", batch=33, tp=8, ep=8, context=4096),
        advisor.WorkloadSpec(
            phase="decode",
            batch=33,
            tp=8,
            ep=8,
            context=4096,
            max_rows_per_call=64,
        ),
    ]
    batch_items: list[_BatchScenarioItem] = [
        {"only_validate": False, "model_name": model_name, "wl": wl} for wl in workloads
    ]
    expected = [_python_reference_eval(model_name, wl, chip_spec) for wl in workloads]

    in_json = tmp_path / "row_limit_parity_in.json"
    out_json = tmp_path / "row_limit_parity_out.json"
    in_json.write_text(
        json.dumps(
            {
                "engine_path": str(engine_path),
                "seed_data": {
                    "chip_spec": chip_dict,
                    **workbench_engine.build_engine_seed_defaults(),
                },
                "items": [
                    {
                        "params": {
                            "only_validate": False,
                            "profile": prof_dict,
                            "chip": chip_dict,
                            "wl": dataclasses.asdict(wl),
                        }
                    }
                    for wl in workloads
                ],
            }
        ),
        encoding="utf-8",
    )
    _run_engine_runner(js_runtime, runner_path, in_json, out_json)
    records = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(records) == len(expected)

    # batch=32: within core limits, no row limit blocker
    exp_ws_32 = expected[0]["scopes"]["whole_step"]
    rec_ws_32 = records[0]["eval"]["scopes"]["whole_step"]
    assert exp_ws_32["support"] == "within_core_limits"
    assert "row_limit_not_stated" not in exp_ws_32["blockers"]
    assert rec_ws_32["support"] == "within_core_limits"
    assert "row_limit_not_stated" not in rec_ws_32["blockers"]

    # batch=33: exceeds default bucket, requires stated limit
    exp_ws_33 = expected[1]["scopes"]["whole_step"]
    rec_ws_33 = records[1]["eval"]["scopes"]["whole_step"]
    assert exp_ws_33["support"] == "hypothetical"
    assert "row_limit_not_stated" in exp_ws_33["blockers"]
    assert rec_ws_33["support"] == "hypothetical"
    assert "row_limit_not_stated" in rec_ws_33["blockers"]

    # batch=33 with max_rows_per_call=64: within limits, blocker cleared
    exp_ws_64 = expected[2]["scopes"]["whole_step"]
    rec_ws_64 = records[2]["eval"]["scopes"]["whole_step"]
    assert exp_ws_64["support"] == "within_core_limits"
    assert "row_limit_not_stated" not in exp_ws_64["blockers"]
    assert rec_ws_64["support"] == "within_core_limits"
    assert "row_limit_not_stated" not in rec_ws_64["blockers"]

    _verify_parity_records(batch_items, expected, records)


def test_js_engine_parity_composition_busiest_groups_and_per_op_boundaries(
    tmp_path: Path,
) -> None:
    """Verify JS parity on collective composition, owner groups, and boundaries."""
    js_runtime = shutil.which("node") or shutil.which("gjs")
    if js_runtime is None:
        msg = "Neither node nor gjs runtime found; install nodejs"
        raise RuntimeError(msg)

    seed_defaults = workbench_engine.build_engine_seed_defaults()
    assert (
        seed_defaults["per_op_mixer_kernels_per_layer"]
        == constants.PER_OP_MIXER_KERNELS_PER_LAYER
    )
    core_limits_seed = seed_defaults["core_limits"]
    assert isinstance(core_limits_seed, dict)
    declared_map = core_limits_seed["declared_megakernels"]
    assert isinstance(declared_map, dict)
    kimi_entries = declared_map["kimi-k3"]
    assert isinstance(kimi_entries, list)
    kimi_preset = advisor.DEPLOYMENT_PRESETS["kimi-k3"]
    assert kimi_preset.decode is not None
    assert kimi_preset.decode.megakernel is not None
    expected_owner_groups = int(
        str(kimi_preset.decode.megakernel["expert_owner_groups"])
    )
    assert all(
        entry["expert_owner_groups"] == expected_owner_groups for entry in kimi_entries
    )

    runner_path = Path(__file__).with_name("advisor_engine_runner.js")
    engine_path = Path(models.__file__).parent / "assets" / "workbench_engine.js"
    chip_spec = advisor.resolve_chip_spec("tpu7x")
    chip_dict = dataclasses.asdict(chip_spec)
    qwen_workload = advisor.WorkloadSpec(
        phase="verify",
        spec_tokens=3,
        batch=1,
        context=16384,
        osl=1,
        tp=8,
        ep=8,
    )
    scenarios = [
        (
            "kimi-k3",
            advisor.WorkloadSpec(
                phase="decode",
                batch=4,
                context=32768,
                osl=1,
                tp=32,
                ep=32,
                dcp=32,
            ),
        ),
        ("qwen3.5-397b", qwen_workload),
    ]
    batch_items: list[_BatchScenarioItem] = []
    js_items: list[dict[str, object]] = []
    expected: list[dict[str, object]] = []
    for model_name, wl in scenarios:
        prof_dict = dataclasses.asdict(models.resolve_model_profile(model_name))
        batch_items.append({"only_validate": False, "model_name": model_name, "wl": wl})
        expected.append(_python_reference_eval(model_name, wl, chip_spec))
        js_items.append(
            {
                "params": {
                    "only_validate": False,
                    "profile": prof_dict,
                    "chip": chip_dict,
                    "wl": dataclasses.asdict(wl),
                }
            }
        )

    in_json = tmp_path / "owner_group_parity_in.json"
    out_json = tmp_path / "owner_group_parity_out.json"
    in_json.write_text(
        json.dumps(
            {
                "engine_path": str(engine_path),
                "seed_data": {
                    "chip_spec": chip_dict,
                    **seed_defaults,
                },
                "items": js_items,
            }
        ),
        encoding="utf-8",
    )
    _run_engine_runner(js_runtime, runner_path, in_json, out_json)
    records = json.loads(out_json.read_text(encoding="utf-8"))
    _verify_parity_records(batch_items, expected, records)

    records_by_model = {
        model_name: rec["eval"]
        for (model_name, _), rec in zip(scenarios, records, strict=True)
    }
    qwen_profile = models.resolve_model_profile("qwen3.5-397b")
    qwen_eff = qwen_workload.efficiency
    extra_per_op_boundaries = (
        constants.PER_OP_MIXER_KERNELS_PER_LAYER
        + constants.PER_OP_BASE_FFN_KERNELS_PER_LAYER
        + constants.PER_OP_SHARED_EXPERT_KERNELS_PER_LAYER
        - constants.SUBLAYER_BOUNDARIES_PER_LAYER
    ) * qwen_profile.num_layers
    per_boundary_ach_us = (
        (constants.DENSE_SLOT_DEFAULT_BYTES / chip_spec.hbm_bw_bytes_per_s)
        / qwen_eff.eta_hbm
        + (constants.BOUNDARY_SYNC_HOP_FACTOR * chip_spec.collective_latency_s)
        / qwen_eff.eta_ici
    ) * 1e6
    qwen_per_op = records_by_model["qwen3.5-397b"]["scopes"]["per_op"]
    qwen_whole_step = records_by_model["qwen3.5-397b"]["scopes"]["whole_step"]
    assert (
        qwen_per_op["step_achievable_us"] - qwen_per_op["step_bound_us"]
        >= extra_per_op_boundaries * per_boundary_ach_us
    )
    assert qwen_whole_step["step_achievable_us"] >= (
        qwen_whole_step["hbm_us"] / qwen_eff.eta_hbm
        + qwen_whole_step["comm_us"] / qwen_eff.eta_ici
    )


def test_js_engine_matches_python_on_declared_collectives_expert_mode_and_host_head(
    tmp_path: Path,
) -> None:
    """Verify Python == JS parity on declared steps, expert_mode, and host head."""
    js_runtime = shutil.which("node") or shutil.which("gjs")
    if js_runtime is None:
        msg = "Neither node nor gjs runtime found; install nodejs"
        raise RuntimeError(msg)

    runner_path = Path(__file__).with_name("advisor_engine_runner.js")
    engine_path = Path(models.__file__).parent / "assets" / "workbench_engine.js"
    chip_spec = advisor.resolve_chip_spec("tpu7x")
    chip_dict = dataclasses.asdict(chip_spec)

    collective_and_host_head_cases: list[tuple[str, advisor.WorkloadSpec]] = [
        (
            "kimi-k3",
            advisor.WorkloadSpec(
                phase="decode",
                batch=4,
                context=4096,
                osl=1,
                tp=32,
                ep=32,
                dcp=32,
            ),
        ),
        (
            "kimi-k3",
            advisor.WorkloadSpec(
                phase="decode",
                batch=4,
                context=4096,
                osl=1,
                tp=32,
                ep=32,
                dcp=32,
                pcp=2,
            ),
        ),
        (
            "qwen3.5-397b",
            advisor.WorkloadSpec(
                phase="verify",
                batch=3,
                spec_tokens=3,
                context=4096,
                tp=8,
                ep=8,
                dcp=4,
            ),
        ),
        (
            "qwen3.5-397b",
            advisor.WorkloadSpec(
                phase="verify",
                batch=4,
                spec_tokens=3,
                context=4096,
                tp=8,
                ep=8,
                dcp=4,
            ),
        ),
        (
            "qwen3.5-397b",
            advisor.WorkloadSpec(
                phase="verify",
                batch=56,
                spec_tokens=3,
                context=4096,
                tp=8,
                ep=8,
                dcp=4,
                dispatch_regime="zero_floor",
                host_head_per_op_us=1800.0,
                host_head_fused_us=350.0,
                host_head_explicit=True,
            ),
        ),
    ]

    batch_items: list[_BatchScenarioItem] = []
    expected_refs: list[_EvalParityResult] = []
    payload_items: list[dict[str, object]] = []
    for model_name, wl in collective_and_host_head_cases:
        prof_dict = dataclasses.asdict(models.resolve_model_profile(model_name))
        batch_items.append({"only_validate": False, "model_name": model_name, "wl": wl})
        expected_refs.append(_python_reference_eval(model_name, wl, chip_spec))
        payload_items.append(
            {
                "params": {
                    "only_validate": False,
                    "profile": prof_dict,
                    "chip": chip_dict,
                    "wl": dataclasses.asdict(wl),
                }
            }
        )

    # Guard declared-collective and host-head invariants before checking JS parity.
    expected_kimi_comm_us = compute_expected_kimi_k3_decode_comm_us(
        models.resolve_model_profile("kimi-k3"), chip_spec, 4
    )
    assert expected_refs[0]["scopes"]["whole_step"]["comm_us"] == pytest.approx(
        expected_kimi_comm_us, abs=0.05
    )
    assert "layout_not_declared" in expected_refs[1]["scopes"]["whole_step"]["blockers"]
    assert expected_refs[2]["scopes"]["whole_step"]["expert_mode"] == "tp"
    assert expected_refs[3]["scopes"]["whole_step"]["expert_mode"] == "ep"
    expected_fused_head_us = 350.0
    assert (
        expected_refs[4]["scopes"]["whole_step"]["host_head_us"]
        == expected_fused_head_us
    )
    assert (
        expected_refs[4]["scopes"]["whole_step"]["break_even_host_head_delta_us"] > 0.0
    )

    in_json = tmp_path / "collective_parity_in.json"
    out_json = tmp_path / "collective_parity_out.json"
    in_json.write_text(
        json.dumps(
            {
                "engine_path": str(engine_path),
                "seed_data": {
                    "chip_spec": chip_dict,
                    **workbench_engine.build_engine_seed_defaults(),
                },
                "items": payload_items,
            }
        ),
        encoding="utf-8",
    )
    _run_engine_runner(js_runtime, runner_path, in_json, out_json)
    out_records = json.loads(out_json.read_text(encoding="utf-8"))
    _verify_parity_records(batch_items, expected_refs, out_records)


def test_glm_moe_dsa_and_dcp_engine_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify Python/JS parity on GLM-5.2 DSA, indexer, DCP, VMEM, and HBM."""
    js_runtime = shutil.which("node") or shutil.which("gjs")
    if js_runtime is None:
        msg = "Neither node nor gjs runtime found; install nodejs"
        raise RuntimeError(msg)

    declared_dsa_deployment = models.DeploymentPreset(
        decode=models.ParallelLayout(
            tp=8,
            ep=1,
            dp=1,
            pp=1,
            dcp=4,
            megakernel={
                "scopes": ["whole_step"],
                "max_rows": 64,
                "ranks": 8,
                "phases": ["decode"],
                "collectives": {
                    "steps": [
                        {
                            "kind": "dcp_gather",
                            "group_size": 4,
                            "calls_per_layer": 2,
                        }
                    ]
                },
                "source": "declared DSA DCP parity contract test",
            },
        ),
        prefill=models.ParallelLayout(tp=8, ep=1),
    )
    monkeypatch.setitem(
        support.DEPLOYMENT_PRESETS, "homo-dsa-declared", declared_dsa_deployment
    )
    seed_defaults = workbench_engine.build_engine_seed_defaults()
    runner_path = Path(__file__).with_name("advisor_engine_runner.js")
    engine_path = Path(models.__file__).parent / "assets" / "workbench_engine.js"
    chip_spec = advisor.resolve_chip_spec("tpu7x")
    chip_dict = dataclasses.asdict(chip_spec)

    glm_profile = models.resolve_model_profile("glm-5.2")
    homogeneous_dsa_profile = models.finalize_model_profile(
        advisor.ModelProfile(
            name="homo-dsa",
            model_type="glm_moe_dsa",
            num_layers=32,
            num_attention_layers=32,
            num_moe_layers=32,
            hidden_size=4096,
            intermediate_size=8192,
            moe_intermediate_size=1024,
            shared_expert_intermediate_size=1024,
            num_attention_heads=32,
            num_kv_heads=32,
            head_dim=128,
            q_lora_rank=1024,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            num_experts=64,
            top_k=8,
            index_topk=2048,
            index_head_dim=128,
            index_n_heads=16,
            num_indexer_layers=32,
            weight_dtype="fp8",
            expert_dtype="fp8",
        )
    )
    declared_dsa_profile = dataclasses.replace(
        homogeneous_dsa_profile, name="homo-dsa-declared"
    )
    cases: list[tuple[advisor.ModelProfile, advisor.WorkloadSpec]] = [
        (
            glm_profile,
            advisor.WorkloadSpec(
                phase="decode", batch=4, context=131076, tp=16, ep=1, dcp=1
            ),
        ),
        (
            glm_profile,
            advisor.WorkloadSpec(
                phase="verify",
                spec_tokens=3,
                batch=1,
                context=1024,
                tp=16,
                ep=1,
                dcp=8,
            ),
        ),
        (
            glm_profile,
            advisor.WorkloadSpec(
                phase="verify",
                spec_tokens=3,
                batch=1,
                context=131076,
                tp=16,
                ep=1,
                dcp=8,
            ),
        ),
        (
            glm_profile,
            advisor.WorkloadSpec(
                phase="verify",
                spec_tokens=3,
                batch=4,
                context=131076,
                tp=16,
                ep=1,
                dcp=8,
            ),
        ),
        (
            glm_profile,
            advisor.WorkloadSpec(
                phase="verify",
                spec_tokens=3,
                batch=8,
                context=131076,
                tp=16,
                ep=1,
                dcp=8,
            ),
        ),
        (
            glm_profile,
            advisor.WorkloadSpec(
                phase="decode", batch=32, context=131076, tp=16, ep=16, dcp=16
            ),
        ),
        (
            glm_profile,
            advisor.WorkloadSpec(
                phase="prefill", batch=1, isl=4096, chunk_size=2048, tp=16, ep=1, dcp=1
            ),
        ),
        (
            glm_profile,
            advisor.WorkloadSpec(
                phase="mixed",
                batch=4,
                isl=4096,
                chunk_size=512,
                context=16384,
                tp=16,
                ep=1,
                dcp=8,
            ),
        ),
        (
            glm_profile,
            advisor.WorkloadSpec(
                phase="verify",
                spec_tokens=3,
                batch=8,
                context=131076,
                tp=16,
                ep=1,
                dcp=8,
                max_rows_per_call=16,
            ),
        ),
        (
            homogeneous_dsa_profile,
            advisor.WorkloadSpec(
                phase="decode", batch=8, context=8192, tp=8, ep=1, dcp=4
            ),
        ),
        (
            declared_dsa_profile,
            advisor.WorkloadSpec(
                phase="decode", batch=8, context=8192, tp=8, ep=1, dcp=4
            ),
        ),
    ]
    items = [
        {
            "params": {
                "only_validate": False,
                "profile": dataclasses.asdict(model_profile),
                "chip": chip_dict,
                "wl": dataclasses.asdict(workload),
            }
        }
        for model_profile, workload in cases
    ]
    in_json = tmp_path / "glm_dsa_in.json"
    out_json = tmp_path / "glm_dsa_out.json"
    in_json.write_text(
        json.dumps(
            {
                "engine_path": str(engine_path),
                "seed_data": {"chip_spec": chip_dict, **seed_defaults},
                "items": items,
            }
        ),
        encoding="utf-8",
    )
    _run_engine_runner(js_runtime, runner_path, in_json, out_json)
    records = json.loads(out_json.read_text(encoding="utf-8"))

    for (model_profile, workload), record in zip(cases, records, strict=True):
        expected_reasons = list(
            advisor.validate_workload(model_profile, chip_spec, workload)
        )
        scope_context = _build_scope_context(model_profile, workload, chip_spec)
        report = advisor.advise(model_profile, workload, chip_spec)
        got = record["eval"]
        full = got["full"]
        assert list(got["invalid_reasons"]) == expected_reasons
        assert full["hbmUsedBytes"] == pytest.approx(scope_context.hbm_used, rel=1e-6)
        assert full["fastestScope"] == report.fastest_scope
        assert full["recScope"] == report.recommended_scope
        for scope in report.scopes:
            got_scope = got["scopes"][scope.scope]
            assert got_scope["t_vmem_max"] == scope.t_vmem_max
            assert got_scope["n_splits"] == scope.n_splits
            assert got_scope["support"] == scope.support
            assert list(got_scope["blockers"]) == list(scope.blockers)
            for field_name in (
                "step_bound_us",
                "step_floor_strict_us",
                "step_achievable_us",
                "step_achievable_upper_us",
                "step_upper_us",
                "act_hbm_mb",
                "comm_us",
                "chain_us",
            ):
                assert got_scope[field_name] == pytest.approx(
                    getattr(scope, field_name), abs=DISPLAY_ROUND_TOL_US
                ), f"{model_profile.name} {workload.phase} {scope.scope} {field_name}"


def _assert_scope_parity_fields(
    got_scope: dict[str, object],
    expected_scope: advisor.ScopeEstimate,
    label: str,
) -> None:
    """Assert JS scope estimate matches Python support, blockers, and latencies."""
    assert got_scope["support"] == expected_scope.support
    assert list(cast("list[str]", got_scope["blockers"])) == list(
        expected_scope.blockers
    )
    for field_name, exp_val in (
        ("step_bound_us", expected_scope.step_bound_us),
        ("step_floor_strict_us", expected_scope.step_floor_strict_us),
        ("step_achievable_us", expected_scope.step_achievable_us),
        ("step_achievable_upper_us", expected_scope.step_achievable_upper_us),
        ("step_upper_us", expected_scope.step_upper_us),
    ):
        assert got_scope[field_name] == pytest.approx(
            exp_val, abs=DISPLAY_ROUND_TOL_US
        ), f"{label} {expected_scope.scope} {field_name}"


def _sublayer_item(layout: models.ParallelLayout) -> dict[str, object]:
    """Build runner payload item to query JS sublayerLabel for one layout."""
    owner_groups = (
        layout.megakernel.get("expert_owner_groups")
        if layout.megakernel is not None
        else None
    )
    return {
        "sublayer_layout": {
            "tp": layout.tp,
            "ep": layout.ep,
            "dp": layout.dp,
            "pp": layout.pp,
            "dcp": layout.dcp,
            "pcp": layout.pcp,
        },
        "expert_owner_groups": owner_groups,
    }


def _run_engine_runner_batch(
    tmp_path: Path,
    batch_slug: str,
    items: list[dict[str, object]],
    chip_dict: dict[str, object],
) -> list[dict[str, object]]:
    """Run the JS engine runner on a list of evaluation or sublayer items."""
    js_runtime = shutil.which("node") or shutil.which("gjs")
    if js_runtime is None:
        msg = "Neither node nor gjs runtime found; install nodejs"
        raise RuntimeError(msg)

    seed_defaults = workbench_engine.build_engine_seed_defaults()
    runner_path = Path(__file__).with_name("advisor_engine_runner.js")
    engine_path = Path(models.__file__).parent / "assets" / "workbench_engine.js"
    in_json = tmp_path / f"{batch_slug}_in.json"
    out_json = tmp_path / f"{batch_slug}_out.json"
    in_json.write_text(
        json.dumps(
            {
                "engine_path": str(engine_path),
                "seed_data": {"chip_spec": chip_dict, **seed_defaults},
                "items": items,
            }
        ),
        encoding="utf-8",
    )
    _run_engine_runner(js_runtime, runner_path, in_json, out_json)
    return cast("list[dict[str, object]]", json.loads(out_json.read_text("utf-8")))


def _assert_eval_cases_parity(
    tmp_path: Path,
    batch_slug: str,
    cases: list[tuple[advisor.ModelProfile, advisor.WorkloadSpec]],
) -> None:
    """Assert JS engine scope estimates and winners match Python advise."""
    chip_spec = advisor.resolve_chip_spec("tpu7x")
    chip_dict = dataclasses.asdict(chip_spec)
    items: list[dict[str, object]] = [
        {
            "params": {
                "only_validate": False,
                "profile": dataclasses.asdict(model_profile),
                "chip": chip_dict,
                "wl": dataclasses.asdict(workload),
            }
        }
        for model_profile, workload in cases
    ]
    records = _run_engine_runner_batch(tmp_path, batch_slug, items, chip_dict)
    for (model_profile, workload), rec in zip(cases, records, strict=True):
        report = advisor.advise(model_profile, workload, chip_spec)
        got = cast("dict[str, object]", rec["eval"])
        full = cast("dict[str, object]", got["full"])
        scopes_map = cast("dict[str, dict[str, object]]", got["scopes"])
        assert full["fastestScope"] == report.fastest_scope
        assert full["recScope"] == report.recommended_scope
        for scope in report.scopes:
            _assert_scope_parity_fields(
                scopes_map[scope.scope],
                scope,
                f"{model_profile.name} {workload.phase}",
            )


def test_prefill_sparse_dsa_and_kimi_k31_engine_parity(tmp_path: Path) -> None:
    """Verify prefill sparse DSA and Kimi-K3 JS engine parity."""
    kimi31_prof = models.resolve_model_profile("kimi-k3")
    glm_prof = models.resolve_model_profile("glm-5.2")
    cases: list[tuple[advisor.ModelProfile, advisor.WorkloadSpec]] = [
        (
            kimi31_prof,
            advisor.WorkloadSpec(
                phase="decode", batch=4, context=4096, tp=32, ep=32, dcp=32
            ),
        ),
        (
            kimi31_prof,
            advisor.WorkloadSpec(
                phase="prefill", batch=1, isl=4096, tp=32, ep=32, pcp=32
            ),
        ),
        (
            glm_prof,
            advisor.WorkloadSpec(phase="prefill", batch=1, isl=4096, tp=16),
        ),
    ]
    _assert_eval_cases_parity(tmp_path, "prefill_dsa_kimi31", cases)


def test_core_gap_blockers_engine_parity(tmp_path: Path) -> None:
    """Verify core dtype and family gap blockers match Python advise in JS."""
    kimi_prof = models.resolve_model_profile("kimi-k3")
    glm_prof = models.resolve_model_profile("glm-5.2")
    cases: list[tuple[advisor.ModelProfile, advisor.WorkloadSpec]] = [
        (
            kimi_prof,
            advisor.WorkloadSpec(phase="decode", batch=4, context=4096, tp=8, ep=8),
        ),
        (
            glm_prof,
            advisor.WorkloadSpec(phase="decode", batch=4, context=4096, tp=16),
        ),
    ]
    _assert_eval_cases_parity(tmp_path, "core_gap_blockers", cases)


def test_sublayer_mesh_label_engine_parity(tmp_path: Path) -> None:
    """Verify JS sublayerLabel matches Python labels with explicit and auto groups."""
    chip_dict = dataclasses.asdict(advisor.resolve_chip_spec("tpu7x"))
    layout_cases: list[tuple[str, str, models.ParallelLayout]] = []
    for preset_slug in ("kimi-k3", "qwen3.5-397b"):
        preset = advisor.DEPLOYMENT_PRESETS[preset_slug]
        candidates: list[tuple[str, models.ParallelLayout | None]] = [
            ("decode", preset.decode),
            ("prefill", preset.prefill),
            ("saturation", preset.saturation),
        ]
        candidates.extend(
            (f"megakernel_{idx}", mk_layout)
            for idx, mk_layout in enumerate(preset.megakernel_layouts)
        )
        for role_tag, layout in candidates:
            if layout is not None:
                layout_cases.append((preset_slug, role_tag, layout))

    preset_profiles = {
        slug: dataclasses.asdict(models.resolve_model_profile(slug))
        for slug in ("kimi-k3", "qwen3.5-397b")
    }
    items: list[dict[str, object]] = [
        _sublayer_item(layout) for _, _, layout in layout_cases
    ]
    items.extend(
        {
            "sublayer_layout": _sublayer_item(layout)["sublayer_layout"],
            "sublayer_profile": preset_profiles[preset_slug],
        }
        for preset_slug, _, layout in layout_cases
    )

    records = _run_engine_runner_batch(tmp_path, "sublayer_labels", items, chip_dict)
    auto_start = len(layout_cases)
    for (preset_slug, role_tag, layout), rec in zip(
        layout_cases, records[:auto_start], strict=True
    ):
        assert rec["sublayer_label"] == layout.submodule_label(), (
            f"{preset_slug} {role_tag}"
        )
    for (preset_slug, role_tag, layout), rec in zip(
        layout_cases, records[auto_start:], strict=True
    ):
        assert rec["sublayer_label"] == layout.submodule_label(), (
            f"{preset_slug} {role_tag} auto owner groups"
        )
