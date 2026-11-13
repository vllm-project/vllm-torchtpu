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
import os
import shutil
import signal
import time
from pathlib import Path
from typing import TypedDict

import pytest

from vllm_torchtpu.megakernel import advisor
from vllm_torchtpu.megakernel.advisor import (
    constants,
    models,
    support,
    workbench_engine,
)


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


def _select_five_models() -> tuple[str, ...]:
    """Return 5 distinct model presets spanning GQA, GDN, MLA, MoE, and dense FFN."""
    return (
        "qwen3.5-397b",
        "kimi-k3",
        "qwen3.8-2.4t",
        "deepseek-v3",
        "llama-3.1-405b",
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
    models_sample = _select_five_models()
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
    expected_scenario_count = 81  # 81 scenario combinations
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
    unv_prof = dataclasses.replace(
        qwen_prof,
        name="custom-unverified",
        model_type="future_arch",
        architecture_unverified=True,
    )

    cases = [
        (qwen_prof, advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=1)),
        (qwen_prof, advisor.WorkloadSpec(phase="decode", batch=256, tp=8, ep=1)),
        (kimi_prof, advisor.WorkloadSpec(phase="decode", batch=64, tp=4, dp=8, ep=32)),
        (unv_prof, advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=1)),
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


def test_sparse_csa_and_hyper_connection_engine_parity(tmp_path: Path) -> None:
    """Verify JS parity for deepseek-v4 (CSA + mHC) and qwen4-exp (4x residual)."""
    js_runtime = shutil.which("node") or shutil.which("gjs")
    if js_runtime is None:
        msg = "Neither node nor gjs runtime found; install nodejs"
        raise RuntimeError(msg)

    seed_defaults = workbench_engine.build_engine_seed_defaults()
    runner_path = Path(__file__).with_name("advisor_engine_runner.js")
    engine_path = Path(models.__file__).parent / "assets" / "workbench_engine.js"
    chip_spec = advisor.resolve_chip_spec("tpu7x")
    chip_dict = dataclasses.asdict(chip_spec)

    cases: list[tuple[advisor.ModelProfile, advisor.WorkloadSpec]] = []
    for model_name in ("deepseek-v4", "qwen4-exp"):
        model_profile = models.resolve_model_profile(model_name)
        cases.extend(
            [
                (
                    model_profile,
                    advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=1),
                ),
                (
                    model_profile,
                    advisor.WorkloadSpec(
                        phase="prefill",
                        batch=1,
                        isl=4096,
                        chunk_size=512,
                        tp=8,
                        ep=1,
                    ),
                ),
            ]
        )

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
    in_json = tmp_path / "csa_mhc_in.json"
    out_json = tmp_path / "csa_mhc_out.json"
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
            assert got_sc["step_bound_us"] == pytest.approx(
                scope.step_bound_us, abs=0.015
            )
            assert got_sc["step_floor_strict_us"] == pytest.approx(
                scope.step_floor_strict_us, abs=0.015
            )
            assert got_sc["step_achievable_us"] == pytest.approx(
                scope.step_achievable_us, abs=0.015
            )
            assert got_sc["step_achievable_upper_us"] == pytest.approx(
                scope.step_achievable_upper_us, abs=0.015
            )
            assert got_sc["act_hbm_mb"] == pytest.approx(scope.act_hbm_mb, abs=0.015)


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
        dep, decode=dataclasses.replace(dep.decode, megakernel=declaration)
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
    kv_less_prof = dataclasses.replace(
        models.resolve_model_profile("Qwen/Qwen3-30B-A3B-FP8"),
        num_kv_heads=0,
    )

    invalid_cases: list[tuple[advisor.ModelProfile, advisor.WorkloadSpec]] = [
        (
            models.resolve_model_profile("Qwen/Qwen3-30B-A3B-FP8"),
            advisor.WorkloadSpec(phase="decode", batch=8, tp=2, ep=1),
        ),
        (
            models.resolve_model_profile("Qwen/Qwen3-30B-A3B-FP8"),
            advisor.WorkloadSpec(phase="decode", batch=8, tp=8, dp=3),
        ),
        (
            models.resolve_model_profile("Qwen/Qwen3-30B-A3B-FP8"),
            advisor.WorkloadSpec(phase="decode", batch=8, tp=8, pcp=3),
        ),
        (
            models.resolve_model_profile("Qwen/Qwen3-30B-A3B-FP8"),
            advisor.WorkloadSpec(phase="decode", batch=4, tp=8, dp=8),
        ),
        (
            models.resolve_model_profile("Qwen/Qwen3-30B-A3B-FP8"),
            advisor.WorkloadSpec(phase="decode", batch=8, tp=8, pp=2, pcp=2),
        ),
        (
            models.resolve_model_profile("Qwen/Qwen3-30B-A3B-FP8"),
            advisor.WorkloadSpec(phase="verify", batch=8, spec_tokens=0, tp=8),
        ),
        (
            kv_less_prof,
            advisor.WorkloadSpec(phase="decode", batch=8, tp=8, kv_split=2),
        ),
    ]
    eval_cases: list[tuple[advisor.ModelProfile, advisor.WorkloadSpec]] = [
        (
            models.resolve_model_profile("qwen3.5-0.8b"),
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
