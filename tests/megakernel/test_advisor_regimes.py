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
"""Unit tests for megakernel advisor regime sweeps, plain-text formatting, and CLI."""

from __future__ import annotations

import ast
import dataclasses
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Mapping

import pytest

from tests.megakernel.test_advisor_bounds import iter_case_grid
from tests.megakernel.test_advisor_serving_sanity import (
    _MAX_SHOWN_VIOLATIONS,
    _MOE_SETTINGS,
    _PRESET_SLUGS,
    _apply_moe_setting,
    _check_scope_envelope_invariants,
    _check_verdict_and_core_invariants,
)
from tests.megakernel.test_advisor_specs import (
    HF_CONFIG_DIR,
    load_qwen35_snapshot_config,
)
from vllm_torchtpu import megakernel as mk
from vllm_torchtpu.megakernel import advisor
from vllm_torchtpu.megakernel.advisor import (
    cli,
    constants,
    models,
    specs,
    text_format,
)
from vllm_torchtpu.megakernel.advisor.chips import resolve_chip_spec
from vllm_torchtpu.megakernel.advisor.regimes import (
    _build_reference_workloads,
    _compute_disagg_handoff,
    _with_preset_prefill,
)
from vllm_torchtpu.megakernel.core.errors import MegakernelConfigError

KERNEL_NAME_DENYLIST = re.compile(
    r"gmm|ragged_dot|megablox|sparsecore|mxu|tensorcore|vpu|\bllo\b", re.IGNORECASE
)
TEST_ROW_LIMIT = 80
EXPECTED_DEFAULT_REGIMES = 12


def _regime(
    profile_or_slug: advisor.ModelProfile | str,
    prefix: str,
    options: advisor.RegimeSweepOptions | None = None,
) -> advisor.RegimeEntry:
    """Return the regime row whose label starts with `prefix`."""
    sweep = advisor.advise_regimes(profile_or_slug, "tpu7x", options=options)
    return next(entry for entry in sweep.regimes if entry.label.startswith(prefix))


def _assert_core_readiness_entry(
    entry: advisor.RegimeEntry,
    *,
    recommended_scope: advisor.FusionScope,
    core_support: str,
    core_ready: bool,
    core_ready_scope: advisor.FusionScope,
) -> None:
    """Assert physics recommendation and core readiness fields on one RegimeEntry."""
    assert entry.recommended_scope == recommended_scope
    assert entry.core_support == core_support
    assert entry.core_ready is core_ready
    assert entry.core_ready_scope == core_ready_scope


def _assert_qwen_auto_sweeps() -> tuple[
    dict[str, advisor.RegimeEntry],
    advisor.AdvisorSweepReport,
    advisor.AdvisorSweepReport,
]:
    """Assert Qwen3.5 fp32 vs default bf16 auto-parallelism 12-regime sweeps."""
    qwen_fp32 = advisor.resolve_model_profile("qwen3.5-397b", state_dtype="fp32")
    sweep_fp32 = advisor.advise_regimes(
        qwen_fp32,
        "tpu7x",
        options=advisor.RegimeSweepOptions(deployment_preset=False),
    )
    assert sweep_fp32.parallelism.auto_selected is True
    assert (sweep_fp32.parallelism.tp, sweep_fp32.parallelism.ep) == (8, 8)
    assert "2x2x1" in sweep_fp32.parallelism.topology
    assert len(sweep_fp32.regimes) == EXPECTED_DEFAULT_REGIMES
    fp32_by_label = {entry.label: entry for entry in sweep_fp32.regimes}
    fp32_small = fp32_by_label["Decode Small (B=4, T=1, 4K)"]
    _assert_core_readiness_entry(
        fp32_small,
        recommended_scope="whole_step",
        core_support="outside_core_limits",
        core_ready=False,
        core_ready_scope="per_op",
    )
    assert fp32_small.fastest_scope == "whole_step"
    assert "kv_or_state_dtype_not_in_core" in fp32_small.core_blockers
    assert fp32_small.evidence == "outside_core_limits"
    assert "hypothetical_best" not in fp32_small.to_dict()
    fp32_whole_scope = next(
        sc for sc in fp32_small.scopes if sc["scope"] == "whole_step"
    )
    assert fp32_whole_scope["support"] == "outside_core_limits"
    whole_blockers = cast("list[str] | tuple[str, ...]", fp32_whole_scope["blockers"])
    assert "kv_or_state_dtype_not_in_core" in whole_blockers

    qwen_bf16 = advisor.resolve_model_profile("qwen3.5-397b")
    assert qwen_bf16.state_dtype == "bf16"
    sweep_bf16 = advisor.advise_regimes(
        qwen_bf16,
        "tpu7x",
        options=advisor.RegimeSweepOptions(deployment_preset=False),
    )
    return fp32_by_label, sweep_fp32, sweep_bf16


def _assert_qwen_bf16_regimes(bf16_by_label: Mapping[str, advisor.RegimeEntry]) -> None:
    """Assert default bf16 Qwen3.5 regime scopes, core readiness, and MoE modes."""
    bf16_small = bf16_by_label["Decode Small (B=4, T=1, 4K)"]
    _assert_core_readiness_entry(
        bf16_small,
        recommended_scope="whole_step",
        core_support="within_core_limits",
        core_ready=True,
        core_ready_scope="whole_step",
    )
    assert bf16_small.core_blockers == ()
    assert bf16_small.recommended_moe_mode == "masked_on_chip"

    bf16_high = bf16_by_label["Decode High (B=64, T=1, 32K)"]
    _assert_core_readiness_entry(
        bf16_high,
        recommended_scope="whole_step",
        core_support="outside_core_limits",
        core_ready=False,
        core_ready_scope="per_op",
    )
    assert bf16_high.fastest_scope == "whole_step"
    assert "row_limit_not_stated" in bf16_high.core_blockers
    assert bf16_high.recommended_moe_mode == "grouped_on_chip"

    bf16_chunked = bf16_by_label["Chunked Prefill (B=1, T=256)"]
    _assert_core_readiness_entry(
        bf16_chunked,
        recommended_scope="whole_step",
        core_support="outside_core_limits",
        core_ready=False,
        core_ready_scope="per_op",
    )
    assert bf16_chunked.fastest_scope == "whole_step"
    assert bf16_chunked.recommended_moe_mode == "grouped_on_chip"

    bf16_long = bf16_by_label["Long Prefill (B=1, T=4096)"]
    _assert_core_readiness_entry(
        bf16_long,
        recommended_scope="per_op",
        core_support="baseline",
        core_ready=False,
        core_ready_scope="per_op",
    )
    assert bf16_long.recommended_moe_mode == "grouped_in_hbm"


def test_bare_minimum_auto_parallelism_and_regime_table() -> None:
    """Verify auto-parallelism 12-regime sweep, state_dtype core gate, and table."""
    fp32_by_label, sweep_fp32, sweep_bf16 = _assert_qwen_auto_sweeps()
    _assert_qwen_bf16_regimes({entry.label: entry for entry in sweep_bf16.regimes})

    sweep_dcp = advisor.advise_regimes(
        "qwen3.5-397b",
        "tpu7x",
        options=advisor.RegimeSweepOptions(dcp=4, pcp=2),
    )
    assert (sweep_dcp.parallelism.dcp, sweep_dcp.parallelism.pcp) == (4, 2)
    dcp_by_label = {entry.label: entry for entry in sweep_dcp.regimes}
    assert (
        dcp_by_label["Long Prefill (B=1, T=4096)"].active_tokens
        == fp32_by_label["Long Prefill (B=1, T=4096)"].active_tokens // 2
    )

    sweep_k3 = advisor.advise_regimes("kimi-k3", "tpu7x")
    assert (sweep_k3.parallelism.tp, sweep_k3.parallelism.ep) == (32, 32)
    assert "2x2x4" in sweep_k3.parallelism.topology
    k3_small = sweep_k3.regimes[0]
    _assert_core_readiness_entry(
        k3_small,
        recommended_scope="whole_step",
        core_support="outside_core_limits",
        core_ready=False,
        core_ready_scope="per_op",
    )
    assert k3_small.fastest_scope == "whole_step"
    assert "kv_or_state_dtype_not_in_core" in k3_small.core_blockers
    _assert_core_readiness_entry(
        sweep_k3.regimes[-1],
        recommended_scope="layer_group",
        core_support="outside_core_limits",
        core_ready=False,
        core_ready_scope="per_op",
    )

    table_text = advisor.format_regime_table(sweep_bf16)
    for marker in (
        "Decode Small (B=4, T=1, 4K)",
        "Core Support",
        "Core-Ready",
        "Core Blockers",
        "within_core_limits",
        "Scope (where the kernel boundary falls):",
        "Tailoring Guide:",
        "Derated Achievable",
    ):
        assert marker in table_text

    fp32_table_text = advisor.format_regime_table(sweep_fp32)
    assert "outside_core_limits" in fp32_table_text
    assert "kv_or_state_dtype_not_in_core" in fp32_table_text


@pytest.mark.parametrize(
    "bad_extra",
    [
        ["--prefix-hit-rate", "1.0"],
        ["--ep", "0"],
        ["--kv-split", "0"],
        ["--op-overhead-us", "-1.0"],
        ["--dma-buffers", "4"],
        ["--vmem-budget-mib", "1.0"],
        ["--hbm-utilization-cap", "0.0"],
        ["--hbm-utilization-cap", "1.2"],
        ["--layer-groups", ","],
        ["--layer-groups", "8,,8"],
    ],
)
def test_cli_parameterized_validation_errors(
    tmp_path: Path, bad_extra: list[str]
) -> None:
    """Verify parameterized CLI validation boundaries raise `MegakernelConfigError`."""
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(load_qwen35_snapshot_config()), encoding="utf-8")
    with pytest.raises(MegakernelConfigError):
        advisor.main(["--config", str(config_file), *bad_extra])


@pytest.mark.parametrize(
    "bad_argv",
    [
        ["--model", ""],
        ["--model", "   "],
        ["--hf-repo", ""],
        ["--list-models", "--json"],
        ["--list-models", "--unresolved"],
        ["--list-models", "--kv-dtype", "fp8"],
        ["--model", "qwen3.5-397b", "--regime-sweep", "--search-layer-groups"],
        ["--model", "qwen3.5-397b", "--regime-sweep", "--explain"],
        ["--model", "qwen3.5-397b", "--regime-sweep", "--batch", "4"],
        ["--model", "qwen3.5-397b", "--phase", "prefill", "--decoding-seqs", "8"],
        [
            "--model",
            "qwen3.5-397b",
            "--phase",
            "prefill",
            "--output-tok-s",
            "500.0",
            "--mean-tpot-ms",
            "10.0",
        ],
        [
            "--model",
            "qwen3.5-397b",
            "--tp",
            "8",
            "--batch",
            "4",
            "--decoding-seqs",
            "4",
        ],
        ["--model", "qwen3.5-397b", "--tp", "8", "--output-tok-s", "500.0"],
        [
            "--model",
            "qwen3.5-397b",
            "--tp",
            "8",
            "--output-tok-s",
            "-1.0",
            "--mean-tpot-ms",
            "10.0",
        ],
        ["--model", "qwen3.5-397b", "--tp", "8", "--decoding-seqs", "0"],
        ["--model", "qwen3.5-397b", "--tp", "8", "--host-head-us", "per_op=-5"],
        ["--model", "qwen3.5-397b", "--tp", "8", "--host-head-us", "bad_key=10"],
        ["--model", "qwen3.5-397b", "--tp", "8", "--per-op-step-us", "0"],
        ["--model", "qwen3.5-397b", "--tp", "8", "--step-coverage", "1.2"],
    ],
)
def test_cli_rejects_empty_targets_and_incompatible_modes(
    bad_argv: list[str],
) -> None:
    """Verify empty targets and incompatible CLI mode/flag combos fail closed."""
    with pytest.raises(MegakernelConfigError):
        advisor.main(bad_argv)


def test_cli_prefill_defaults_osl_zero_and_clean_arg_type_errors(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify prefill defaults `--osl` to 0 and invalid numeric types name `int`."""
    captured_workloads: list[advisor.WorkloadSpec] = []
    orig_advise = cli.advise

    def _capture_advise(
        profile: advisor.ModelProfile | str | Mapping[str, object],
        workload: advisor.WorkloadSpec,
        chip: advisor.ChipSpec | str = constants.DEFAULT_CHIP,
        plan_error: str = "",
        measurements: specs.MeasurementsData
        | str
        | Path
        | Mapping[str, object]
        | None = None,
    ) -> advisor.AdvisorReport:
        """Record dispatched WorkloadSpec and forward to real `advise`."""
        captured_workloads.append(workload)
        return orig_advise(profile, workload, chip, plan_error, measurements)

    monkeypatch.setattr(cli, "advise", _capture_advise)
    assert (
        advisor.main(
            ["--model", "qwen3.5-397b", "--phase", "prefill", "--tp", "8", "--ep", "8"]
        )
        == 0
    )
    assert (
        advisor.main(
            ["--model", "qwen3.5-397b", "--phase", "decode", "--tp", "8", "--ep", "8"]
        )
        == 0
    )
    assert captured_workloads[0].osl == 0
    expected_decode_osl = 1024
    assert captured_workloads[1].osl == expected_decode_osl

    parser = advisor.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--model", "qwen3.5-397b", "--batch", "abc"])
    stderr_text = capsys.readouterr().err
    assert "invalid int value: 'abc'" in stderr_text
    assert "_MinIntArg(" not in stderr_text


def _assert_cli_core_json_fields(
    payload: Mapping[str, object],
    *,
    core_support: str,
    core_ready: bool,
    core_ready_scope: str,
    build_megakernel: bool,
) -> None:
    """Assert physics scope and core readiness fields on a CLI JSON payload."""
    assert payload["recommended_scope"] == "whole_step"
    assert payload["fastest_scope"] == "whole_step"
    assert payload["core_support"] == core_support
    assert payload["core_ready"] is core_ready
    assert payload["core_ready_scope"] == core_ready_scope
    assert payload["build_megakernel"] is build_megakernel
    assert payload["evidence"] == core_support
    assert "hypothetical_best" not in payload


def test_cli_json_and_plain_text_outputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify CLI plain-text, JSON output, `--state-dtype`, and missing paths."""
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(load_qwen35_snapshot_config()), encoding="utf-8")
    base_args = [
        "--config",
        str(config_file),
        "--chip",
        "tpu7x",
        "--tp",
        "8",
        "--ep",
        "8",
        "--json",
    ]
    assert advisor.main(base_args) == 0
    default_payload = json.loads(capsys.readouterr().out)
    _assert_cli_core_json_fields(
        default_payload,
        core_support="within_core_limits",
        core_ready=True,
        core_ready_scope="whole_step",
        build_megakernel=True,
    )
    assert default_payload["core_blockers"] == []

    assert advisor.main([*base_args, "--state-dtype", "fp32"]) == 0
    fp32_payload = json.loads(capsys.readouterr().out)
    _assert_cli_core_json_fields(
        fp32_payload,
        core_support="outside_core_limits",
        core_ready=False,
        core_ready_scope="per_op",
        build_megakernel=True,
    )
    assert "kv_or_state_dtype_not_in_core" in fp32_payload["core_blockers"]

    assert advisor.main([*base_args, "--state-dtype", "bf16"]) == 0
    bf16_payload = json.loads(capsys.readouterr().out)
    _assert_cli_core_json_fields(
        bf16_payload,
        core_support="within_core_limits",
        core_ready=True,
        core_ready_scope="whole_step",
        build_megakernel=True,
    )
    assert bf16_payload["recommended_moe_mode"] == "masked_on_chip"
    assert bf16_payload["core_blockers"] == []
    assert bf16_payload["hbm_pool_overflow"] is False
    assert bf16_payload["vmem_budget_mib"] == advisor.fit_vmem_budget_bytes(
        chip_spec=resolve_chip_spec("tpu7x")
    ) / (1024 * 1024)

    rc_sweep = advisor.main(
        [
            "--model",
            "qwen3.5-397b",
            "--regime-sweep",
            "--vmem-budget-mib",
            "60.0",
            "--dma-buffers",
            "2",
        ]
    )
    assert rc_sweep == 0
    sweep_stdout = capsys.readouterr().out
    assert "Decode Parallelism: TP8 x EP8 x DCP4" in sweep_stdout
    assert "limit=62.0 MiB, fit=60.0 MiB" in sweep_stdout

    sweep_json = tmp_path / "sweep.json"
    rc_sweep_json = advisor.main(
        [
            "--model",
            "qwen3.5-397b",
            "--regime-sweep",
            "--json",
            str(sweep_json),
        ]
    )
    assert rc_sweep_json == 0
    sweep_payload = json.loads(sweep_json.read_text(encoding="utf-8"))
    assert sweep_payload["model"] == "qwen3.5-397b"
    assert len(sweep_payload["regimes"]) >= EXPECTED_DEFAULT_REGIMES
    for reg in sweep_payload["regimes"]:
        assert {"core_support", "core_blockers", "core_ready", "core_ready_scope"} <= (
            reg.keys()
        )

    assert (
        advisor.main(
            [
                "--model",
                "moonshotai/Kimi-K3",
                "--tp",
                "32",
                "--ep",
                "32",
                "--vmem-fit-headroom",
                "--no-vmem-alias",
            ]
        )
        == 0
    )
    assert "Model: kimi-k3" in capsys.readouterr().out

    for bad_argv in (
        ["--config", str(tmp_path / "missing.json")],
        ["--model", str(tmp_path / "missing_dir")],
        ["--model", "org/unknown-preset"],
    ):
        with pytest.raises(MegakernelConfigError):
            advisor.main(bad_argv)


@pytest.mark.parametrize("model_slug", sorted(models.MODEL_PRESETS))
def test_text_reports_name_mechanisms_not_kernels(model_slug: str) -> None:
    """Verify plain-text regime and summary formatters name mechanisms only."""
    sweep = advisor.advise_regimes(model_slug, "tpu7x")
    dec_layout = advisor.DEPLOYMENT_PRESETS[model_slug].decode
    report = advisor.advise(
        model_slug,
        advisor.WorkloadSpec(
            phase="decode",
            batch=16 * dec_layout.dp,
            tp=dec_layout.tp,
            ep=dec_layout.ep,
            dp=dec_layout.dp,
            pp=dec_layout.pp,
            dcp=dec_layout.dcp,
            pcp=dec_layout.pcp,
        ),
        "tpu7x",
    )
    rendered = "\n".join(
        [
            advisor.format_regime_table(sweep, unresolved=True),
            advisor.format_advisor_summary(report, explain=True, unresolved=True),
            advisor.format_model_listing(),
        ]
    )
    assert KERNEL_NAME_DENYLIST.search(rendered) is None


def test_hbm_utilization_cap_default_and_cli_overflow(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify `--hbm-utilization-cap` defaults to 0.92 and separates OOM from cap."""
    parsed = advisor.build_parser().parse_args(["--model", "qwen3.5-397b"])
    assert parsed.hbm_utilization_cap == constants.HBM_UTILIZATION_CAP
    expected_cap = 0.92
    assert parsed.hbm_utilization_cap == expected_cap

    base_cmd = ["--model", "glm-5.2", "--tp", "16", "--ep", "16", "--batch", "16"]
    assert advisor.main([*base_cmd, "--json"]) == 0
    default_payload = json.loads(capsys.readouterr().out)
    assert default_payload["hbm_fits"] is True
    assert default_payload["hbm_pool_overflow"] is False

    assert advisor.main([*base_cmd, "--hbm-utilization-cap", "0.45", "--json"]) == 0
    capped_payload = json.loads(capsys.readouterr().out)
    assert capped_payload["hbm_fits"] is True
    assert capped_payload["hbm_pool_overflow"] is True

    assert advisor.main([*base_cmd, "--hbm-utilization-cap", "0.45"]) == 0
    summary_text = capsys.readouterr().out
    assert "hbm_pool_overflow=True" in summary_text
    assert "hbm_fits=True" in summary_text


@pytest.mark.parametrize("model_slug", sorted(models.MODEL_PRESETS))
def test_every_output_number_has_provenance_tag(model_slug: str) -> None:
    """Verify every metric in single-point and sweep dicts carries a valid tag."""
    valid_tags = {"DERIVED", "ASSUMED", "MEASURED"}
    required_keys = (
        "fits_hbm",
        "tokens_per_core",
        "onchip_ceiling",
        "compute_bound_at",
        "step_floor_us",
        "throughput_tok_s_per_chip",
        "boundary_cost_us",
        "fusion_gain_max",
        "step_coverage",
        "expected_gain_max",
    )
    dec_layout = advisor.DEPLOYMENT_PRESETS[model_slug].decode
    report = advisor.advise(
        model_slug,
        advisor.WorkloadSpec(
            phase="decode",
            batch=16 * dec_layout.dp,
            tp=dec_layout.tp,
            ep=dec_layout.ep,
            dp=dec_layout.dp,
            pp=dec_layout.pp,
            dcp=dec_layout.dcp,
            pcp=dec_layout.pcp,
        ),
        "tpu7x",
    )
    metrics = report.to_dict()["metrics"]
    assert isinstance(metrics, dict)
    for key in required_keys:
        assert key in metrics
    for entry in metrics.values():
        assert entry["tag"] in valid_tags
        assert bool(entry["source"])
        assert bool(entry["unit"])
        band = entry["band"]
        assert band[0] <= band[1]

    sweep = advisor.advise_regimes(model_slug, "tpu7x")
    raw_regimes = sweep.to_dict()["regimes"]
    assert isinstance(raw_regimes, list)
    for reg in raw_regimes:
        assert isinstance(reg, dict)
        reg_metrics = reg["metrics"]
        assert isinstance(reg_metrics, dict)
        for key in required_keys:
            assert key in reg_metrics
        for entry in reg_metrics.values():
            assert entry["tag"] in valid_tags
            assert entry["band"][0] <= entry["band"][1]


def test_single_layer_is_fused_layer_not_megakernel() -> None:
    """Verify one call per layer is reported as a fused layer, not a megakernel."""
    row = _regime("kimi-k3", "Decode Saturated")
    assert row.fastest_scope == "single_layer"
    assert row.fastest_verdict == "fused_layer_likely"
    assert _regime("kimi-k3", "Preset Prefill Window (W=2048)").fastest_verdict == (
        "megakernel_likely"
    )


def test_throughput_counts_all_window_tokens_per_physical_chip() -> None:
    """Verify tok/s/chip conserves DP * TP * PP * PCP / devices_per_chip."""
    row = _regime("kimi-k3", "Preset Prefill Window (W=2048)")
    expected = 2048 / 512 / (row.step_bound_us * 1e-6)
    got = float(row.metrics["throughput_tok_s_per_chip"]["value"])
    assert row.active_tokens == 2048 // 32
    assert got == pytest.approx(expected, rel=1e-3)

    dec = _regime("kimi-k3", "Decode Small")
    expected_decode = 4 / 16 / (dec.step_bound_us * 1e-6)
    got_decode = float(dec.metrics["throughput_tok_s_per_chip"]["value"])
    assert got_decode == pytest.approx(expected_decode, rel=1e-3)

    orth_report = advisor.advise(
        "kimi-k3",
        advisor.WorkloadSpec(phase="prefill", batch=1, isl=2048, tp=8, ep=32, pcp=4),
        "tpu7x",
    )
    assert orth_report.active_tokens == 2048 // 4
    orth_rec = next(
        sc for sc in orth_report.scopes if sc.scope == orth_report.recommended_scope
    )
    expected_orth = 2048 / 16 / (orth_rec.step_bound_us * 1e-6)
    got_orth = float(orth_report.metrics["throughput_tok_s_per_chip"]["value"])
    assert got_orth == pytest.approx(expected_orth, rel=1e-3)


@pytest.mark.parametrize("model_slug", sorted(models.MODEL_PRESETS))
def test_rationale_and_assumptions_across_sweeps(model_slug: str) -> None:
    """Verify runner-up margin formatting, coverage placement, and launch gaps."""
    row = _regime(model_slug, "Decode Small")
    assert row.step_upper_us > row.step_bound_us
    assert "vs runner-up " in row.rationale
    assert re.search(r"\+\d+\.\d us, \d+\.\d%\)", row.rationale)

    for reg in advisor.advise_regimes(model_slug, "tpu7x").regimes:
        assert "coverage_assumed" not in reg.flags
        assert "step_coverage" not in reg.verdict
        coverage_items = [
            item
            for item in reg.unresolved_assumptions
            if item["name"] == "step_coverage"
        ]
        assert all(item["affects"] == "gain" for item in coverage_items)

    dec_layout = advisor.DEPLOYMENT_PRESETS[model_slug].decode
    report = advisor.advise(
        model_slug,
        advisor.WorkloadSpec(
            phase="decode",
            batch=4 * dec_layout.dp,
            tp=dec_layout.tp,
            ep=dec_layout.ep,
            dp=dec_layout.dp,
            pp=dec_layout.pp,
            dcp=dec_layout.dcp,
            pcp=dec_layout.pcp,
        ),
        "tpu7x",
    )
    assert (
        specs.scalar_float(report.metrics["boundary_cost_us"]["value"], "boundary_cost")
        >= 0.0
    )
    assert report.metrics["launch_gap_us"]["tag"] == "ASSUMED"
    assert (
        specs.scalar_float(report.metrics["launch_gap_us"]["value"], "launch_gap") > 0.0
    )


def test_import_without_jax_and_torch() -> None:
    """Verify advisor is torch-free and restricts JAX to chip and bridge modules."""
    mk_init = Path(mk.__file__)
    adv_dir = mk_init.parent / "advisor"
    jax_allowed = frozenset({"chips.py", "spec_bridge.py"})
    for py_path in (mk_init, *sorted(adv_dir.glob("*.py"))):
        tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
        allow_jax = py_path.name in jax_allowed
        for ast_node in ast.walk(tree):
            if isinstance(ast_node, ast.Import):
                for alias in ast_node.names:
                    root = alias.name.split(".", 1)[0]
                    assert root != "torch", f"Unexpected torch import in {py_path}"
                    if not allow_jax:
                        assert root != "jax", (
                            f"Unexpected jax import in non-bridge {py_path}"
                        )
            elif isinstance(ast_node, ast.ImportFrom) and ast_node.module:
                root = ast_node.module.split(".", 1)[0]
                assert root != "torch", f"Unexpected torch import in {py_path}"
                if not allow_jax:
                    assert root != "jax", (
                        f"Unexpected jax import in non-bridge {py_path}"
                    )


def test_tpu7x_topologies_start_at_2x2x1_and_tpu6e_alias() -> None:
    """Verify `TPU7X_TOPOLOGIES` starts at `2x2x1` (8 cores) and `tpu6e` resolves."""
    min_torus_cores = 8
    assert min(advisor.TPU7X_TOPOLOGIES) == min_torus_cores
    assert advisor.TPU7X_TOPOLOGIES[8].startswith("2x2x1")
    for sub_host in (1, 2, 4):
        assert sub_host not in advisor.TPU7X_TOPOLOGIES
    c6e = advisor.resolve_chip_spec("tpu6e")
    assert c6e.name == "v6e"
    assert min(c6e.topologies) == 1
    par = advisor.recommend_parallelism(
        advisor.resolve_model_profile("qwen3.5-397b"), "tpu7x"
    )
    assert par.tp * par.dp * par.pp >= min_torus_cores


def _assert_qwen_disagg_regimes() -> advisor.AdvisorSweepReport:
    """Assert Qwen3.5 default bf16 and explicit fp32 P/D and saturation handoffs."""
    q_sweep = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    q_by_lbl = {entry.label: entry for entry in q_sweep.regimes}
    q_sat = next(
        entry for entry in q_sweep.regimes if entry.label.startswith("Decode Saturated")
    )
    q_dec_med = q_by_lbl["Decode Med (B=16, T=1, 4K)"]
    _assert_core_readiness_entry(
        q_dec_med,
        recommended_scope="whole_step",
        core_support="within_core_limits",
        core_ready=True,
        core_ready_scope="whole_step",
    )
    assert q_dec_med.fastest_scope == "whole_step"
    assert q_dec_med.core_blockers == ()
    q_dec_high = q_by_lbl["Decode High (B=64, T=1, 32K)"]
    _assert_core_readiness_entry(
        q_dec_high,
        recommended_scope="whole_step",
        core_support="outside_core_limits",
        core_ready=False,
        core_ready_scope="per_op",
    )
    assert q_dec_high.fastest_scope == "whole_step"
    assert q_dec_high.hbm_fits is True
    assert q_sat.layout == "TP2 x DP4 x EP8"
    assert q_sat.hbm_fits is True
    assert q_sat.hbm_pool_overflow is True
    assert q_sat.num_vmem_splits == 1
    _assert_core_readiness_entry(
        q_sat,
        recommended_scope="whole_step",
        core_support="outside_core_limits",
        core_ready=False,
        core_ready_scope="per_op",
    )
    q_prefill = q_by_lbl["Preset Prefill Window (W=2176)"]
    assert q_prefill.layout == "PP8"
    _assert_core_readiness_entry(
        q_prefill,
        recommended_scope="single_layer",
        core_support="outside_core_limits",
        core_ready=False,
        core_ready_scope="per_op",
    )
    assert "pipeline_parallel_not_in_core" in q_prefill.core_blockers
    assert q_sweep.disagg_handoff is not None
    assert q_sweep.disagg_handoff.dcn_provenance == "ASSUMED"
    default_dcn_bw_gbps = 100.0
    assert q_sweep.disagg_handoff.dcn_bw_gbps == default_dcn_bw_gbps
    assert q_sweep.disagg_handoff.dcn_bw_gbps_bracket == (50.0, 200.0)

    q_fp32_sweep = advisor.advise_regimes(
        advisor.resolve_model_profile("qwen3.5-397b", state_dtype="fp32"), "tpu7x"
    )
    q_fp32_med = {entry.label: entry for entry in q_fp32_sweep.regimes}[
        "Decode Med (B=16, T=1, 4K)"
    ]
    _assert_core_readiness_entry(
        q_fp32_med,
        recommended_scope="whole_step",
        core_support="outside_core_limits",
        core_ready=False,
        core_ready_scope="per_op",
    )
    assert "kv_or_state_dtype_not_in_core" in q_fp32_med.core_blockers
    q_fp32_sat = next(
        entry
        for entry in q_fp32_sweep.regimes
        if entry.label.startswith("Decode Saturated")
    )
    assert q_fp32_sat.hbm_fits is False
    assert q_fp32_sat.recommended_scope == "per_op"
    assert "[PHYSICAL OOM]" in advisor.format_regime_table(q_fp32_sweep)
    return q_sweep


def test_pd_disagg_alignment_qwen35_kimi_k3_and_dcn_handoff(
    tmp_path: Path,
) -> None:
    """Verify Qwen3.5-397B, Kimi-K3, and Qwen physical OOM saturation handoffs."""
    _assert_qwen_disagg_regimes()

    k_sweep = advisor.advise_regimes("kimi-k3", "tpu7x")
    k_by_lbl = {entry.label: entry for entry in k_sweep.regimes}
    _assert_core_readiness_entry(
        k_by_lbl["Decode Med (B=16, T=1, 4K)"],
        recommended_scope="whole_step",
        core_support="outside_core_limits",
        core_ready=False,
        core_ready_scope="per_op",
    )
    assert k_by_lbl["Preset Prefill Window (W=2048)"].layout == "TP32 x EP32 x PCP32"
    _assert_core_readiness_entry(
        k_by_lbl["Preset Prefill Window (W=2048)"],
        recommended_scope="layer_group",
        core_support="outside_core_limits",
        core_ready=False,
        core_ready_scope="per_op",
    )
    assert k_by_lbl["Preset Prefill Window (W=8192)"].recommended_scope == "layer_group"

    measurements_file = tmp_path / "dcn_meas.json"
    measurements_file.write_text(
        json.dumps(
            {
                "schema": "advisor.measurements.v1",
                "source": {"kind": "xplane", "file": "dcn.xplane.pb"},
                "dcn_bw_gbps": 160.0,
            }
        ),
        encoding="utf-8",
    )
    q_meas = advisor.advise_regimes(
        "qwen3.5-397b", "tpu7x", measurements=measurements_file
    )
    assert q_meas.disagg_handoff is not None
    assert q_meas.disagg_handoff.dcn_provenance == "MEASURED"
    measured_dcn_bw_gbps = 160.0
    assert q_meas.disagg_handoff.dcn_bw_gbps == measured_dcn_bw_gbps


def test_glm52_regime_sweep_and_disagg_indexer_wire_bytes() -> None:
    """Verify GLM-5.2 regime sweep separates physics winners from core blockers."""
    glm_profile = advisor.resolve_model_profile("glm-5.2")
    sweep = advisor.advise_regimes(glm_profile, "tpu7x")
    assert all(
        entry.core_ready is False and entry.core_ready_scope == "per_op"
        for entry in sweep.regimes
    )
    assert any(
        entry.recommended_scope == "whole_step"
        and entry.core_support == "outside_core_limits"
        and "family_not_in_core" in entry.core_blockers
        for entry in sweep.regimes
    )
    assert sweep.disagg_handoff is not None
    # ISL=4096 wire payload: 78*4096*(512+64)*1 + 21*4096*(128+4) = 195.38 MB.
    expected_kv_wire_mb = 195.38
    assert sweep.disagg_handoff.kv_wire_mb == pytest.approx(
        expected_kv_wire_mb, abs=0.01
    )
    doubled = _compute_disagg_handoff(
        glm_profile,
        resolve_chip_spec("tpu7x"),
        advisor.DEPLOYMENT_PRESETS["glm-5.2"],
        None,
        isl=2 * sweep.disagg_handoff.isl,
    )
    assert doubled is not None
    assert doubled.kv_wire_mb == pytest.approx(2.0 * expected_kv_wire_mb, abs=0.02)
    table_text = advisor.format_regime_table(sweep)
    assert "outside_core_limits" in table_text
    assert "family_not_in_core" in table_text
    assert (
        f"Disaggregated Prefill/Decode DCN Handoff (ISL={sweep.disagg_handoff.isl})"
        in table_text
    )
    assert KERNEL_NAME_DENYLIST.search(table_text) is None


def _assert_capped_crossover(
    qwen_bf16: advisor.ModelProfile, base_workload: advisor.WorkloadSpec, b_cross: int
) -> None:
    """Assert explicit `max_rows_per_call` vs VMEM cap crossover and core blockers."""
    at_capped = advisor.advise(
        qwen_bf16,
        dataclasses.replace(
            base_workload,
            batch=b_cross,
            max_rows_per_call=b_cross,
            batch_explicit=True,
        ),
        "tpu7x",
    )
    above_capped = advisor.advise(
        qwen_bf16,
        dataclasses.replace(
            base_workload,
            batch=b_cross + 1,
            max_rows_per_call=b_cross,
            batch_explicit=True,
        ),
        "tpu7x",
    )
    assert at_capped.fastest_scope == "whole_step"
    assert at_capped.recommended_scope == "whole_step"
    assert at_capped.core_support == "outside_core_limits"
    assert at_capped.core_ready is False
    assert at_capped.core_ready_scope == "per_op"
    assert "rows_exceed_core_max_batch_rows" in at_capped.core_blockers
    at_capped_whole = next(s for s in at_capped.scopes if s.scope == "whole_step")
    assert "rows_exceed_core_max_batch_rows" in at_capped_whole.blockers
    assert above_capped.recommended_scope == "whole_step"
    assert "microbatch_rerun" in above_capped.flags
    assert above_capped.core_support == "outside_core_limits"
    assert above_capped.core_ready is False
    assert above_capped.core_ready_scope == "per_op"
    assert "rows_exceed_core_max_batch_rows" in above_capped.core_blockers


def test_analytical_crossover_batch_flips_recommendation() -> None:
    """Derive exact VMEM batch crossover from physics and verify flip at `b + 1`."""
    qwen_bf16 = advisor.resolve_model_profile("qwen3.5-397b", state_dtype="bf16")
    base_workload = advisor.WorkloadSpec(
        phase="decode",
        batch=1,
        context=4096,
        tp=8,
        ep=8,
        dcp=4,
    )
    base_report = advisor.advise(qwen_bf16, base_workload, "tpu7x")
    base_whole_step = next(
        scope_est for scope_est in base_report.scopes if scope_est.scope == "whole_step"
    )
    b_cross = base_whole_step.max_vmem_tokens
    assert b_cross > 1

    at_cross = advisor.advise(
        qwen_bf16,
        dataclasses.replace(base_workload, batch=b_cross, batch_explicit=True),
        "tpu7x",
    )
    above_cross = advisor.advise(
        qwen_bf16,
        dataclasses.replace(base_workload, batch=b_cross + 1, batch_explicit=True),
        "tpu7x",
    )
    at_whole = next(s for s in at_cross.scopes if s.scope == "whole_step")
    above_whole = next(s for s in above_cross.scopes if s.scope == "whole_step")
    assert at_whole.num_vmem_splits == 1
    assert at_cross.fastest_scope == "whole_step"
    expected_split_passes = 2
    assert above_whole.num_vmem_splits == expected_split_passes
    assert above_cross.fastest_scope == "per_op"
    assert above_cross.recommended_scope == "per_op"

    _assert_capped_crossover(qwen_bf16, base_workload, b_cross)

    small_vmem_mib = 8.0
    small_base = dataclasses.replace(base_workload, vmem_budget_mib=small_vmem_mib)
    small_report = advisor.advise(qwen_bf16, small_base, "tpu7x")
    small_cross = next(
        s for s in small_report.scopes if s.scope == "whole_step"
    ).max_vmem_tokens
    assert 1 < small_cross <= constants.CORE_LIMITS.max_batch_rows
    at_small = advisor.advise(
        qwen_bf16,
        dataclasses.replace(
            small_base,
            batch=small_cross,
            max_rows_per_call=small_cross,
            batch_explicit=True,
        ),
        "tpu7x",
    )
    above_small = advisor.advise(
        qwen_bf16,
        dataclasses.replace(
            small_base,
            batch=small_cross + 1,
            max_rows_per_call=small_cross,
            batch_explicit=True,
        ),
        "tpu7x",
    )
    assert at_small.recommended_scope == "whole_step"
    assert at_small.core_support == "within_core_limits"
    assert at_small.core_ready is True
    assert at_small.core_ready_scope == "whole_step"
    assert above_small.recommended_scope == "per_op"


def test_advisor_cli_with_bundled_measurements(tmp_path: Path) -> None:
    """Verify CLI `estimate` with `--measurements bundled` and missing-chip error."""
    out_json = tmp_path / "bundled_out.json"
    base_argv = [
        "estimate",
        "--model",
        "qwen3.5-397b",
        "--phase",
        "decode",
        "--batch",
        "16",
        "--tp",
        "8",
        "--ep",
        "8",
        "--measurements",
        "bundled",
    ]
    rc = advisor.main([*base_argv, "--chip", "tpu7x", "--json", str(out_json)])
    assert rc == 0
    assert out_json.is_file()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["confidence"] == "prior_only"
    assert payload["metrics"]["collective_latency_us"]["tag"] == "ASSUMED"
    assert payload["metrics"]["launch_gap_us"]["tag"] == "ASSUMED"

    with pytest.raises(MegakernelConfigError, match="bundled architectural priors"):
        advisor.main([*base_argv, "--chip", "tpu6e"])


def test_cli_invalid_sharding_exits_nonzero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify invalid sharding exits 2 on soft mismatches and raises on hard ones."""
    rc = advisor.main(["--model", "qwen3.5-397b", "--chip", "tpu7x", "--tp", "2"])
    assert rc == constants.INVALID_CONFIG_EXIT_CODE
    out = capsys.readouterr().out
    assert "invalid_config(" in out

    rc_sweep = advisor.main(
        ["--model", "qwen3.5-397b", "--chip", "tpu7x", "--tp", "2", "--regime-sweep"]
    )
    assert rc_sweep == constants.INVALID_CONFIG_EXIT_CODE
    sweep_out = capsys.readouterr().out
    assert "invalid_config(" in sweep_out

    rc_search = advisor.main(
        [
            "--model",
            "qwen3.5-397b",
            "--chip",
            "tpu7x",
            "--tp",
            "2",
            "--search-layer-groups",
            "--json",
        ]
    )
    assert rc_search == constants.INVALID_CONFIG_EXIT_CODE
    search_payload = json.loads(capsys.readouterr().out)
    assert search_payload["best"] is None

    for extra in ([], ["--regime-sweep"], ["--search-layer-groups"]):
        with pytest.raises(MegakernelConfigError, match="tp_not_power_of_two"):
            advisor.main(
                ["--model", "qwen3.5-397b", "--chip", "tpu7x", "--tp", "7", *extra]
            )


def test_cli_prints_evidence_blockers_and_inputs(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Verify summary prints physics winner, Core Readiness, Evidence, and blockers."""
    glm_snapshot = json.loads(
        (HF_CONFIG_DIR / "glm-5.2.json").read_text(encoding="utf-8")
    )
    config_path = tmp_path / "glm_5_2_config.json"
    config_path.write_text(json.dumps(glm_snapshot["config"]), encoding="utf-8")
    rc = advisor.main(["--model", str(config_path), "--tp", "16", "--explain"])
    assert rc == 0
    out = capsys.readouterr().out
    for marker in (
        "Recommended: whole_step",
        (
            "Core Readiness: outside_core_limits "
            "(core_ready=False, core_ready_scope=per_op"
        ),
        "Evidence: outside_core_limits",
        "Blocked by:",
        "family_not_in_core",
        "Inputs:",
    ):
        assert marker in out


def test_cli_max_rows_per_call_prices_microbatches(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify `--max-rows-per-call` caps `whole_step` only in CLI and sweep."""
    base = ["--model", "qwen3.5-397b", "--chip", "tpu7x", "--state-dtype", "bf16"]
    mesh = ["--tp", "8", "--dcp", "4", "--ep", "8"]
    verify = ["--phase", "verify", "--batch", "28", "--spec-tokens", "3", "--json"]
    limit = ["--max-rows-per-call", str(TEST_ROW_LIMIT)]
    assert advisor.main([*base, *mesh, *verify, *limit]) == 0
    split = json.loads(capsys.readouterr().out)
    whole = next(s for s in split["scopes"] if s["scope"] == "whole_step")
    assert whole["max_vmem_tokens"] == TEST_ROW_LIMIT
    assert whole["num_vmem_splits"] > 1
    assert "microbatch_rerun" in split["flags"]

    assert advisor.main([*base, *mesh, *verify]) == 0
    assert "microbatch_rerun" not in json.loads(capsys.readouterr().out)["flags"]

    assert advisor.main([*base, *mesh, "--regime-sweep", "--json", *limit]) == 0
    assert json.loads(capsys.readouterr().out)["max_vmem_tokens"] == TEST_ROW_LIMIT

    qwen_bf16 = advisor.resolve_model_profile("qwen3.5-397b", state_dtype="bf16")
    uncapped_wl = advisor.WorkloadSpec(
        phase="verify",
        batch=32,
        spec_tokens=3,
        context=32768,
        tp=8,
        dcp=4,
        ep=8,
        layer_groups=(8, 8, 8, 8, 8, 8, 8, 4),
    )
    capped_wl = dataclasses.replace(uncapped_wl, max_rows_per_call=TEST_ROW_LIMIT)
    uncapped = {
        s.scope: s for s in advisor.advise(qwen_bf16, uncapped_wl, "tpu7x").scopes
    }
    capped_rep = advisor.advise(qwen_bf16, capped_wl, "tpu7x")
    capped = {s.scope: s for s in capped_rep.scopes}
    assert uncapped["whole_step"].max_vmem_tokens > TEST_ROW_LIMIT
    assert capped["whole_step"].max_vmem_tokens == TEST_ROW_LIMIT
    assert capped["whole_step"].num_vmem_splits > uncapped["whole_step"].num_vmem_splits
    assert capped["layer_group"].max_vmem_tokens == TEST_ROW_LIMIT
    assert (
        capped["layer_group"].num_vmem_splits > uncapped["layer_group"].num_vmem_splits
    )
    for scope_name in ("per_op", "single_layer"):
        assert (
            capped[scope_name].max_vmem_tokens == uncapped[scope_name].max_vmem_tokens
        )
        assert (
            capped[scope_name].num_vmem_splits == uncapped[scope_name].num_vmem_splits
        )
        assert capped[scope_name].step_bound_us == uncapped[scope_name].step_bound_us


def test_regime_entry_uses_recommended_scope_not_hypothetical() -> None:
    """Verify `RegimeEntry` tracks physics `recommended_scope` and core readiness."""
    k3_sweep = advisor.advise_regimes("kimi-k3", "tpu7x")
    dec_small = next(r for r in k3_sweep.regimes if r.label.startswith("Decode Small"))
    assert dec_small.recommended_scope == "whole_step"
    assert dec_small.fastest_scope == "whole_step"
    assert dec_small.core_support == "outside_core_limits"
    assert dec_small.core_ready is False
    assert dec_small.core_ready_scope == "per_op"
    assert "kv_or_state_dtype_not_in_core" in dec_small.core_blockers
    assert dec_small.core_support == "outside_core_limits"
    whole_sc = next(s for s in dec_small.scopes if s["scope"] == "whole_step")
    assert dec_small.step_bound_us == whole_sc["step_bound_us"]
    assert dec_small.step_upper_us == whole_sc["step_upper_us"]
    assert dec_small.max_vmem_tokens == whole_sc["max_vmem_tokens"]
    serialized = dec_small.to_dict()
    assert serialized["core_support"] == "outside_core_limits"
    assert serialized["core_ready"] is False
    assert serialized["core_ready_scope"] == "per_op"
    assert "kv_or_state_dtype_not_in_core" in cast(
        "tuple[str, ...] | list[str]", serialized["core_blockers"]
    )

    q_sweep = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    q_ref = advisor.advise(
        "qwen3.5-397b",
        advisor.WorkloadSpec(
            phase="decode",
            batch=constants.RECOMMEND_DEFAULT_BATCH,
            tp=q_sweep.parallelism.tp,
            ep=q_sweep.parallelism.ep,
            dp=q_sweep.parallelism.dp,
            pp=q_sweep.parallelism.pp,
            dcp=q_sweep.parallelism.dcp,
            pcp=q_sweep.parallelism.pcp,
        ),
        "tpu7x",
    )
    q_ref_whole = next(s for s in q_ref.scopes if s.scope == "whole_step")
    assert q_sweep.max_vmem_tokens == q_ref_whole.max_vmem_tokens


def test_recommend_parallelism_raises_and_handles_v6e_dcp_and_options() -> None:
    """Verify `recommend_parallelism` rejects OOM/invalid phase and honors EP/DCP."""
    model_profile = advisor.resolve_model_profile("kimi-k3")
    tiny_chip = dataclasses.replace(
        advisor.resolve_chip_spec("tpu7x"),
        hbm_capacity_bytes=1024 * 1024,
    )
    with pytest.raises(MegakernelConfigError, match="fits in HBM"):
        advisor.recommend_parallelism(model_profile, tiny_chip)

    with pytest.raises(MegakernelConfigError, match="Unsupported phase"):
        advisor.ParallelismSearchOptions(phase=cast("advisor.Phase", "bogus"))

    v6e_spec = advisor.resolve_chip_spec("v6e")
    qwen35_profile = advisor.resolve_model_profile("qwen3.5-397b")
    par_v6e = advisor.recommend_parallelism(qwen35_profile, v6e_spec)
    assert par_v6e.tp * par_v6e.dp * par_v6e.pp in v6e_spec.topologies
    v6e_sweep = advisor.advise_regimes(qwen35_profile, "v6e")
    assert not v6e_sweep.regimes[0].verdict.startswith("invalid_config(")

    dp2_sweep = advisor.advise_regimes(
        "qwen3.5-397b",
        "tpu7x",
        options=advisor.RegimeSweepOptions(tp=4, dp=2),
    )
    assert len(dp2_sweep.regimes) == EXPECTED_DEFAULT_REGIMES
    for reg in dp2_sweep.regimes:
        assert not reg.verdict.startswith("invalid_config(")
        assert int(str(reg.workload["batch"])) >= int(str(reg.workload["dp"]))

    expected_tp = 8
    expected_ep = 8
    rec_default = advisor.recommend_parallelism(
        qwen35_profile,
        "tpu7x",
        options=advisor.ParallelismSearchOptions(dcp=0),
    )
    assert (rec_default.tp, rec_default.ep, rec_default.dcp) == (
        expected_tp,
        expected_ep,
        1,
    )
    expected_dcp = 4
    rec_explicit = advisor.recommend_parallelism(
        qwen35_profile,
        "tpu7x",
        options=advisor.ParallelismSearchOptions(dcp=expected_dcp, phase="verify"),
    )
    assert (rec_explicit.tp, rec_explicit.ep, rec_explicit.dcp) == (
        expected_tp,
        expected_ep,
        expected_dcp,
    )
    rec_ep_dp = advisor.recommend_parallelism(
        qwen35_profile,
        "tpu7x",
        options=advisor.ParallelismSearchOptions(dp=4, ep=2),
    )
    expected_dp4_tp, expected_dp4_ep = 4, 2
    assert (rec_ep_dp.tp, rec_ep_dp.ep, rec_ep_dp.dp) == (
        expected_dp4_tp,
        expected_dp4_ep,
        4,
    )


def test_cli_and_text_render_layer_group_search(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Verify `--search-layer-groups` plain-text and JSON outputs."""
    rc = advisor.main(
        [
            "--model",
            "kimi-k3",
            "--phase",
            "prefill",
            "--isl",
            "512",
            "--tp",
            "32",
            "--ep",
            "16",
            "--pcp",
            "32",
            "--search-layer-groups",
        ]
    )
    assert rc == 0
    stdout_text = capsys.readouterr().out
    assert "Layer-Group Partition Search: kimi-k3" in stdout_text
    assert "one fused kernel per group" in stdout_text

    json_out = tmp_path / "lg_search.json"
    rc_json = advisor.main(
        [
            "--model",
            "kimi-k3",
            "--phase",
            "prefill",
            "--isl",
            "512",
            "--tp",
            "32",
            "--ep",
            "16",
            "--pcp",
            "32",
            "--search-layer-groups",
            "--json",
            str(json_out),
        ]
    )
    assert rc_json == 0
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["best"]["num_groups"] == 1


def test_cli_decoding_sequences_littles_law_and_host_head_flags(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify Little's Law decoding sequences, host-head flags, and `--unresolved`."""
    # Closed-form Little's Law: 605.0 * 13.42 / 1000.0 = 8.1191 -> 8 sequences.
    expected_exact_seqs = 8.1191
    expected_rounded_seqs = 8
    exact_seqs, rounded_seqs = advisor.derive_decoding_sequences(605.0, 13.42)
    assert exact_seqs == pytest.approx(expected_exact_seqs, abs=1e-6)
    assert rounded_seqs == expected_rounded_seqs

    rc_littles = advisor.main(
        [
            "estimate",
            "--model",
            "qwen3.5-397b",
            "--chip",
            "tpu7x",
            "--phase",
            "decode",
            "--output-tok-s",
            "605.0",
            "--mean-tpot-ms",
            "13.42",
            "--tp",
            "8",
            "--ep",
            "8",
            "--dcp",
            "4",
            "--json",
        ]
    )
    assert rc_littles == 0
    littles_payload = json.loads(capsys.readouterr().out)
    assert littles_payload["active_tokens"] == expected_rounded_seqs

    cal_args = [
        "estimate",
        "--model",
        "qwen3.5-397b",
        "--state-dtype",
        "bf16",
        "--chip",
        "tpu7x",
        "--phase",
        "decode",
        "--decoding-seqs",
        "8",
        "--tp",
        "8",
        "--ep",
        "8",
        "--dcp",
        "4",
        "--host-head-us",
        "per_op=1800,fused=350",
        "--per-op-step-us",
        "9500",
        "--step-coverage",
        "0.85",
    ]
    assert advisor.main([*cal_args, "--json"]) == 0
    cal_payload = json.loads(capsys.readouterr().out)
    assert cal_payload["gain_basis"] == constants.GAIN_BASIS_CALIBRATED_E2E
    scopes_by_name = {s["scope"]: s for s in cal_payload["scopes"]}
    expected_per_op_head = 1800.0
    expected_fused_head = 350.0
    assert scopes_by_name["per_op"]["host_head_us"] == expected_per_op_head
    assert scopes_by_name["whole_step"]["host_head_us"] == expected_fused_head

    assert advisor.main(cal_args) == 0
    assert "Whole-server E2E gain (calibrated):" in capsys.readouterr().out

    rc_unresolved = advisor.main(
        [
            "estimate",
            "--model",
            "qwen3.5-397b",
            "--chip",
            "tpu7x",
            "--phase",
            "decode",
            "--tp",
            "8",
            "--ep",
            "8",
            "--dcp",
            "4",
            "--unresolved",
        ]
    )
    assert rc_unresolved == 0
    unres_out = capsys.readouterr().out
    assert "Unresolved Assumptions:" in unres_out
    assert "assumed=" in unres_out
    assert "| affects=" in unres_out
    assert "| settle:" in unres_out
    assert "decoding_sequences:" in unres_out


def test_cli_list_models_physics_only_and_reference_topologies(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify `--list-models` lists all shipped models and pure 6-axis layouts."""
    exit_code = advisor.main(["--list-models"])
    assert exit_code == 0
    listing_text = capsys.readouterr().out
    assert "tpu7x" in listing_text
    assert "v6e" in listing_text
    assert "chip_without_core_budget" in listing_text
    assert "priced from hardware rates only" in listing_text
    assert "tier=" not in listing_text
    assert "declared megakernel" not in listing_text
    assert "not modeled:" not in listing_text

    assert len(advisor.MODEL_PRESETS) > 0
    for slug in advisor.MODEL_PRESETS:
        assert slug in listing_text
        assert advisor.models.MODEL_PRESET_DTYPE_BASIS[slug] == "config"
        deployment = advisor.DEPLOYMENT_PRESETS[slug]
        assert deployment.decode.label() in listing_text
        assert deployment.decode.submodule_label() in listing_text


def test_format_report_fastest_scope_and_low_gain_phrasing() -> None:
    """Verify report formatting surfaces core-readiness, blockers, and plain terms."""
    kimi_table = advisor.format_regime_table(advisor.advise_regimes("kimi-k3", "tpu7x"))
    assert "Intensity (FLOP/B)" in kimi_table
    assert "Core Support" in kimi_table
    assert "Core-Ready" in kimi_table
    assert "Core Blockers" in kimi_table
    assert "outside_core_limits" in kimi_table
    assert "core_ready=per_op" in kimi_table
    assert "kv_or_state_dtype_not_in_core" in kimi_table
    assert "% vs per_op)" in kimi_table
    assert "Fusion gain (estimate)=" in kimi_table

    glm_table = advisor.format_regime_table(advisor.advise_regimes("glm-5.2", "tpu7x"))
    assert "[LOW PHYSICS GAIN <1%]" in glm_table
    assert "family_not_in_core" in glm_table


def test_saturation_batch_shared_between_summary_and_regime_rows() -> None:
    """Verify decode and verify saturation rows share unified `saturation_batch`."""
    sweep = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    decode_sat = next(
        entry for entry in sweep.regimes if entry.label.startswith("Decode Saturated")
    )
    verify_sat = next(
        entry
        for entry in sweep.regimes
        if entry.label.startswith("Spec Verify Saturated")
    )
    sat_batch = int(str(decode_sat.workload["batch"]))
    assert f"B={sat_batch}" in decode_sat.label
    verify_rows = int(str(verify_sat.workload["batch"])) * (
        1 + int(str(verify_sat.workload["spec_tokens"]))
    )
    assert verify_rows >= sat_batch

    sat_dp2_layout = models.ParallelLayout(tp=8, ep=8, dp=2, spec_tokens=3)
    expected_unified_batch = 512
    for decode_dp, raw_batch in ((3, 258), (5, 260)):
        decode_layout = models.ParallelLayout(tp=8, ep=8, dp=decode_dp, spec_tokens=3)
        preset = models.DeploymentPreset(
            decode=decode_layout,
            prefill=sat_dp2_layout,
            saturation=sat_dp2_layout,
        )
        base_wl = advisor.WorkloadSpec(
            phase="decode", batch=raw_batch, tp=8, ep=8, dp=decode_dp
        )
        moved = _with_preset_prefill(
            (
                (f"Decode Saturated (B={raw_batch}, T=1, 32K)", base_wl),
                ("Spec Verify High (B=32, K=3)", base_wl),
            ),
            preset,
        )
        moved_dec_label, moved_dec_wl = next(
            (lbl, wl) for lbl, wl in moved if lbl.startswith("Decode Saturated")
        )
        moved_ver_label, moved_ver_wl = next(
            (lbl, wl) for lbl, wl in moved if lbl.startswith("Spec Verify Saturated")
        )
        assert moved_dec_wl.batch == expected_unified_batch
        assert moved_ver_wl.batch == expected_unified_batch
        assert f"B={expected_unified_batch}" in moved_dec_label
        assert f"B={expected_unified_batch}" in moved_ver_label


def test_cli_accepts_model_dir_and_hbm_utilization_cap(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify `--config` resolves local directories and honors HBM cap in reports."""
    model_dir = tmp_path / "qwen_local_dir"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(
        json.dumps(load_qwen35_snapshot_config()), encoding="utf-8"
    )
    cap_fraction = 0.85
    rc = advisor.main(
        [
            "--config",
            str(model_dir),
            "--chip",
            "tpu7x",
            "--tp",
            "8",
            "--ep",
            "8",
            "--hbm-utilization-cap",
            str(cap_fraction),
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["hbm_utilization_cap"] == pytest.approx(cap_fraction)


def test_advise_regimes_explicit_ep_without_tp_picks_hbm_fitting_tp() -> None:
    """Verify explicit EP without TP evaluates HBM fit using requested EP."""
    sweep = advisor.advise_regimes(
        "qwen3.5-397b",
        "tpu7x",
        options=advisor.RegimeSweepOptions(dp=4, ep=2),
    )
    expected_tp, expected_ep, expected_dcp = 4, 2, 2
    assert sweep.parallelism.tp == expected_tp
    assert sweep.parallelism.ep == expected_ep
    assert sweep.parallelism.dcp == expected_dcp
    direct = advisor.advise(
        "qwen3.5-397b",
        advisor.WorkloadSpec(
            phase="decode",
            batch=constants.RECOMMEND_DEFAULT_BATCH,
            tp=expected_tp,
            dp=4,
            ep=expected_ep,
            dcp=expected_dcp,
            pcp=sweep.parallelism.pcp,
        ),
        "tpu7x",
    )
    assert sweep.parallelism.hbm_used_gb == pytest.approx(direct.hbm_used_gb)
    assert all(
        entry.hbm_fits
        for entry in sweep.regimes
        if not entry.label.startswith("Decode Saturated")
    )

    sweep_dcp1 = advisor.advise_regimes(
        "qwen3.5-397b",
        "tpu7x",
        options=advisor.RegimeSweepOptions(dp=4, ep=2, dcp=1),
    )
    expected_dcp1_tp = 8
    assert sweep_dcp1.parallelism.tp == expected_dcp1_tp
    assert sweep_dcp1.parallelism.ep == expected_ep
    assert sweep_dcp1.parallelism.dcp == 1
    direct_dcp1 = advisor.advise(
        "qwen3.5-397b",
        advisor.WorkloadSpec(
            phase="decode",
            batch=constants.RECOMMEND_DEFAULT_BATCH,
            tp=expected_dcp1_tp,
            dp=4,
            ep=expected_ep,
            dcp=1,
            pcp=sweep_dcp1.parallelism.pcp,
        ),
        "tpu7x",
    )
    assert sweep_dcp1.parallelism.hbm_used_gb == pytest.approx(direct_dcp1.hbm_used_gb)
    assert all(entry.hbm_fits for entry in sweep_dcp1.regimes)


@pytest.mark.parametrize("model_slug", sorted(models.MODEL_PRESETS))
def test_cli_sweep_json_matches_advise_regimes_exact(
    model_slug: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify CLI `--regime-sweep --json` matches `advise_regimes` JSON verbatim."""
    rc = advisor.main(["--model", model_slug, "--regime-sweep", "--json"])
    assert rc == 0
    cli_payload = json.loads(capsys.readouterr().out)
    sweep = advisor.advise_regimes(model_slug, "tpu7x")
    api_payload = json.loads(json.dumps(sweep.to_dict()))
    assert cli_payload == api_payload
    assert all(
        item["name"] != "decoding_sequences"
        for item in sweep.regimes[0].unresolved_assumptions
    )


def test_regime_table_vmem_header_names_single_layer_when_whole_step_infeasible() -> (
    None
):
    """Verify sweep VMEM header labels `single_layer` when `whole_step` overflows."""
    sweep = advisor.advise_regimes(
        "qwen3.5-397b",
        "tpu7x",
        options=advisor.RegimeSweepOptions(
            base_workload=advisor.WorkloadSpec(vmem_budget_mib=6.5)
        ),
    )
    table = advisor.format_regime_table(sweep)
    assert "(single_layer, 2-slot DMA)" in table
    assert "Ideal-kernel VMEM ceiling (whole_step," not in table


def test_text_format_constants_decode_ref_and_none_ring_safety() -> None:
    """Verify gain constant, None ring safety, empty bands, and Decode Med ref."""
    assert text_format.GAIN_BASIS_CALIBRATED_E2E is constants.GAIN_BASIS_CALIBRATED_E2E
    base_report = advisor.advise(
        "qwen3.5-397b",
        advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8),
        "tpu7x",
    )
    empty_band_metrics = dict(base_report.metrics)
    empty_band_metrics["launch_gap_us"] = {
        **base_report.metrics["launch_gap_us"],
        "band": [],
    }
    patched_report = dataclasses.replace(base_report, metrics=empty_band_metrics)
    rendered = text_format.format_advisor_summary(patched_report, explain=True)
    assert "Provenance Breakdown:" in rendered

    sweep = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    first_entry = sweep.regimes[0]
    none_ring_scopes = tuple(
        {**sc, "selected_dense_slots": None, "selected_expert_slots": None}
        for sc in first_entry.scopes
    )
    patched_sweep = dataclasses.replace(
        sweep,
        regimes=(dataclasses.replace(first_entry, scopes=none_ring_scopes),),
    )
    assert "ring=-" in text_format.format_regime_table(patched_sweep)

    decode_med = next(
        entry for entry in sweep.regimes if entry.label.startswith("Decode Med")
    )
    med_step_item = next(
        item
        for item in decode_med.unresolved_assumptions
        if item["name"] == "per_op_step_us"
    )
    table = text_format.format_regime_table(sweep, unresolved=True)
    unres_section = table.split("Unresolved Assumptions (Decode Reference):", 1)[1]
    assert f"per_op_step_us: assumed={med_step_item['value']}" in unres_section


def test_tailoring_guide_derives_row_tile_and_handles_dense_models() -> None:
    """Verify tailoring guide derives row tile, dense rules, and core-limit note."""
    v6e_sweep = advisor.advise_regimes("kimi-k3", "v6e")
    assert "T < 8" in v6e_sweep.tailoring_guide[0]
    assert "T >= 8" in v6e_sweep.tailoring_guide[0]
    assert "T < 32" not in v6e_sweep.tailoring_guide[0]
    assert "outside_core_limits" in v6e_sweep.tailoring_guide[0]

    glm_sweep = advisor.advise_regimes("glm-5.2", "tpu7x")
    assert "outside_core_limits" in glm_sweep.tailoring_guide[0]

    dense_profile = models.finalize_model_profile(
        advisor.ModelProfile(
            name="dense-small-gqa",
            num_layers=8,
            hidden_size=4096,
            intermediate_size=14336,
            num_attention_heads=32,
            num_kv_heads=8,
            head_dim=128,
            weight_dtype="fp8",
            kv_dtype="fp8",
        )
    )
    dense_sweep = advisor.advise_regimes(
        dense_profile,
        "tpu7x",
        options=advisor.RegimeSweepOptions(tp=8),
    )
    joined_guide = "\n".join(dense_sweep.tailoring_guide)
    assert "dense_mlp" in joined_guide
    assert "masked_on_chip" not in joined_guide
    assert (
        f"({dense_sweep.max_vmem_tokens} < T <= {dense_sweep.ridge_tokens}"
        not in joined_guide
    )


def test_cli_rejects_empty_and_directory_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject empty, whitespace, and directory values on CLI path and name flags."""
    monkeypatch.chdir(tmp_path)
    qwen_snapshot = json.loads(
        (HF_CONFIG_DIR / "qwen3.5-397b.json").read_text(encoding="utf-8")
    )
    (tmp_path / "config.json").write_text(
        json.dumps(qwen_snapshot["config"]), encoding="utf-8"
    )
    out_dir = tmp_path / "out_dir"
    out_dir.mkdir()
    bad_argvs = (
        ["--config", ""],
        ["--config", "   "],
        ["--model", ""],
        ["--model", "   "],
        ["--hf-repo", ""],
        ["--hf-repo", "   "],
        ["--model", "qwen3.5-397b", "--json", ""],
        ["--model", "qwen3.5-397b", "--json", "   "],
        ["--model", "qwen3.5-397b", "--json", str(out_dir)],
        ["--model", "qwen3.5-397b", "--measurements", ""],
        ["--model", "qwen3.5-397b", "--measurements", "   "],
        ["--model", "qwen3.5-397b", "--measurements", str(out_dir)],
        ["--model", "qwen3.5-397b", "--chip", ""],
        ["--model", "qwen3.5-397b", "--chip", "   "],
        ["--model", "qwen3.5-397b", "--act-dtype", ""],
        ["--model", "qwen3.5-397b", "--kv-dtype", ""],
        ["--model", "qwen3.5-397b", "--state-dtype", ""],
    )
    for argv in bad_argvs:
        before = set(tmp_path.iterdir())
        try:
            rc = advisor.main(argv)
        except MegakernelConfigError:
            rc = constants.INVALID_CONFIG_EXIT_CODE
        except SystemExit as exc:
            rc = exc.code if isinstance(exc.code, int) else 1
        assert rc != 0, f"{argv} returned 0 unexpectedly"
        assert set(tmp_path.iterdir()) == before, f"{argv} modified cwd files"


def test_cli_json_matches_api_over_case_grid(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify CLI JSON output matches `advise()` verbatim across base case grid."""
    for label, slug, profile, phase, lay_name, wl in iter_case_grid():
        if (
            lay_name not in ("preset", "tp_ep_only")
            or label != f"{slug}/{phase}/{lay_name}"
        ):
            continue
        argv = [
            "--model",
            slug,
            "--phase",
            phase,
            "--batch",
            str(wl.batch),
            "--isl",
            str(wl.isl),
            "--osl",
            str(wl.osl),
            "--context",
            str(wl.context),
            "--spec-tokens",
            str(wl.spec_tokens),
            "--chunk-size",
            str(wl.chunk_size),
            "--tp",
            str(wl.tp),
            "--ep",
            str(wl.ep),
            "--dp",
            str(wl.dp),
            "--pp",
            str(wl.pp),
            "--dcp",
            str(wl.dcp),
            "--pcp",
            str(wl.pcp),
            "--json",
        ]
        rc = advisor.main(argv)
        cli_payload = json.loads(capsys.readouterr().out)
        api_rep = advisor.advise(
            profile, dataclasses.replace(wl, batch_explicit=True), "tpu7x"
        )
        api_payload = json.loads(json.dumps(api_rep.to_dict()))
        assert cli_payload == api_payload, f"{label}: CLI JSON != advise()"
        expected_rc = (
            constants.INVALID_CONFIG_EXIT_CODE
            if api_rep.verdict.startswith("invalid_config(")
            else 0
        )
        assert rc == expected_rc, f"{label}: rc={rc} != {expected_rc}"


def test_auto_layout_fits_hbm_when_a_searched_layout_fits() -> None:
    """Verify auto-search finds HBM-fitting DCP layout and honors explicit dcp=1."""
    for slug in sorted(models.MODEL_PRESETS):
        sweep = advisor.advise_regimes(
            slug,
            "tpu7x",
            options=advisor.RegimeSweepOptions(deployment_preset=False),
        )
        decode_high = [
            entry for entry in sweep.regimes if entry.label.startswith("Decode High")
        ]
        assert decode_high
        assert all(entry.hbm_fits is True for entry in decode_high)

    sweep_explicit = advisor.advise_regimes(
        "qwen3.5-397b",
        "tpu7x",
        options=advisor.RegimeSweepOptions(dcp=1),
    )
    assert sweep_explicit.parallelism.dcp == 1


def test_sweep_rows_validate_for_every_preset_role() -> None:
    """Verify sweep prefill OSL=0, zero-spec saturation fallback, and role groups."""
    for slug in sorted(models.MODEL_PRESETS):
        sweep = advisor.advise_regimes(slug, "tpu7x")
        for entry in sweep.regimes:
            if entry.phase == "prefill":
                assert entry.workload["osl"] == constants.DEFAULT_PREFILL_OSL

    profile = advisor.resolve_model_profile("kimi-k3")
    chip_tpu7x = resolve_chip_spec("tpu7x")
    base_preset = advisor.DEPLOYMENT_PRESETS["kimi-k3"]
    assert base_preset.saturation is not None
    half_layers = profile.num_layers // 2
    decode_groups = (half_layers, profile.num_layers - half_layers)
    custom_preset = models.DeploymentPreset(
        decode=dataclasses.replace(
            base_preset.decode, spec_tokens=0, layer_groups=decode_groups
        ),
        prefill=dataclasses.replace(base_preset.prefill, layer_groups=()),
        saturation=dataclasses.replace(base_preset.saturation, layer_groups=()),
        prefill_windows=base_preset.prefill_windows,
    )
    base_wl = advisor.WorkloadSpec(
        tp=custom_preset.decode.tp,
        ep=custom_preset.decode.ep,
        dp=custom_preset.decode.dp,
        pp=custom_preset.decode.pp,
        dcp=custom_preset.decode.dcp,
        pcp=custom_preset.decode.pcp,
        layer_groups=decode_groups,
        batch_explicit=True,
    )
    moved = _with_preset_prefill(
        _build_reference_workloads(base_wl, constants.DEFAULT_SPEC_TOKENS),
        custom_preset,
    )
    for label, role_wl in moved:
        advisor.validate_advise_inputs(profile, role_wl, chip_tpu7x)
        if role_wl.phase == "prefill" or "Saturated" in label:
            assert role_wl.layer_groups == ()


def test_cli_kv_dtype_override_and_core_readiness_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify `--kv-dtype` overrides, fail-closed invalid dtypes, and core output."""
    base_cmd = [
        "--model",
        "qwen3.5-397b",
        "--chip",
        "tpu7x",
        "--tp",
        "8",
        "--ep",
        "8",
        "--batch",
        "8",
    ]
    assert advisor.main([*base_cmd, "--kv-dtype", "fp8_e4m3fn", "--json"]) == 0
    fp8_payload = json.loads(capsys.readouterr().out)
    assert fp8_payload["recommended_scope"] == "whole_step"
    assert fp8_payload["core_support"] == "within_core_limits"
    assert fp8_payload["core_ready"] is True
    assert fp8_payload["core_blockers"] == []
    assert fp8_payload["core_ready_scope"] == "whole_step"

    assert advisor.main([*base_cmd, "--kv-dtype", "bf16", "--json"]) == 0
    bf16_payload = json.loads(capsys.readouterr().out)
    assert bf16_payload["recommended_scope"] == "whole_step"
    assert bf16_payload["core_support"] == "outside_core_limits"
    assert bf16_payload["core_ready"] is False
    assert bf16_payload["core_ready_scope"] == "per_op"
    assert "kv_or_state_dtype_not_in_core" in bf16_payload["core_blockers"]

    assert advisor.main([*base_cmd, "--kv-dtype", "bf16"]) == 0
    text_out = capsys.readouterr().out
    assert "Recommended: whole_step" in text_out
    assert (
        "Core Readiness: outside_core_limits "
        "(core_ready=False, core_ready_scope=per_op, "
        "core_blockers=kv_or_state_dtype_not_in_core)"
    ) in text_out
    assert "Blocked by: kv_or_state_dtype_not_in_core" in text_out

    assert (
        advisor.main(
            [
                "--model",
                "qwen3.5-397b",
                "--regime-sweep",
                "--kv-dtype",
                "bf16",
                "--json",
            ]
        )
        == 0
    )
    sweep_json = json.loads(capsys.readouterr().out)
    assert "kv=bf16" in sweep_json["model_sources"]["dtypes"]
    decode_small = next(
        reg for reg in sweep_json["regimes"] if reg["label"].startswith("Decode Small")
    )
    assert decode_small["recommended_scope"] == "whole_step"
    assert decode_small["core_support"] == "outside_core_limits"
    assert decode_small["core_ready"] is False
    assert decode_small["core_ready_scope"] == "per_op"
    assert "kv_or_state_dtype_not_in_core" in decode_small["core_blockers"]

    for bad_kv in ("invalid_dtype", "   "):
        with pytest.raises(MegakernelConfigError):
            advisor.main([*base_cmd, "--kv-dtype", bad_kv])


def test_format_regime_table_empty_sweep_and_compact_blocker_rows() -> None:
    """Verify empty sweeps format cleanly and long blocker lists compact in rows."""
    sweep = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    empty_table = advisor.format_regime_table(dataclasses.replace(sweep, regimes=()))
    assert "Request Regime" in empty_table
    assert "Core Blockers" in empty_table
    assert "Investment Triage" not in empty_table

    long_blockers = ("blocker_a", "blocker_b", "blocker_c", "blocker_d")
    patched_entry = dataclasses.replace(
        sweep.regimes[0],
        core_support="outside_core_limits",
        core_blockers=long_blockers,
        blockers=long_blockers,
    )
    compact_table = advisor.format_regime_table(
        dataclasses.replace(sweep, regimes=(patched_entry,))
    )
    table_part, triage_part = compact_table.split("Investment Triage", 1)
    assert "blocker_a, blocker_b (+2)" in table_part
    assert "blocker_c" not in table_part
    assert "Core Blockers: blocker_a, blocker_b, blocker_c, blocker_d" in triage_part


def _opt_workload_float(workload_map: Mapping[str, object], key: str) -> float | None:
    """Return optional finite float from `workload_map[key]` or `None`."""
    raw = workload_map.get(key)
    return None if raw is None else specs.scalar_float(raw, key)


def _rebuild_sweep_workload(
    workload_map: Mapping[str, object],
) -> advisor.WorkloadSpec:
    """Rebuild an unpinned `WorkloadSpec` from a serialized `RegimeEntry.workload`."""
    eff_raw = specs.as_mapping(workload_map.get("efficiency"))
    efficiency = (
        advisor.EfficiencySpec(
            eta_hbm=specs.scalar_float(eff_raw["eta_hbm"], "eta_hbm"),
            eta_dense=specs.scalar_float(eff_raw["eta_dense"], "eta_dense"),
            eta_moe_grouped=specs.scalar_float(
                eff_raw["eta_moe_grouped"], "eta_moe_grouped"
            ),
            eta_ici=specs.scalar_float(eff_raw["eta_ici"], "eta_ici"),
        )
        if eff_raw is not None
        else advisor.EfficiencySpec()
    )
    ring_raw = specs.as_mapping(workload_map.get("ring_plan"))
    ring_plan = (
        advisor.RingPlan(
            dense_slots=specs.scalar_int(ring_raw["dense_slots"], "dense_slots"),
            expert_slots=specs.scalar_int(ring_raw["expert_slots"], "expert_slots"),
            dense_slot_bytes=specs.scalar_int(
                ring_raw["dense_slot_bytes"], "dense_slot_bytes"
            ),
            expert_slot_bytes=specs.scalar_int(
                ring_raw["expert_slot_bytes"], "expert_slot_bytes"
            ),
            compiler_spill_reserve_bytes=specs.scalar_int(
                ring_raw["compiler_spill_reserve_bytes"],
                "compiler_spill_reserve_bytes",
            ),
        )
        if ring_raw is not None
        else None
    )
    return advisor.WorkloadSpec(
        phase=cast("advisor.Phase", str(workload_map["phase"])),
        batch=specs.scalar_int(workload_map["batch"], "batch"),
        isl=specs.scalar_int(workload_map["isl"], "isl"),
        osl=specs.scalar_int(workload_map["osl"], "osl"),
        context=specs.scalar_int(workload_map["context"], "context"),
        prefix_hit_rate=specs.scalar_float(
            workload_map["prefix_hit_rate"], "prefix_hit_rate"
        ),
        spec_tokens=specs.scalar_int(workload_map["spec_tokens"], "spec_tokens"),
        chunk_size=specs.scalar_int(workload_map["chunk_size"], "chunk_size"),
        microbatches=specs.scalar_int(workload_map["microbatches"], "microbatches"),
        tp=specs.scalar_int(workload_map["tp"], "tp"),
        pp=specs.scalar_int(workload_map["pp"], "pp"),
        dp=specs.scalar_int(workload_map["dp"], "dp"),
        ep=specs.scalar_int(workload_map["ep"], "ep"),
        kv_split=specs.scalar_int(workload_map["kv_split"], "kv_split"),
        dcp=specs.scalar_int(workload_map["dcp"], "dcp"),
        pcp=specs.scalar_int(workload_map["pcp"], "pcp"),
        segment_layers=specs.scalar_int(
            workload_map["segment_layers"], "segment_layers"
        ),
        layer_groups=specs.int_tuple(workload_map.get("layer_groups"), "layer_groups"),
        moe_mode=cast("advisor.MoEMode", str(workload_map["moe_mode"])),
        vmem_alias=specs.scalar_bool(workload_map["vmem_alias"], "vmem_alias"),
        vmem_fit_headroom=specs.scalar_bool(
            workload_map["vmem_fit_headroom"], "vmem_fit_headroom"
        ),
        op_overhead_us=specs.scalar_float(
            workload_map["op_overhead_us"], "op_overhead_us"
        ),
        vmem_budget_mib=_opt_workload_float(workload_map, "vmem_budget_mib"),
        dma_buffers=specs.scalar_int(workload_map["dma_buffers"], "dma_buffers"),
        ring_plan=ring_plan,
        objective=cast("advisor.ObjectiveMode", str(workload_map["objective"])),
        hbm_utilization_cap=specs.scalar_float(
            workload_map["hbm_utilization_cap"], "hbm_utilization_cap"
        ),
        dispatch_regime=cast(
            "advisor.DispatchRegime", str(workload_map["dispatch_regime"])
        ),
        efficiency=efficiency,
        max_rows_per_call=specs.scalar_int(
            workload_map["max_rows_per_call"], "max_rows_per_call"
        ),
        host_head_per_op_us=specs.scalar_float(
            workload_map["host_head_per_op_us"], "host_head_per_op_us"
        ),
        host_head_fused_us=specs.scalar_float(
            workload_map["host_head_fused_us"], "host_head_fused_us"
        ),
        host_head_explicit=specs.scalar_bool(
            workload_map["host_head_explicit"], "host_head_explicit"
        ),
        per_op_step_us=_opt_workload_float(workload_map, "per_op_step_us"),
        step_coverage=_opt_workload_float(workload_map, "step_coverage"),
        batch_explicit=specs.scalar_bool(
            workload_map["batch_explicit"], "batch_explicit"
        ),
        latent_proj_tp=specs.scalar_int(
            workload_map.get("latent_proj_tp", 0), "latent_proj_tp"
        ),
    )


def _check_regime_entry_against_per_op(
    entry: advisor.RegimeEntry,
    tag: str,
    violations: list[str],
) -> None:
    """Append floor and estimate violations comparing RegimeEntry against per_op."""
    if entry.recommended_scope == "per_op":
        return
    rec_scope = next(
        sc for sc in entry.scopes if sc.get("scope") == entry.recommended_scope
    )
    per_op_scope = next(sc for sc in entry.scopes if sc.get("scope") == "per_op")
    rec_est = specs.scalar_float(rec_scope["step_achievable_us"], "step_achievable_us")
    per_op_est = specs.scalar_float(
        per_op_scope["step_achievable_us"], "step_achievable_us"
    )
    if rec_est > per_op_est:
        violations.append(
            f"{tag} | entry_fused_est_le_per_op | "
            f"rec={entry.recommended_scope} est={rec_est}>{per_op_est}"
        )
    rec_floor = specs.scalar_float(rec_scope["step_bound_us"], "step_bound_us")
    per_op_floor = specs.scalar_float(per_op_scope["step_bound_us"], "step_bound_us")
    if rec_floor > per_op_floor:
        violations.append(
            f"{tag} | entry_fused_floor_le_per_op | "
            f"rec={entry.recommended_scope} floor={rec_floor}>{per_op_floor}"
        )


@pytest.mark.parametrize("preset_slug", _PRESET_SLUGS)
@pytest.mark.parametrize("moe_setting", _MOE_SETTINGS)
def test_sweep_regime_grid_cross_field_invariants(
    preset_slug: str,
    moe_setting: str,
) -> None:
    """Verify 12 per-report and sweep-entry invariants across preset regimes."""
    profile = advisor.resolve_model_profile(preset_slug)
    sweep = advisor.advise_regimes(profile, "tpu7x")
    assert sweep.regimes
    violations: list[str] = []
    for entry in sweep.regimes:
        tag = f"{preset_slug} | {entry.label}"
        if moe_setting == "unpinned":
            _check_regime_entry_against_per_op(entry, tag, violations)
        rebuilt = _rebuild_sweep_workload(entry.workload)
        workload = _apply_moe_setting(rebuilt, moe_setting)
        report = advisor.advise(profile, workload, "tpu7x")
        _check_scope_envelope_invariants(report, tag, violations)
        _check_verdict_and_core_invariants(report, tag, violations)
    assert not violations, (
        f"{preset_slug} ({moe_setting}) {len(violations)} violations: "
        f"{'; '.join(violations[:_MAX_SHOWN_VIOLATIONS])}"
    )


def test_cli_moe_mode_pins_when_passed_and_searches_when_omitted(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify `--moe-mode` pins selection while omission searches admissible modes."""
    base_argv = [
        "--model",
        "qwen3.5-397b",
        "--batch",
        "600",
        "--context",
        "32768",
        "--tp",
        "2",
        "--dp",
        "4",
        "--ep",
        "8",
    ]
    assert advisor.main([*base_argv, "--moe-mode", "masked_on_chip"]) == 0
    pinned_out = capsys.readouterr().out
    assert "Recommended: single_layer" in pinned_out
    assert "whole_step skipped despite lower floor" in pinned_out

    assert advisor.main(base_argv) == 0
    unpinned_out = capsys.readouterr().out
    assert "Recommended: whole_step [moe=grouped_on_chip]" in unpinned_out
    assert "build=True" in unpinned_out
