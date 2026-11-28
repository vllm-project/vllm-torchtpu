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
"""CPU unit tests for physics-only ceiling curve export and solve JSON payloads."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import pytest

import vllm_torchtpu.roofline as rf_pkg
import vllm_torchtpu.roofline.constants as rf_const
from tests.roofline.conftest import LOCKED_FIXTURE_FILES, load_locked_bundle
from vllm_torchtpu.roofline import (
    CHIPS,
    CeilingRow,
    SolveOptions,
    TrafficScenario,
    build_solve_payload,
    cli,
    emit_ceiling_rows,
    format_solve_json,
    format_submodule_parallelism,
    solve_serving_plan,
)
from vllm_torchtpu.roofline.constants import NO_FEASIBLE_LAYOUT_WARNING

if TYPE_CHECKING:
    from pathlib import Path

_CEILING_KEYS = [
    "est_interactivity_tok_s",
    "token_throughput_per_chip",
    "output_throughput_per_chip",
    "est_ttft_ms",
    "est_tpot_ms",
    "concurrency",
    "total_chips",
    "mode",
    "binding_unit",
    "slo_met",
    "is_pareto",
    "decode_topology",
    "decode_parallelism",
    "prefill_topology",
    "prefill_parallelism",
]


def test_solve_output_has_no_launch_commands(
    locked_config_path: Callable[[str], Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify solve, ceiling, and package exports emit zero launch commands or flags."""
    assert not hasattr(rf_pkg, "emit_launch_args")
    assert not hasattr(rf_pkg, "pool_cli_args")
    assert not hasattr(rf_pkg, "LaunchPayload")
    assert not hasattr(rf_const, "DEFAULT_PCP_KV_INTERLEAVE_SIZE")
    assert CeilingRow is rf_pkg.CeilingRow

    with pytest.raises(SystemExit) as emit_exc:
        cli.main(["emit", "--help"])
    assert emit_exc.value.code != 0
    capsys.readouterr()
    with pytest.raises(SystemExit) as help_exc:
        cli.main(["--help"])
    assert help_exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "ceiling" in help_text
    assert "solve" in help_text

    cfg_path = locked_config_path("qwen3_5_397b_fp8.json")
    rc_solve = cli.main(
        [
            "solve",
            "--config",
            str(cfg_path),
            "--example-scenario",
            "chat-8k1k",
            "--mode",
            "mixed",
            "--max-chips",
            "4",
            "--concurrency",
            "8",
            "--json",
        ]
    )
    assert rc_solve == 0
    solve_raw = capsys.readouterr().out
    assert "vllm serve" not in solve_raw
    solve_payload = json.loads(solve_raw)
    assert "commands" not in solve_payload
    assert "cli_args" not in solve_payload
    assert "prefill_cli_args" not in solve_payload


@pytest.mark.parametrize("fixture_name", LOCKED_FIXTURE_FILES)
@pytest.mark.parametrize("mode", ("mixed", "disagg", "auto"))
def test_emit_ceiling_rows_covers_every_candidate_and_cli_formats(
    fixture_name: str,
    mode: Literal["mixed", "disagg", "auto"],
    locked_config_path: Callable[[str], Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify ceiling rows cover every candidate across fixtures, modes, and formats."""
    bundle = load_locked_bundle(fixture_name)
    scenario = TrafficScenario(
        name="chat-8k1k",
        isl=2048,
        osl=256,
        prefix_hit_rate=0.5,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(1, 8),
        kv_working_set_turns=1.0,
    )
    serving_plan = solve_serving_plan(
        bundle.geometry,
        bundle.dtypes,
        CHIPS["tpu7x"],
        scenario,
        SolveOptions(
            mode=mode,
            max_chips=8,
            include_pairs=mode in ("disagg", "auto"),
        ),
    )
    rows: list[CeilingRow] = emit_ceiling_rows(serving_plan)
    assert len(rows) == len(serving_plan.candidates)
    assert rows
    assert list(rows[0].keys()) == _CEILING_KEYS
    assert any(r["is_pareto"] for r in rows)

    cfg_path = locked_config_path(fixture_name)
    base_argv = [
        "ceiling",
        "--config",
        str(cfg_path),
        "--example-scenario",
        "chat-8k1k",
        "--mode",
        mode,
        "--max-chips",
        "8",
        "--concurrency",
        "8",
    ]
    rc_json = cli.main([*base_argv, "--json"])
    assert rc_json == 0
    json_rows = json.loads(capsys.readouterr().out)
    assert json_rows
    assert list(json_rows[0].keys()) == _CEILING_KEYS

    rc_jsonl = cli.main(base_argv)
    assert rc_jsonl == 0
    jsonl_lines = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    ]
    assert jsonl_lines == json_rows


def test_ceiling_cli_infeasible_slo_still_exports_all_candidates_and_warns(
    locked_config_path: Callable[[str], Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify infeasible SLO keeps all legal rows with slo_met=False and warns."""
    bundle = load_locked_bundle("qwen3_5_397b_fp8.json")
    cfg_path = locked_config_path("qwen3_5_397b_fp8.json")
    infeasible_plan = solve_serving_plan(
        bundle.geometry,
        bundle.dtypes,
        CHIPS["tpu7x"],
        TrafficScenario(
            name="infeasible-slo",
            isl=4096,
            osl=256,
            prefix_hit_rate=0.0,
            accept_len=1.0,
            think_time_s=0.0,
            concurrencies=(4,),
            kv_working_set_turns=1.0,
            tpot_slo_ms=0.001,
        ),
        SolveOptions(mode="mixed", max_chips=4),
    )
    assert infeasible_plan.recommended is None
    assert infeasible_plan.candidates
    assert all(not p.slo_met for p in infeasible_plan.candidates)
    assert any("INFEASIBLE_SLO" in w for w in infeasible_plan.warnings)
    assert any("SLO_UNMET" in w for w in infeasible_plan.warnings)

    rc_ceil = cli.main(
        [
            "ceiling",
            "--config",
            str(cfg_path),
            "--example-scenario",
            "chat-8k1k",
            "--mode",
            "mixed",
            "--max-chips",
            "4",
            "--concurrency",
            "16",
            "--tpot-slo-ms",
            "0.001",
            "--json",
        ]
    )
    captured = capsys.readouterr()
    assert rc_ceil == 0
    rows = json.loads(captured.out)
    assert len(rows) >= 1
    assert all(not r["slo_met"] for r in rows)
    assert "WARNING:" in captured.err


def test_ceiling_cli_warns_and_exits_nonzero_when_no_layout_fits(
    locked_config_path: Callable[[str], Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Write no-feasible-layout warning and exit 1 when HBM reserve blocks all fits."""
    cfg_path = locked_config_path("qwen3_5_397b_fp8.json")
    exit_code = cli.main(
        [
            "ceiling",
            "--config",
            str(cfg_path),
            "--example-scenario",
            "chat-8k1k",
            "--mode",
            "mixed",
            "--max-chips",
            "4",
            "--concurrency",
            "8",
            "--hbm-reserve-gb",
            "102.9",
            "--json",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out.strip() == "[]"
    assert f"WARNING: {NO_FEASIBLE_LAYOUT_WARNING}" in captured.err

    rc_disagg_4 = cli.main(
        [
            "ceiling",
            "--config",
            str(cfg_path),
            "--example-scenario",
            "chat-8k1k",
            "--mode",
            "disagg",
            "--max-chips",
            "4",
            "--concurrency",
            "8",
            "--json",
        ]
    )
    captured_disagg_4 = capsys.readouterr()
    assert rc_disagg_4 == 1
    assert captured_disagg_4.out.strip() == "[]"
    assert "WARNING:" in captured_disagg_4.err

    rc_disagg_inst = cli.main(
        [
            "ceiling",
            "--config",
            str(cfg_path),
            "--example-scenario",
            "chat-8k1k",
            "--mode",
            "disagg",
            "--max-instances",
            "1",
            "--concurrency",
            "8",
            "--json",
        ]
    )
    captured_inst = capsys.readouterr()
    assert rc_disagg_inst == 1
    assert captured_inst.out.strip() == "[]"
    assert "DISAGG_PAIRS_NOT_EVALUATED" in captured_inst.err


def test_disagg_pool_only_warnings_and_empty_plan_ceiling() -> None:
    """Verify pool-only disagg emits DISAGG_PAIRS_NOT_EVALUATED and SLO_UNMET."""
    bundle = load_locked_bundle("qwen3_5_397b_fp8.json")
    pool_only_plan = solve_serving_plan(
        bundle.geometry,
        bundle.dtypes,
        CHIPS["tpu7x"],
        TrafficScenario(
            name="disagg-pool-only-slo",
            isl=4096,
            osl=256,
            prefix_hit_rate=0.0,
            accept_len=1.0,
            think_time_s=0.0,
            concurrencies=(8,),
            kv_working_set_turns=1.0,
            tpot_slo_ms=0.001,
        ),
        SolveOptions(mode="disagg", max_chips=8, include_pairs=False),
    )
    assert emit_ceiling_rows(pool_only_plan) == []
    assert pool_only_plan.prefill_pools
    assert pool_only_plan.decode_pools
    assert any("DISAGG_PAIRS_NOT_EVALUATED" in w for w in pool_only_plan.warnings)
    assert any("SLO_UNMET" in w for w in pool_only_plan.warnings)

    infeasible_plan = solve_serving_plan(
        bundle.geometry,
        bundle.dtypes,
        CHIPS["tpu7x"],
        TrafficScenario(
            name="no-feasible-layout",
            isl=4096,
            osl=256,
            prefix_hit_rate=0.0,
            accept_len=1.0,
            think_time_s=0.0,
            concurrencies=(8,),
            kv_working_set_turns=1.0,
        ),
        SolveOptions(mode="mixed", max_chips=4, hbm_reserve_gb=191.9),
    )
    assert infeasible_plan.candidates == ()
    assert NO_FEASIBLE_LAYOUT_WARNING in infeasible_plan.warnings


def test_solve_default_returns_all_layouts_ordered_per_concurrency(
    locked_config_path: Callable[[str], Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify solve --json returns all legal layouts ordered per concurrency."""
    cfg_path = locked_config_path("qwen3_5_397b_fp8.json")
    rc = cli.main(
        [
            "solve",
            "--config",
            str(cfg_path),
            "--example-scenario",
            "chat-8k1k",
            "--mode",
            "auto",
            "--max-chips",
            "4",
            "--concurrency",
            "1,16",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == rf_const.SOLVE_SCHEMA
    assert payload["top_k_limit"] is None
    assert payload["sort_by"] == "throughput"
    assert payload["prefill_pools"]
    assert payload["decode_pools"]
    assert payload["provenance"]["calibrated"] is False
    assert "p90" not in payload["provenance"]["basis"].lower()
    assert payload["model"] == str(cfg_path)
    for concurrency in (1, 16):
        cands = [p for p in payload["candidates"] if p["concurrency"] == concurrency]
        assert len(cands) > rf_const.DEFAULT_TOP_K
        assert any(p["is_pareto"] for p in cands)
        tputs = [p["token_throughput_per_chip"] for p in cands]
        assert tputs == sorted(tputs, reverse=True)
    for entry in payload["top_k"]:
        assert "launch" not in entry


@pytest.mark.parametrize("fixture_name", LOCKED_FIXTURE_FILES)
@pytest.mark.parametrize("sort_by", ("throughput", "interactivity"))
def test_sort_by_roles_margins_floor_bounds_and_residual(
    fixture_name: str,
    sort_by: Literal["throughput", "interactivity"],
) -> None:
    """Verify sort_by roles/margins, FloorTimes 4-bound max, and residual payload."""
    bundle = load_locked_bundle(fixture_name)
    chip = CHIPS["tpu7x"]
    scenario = TrafficScenario(
        name="sort-roles-check",
        isl=4096,
        osl=256,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(1, 16),
        kv_working_set_turns=1.0,
    )
    plan = solve_serving_plan(
        bundle.geometry,
        bundle.dtypes,
        chip,
        scenario,
        SolveOptions(mode="mixed", max_chips=4, sort_by=sort_by),
    )
    payload = build_solve_payload(plan, chip=chip, dtypes=bundle.dtypes, top_k=3)
    assert format_solve_json(payload)
    assert payload["sort_by"] == sort_by
    assert "p90" not in payload["provenance"]["basis"].lower()
    top_entries = payload["top_k"]
    assert top_entries
    expected_rank1_role = (
        "max_interactivity" if sort_by == "interactivity" else "max_throughput"
    )
    assert top_entries[0]["frontier_role"] == expected_rank1_role
    assert top_entries[0]["margin_vs_rank1_pct"] == 0.0
    for entry in top_entries[1:]:
        assert entry["margin_vs_rank1_pct"] <= 0.0
    if len(top_entries) >= 2:
        expected_opp_role = (
            "max_throughput" if sort_by == "interactivity" else "max_interactivity"
        )
        opp_entries = [
            entry
            for entry in top_entries
            if entry["frontier_role"] == expected_opp_role
        ]
        assert len(opp_entries) == 1
        slo_cands = [pt for pt in plan.candidates if pt.slo_met]
        if sort_by == "interactivity":
            global_opp = max(slo_cands, key=lambda pt: pt.token_throughput_per_chip)
            assert opp_entries[0]["point"][
                "token_throughput_per_chip"
            ] == pytest.approx(global_opp.token_throughput_per_chip, rel=1e-9)
        else:
            global_opp = max(slo_cands, key=lambda pt: pt.est_interactivity_tok_s)
            assert opp_entries[0]["point"]["est_interactivity_tok_s"] == pytest.approx(
                global_opp.est_interactivity_tok_s, rel=1e-9
            )
    for point in plan.candidates:
        assert point.decode_floor is not None
        assert point.decode_pool.residual in ("replicated", "sp")
        max_bound = max(
            point.decode_floor.hbm_bound_ms,
            point.decode_floor.matmul_bound_ms,
            point.decode_floor.vector_bound_ms,
            point.decode_floor.coll_bound_ms,
        )
        assert max_bound == pytest.approx(point.decode_floor.hard_floor_ms, rel=1e-9)


@pytest.mark.parametrize("top_k", (None, 1, 3))
def test_top_k_preserves_frontier_and_single_host_fit_while_slicing_payload(
    top_k: int | None,
) -> None:
    """Verify top_k slices payload lists per concurrency without truncating plan."""
    bundle = load_locked_bundle("qwen3_5_397b_fp8.json")
    chip = CHIPS["tpu7x"]
    scenario = TrafficScenario(
        name="top-k-presentation-check",
        isl=4096,
        osl=256,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(1, 16),
        kv_working_set_turns=1.0,
    )
    full_plan = solve_serving_plan(
        bundle.geometry,
        bundle.dtypes,
        chip,
        scenario,
        SolveOptions(mode="auto", max_chips=8, top_k=None),
    )
    k_plan = solve_serving_plan(
        bundle.geometry,
        bundle.dtypes,
        chip,
        scenario,
        SolveOptions(mode="auto", max_chips=8, top_k=top_k),
    )
    assert len(k_plan.candidates) == len(full_plan.candidates)
    assert len(k_plan.frontier) == len(full_plan.frontier)
    full_payload = build_solve_payload(full_plan, chip=chip, top_k=None)
    k_payload = build_solve_payload(k_plan, chip=chip, top_k=top_k)
    assert (
        k_payload["single_host_fit"]["status"]
        == full_payload["single_host_fit"]["status"]
    )
    assert len(k_payload["frontier"]) == len(full_payload["frontier"])
    if top_k is not None:
        assert len(k_payload["top_k"]) <= top_k
        for concurrency in (1, 16):
            cands = [
                p for p in k_payload["candidates"] if p["concurrency"] == concurrency
            ]
            pref = [
                r for r in k_payload["prefill_pools"] if r["concurrency"] == concurrency
            ]
            dec = [
                r for r in k_payload["decode_pools"] if r["concurrency"] == concurrency
            ]
            assert len(cands) == top_k
            assert len(pref) == top_k
            assert len(dec) == top_k


def test_solve_payload_preserves_pins_optional_think_time_and_validation(
    locked_config_path: Callable[[str], Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify pinned-budget CLI, optional think-time, pin validation, and submodule."""
    cfg_path = locked_config_path("qwen3_5_397b_fp8.json")
    rc = cli.main(
        [
            "solve",
            "--config",
            str(cfg_path),
            "--isl",
            "1024",
            "--osl",
            "128",
            "--prefix-hit-rate",
            "0.0",
            "--accept-len",
            "1.0",
            "--concurrency",
            "4",
            "--mode",
            "disagg",
            "--prefill-topology",
            "2x2x1",
            "--decode-topology",
            "2x2x1",
            "--pd",
            "1:1",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["budget"] == 8
    assert payload["budget_source"] == "pins"
    assert payload["pins"] == {
        "prefill_topology": "2x2x1",
        "decode_topology": "2x2x1",
        "pd_ratio": "1:1",
    }
    assert payload["scenario"]["think_time_s"] is None
    assert payload["candidates"]
    assert all(p["total_chips"] == 8 for p in payload["candidates"])

    # Asymmetric disagg pins must prune competing legal pairs under 16 chips.
    rc_asym = cli.main(
        [
            "solve",
            "--config",
            str(cfg_path),
            "--isl",
            "1024",
            "--osl",
            "128",
            "--prefix-hit-rate",
            "0.0",
            "--accept-len",
            "1.0",
            "--concurrency",
            "4",
            "--mode",
            "disagg",
            "--prefill-topology",
            "2x2x1",
            "--decode-topology",
            "2x2x2",
            "--pd",
            "2:1",
            "--json",
        ]
    )
    assert rc_asym == 0
    asym_payload = json.loads(capsys.readouterr().out)
    assert asym_payload["budget"] == 16
    assert asym_payload["budget_source"] == "pins"
    assert asym_payload["candidates"]
    for point in asym_payload["candidates"]:
        assert point["prefill_pool"]["instances"] == 2
        assert point["prefill_pool"]["topology"] == "2x2x1"
        assert point["decode_pool"]["instances"] == 1
        assert point["decode_pool"]["topology"] == "2x2x2"
        assert point["total_chips"] == 16

    # Direct library SolveOptions(max_instances=2) derives budget_source automatically.
    bundle = load_locked_bundle("qwen3_5_397b_fp8.json")
    lib_scen = TrafficScenario(
        name="lib-max-inst",
        isl=1024,
        osl=128,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=None,
        concurrencies=(4,),
        kv_working_set_turns=1.0,
    )
    lib_plan = solve_serving_plan(
        bundle.geometry,
        bundle.dtypes,
        CHIPS["tpu7x"],
        lib_scen,
        SolveOptions(mode="mixed", max_instances=2),
    )
    assert lib_plan.budget_source == "max_instances"
    lib_payload = build_solve_payload(lib_plan, chip=CHIPS["tpu7x"])
    assert lib_payload["budget_source"] == "max_instances"
    assert lib_payload["chip_budget"]["max_chips"] is None

    # Invalid topology pin and --mode mixed --pd must fail closed with ValueError.
    base_scen = TrafficScenario(
        name="pin-err",
        isl=1024,
        osl=128,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=None,
        concurrencies=(4,),
        kv_working_set_turns=1.0,
        tpot_slo_ms=25.0,
    )
    with pytest.raises(ValueError, match="Unknown decode_topology"):
        solve_serving_plan(
            bundle.geometry,
            bundle.dtypes,
            CHIPS["tpu7x"],
            base_scen,
            SolveOptions(mode="mixed", max_chips=4, decode_topology="9x9x9"),
        )
    with pytest.raises(ValueError, match="mode='mixed'"):
        solve_serving_plan(
            bundle.geometry,
            bundle.dtypes,
            CHIPS["tpu7x"],
            base_scen,
            SolveOptions(mode="mixed", max_chips=8, pd_ratio=(1, 1)),
        )
    single_slo_plan = solve_serving_plan(
        bundle.geometry,
        bundle.dtypes,
        CHIPS["tpu7x"],
        base_scen,
        SolveOptions(mode="mixed", max_chips=4),
    )
    assert rf_const.UNKNOWN_THINK_TIME_WARNING in single_slo_plan.warnings

    # Subgroup MoE display renders EP x TP when 1 < ep < tp.
    assert (
        format_submodule_parallelism({"tp": 8, "ep": 4, "dp": 1, "pp": 1, "pcp": 1})
        == "Attn: TP8 · MoE: EP4 x TP2"
    )

    # max_instances without max_chips keeps chip_budget["max_chips"] as None.
    rc_inst = cli.main(
        [
            "solve",
            "--config",
            str(cfg_path),
            "--example-scenario",
            "chat-8k1k",
            "--mode",
            "mixed",
            "--max-instances",
            "1",
            "--concurrency",
            "4",
            "--json",
        ]
    )
    assert rc_inst == 0
    inst_payload = json.loads(capsys.readouterr().out)
    assert inst_payload["chip_budget"]["max_chips"] is None


def test_pool_label_joins_axes_with_x_and_omits_ones() -> None:
    """Verify format_parallelism and pool_label omit size-1 axes and join with x."""
    sample_pool = {
        "instances": 1,
        "topology": "2x2x1",
        "chips_per_instance": 4,
        "tp": 8,
        "dp": 1,
        "ep": 8,
        "pp": 1,
        "pcp": 1,
        "kv_split": 4,
    }
    assert rf_pkg.format_parallelism(sample_pool) == "TP8 x EP8 x DCP4"
    label = rf_pkg.pool_label(sample_pool)
    assert label.endswith(": TP8 x EP8 x DCP4")

    all_one_pool = {
        "instances": 1,
        "topology": "1x1x1",
        "chips_per_instance": 1,
        "tp": 1,
        "dp": 1,
        "ep": 1,
        "pp": 1,
        "pcp": 1,
        "kv_split": 1,
    }
    assert rf_pkg.format_parallelism(all_one_pool) == "TP1"


def test_format_submodule_parallelism_splits_attn_and_moe() -> None:
    """format_submodule_parallelism separates Attention, PP, and MoE sharding."""
    pcp_ep_pool = {"tp": 1, "dp": 1, "ep": 8, "pp": 1, "pcp": 8, "kv_split": 1}
    assert (
        rf_pkg.format_submodule_parallelism(pcp_ep_pool, has_moe=True)
        == "Attn: PCP8 · MoE: EP8"
    )

    tp_sliced_pool = {"tp": 8, "dp": 1, "ep": 1, "pp": 1, "pcp": 1, "kv_split": 4}
    assert (
        rf_pkg.format_submodule_parallelism(tp_sliced_pool, has_moe=True)
        == "Attn: TP8 x DCP4 · MoE: TP8"
    )

    pp_pcp_ep_pool = {"tp": 1, "dp": 1, "ep": 2, "pp": 4, "pcp": 2, "kv_split": 1}
    assert (
        rf_pkg.format_submodule_parallelism(pp_pcp_ep_pool, has_moe=True)
        == "PP4 · Attn: PCP2 · MoE: EP2"
    )

    dense_tp_pool = {"tp": 8, "dp": 1, "ep": 1, "pp": 1, "pcp": 1, "kv_split": 1}
    assert rf_pkg.format_submodule_parallelism(dense_tp_pool, has_moe=False) == "TP8"
