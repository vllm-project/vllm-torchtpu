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


def test_ordered_candidates_distinguish_prefill_mesh_families() -> None:
    """Real disagg solve keeps distinct prefill mesh families in ordered output."""
    bundle = load_locked_bundle("qwen3_5_397b_fp8.json")
    geometry, dtypes = bundle.geometry, bundle.dtypes
    target_concurrency = 16
    scenario = rf_pkg.TrafficScenario(
        name="disagg-prefill-diversity",
        isl=4096,
        osl=512,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=None,
        concurrencies=(target_concurrency,),
        kv_working_set_turns=1.0,
    )
    serving_plan = rf_pkg.solve_serving_plan(
        geometry,
        dtypes,
        rf_pkg.CHIPS["tpu7x"],
        scenario,
        rf_pkg.SolveOptions(
            mode="disagg",
            max_chips=8,
            prefill_topology="2x2x1",
            decode_topology="2x2x1",
            pd_ratio=(1, 1),
            include_pairs=True,
        ),
    )
    ordered = rf_pkg.ordered_candidates_by_concurrency(serving_plan)
    assert target_concurrency in ordered
    prefill_meshes = {
        (
            point.prefill_pool.tp,
            point.prefill_pool.dp,
            point.prefill_pool.ep,
            point.prefill_pool.pp,
            point.prefill_pool.pcp,
        )
        for point in ordered[target_concurrency]
        if point.prefill_pool is not None
    }
    assert len(prefill_meshes) > 1
    families = rf_pkg.build_bound_families_by_concurrency(serving_plan)
    assert len(families[str(target_concurrency)]) >= len(prefill_meshes)


def test_reference_layout_rank_compares_same_concurrency_peers() -> None:
    """Reference layout ranking compares strictly against same-concurrency peers."""
    bundle = load_locked_bundle("qwen3_5_397b_fp8.json")
    geometry, dtypes = bundle.geometry, bundle.dtypes
    scenario = rf_pkg.TrafficScenario(
        name="ref-rank-same-conc",
        isl=2048,
        osl=256,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=None,
        concurrencies=(1, 64),
        kv_working_set_turns=1.0,
    )
    serving_plan = rf_pkg.solve_serving_plan(
        geometry,
        dtypes,
        rf_pkg.CHIPS["tpu7x"],
        scenario,
        rf_pkg.SolveOptions(mode="mixed", max_chips=4),
    )
    c1_points = [point for point in serving_plan.candidates if point.concurrency == 1]
    assert len(c1_points) > 1
    assert len(serving_plan.candidates) > len(c1_points)
    report = rf_pkg.reference_layout_report(
        serving_plan,
        rf_pkg.top_layouts(serving_plan),
        "TP8 x EP8",
        "",
    )
    assert report["found"] is True
    assert report["ruled_out"] is False
    assert report["concurrency"] in (1, 64)
    same_c_points = [
        point
        for point in serving_plan.candidates
        if point.concurrency == report["concurrency"]
    ]
    axis_groups: dict[
        tuple[int, int, int, int, int, int], list[rf_pkg.ParetoPoint]
    ] = {}
    for point in same_c_points:
        pool = point.decode_pool
        key = (pool.tp, pool.dp, pool.ep, pool.pp, pool.pcp, pool.kv_split)
        axis_groups.setdefault(key, []).append(point)
    ref_key = (8, 1, 8, 1, 1, 1)
    assert ref_key in axis_groups
    expected_ref_bound = min(
        point.decode_floor.hard_floor_ms for point in axis_groups[ref_key]
    )
    expected_ref_tput = max(
        point.token_throughput_per_chip for point in axis_groups[ref_key]
    )
    expected_bound_rank = 1 + sum(
        1
        for key, pts in axis_groups.items()
        if key != ref_key
        and min(p.decode_floor.hard_floor_ms for p in pts) < expected_ref_bound - 1e-9
    )
    expected_tput_rank = 1 + sum(
        1
        for key, pts in axis_groups.items()
        if key != ref_key
        and max(p.token_throughput_per_chip for p in pts) > expected_ref_tput + 1e-9
    )
    assert report["candidates"] == len(axis_groups)
    assert report["rank_by_tpot_bound"] == expected_bound_rank
    assert report["rank_by_estimated_throughput"] == expected_tput_rank
    assert report["point"] is not None
    assert report["point"]["concurrency"] == report["concurrency"]


@pytest.mark.parametrize(
    ("layout_text", "mode", "opts_kwargs", "override_gdn", "expected_sub"),
    [
        (
            "P: PP4 / D: TP8 x EP8",
            "mixed",
            {"max_chips": 4},
            False,
            "disaggregated mode",
        ),
        ("TP8 x EP8", "disagg", {"max_chips": 8}, False, "single-pool layout"),
        (
            "TP3",
            "mixed",
            {"max_chips": 4},
            False,
            "does not divide the attention heads",
        ),
        (
            "TP16 x EP16",
            "mixed",
            {"max_chips": 8},
            True,
            "not a valid linear-attention shard degree",
        ),
        ("PP999", "mixed", {"max_chips": 4}, False, "exceeds model layers"),
        (
            "TP2 x DCP4",
            "mixed",
            {"max_chips": 4},
            False,
            "is not physically valid for TP2",
        ),
        ("TP8 x EP3", "mixed", {"max_chips": 4}, False, "is not physically divisible"),
        (
            "TP4 x PCP2 x EP8",
            "mixed",
            {"max_chips": 4},
            False,
            "mixed layout requires PCP1",
        ),
        (
            "P: TP8 x EP8 / D: TP4 x PCP2 x EP8",
            "disagg",
            {"max_chips": 8, "include_pairs": True},
            False,
            "decode pool requires PCP1",
        ),
        (
            "TP16 x EP16",
            "mixed",
            {"max_chips": 8, "decode_topology": "2x2x1"},
            False,
            "pinned decode topology 2x2x1",
        ),
        (
            "TP2 x DP3",
            "mixed",
            {"max_chips": 4},
            False,
            "no searched slice has that many",
        ),
        (
            "TP16 x EP16",
            "mixed",
            {"max_chips": 4},
            False,
            "exceeding max_chips=4",
        ),
        (
            "P: TP8 x EP8 / D: TP8 x EP8",
            "disagg",
            {"max_chips": 8, "pd_ratio": (2, 2), "include_pairs": True},
            False,
            "exceeding max_chips=8",
        ),
        (
            "P: TP8 x EP8 / D: TP8 x EP8",
            "disagg",
            {"max_chips": 4, "include_pairs": True},
            False,
            "1P+1D",
        ),
        (
            "P: TP8 x EP8 / D: TP8 x EP8",
            "disagg",
            {
                "max_chips": 32,
                "pd_ratio": (2, 2),
                "max_instances": 3,
                "include_pairs": True,
            },
            False,
            "exceeding max_instances=3",
        ),
        (
            "P: TP8 x EP8 / D: TP8 x EP8",
            "disagg",
            {"max_chips": 32, "max_instances": 1, "include_pairs": True},
            False,
            "exceeding max_instances=1",
        ),
        (
            "P: TP8 x EP8 / D: TP8 x EP8",
            "disagg",
            {"max_chips": 8, "include_pairs": False},
            False,
            "pair candidates are off",
        ),
        (
            "TP8 x EP8",
            "mixed",
            {"max_chips": 4},
            False,
            "enumerated, but no searched point",
        ),
    ],
)
def test_parse_layout_and_explain_rejection_diagnostics(
    layout_text: str,
    mode: Literal["auto", "mixed", "disagg"],
    opts_kwargs: dict[str, object],
    override_gdn: bool,
    expected_sub: str,
) -> None:
    """Verify explain_layout_rejection names the exact failing physical rule."""
    from dataclasses import replace

    geometry = load_locked_bundle("qwen3_5_397b_fp8.json").geometry
    if override_gdn:
        geometry = replace(geometry, linear_v_heads=8, linear_k_heads=4)
    topos = tuple(rf_pkg.TOPOLOGIES.values())
    opts = rf_pkg.SolveOptions(mode=mode, **opts_kwargs)  # type: ignore[arg-type]
    reason = rf_pkg.explain_layout_rejection(
        geometry, rf_pkg.parse_layout(layout_text), topos, opts
    )
    assert expected_sub in reason


def test_explain_layout_rejection_not_enumerated_and_parse_errors(
    monkeypatch: pytest.MonkeyPatch,
    locked_config_path: Callable[[str], Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Cover not-enumerated path, parse_layout errors/D:, strict_rank ties, and CLI."""
    from vllm_torchtpu.roofline import solver as rf_solver

    # Valid multi-pool and single-pool D: syntax.
    parsed_pd = rf_pkg.parse_layout("P: PP4 x EP1 / D: TP8 x EP8 x DCP2")
    assert parsed_pd["prefill"]["pp"] == 4
    assert parsed_pd["decode"]["tp"] == 8
    assert parsed_pd["decode"]["kv_split"] == 2
    parsed_d = rf_pkg.parse_layout("D: TP8 x EP8")
    assert "prefill" not in parsed_d
    assert parsed_d["decode"]["tp"] == 8
    assert parsed_d["decode"]["ep"] == 8

    # All parse_layout error branches.
    invalid_specs = [
        "",
        "   ",
        "TP8 x QP2",
        "TP4 x TP2",
        "TP0",
        "TPbad",
        "P: TP8 / D: TP8 / D: TP4",
        "X: TP8 / D: TP8",
        "P: TP8 / X: TP8",
        "P: TP8",
    ]
    for bad in invalid_specs:
        with pytest.raises(ValueError):
            rf_pkg.parse_layout(bad)

    # strict_rank exact ordering and ties (both directions).
    assert rf_pkg.strict_rank(10.0, [5.0, 10.0, 10.0, 20.0], lower_is_better=True) == 2
    assert rf_pkg.strict_rank(10.0, [5.0, 10.0, 15.0, 20.0], lower_is_better=False) == 3
    assert rf_pkg.strict_rank(5.0, [5.0, 5.0], lower_is_better=True) == 1

    # Not-enumerated slice branch.
    geometry = load_locked_bundle("qwen3_5_397b_fp8.json").geometry
    monkeypatch.setattr(rf_solver, "_candidate_decode_parallels", lambda _g, _t: [])
    reason = rf_pkg.explain_layout_rejection(
        geometry,
        rf_pkg.parse_layout("TP8 x EP8"),
        tuple(rf_pkg.TOPOLOGIES.values()),
        rf_pkg.SolveOptions(mode="mixed", max_chips=4),
    )
    assert "not enumerated on a 8-core slice" in reason
    monkeypatch.undo()

    # Invalid --reference-layout fails fast before solve via parser.error (exit 2).
    cfg_path = locked_config_path("qwen3_5_397b_fp8.json")
    with pytest.raises(SystemExit) as exc_info:
        cli.main(
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
                "1",
                "--reference-layout",
                "TP8 x QP2",
            ]
        )
    assert exc_info.value.code == 2
    capsys.readouterr()

    # Rejected valid --reference-layout populates found=False in JSON and text summary.
    rc_json = cli.main(
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
            "1",
            "--reference-layout",
            "TP4 x PCP2 x EP8",
            "--json",
        ]
    )
    assert rc_json == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["reference"]["found"] is False
    assert "PCP1" in payload["reference"]["rejected_by"]

    rc_text = cli.main(
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
            "1",
            "--reference-layout",
            "TP4 x PCP2 x EP8",
        ]
    )
    assert rc_text == 0
    text_out = capsys.readouterr().out
    assert "not found" in text_out
    assert "PCP1" in text_out


def test_concurrency_top_k_never_crowns_slo_failing_points() -> None:
    """concurrency_top_k labels SLO-failing candidates as ruled_out, never frontier."""
    bundle = load_locked_bundle("qwen3_5_397b_fp8.json")
    geometry, dtypes = bundle.geometry, bundle.dtypes
    # First get an unconstrained solve to pick a tight TPOT SLO that fails some points.
    base_scen = rf_pkg.TrafficScenario(
        name="slo-split-probe",
        isl=2048,
        osl=256,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=None,
        concurrencies=(16,),
        kv_working_set_turns=1.0,
    )
    base_plan = rf_pkg.solve_serving_plan(
        geometry,
        dtypes,
        rf_pkg.CHIPS["tpu7x"],
        base_scen,
        rf_pkg.SolveOptions(mode="mixed", max_chips=8),
    )
    tpots = sorted(pt.est_tpot_ms for pt in base_plan.candidates)
    assert len(tpots) >= 3
    mid_tpot = 0.5 * (tpots[0] + tpots[-1])

    # Case 1: mixed SLO-met and SLO-failing points at the same concurrency.
    mixed_scen = rf_pkg.TrafficScenario(
        name="slo-split-mixed",
        isl=2048,
        osl=256,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=None,
        concurrencies=(16,),
        kv_working_set_turns=1.0,
        tpot_slo_ms=mid_tpot,
    )
    mixed_plan = rf_pkg.solve_serving_plan(
        geometry,
        dtypes,
        rf_pkg.CHIPS["tpu7x"],
        mixed_scen,
        rf_pkg.SolveOptions(mode="mixed", max_chips=8),
    )
    mixed_payload = build_solve_payload(
        mixed_plan, chip=rf_pkg.CHIPS["tpu7x"], dtypes=dtypes
    )
    entries = mixed_payload["concurrency_top_k"]["16"]
    assert any(e["point"]["slo_met"] for e in entries)
    assert any(not e["point"]["slo_met"] for e in entries)
    for entry in entries:
        if not entry["point"]["slo_met"]:
            expected_role = entry["point"].get("slo_verdict", "ruled_out")
            assert entry["frontier_role"] == expected_role
        else:
            assert entry["frontier_role"] not in ("ruled_out", "undecided")

    # Case 2: impossible SLO where all points fail -> zero frontier roles crowned.
    impossible_scen = rf_pkg.TrafficScenario(
        name="slo-split-none",
        isl=2048,
        osl=256,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=None,
        concurrencies=(16,),
        kv_working_set_turns=1.0,
        tpot_slo_ms=0.001,
    )
    impossible_plan = rf_pkg.solve_serving_plan(
        geometry,
        dtypes,
        rf_pkg.CHIPS["tpu7x"],
        impossible_scen,
        rf_pkg.SolveOptions(mode="mixed", max_chips=4),
    )
    impossible_payload = build_solve_payload(
        impossible_plan, chip=rf_pkg.CHIPS["tpu7x"], dtypes=dtypes
    )
    imp_entries = impossible_payload["concurrency_top_k"]["16"]
    assert imp_entries
    assert all(entry["frontier_role"] == "ruled_out" for entry in imp_entries)


@pytest.mark.parametrize("fixture_name", LOCKED_FIXTURE_FILES)
def test_bound_families_apply_speculative_accept_len(fixture_name: str) -> None:
    """Speculative decode bound families divide TPOT lower bound by accept_len."""
    bundle = load_locked_bundle(fixture_name)
    geometry, dtypes = bundle.geometry, bundle.dtypes
    scenario = rf_pkg.TrafficScenario(
        name="spec-decode-bound-family",
        isl=2048,
        osl=256,
        prefix_hit_rate=0.5,
        accept_len=3.0,
        think_time_s=None,
        concurrencies=(1, 16),
        kv_working_set_turns=1.0,
    )
    serving_plan = rf_pkg.solve_serving_plan(
        geometry,
        dtypes,
        rf_pkg.CHIPS["tpu7x"],
        scenario,
        rf_pkg.SolveOptions(mode="mixed", max_chips=8),
    )
    families_by_conc = rf_pkg.build_bound_families_by_concurrency(serving_plan)
    assert families_by_conc
    for entries in families_by_conc.values():
        assert entries
        prev_bound = 0.0
        for entry in entries:
            min_bound = entry["min_bound_tpot_ms"]
            assert min_bound is not None
            assert min_bound > 0.0
            expected_min = min(
                member["decode_floor"]["hard_floor_ms"] / 3.0
                for member in entry["members"]
            )
            assert min_bound == pytest.approx(expected_min)
            assert min_bound >= prev_bound - 1e-9
            prev_bound = min_bound
            for member in entry["members"]:
                est_tpot = member["est_tpot_ms"]
                assert isinstance(est_tpot, (int, float))
                assert min_bound <= est_tpot + 1e-6


@pytest.mark.parametrize("fixture_name", LOCKED_FIXTURE_FILES)
def test_pool_only_disagg_families_rank_roles_separately_and_render_clean_html(
    fixture_name: str,
) -> None:
    """Pool-only disagg ranks decode and prefill separately with clean HTML members."""
    bundle = load_locked_bundle(fixture_name)
    geometry, dtypes = bundle.geometry, bundle.dtypes
    scenario = rf_pkg.TrafficScenario(
        name="pool-only-disagg",
        isl=2048,
        osl=256,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=None,
        concurrencies=(4,),
        kv_working_set_turns=1.0,
    )
    serving_plan = rf_pkg.solve_serving_plan(
        geometry,
        dtypes,
        rf_pkg.CHIPS["tpu7x"],
        scenario,
        rf_pkg.SolveOptions(mode="disagg", max_chips=8, include_pairs=False),
    )
    payload = build_solve_payload(
        serving_plan, chip=rf_pkg.CHIPS["tpu7x"], dtypes=dtypes
    )
    families = payload["bound_families_by_concurrency"]["4"]
    decode_entries = [entry for entry in families if entry["family"][0] == "decode"]
    prefill_entries = [entry for entry in families if entry["family"][0] == "prefill"]
    assert decode_entries
    assert prefill_entries
    assert families[: len(decode_entries)] == decode_entries
    assert [entry["rank"] for entry in decode_entries] == list(
        range(1, len(decode_entries) + 1)
    )
    assert [entry["rank"] for entry in prefill_entries] == list(
        range(1, len(prefill_entries) + 1)
    )
    for entry in decode_entries:
        assert entry["min_bound_tpot_ms"] is not None
        assert entry["min_bound_ttft_ms"] is None
    for entry in prefill_entries:
        assert entry["min_bound_tpot_ms"] is None
        assert entry["min_bound_ttft_ms"] is not None
    html = rf_pkg.render_html(payload)
    assert "{'role'" not in html
    assert "{&#x27;role&#x27;" not in html


@pytest.mark.parametrize("fixture_name", LOCKED_FIXTURE_FILES)
@pytest.mark.parametrize("sort_by", ("throughput", "interactivity"))
def test_reference_layout_multi_ratio_disagg_and_sort_by(
    fixture_name: str,
    sort_by: Literal["throughput", "interactivity"],
) -> None:
    """Multi-ratio disagg reference ranking uses symmetric floor and peer count."""
    bundle = load_locked_bundle(fixture_name)
    geometry, dtypes = bundle.geometry, bundle.dtypes
    topo_2x2x2 = rf_pkg.TOPOLOGIES["2x2x2"]
    scenario = rf_pkg.TrafficScenario(
        name="disagg-multi-ratio",
        isl=2048,
        osl=256,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=None,
        concurrencies=(4,),
        kv_working_set_turns=1.0,
    )
    serving_plan = rf_pkg.solve_serving_plan(
        geometry,
        dtypes,
        rf_pkg.CHIPS["tpu7x"],
        scenario,
        rf_pkg.SolveOptions(
            mode="disagg",
            max_chips=24,
            topologies=(topo_2x2x2,),
            sort_by=sort_by,
            include_pairs=True,
        ),
    )
    assert serving_plan.candidates
    sample = serving_plan.candidates[0]
    assert sample.prefill_pool is not None
    prefill_label = rf_pkg.format_parallelism(
        {
            "tp": sample.prefill_pool.tp,
            "dp": sample.prefill_pool.dp,
            "ep": sample.prefill_pool.ep,
            "pp": sample.prefill_pool.pp,
            "pcp": sample.prefill_pool.pcp,
            "kv_split": sample.prefill_pool.kv_split,
        }
    )
    decode_label = rf_pkg.format_parallelism(
        {
            "tp": sample.decode_pool.tp,
            "dp": sample.decode_pool.dp,
            "ep": sample.decode_pool.ep,
            "pp": sample.decode_pool.pp,
            "pcp": sample.decode_pool.pcp,
            "kv_split": sample.decode_pool.kv_split,
        }
    )
    layout_text = f"P: {prefill_label} / D: {decode_label}"
    report = rf_pkg.reference_layout_report(
        serving_plan,
        rf_pkg.top_layouts(serving_plan),
        layout_text,
        "",
    )
    assert report["found"] is True
    assert report["concurrency"] == 4
    axis_groups: dict[
        tuple[
            tuple[int, int, int, int, int, int] | None,
            tuple[int, int, int, int, int, int],
        ],
        list[rf_pkg.ParetoPoint],
    ] = {}
    for cand in serving_plan.candidates:
        if cand.concurrency != 4:
            continue
        p_pool = cand.prefill_pool
        d_pool = cand.decode_pool
        p_key = (
            (p_pool.tp, p_pool.dp, p_pool.ep, p_pool.pp, p_pool.pcp, p_pool.kv_split)
            if p_pool is not None
            else None
        )
        d_key = (
            d_pool.tp,
            d_pool.dp,
            d_pool.ep,
            d_pool.pp,
            d_pool.pcp,
            d_pool.kv_split,
        )
        axis_groups.setdefault((p_key, d_key), []).append(cand)
    sample_p = sample.prefill_pool
    sample_d = sample.decode_pool
    ref_key = (
        (
            sample_p.tp,
            sample_p.dp,
            sample_p.ep,
            sample_p.pp,
            sample_p.pcp,
            sample_p.kv_split,
        ),
        (
            sample_d.tp,
            sample_d.dp,
            sample_d.ep,
            sample_d.pp,
            sample_d.pcp,
            sample_d.kv_split,
        ),
    )
    matching = axis_groups[ref_key]
    expected_best = min(
        matching,
        key=lambda cand: (
            not cand.slo_met,
            -(
                cand.est_interactivity_tok_s
                if sort_by == "interactivity"
                else cand.token_throughput_per_chip
            ),
            -(
                cand.token_throughput_per_chip
                if sort_by == "interactivity"
                else cand.est_interactivity_tok_s
            ),
            cand.total_chips,
        ),
    )
    expected_ref_bound = min(cand.decode_floor.hard_floor_ms for cand in matching)
    expected_ref_tput = max(cand.token_throughput_per_chip for cand in matching)
    expected_bound_rank = 1 + sum(
        1
        for key, pts in axis_groups.items()
        if key != ref_key
        and min(p.decode_floor.hard_floor_ms for p in pts) < expected_ref_bound - 1e-9
    )
    expected_tput_rank = 1 + sum(
        1
        for key, pts in axis_groups.items()
        if key != ref_key
        and max(p.token_throughput_per_chip for p in pts) > expected_ref_tput + 1e-9
    )
    assert report["candidates"] == len(axis_groups)
    assert report["rank_by_tpot_bound"] == expected_bound_rank
    assert report["rank_by_estimated_throughput"] == expected_tput_rank
    assert report["point"] is not None
    assert report["point"]["total_chips"] == expected_best.total_chips
    assert report["point"]["token_throughput_per_chip"] == pytest.approx(
        expected_best.token_throughput_per_chip
    )
    assert report["point"]["est_interactivity_tok_s"] == pytest.approx(
        expected_best.est_interactivity_tok_s
    )


@pytest.mark.parametrize("fixture_name", LOCKED_FIXTURE_FILES)
def test_top_k_one_preserves_reference_and_interactivity_roles(
    fixture_name: str,
    locked_config_path: Callable[[str], Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI --top-k 1 keeps families and valid reference layout, honoring --sort-by."""
    cfg_path = locked_config_path(fixture_name)
    bundle = load_locked_bundle(fixture_name)
    geometry, dtypes = bundle.geometry, bundle.dtypes
    scenario = rf_pkg.build_example_scenario("chat-8k1k").__class__(
        name="chat-8k1k-short",
        isl=8192,
        osl=1024,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=None,
        concurrencies=(1, 4),
        kv_working_set_turns=1.0,
    )
    full_plan = rf_pkg.solve_serving_plan(
        geometry,
        dtypes,
        rf_pkg.CHIPS["tpu7x"],
        scenario,
        rf_pkg.SolveOptions(mode="mixed", max_chips=8, sort_by="interactivity"),
    )
    full_families = rf_pkg.build_bound_families_by_concurrency(full_plan)
    ref_cand = full_plan.candidates[-1]
    ref_layout = rf_pkg.format_parallelism(
        {
            "tp": ref_cand.decode_pool.tp,
            "dp": ref_cand.decode_pool.dp,
            "ep": ref_cand.decode_pool.ep,
            "pp": ref_cand.decode_pool.pp,
            "pcp": ref_cand.decode_pool.pcp,
            "kv_split": ref_cand.decode_pool.kv_split,
        }
    )
    rc_json = cli.main(
        [
            "solve",
            "--config",
            str(cfg_path),
            "--example-scenario",
            "chat-8k1k",
            "--mode",
            "mixed",
            "--max-chips",
            "8",
            "--concurrency",
            "1,4",
            "--sort-by",
            "interactivity",
            "--top-k",
            "1",
            "--reference-layout",
            ref_layout,
            "--json",
        ]
    )
    assert rc_json == 0
    out = json.loads(capsys.readouterr().out)
    assert out["reference"]["found"] is True
    assert out["reference"]["concurrency"] in (1, 4)
    assert "launch" not in out["reference"]
    assert len(out["bound_families_by_concurrency"]["1"]) == len(full_families["1"])
    conc_top_k = out["concurrency_top_k"]["1"]
    assert len(conc_top_k) == 1
    assert conc_top_k[0]["frontier_role"] == "max_interactivity"
    assert conc_top_k[0]["margin_vs_rank1_pct"] == 0.0

    rc_text = cli.main(
        [
            "solve",
            "--config",
            str(cfg_path),
            "--example-scenario",
            "chat-8k1k",
            "--mode",
            "mixed",
            "--max-chips",
            "8",
            "--concurrency",
            "1,4",
            "--sort-by",
            "interactivity",
            "--top-k",
            "1",
            "--reference-layout",
            ref_layout,
        ]
    )
    assert rc_text == 0
    text_out = capsys.readouterr().out
    assert f"reference {ref_layout}: c=" in text_out
