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
"""CPU unit tests for physics-only ceiling curve export and launch-free CLI."""

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
    cli,
    emit_ceiling_rows,
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

    # Disagg ceiling on 4 chips cannot form a 1P+1D pair: warns on stderr and exits 1.
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

    # Disagg ceiling with --max-instances 1 (no --max-chips) skips pair grid and warns.
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
