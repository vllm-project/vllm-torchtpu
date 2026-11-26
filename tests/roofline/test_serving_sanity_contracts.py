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
"""Permanent domain sanity contracts for TPU serving physics and pure-L1 HTML."""

from __future__ import annotations

from dataclasses import replace

import pytest

import vllm_torchtpu.roofline as rf
from tests.roofline.conftest import load_locked_bundle
from vllm_torchtpu.roofline.constants import BF16_BYTES, FP8_BYTES, FP32_BYTES


def test_dtypes_default_names_match_bytes_and_state_provenance() -> None:
    """Dtypes defaults keep name/byte parity and record --state-dtype provenance."""
    defaults = rf.Dtypes()
    assert defaults.w_dense == FP8_BYTES
    assert defaults.w_dense_dtype == "fp8"
    assert defaults.w_expert == FP8_BYTES
    assert defaults.w_expert_dtype == "fp8"
    assert defaults.kv == FP8_BYTES
    assert defaults.kv_dtype == "fp8"
    assert defaults.state == FP32_BYTES
    assert defaults.state_dtype == "float32"
    assert defaults.state_source == "default"

    bundle = load_locked_bundle("qwen3_5_397b_fp8.json")
    assert bundle.dtypes.state == FP32_BYTES
    assert bundle.dtypes.state_dtype == "float32"
    assert bundle.dtypes.state_source == "config"

    tpu_dtypes = replace(
        bundle.dtypes,
        kv=FP8_BYTES,
        kv_dtype="fp8",
        kv_source="flag",
        state=BF16_BYTES,
        state_dtype="bf16",
        state_source="flag",
    )
    scenario = rf.TrafficScenario(
        name="state-provenance",
        isl=2048,
        osl=256,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(1,),
        kv_working_set_turns=1.0,
    )
    plan = rf.solve_serving_plan(
        bundle.geometry,
        tpu_dtypes,
        rf.CHIPS["tpu7x"],
        scenario,
        rf.SolveOptions(mode="mixed", max_chips=4),
    )
    payload = rf.build_solve_payload(plan, chip=rf.CHIPS["tpu7x"], dtypes=tpu_dtypes)
    supplied = {
        entry["name"]: entry for entry in payload["provenance"]["supplied_values"]
    }
    assert supplied["kv_dtype"]["flag"] == "--kv-dtype"
    assert supplied["kv_dtype"]["value"] == "fp8"
    assert supplied["state_dtype"]["flag"] == "--state-dtype"
    assert supplied["state_dtype"]["value"] == "bf16"


def test_glm52_fp8_tpu_cache_policy_contract() -> None:
    """Explicit FP8 KV cache increases GLM-5.2 2x2x2 survival and c=256 ceiling."""
    bundle = load_locked_bundle("glm_5_2_fp8.json")
    tpu_dtypes = replace(bundle.dtypes, kv=FP8_BYTES, kv_dtype="fp8", kv_source="flag")
    scenario = rf.build_example_scenario("agentic-64k")
    opts = rf.SolveOptions(
        mode="mixed",
        max_chips=8,
        topologies=(rf.TOPOLOGIES["2x2x2"],),
    )
    plan_hf = rf.solve_serving_plan(
        bundle.geometry, bundle.dtypes, rf.CHIPS["tpu7x"], scenario, opts
    )
    plan_tpu = rf.solve_serving_plan(
        bundle.geometry, tpu_dtypes, rf.CHIPS["tpu7x"], scenario, opts
    )
    assert plan_hf.search is not None and plan_tpu.search is not None
    assert plan_tpu.search.not_ruled_out > plan_hf.search.not_ruled_out
    assert plan_tpu.recommended is not None
    assert plan_tpu.recommended.concurrency == 256
    assert plan_tpu.recommended.token_throughput_per_chip > 100_000.0

    c64_tpu = [pt for pt in plan_tpu.candidates if pt.concurrency == 64]
    assert c64_tpu
    top_c64 = c64_tpu[0]
    assert top_c64.decode_pool.tp == 2
    assert top_c64.decode_pool.dp == 8
    assert top_c64.decode_pool.ep == 16
    assert top_c64.decode_pool.kv_split == 1


def test_qwen35_fp8_resv24_c128_tpu_cache_policy_contract() -> None:
    """Explicit TPU FP8 KV + BF16 GDN state unlocks c=128 on Qwen3.5 at resv=24GB."""
    bundle = load_locked_bundle("qwen3_5_397b_fp8.json")
    tpu_dtypes = replace(
        bundle.dtypes,
        kv=FP8_BYTES,
        kv_dtype="fp8",
        kv_source="flag",
        state=BF16_BYTES,
        state_dtype="bf16",
        state_source="flag",
    )
    scenario = rf.build_example_scenario("agentic-64k")
    opts = rf.SolveOptions(
        mode="mixed",
        max_chips=4,
        hbm_reserve_gb=24.0,
        topologies=(rf.TOPOLOGIES["2x2x1"],),
    )
    plan_hf = rf.solve_serving_plan(
        bundle.geometry, bundle.dtypes, rf.CHIPS["tpu7x"], scenario, opts
    )
    plan_tpu = rf.solve_serving_plan(
        bundle.geometry, tpu_dtypes, rf.CHIPS["tpu7x"], scenario, opts
    )
    assert plan_hf.recommended is not None
    assert plan_tpu.recommended is not None
    # Unspecified HF fallback (BF16 KV + FP32 state) stalls at c=64 (~105k).
    assert plan_hf.recommended.concurrency == 64
    assert plan_hf.recommended.token_throughput_per_chip == pytest.approx(
        104_977.6, rel=0.03
    )
    # Explicit TPU serving cache policy (FP8 KV + BF16 state) unlocks c=128 (~152k).
    assert plan_tpu.recommended.concurrency == 128
    assert plan_tpu.recommended.token_throughput_per_chip > 145_000.0
    assert plan_tpu.recommended.is_pareto is True


def test_html_report_pure_l1_per_concurrency_and_compact_size() -> None:
    """Pure-L1 HTML omits shortlist language, shows ceiling, and stays compact."""
    bundle = load_locked_bundle("qwen3_5_397b_fp8.json")
    tpu_dtypes = replace(
        bundle.dtypes,
        kv=FP8_BYTES,
        kv_dtype="fp8",
        kv_source="flag",
        state=BF16_BYTES,
        state_dtype="bf16",
        state_source="flag",
    )
    scenario = rf.build_example_scenario("coding-agent")
    plan = rf.solve_serving_plan(
        bundle.geometry,
        tpu_dtypes,
        rf.CHIPS["tpu7x"],
        scenario,
        rf.SolveOptions(
            mode="disagg",
            max_chips=28,
            prefill_topology="2x2x1",
            decode_topology="2x2x1",
            pd_ratio=(4, 3),
            include_pairs=True,
        ),
    )
    payload = rf.build_solve_payload(plan, chip=rf.CHIPS["tpu7x"], dtypes=tpu_dtypes)
    html = rf.render_html(payload)
    lower = html.lower()
    assert "shortlist" not in lower
    assert "est. 2d frontier" not in lower
    assert 'id="physical-envelope"' in html
    assert 'id="concurrency-controls"' in html
    assert 'id="candidate-explorer"' in html
    assert 'id="bound-families-table"' in html
    assert "Tput Ceiling (tok/s/chip) [bound]" in html
    assert "state=bf16" in html
    assert len(html.encode("utf-8")) < 1_000_000
    assert "https://" not in html.split("</head>", 1)[0]
