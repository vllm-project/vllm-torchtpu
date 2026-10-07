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
"""Unit tests for ``scripts/dsv41_bench/bench_decoder_layer.py``."""

from __future__ import annotations

import pathlib
import sys

import pytest

BENCH_DIR = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "dsv41_bench"
sys.path.insert(0, str(BENCH_DIR))

import bench_decoder_layer  # noqa: E402


@pytest.mark.parametrize("subop", bench_decoder_layer.SUBOPS)
@pytest.mark.parametrize("mode", bench_decoder_layer.MODES)
def test_compute_layer_roofline_all_subops_and_modes(subop: str, mode: str) -> None:
    roof = bench_decoder_layer.compute_layer_roofline(
        subop, num_tokens=2048, kv_len=90112, tp_size=8, mode=mode
    )
    assert roof["subop"] == subop
    assert roof["mode"] == mode
    assert roof["flops"] > 0
    assert roof["hbm_bytes"] > 0
    assert roof["comm_bytes"] > 0
    assert roof["lower_bound_ms"] > 0.0
    if subop == "decoder_ced":
        assert roof["num_tokens"] == 128
    else:
        assert roof["num_tokens"] == 2048


def test_reconcile_40_layer_step_matches_e2e_baseline() -> None:
    rec = bench_decoder_layer.reconcile_40_layer_step(
        num_tokens=2048, ced_decoder_tokens=128
    )
    base = rec["baseline_tp"]
    opt_tp = rec["optimized_tp_ced1"]
    opt_sp_ced1 = rec["optimized_sp_ced1"]
    opt_sp = rec["optimized_sp_ced0"]
    ref = rec["e2e_reference"]

    # Cited isolated micro-benchmark sum is strictly less than total E2E XProf
    # device chunk time (78.59 ms), with the unmeasured residual (dense
    # projections, MoE dispatch/combine, KV compressor, Engram) reported.
    assert (
        0.0
        < base["ced1_chunks_1_to_48_isolated_ms"]
        < ref["baseline_100k_xprof_device_chunk_ms"]
    )
    assert ref["baseline_tp_ced1_unmeasured_residual_ms_unverified"] > 0.0
    assert base["ced0_all_40L_isolated_ms"] > base["ced1_chunks_1_to_48_isolated_ms"]

    # Runnable TP+CED=1 reduces isolated encoder & 40L kernel sums
    # without mixing SP mHC/RS+AG numbers into CED=1.
    assert (
        opt_tp["ced1_chunks_1_to_48_isolated_ms"]
        < base["ced1_chunks_1_to_48_isolated_ms"]
    )
    assert opt_tp["isolated_delta_vs_baseline_tp_ms"] == pytest.approx(-12.83, abs=1e-2)

    # Stage-scoped Encoder-SP + Decoder-TP under CED=1 reduces isolated encoder_stage
    # sum by an additional -8.88 ms/chunk at T=2048 (-21.71 ms vs baseline_tp).
    assert (
        opt_sp_ced1["ced1_chunks_1_to_48_isolated_ms"]
        < opt_tp["ced1_chunks_1_to_48_isolated_ms"]
    )
    assert opt_sp_ced1["isolated_delta_vs_optimized_tp_ced1_ms"] == pytest.approx(
        -8.88, abs=1e-2
    )
    assert opt_sp_ced1["isolated_delta_vs_baseline_tp_ms"] == pytest.approx(
        -21.71, abs=1e-2
    )

    # Runnable SP+CED=0 reports both the isolated 40L sum and the unmeasured
    # residual against the measured 144.07 ms/chunk wall time.
    assert opt_sp["ced0_all_40L_isolated_ms"] < opt_tp["ced0_all_40L_isolated_ms"]
    assert opt_sp["unmeasured_40L_dense_ep_compressor_host_gap_ms_unverified"] > 0.0


@pytest.mark.parametrize("subop", bench_decoder_layer.SUBOPS)
def test_run_decoder_layer_verification_all_subops(subop: str) -> None:
    res = bench_decoder_layer.run_decoder_layer_verification(
        subop, num_tokens=64, num_seqs=2, tp_size=8
    )
    assert res["subop"] == subop
    assert len(res["comparisons"]) > 0
    for name, comp in res["comparisons"].items():
        assert comp["ok"], f"{name} failed verification: {comp}"
        if comp["kind"] == "exact":
            assert comp["bitwise_equal"], f"{name} expected bitwise equal: {comp}"
        else:
            assert comp["cos_sim"] >= 0.9999


def test_cli_cpu_only_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    rc = bench_decoder_layer.main(
        ["--subop", "all", "--chunk-size", "128", "--cpu-only"]
    )
    assert rc == 0
    captured = capsys.readouterr().out
    assert '"four_layer_e2e_tp_vs_sp"' in captured
    assert '"four_layer_ced1_stage_split_exact"' in captured
    assert '"four_layer_ced1_replay_tail_vs_full"' in captured
