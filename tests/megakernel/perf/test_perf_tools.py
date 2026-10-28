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
"""Unit tests for C2-lite perf tooling and T1 gate (`tests/megakernel/perf/`)."""

from __future__ import annotations

import pathlib

import pytest

from tests.megakernel.perf import ablate, compiler_stats, report, sweep


def test_toolchain_mismatch_returns_invalid_toolchain() -> None:
    """Toolchain mismatch must be reported as INVALID_TOOLCHAIN (invalid gate,
    not fail).
    """
    tc_base = {
        "python": "3.12.9",
        "jax": "0.9.1",
        "jaxlib": "0.9.1",
        "libtpu": "0.0.50",
        "libtpu_so": "sha256:0123456789abcdef",
        "topology_name": "tpu7x:2x2x1",
        "libtpu_init_args": "",
        "xla_flags": "",
    }
    ok, msg = compiler_stats.check_toolchain_match(tc_base, dict(tc_base))
    assert ok is True
    assert msg == "OK"

    tc_diff = {**tc_base, "libtpu_so": "sha256:fedcba9876543210"}
    ok2, msg2 = compiler_stats.check_toolchain_match(tc_base, tc_diff)
    assert ok2 is False
    assert msg2.startswith("INVALID_TOOLCHAIN")
    assert "libtpu_so" in msg2

    d = compiler_stats.diff_compiled_buckets(
        {"toolchain": tc_base, "llo": {}},
        {"toolchain": tc_diff, "llo": {}},
    )
    assert d["status"] == "INVALID_TOOLCHAIN"
    assert d["exact_match"] is False


def test_ensure_libtpu_on_path_rejects_missing_site(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify MK_TPU_AOT_SITE raises a clear error if set to a non-existent path."""
    monkeypatch.setenv("MK_TPU_AOT_SITE", "/nonexistent/tpu_aot_site_path")
    with pytest.raises(RuntimeError, match="MK_TPU_AOT_SITE"):
        compiler_stats.ensure_libtpu_on_path(require=True)


def test_roofline_stats_matches_analytic_model() -> None:
    """Verify analytic roofline floor and headroom calculation against
    megakernel_roofline.md.
    """
    rl_b4 = compiler_stats.roofline_stats(B=4, T=1, ctx=4096, median_us=3295.2)
    assert 1950.0 <= rl_b4["roofline_floor_us"] <= 2050.0
    assert 7.20 <= rl_b4["crit_bytes_gb"] <= 7.55
    assert 58.0 <= rl_b4["hbm_bw_util_pct"] <= 63.0
    assert rl_b4["headroom_us"] > 1200.0

    rl_b64 = compiler_stats.roofline_stats(B=64, T=1, ctx=4096, median_us=20810.9)
    assert 11200.0 <= rl_b64["roofline_floor_us"] <= 11600.0
    assert 41.0 <= rl_b64["crit_bytes_gb"] <= 43.0
    assert 53.0 <= rl_b64["hbm_bw_util_pct"] <= 57.0

    # MoE TP eliminates EP load imbalance at B=4
    rl_b4_tp = compiler_stats.roofline_stats(B=4, T=1, ctx=4096, moe_tp=True)
    assert rl_b4_tp["ep_imbalance"] == 1.0
    assert rl_b4_tp["roofline_floor_us"] < rl_b4["roofline_floor_us"]


def test_plan_bucket_stats_exact_and_approx() -> None:
    """Verify CPU static planning for both exact and approx profiles and split/fused
    verify.
    """
    st_exact = compiler_stats.plan_bucket_stats("qwen35-baseline", 1, 8)
    assert st_exact["numerics"] == "exact"
    assert st_exact["fits_vmem"] is True
    assert 0 < st_exact["vmem_mib"] <= 62.0
    assert st_exact["smem_kib"] > 0

    st_split = compiler_stats.plan_bucket_stats("qwen35-candidate30-fused", 4, 32)
    assert st_split["numerics"] == "approx"
    assert st_split["fits_vmem"] is True
    assert st_split["FUSED"] is False
    assert st_split["GPR"] is True

    st_fused = compiler_stats.plan_bucket_stats(
        "qwen35-candidate30-fused", 4, 32, fused_pool_io_max_b=64
    )
    assert st_fused["fits_vmem"] is True
    assert st_fused["FUSED"] is True
    assert st_fused["GPR"] is False
    assert st_fused["GCH"] is True
    assert st_fused["vmem_mib"] < st_split["vmem_mib"]


def test_parse_and_diff_llo_dump_dir(tmp_path: pathlib.Path) -> None:
    """Verify full LLO + Mosaic normalization and per-subkernel diffing."""
    d_base = tmp_path / "base_dump"
    d_head = tmp_path / "head_dump"
    d_base.mkdir()
    d_head.mkdir()

    # Dynamic kernel counter IDs (.1 vs .9) must normalize to the same canonical
    # subkernel
    for idx, d in ((1, d_base), (9, d_head)):
        (d / "1777-0000-mosaic-dump-anonymous-post-deserialization.txt").write_text(
            f"module @qwen3_5_decode_megakernel.{idx} {{ stack_frame_id={idx} }}\n"
        )
        (d / "1777-TLP-00-memory-space-assignment-bufferinfo.txt").write_text(
            f"scoped allocation for qwen3_5_decode_megakernel.{idx} : size 57368576\n"
        )
        (d / f"1777-qwen3_5_decode_megakernel.{idx}-00-fingerprints.txt").write_text(
            "000: abcdef\n072: 123456\n"
        )
        sched_fname = (
            f"1777-qwen3_5_decode_megakernel.{idx}"
            "-71-schedule-analysis_final_bundles.txt"
        )
        (d / sched_fname).write_text(
            "total scheduled bundles: 35697\n"
            "empty scheduled bundles: 916\n"
            "non empty scheduled bundles: 34781\n"
        )
        (d / f"1777-qwen3_5_decode_megakernel.{idx}-72-final_bundles.txt").write_text(
            "0x0: { vadd }\n0x1: { vmul }\n"
        )

    llo_b = compiler_stats.parse_llo_dump_dir(d_base)
    llo_h = compiler_stats.parse_llo_dump_dir(d_head)
    assert llo_b["full_llo_mosaic_sha256"] == llo_h["full_llo_mosaic_sha256"]
    sk_b = llo_b["subkernels"]["qwen3_5_decode_megakernel"]
    assert sk_b["total_bundles"] == 35697
    assert sk_b["non_empty_bundles"] == 34781
    assert sk_b["spill_count"] == 0
    assert sk_b["scoped_vmem_bytes"] == 57368576

    tc = compiler_stats.get_toolchain_info()
    res_b = {
        "profile": "default",
        "T": 1,
        "B": 8,
        "vmem_bytes": 57368576,
        "smem_bytes": 16384,
        "primary_total_bundles": 35697,
        "toolchain": tc,
        "llo": llo_b,
        "dump_dir": str(d_base),
    }
    res_h = {
        **res_b,
        "llo": llo_h,
        "dump_dir": str(d_head),
    }
    diff_eq = compiler_stats.diff_compiled_buckets(res_b, res_h)
    assert diff_eq["status"] == "EQUAL"
    assert diff_eq["exact_match"] is True

    # Introduce a 1-bundle change in head
    (d_head / "1777-qwen3_5_decode_megakernel.9-72-final_bundles.txt").write_text(
        "0x0: { vadd }\n0x1: { vmul }\n0x2: { vsub }\n"
    )
    llo_h2 = compiler_stats.parse_llo_dump_dir(d_head)
    res_h2 = {**res_h, "llo": llo_h2, "primary_total_bundles": 35698}
    diff_out = tmp_path / "diffs"
    diff_ne = compiler_stats.diff_compiled_buckets(res_b, res_h2, diff_out_dir=diff_out)
    assert diff_ne["status"] == "DIFF"
    assert diff_ne["exact_match"] is False
    assert (
        diff_ne["subkernels"]["qwen3_5_decode_megakernel"]["final_bundles_match"]
        is False
    )
    assert len(diff_ne["diff_files"]) == 1


def test_paired_significance_and_gsm8k_gate() -> None:
    """Verify paired significance rules and GSM8K 2*SE / McNemar accuracy gate."""
    # Single run is always rejected
    s1 = report.paired_significance([10.0], [9.5])
    assert s1["status"] == "INVALID_SINGLE_RUN"

    # Statistically significant latency regression (even if small, e.g. +0.8% with tight
    # CI)
    # must fail as SIGNIFICANT_REGRESSION (noise floor is NOT an allowed slowdown!)
    s_reg = report.paired_significance(
        [10.00, 10.00, 10.00, 10.00],
        [10.08, 10.09, 10.07, 10.08],
        higher_is_better=False,
        noise_floor_pct=1.5,
    )
    assert s_reg["status"] == "SIGNIFICANT_REGRESSION"

    # Statistically significant latency improvement
    s_imp = report.paired_significance(
        [10.00, 10.02, 9.98, 10.00],
        [9.50, 9.52, 9.48, 9.50],
        higher_is_better=False,
        noise_floor_pct=1.5,
    )
    assert s_imp["status"] == "SIGNIFICANT_IMPROVEMENT"

    # Equivalent within noise floor
    s_eq = report.paired_significance(
        [10.00, 10.02, 9.98, 10.00],
        [10.01, 9.99, 10.00, 10.00],
        higher_is_better=False,
        noise_floor_pct=1.5,
    )
    assert s_eq["status"] == "WITHIN_NOISE_EQUIVALENT"

    # High variance (0 in CI, half-width > noise_floor_pct) ->
    # INCONCLUSIVE_HIGH_VARIANCE
    s_var = report.paired_significance(
        [10.0, 10.0, 10.0],
        [10.5, 9.5, 10.1],
        higher_is_better=False,
        noise_floor_pct=1.5,
    )
    assert s_var["status"] == "INCONCLUSIVE_HIGH_VARIANCE"

    # GSM8K gate: n=1319 has SE ~0.0041, 2*SE ~0.0083
    g_ok = report.gsm8k_accuracy_gate(
        0.9200,
        0.9242,
        n_questions=1319,
        acceptance_lengths=[3.38, 3.39, 3.41],
    )
    assert g_ok["passed"] is True
    assert g_ok["acceptance_length"]["spread"] == pytest.approx(0.03, abs=1e-4)


def test_markdown_tables_separate_exact_and_approx() -> None:
    """Verify report, ablate, and sweep strictly separate exact and approx profiles."""
    rows = report.collect_profile_rows(
        ["qwen35-baseline", "qwen35-candidate30-fused"],
        [(1, 8)],
        aot_compile=False,
    )
    md = report.format_markdown_tables(rows)
    assert "Table A: Exact Numerics Profiles" in md
    assert "Table B: Approx Numerics Profiles" in md

    ab_rows = ablate.run_ablation_table(
        mode="leave-one-out",
        buckets=[(1, 8)],
        flags=["tiles.gqa.kv_sub", "fused_pool_io_all_b"],
        aot_compile=False,
    )
    ab_md = ablate.format_ablation_markdown(ab_rows)
    assert "Exact-Numerics Flag Ablations" in ab_md
    assert "Approx-Numerics Flag Ablations" in ab_md

    sw_rows = sweep.sweep_bucket(
        "qwen35-candidate30-fused",
        4,
        32,
        nbd_values=(10, 12),
        nbe_values=(12, 16),
        nkv_values=(3,),
        fused_max_b_values=(64,),
    )
    assert len(sw_rows) > 0
    assert all(r["fits_vmem"] for r in sw_rows)


def test_no_frozen_references_in_owned_files() -> None:
    """Ensure owned profile/model/perf files contain no frozen kernel or internal
    provenance refs.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    owned_paths = [
        repo_root / "src/vllm_torchtpu/megakernel/models/qwen35_profile.py",
        repo_root / "src/vllm_torchtpu/megakernel/models/qwen35_candidate30.py",
        *sorted((repo_root / "src/vllm_torchtpu/megakernel/profiles").glob("*.py")),
        *sorted((repo_root / "tests/megakernel/perf").glob("*.py")),
        *sorted((repo_root / "tests/megakernel/perf").glob("*.sh")),
    ]
    forbidden = (
        "_".join(("frozen", "qwen35")),
        "_".join(("MK", "FROZEN", "QWEN35", "DIR")),
        "-".join(("hand", "written")),
        "_".join(("CANDIDATE30", "BASELINE")),
        "".join(("nan", "wei")),
        "".join(("venv", "312")),
        "_".join(("tpu", "aot", "pkgs")),
    )
    for p in owned_paths:
        txt = p.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in txt, f"Forbidden reference {token!r} found in {p}"
