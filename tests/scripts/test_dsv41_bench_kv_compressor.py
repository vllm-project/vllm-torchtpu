"""Unit tests for scripts/dsv41_bench/bench_kv_compressor.py."""

from __future__ import annotations

import pathlib
import sys
from pathlib import Path

sys.path.insert(
    0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts" / "dsv41_bench")
)

from bench_kv_compressor import (  # noqa: E402
    SUBOPS,
    compute_engram_hbm_savings_bytes,
    main,
)


def test_bench_kv_compressor_all_subops_verify_cpu(tmp_path: Path) -> None:
    json_out = tmp_path / "summary.json"
    summary = main(
        [
            "--device=cpu",
            "--subop=all",
            "--num-tokens=64",
            "--num-pages=4",
            "--page-size=16",
            "--engram-rows=512",
            "--warmup=1",
            "--iters=1",
            "--verify",
            f"--json-out={json_out}",
        ]
    )
    assert json_out.is_file()
    assert len(summary["rooflines"]) == 5
    assert len(summary["verifications"]) == 4
    for v in summary["verifications"]:
        assert v["ok"] is True
    by_name = {v["name"]: v for v in summary["verifications"]}
    assert by_name["compressor.write_fp8_only.main_words"]["bitwise_equal"] is True
    assert by_name["engram.host_vs_hbm"]["bitwise_equal"] is True


def test_engram_hbm_savings_calculation() -> None:
    savings = compute_engram_hbm_savings_bytes(tp_size=8)
    assert 20.0 <= savings["per_rank_gib"] <= 25.0
    assert 160.0 <= savings["total_gib"] <= 200.0
    assert "engram_host_vs_hbm" in SUBOPS
