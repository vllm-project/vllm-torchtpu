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
"""CLI unit tests for plan, explain, solve, emit, and ceiling subcommands."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from vllm_torchtpu.roofline import (
    CHIPS,
    Dtypes,
    Parallel,
    ServingRequest,
    cli,
    format_parallelism,
    from_hf_config,
    plan,
)

if TYPE_CHECKING:
    from pathlib import Path

QWEN35_CFG: dict[str, object] = {
    "text_config": {
        "hidden_size": 4096,
        "num_hidden_layers": 60,
        "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 15,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "attn_output_gate": True,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 64,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 1024,
        "shared_expert_intermediate_size": 1024,
        "vocab_size": 248320,
        "mtp_num_hidden_layers": 1,
        "dtype": "bfloat16",
    }
}


def _iter_subcommand_options(
    parser: argparse.ArgumentParser,
) -> list[tuple[str, tuple[str, ...], str | None]]:
    """Collect ``(subcommand, option_strings, help)`` across root and subparsers."""
    items: list[tuple[str, tuple[str, ...], str | None]] = []

    def _walk(cmd_name: str, subparser: argparse.ArgumentParser) -> None:
        """Walk parser actions via stdlib attributes lacking public accessors."""
        actions: Sequence[argparse.Action] = getattr(subparser, "_actions", ())
        for action in actions:
            if action.option_strings:
                items.append((cmd_name, tuple(action.option_strings), action.help))
            choices = getattr(action, "choices", None)
            if isinstance(choices, Mapping):
                for sub_name, child in choices.items():
                    if isinstance(child, argparse.ArgumentParser):
                        _walk(str(sub_name), child)

    _walk("root", parser)
    return items


def test_cli_plan_json_schema(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify JSON output schema and hard floor for plan command."""
    expected_concurrency_rows = 2
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(QWEN35_CFG))
    rc = cli.main(
        [
            "plan",
            "--config",
            str(cfg_path),
            "--tp",
            "8",
            "--context",
            "1024",
            "--concurrency",
            "1,32",
            "--json",
        ]
    )
    assert rc == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == expected_concurrency_rows
    expected_keys = {
        "c",
        "rows",
        "binding",
        "hard_floor_ms",
        "serial_estimate_ms",
        "tpot_floor_ms",
        "interactivity_ceil_tok_s",
        "output_tput_ceil_per_chip",
        "flag",
    }
    assert set(rows[0].keys()) == expected_keys
    assert rows[0]["c"] == 1
    assert rows[0]["hard_floor_ms"] <= rows[1]["hard_floor_ms"]


def test_cli_explain_spec_window_includes_draft_modules(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify explain command includes speculative draft modules."""
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(QWEN35_CFG))
    rc = cli.main(
        [
            "explain",
            "--config",
            str(cfg_path),
            "--tp",
            "8",
            "--context",
            "90036",
            "--spec-window",
            "3",
            "--json",
        ]
    )
    assert rc == 0
    mods = json.loads(capsys.readouterr().out)
    names = [m["name"] for m in mods]
    assert "gqa" in names
    assert "draft_gqa" in names
    assert "draft_fc" in names


def test_cli_rejects_non_positive_tp(tmp_path: Path) -> None:
    """Verify explain command rejects non-positive tensor parallel values."""
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(QWEN35_CFG))
    with pytest.raises(ValueError, match="--tp must be >= 1"):
        cli.main(["explain", "--config", str(cfg_path), "--tp", "0"])


def test_cli_plan_axes_flags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify explain command with MoE expert parallel and FP8 KV cache."""
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(QWEN35_CFG))
    rc = cli.main(
        [
            "explain",
            "--config",
            str(cfg_path),
            "--tp",
            "8",
            "--context",
            "90036",
            "--moe",
            "ep",
            "--kv-dtype",
            "fp8",
            "--json",
        ]
    )
    assert rc == 0
    mods = json.loads(capsys.readouterr().out)
    total_phys = sum(m["hbm_bytes"] for m in mods)
    # Sum of module HBM footprints: gqa (1.603 GB) + gdn (0.715 GB)
    # + moe (2.596 GB) + lm_head (0.254 GB)
    mod_map = {m["name"]: m["hbm_bytes"] for m in mods}
    expected_phys = (
        mod_map["gqa"] + mod_map["gdn"] + mod_map["moe"] + mod_map["lm_head"]
    )
    assert total_phys == pytest.approx(expected_phys, rel=1e-6)

    gqa_mod = next(m for m in mods if m["name"] == "gqa")
    assert gqa_mod.get("hbm_bytes_min") is not None
    # Logical footprint replaces physical GQA padded allocation with
    # unpadded logical KV requirement
    expected_logical = total_phys - (gqa_mod["hbm_bytes"] - gqa_mod["hbm_bytes_min"])
    logical_total = total_phys - (gqa_mod["hbm_bytes"] - gqa_mod["hbm_bytes_min"])
    assert logical_total == pytest.approx(expected_logical, rel=1e-6)


def test_plan_requires_acceptance_length_for_spec_window(tmp_path: Path) -> None:
    """Verify acceptance length requirement when speculative window is positive."""
    geom = from_hf_config(QWEN35_CFG)
    with pytest.raises(
        ValueError, match="acceptance_length is required when spec_window > 0"
    ):
        plan(
            geom,
            CHIPS["tpu7x"],
            Parallel(tp=8),
            Dtypes(),
            ServingRequest(context=1024, concurrencies=[1], spec_window=3),
        )

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(QWEN35_CFG))
    with pytest.raises(SystemExit) as excinfo:
        cli.main(
            [
                "plan",
                "--config",
                str(cfg_path),
                "--tp",
                "8",
                "--spec-window",
                "3",
                "--context",
                "1024",
            ]
        )
    assert excinfo.value.code != 0


def test_cli_from_serve_and_bench_cmd_roundtrip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify CLI solve roundtrip from serve and bench command strings."""
    expected_isl = 4096
    expected_osl = 512
    max_chips = 4
    bf16_cfg_path = tmp_path / "synth_bf16_config.json"
    bf16_cfg_path.write_text(json.dumps(QWEN35_CFG))
    cfg_path = tmp_path / "synth_config.json"
    cfg_path.write_text(
        json.dumps({**QWEN35_CFG, "quantization_config": {"quant_method": "fp8"}})
    )
    html_path = tmp_path / "roundtrip.html"
    rc = cli.main(
        [
            "solve",
            "--from-serve-cmd",
            (
                f"vllm serve {bf16_cfg_path} --quantization fp8 "
                "--kv-cache-dtype fp8 --tensor-parallel-size 8 "
                "--data-parallel-size 1 --enable-expert-parallel "
                "--gpu-memory-utilization 0.90 --kv-offloading-size 32"
            ),
            "--from-bench-cmd",
            (
                "vllm bench serve --dataset-name random "
                "--random-prefix-len 3072 --random-input-len 1024 "
                "--random-output-len 512 --max-concurrency 16"
            ),
            "--dcn-gbit-per-s",
            "200",
            "--html",
            str(html_path),
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["scenario"]["isl"] == expected_isl
    assert payload["scenario"]["osl"] == expected_osl
    assert payload["scenario"]["prefix_hit_rate"] == pytest.approx(0.75)
    assert payload["scenario"]["concurrencies"] == [16]
    assert payload["scenario"]["kv_offload_gb"] == pytest.approx(32.0)
    assert payload["gpu_memory_utilization"] == pytest.approx(0.90)
    assert payload["chip_spec"]["dcn_bw"] == pytest.approx(25.0e9)
    # TP=8, DP=1 on tpu7x (2 TensorCores/chip) budgets 4 physical chips, not 8.
    assert payload["recommended"]["total_chips"] <= max_chips
    assert payload["winner_prefix_hit_range"] is not None
    assert "sweep_candidates" not in payload
    assert "--kv-offloading-size 32" in payload["top_k"][0]["launch"]["commands"][0]
    html_text = html_path.read_text()
    assert "sweep_candidates" not in html_text
    assert "Prefix-Hit Regime" in html_text
    assert ("crossover=" in html_text) or ("stable" in html_text)

    # Sonnet dataset --sonnet-input-len already includes --sonnet-prefix-len, and
    # direct --shared-prefix-len / --kv-offload-gb flags work standalone.
    stable_html = tmp_path / "sonnet_direct.html"
    sonnet_isl = 550
    rc_sonnet = cli.main(
        [
            "solve",
            "--config",
            str(cfg_path),
            "--from-bench-cmd",
            (
                "vllm bench serve --dataset-name sonnet "
                "--sonnet-input-len 550 --sonnet-prefix-len 200 "
                "--sonnet-output-len 150 --max-concurrency 4"
            ),
            "--kv-offload-gb",
            "16",
            "--max-chips",
            "4",
            "--mode",
            "mixed",
            "--html",
            str(stable_html),
            "--json",
        ]
    )
    assert rc_sonnet == 0
    sonnet_payload = json.loads(capsys.readouterr().out)
    assert sonnet_payload["scenario"]["isl"] == sonnet_isl
    assert sonnet_payload["scenario"]["prefix_hit_rate"] == pytest.approx(200.0 / 550.0)
    assert sonnet_payload["scenario"]["kv_offload_gb"] == pytest.approx(16.0)
    stable_text = stable_html.read_text()
    assert ("crossover=" in stable_text) or ("stable" in stable_text)

    rc_direct = cli.main(
        [
            "solve",
            "--config",
            str(cfg_path),
            "--isl",
            "4096",
            "--shared-prefix-len",
            "1024",
            "--osl",
            "256",
            "--concurrency",
            "8",
            "--max-chips",
            "4",
            "--json",
        ]
    )
    assert rc_direct == 0
    direct_payload = json.loads(capsys.readouterr().out)
    assert direct_payload["scenario"]["isl"] == expected_isl
    assert direct_payload["scenario"]["prefix_hit_rate"] == pytest.approx(0.25)

    # Generic --input-len under default random dataset adds --random-prefix-len.
    rc_gen_rand = cli.main(
        [
            "solve",
            "--config",
            str(cfg_path),
            "--from-bench-cmd",
            (
                "vllm bench serve --random-prefix-len 3072 "
                "--input-len 1024 --output-len 256 --max-concurrency 8"
            ),
            "--max-chips",
            "4",
            "--json",
        ]
    )
    assert rc_gen_rand == 0
    gen_rand_payload = json.loads(capsys.readouterr().out)
    assert gen_rand_payload["scenario"]["isl"] == expected_isl
    assert gen_rand_payload["scenario"]["prefix_hit_rate"] == pytest.approx(0.75)


def test_cli_module_dunder_main_help(capsys: pytest.CaptureFixture[str]) -> None:
    """Ensure CLI help output is emitted and exits cleanly."""
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--help"])
    out = capsys.readouterr().out
    assert excinfo.value.code == 0
    assert "usage:" in out.lower()


def test_cli_forwards_spec_tokens_and_token_caps(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify CLI forwards spec_tokens and token caps from serve cmd."""
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(QWEN35_CFG))
    expected_spec_tokens = 3
    expected_prefill_chunk = 2048
    cmd = [
        "solve",
        "--config",
        str(cfg_path),
        "--from-serve-cmd",
        (
            f"vllm serve {cfg_path} --tensor-parallel-size 8 --quantization fp8 "
            "--max-model-len 16384 --max-num-batched-tokens 2048 "
            '--speculative-config \'{"method":"mtp","num_speculative_tokens":3}\''
        ),
        "--json",
    ]
    rc = cli.main(cmd)
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["scenario"]["spec_tokens"] == expected_spec_tokens
    assert payload["scenario"]["prefill_chunk"] == expected_prefill_chunk
    launch_args = payload["top_k"][0]["launch"]["cli_args"]
    assert "--max-model-len" in launch_args
    assert "16384" in launch_args
    assert "--speculative-config" in launch_args


def test_every_cli_flag_has_help(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify every CLI flag has help text with unit/scope, aliases, and assumptions."""
    parser = cli.build_parser()
    sub_opts = _iter_subcommand_options(parser)
    assert len(sub_opts) > 0
    for sub_name, option_strings, raw_help in sub_opts:
        if not option_strings or "-h" in option_strings:
            continue
        assert raw_help is not None, f"{sub_name} {option_strings} missing help"
        help_str = raw_help.strip()
        assert bool(help_str), f"{sub_name} {option_strings} empty help"
        assert help_str != "==SUPPRESS==", (
            f"{sub_name} {option_strings} suppressed help"
        )
        assert "scope:" in help_str.lower(), (
            f"{sub_name} {option_strings} help missing scope: {help_str}"
        )
        assert "unit:" in help_str.lower(), (
            f"{sub_name} {option_strings} help missing unit: {help_str}"
        )

    rc_assump = cli.main(["--dcn-gbps", "200", "--assumptions", "--json"])
    assert rc_assump == 0
    assump_payload = json.loads(capsys.readouterr().out)
    assert "catalog" in assump_payload
    assert assump_payload["catalog"]["dcn_bw"]["value"] == pytest.approx(25.0e9)

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(QWEN35_CFG))
    rc_alias = cli.main(
        [
            "explain",
            "--config",
            str(cfg_path),
            "--tp",
            "8",
            "--ici-bw-gbytes-per-s",
            "250",
            "--json",
        ]
    )
    assert rc_alias == 0
    _ = capsys.readouterr()


def test_solve_exits_nonzero_on_empty_frontier(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify solve exits 1 on empty frontier and 2 on invalid subhost or loader err."""
    expected_error_exit = 2
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(QWEN35_CFG))

    # Empty frontier (BF16 397B weights cannot fit on 2x2x1 = 4 chips) exits 1.
    rc_empty = cli.main(
        [
            "solve",
            "--config",
            str(cfg_path),
            "--max-chips",
            "4",
            "--example-scenario",
            "chat-8k1k",
        ]
    )
    captured_empty = capsys.readouterr()
    assert rc_empty == 1
    assert "WARNING:" in captured_empty.err

    # Subhost --max-chips 1 exits cleanly with single error line and code 2.
    rc_subhost = cli.main(
        [
            "solve",
            "--config",
            str(cfg_path),
            "--max-chips",
            "1",
            "--example-scenario",
            "chat-8k1k",
        ]
    )
    captured_subhost = capsys.readouterr()
    assert rc_subhost == expected_error_exit
    assert "error:" in captured_subhost.err

    # Missing config path exits 2 with a single error line.
    rc_err = cli.main(
        [
            "solve",
            "--config",
            str(tmp_path / "missing_config.json"),
            "--max-chips",
            "4",
            "--example-scenario",
            "chat-8k1k",
        ]
    )
    captured_err = capsys.readouterr()
    assert rc_err == expected_error_exit
    err_lines = captured_err.err.strip().splitlines()
    assert len(err_lines) == 1
    assert err_lines[0].startswith("error:")


def test_plan_row_serial_estimate_ms_and_ici_measure_flag() -> None:
    """Verify plan emits serial_estimate_ms and flags ICI-bound configs to MEASURE."""
    geom = from_hf_config(QWEN35_CFG)
    low_ici_chip = replace(
        CHIPS["tpu7x"],
        ici_link_bw=1.0e6,
        ici_bw=1.0e6,
        d2d_bw=2.0e6,
    )
    rows = plan(
        geom,
        low_ici_chip,
        Parallel(tp=8),
        Dtypes(),
        ServingRequest(context=128, concurrencies=[1]),
    )
    assert len(rows) == 1
    assert rows[0].binding == "ici"
    assert rows[0].flag == "MEASURE"
    assert rows[0].serial_estimate_ms >= rows[0].hard_floor_ms


def test_cli_solve_top_k_ascii_table_and_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify --top-k defaults to 5, drives JSON and ASCII tables, and rejects <=0."""
    expected_default_k = 5
    expected_k_2 = 2
    expected_pp_8 = 8
    expected_err_code = 2
    cfg_path = tmp_path / "qwen35_fp8.json"
    raw_text_cfg = QWEN35_CFG["text_config"]
    assert isinstance(raw_text_cfg, dict)
    cfg_path.write_text(
        json.dumps(
            {
                **QWEN35_CFG,
                "text_config": {
                    **raw_text_cfg,
                    "quantization_config": {"quant_method": "fp8"},
                },
            }
        )
    )

    rc_5 = cli.main(
        [
            "solve",
            "--config",
            str(cfg_path),
            "--example-scenario",
            "coding-agent",
            "--max-chips",
            "64",
            "--kv-offload-gb",
            "64",
            "--json",
        ]
    )
    assert rc_5 == 0
    payload_5 = json.loads(capsys.readouterr().out)
    assert len(payload_5["top_k"]) == expected_default_k
    assert any(
        (entry["point"].get("prefill_pool") or {}).get("pp") == expected_pp_8
        for entry in payload_5["top_k"]
    )

    rc_2_json = cli.main(
        [
            "solve",
            "--config",
            str(cfg_path),
            "--example-scenario",
            "coding-agent",
            "--max-chips",
            "32",
            "--top-k",
            "2",
            "--json",
        ]
    )
    assert rc_2_json == 0
    payload_2 = json.loads(capsys.readouterr().out)
    assert len(payload_2["top_k"]) == expected_k_2

    rc_2_txt = cli.main(
        [
            "solve",
            "--config",
            str(cfg_path),
            "--example-scenario",
            "coding-agent",
            "--max-chips",
            "32",
            "--top-k",
            "2",
        ]
    )
    assert rc_2_txt == 0
    out_lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert "rank" in out_lines[0]
    assert "role" in out_lines[0]
    assert len(out_lines) == 1 + expected_k_2
    assert out_lines[1].split()[0] == "1"
    assert out_lines[2].split()[0] == "2"

    with pytest.raises(SystemExit) as excinfo:
        cli.main(
            [
                "solve",
                "--config",
                str(cfg_path),
                "--example-scenario",
                "coding-agent",
                "--max-chips",
                "64",
                "--top-k",
                "0",
            ]
        )
    assert excinfo.value.code == expected_err_code
    _ = capsys.readouterr()


def test_cli_solve_rejects_nonpositive_max_instances(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify solve rejects --max-instances <= 0 with parser error exit code."""
    expected_err_code = 2
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "model_type": "qwen3_moe",
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 4,
                "head_dim": 128,
                "intermediate_size": 12288,
                "moe_intermediate_size": 1536,
                "num_experts": 128,
                "num_experts_per_tok": 8,
                "vocab_size": 151936,
                "dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as excinfo:
        cli.main(
            [
                "solve",
                "--config",
                str(cfg_path),
                "--chip",
                "tpu7x",
                "--example-scenario",
                "coding-agent",
                "--max-instances",
                "0",
            ]
        )
    assert excinfo.value.code == expected_err_code
    captured = capsys.readouterr()
    assert "--max-instances" in captured.err


def test_cli_plan_and_explain_auto_resolve_hf_dtypes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify plan/explain inherit dtypes from HF config when flags omitted."""
    cfg_dict = {
        "model_type": "llama",
        "hidden_size": 2048,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "intermediate_size": 8192,
        "vocab_size": 32000,
        "torch_dtype": "float16",
        "quantization_config": {"quant_method": "fp8"},
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg_dict), encoding="utf-8")

    # Test plan subcommand
    rc_plan = cli.main(
        [
            "plan",
            "--config",
            str(cfg_path),
            "--chip",
            "tpu7x",
            "--tp",
            "4",
            "--context",
            "1024",
            "--concurrency",
            "1,4",
            "--json",
        ]
    )
    assert rc_plan == 0
    plan_auto = json.loads(capsys.readouterr().out)
    expected_concs = 2
    assert len(plan_auto) == expected_concs

    rc_plan_fp8 = cli.main(
        [
            "plan",
            "--config",
            str(cfg_path),
            "--chip",
            "tpu7x",
            "--tp",
            "4",
            "--context",
            "1024",
            "--concurrency",
            "1,4",
            "--weight-dtype",
            "fp8",
            "--json",
        ]
    )
    assert rc_plan_fp8 == 0
    plan_fp8 = json.loads(capsys.readouterr().out)
    assert plan_auto[0]["tpot_floor_ms"] == pytest.approx(plan_fp8[0]["tpot_floor_ms"])

    # Test explain subcommand auto-resolves fp8 weight bytes (< bf16 weight bytes)
    rc_explain = cli.main(
        [
            "explain",
            "--config",
            str(cfg_path),
            "--chip",
            "tpu7x",
            "--tp",
            "4",
            "--batch",
            "4",
            "--context",
            "1024",
            "--json",
        ]
    )
    assert rc_explain == 0
    explain_auto = json.loads(capsys.readouterr().out)
    rc_explain_bf16 = cli.main(
        [
            "explain",
            "--config",
            str(cfg_path),
            "--chip",
            "tpu7x",
            "--tp",
            "4",
            "--batch",
            "4",
            "--context",
            "1024",
            "--weight-dtype",
            "bf16",
            "--json",
        ]
    )
    assert rc_explain_bf16 == 0
    explain_bf16 = json.loads(capsys.readouterr().out)
    auto_hbm = sum(float(row["hbm_bytes"]) for row in explain_auto)
    bf16_hbm = sum(float(row["hbm_bytes"]) for row in explain_bf16)
    assert auto_hbm < bf16_hbm * 0.75

    bf16_cfg = {
        "model_type": "llama",
        "hidden_size": 2048,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "intermediate_size": 8192,
        "vocab_size": 32000,
        "torch_dtype": "bfloat16",
    }
    bf16_cfg_path = tmp_path / "bf16_config.json"
    bf16_cfg_path.write_text(json.dumps(bf16_cfg), encoding="utf-8")

    rc_explain_bf16_auto = cli.main(
        [
            "explain",
            "--config",
            str(bf16_cfg_path),
            "--chip",
            "tpu7x",
            "--tp",
            "4",
            "--batch",
            "4",
            "--context",
            "1024",
            "--json",
        ]
    )
    assert rc_explain_bf16_auto == 0
    explain_bf16_auto = json.loads(capsys.readouterr().out)
    assert sum(float(row["hbm_bytes"]) for row in explain_bf16_auto) == pytest.approx(
        bf16_hbm
    )


def test_cli_subcommand_module_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify python -m vllm_torchtpu.roofline --help runs and prints help."""
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "usage:" in out.lower()
    assert "roofline" in out.lower()


def test_cli_solve_ascii_table_format(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify non-JSON solve outputs formatted ASCII table with headers."""
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(QWEN35_CFG), encoding="utf-8")
    rc = cli.main(
        [
            "solve",
            "--config",
            str(cfg_path),
            "--chip",
            "tpu7x",
            "--example-scenario",
            "coding-agent",
            "--max-chips",
            "32",
            "--top-k",
            "3",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "rank" in out
    assert "role" in out
    assert "pools" in out
    assert "# Note: bound_* columns are hard physical lower bounds;" in out


def test_cli_solve_html_writes_report(tmp_path: Path) -> None:
    """Verify solve --html writes report containing Search Budget:."""
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(QWEN35_CFG), encoding="utf-8")
    html_path = tmp_path / "report.html"
    rc = cli.main(
        [
            "solve",
            "--config",
            str(cfg_path),
            "--chip",
            "tpu7x",
            "--example-scenario",
            "coding-agent",
            "--max-chips",
            "32",
            "--html",
            str(html_path),
            "--json",
        ]
    )
    assert rc == 0
    assert html_path.exists()
    html_text = html_path.read_text(encoding="utf-8")
    assert "Search Budget:" in html_text
    assert "TPU Roofline Solve Report" in html_text


def test_serve_command_acceptance_length_is_parsed_and_no_slope_is_fitted(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify synthetic_acceptance_length is parsed with no fitted slope."""
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(QWEN35_CFG))

    spec_cfg = (
        '{"method":"mtp","num_speculative_tokens":3,"synthetic_acceptance_length":3.32}'
    )
    cmd_with_al = [
        "solve",
        "--config",
        str(cfg_path),
        "--from-serve-cmd",
        (
            f"vllm serve {cfg_path} --tensor-parallel-size 8 --quantization fp8 "
            f"--max-model-len 16384 --max-num-batched-tokens 2048 "
            f"--speculative-config '{spec_cfg}'"
        ),
        "--json",
    ]
    rc1 = cli.main(cmd_with_al)
    assert rc1 == 0
    payload1 = json.loads(capsys.readouterr().out)
    assert payload1["scenario"]["accept_len"] == pytest.approx(3.32)

    cmd_without_al = [
        "solve",
        "--config",
        str(cfg_path),
        "--from-serve-cmd",
        (
            f"vllm serve {cfg_path} --tensor-parallel-size 8 --quantization fp8 "
            "--max-model-len 16384 --max-num-batched-tokens 2048 "
            '--speculative-config \'{"method":"mtp","num_speculative_tokens":3}\''
        ),
        "--json",
    ]
    rc2 = cli.main(cmd_without_al)
    assert rc2 == 0
    payload2 = json.loads(capsys.readouterr().out)
    assert payload2["scenario"]["accept_len"] == pytest.approx(4.0)
    assert any(
        "acceptance length" in str(w).lower() for w in payload2.get("warnings", ())
    )
    assert not hasattr(cli, "_SPEC_ACCEPT_SLOPE")


def test_top_layout_tables_have_no_margin_column(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify solve text and HTML output contain no margin column or signed percent."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(QWEN35_CFG), encoding="utf-8")
    html_path = tmp_path / "report.html"
    exit_code = cli.main(
        [
            "solve",
            "--config",
            str(config_path),
            "--chip",
            "tpu7x",
            "--example-scenario",
            "coding-agent",
            "--max-chips",
            "32",
            "--top-k",
            "3",
            "--html",
            str(html_path),
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    header_line = out.splitlines()[0]
    assert "margin" not in header_line.lower()
    signed_pct_pattern = re.compile(r"[+-]\d+(\.\d+)?%")
    for line in out.splitlines()[1:]:
        assert not signed_pct_pattern.search(line)

    assert html_path.exists()
    html = html_path.read_text(encoding="utf-8")
    table_match = re.search(
        r'<section id="top-layouts-table".*?</section>', html, re.DOTALL
    )
    assert table_match is not None
    table_html = table_match.group(0)
    thead_match = re.search(r"<thead>(.*?)</thead>", table_html, re.DOTALL)
    assert thead_match is not None
    assert "margin" not in thead_match.group(1).lower()
    assert not signed_pct_pattern.search(table_html)


def test_reference_layout_is_reported_with_both_ranks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify --reference-layout reports ranks for matches and rejection reason."""
    config_path = tmp_path / "qwen35_fp8.json"
    raw_text_config = QWEN35_CFG["text_config"]
    assert isinstance(raw_text_config, dict)
    config_path.write_text(
        json.dumps(
            {
                **QWEN35_CFG,
                "text_config": {
                    **raw_text_config,
                    "quantization_config": {"quant_method": "fp8"},
                },
            }
        )
    )

    base_arguments = [
        "solve",
        "--config",
        str(config_path),
        "--isl",
        "105000",
        "--osl",
        "1000",
        "--prefix-hit-rate",
        "0.95",
        "--accept-len",
        "3.39",
        "--think-time-s",
        "0.0",
        "--concurrency",
        "8",
        "--mode",
        "mixed",
        "--max-chips",
        "4",
        "--json",
    ]

    exit_code_base = cli.main(base_arguments)
    assert exit_code_base == 0
    payload_base = json.loads(capsys.readouterr().out)
    assert "reference" not in payload_base

    # Match case: TP8 x EP8
    exit_code_match = cli.main([*base_arguments, "--reference-layout", "TP8 x EP8"])
    assert exit_code_match == 0
    payload_match = json.loads(capsys.readouterr().out)
    assert "reference" in payload_match
    reference_match = payload_match["reference"]
    assert reference_match["found"] is True
    assert 1 <= reference_match["rank_by_tpot_bound"] <= reference_match["candidates"]
    assert (
        1
        <= reference_match["rank_by_estimated_throughput"]
        <= reference_match["candidates"]
    )
    assert payload_match["top_k"] == payload_base["top_k"]

    # Rejection case: TP3
    exit_code_rejection = cli.main([*base_arguments, "--reference-layout", "TP3"])
    assert exit_code_rejection == 0
    payload_rejection = json.loads(capsys.readouterr().out)
    assert "reference" in payload_rejection
    reference_rejection = payload_rejection["reference"]
    assert reference_rejection["found"] is False
    assert bool(reference_rejection["rejected_by"])
    assert payload_rejection["top_k"] == payload_base["top_k"]


def test_reference_layout_rejection_names_the_rule(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify reference layout rejection reports the specific rule breached."""
    config_path = tmp_path / "qwen35_fp8.json"
    raw_text_config = QWEN35_CFG["text_config"]
    assert isinstance(raw_text_config, dict)
    config_path.write_text(
        json.dumps(
            {
                **QWEN35_CFG,
                "text_config": {
                    **raw_text_config,
                    "quantization_config": {"quant_method": "fp8"},
                },
            }
        )
    )

    base_arguments = [
        "solve",
        "--config",
        str(config_path),
        "--isl",
        "105000",
        "--osl",
        "1000",
        "--prefix-hit-rate",
        "0.95",
        "--accept-len",
        "3.39",
        "--think-time-s",
        "0.0",
        "--concurrency",
        "8",
        "--json",
    ]

    # (a) "P: TP8 x DP2 / D: TP8" in disaggregated mode: rejected_by contains "DP1"
    exit_code_a = cli.main(
        [
            *base_arguments,
            "--mode",
            "disagg",
            "--max-chips",
            "8",
            "--reference-layout",
            "P: TP8 x DP2 / D: TP8",
        ]
    )
    assert exit_code_a == 0
    payload_a = json.loads(capsys.readouterr().out)
    assert "DP1" in payload_a["reference"]["rejected_by"]

    # (b) a single-pool layout in disaggregated mode: contains "mixed mode"
    exit_code_b = cli.main(
        [
            *base_arguments,
            "--mode",
            "disagg",
            "--max-chips",
            "8",
            "--reference-layout",
            "TP8 x EP8",
        ]
    )
    assert exit_code_b == 0
    payload_b = json.loads(capsys.readouterr().out)
    assert "mixed mode" in payload_b["reference"]["rejected_by"]

    # (c) "P: PP8 / D: TP8 x EP8 x DCP4" in mixed mode: contains "disaggregated mode"
    exit_code_c = cli.main(
        [
            *base_arguments,
            "--mode",
            "mixed",
            "--max-chips",
            "4",
            "--reference-layout",
            "P: PP8 / D: TP8 x EP8 x DCP4",
        ]
    )
    assert exit_code_c == 0
    payload_c = json.loads(capsys.readouterr().out)
    assert "disaggregated mode" in payload_c["reference"]["rejected_by"]

    # (d) in disaggregated mode, take first candidate of unflagged run's top_k:
    exit_code_d_base = cli.main(
        [*base_arguments, "--mode", "disagg", "--max-chips", "8"]
    )
    assert exit_code_d_base == 0
    payload_d_base = json.loads(capsys.readouterr().out)
    top_candidate = payload_d_base["top_k"][0]
    prefill_pool = top_candidate["point"]["prefill_pool"]
    decode_pool = top_candidate["point"]["decode_pool"]
    reference_layout_d = (
        f"P: {format_parallelism(prefill_pool)} / D: {format_parallelism(decode_pool)}"
    )

    exit_code_d = cli.main(
        [
            *base_arguments,
            "--mode",
            "disagg",
            "--max-chips",
            "8",
            "--reference-layout",
            reference_layout_d,
        ]
    )
    assert exit_code_d == 0
    payload_d = json.loads(capsys.readouterr().out)
    reference_d = payload_d["reference"]
    assert reference_d["found"] is True
    assert 1 <= reference_d["rank_by_tpot_bound"] <= reference_d["candidates"]
    assert 1 <= reference_d["rank_by_estimated_throughput"] <= reference_d["candidates"]
    assert reference_d["in_top_k"] is True


def test_reference_layout_row_in_html_report(tmp_path: Path) -> None:
    """Verify HTML report includes rejection sentence or reference row with launch."""
    config_path = tmp_path / "qwen35_fp8.json"
    raw_text_config = QWEN35_CFG["text_config"]
    assert isinstance(raw_text_config, dict)
    config_path.write_text(
        json.dumps(
            {
                **QWEN35_CFG,
                "text_config": {
                    **raw_text_config,
                    "quantization_config": {"quant_method": "fp8"},
                },
            }
        )
    )
    html_path_unfound = tmp_path / "report_unfound.html"
    exit_code_unfound = cli.main(
        [
            "solve",
            "--config",
            str(config_path),
            "--isl",
            "105000",
            "--osl",
            "1000",
            "--prefix-hit-rate",
            "0.95",
            "--accept-len",
            "3.39",
            "--think-time-s",
            "0.0",
            "--concurrency",
            "8",
            "--mode",
            "mixed",
            "--max-chips",
            "4",
            "--reference-layout",
            "TP3",
            "--html",
            str(html_path_unfound),
            "--json",
        ]
    )
    assert exit_code_unfound == 0
    html_text_unfound = html_path_unfound.read_text(encoding="utf-8")
    assert "# reference layout TP3: not enumerated" in html_text_unfound
    assert "TP3 does not divide the attention heads" in html_text_unfound

    html_path_found = tmp_path / "report_found.html"
    exit_code_found = cli.main(
        [
            "solve",
            "--config",
            str(config_path),
            "--isl",
            "105000",
            "--osl",
            "1000",
            "--prefix-hit-rate",
            "0.95",
            "--accept-len",
            "3.39",
            "--think-time-s",
            "0.0",
            "--concurrency",
            "8",
            "--mode",
            "mixed",
            "--max-chips",
            "4",
            "--top-k",
            "1",
            "--reference-layout",
            "TP8 x EP8",
            "--html",
            str(html_path_found),
            "--json",
        ]
    )
    assert exit_code_found == 0
    html_text_found = html_path_found.read_text(encoding="utf-8")
    reference_row = re.search(
        r"<tr><td>-</td><td><code>reference</code></td>.*?</tr>",
        html_text_found,
        re.DOTALL,
    )
    assert reference_row is not None
    assert "vllm serve" in reference_row.group(0)


def test_example_scenario_with_other_draft_count_assumes_all_drafts_accepted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify other draft count assumes all drafts accepted when unstated."""
    config_path = tmp_path / "config.json"
    raw_text_config = QWEN35_CFG["text_config"]
    assert isinstance(raw_text_config, dict)
    config_path.write_text(
        json.dumps(
            {
                **QWEN35_CFG,
                "text_config": {
                    **raw_text_config,
                    "quantization_config": {"quant_method": "fp8"},
                },
            }
        )
    )

    spec_argument = (
        '--speculative-config \'{"method":"mtp","num_speculative_tokens":3}\''
    )

    # (a) No draft tokens of its own, draft count 3 in serve cmd: accept_len == 4.0
    exit_code_a = cli.main(
        [
            "solve",
            "--config",
            str(config_path),
            "--chip",
            "tpu7x",
            "--example-scenario",
            "chat-8k1k",
            "--max-chips",
            "4",
            "--from-serve-cmd",
            f"vllm serve model --tensor-parallel-size 4 {spec_argument}",
            "--json",
        ]
    )
    assert exit_code_a == 0
    payload_a = json.loads(capsys.readouterr().out)
    expected_accept_len_a = 4.0
    assert payload_a["scenario"]["accept_len"] == expected_accept_len_a
    assert any("acceptance length" in w.lower() for w in payload_a["warnings"])

    # (b) With --accept-len 2.5: accept_len == 2.5 and no acceptance length warning
    expected_accept_len_b = 2.5
    exit_code_b = cli.main(
        [
            "solve",
            "--config",
            str(config_path),
            "--chip",
            "tpu7x",
            "--example-scenario",
            "chat-8k1k",
            "--accept-len",
            str(expected_accept_len_b),
            "--max-chips",
            "4",
            "--from-serve-cmd",
            f"vllm serve model --tensor-parallel-size 4 {spec_argument}",
            "--json",
        ]
    )
    assert exit_code_b == 0
    payload_b = json.loads(capsys.readouterr().out)
    assert payload_b["scenario"]["accept_len"] == expected_accept_len_b
    assert not any("acceptance length" in w.lower() for w in payload_b["warnings"])

    # (c) Example scenario whose own draft count equals serve command's: kept.
    exit_code_c = cli.main(
        [
            "solve",
            "--config",
            str(config_path),
            "--chip",
            "tpu7x",
            "--example-scenario",
            "coding-agent",
            "--max-chips",
            "4",
            "--from-serve-cmd",
            f"vllm serve model --tensor-parallel-size 4 {spec_argument}",
            "--json",
        ]
    )
    assert exit_code_c == 0
    payload_c = json.loads(capsys.readouterr().out)
    expected_accept_len_c = 3.0
    assert payload_c["scenario"]["accept_len"] == expected_accept_len_c
    assert not any("acceptance length" in w.lower() for w in payload_c["warnings"])
