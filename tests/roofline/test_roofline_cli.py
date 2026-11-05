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
"""CLI unit tests for plan and explain subcommands."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from vllm_torchtpu.roofline import CHIPS, Dtypes, Parallel, cli, from_hf_config, plan

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


def test_cli_plan_json_schema(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
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
            "90036",
            "--kv-dtype",
            "bf16",
            "--aux-dtype",
            "fp8",
            "--concurrency",
            "1,4",
            "--json",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert type(out) is list and len(out) == 2
    expected_keys = {
        "c",
        "rows",
        "binding",
        "hard_floor_ms",
        "serial_floor_ms",
        "tpot_floor_ms",
        "interactivity_ceil_tok_s",
        "output_tput_ceil_per_chip",
        "flag",
    }
    for row in out:
        assert set(row.keys()) == expected_keys
    assert out[0]["c"] == 1
    assert out[0]["hard_floor_ms"] == pytest.approx(1.0429799783783784, rel=1e-9)


def test_cli_explain_json_and_spec_window(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
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
    assert "gqa" in names and "draft_gqa" in names and "draft_fc" in names


def test_cli_rejects_non_positive_tp(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(QWEN35_CFG))
    with pytest.raises(ValueError, match="--tp must be >= 1"):
        cli.main(["explain", "--config", str(cfg_path), "--tp", "0"])


def test_cli_plan_axes_flags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
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
    total = sum(m["hbm_bytes"] for m in mods)
    assert total == pytest.approx(4_476_803_544.77, abs=1.0)


def test_plan_requires_acceptance_length_for_spec_window(tmp_path: Path) -> None:
    geom = from_hf_config(QWEN35_CFG)
    with pytest.raises(
        ValueError, match="acceptance_length is required when spec_window > 0"
    ):
        plan(
            geom,
            CHIPS["tpu7x"],
            Parallel(tp=8),
            Dtypes(),
            context=1024,
            concurrencies=[1],
            spec_window=3,
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
            "--dcn-gbps",
            "200",
            "--html",
            str(html_path),
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["scenario"]["isl"] == 4096
    assert payload["scenario"]["osl"] == 512
    assert payload["scenario"]["prefix_hit_rate"] == pytest.approx(0.75)
    assert payload["scenario"]["concurrencies"] == [16]
    assert payload["scenario"]["kv_offload_gb"] == pytest.approx(32.0)
    assert payload["gpu_memory_utilization"] == pytest.approx(0.90)
    assert payload["chip_spec"]["dcn_bw"] == pytest.approx(25.0e9)
    # TP=8, DP=1 on tpu7x (2 TensorCores/chip) budgets 4 physical chips, not 8.
    assert payload["recommended"]["total_chips"] <= 4
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
    assert sonnet_payload["scenario"]["isl"] == 550
    assert sonnet_payload["scenario"]["prefix_hit_rate"] == pytest.approx(200.0 / 550.0)
    assert sonnet_payload["scenario"]["kv_offload_gb"] == pytest.approx(16.0)
    assert ("crossover=" in stable_html.read_text()) or (
        "stable" in stable_html.read_text()
    )

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
    assert direct_payload["scenario"]["isl"] == 4096
    assert direct_payload["scenario"]["prefix_hit_rate"] == pytest.approx(0.25)

    # R3: Generic --input-len under default random dataset adds --random-prefix-len.
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
    assert gen_rand_payload["scenario"]["isl"] == 4096
    assert gen_rand_payload["scenario"]["prefix_hit_rate"] == pytest.approx(0.75)


def test_cli_module_dunder_main_help() -> None:
    py_exe = sys.executable or "python3"
    proc = subprocess.run(
        [py_exe, "-m", "vllm_torchtpu.roofline.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "usage:" in proc.stdout.lower()
    sp = cli._parse_serve_cmd(
        "vllm serve org/m --max-model-len 32768 --max-num-batched-tokens 2048 "
        '--speculative-config \'{"method":"mtp","num_speculative_tokens":3}\''
    )
    assert sp["max_model_len"] == 32768
    assert sp["max_num_batched_tokens"] == 2048
    assert sp["spec_tokens"] == 3
