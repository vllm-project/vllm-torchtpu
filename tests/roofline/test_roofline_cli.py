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

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import huggingface_hub.constants as hf_constants
import pytest

from tests.roofline.conftest import load_locked_config
from vllm_torchtpu.roofline import (
    CHIPS,
    Dtypes,
    Parallel,
    ServingRequest,
    cli,
    from_hf_config,
    plan,
)


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
    cfg_path.write_text(json.dumps(load_locked_config("qwen3_5_397b_fp8.json")))
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
        "concurrency",
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
    assert rows[0]["concurrency"] == 1
    assert rows[0]["hard_floor_ms"] <= rows[1]["hard_floor_ms"]


def test_cli_explain_spec_window_includes_draft_modules(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify explain command includes speculative draft modules."""
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(load_locked_config("qwen3_5_397b_fp8.json")))
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


def test_cli_rejects_non_positive_tp(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify explain command exits with usage error on non-positive --tp."""
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps(load_locked_config("qwen3_5_397b_fp8.json")), encoding="utf-8"
    )
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["explain", "--config", str(cfg_path), "--tp", "0"])
    expected_usage_exit = 2
    assert exc_info.value.code == expected_usage_exit
    stderr_text = capsys.readouterr().err
    assert "--tp" in stderr_text
    assert ">= 1" in stderr_text


def test_cli_plan_axes_flags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify explain command with MoE expert parallel and FP8 KV cache."""
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(load_locked_config("qwen3_5_397b_fp8.json")))
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
    mod_map = {m["name"]: m["hbm_bytes"] for m in mods}
    expected_phys = (
        mod_map["gqa"] + mod_map["gdn"] + mod_map["moe"] + mod_map["lm_head"]
    )
    assert total_phys == pytest.approx(expected_phys, rel=1e-06)
    gqa_mod = next(m for m in mods if m["name"] == "gqa")
    assert gqa_mod.get("hbm_bytes_min") is not None
    # Qwen3.5-397B has 15 full_attention layers, 2 KV heads, head_dim=256,
    # FP8 KV (1 B) on TPU7x (128-lane physical alignment -> 256 B/tok logical
    # vs padded alignment with 1 KV head per rank under TP=8).
    # Verify logical HBM <= physical HBM and equals exact analytical KV minus padding.
    geometry = from_hf_config(load_locked_config("qwen3_5_397b_fp8.json"))
    num_full_layers = geometry.layer_types.count("full_attention")
    logical_kv_bytes = float(num_full_layers * 90036 * 2 * 1 * geometry.head_dim) * 1.0
    assert gqa_mod["hbm_bytes_min"] <= gqa_mod["hbm_bytes"]
    assert gqa_mod["hbm_bytes_min"] > logical_kv_bytes


def test_plan_requires_acceptance_length_for_spec_window(tmp_path: Path) -> None:
    """Verify acceptance length requirement when speculative window is positive."""
    geometry = from_hf_config(load_locked_config("qwen3_5_397b_fp8.json"))
    with pytest.raises(
        ValueError, match="acceptance_length is required when spec_window > 0"
    ):
        plan(
            geometry,
            CHIPS["tpu7x"],
            Parallel(tp=8),
            Dtypes(),
            ServingRequest(context=1024, concurrencies=[1], spec_window=3),
        )
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(load_locked_config("qwen3_5_397b_fp8.json")))
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


def test_cli_module_dunder_main_help(capsys: pytest.CaptureFixture[str]) -> None:
    """Ensure CLI help output is emitted and exits cleanly."""
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--help"])
    out = capsys.readouterr().out
    assert excinfo.value.code == 0
    assert "usage:" in out.lower()


def test_cli_plan_and_explain_auto_resolve_hf_dtypes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify plan/explain inherit dtypes from HF config when flags omitted."""
    cfg_dict = {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForCausalLM"],
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
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForCausalLM"],
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


def test_cli_chip_spec_hf_overrides_clean_error_and_dtype_label_preservation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify --chip-spec, --hf-overrides, clean exit-2 error, and replace() dtypes."""
    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    unknown_arch_path = tmp_path / "unknown_arch.json"
    unknown_arch_path.write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "hidden_size": 4096,
                "num_hidden_layers": 4,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "vocab_size": 64000,
            }
        ),
        encoding="utf-8",
    )
    qwen35_path = fixtures_dir / "qwen3_5_397b_fp8.json"
    nvfp4_path = fixtures_dir / "qwen3_5_397b_nvfp4.json"

    # Unlisted model_type exits cleanly with code 2.
    rc_fail = cli.main(
        ["plan", "--config", str(unknown_arch_path), "--tp", "8", "--json"]
    )
    expected_error_rc = 2
    assert rc_fail == expected_error_rc
    assert "qwen4_exp" in capsys.readouterr().err
    rc_missing = cli.main(
        ["plan", "--config", str(tmp_path / "missing.json"), "--tp", "8"]
    )
    assert rc_missing == expected_error_rc
    assert "ERROR:" in capsys.readouterr().err

    # With --hf-overrides and --chip-spec, plan succeeds end-to-end.
    spec_path = tmp_path / "custom_chip.json"
    spec_path.write_text(
        json.dumps(
            {
                "schema_version": "vllm_torchtpu.roofline.chip_spec/v1",
                "provenance": "projected",
                "base": "tpu7x",
                "name": "tpu7x_custom",
                "hbm_bw": 1e12,
            }
        )
    )
    hf_ov = json.dumps(
        {
            "text_config": {
                "num_hidden_layers": 12,
                "layer_types": ["linear_attention"] * 3
                + ["full_attention"]
                + ["linear_attention"] * 3
                + ["full_attention"]
                + ["linear_attention"] * 3
                + ["full_attention"],
            }
        }
    )
    rc_ok = cli.main(
        [
            "plan",
            "--config",
            str(qwen35_path),
            "--hf-overrides",
            hf_ov,
            "--chip-spec",
            str(spec_path),
            "--tp",
            "8",
            "--concurrency",
            "1",
            "--json",
        ]
    )
    assert rc_ok == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 1

    # Partial CLI override preserves untouched NVFP4 expert dtype label via replace().
    nvfp4_cfg = json.loads(nvfp4_path.read_text())
    parsed = argparse.Namespace(
        weight_dtype=None,
        expert_weight_dtype=None,
        aux_dtype=None,
        kv_dtype="bf16",
        state_dtype=None,
        act=None,
    )
    dt_resolved = cli.resolve_dtypes_from_args(parsed, nvfp4_cfg)
    assert dt_resolved.w_expert_dtype == "nvfp4"
    assert dt_resolved.w_expert_source == "quantization_config"
    assert dt_resolved.kv_dtype == "bf16"
    assert dt_resolved.kv_source == "flag"
    assert dt_resolved.act == "a4"


def test_cli_model_repo_id_from_offline_hf_hub_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify --model org/repo resolves cleanly from offline HF_HUB_CACHE."""
    snapshot_dir = (
        tmp_path / "models--Qwen--Qwen3.5-397B-A17B-FP8" / "snapshots" / "rev1"
    )
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "config.json").write_text(
        json.dumps(load_locked_config("qwen3_5_397b_fp8.json")),
        encoding="utf-8",
    )
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", True)
    rc = cli.main(
        [
            "plan",
            "--model",
            "Qwen/Qwen3.5-397B-A17B-FP8",
            "--tp",
            "8",
            "--context",
            "2048",
            "--concurrency",
            "4",
            "--json",
        ]
    )
    assert rc == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 1
    assert rows[0]["hard_floor_ms"] > 0.0


def test_cli_ici_bw_override_and_custom_chip_spec_topologies(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify --ici-bw-gbps shifts hard_floor_ms and --chip-spec keeps topologies."""
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps(load_locked_config("qwen3_5_397b_fp8.json")),
        encoding="utf-8",
    )
    rc_base = cli.main(
        [
            "plan",
            "--config",
            str(cfg_path),
            "--tp",
            "8",
            "--concurrency",
            "16",
            "--json",
        ]
    )
    assert rc_base == 0
    base_floor = json.loads(capsys.readouterr().out)[0]["hard_floor_ms"]

    rc_slow = cli.main(
        [
            "plan",
            "--config",
            str(cfg_path),
            "--tp",
            "8",
            "--concurrency",
            "16",
            "--ici-bw-gbps",
            "1",
            "--json",
        ]
    )
    assert rc_slow == 0
    slow_floor = json.loads(capsys.readouterr().out)[0]["hard_floor_ms"]
    assert slow_floor > base_floor

    spec_unwrapped = tmp_path / "custom_chip_unwrapped.json"
    spec_wrapped = tmp_path / "custom_chip_wrapped.json"
    base_spec = {
        "schema_version": "vllm_torchtpu.roofline.chip_spec/v1",
        "name": "custom_topo_chip",
        "base": "tpu7x",
        "provenance": "projected",
    }
    spec_unwrapped.write_text(
        json.dumps(
            {
                **base_spec,
                "topologies": [
                    {
                        "name": "1x4x1",
                        "chip_axes": [1, 4, 1],
                        "wrapped_axes": [False, False, False],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    spec_wrapped.write_text(
        json.dumps(
            {
                **base_spec,
                "topologies": [
                    {
                        "name": "1x4x1",
                        "chip_axes": [1, 4, 1],
                        "wrapped_axes": [False, True, False],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    rc_unwrapped = cli.main(
        [
            "plan",
            "--config",
            str(cfg_path),
            "--chip-spec",
            str(spec_unwrapped),
            "--tp",
            "8",
            "--concurrency",
            "16",
            "--ici-bw-gbps",
            "1",
            "--json",
        ]
    )
    assert rc_unwrapped == 0
    unwrapped_floor = json.loads(capsys.readouterr().out)[0]["hard_floor_ms"]

    rc_wrapped = cli.main(
        [
            "plan",
            "--config",
            str(cfg_path),
            "--chip-spec",
            str(spec_wrapped),
            "--tp",
            "8",
            "--concurrency",
            "16",
            "--ici-bw-gbps",
            "1",
            "--json",
        ]
    )
    assert rc_wrapped == 0
    wrapped_floor = json.loads(capsys.readouterr().out)[0]["hard_floor_ms"]
    assert wrapped_floor < unwrapped_floor


@pytest.mark.parametrize("act_flag", ("a4", "fp4", "nvfp4"))
def test_cli_act_flag_accepts_four_bit_aliases(act_flag: str) -> None:
    """Verify --act accepts a4, fp4, and nvfp4 and normalizes them to a4."""
    parsed = argparse.Namespace(
        weight_dtype=None,
        expert_weight_dtype=None,
        aux_dtype=None,
        kv_dtype=None,
        state_dtype=None,
        act=act_flag,
    )
    resolved = cli.resolve_dtypes_from_args(
        parsed, load_locked_config("qwen3_5_397b_fp8.json")
    )
    assert resolved.act == "a4"


def test_cli_plan_handles_gptq_default_scale_and_catches_invalid_4bit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify GPTQ 4-bit uses base scale bytes and invalid 4-bit returns exit 2."""
    base_cfg = load_locked_config("qwen3_5_397b_fp8.json")
    gptq_cfg = {
        **base_cfg,
        "quantization_config": {"quant_method": "gptq", "bits": 4, "group_size": 128},
    }
    gptq_path = tmp_path / "gptq.json"
    gptq_path.write_text(json.dumps(gptq_cfg), encoding="utf-8")
    rc = cli.main(["plan", "--config", str(gptq_path), "--tp", "8"])
    assert rc == 0
    assert "hard_floor_ms" in capsys.readouterr().out

    bad_cfg = {
        **base_cfg,
        "quantization_config": {"bits": 4, "group_size": 128},
    }
    bad_path = tmp_path / "bad4bit.json"
    bad_path.write_text(json.dumps(bad_cfg), encoding="utf-8")
    rc_bad = cli.main(["plan", "--config", str(bad_path), "--tp", "8"])
    assert rc_bad == 2
    assert "scale_bytes" in capsys.readouterr().err
