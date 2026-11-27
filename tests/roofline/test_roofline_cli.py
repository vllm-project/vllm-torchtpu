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
from collections.abc import Callable, Mapping, Sequence
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
        act_dtype=None,
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
        act_dtype=act_flag,
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


def test_cli_help_clarifies_physics_pruning_vs_uncalibrated_rankings(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify CLI help distinguishes physical pruning from uncalibrated rankings."""
    for argv, expected_substrings in (
        (["--help"], ("physical lower bounds", "uncalibrated serial-phase")),
        (
            ["solve", "--help"],
            ("Prune parallel layouts via physical HBM bounds", "uncalibrated"),
        ),
        (["ceiling", "--help"], ("uncalibrated", "physics candidate")),
        (["plan", "--help"], ("hard physical step floors", "uncalibrated")),
    ):
        with pytest.raises(SystemExit) as exc_info:
            cli.main(argv)
        assert exc_info.value.code == 0
        help_text = capsys.readouterr().out
        assert "winning" not in help_text.lower()
        for expected in expected_substrings:
            assert expected in help_text


@pytest.mark.parametrize("mode", ("mixed", "disagg", "auto"))
def test_cli_solve_and_ceiling_support_chip_spec_and_dtype_overrides(
    mode: str,
    tmp_path: Path,
    locked_config_path: Callable[[str], Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify solve and ceiling honor --chip-spec, dtype flags, and 11-field parity."""
    qwen35_path = locked_config_path("qwen3_5_397b_fp8.json")
    spec_path = tmp_path / "custom_chip.json"
    spec_path.write_text(
        json.dumps(
            {
                "schema_version": "vllm_torchtpu.roofline.chip_spec/v1",
                "provenance": "projected",
                "base": "tpu7x",
                "name": "tpu7x_override_test",
            }
        ),
        encoding="utf-8",
    )
    common_flags = [
        "--config",
        str(qwen35_path),
        "--chip-spec",
        str(spec_path),
        "--kv-dtype",
        "fp8",
        "--act",
        "a16",
        "--isl",
        "1024",
        "--osl",
        "128",
        "--prefix-hit-rate",
        "0.0",
        "--accept-len",
        "1.0",
        "--think-time-s",
        "0.0",
        "--concurrency",
        "4",
        "--mode",
        mode,
        "--max-chips",
        "8",
        "--json",
    ]
    solve_argv = ["solve", *common_flags]
    if mode in ("disagg", "auto"):
        solve_argv.append("--pairs")
    rc_solve = cli.main(solve_argv)
    assert rc_solve == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["chip"] == "tpu7x_override_test"
    assert payload["candidates"]

    rc_ceil = cli.main(["ceiling", *common_flags])
    assert rc_ceil == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == len(payload["candidates"])
    for row, cand in zip(rows, payload["candidates"], strict=True):
        assert row == {
            "est_interactivity_tok_s": cand["est_interactivity_tok_s"],
            "token_throughput_per_chip": cand["token_throughput_per_chip"],
            "output_throughput_per_chip": cand["output_throughput_per_chip"],
            "est_ttft_ms": cand["est_ttft_ms"],
            "est_tpot_ms": cand["est_tpot_ms"],
            "bound_tpot_ms": cand["bound_tpot_ms"],
            "concurrency": cand["concurrency"],
            "total_chips": cand["total_chips"],
            "mode": cand["mode"],
            "binding_unit": cand["binding"],
            "min_decode_seqs": cand["min_decode_seqs"],
            "floor_throughput_per_chip": cand["floor_throughput_per_chip"],
            "slo_met": cand["slo_met"],
            "is_pareto": cand["is_pareto"],
            "decode_topology": cand["decode_pool"]["topology"],
            "decode_parallelism": row["decode_parallelism"],
            "prefill_topology": (
                cand["prefill_pool"]["topology"]
                if cand["prefill_pool"] is not None
                else None
            ),
            "prefill_parallelism": row["prefill_parallelism"],
        }
        assert row["decode_parallelism"]
        if cand["prefill_pool"] is None:
            assert row["prefill_parallelism"] is None
        else:
            assert row["prefill_parallelism"]


def test_cli_moe_activation_and_ruled_out_reference_text(
    locked_config_path: Callable[[str], Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify --moe-activation valid/invalid flags and ruled-out reference summary."""
    cfg_path = locked_config_path("qwen3_5_397b_fp8.json")

    rc_pig = cli.main(
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
            "--moe-activation",
            "pigeonhole",
            "--json",
        ]
    )
    assert rc_pig == 0
    pig_payload = json.loads(capsys.readouterr().out)
    supplied = {
        entry["name"]: entry["value"]
        for entry in pig_payload["provenance"]["supplied_values"]
    }
    assert supplied["moe_activation"] == "pigeonhole"

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
                "--moe-activation",
                "bogus",
            ]
        )
    assert exc_info.value.code == 2
    capsys.readouterr()

    rc_ruled = cli.main(
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
            "--tpot-slo-ms",
            "0.001",
            "--reference-layout",
            "TP8 x EP8",
        ]
    )
    assert rc_ruled == 0
    text_out = capsys.readouterr().out
    assert "enumerated, ruled out by SLO" in text_out


def test_cli_solve_mixed_instances_text_and_reference_top_k(
    locked_config_path: Callable[[str], Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify mixed text prints decode instances and reference honors --top-k."""
    qwen35_path = locked_config_path("qwen3_5_397b_fp8.json")
    base_argv = [
        "solve",
        "--config",
        str(qwen35_path),
        "--isl",
        "4096",
        "--osl",
        "512",
        "--prefix-hit-rate",
        "0.5",
        "--accept-len",
        "1.0",
        "--think-time-s",
        "0.0",
        "--concurrency",
        "32",
        "--mode",
        "mixed",
        "--decode-topology",
        "2x2x1",
        "--max-chips",
        "8",
    ]
    assert cli.main([*base_argv, "--top-k", "20"]) == 0
    table_text = capsys.readouterr().out
    assert "2x 2x2x1 " in table_text

    assert cli.main([*base_argv, "--top-k", "10", "--json"]) == 0
    full_payload = json.loads(capsys.readouterr().out)
    top_candidates = full_payload["candidates"]
    assert len(top_candidates) >= 2
    first_pool = top_candidates[0]["decode_pool"]
    first_sig = (
        first_pool["instances"],
        first_pool["tp"],
        first_pool["dp"],
        first_pool["ep"],
        first_pool["pp"],
        first_pool["pcp"],
    )
    second_pool = next(
        cand["decode_pool"]
        for cand in top_candidates[1:]
        if (
            cand["decode_pool"]["instances"],
            cand["decode_pool"]["tp"],
            cand["decode_pool"]["dp"],
            cand["decode_pool"]["ep"],
            cand["decode_pool"]["pp"],
            cand["decode_pool"]["pcp"],
        )
        != first_sig
    )
    prefix = f"{second_pool['instances']}x " if second_pool["instances"] > 1 else ""
    parts = [
        f"{name}{second_pool[key]}"
        for name, key in (
            ("TP", "tp"),
            ("DP", "dp"),
            ("EP", "ep"),
            ("PP", "pp"),
            ("PCP", "pcp"),
        )
        if second_pool[key] > 1
    ]
    ref_spec = f"{prefix}{' x '.join(parts)}"

    assert (
        cli.main([*base_argv, "--top-k", "1", "--reference-layout", ref_spec, "--json"])
        == 0
    )
    narrow_payload = json.loads(capsys.readouterr().out)
    assert narrow_payload["reference"]["found"] is True
    assert narrow_payload["reference"]["family_in_ordered_top_k"] is False

    assert (
        cli.main(
            [*base_argv, "--top-k", "10", "--reference-layout", ref_spec, "--json"]
        )
        == 0
    )
    wide_payload = json.loads(capsys.readouterr().out)
    assert wide_payload["reference"]["found"] is True
    assert wide_payload["reference"]["family_in_ordered_top_k"] is True

    conc1_argv = [
        "solve",
        "--config",
        str(qwen35_path),
        "--isl",
        "4096",
        "--osl",
        "512",
        "--prefix-hit-rate",
        "0.5",
        "--accept-len",
        "1.0",
        "--think-time-s",
        "0.0",
        "--concurrency",
        "1",
        "--mode",
        "mixed",
        "--max-chips",
        "8",
        "--reference-layout",
        "2x TP8 x EP8",
        "--json",
    ]
    assert cli.main(conc1_argv) == 0
    conc1_payload = json.loads(capsys.readouterr().out)
    assert conc1_payload["reference"]["found"] is False
    assert "exceeds max concurrency 1" in conc1_payload["reference"]["rejected_by"]


def test_every_cli_flag_has_help() -> None:
    """Verify every registered CLI flag carries non-empty help text."""
    parser = cli.build_parser()
    options = _iter_subcommand_options(parser)
    assert options
    subcommands = {cmd for cmd, _, _ in options if cmd != "root"}
    assert subcommands == {"solve", "ceiling", "plan", "explain", "inspect"}
    for cmd, opt_strings, help_text in options:
        assert help_text, f"{cmd} {opt_strings} missing help text"


def test_cli_assumptions_and_solve_ascii_ordered_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify --assumptions text output and solve aligned ASCII ordered table."""
    rc_assump = cli.main(["--assumptions"])
    assert rc_assump == 0
    assump_out = capsys.readouterr().out
    assert "Constants catalog for chip=tpu7x" in assump_out
    assert "Solver assumptions:" in assump_out
    assert "hbm_bytes" in assump_out
    assert "bf16_flops" in assump_out
    assert "hbm_bw" in assump_out

    cfg_path = Path(__file__).resolve().parent / "fixtures" / "qwen3_5_397b_fp8.json"
    rc_solve = cli.main(
        [
            "solve",
            "--config",
            str(cfg_path),
            "--mode",
            "mixed",
            "--max-chips",
            "4",
            "--isl",
            "4096",
            "--osl",
            "512",
            "--prefix-hit-rate",
            "0.0",
            "--accept-len",
            "1.0",
            "--concurrency",
            "16",
            "--top-k",
            "3",
        ]
    )
    assert rc_solve == 0
    out = capsys.readouterr().out
    assert "bound_tpot_ms" in out
    assert "# Ordered Candidates by Concurrency" in out
    assert "# c=16:" in out
    assert "#   #1 " in out
    assert "#   #3 " in out
    assert "#   #4 " not in out
    assert "skyline" not in out.lower()

    # Ceiling accepts --sort-by, --pairs, and --top-k cleanly.
    rc_ceil = cli.main(
        [
            "ceiling",
            "--config",
            str(cfg_path),
            "--mode",
            "disagg",
            "--max-chips",
            "8",
            "--isl",
            "4096",
            "--osl",
            "256",
            "--prefix-hit-rate",
            "0.0",
            "--accept-len",
            "1.0",
            "--concurrency",
            "8",
            "--sort-by",
            "interactivity",
            "--pairs",
            "--top-k",
            "2",
            "--json",
        ]
    )
    assert rc_ceil == 0
    ceil_rows = json.loads(capsys.readouterr().out)
    assert 1 <= len(ceil_rows) <= 2

    # Pool-only disagg text honors --top-k and prints --reference-layout summary.
    rc_pool = cli.main(
        [
            "solve",
            "--config",
            str(cfg_path),
            "--mode",
            "disagg",
            "--max-chips",
            "8",
            "--isl",
            "4096",
            "--osl",
            "256",
            "--prefix-hit-rate",
            "0.0",
            "--accept-len",
            "1.0",
            "--concurrency",
            "8",
            "--top-k",
            "2",
            "--reference-layout",
            "P: PP8 / D: TP8 x EP8",
        ]
    )
    assert rc_pool == 0
    pool_out = capsys.readouterr().out
    prefill_rows = [ln for ln in pool_out.splitlines() if ln.startswith("prefill ")]
    decode_rows = [ln for ln in pool_out.splitlines() if ln.startswith("decode  ")]
    assert 1 <= len(prefill_rows) <= 2
    assert 1 <= len(decode_rows) <= 2
    assert "reference P: PP8 / D: TP8 x EP8: not found (" in pool_out


def test_cli_default_ascii_rank1_matches_json_top_k_and_preserves_a4(
    locked_config_path: Callable[[str], Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify omitted --top-k ASCII rank 1 matches JSON top_k[0] and --act a4 works."""
    qwen35_path = locked_config_path("qwen3_5_397b_fp8.json")
    base_argv = [
        "solve",
        "--config",
        str(qwen35_path),
        "--mode",
        "mixed",
        "--max-chips",
        "4",
        "--isl",
        "4096",
        "--osl",
        "512",
        "--prefix-hit-rate",
        "0.0",
        "--accept-len",
        "1.0",
        "--think-time-s",
        "0.0",
        "--concurrency",
        "1,64",
        "--act-dtype",
        "a4",
    ]
    assert cli.main([*base_argv, "--json"]) == 0
    json_payload = json.loads(capsys.readouterr().out)
    assert json_payload["dtypes"]["act"] == "a4"
    rank1_json = json_payload["top_k"][0]["point"]

    assert cli.main(base_argv) == 0
    ascii_out = capsys.readouterr().out
    assert "act=a4" in ascii_out
    assert "state=float32 (4B, config)" in ascii_out
    rank1_line = next(
        line for line in ascii_out.splitlines() if line.lstrip().startswith("1  ")
    )
    assert str(rank1_json["concurrency"]) in rank1_line
    assert f"{rank1_json['token_throughput_per_chip']:.1f}" in rank1_line

    assert cli.main([*base_argv, "--state-dtype", "bf16"]) == 0
    bf16_out = capsys.readouterr().out
    assert "state=bf16 (2B, flag)" in bf16_out
