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
"""Unit tests for the Qwen3.5 serving acceptance benchmark harness."""

from __future__ import annotations

import ast
import importlib.util
import json
import pathlib
import re
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
HARNESS_PATH = (
    REPO_ROOT
    / "src"
    / "vllm_torchtpu"
    / "models"
    / "qwen3_5"
    / "megakernel"
    / "serving"
    / "acceptance_benchmark.py"
)


def _load_harness():
    spec = importlib.util.spec_from_file_location("acceptance_benchmark", HARNESS_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["acceptance_benchmark"] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_sweep(
    mod,
    *,
    sweep_id: str,
    concurrency: str,
    pair_index: int,
    arm: str,
    tps: float,
    p50: float,
    p90: float,
    p99: float,
    mode: str = "staggered",
    mk_steps: int = 50000,
    decline_codes: tuple[str, ...] = (),
    dropped: int = 0,
    acc_len: float = 3.3902,
    duration_s: float = 900.0,
):
    mid_steps = 4200 if mode == "bunched" else 1800
    b4_steps = max(0, mk_steps - mid_steps)
    total_steps = mk_steps + 1000 if mk_steps > 0 else 1000
    counts = (
        {"mk_t4_b4": b4_steps, "mk_t4_b32": mid_steps, "stock_prefill": 1000}
        if mk_steps > 0
        else {"stock_prefill": 1000}
    )
    mid_pct, computed_mode = mod.classify_c16_packing_mode(counts)
    sm = mod.StepMixSummary(
        mk_steps=mk_steps,
        total_steps=total_steps,
        mk_pct=round(100.0 * mk_steps / total_steps, 4),
        bucket_counts=counts,
        bucket_shares_pct={
            k: round(100.0 * v / total_steps, 4) for k, v in counts.items()
        },
        decline_codes=decline_codes,
        mid_verify_share_pct=mid_pct,
        packing_mode=computed_mode,
    )
    rec = mod.SweepRecord(
        sweep_id=sweep_id,
        concurrency=concurrency,
        pair_index=pair_index,
        arm=arm,
        arm_label=arm,
        numerics="approx",
        expect_megakernel=True,
        duration_s=duration_s,
        output_tok_per_s=tps,
        itl_p50_ms=p50,
        itl_p90_ms=p90,
        itl_p99_ms=p99,
        itl_mean_ms=p50 * 1.15,
        records_error_dropped=dropped,
        server_healthy_after=True,
        spec_mean_acceptance_length=acc_len,
        prefix_cache_hit_rate=0.82,
        step_mix=sm,
    )
    return mod.validate_sweep_record(rec)


def test_arm_commands_and_env_match_candidate30_specs(monkeypatch):
    mod = _load_harness()
    monkeypatch.delenv(mod.REF_IMAGE_ENV, raising=False)

    # Image reference mode requires explicit image URI (never hardcoded in src/).
    with pytest.raises(ValueError, match=mod.REF_IMAGE_ENV):
        mod.build_arm_server_command(
            "reference", mod.CONCURRENCY_SPECS["c8"], reference_mode="image"
        )

    cmd_a_c16 = mod.build_arm_server_command(
        "reference",
        mod.CONCURRENCY_SPECS["c16"],
        reference_mode="image",
        reference_image="registry.example/vllm-tpu:candidate30",
    )
    assert cmd_a_c16[:3] == ["docker", "run", "--rm"]
    assert "registry.example/vllm-tpu:candidate30" in cmd_a_c16
    assert "--block-size" in cmd_a_c16
    assert cmd_a_c16[cmd_a_c16.index("--block-size") + 1] == "256"
    assert cmd_a_c16[cmd_a_c16.index("--max-num-seqs") + 1] == "16"
    assert "--enable-prefix-caching" in cmd_a_c16
    assert "--async-scheduling" in cmd_a_c16
    assert "--mamba-ssm-cache-dtype" not in cmd_a_c16
    kv_cfg_c16 = json.loads(cmd_a_c16[cmd_a_c16.index("--kv-transfer-config") + 1])
    assert kv_cfg_c16["kv_connector_extra_config"]["cpu_bytes_to_use"] == 400 * 1024**3

    cmd_b_c32 = mod.build_arm_server_command(
        "candidate",
        mod.CONCURRENCY_SPECS["c32"],
    )
    assert cmd_b_c32[:2] == ["vllm", "serve"]
    assert cmd_b_c32[cmd_b_c32.index("--mamba-ssm-cache-dtype") + 1] == "bfloat16"
    assert cmd_b_c32[cmd_b_c32.index("--max-num-seqs") + 1] == "32"
    spec_cfg = json.loads(cmd_b_c32[cmd_b_c32.index("--speculative-config") + 1])
    assert spec_cfg["method"] == "mtp"
    assert spec_cfg["num_speculative_tokens"] == 3
    assert spec_cfg["synthetic_acceptance_length"] == pytest.approx(3.39)
    add_cfg = json.loads(cmd_b_c32[cmd_b_c32.index("--additional-config") + 1])
    assert add_cfg == {
        "megakernel": {
            "options": {"mode": "ep", "profile": "qwen35-candidate30-fused"},
            "plugin": "vllm_torchtpu.models.qwen3_5.megakernel",
            "stats_period_s": 60,
        }
    }
    kv_cfg_c32 = json.loads(cmd_b_c32[cmd_b_c32.index("--kv-transfer-config") + 1])
    assert kv_cfg_c32["kv_connector_extra_config"]["cpu_bytes_to_use"] == 500 * 1024**3

    # Stock reference mode disables megakernel and matches bfloat16 SSM flag.
    cmd_stock_c8 = mod.build_arm_server_command(
        "reference",
        mod.CONCURRENCY_SPECS["c8"],
        reference_mode="stock",
    )
    assert "--kv-transfer-config" not in cmd_stock_c8
    assert cmd_stock_c8[cmd_stock_c8.index("--mamba-ssm-cache-dtype") + 1] == "bfloat16"
    stock_add = json.loads(cmd_stock_c8[cmd_stock_c8.index("--additional-config") + 1])
    assert stock_add == {"megakernel": {"enabled": False}}


def test_parse_mk_stats_log_sums_windows_and_catches_declines():
    mod = _load_harness()
    log = "\n".join(
        [
            "INFO 10-29 01:00:00 qwen35 megakernel step mix over 60s: "
            "1900/2000 MK (95.0%) "
            "{'mk_t4_b4': 1700, 'mk_t4_b32': 120, "
            "'mk_t4_b40': 80, 'stock_prefill': 100}",
            "INFO 10-29 01:01:00 qwen35 megakernel step mix over 60s: "
            "1950/2000 MK (97.5%) "
            "{'mk_t4_b4': 1850, 'mk_t4_b32': 60, "
            "'mk_t4_b48': 40, 'stock_prefill': 50}",
        ]
    )
    sm = mod.parse_mk_stats_log(log)
    assert sm.mk_steps == 3850
    assert sm.total_steps == 4000
    assert sm.mk_pct == pytest.approx(96.25)
    # mid_steps = 120 + 80 + 60 + 40 = 300 / 4000 = 7.5% >= 6.0% -> bunched
    assert sm.mid_verify_share_pct == pytest.approx(7.5)
    assert sm.packing_mode == "bunched"
    assert sm.decline_codes == ()

    decl_log = (
        "WARNING 10-29 01:00:00 qwen35 megakernel declined "
        "(conv_state_dtype_unsupported): "
        "conv state dtype torch.float32 is unsupported\n"
        "INFO 10-29 01:01:00 qwen35 megakernel step mix over 60s: 0/500 MK (0.0%) "
        "{'stock_conv_state_dtype_unsupported': 500}\n"
    )
    sm_decl = mod.parse_mk_stats_log(decl_log)
    assert sm_decl.mk_steps == 0
    assert sm_decl.decline_codes == ("conv_state_dtype_unsupported",)


def test_validity_gates_reject_declines_drops_and_acceptance_drift():
    mod = _load_harness()
    bad_decl = _make_sweep(
        mod,
        sweep_id="c8_p1_candidate",
        concurrency="c8",
        pair_index=1,
        arm="candidate",
        tps=570.0,
        p50=1.50,
        p90=2.50,
        p99=6.00,
        mk_steps=0,
        decline_codes=("conv_state_dtype_unsupported",),
    )
    assert not bad_decl.valid
    assert (
        "megakernel_declined:conv_state_dtype_unsupported" in bad_decl.invalid_reasons
    )
    assert "mk_steps_zero" in bad_decl.invalid_reasons

    bad_drop = _make_sweep(
        mod,
        sweep_id="c8_p1_reference",
        concurrency="c8",
        pair_index=1,
        arm="reference",
        tps=570.0,
        p50=1.50,
        p90=2.50,
        p99=6.00,
        dropped=2,
        acc_len=3.20,
        duration_s=500.0,
    )
    assert not bad_drop.valid
    assert "records_error_dropped=2" in bad_drop.invalid_reasons
    assert any(
        r.startswith("spec_acceptance_out_of_band") for r in bad_drop.invalid_reasons
    )
    assert any(r.startswith("duration_out_of_window") for r in bad_drop.invalid_reasons)


def test_c16_mode_stratification_and_strict_regression_rule():
    mod = _load_harness()

    # Noise floor is NOT an allowed slowdown: a consistent -0.9% throughput drop
    # with tight variance (CI excludes 0) must flag SIGNIFICANT_REGRESSION -> FAIL.
    tight_reg = mod.paired_student_t(
        [1000.0, 1000.0, 1000.0],
        [991.0, 991.2, 990.8],
        higher_is_better=True,
        noise_floor_pct=1.5,
    )
    assert tight_reg["status"] == "SIGNIFICANT_REGRESSION"

    # Build a full c8/c16/c32 passing suite where c16 has 2 staggered matched pairs
    # and 1 bunched matched pair.
    recs = []
    for p in (1, 2, 3):
        recs.append(
            _make_sweep(
                mod,
                sweep_id=f"c8_p{p}_reference",
                concurrency="c8",
                pair_index=p,
                arm="reference",
                tps=568.0 + p,
                p50=1.51,
                p90=2.48,
                p99=6.00,
            )
        )
        recs.append(
            _make_sweep(
                mod,
                sweep_id=f"c8_p{p}_candidate",
                concurrency="c8",
                pair_index=p,
                arm="candidate",
                tps=569.0 + p,
                p50=1.50,
                p90=2.47,
                p99=5.98,
            )
        )
        c16_mode = "bunched" if p == 2 else "staggered"
        base_tps = 770.0 if c16_mode == "bunched" else 815.0
        base_p50 = 2.52 if c16_mode == "bunched" else 2.15
        recs.append(
            _make_sweep(
                mod,
                sweep_id=f"c16_p{p}_reference",
                concurrency="c16",
                pair_index=p,
                arm="reference",
                tps=base_tps,
                p50=base_p50,
                p90=5.90,
                p99=17.2,
                mode=c16_mode,
            )
        )
        recs.append(
            _make_sweep(
                mod,
                sweep_id=f"c16_p{p}_candidate",
                concurrency="c16",
                pair_index=p,
                arm="candidate",
                tps=base_tps + 1.0,
                p50=base_p50 - 0.01,
                p90=5.86,
                p99=17.0,
                mode=c16_mode,
            )
        )
        recs.append(
            _make_sweep(
                mod,
                sweep_id=f"c32_p{p}_reference",
                concurrency="c32",
                pair_index=p,
                arm="reference",
                tps=1310.0,
                p50=9.55,
                p90=17.80,
                p99=30.10,
            )
        )
        recs.append(
            _make_sweep(
                mod,
                sweep_id=f"c32_p{p}_candidate",
                concurrency="c32",
                pair_index=p,
                arm="candidate",
                tps=1311.2,
                p50=9.52,
                p90=17.65,
                p99=30.00,
            )
        )

    suite = mod.evaluate_acceptance_suite(recs)
    assert suite["overall_verdict"] == "PASS"
    c16_rep = suite["concurrencies"]["c16"]
    assert c16_rep["c16_stratification"]["mode_matched_pairs"] == 3
    assert set(c16_rep["c16_stratification"]["by_mode"]) == {"staggered", "bunched"}
    md = mod.format_acceptance_markdown(suite)
    assert "`c16` (`staggered`)" in md
    assert "`c16` (`bunched`)" in md
    assert "`approx`" in md


def test_prometheus_metrics_parser():
    mod = _load_harness()
    before = mod.parse_prometheus_metrics(
        'vllm:spec_decode_num_accepted_tokens_total{model_name="qwen"} 100\n'
        'vllm:spec_decode_num_drafts_total{model_name="qwen"} 50\n'
        'vllm:prefix_cache_hits_total{model_name="qwen"} 10\n'
        'vllm:prefix_cache_queries_total{model_name="qwen"} 20\n'
    )
    after = mod.parse_prometheus_metrics(
        'vllm:spec_decode_num_accepted_tokens_total{model_name="qwen"} 2490\n'
        'vllm:spec_decode_num_drafts_total{model_name="qwen"} 1050\n'
        'vllm:prefix_cache_hits_total{model_name="qwen"} 810\n'
        'vllm:prefix_cache_queries_total{model_name="qwen"} 1020\n'
    )
    deltas = mod.compute_prometheus_deltas(before, after)
    assert deltas["spec_mean_acceptance_length"] == pytest.approx(3.39)
    assert deltas["prefix_cache_hit_rate"] == pytest.approx(0.80)


def test_gsm8k_separate_pass_real_rejection_sampling_and_spread(capsys):
    mod = _load_harness()

    cmd_a_gsm = mod.build_gsm8k_server_command(
        "reference",
        mod.CONCURRENCY_SPECS["c32"],
        reference_mode="image",
        reference_image="registry.example/vllm-tpu:candidate30",
    )
    spec_a = json.loads(cmd_a_gsm[cmd_a_gsm.index("--speculative-config") + 1])
    assert spec_a == {"method": "mtp", "num_speculative_tokens": 3}
    assert "rejection_sample_method" not in spec_a
    assert "synthetic_acceptance_length" not in spec_a

    cmd_b_gsm = mod.build_gsm8k_server_command(
        "candidate",
        mod.CONCURRENCY_SPECS["c32"],
    )
    spec_b = json.loads(cmd_b_gsm[cmd_b_gsm.index("--speculative-config") + 1])
    assert spec_b == {"method": "mtp", "num_speculative_tokens": 3}
    add_b = json.loads(cmd_b_gsm[cmd_b_gsm.index("--additional-config") + 1])
    assert add_b["megakernel"]["options"] == {
        "mode": "ep",
        "profile": "qwen35-candidate30-fused",
    }

    # print-commands --pass-mode gsm8k emits real rejection sampling + greedy params.
    rc = mod.main(
        [
            "print-commands",
            "--pass-mode",
            "gsm8k",
            "--reference-mode",
            "stock",
        ]
    )
    assert rc == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["pass_mode"] == "gsm8k"
    assert printed["rejection_sample_mode"] == "real"
    assert printed["client_sampling_params"] == {"temperature": 0.0}
    for arm_key in ("reference", "candidate"):
        arm_cmd = printed[arm_key]["cmd"]
        arm_spec = json.loads(arm_cmd[arm_cmd.index("--speculative-config") + 1])
        assert arm_spec == {"method": "mtp", "num_speculative_tokens": 3}

    # Synthetic rejection sampling or non-greedy temperature must be rejected.
    with pytest.raises(ValueError, match="rejection_sample_mode='real'"):
        mod.evaluate_gsm8k_gate(0.9242, 0.9242, rejection_sample_mode="synthetic")
    with pytest.raises(ValueError, match="temperature=0.0"):
        mod.evaluate_gsm8k_gate(0.9242, 0.9242, temperature=0.6)

    gsm = mod.evaluate_gsm8k_gate(
        0.9234,
        0.9242,
        n_questions=1319,
        ref_only_correct=14,
        cand_only_correct=13,
        reference_acceptance_lengths=[3.182, 3.190],
        candidate_acceptance_lengths=[3.176, 3.188],
    )
    assert gsm["passed"] is True
    assert gsm["rejection_sample_mode"] == "real"
    assert gsm["temperature"] == 0.0
    assert gsm["acceptance_length_spread"] == pytest.approx(0.014)
    assert gsm["arm_acceptance_delta"] == pytest.approx(0.004)

    suite = mod.evaluate_acceptance_suite(
        [], required_concurrencies=(), gsm8k_result=gsm
    )
    md = mod.format_acceptance_markdown(suite)
    assert "Separate Real Rejection Sampling Pass, Greedy" in md
    assert "spread across runs: `0.0140`" in md


def test_acceptance_script_stdlib_and_template_hygiene():
    assert HARNESS_PATH.is_file()
    # Ensure no unintended __init__.py was added before template package creation.
    assert not (HARNESS_PATH.parent / "__init__.py").exists()

    src = HARNESS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(HARNESS_PATH))
    stdlib = sys.stdlib_module_names

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                assert root in stdlib, (
                    f"Non-stdlib import {alias.name!r} in serving harness"
                )
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0 and node.module is not None
            root = node.module.split(".")[0]
            assert root in stdlib, (
                f"Non-stdlib from-import {node.module!r} in serving harness"
            )

    forbidden_patterns = [
        r"\b(?:C2|C3|B1|B6|B7|B8|S1|S2|S3|S4)\b",
        r"\bStage\s*[12][ab]?\b",
        "".join(("nan", "wei")),
        "".join(("cao", "jx")),
        "".join(("venv", "312")),
        "_".join(("tpu", "aot", "pkgs")),
        "_".join(("MK", "FROZEN")),
        "_".join(("frozen", "qwen35")),
        "_".join(("CANDIDATE30", "BASELINE")),
        "-".join(("hand", "written")),
    ]
    for pat in forbidden_patterns:
        assert re.search(pat, src) is None, (
            f"Forbidden pattern {pat!r} found in {HARNESS_PATH}"
        )

    # Verify fresh-interpreter import succeeds with torch, jax, and vllm blocked,
    # both in-tree and when the model folder is copied under a custom package name.
    exp_pkg = "vllm_torchtpu.models.qwen3_5.megakernel"
    code = (
        "import importlib.util, pathlib, shutil, sys, tempfile; "
        "sys.modules.update({'torch': None, 'jax': None, 'vllm': None}); "
        f"spec = importlib.util.spec_from_file_location('ab', {str(HARNESS_PATH)!r}); "
        "mod = importlib.util.module_from_spec(spec); "
        "sys.modules['ab'] = mod; "
        "spec.loader.exec_module(mod); "
        f"assert mod.default_plugin_module() == {exp_pkg!r}; "
        "td = pathlib.Path(tempfile.mkdtemp()); "
        f"dst = td / 'copied_model_pkg'; "
        f"shutil.copytree(pathlib.Path({str(HARNESS_PATH.parent.parent)!r}), dst); "
        "cp_path = dst / 'serving' / 'acceptance_benchmark.py'; "
        "spec2 = importlib.util.spec_from_file_location('ab_cp', cp_path); "
        "mod2 = importlib.util.module_from_spec(spec2); "
        "sys.modules['ab_cp'] = mod2; "
        "spec2.loader.exec_module(mod2); "
        "assert mod2.default_plugin_module() == 'copied_model_pkg'; "
        "shutil.rmtree(td)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
