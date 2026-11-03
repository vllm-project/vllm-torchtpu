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
"""Unit tests for `vllm_torchtpu.megakernel.advisor`."""

from __future__ import annotations

import dataclasses
import importlib
import json
import sys
from pathlib import Path

import pytest
from jax.experimental.pallas import tpu as pltpu

from tests.megakernel.megakernel_fakes import TINY_TILES, make_fake_model_spec
from vllm_torchtpu import megakernel as mk
from vllm_torchtpu.megakernel import advisor
from vllm_torchtpu.megakernel.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.exec_spec import ExecSpec, Tiles

# HuggingFace Qwen3.5, Mixtral-8x7B, DeepSeek-V3, and Kimi-K3-Pro configs.
_QWEN35_08B_CFG = {
    "num_hidden_layers": 24,
    "hidden_size": 1024,
    "intermediate_size": 3584,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "full_attention_interval": 4,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 16,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "top_k": 50,
    "quantization_config": {"quant_method": "fp8"},
}
_QWEN35_397B_CFG = {
    "num_hidden_layers": 60,
    "hidden_size": 4096,
    "intermediate_size": 1024,
    "num_attention_heads": 32,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "full_attention_interval": 4,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 64,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "moe_intermediate_size": 1024,
    "shared_expert_intermediate_size": 1024,
    "quantization_config": {"quant_method": "fp8"},
}
_MIXTRAL_8X7B_CFG = {
    "num_hidden_layers": 32,
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "num_local_experts": 8,
    "num_experts_per_tok": 2,
    "torch_dtype": "bfloat16",
}
_DEEPSEEK_V3_CFG = {
    "num_hidden_layers": 61,
    "first_k_dense_replace": 3,
    "hidden_size": 7168,
    "intermediate_size": 18432,
    "moe_intermediate_size": 2048,
    "num_attention_heads": 128,
    "num_key_value_heads": 128,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "num_experts_per_tok": 8,
    "quantization_config": {"quant_method": "fp8"},
}
_KIMI_K3_PRO_CFG = {
    "num_hidden_layers": 93,
    "first_k_dense_replace": 1,
    "hidden_size": 7168,
    "intermediate_size": 33792,
    "moe_intermediate_size": 3072,
    "shared_expert_intermediate_size": 6144,
    "num_attention_heads": 96,
    "num_key_value_heads": 96,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "num_experts": 896,
    "num_experts_per_tok": 16,
    "linear_attn_config": {
        "kda_layers": list(range(69)),
        "num_heads": 96,
        "head_k_dim": 128,
        "head_v_dim": 128,
        "short_conv_kernel_size": 4,
    },
    "quantization_config": {"quant_method": "fp8", "expert_dtype": "mxfp4"},
}


def test_package_reexport_without_prior_submodule_import():
    """Verify `vllm_torchtpu.megakernel` re-exports `advisor` in `__all__`."""
    saved = sys.modules.pop("vllm_torchtpu.megakernel.advisor", None)
    try:
        if "advisor" in mk.__dict__:
            delattr(mk, "advisor")
        importlib.reload(mk)
        assert "advisor" in mk.__all__
        assert mk.advisor.__name__ == "vllm_torchtpu.megakernel.advisor"
    finally:
        if saved is not None:
            sys.modules["vllm_torchtpu.megakernel.advisor"] = saved
            mk.advisor = saved


@pytest.mark.parametrize(
    ("slug", "repo_id", "layers", "attn_l", "lin_l", "moe_l", "wd", "ed"),
    [
        ("qwen3.5-0.8b", "Qwen/Qwen3.5-0.8B", 24, 6, 18, 0, "fp8", "fp8"),
        ("qwen3.5-397b", "Qwen/Qwen3.5-397B-A17B-FP8", 60, 15, 45, 60, "fp8", "fp8"),
        ("qwen3.8-2.4t", "Qwen/Qwen3.8-2.4T", 92, 23, 69, 92, "fp8", "nvfp4"),
        ("kimi-k3", "moonshotai/Kimi-K3", 93, 24, 69, 92, "fp8", "mxfp4"),
        ("kimi-k3.1-pro", "moonshotai/Kimi-K3.1-Pro", 93, 24, 69, 92, "fp8", "mxfp4"),
        ("kimi-k3-pro", "moonshotai/Kimi-K3-Pro", 93, 24, 69, 92, "fp8", "mxfp4"),
        ("deepseek-v3", "deepseek-ai/DeepSeek-V3", 61, 61, 0, 58, "fp8", "fp8"),
        ("glm-5.2", "zai-org/GLM-5.2-FP8", 78, 78, 0, 75, "fp8", "fp8"),
        (
            "mistral-large",
            "mistralai/Mistral-Large-Instruct-2411",
            88,
            88,
            0,
            0,
            "fp8",
            "fp8",
        ),
        (
            "mixtral-8x22b",
            "mistralai/Mixtral-8x22B-v0.1",
            56,
            56,
            0,
            56,
            "bf16",
            "bf16",
        ),
        (
            "llama-3.1-405b",
            "meta-llama/Llama-3.1-405B-Instruct-FP8",
            126,
            126,
            0,
            0,
            "fp8",
            "fp8",
        ),
        (
            "llama-4-maverick",
            "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            48,
            48,
            0,
            48,
            "fp8",
            "fp8",
        ),
        ("gpt-oss-120b", "openai/gpt-oss-120b", 36, 36, 0, 36, "fp8", "mxfp4"),
    ],
)
def test_builtin_model_presets_and_hf_aliases(
    slug: str,
    repo_id: str,
    layers: int,
    attn_l: int,
    lin_l: int,
    moe_l: int,
    wd: str,
    ed: str,
) -> None:
    """Verify all 12 Roofline RFC, Kimi K3/Pro, and Mistral/Mixtral presets."""
    prof = advisor.resolve_model_profile(slug)
    assert advisor.resolve_model_profile(repo_id) == prof
    assert advisor.resolve_model_profile(prof) == prof
    assert (
        prof.num_layers,
        prof.num_attention_layers,
        prof.num_linear_layers,
        prof.num_moe_layers,
        prof.weight_dtype,
        prof.expert_dtype,
    ) == (layers, attn_l, lin_l, moe_l, wd, ed)
    rep = advisor.advise(prof, advisor.WorkloadSpec(tp=8, batch=16), "tpu7x")
    assert rep.model == slug and len(rep.scopes) == 4


def test_kimi_k3_linear_attn_config_normalization():
    """Verify KimiLinear / Kimi-K3.1-Pro nested `linear_attn_config` normalization."""
    prof = advisor.model_profile_from_hf_config(_KIMI_K3_PRO_CFG, name="kimi-k3.1-pro")
    assert prof == advisor.MODEL_PRESETS["kimi-k3.1-pro"]
    assert (
        dataclasses.replace(prof, name="kimi-k3-pro")
        == advisor.MODEL_PRESETS["kimi-k3-pro"]
    )
    assert advisor.resolve_model_profile(_KIMI_K3_PRO_CFG).num_linear_layers == 69


def test_chip_specs_match_jax_pallas_tpu_info(monkeypatch: pytest.MonkeyPatch):
    """Verify all 5 public TPU chips, `auto`, and Megacore match `tpu_info`."""
    assert mk.advisor is advisor
    chips = (
        ("v4", "v4"),
        ("v5e", "v5e"),
        ("v5p", "v5p"),
        ("v6e", "v6e"),
        ("tpu7x", "7x"),
    )
    for name, cv_str in chips:
        cs = advisor.resolve_chip_spec(name)
        assert advisor.resolve_chip_spec(cs) is cs
        info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion(cv_str), 1)
        assert cs.vmem_capacity_bytes == info.vmem_capacity_bytes
        assert cs.hbm_capacity_bytes == info.hbm_capacity_bytes
        assert cs.hbm_bw_bytes_per_s == float(info.mem_bw_bytes_per_second)
        assert cs.bf16_flops_per_s == float(info.bf16_ops_per_second)

    mc = advisor.resolve_chip_spec("v5p", megacore=True)
    sp = advisor.resolve_chip_spec("v5p", megacore=False)
    assert mc.cores_per_device == 2
    assert mc.vmem_capacity_bytes == 2 * sp.vmem_capacity_bytes
    assert mc.ici_egress_bw_bytes_per_s == 2 * sp.ici_egress_bw_bytes_per_s

    with pytest.raises(MegakernelConfigError, match="requires TPU"):
        advisor.resolve_chip_spec("auto")
    fake_7x = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion("7x"), 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: fake_7x)
    auto_cs = advisor.resolve_chip_spec("auto")
    assert auto_cs.name == "tpu7x" and auto_cs.cores_per_device == 1

    with pytest.raises(MegakernelConfigError, match="Megacore"):
        advisor.resolve_chip_spec("tpu7x", megacore=True)


def test_multi_tpu_vmem_and_fp8_dtype_physics():
    """Verify TPU FP8 physics, VMEM ring overflow, Mixtral MoE, and DeepSeek MLA."""
    c7x = advisor.resolve_chip_spec("tpu7x")
    cv6e = advisor.resolve_chip_spec("v6e")
    cv5e = advisor.resolve_chip_spec("v5e")
    cv5p = advisor.resolve_chip_spec("v5p")
    cv4 = advisor.resolve_chip_spec("v4")

    assert c7x.native_ocp_fp8 is True and c7x.peak_flops("fp8") == 2.30e15
    assert c7x.lhs_row_floor("fp8") == 32 and c7x.vmem_usable_bytes == 62 << 20
    assert cv6e.native_ocp_fp8 is False and cv6e.peak_flops("fp8") == 9.20e14
    assert cv5p.peak_flops("fp8") == cv5p.bf16_flops_per_s
    assert cv6e.vmem_usable_bytes == 126 << 20 and cv4.vmem_usable_bytes == 14 << 20

    p397 = advisor.model_profile_from_hf_config(_QWEN35_397B_CFG, name="qwen35-397b")
    rep_v4 = advisor.advise(p397, advisor.WorkloadSpec(tp=8, batch=4), cv4)
    v4_cross = next(s for s in rep_v4.scopes if s.scope == "cross_layer")
    assert rep_v4.hbm_fits is False and rep_v4.build_megakernel is False
    assert v4_cross.t_vmem_max == 0 and v4_cross.reason == "Rings exceed VMEM"
    rep_v6e_tp8 = advisor.advise(p397, advisor.WorkloadSpec(tp=8, batch=4), cv6e)
    assert rep_v6e_tp8.hbm_fits is False

    # Mixtral-8x7B (`num_local_experts`, `intermediate_size`) exceeds v5e HBM at TP4;
    # `ffn_dim` fallback also populates `moe_intermediate_size`.
    pmix = advisor.model_profile_from_hf_config(_MIXTRAL_8X7B_CFG, name="mixtral-8x7b")
    assert pmix.num_moe_layers == 32 and pmix.moe_intermediate_size == 14336
    rep_mix = advisor.advise(pmix, advisor.WorkloadSpec(tp=4, batch=16), cv5e)
    assert rep_mix.hbm_fits is False and rep_mix.hbm_used_gb > 24.0
    p_ffn = advisor.model_profile_from_hf_config(
        {
            "num_hidden_layers": 2,
            "hidden_size": 512,
            "ffn_dim": 2048,
            "num_attention_heads": 8,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
        }
    )
    assert p_ffn.intermediate_size == 2048 and p_ffn.moe_intermediate_size == 2048

    # DeepSeek-V3 MLA (`kv_lora_rank`, `q_lora_rank`, `v_head_dim`, shared expert):
    # 61 * (36,634,624 MLA + 21,504 min FFN) = 2.2360 GB/rank at TP8.
    pdsv3 = advisor.model_profile_from_hf_config(_DEEPSEEK_V3_CFG, name="deepseek-v3")
    assert pdsv3.head_dim == 128 and pdsv3.shared_expert_intermediate_size == 2048
    assert pdsv3.num_moe_layers == 58 and pdsv3.is_periodic is False
    pdsv3_dense_only = dataclasses.replace(
        pdsv3, num_moe_layers=0, num_experts=0, top_k=0, intermediate_size=0
    )
    r_mla_only = advisor.advise(
        pdsv3_dense_only, advisor.WorkloadSpec(tp=8, batch=1, context=4096), c7x
    )
    assert r_mla_only.scopes[0].w_step_gb == 2.236
    r_mla_pre = advisor.advise(
        pdsv3_dense_only,
        advisor.WorkloadSpec(phase="prefill", tp=8, batch=1, isl=4096, chunk_size=1),
        c7x,
    )
    assert r_mla_only.step_arithmetic_intensity > r_mla_pre.step_arithmetic_intensity
    r_ds1 = advisor.advise(
        pdsv3, advisor.WorkloadSpec(tp=8, batch=32, context=8192), c7x
    )
    r_ds2 = advisor.advise(
        pdsv3, advisor.WorkloadSpec(tp=8, kv_split=2, batch=32, context=8192), c7x
    )
    assert r_ds2.hbm_used_gb < r_ds1.hbm_used_gb
    assert next(s for s in r_ds1.scopes if s.scope == "cross_layer").feasible is False
    r_ver = advisor.advise(
        pdsv3, advisor.WorkloadSpec(phase="verify", tp=8, batch=8, spec_tokens=3), c7x
    )
    r_mix = advisor.advise(
        pdsv3,
        advisor.WorkloadSpec(
            phase="mixed", tp=8, batch=8, spec_tokens=3, chunk_size=128
        ),
        c7x,
    )
    assert r_ver.active_tokens == 32 and r_mix.active_tokens == 160


def test_asymptotic_decode_intensity_and_vmem_fit_headroom():
    """Verify `asymptotic_decode_intensity`, `fit_vmem_budget_bytes`, `w_crit_gb`."""
    assert advisor.fit_vmem_budget_bytes() == 56 << 20
    assert advisor.fit_vmem_budget_bytes(Tiles(vmem_mib=62)) == 56 << 20
    p08 = advisor.model_profile_from_hf_config(
        _QWEN35_08B_CFG, name="qwen35-0.8b", state_dtype="fp32"
    )
    r_b16 = advisor.advise(
        p08,
        advisor.WorkloadSpec(phase="verify", batch=16, spec_tokens=3, tp=8),
        "tpu7x",
    )
    r_b128 = advisor.advise(
        p08,
        advisor.WorkloadSpec(phase="verify", batch=128, spec_tokens=3, tp=8),
        "tpu7x",
    )
    assert r_b16.asymptotic_decode_intensity is not None
    assert r_b16.asymptotic_decode_intensity == r_b128.asymptotic_decode_intensity
    assert r_b16.asymptotic_decode_intensity < r_b16.ridge_dense_flop_per_byte
    assert r_b16.step_arithmetic_intensity < r_b16.asymptotic_decode_intensity

    r_alias = advisor.advise(
        p08, advisor.WorkloadSpec(phase="prefill", batch=1, isl=8192, tp=8), "tpu7x"
    )
    r_no_alias = advisor.advise(
        p08,
        advisor.WorkloadSpec(
            phase="prefill",
            batch=1,
            isl=8192,
            tp=8,
            vmem_alias=False,
            vmem_fit_headroom=True,
        ),
        "tpu7x",
    )
    r_no_alias_disp = advisor.advise(
        p08,
        advisor.WorkloadSpec(
            phase="prefill",
            batch=1,
            isl=8192,
            tp=8,
            vmem_alias=False,
            vmem_fit_headroom=True,
            op_overhead_us=5.0,
        ),
        "tpu7x",
    )
    assert r_alias.vmem_budget_mib == 62.0 and r_no_alias.vmem_budget_mib == 56.0
    cx_alias = next(s for s in r_alias.scopes if s.scope == "cross_layer")
    cx_no_al = next(s for s in r_no_alias.scopes if s.scope == "cross_layer")
    cx_disp = next(s for s in r_no_alias_disp.scopes if s.scope == "cross_layer")
    assert cx_alias.n_splits == 1 and cx_no_al.n_splits > 1
    assert cx_no_al.w_crit_gb is not None and cx_disp.w_crit_gb is not None
    assert cx_disp.w_crit_gb > cx_no_al.w_crit_gb
    assert cx_disp.step_bound_us > cx_no_al.step_bound_us


def test_all_four_fusion_scopes_reachable():
    """Prove `cross_layer`, `single_layer`, `stage`, and `sublayer` each win."""
    p08 = advisor.model_profile_from_hf_config(_QWEN35_08B_CFG, name="qwen35-0.8b")
    p397 = advisor.model_profile_from_hf_config(_QWEN35_397B_CFG, name="qwen35-397b")

    # 1. `cross_layer` wins on 397B TP8 decode (B=16, B=32) and 0.8B 8K prefill.
    r_dec = advisor.advise(
        p397, advisor.WorkloadSpec(phase="decode", batch=16, tp=8), "tpu7x"
    )
    assert r_dec.recommended_scope == "cross_layer" and r_dec.build_megakernel is True
    r_b32 = advisor.advise(
        p397, advisor.WorkloadSpec(phase="decode", batch=32, tp=8), "tpu7x"
    )
    assert r_b32.recommended_scope == "cross_layer"
    r_08_pre = advisor.advise(
        p08, advisor.WorkloadSpec(phase="prefill", batch=1, isl=8192, tp=8), "tpu7x"
    )
    assert r_08_pre.recommended_scope == "cross_layer"

    # 2. `single_layer` wins in the intermediate VMEM cliff band (B=96 on tpu7x),
    # whereas 128 MiB v6e (TP16) and v5p Megacore (TP8) keep `cross_layer` in 1 pass,
    # and saturated DP4 x TP2 x EP8 decode (B_pod=600) selects `sublayer`.
    r_b96_7x = advisor.advise(
        p397, advisor.WorkloadSpec(phase="decode", batch=96, tp=8), "tpu7x"
    )
    assert r_b96_7x.recommended_scope == "single_layer"
    r_b96_v6e = advisor.advise(
        p397, advisor.WorkloadSpec(phase="decode", batch=96, tp=16), "v6e"
    )
    assert r_b96_v6e.recommended_scope == "cross_layer"
    v5p_mc = advisor.resolve_chip_spec("v5p", megacore=True)
    r_b96_v5p_mc = advisor.advise(
        p397, advisor.WorkloadSpec(phase="decode", batch=96, tp=8), v5p_mc
    )
    assert r_b96_v5p_mc.recommended_scope == "cross_layer"
    r_ep8_sat = advisor.advise(
        p397, advisor.WorkloadSpec(phase="decode", batch=600, tp=2, dp=4, ep=8), "tpu7x"
    )
    assert r_ep8_sat.recommended_scope == "sublayer"

    # 3. `stage` wins on PP8 x TP1 tail (chunk=64) and segmented TP8 (`segment=10`).
    r_stg = advisor.advise(
        p397,
        advisor.WorkloadSpec(
            phase="prefill", batch=1, isl=4096, chunk_size=64, tp=1, pp=8
        ),
        "tpu7x",
    )
    assert r_stg.recommended_scope == "stage"
    r_seg = advisor.advise(
        p397,
        advisor.WorkloadSpec(phase="decode", batch=16, tp=8, segment_layers=10),
        "tpu7x",
    )
    assert r_seg.recommended_scope == "stage"
    assert next(s for s in r_seg.scopes if s.scope == "stage").ops_per_step == 18
    r_16k = advisor.advise(
        p397,
        advisor.WorkloadSpec(phase="prefill", batch=8, isl=16384, tp=1, pp=8),
        "tpu7x",
    )
    assert r_16k.recommended_scope == "sublayer" and r_16k.build_megakernel is False


def test_advise_from_model_spec_and_planner():
    """Verify `advise_from_spec` bridges periodic and non-periodic `ModelSpec`."""
    ms = make_fake_model_spec(L=4, att_every=4, H=256)
    ex = ExecSpec(B=4, T=1, R=2, ctx=1024, tiles=TINY_TILES)
    rep = advisor.advise_from_spec(ms, ex, chip="tpu7x", moe_mode="binned_vmem")
    assert rep.recommended_scope == "cross_layer"
    rep_def = advisor.advise_from_spec(
        ms, ExecSpec(B=4, T=1, R=2, ctx=1024), chip="tpu7x", moe_mode="binned_vmem"
    )
    assert (
        rep_def.recommended_scope == "cross_layer" and rep_def.vmem_budget_mib == 62.0
    )

    ms_non_periodic = make_fake_model_spec(L=5, att_every=4, H=256)
    rep_np = advisor.advise_from_spec(
        ms_non_periodic, ex, chip="tpu7x", moe_mode="binned_vmem"
    )
    cross = next(s for s in rep_np.scopes if s.scope == "cross_layer")
    assert cross.feasible is False and rep_np.recommended_scope == "single_layer"


@pytest.mark.parametrize(
    "bad_extra",
    [
        ["--prefix-hit-rate", "1.0"],
        ["--ep", "0"],
        ["--kv-split", "0"],
        ["--op-overhead-us", "-1.0"],
    ],
)
def test_cli_parameterized_validation_errors(
    tmp_path: Path, bad_extra: list[str]
) -> None:
    """Verify parameterized CLI validation boundaries raise `MegakernelConfigError`."""
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(_QWEN35_08B_CFG), encoding="utf-8")
    with pytest.raises(MegakernelConfigError):
        advisor.main(["--config", str(cfg_file), *bad_extra])


def test_cli_json_presets_and_validation_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """Verify CLI JSON/table output, preset resolution, and error wrapping."""
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(_QWEN35_08B_CFG), encoding="utf-8")
    rc = advisor.main(
        ["--config", str(cfg_file), "--chip", "tpu7x", "--tp", "8", "--json"]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert (
        payload["recommended_scope"] == "cross_layer"
        and payload["build_megakernel"] is True
        and payload["vmem_budget_mib"] == 62.0
    )
    assert advisor.main(["--model", str(tmp_path), "--chip", "tpu7x", "--tp", "8"]) == 0
    assert "Recommended: cross_layer" in capsys.readouterr().out

    assert (
        advisor.main(
            [
                "--model",
                "moonshotai/Kimi-K3-Pro",
                "--tp",
                "16",
                "--ep",
                "16",
                "--vmem-fit-headroom",
                "--no-vmem-alias",
            ]
        )
        == 0
    )
    assert "Model: kimi-k3-pro" in capsys.readouterr().out

    for bad_argv in (
        ["--config", str(tmp_path / "missing.json")],
        ["--model", str(tmp_path / "missing_dir")],
        ["--model", "org/unknown-preset"],
    ):
        with pytest.raises(MegakernelConfigError):
            advisor.main(bad_argv)

    bad_dense = dataclasses.replace(
        advisor.model_profile_from_hf_config(_QWEN35_08B_CFG), top_k=2
    )
    with pytest.raises(MegakernelConfigError):
        advisor.advise(bad_dense, advisor.WorkloadSpec(tp=8))
