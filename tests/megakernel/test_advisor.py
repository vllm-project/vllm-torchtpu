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
import re
import sys
from html.parser import HTMLParser
from typing import TYPE_CHECKING

import pytest
from jax.experimental.pallas import tpu as pltpu

from tests.megakernel.megakernel_fakes import TINY_TILES, make_fake_model_spec
from vllm_torchtpu import megakernel as mk
from vllm_torchtpu.megakernel import advisor
from vllm_torchtpu.megakernel.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.exec_spec import ExecSpec, Tiles

if TYPE_CHECKING:
    from pathlib import Path

# HuggingFace Qwen3.5, Mixtral-8x7B, DeepSeek-V3, and Kimi-K3-Pro configs.
_QWEN35_08B_CFG = {
    "model_type": "qwen3_5",
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
    "model_type": "qwen3_5_moe",
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
    "model_type": "mixtral",
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
    "model_type": "deepseek_v3",
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
    "model_type": "kimi_linear",
    "num_hidden_layers": 93,
    "first_k_dense_replace": 1,
    "hidden_size": 7168,
    "intermediate_size": 33792,
    "moe_intermediate_size": 3072,
    "shared_expert_intermediate_size": 6144,
    "routed_expert_hidden_size": 3584,
    "num_attention_heads": 96,
    "num_key_value_heads": 96,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "num_experts": 896,
    "num_experts_per_token": 16,
    "linear_attn_config": {
        "kda_layers": list(range(69)),
        "num_heads": 96,
        "head_k_dim": 128,
        "head_v_dim": 128,
        "short_conv_kernel_size": 4,
    },
    "quantization_config": {"quant_method": "fp8", "expert_dtype": "mxfp4"},
}

_NUM_SCOPES = 4
_NUM_REGIMES = 12
_KIMI_ROUTED_HIDDEN = 3584
_KIMI_KDA_LAYERS = 69
_MEGACORE_CORES = 2
_TPU7X_FP8_FLOPS = 2.30e15
_TPU7X_FP8_ROW_FLOOR = 32
_V6E_FP8_FLOPS = 9.20e14
_MIXTRAL_LAYERS = 32
_MIXTRAL_MOE_INTER = 14336
_V5E_HBM_CAP_GB = 24.0
_FFN_DIM_2048 = 2048
_DSV3_HEAD_DIM = 128
_DSV3_MOE_LAYERS = 58
_DSV3_MLA_W_STEP_GB = 2.236
_VERIFY_ACTIVE_TOKENS = 32
_MIXED_ACTIVE_TOKENS = 160
_TPU7X_VMEM_MIB = 62.0
_FIT_VMEM_MIB = 56.0
_CUSTOM_VMEM_MIB = 58.0
_STAGE_OPS_PER_STEP = 18
_KIMI_GROUPED_B_TOK = 139264
_KIMI_CROSS_T_MAX = 67
_KIMI_1L_T_MAX = 384
_QWEN38_CROSS_T_MAX = 42
# Shipped Qwen3.5 decode megakernel: largest single-call verify bucket (rows).
_QWEN35_SHIPPED_VERIFY_ROWS = 80
# 16 sequences x MTP-3 verify (4 rows per sequence) in one call.
_QWEN35_VERIFY_ROWS_MIN = 64
# Shipped Kimi-K3 prefill calls: embedding + layers 0-3, 11 groups, last layer.
_K3_PREFILL_CALLS = 13
_KERNEL_NAME_DENYLIST = re.compile(
    r"gmm|ragged_dot|megablox|sparsecore|mxu", re.IGNORECASE
)


def test_package_reexport_without_prior_submodule_import() -> None:
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
    "case",
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
    case: tuple[str, str, int, int, int, int, str, str],
) -> None:
    """Verify all 13 Roofline RFC, Kimi K3/Pro, and Mistral/Mixtral presets."""
    slug, repo_id, layers, attn_l, lin_l, moe_l, wd, ed = case
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
    assert rep.model == slug
    assert len(rep.scopes) == _NUM_SCOPES


def test_kimi_k3_linear_attn_config_normalization() -> None:
    """Verify KimiLinear / Kimi-K3.1-Pro nested `linear_attn_config` and latent MoE."""
    prof = advisor.model_profile_from_hf_config(_KIMI_K3_PRO_CFG, name="kimi-k3.1-pro")
    assert prof == advisor.MODEL_PRESETS["kimi-k3.1-pro"]
    assert prof.routed_expert_hidden_size == _KIMI_ROUTED_HIDDEN
    assert (
        dataclasses.replace(prof, name="kimi-k3-pro")
        == advisor.MODEL_PRESETS["kimi-k3-pro"]
    )
    assert (
        advisor.resolve_model_profile(_KIMI_K3_PRO_CFG).num_linear_layers
        == _KIMI_KDA_LAYERS
    )


@pytest.mark.parametrize(
    "bad_cfg",
    [
        {},
        {
            "model_type": "bert",
            "num_hidden_layers": 12,
            "hidden_size": 768,
            "num_attention_heads": 12,
        },
        {
            "model_type": "mamba2",
            "num_hidden_layers": 24,
            "hidden_size": 2048,
            "num_attention_heads": 16,
        },
        {
            "model_type": "gemma3_unregistered",
            "num_hidden_layers": 24,
            "hidden_size": 2048,
            "num_attention_heads": 16,
        },
        {"text_config": "invalid_not_a_dict"},
        {"vision_config": {}, "text_config": {"num_hidden_layers": 12}},
        {
            "model_type": "qwen3_5",
            "num_hidden_layers": 4,
            "hidden_size": 1024,
            "num_attention_heads": 8,
            "layer_types": ["full_attention", "unknown_layer_token", "gdn", "gdn"],
        },
        {
            "model_type": "qwen3_5",
            "num_hidden_layers": 4,
            "hidden_size": 1024,
            "num_attention_heads": 8,
            "full_attention_interval": 4,
        },
    ],
)
def test_unregistered_and_malformed_model_rejection(bad_cfg: dict[str, object]) -> None:
    """Verify unregistered model_types raise UnsupportedArchitectureError."""
    with pytest.raises(advisor.UnsupportedArchitectureError):
        advisor.model_profile_from_hf_config(bad_cfg)


def test_per_token_vmem_live_interval_and_sharding() -> None:
    """Verify per-token live VMEM interval math and TP/EP sharding on Kimi-K3.1-Pro."""
    pk3 = advisor.resolve_model_profile("kimi-k3.1-pro")
    wl_grp = advisor.WorkloadSpec(
        phase="verify",
        batch=16,
        spec_tokens=3,
        tp=32,
        ep=32,
        moe_mode="grouped_on_chip",
    )
    parts_grp = advisor.compute_per_token_vmem_bytes(pk3, wl_grp)
    assert parts_grp["total_per_token_bytes"] == _KIMI_GROUPED_B_TOK
    assert parts_grp["b_tok"] == _KIMI_GROUPED_B_TOK
    rep_grp = advisor.advise(pk3, wl_grp, "tpu7x")
    cross_sc = next(s for s in rep_grp.scopes if s.scope == "whole_step")
    one_l_sc = next(s for s in rep_grp.scopes if s.scope == "single_layer")
    assert cross_sc.max_vmem_tokens == _KIMI_CROSS_T_MAX
    assert one_l_sc.max_vmem_tokens == _KIMI_1L_T_MAX

    wl_msk = dataclasses.replace(wl_grp, moe_mode="masked_on_chip")
    parts_msk = advisor.compute_per_token_vmem_bytes(pk3, wl_msk)
    assert parts_grp["total_per_token_bytes"] < parts_msk["total_per_token_bytes"]

    wl_no_alias = dataclasses.replace(wl_grp, vmem_alias=False)
    parts_no_alias = advisor.compute_per_token_vmem_bytes(pk3, wl_no_alias)
    assert parts_no_alias["total_per_token_bytes"] > parts_grp["total_per_token_bytes"]


def test_chip_specs_match_jax_pallas_tpu_info(monkeypatch: pytest.MonkeyPatch) -> None:
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
    assert mc.cores_per_device == _MEGACORE_CORES
    assert mc.vmem_capacity_bytes == _MEGACORE_CORES * sp.vmem_capacity_bytes
    assert (
        mc.ici_egress_bw_bytes_per_s == _MEGACORE_CORES * sp.ici_egress_bw_bytes_per_s
    )

    with pytest.raises(MegakernelConfigError, match="requires TPU"):
        advisor.resolve_chip_spec("auto")
    fake_7x = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion("7x"), 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: fake_7x)
    auto_cs = advisor.resolve_chip_spec("auto")
    assert auto_cs.name == "tpu7x"
    assert auto_cs.cores_per_device == 1

    with pytest.raises(MegakernelConfigError, match="Megacore"):
        advisor.resolve_chip_spec("tpu7x", megacore=True)


def test_multi_tpu_vmem_and_fp8_dtype_physics() -> None:
    """Verify TPU FP8 physics, VMEM buffer overflow, Mixtral MoE, and DeepSeek MLA."""
    c7x = advisor.resolve_chip_spec("tpu7x")
    cv6e = advisor.resolve_chip_spec("v6e")
    cv5e = advisor.resolve_chip_spec("v5e")
    cv5p = advisor.resolve_chip_spec("v5p")
    cv4 = advisor.resolve_chip_spec("v4")

    assert (c7x.native_ocp_fp8, c7x.peak_flops("fp8"), c7x.lhs_row_floor("fp8")) == (
        True,
        _TPU7X_FP8_FLOPS,
        _TPU7X_FP8_ROW_FLOOR,
    )
    assert (cv6e.native_ocp_fp8, cv6e.peak_flops("fp8"), cv5p.peak_flops("fp8")) == (
        False,
        _V6E_FP8_FLOPS,
        cv5p.bf16_flops_per_s,
    )
    assert (
        c7x.vmem_usable_bytes,
        cv6e.vmem_usable_bytes,
        cv4.vmem_usable_bytes,
    ) == (62 << 20, 126 << 20, 14 << 20)

    p397 = advisor.model_profile_from_hf_config(_QWEN35_397B_CFG, name="qwen35-397b")
    rep_v4 = advisor.advise(p397, advisor.WorkloadSpec(tp=8, batch=4), cv4)
    v4_cross = next(s for s in rep_v4.scopes if s.scope == "whole_step")
    assert rep_v4.hbm_fits is False
    assert rep_v4.build_megakernel is False
    assert v4_cross.t_vmem_max == 0
    assert v4_cross.reason == "Weight buffers exceed VMEM"
    rep_v6e_tp8 = advisor.advise(p397, advisor.WorkloadSpec(tp=8, batch=4), cv6e)
    assert rep_v6e_tp8.hbm_fits is False

    pmix = advisor.model_profile_from_hf_config(_MIXTRAL_8X7B_CFG, name="mixtral-8x7b")
    assert pmix.num_moe_layers == _MIXTRAL_LAYERS
    assert pmix.moe_intermediate_size == _MIXTRAL_MOE_INTER
    rep_mix = advisor.advise(pmix, advisor.WorkloadSpec(tp=4, batch=16), cv5e)
    assert rep_mix.hbm_fits is False
    assert rep_mix.hbm_used_gb > _V5E_HBM_CAP_GB
    p_ffn = advisor.model_profile_from_hf_config(
        {
            "model_type": "mixtral",
            "num_hidden_layers": 2,
            "hidden_size": 512,
            "ffn_dim": 2048,
            "num_attention_heads": 8,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
        }
    )
    assert p_ffn.intermediate_size == _FFN_DIM_2048
    assert p_ffn.moe_intermediate_size == _FFN_DIM_2048

    pdsv3 = advisor.model_profile_from_hf_config(_DEEPSEEK_V3_CFG, name="deepseek-v3")
    assert pdsv3.head_dim == _DSV3_HEAD_DIM
    assert pdsv3.shared_expert_intermediate_size == _FFN_DIM_2048
    assert pdsv3.num_moe_layers == _DSV3_MOE_LAYERS
    assert pdsv3.is_periodic is False
    pdsv3_dense_only = dataclasses.replace(
        pdsv3, num_moe_layers=0, num_experts=0, top_k=0, intermediate_size=0
    )
    r_mla_only = advisor.advise(
        pdsv3_dense_only, advisor.WorkloadSpec(tp=8, batch=1, context=4096), c7x
    )
    assert r_mla_only.scopes[0].w_step_gb == _DSV3_MLA_W_STEP_GB
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
    assert next(s for s in r_ds1.scopes if s.scope == "whole_step").feasible is False
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
    assert r_ver.active_tokens == _VERIFY_ACTIVE_TOKENS
    assert r_mix.active_tokens == _MIXED_ACTIVE_TOKENS


def test_asymptotic_decode_intensity_and_vmem_fit_headroom() -> None:
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
        p08, advisor.WorkloadSpec(phase="prefill", batch=1, isl=2048, tp=8), "tpu7x"
    )
    r_no_alias = advisor.advise(
        p08,
        advisor.WorkloadSpec(
            phase="prefill",
            batch=1,
            isl=2048,
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
            isl=2048,
            tp=8,
            vmem_alias=False,
            vmem_fit_headroom=True,
            op_overhead_us=5.0,
        ),
        "tpu7x",
    )
    assert r_alias.vmem_budget_mib == _TPU7X_VMEM_MIB
    assert r_no_alias.vmem_budget_mib == _FIT_VMEM_MIB
    cx_alias = next(s for s in r_alias.scopes if s.scope == "whole_step")
    cx_no_al = next(s for s in r_no_alias.scopes if s.scope == "whole_step")
    cx_disp = next(s for s in r_no_alias_disp.scopes if s.scope == "whole_step")
    assert cx_alias.n_splits == 1
    assert cx_no_al.n_splits > 1
    assert cx_no_al.w_crit_gb is not None
    assert cx_disp.w_crit_gb is not None
    assert cx_disp.w_crit_gb > cx_no_al.w_crit_gb
    assert cx_disp.step_bound_us > cx_no_al.step_bound_us


def test_all_four_fusion_scopes_reachable() -> None:
    """Prove `whole_step`, `single_layer`, `stage`, and `per_op` each win."""
    p08 = advisor.model_profile_from_hf_config(_QWEN35_08B_CFG, name="qwen35-0.8b")
    p397 = advisor.model_profile_from_hf_config(_QWEN35_397B_CFG, name="qwen35-397b")
    pllama = advisor.resolve_model_profile("llama-3.1-405b")

    # 1. `whole_step` (`whole_step`) wins on 397B TP8 decode and 0.8B prefill.
    r_dec = advisor.advise(
        p397, advisor.WorkloadSpec(phase="decode", batch=16, tp=8), "tpu7x"
    )
    assert r_dec.recommended_scope == "whole_step"
    assert r_dec.recommended_moe_mode == "masked_on_chip"
    assert r_dec.build_megakernel is True
    r_b32 = advisor.advise(
        p397, advisor.WorkloadSpec(phase="decode", batch=32, tp=8), "tpu7x"
    )
    assert r_b32.recommended_scope == "whole_step"
    assert r_b32.recommended_moe_mode == "grouped_on_chip"
    r_08_pre = advisor.advise(
        p08, advisor.WorkloadSpec(phase="prefill", batch=1, isl=2048, tp=8), "tpu7x"
    )
    assert r_08_pre.recommended_scope == "whole_step"
    assert r_08_pre.recommended_moe_mode == "dense_mlp"

    # 2. `single_layer` (`layer`) wins in the B=96 VMEM cliff band on tpu7x,
    # whereas v6e (TP16) and v5p Megacore (TP8) keep `whole_step` in 1 pass.
    r_b96_7x = advisor.advise(
        p397, advisor.WorkloadSpec(phase="decode", batch=96, tp=8), "tpu7x"
    )
    assert r_b96_7x.recommended_scope == "single_layer"
    assert r_b96_7x.recommended_moe_mode == "grouped_on_chip"
    r_b96_v6e = advisor.advise(
        p397, advisor.WorkloadSpec(phase="decode", batch=96, tp=16), "v6e"
    )
    assert r_b96_v6e.recommended_scope == "whole_step"
    v5p_mc = advisor.resolve_chip_spec("v5p", megacore=True)
    r_b96_v5p_mc = advisor.advise(
        p397, advisor.WorkloadSpec(phase="decode", batch=96, tp=8), v5p_mc
    )
    assert r_b96_v5p_mc.recommended_scope == "whole_step"
    r_ep8_sat = advisor.advise(
        p397, advisor.WorkloadSpec(phase="decode", batch=600, tp=2, dp=4, ep=8), "tpu7x"
    )
    assert r_ep8_sat.recommended_scope == "per_op"
    assert r_ep8_sat.recommended_moe_mode == "grouped_in_hbm"

    r_llama_pre = advisor.advise(
        pllama, advisor.WorkloadSpec(phase="prefill", batch=1, isl=4096, tp=8), "tpu7x"
    )
    assert r_llama_pre.recommended_scope == "per_op"
    assert r_llama_pre.recommended_moe_mode == "dense_mlp"
    assert r_llama_pre.build_megakernel is False

    # 3. `layer_group` wins on PP8 tail and segmented TP8 (`segment_layers=10`).
    r_stg = advisor.advise(
        p397,
        advisor.WorkloadSpec(
            phase="prefill", batch=1, isl=4096, chunk_size=64, tp=1, pp=8
        ),
        "tpu7x",
    )
    assert r_stg.recommended_scope == "layer_group"
    r_seg = advisor.advise(
        p397,
        advisor.WorkloadSpec(phase="decode", batch=16, tp=8, segment_layers=10),
        "tpu7x",
    )
    assert r_seg.recommended_scope == "layer_group"
    assert (
        next(s for s in r_seg.scopes if s.scope == "layer_group").ops_per_step
        == _STAGE_OPS_PER_STEP
    )
    r_16k = advisor.advise(
        p397,
        advisor.WorkloadSpec(phase="prefill", batch=8, isl=16384, tp=1, pp=8),
        "tpu7x",
    )
    assert r_16k.recommended_scope == "per_op"
    assert r_16k.build_megakernel is False


def test_bare_minimum_auto_parallelism_and_html_regime_report(tmp_path: Path) -> None:
    """Verify bare-minimum Model + TPUv7x 12-regime sweep and HTML SVG export."""
    sweep_397 = advisor.advise_regimes("qwen3.5-397b", "tpu7x", deployment_preset=False)
    assert sweep_397.parallelism.auto_selected is True
    assert (sweep_397.parallelism.tp, sweep_397.parallelism.ep) == (8, 8)
    assert "2x2x1" in sweep_397.parallelism.topology
    assert len(sweep_397.regimes) == _NUM_REGIMES
    by_label = {r.label: r for r in sweep_397.regimes}
    assert by_label["Decode Small (B=4, T=1, 4K)"].recommended_scope == "whole_step"
    assert (
        by_label["Decode Small (B=4, T=1, 4K)"].recommended_moe_mode == "masked_on_chip"
    )
    assert by_label["Decode High (B=64, T=1, 32K)"].recommended_scope == "whole_step"
    assert (
        by_label["Decode High (B=64, T=1, 32K)"].recommended_moe_mode
        == "grouped_on_chip"
    )
    assert by_label["Chunked Prefill (B=1, T=256)"].recommended_scope == "single_layer"
    assert (
        by_label["Chunked Prefill (B=1, T=256)"].recommended_moe_mode
        == "grouped_on_chip"
    )
    assert by_label["Long Prefill (B=1, T=4096)"].recommended_scope == "per_op"
    assert (
        by_label["Long Prefill (B=1, T=4096)"].recommended_moe_mode == "grouped_in_hbm"
    )

    sweep_dcp = advisor.advise_regimes("qwen3.5-397b", "tpu7x", dcp=4, pcp=2)
    assert (sweep_dcp.parallelism.dcp, sweep_dcp.parallelism.pcp) == (4, 2)
    by_label_dcp = {r.label: r for r in sweep_dcp.regimes}
    assert (
        by_label_dcp["Long Prefill (B=1, T=4096)"].active_tokens
        == by_label["Long Prefill (B=1, T=4096)"].active_tokens // 2
    )

    sweep_k3 = advisor.advise_regimes("kimi-k3.1-pro", "tpu7x")
    assert (sweep_k3.parallelism.tp, sweep_k3.parallelism.ep) == (32, 32)
    assert "2x2x4" in sweep_k3.parallelism.topology
    assert sweep_k3.regimes[0].recommended_scope == "whole_step"
    assert sweep_k3.regimes[-1].recommended_scope == "per_op"

    html_file = tmp_path / "advisor_report.html"
    advisor.write_advisor_html(sweep_397, html_file)
    html_text = html_file.read_text("utf-8")
    assert "<svg" in html_text
    assert "advisor-data" in html_text
    assert "qwen3.5-397b" in html_text
    assert "per_op" in html_text
    assert "masked_on_chip" in html_text
    assert "grouped_on_chip" in html_text
    assert "grouped_in_hbm" in html_text
    table_txt = advisor.format_regime_table(sweep_397)
    assert "Decode Small (B=4, T=1, 4K)" in table_txt
    assert "Scope (where the kernel boundary falls):" in table_txt
    assert "Tailoring Guide:" in table_txt


def test_qwen38_24t_decode_regime_crossover_and_moe_modes() -> None:
    """Verify Qwen3.8-2.4T TP32xEP32 recommends whole_step for B=4, 8, 16 decode."""
    sweep_24t = advisor.advise_regimes("qwen3.8-2.4t", "tpu7x")
    assert (sweep_24t.parallelism.tp, sweep_24t.parallelism.ep) == (32, 32)
    assert sweep_24t.max_vmem_tokens == _QWEN38_CROSS_T_MAX
    by_label = {r.label: r for r in sweep_24t.regimes}

    for small_lbl in (
        "Decode Small (B=4, T=1, 4K)",
        "Decode Low (B=8, T=1, 4K)",
        "Decode Med (B=16, T=1, 4K)",
        "Spec Verify Small (B=4, K=3)",
    ):
        entry = by_label[small_lbl]
        assert entry.recommended_scope == "whole_step"
        assert entry.recommended_moe_mode == "masked_on_chip"
        assert entry.num_vmem_splits == 1

    for layer_lbl in (
        "Decode High (B=64, T=1, 32K)",
        "Spec Verify Med (B=16, K=3)",
        "Chunked Prefill (B=1, T=256)",
    ):
        entry = by_label[layer_lbl]
        assert entry.recommended_scope == "single_layer"
        assert entry.recommended_moe_mode == "grouped_on_chip"

    long_pre = by_label["Long Prefill (B=1, T=4096)"]
    assert long_pre.recommended_scope == "per_op"
    assert long_pre.recommended_moe_mode == "grouped_in_hbm"


def test_vmem_budget_override_and_dma_buffers_triple_buffering() -> None:
    """Verify user-configurable `--vmem-budget-mib` and `--dma-buffers {2,3}`."""
    p397 = advisor.resolve_model_profile("qwen3.5-397b")
    rep_2buf = advisor.advise(
        p397, advisor.WorkloadSpec(tp=8, ep=8, batch=16, dma_buffers=2), "tpu7x"
    )
    rep_3buf = advisor.advise(
        p397, advisor.WorkloadSpec(tp=8, ep=8, batch=16, dma_buffers=3), "tpu7x"
    )
    rep_custom_vmem = advisor.advise(
        p397,
        advisor.WorkloadSpec(
            tp=8, ep=8, batch=16, dma_buffers=2, vmem_budget_mib=_CUSTOM_VMEM_MIB
        ),
        "tpu7x",
    )
    cx_2 = next(s for s in rep_2buf.scopes if s.scope == "whole_step")
    cx_3 = next(s for s in rep_3buf.scopes if s.scope == "whole_step")
    cx_vmem = next(s for s in rep_custom_vmem.scopes if s.scope == "whole_step")
    assert rep_2buf.vmem_budget_mib == _TPU7X_VMEM_MIB
    assert rep_custom_vmem.vmem_budget_mib == _CUSTOM_VMEM_MIB
    assert cx_vmem.max_vmem_tokens < cx_2.max_vmem_tokens
    assert cx_3.vmem_static_mib > cx_2.vmem_static_mib
    assert cx_3.max_vmem_tokens < cx_2.max_vmem_tokens
    assert cx_3.step_bound_us >= cx_2.step_bound_us


@pytest.mark.parametrize(
    ("tp", "ep", "pcp", "accepted"),
    [(4, 8, 2, True), (4, 8, 1, False), (4, 16, 2, False)],
)
def test_expert_parallel_spans_context_parallel_ranks(
    tp: int, ep: int, pcp: int, *, accepted: bool
) -> None:
    """Verify `ep` may span `tp * dp * pcp` ranks and no wider."""
    p397 = advisor.resolve_model_profile("qwen3.5-397b")
    wl = advisor.WorkloadSpec(phase="prefill", batch=1, isl=4096, tp=tp, ep=ep, pcp=pcp)
    if accepted:
        rep = advisor.advise(p397, wl, "tpu7x")
        assert rep.active_tokens == 4096 // pcp
    else:
        with pytest.raises(MegakernelConfigError):
            advisor.advise(p397, wl, "tpu7x")


def test_advise_from_model_spec_and_planner() -> None:
    """Verify `advise_from_spec` bridges periodic and non-periodic `ModelSpec`."""
    ms = make_fake_model_spec(L=4, att_every=4, H=256)
    ex = ExecSpec(B=4, T=1, R=2, ctx=1024, tiles=TINY_TILES)
    rep = advisor.advise_from_spec(ms, ex, chip="tpu7x", moe_mode="grouped_on_chip")
    assert rep.recommended_scope == "whole_step"
    rep_def = advisor.advise_from_spec(
        ms, ExecSpec(B=4, T=1, R=2, ctx=1024), chip="tpu7x", moe_mode="grouped_on_chip"
    )
    assert rep_def.recommended_scope == "whole_step"
    assert rep_def.vmem_budget_mib == _TPU7X_VMEM_MIB

    ms_non_periodic = make_fake_model_spec(L=5, att_every=4, H=256)
    rep_np = advisor.advise_from_spec(
        ms_non_periodic, ex, chip="tpu7x", moe_mode="grouped_on_chip"
    )
    cross = next(s for s in rep_np.scopes if s.scope == "whole_step")
    assert cross.feasible is False
    assert rep_np.recommended_scope == "single_layer"


@pytest.mark.parametrize(
    "bad_extra",
    [
        ["--prefix-hit-rate", "1.0"],
        ["--ep", "0"],
        ["--kv-split", "0"],
        ["--op-overhead-us", "-1.0"],
        ["--dma-buffers", "4"],
        ["--vmem-budget-mib", "1.0"],
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
) -> None:
    """Verify CLI JSON/table/HTML output, bare-minimum auto-TP, and error wrapping."""
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(_QWEN35_08B_CFG), encoding="utf-8")
    rc = advisor.main(
        ["--config", str(cfg_file), "--chip", "tpu7x", "--tp", "8", "--json"]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["recommended_scope"] == "whole_step"
    assert payload["recommended_moe_mode"] == "dense_mlp"
    assert payload["build_megakernel"] is True
    assert payload["vmem_budget_mib"] == _TPU7X_VMEM_MIB
    html_out = tmp_path / "regimes.html"
    assert (
        advisor.main(
            [
                "--model",
                "qwen3.5-397b",
                "--regime-sweep",
                "--vmem-budget-mib",
                "60.0",
                "--dma-buffers",
                "2",
                "--html",
                str(html_out),
            ]
        )
        == 0
    )
    sweep_stdout = capsys.readouterr().out
    assert "Decode Parallelism: TP8 x DCP4 x EP8" in sweep_stdout
    assert "usable=60.0 MiB" in sweep_stdout
    assert html_out.is_file()

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


def _preset_workload(
    layout: advisor.ParallelLayout, **changes: object
) -> advisor.WorkloadSpec:
    """Return a WorkloadSpec on a deployment preset layout with `changes` applied."""
    mesh = advisor.WorkloadSpec(
        tp=layout.tp,
        ep=layout.ep,
        dp=layout.dp,
        pp=layout.pp,
        dcp=layout.dcp,
        pcp=layout.pcp,
        layer_groups=layout.layer_groups,
    )
    return dataclasses.replace(mesh, **changes)


def test_qwen35_decode_preset_whole_step_admits_shipped_verify_rows() -> None:
    """Verify the whole_step ceiling fits 64 verify rows and lands near 80."""
    dep = advisor.DEPLOYMENT_PRESETS["qwen3.5-397b"]
    rows_per_seq = dep.decode.spec_tokens + 1
    rep = advisor.advise(
        advisor.resolve_model_profile("qwen3.5-397b"),
        _preset_workload(
            dep.decode,
            phase="verify",
            batch=_QWEN35_VERIFY_ROWS_MIN // rows_per_seq,
            spec_tokens=dep.decode.spec_tokens,
            moe_mode="grouped_on_chip",
        ),
        "tpu7x",
    )
    whole = next(s for s in rep.scopes if s.scope == "whole_step")
    rows_cap = rows_per_seq * (whole.t_vmem_max // rows_per_seq)
    assert rows_cap >= _QWEN35_VERIFY_ROWS_MIN
    assert abs(rows_cap - _QWEN35_SHIPPED_VERIFY_ROWS) <= rows_per_seq


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_kimi_k3_decode_preset_low_batch_prefers_whole_step(batch: int) -> None:
    """Verify Kimi-K3 decode at batch 1..8 on the TP32 mesh is one call."""
    dep = advisor.DEPLOYMENT_PRESETS["kimi-k3"]
    rep = advisor.advise(
        advisor.resolve_model_profile("kimi-k3"),
        _preset_workload(
            dep.decode,
            phase="decode",
            batch=batch,
        ),
        "tpu7x",
    )
    assert rep.recommended_scope == "whole_step"


@pytest.mark.parametrize("window", [2048, 8192])
def test_kimi_k3_prefill_preset_evaluates_layer_group(window: int) -> None:
    """Verify the shipped 13-call layer grouping is feasible and evaluated."""
    dep = advisor.DEPLOYMENT_PRESETS["kimi-k3"]
    rep = advisor.advise(
        advisor.resolve_model_profile("kimi-k3"),
        _preset_workload(
            dep.prefill,
            phase="prefill",
            batch=1,
            isl=window,
            chunk_size=window,
            moe_mode="grouped_on_chip",
        ),
        "tpu7x",
    )
    group = next(s for s in rep.scopes if s.scope == "layer_group")
    assert group.feasible
    assert group.ops_per_step == _K3_PREFILL_CALLS
    assert rep.active_tokens == window // dep.prefill.pcp


def test_layer_groups_must_cover_every_layer() -> None:
    """Verify a layer grouping that does not sum to num_layers is rejected."""
    with pytest.raises(MegakernelConfigError):
        advisor.advise(
            advisor.resolve_model_profile("kimi-k3"),
            advisor.WorkloadSpec(tp=32, ep=32, layer_groups=(8, 8)),
            "tpu7x",
        )


class _SvgTextCollector(HTMLParser):
    """Collect the viewBox and every <text> element's attributes and content."""

    def __init__(self) -> None:
        """Start with no viewBox, no collected text, and no open element."""
        super().__init__()
        self.view_box = ""
        self.texts: list[tuple[dict[str, str], str]] = []
        self._open: dict[str, str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Record the svg viewBox and open a text element."""
        attr = {k: v or "" for k, v in attrs}
        if tag == "svg":
            self.view_box = attr["viewbox"]
        elif tag == "text":
            self._open = attr

    def handle_data(self, data: str) -> None:
        """Attach character data to the open text element."""
        if self._open is not None:
            self.texts.append((self._open, data))
            self._open = None


@pytest.mark.parametrize("model", ["qwen3.5-397b", "kimi-k3"])
def test_regime_dot_plot_labels_once_and_inside_viewbox(model: str) -> None:
    """Verify each regime label renders once and no text leaves the viewBox."""
    sweep = advisor.advise_regimes(model, "tpu7x")
    parser = _SvgTextCollector()
    parser.feed(advisor.report.render_regime_dot_plot_svg(sweep))
    _, _, width, height = (float(v) for v in parser.view_box.split())
    for entry in sweep.regimes:
        assert sum(text == entry.label for _, text in parser.texts) == 1
    for attr, text in parser.texts:
        size = float(attr["font-size"])
        assert size >= advisor.report.SVG_MIN_FONT_PX
        span = advisor.report.svg_text_width(text, size)
        x, y = float(attr["x"]), float(attr["y"])
        left = {"start": x, "middle": x - span / 2, "end": x - span}[
            attr.get("text-anchor", "start")
        ]
        assert left >= 0.0
        assert left + span <= width
        assert y - size >= 0.0
        assert y <= height


@pytest.mark.parametrize("model", ["qwen3.5-397b", "kimi-k3", "qwen3.8-2.4t"])
def test_reports_name_mechanisms_not_kernels(model: str) -> None:
    """Verify the text table and HTML name no kernel or hardware unit."""
    sweep = advisor.advise_regimes(model, "tpu7x")
    rendered = advisor.format_regime_table(sweep) + advisor.render_advisor_html(sweep)
    assert _KERNEL_NAME_DENYLIST.search(rendered) is None
