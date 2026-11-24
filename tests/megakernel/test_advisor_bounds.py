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
"""VMEM floorplan, roofline bounds, and verdict tests for TPU advisor."""

from __future__ import annotations

import ast
import dataclasses
import math
from pathlib import Path
from typing import Literal

import pytest

from tests.megakernel.core.megakernel_fakes import TINY_TILES, make_fake_model_spec
from tests.megakernel.test_advisor_specs import (
    QWEN35_397B_CFG,
)
from vllm_torchtpu.megakernel import advisor
from vllm_torchtpu.megakernel.advisor import (
    constants,
    models,
    roofline,
    spec_bridge,
    specs,
    support,
)
from vllm_torchtpu.megakernel.advisor.roofline import (
    _build_scope_context,
    _compute_scope_matmuls,
    _core_time,
    _evaluate_scopes,
    _resolve_calibrated_priors,
    _routed_chain,
)
from vllm_torchtpu.megakernel.advisor.verdict import (
    _apply_box,
    _assumed_box,
    _evaluate_corners,
)
from vllm_torchtpu.megakernel.advisor.vmem import _mesh_divisibility_reasons
from vllm_torchtpu.megakernel.core import runtime as core_runtime
from vllm_torchtpu.megakernel.core.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.core.exec_spec import MAX_BATCH_ROWS, ExecSpec, Tiles
from vllm_torchtpu.megakernel.core.profile import _NUMERICS_GRID_R

REPORTED_FLOOR_ROUNDING_STEP_US = 0.01
STRICT_VECTOR_DIFF_TOLERANCE_US = 2 * REPORTED_FLOOR_ROUNDING_STEP_US
ROW_BLOCKER = "rows_exceed_core_max_batch_rows"
ROWS_PER_VERIFY_REQUEST = 4
SPLIT_STEP_ROWS = 192
ROWS_ABOVE_DECLARED_TOTAL = 1000
ROW_OVERRIDE = 64
ROWS_ABOVE_ROW_OVERRIDE = 96
LONG_PREFILL_TOKENS = 8192
SHORT_PREFILL_TOKENS = 512
WEIGHT_PASSES_UPPER_LIMIT = 2
UNBOUNDED_RATE = 1e20
SLOW_VECTOR_WORD_RATE_PER_S = 1e10
MIN_RESOLVABLE_VECTOR_TERM_US = 100 * STRICT_VECTOR_DIFF_TOLERANCE_US


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


@pytest.fixture
def without_declared_megakernels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip deployment `megakernel` blocks to test core-limit mechanics."""
    for model_key, preset in list(advisor.DEPLOYMENT_PRESETS.items()):
        stripped_sat = (
            dataclasses.replace(preset.saturation, megakernel=None)
            if preset.saturation is not None
            else None
        )
        monkeypatch.setitem(
            advisor.DEPLOYMENT_PRESETS,
            model_key,
            dataclasses.replace(
                preset,
                decode=dataclasses.replace(preset.decode, megakernel=None),
                prefill=dataclasses.replace(preset.prefill, megakernel=None),
                saturation=stripped_sat,
                megakernel_layouts=(),
            ),
        )


def test_per_token_vmem_live_interval_and_sharding() -> None:
    """Verify per-token live VMEM interval math and TP/EP sharding on Kimi-K3."""
    pk3 = advisor.resolve_model_profile("kimi-k3")
    wl_grp = advisor.WorkloadSpec(
        phase="verify",
        batch=16,
        spec_tokens=3,
        tp=32,
        ep=32,
        moe_mode="grouped_on_chip",
    )
    parts_grp = advisor.compute_per_token_vmem_bytes(pk3, wl_grp)
    expected_grp_tok = parts_grp["residual_buffer_bytes"] + max(
        parts_grp["attn_scratch_bytes"],
        parts_grp["linear_attn_scratch_bytes"],
        parts_grp["moe_scratch_bytes"],
    )
    assert parts_grp["total_per_token_bytes"] == expected_grp_tok
    assert parts_grp["b_tok"] == expected_grp_tok
    rep_grp = advisor.advise(pk3, wl_grp, "tpu7x")
    cross_sc = next(s for s in rep_grp.scopes if s.scope == "whole_step")
    one_l_sc = next(s for s in rep_grp.scopes if s.scope == "single_layer")
    min_verify_rows = 64  # Minimum verify tokens for Qwen 3.5
    assert cross_sc.max_vmem_tokens >= min_verify_rows
    assert one_l_sc.max_vmem_tokens > cross_sc.max_vmem_tokens

    wl_msk = dataclasses.replace(wl_grp, moe_mode="masked_on_chip")
    parts_msk = advisor.compute_per_token_vmem_bytes(pk3, wl_msk)
    assert parts_grp["total_per_token_bytes"] < parts_msk["total_per_token_bytes"]

    wl_no_alias = dataclasses.replace(wl_grp, vmem_alias=False)
    parts_no_alias = advisor.compute_per_token_vmem_bytes(pk3, wl_no_alias)
    assert parts_no_alias["total_per_token_bytes"] > parts_grp["total_per_token_bytes"]


def test_multi_tpu_vmem_and_fp8_dtype_physics() -> None:
    """Verify TPU FP8 physics, VMEM buffer overflow, Mixtral MoE, and DeepSeek MLA."""
    c7x = advisor.resolve_chip_spec("tpu7x")
    cv6e = advisor.resolve_chip_spec("v6e")

    tpu7x_fp8_flops = 2.30e15  # TPU v7x FP8 peak FLOP/s
    tpu7x_fp8_row_floor = 32  # TPU v7x FP8 row floor
    v6e_fp8_flops = 9.20e14  # TPU v6e FP8 peak FLOP/s

    assert (c7x.native_ocp_fp8, c7x.peak_flops("fp8"), c7x.lhs_row_floor("fp8")) == (
        True,
        tpu7x_fp8_flops,
        tpu7x_fp8_row_floor,
    )
    assert (cv6e.native_ocp_fp8, cv6e.peak_flops("fp8")) == (
        False,
        v6e_fp8_flops,
    )
    assert (
        c7x.vmem_usable_bytes,
        cv6e.vmem_usable_bytes,
    ) == (62 << 20, 126 << 20)

    p397 = advisor.model_profile_from_hf_config(QWEN35_397B_CFG, name="qwen35-397b")
    rep_v6e_tp8 = advisor.advise(p397, advisor.WorkloadSpec(tp=8, batch=4), cv6e)
    assert rep_v6e_tp8.hbm_fits is False

    pglm = advisor.resolve_model_profile("glm-5.2")
    expected_glm_moe_layers = 75
    expected_glm_moe_inter = 2048
    assert pglm.num_moe_layers == expected_glm_moe_layers
    assert pglm.moe_intermediate_size == expected_glm_moe_inter
    with pytest.raises(advisor.UnsupportedArchitectureError):
        advisor.model_profile_from_hf_config(
            {
                "model_type": "mixtral",
                "num_hidden_layers": 32,
                "hidden_size": 4096,
                "intermediate_size": 14336,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
            }
        )


def test_deepseek_v3_mla_kv_split_physics() -> None:
    """Verify MLA parameter sizing and kv_split HBM savings on GLM-5.2."""
    c7x = advisor.resolve_chip_spec("tpu7x")
    pdsv3 = advisor.resolve_model_profile("glm-5.2")
    expected_head_dim = 256
    expected_shared_inter = 2048
    expected_moe_layers = 75
    assert pdsv3.head_dim == expected_head_dim
    assert pdsv3.shared_expert_intermediate_size == expected_shared_inter
    assert pdsv3.num_moe_layers == expected_moe_layers
    assert pdsv3.is_periodic is False
    pdsv3_dense_only = dataclasses.replace(
        pdsv3,
        num_moe_layers=0,
        num_experts=0,
        top_k=0,
        intermediate_size=0,
        index_topk=0,
        index_head_dim=0,
        index_n_heads=0,
        num_indexer_layers=0,
        layer_families=("mla",),
    )
    r_mla_only = advisor.advise(
        pdsv3_dense_only, advisor.WorkloadSpec(tp=8, batch=1, context=4096), c7x
    )
    nh_loc = pdsv3_dense_only.num_attention_heads // 8
    q_tot = nh_loc * (
        pdsv3_dense_only.qk_nope_head_dim + pdsv3_dense_only.qk_rope_head_dim
    )
    q_dim = nh_loc * pdsv3_dense_only.head_dim
    mla_layer_params = (
        pdsv3_dense_only.q_lora_rank * (pdsv3_dense_only.hidden_size + q_tot)
        + pdsv3_dense_only.hidden_size
        * (pdsv3_dense_only.kv_lora_rank + pdsv3_dense_only.qk_rope_head_dim + q_dim)
        + pdsv3_dense_only.kv_lora_rank
        * nh_loc
        * (pdsv3_dense_only.qk_nope_head_dim + pdsv3_dense_only.head_dim)
        + 3 * pdsv3_dense_only.hidden_size
    )
    expected_mla_w_gb = round((pdsv3_dense_only.num_layers * mla_layer_params) / 1e9, 4)
    assert r_mla_only.scopes[0].w_step_gb == expected_mla_w_gb
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
    assert [
        (s.feasible, s.support, "family_not_in_core" in s.blockers)
        for s in r_ds1.scopes
        if s.scope == "whole_step"
    ] == [(True, "hypothetical", True)]
    verify_batch, verify_spec = 8, 3
    mixed_chunk = 128
    r_ver = advisor.advise(
        pdsv3,
        advisor.WorkloadSpec(
            phase="verify", tp=8, batch=verify_batch, spec_tokens=verify_spec
        ),
        c7x,
    )
    r_mix = advisor.advise(
        pdsv3,
        advisor.WorkloadSpec(
            phase="mixed",
            tp=8,
            batch=verify_batch,
            spec_tokens=verify_spec,
            chunk_size=mixed_chunk,
        ),
        c7x,
    )
    assert r_ver.active_tokens == verify_batch * (verify_spec + 1)
    assert r_mix.active_tokens == verify_batch * (verify_spec + 1) + mixed_chunk


def test_asymptotic_decode_intensity_and_vmem_fit_headroom() -> None:
    """Verify `asymptotic_decode_intensity`, `fit_vmem_budget_bytes`, `w_crit_gb`."""
    assert advisor.fit_vmem_budget_bytes() == 62 << 20
    assert advisor.fit_vmem_budget_bytes(Tiles(vmem_mib=62)) == 62 << 20
    p08 = models.finalize_model_profile(
        dataclasses.replace(
            advisor.resolve_model_profile("qwen3.5-397b"),
            name="qwen3.5-397b-small-state",
            num_layers=24,
            num_attention_layers=6,
            num_linear_layers=18,
            hidden_size=1024,
            intermediate_size=3584,
            num_attention_heads=8,
            num_kv_heads=2,
            linear_key_heads=16,
            linear_value_heads=16,
            num_experts=0,
            top_k=0,
            num_moe_layers=0,
            moe_intermediate_size=0,
            shared_expert_intermediate_size=0,
            has_attn_output_gate=False,
            weight_block_k=0,
            state_dtype="fp32",
            layer_families=("gqa", "gdn", "dense_mlp"),
        )
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
    # 64 MiB physical minus 2 MiB reserve
    expected_vmem_mib = float(constants.DEFAULT_USABLE_VMEM_MIB)
    assert r_alias.vmem_budget_mib == expected_vmem_mib
    assert r_no_alias.vmem_budget_mib == advisor.fit_vmem_budget_bytes() / (1 << 20)
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
    p397 = advisor.model_profile_from_hf_config(QWEN35_397B_CFG, name="qwen35-397b")
    p08 = models.finalize_model_profile(
        dataclasses.replace(
            p397,
            name="dense-small-gdn",
            num_layers=24,
            num_attention_layers=6,
            num_linear_layers=18,
            hidden_size=1024,
            intermediate_size=3584,
            num_attention_heads=8,
            num_kv_heads=2,
            linear_key_heads=16,
            linear_value_heads=16,
            num_experts=0,
            top_k=0,
            num_moe_layers=0,
            moe_intermediate_size=0,
            shared_expert_intermediate_size=0,
            has_attn_output_gate=False,
            weight_block_k=0,
            layer_families=("gqa", "gdn", "dense_mlp"),
        )
    )
    pllama = models.finalize_model_profile(
        advisor.ModelProfile(
            name="dense-large-gqa",
            num_layers=126,
            hidden_size=16384,
            intermediate_size=53248,
            num_attention_heads=128,
            num_kv_heads=8,
            head_dim=128,
            weight_dtype="fp8",
            expert_dtype="fp8",
        )
    )

    # 1. `whole_step` wins on 397B TP8 x EP8 decode and small dense prefill.
    r_dec = advisor.advise(
        p397, advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8), "tpu7x"
    )
    assert r_dec.recommended_scope == "whole_step"
    assert r_dec.recommended_moe_mode == "masked_on_chip"
    assert r_dec.build_megakernel is True
    r_b32 = advisor.advise(
        p397, advisor.WorkloadSpec(phase="decode", batch=32, tp=8, ep=8), "tpu7x"
    )
    assert r_b32.recommended_scope == "whole_step"
    assert r_b32.recommended_moe_mode == "grouped_on_chip"
    r_08_pre = advisor.advise(
        p08, advisor.WorkloadSpec(phase="prefill", batch=1, isl=2048, tp=8), "tpu7x"
    )
    assert r_08_pre.fastest_scope == "whole_step"
    assert r_08_pre.recommended_moe_mode == "dense_mlp"

    # 2. `single_layer` (`layer`) wins in the B=488 VMEM cliff band on tpu7x,
    # whereas v6e (TP16) and v5p Megacore (TP8) keep `whole_step` in 1 pass.
    r_cliff_7x = advisor.advise(
        p397,
        advisor.WorkloadSpec(phase="decode", batch=488, context=1024, tp=8),
        "tpu7x",
    )
    assert r_cliff_7x.fastest_scope == "single_layer"
    assert r_cliff_7x.recommended_moe_mode == "grouped_on_chip"
    r_cliff_v6e = advisor.advise(
        p397,
        advisor.WorkloadSpec(phase="decode", batch=488, context=1024, tp=16),
        "v6e",
    )
    assert r_cliff_v6e.fastest_scope == "whole_step"
    r_ep8_sat = advisor.advise(
        p397,
        advisor.WorkloadSpec(
            phase="decode", batch=600, tp=2, dp=4, ep=8, tiles=Tiles()
        ),
        "tpu7x",
    )
    # Saturated DP4 x TP2 decode with locked wide ring: 150 tok/core overflow
    # the wide whole-step ceiling while single_layer fits in 1 pass.
    assert r_ep8_sat.fastest_scope == "single_layer"
    assert r_ep8_sat.recommended_moe_mode == "grouped_on_chip"

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
    assert r_stg.fastest_scope == "layer_group"
    r_seg = advisor.advise(
        p397,
        advisor.WorkloadSpec(phase="decode", batch=16, tp=8, segment_layers=10),
        "tpu7x",
    )
    assert r_seg.fastest_scope == "layer_group"
    # 6 segments of 10 layers across 3 ops per segment
    expected_ops_per_step = (p397.num_layers // 10) * 3
    assert (
        next(s for s in r_seg.scopes if s.scope == "layer_group").ops_per_step
        == expected_ops_per_step
    )
    r_16k = advisor.advise(
        p397,
        advisor.WorkloadSpec(phase="prefill", batch=8, isl=16384, tp=1, pp=8),
        "tpu7x",
    )
    assert r_16k.recommended_scope == "per_op"
    assert r_16k.build_megakernel is False


@pytest.mark.usefixtures("without_declared_megakernels")
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
        advisor.WorkloadSpec(tp=8, ep=8, batch=16, dma_buffers=2, vmem_budget_mib=58.0),
        "tpu7x",
    )
    cx_2 = next(s for s in rep_2buf.scopes if s.scope == "whole_step")
    cx_3 = next(s for s in rep_3buf.scopes if s.scope == "whole_step")
    cx_vmem = next(s for s in rep_custom_vmem.scopes if s.scope == "whole_step")
    # 64 MiB physical minus 2 MiB reserve
    expected_vmem_mib = float(constants.DEFAULT_USABLE_VMEM_MIB)
    custom_vmem_mib = 58.0  # Custom VMEM budget in MiB
    assert rep_2buf.vmem_budget_mib == expected_vmem_mib
    assert rep_custom_vmem.vmem_budget_mib == custom_vmem_mib
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
    workload = advisor.WorkloadSpec(
        phase="prefill", batch=1, isl=4096, tp=tp, ep=ep, pcp=pcp
    )
    if accepted:
        report = advisor.advise(p397, workload, "tpu7x")
        assert report.active_tokens == 4096 // pcp
    else:
        with pytest.raises(MegakernelConfigError):
            advisor.advise(p397, workload, "tpu7x")


def test_advise_from_model_spec_and_planner() -> None:
    """Verify `advise_from_spec` bridges periodic and non-periodic `ModelSpec`."""
    ms = make_fake_model_spec(L=4, att_every=4, H=256)
    ex = ExecSpec(B=4, T=1, R=2, ctx=1024, tiles=TINY_TILES)
    report = advisor.advise_from_spec(
        ms, ex, chip="tpu7x", moe_mode="grouped_on_chip", dp=4
    )
    assert report.recommended_scope == "whole_step"
    rep_def = advisor.advise_from_spec(
        ms,
        ExecSpec(B=4, T=1, R=2, ctx=1024),
        chip="tpu7x",
        moe_mode="grouped_on_chip",
        dp=4,
    )
    # 64 MiB physical minus 2 MiB reserve
    expected_vmem_mib = float(constants.DEFAULT_USABLE_VMEM_MIB)
    assert rep_def.recommended_scope == "whole_step"
    assert rep_def.vmem_budget_mib == expected_vmem_mib

    ms_non_periodic = make_fake_model_spec(L=5, att_every=4, H=256)
    rep_np = advisor.advise_from_spec(
        ms_non_periodic, ex, chip="tpu7x", moe_mode="grouped_on_chip", dp=4
    )
    cross = next(s for s in rep_np.scopes if s.scope == "whole_step")
    assert cross.feasible is False
    assert rep_np.fastest_scope == "single_layer"
    assert rep_np.recommended_scope == "per_op"


def test_spec_bridge_jax_isolation_ast() -> None:
    """Verify spec_bridge.py isolates jax and planner imports inside ast.Try."""
    bridge_path = Path(advisor.__file__).with_name("spec_bridge.py")
    tree = ast.parse(bridge_path.read_text("utf-8"), filename=str(bridge_path))
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names]
            mod = getattr(node, "module", "") or ""
            assert not any("jax" in n for n in names), f"Unguarded jax import: {names}"
            assert "jax" not in mod, f"Unguarded jax module import: {mod}"
            assert not any("planner" in n for n in names), (
                f"Unguarded planner import: {names}"
            )
    try_nodes = [n for n in tree.body if isinstance(n, ast.Try)]
    assert any(
        any(
            (
                isinstance(h.type, ast.Tuple)
                and any(
                    getattr(e, "id", "") in ("ImportError", "RuntimeError")
                    for e in h.type.elts
                )
            )
            or getattr(h.type, "id", "") in ("ImportError", "RuntimeError")
            for h in t.handlers
        )
        for t in try_nodes
    ), "spec_bridge.py must have ast.Try catching (ImportError, RuntimeError)"


def test_advise_from_spec_raises_when_planner_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify advise_from_spec raises MegakernelConfigError when planner is missing."""
    monkeypatch.setattr(spec_bridge, "planner", None)
    ms = make_fake_model_spec(L=4, att_every=4, H=256)
    ex = ExecSpec(B=4, T=1, R=2, ctx=1024, tiles=TINY_TILES)
    with pytest.raises(MegakernelConfigError, match="Megakernel planner requires JAX"):
        advisor.advise_from_spec(ms, ex)


def test_qwen35_decode_preset_whole_step_admits_shipped_verify_rows() -> None:
    """Verify whole_step fits 64 verify rows and reports the ideal VMEM ceiling."""
    dep = advisor.DEPLOYMENT_PRESETS["qwen3.5-397b"]
    rows_per_seq = dep.decode.spec_tokens + 1
    report = advisor.advise(
        advisor.resolve_model_profile("qwen3.5-397b"),
        _preset_workload(
            dep.decode,
            phase="verify",
            batch=64 // rows_per_seq,  # Minimum verify rows for Qwen 3.5
            spec_tokens=dep.decode.spec_tokens,
            moe_mode="grouped_on_chip",
        ),
        "tpu7x",
    )
    whole = next(s for s in report.scopes if s.scope == "whole_step")
    rows_cap = rows_per_seq * (whole.t_vmem_max // rows_per_seq)
    min_verify_rows = 64  # Minimum verify rows for Qwen 3.5
    declared_qwen35_call_rows = 112
    assert rows_cap >= min_verify_rows
    assert whole.t_vmem_max == declared_qwen35_call_rows


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_kimi_k3_decode_preset_low_batch_prefers_whole_step(batch: int) -> None:
    """Verify Kimi-K3 decode preset recommends declared whole_step at low batch."""
    dep = advisor.DEPLOYMENT_PRESETS["kimi-k3"]
    report = advisor.advise(
        advisor.resolve_model_profile("kimi-k3"),
        _preset_workload(dep.decode, phase="decode", batch=batch),
        "tpu7x",
    )
    whole = next(s for s in report.scopes if s.scope == "whole_step")
    assert report.fastest_scope == "whole_step"
    assert report.recommended_scope == "whole_step"
    assert whole.support == "declared"
    assert report.build_megakernel is True


@pytest.mark.parametrize("window", [2048, 8192])
def test_kimi_k3_prefill_preset_evaluates_layer_group(window: int) -> None:
    """Verify the shipped 13-call layer grouping is feasible and evaluated."""
    dep = advisor.DEPLOYMENT_PRESETS["kimi-k3"]
    report = advisor.advise(
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
    # 13 declared groups in Kimi-K3 prefill preset
    expected_prefill_calls = len(
        advisor.DEPLOYMENT_PRESETS["kimi-k3"].prefill.layer_groups
    )
    group = next(s for s in report.scopes if s.scope == "layer_group")
    assert group.feasible
    assert group.ops_per_step == expected_prefill_calls
    assert report.active_tokens == window // dep.prefill.pcp


def test_layer_groups_must_cover_every_layer() -> None:
    """Verify a layer grouping that does not sum to num_layers is rejected."""
    with pytest.raises(MegakernelConfigError):
        advisor.advise(
            advisor.resolve_model_profile("kimi-k3"),
            advisor.WorkloadSpec(tp=32, ep=32, layer_groups=(8, 8)),
            "tpu7x",
        )


@pytest.mark.usefixtures("without_declared_megakernels")
def test_whole_step_tiles_overflow_and_reads_weights_once() -> None:
    """Verify whole_step tiles tokens past the ceiling instead of re-passing."""
    dep = advisor.DEPLOYMENT_PRESETS["qwen3.5-397b"]
    report = advisor.advise(
        advisor.resolve_model_profile("qwen3.5-397b"),
        _preset_workload(
            dep.decode,
            phase="verify",
            batch=(448 // (dep.decode.spec_tokens + 1))
            + 4,  # Ideal VMEM rows for Qwen 3.5
            spec_tokens=dep.decode.spec_tokens,
            moe_mode="grouped_on_chip",
        ),
        "tpu7x",
    )
    tiled = next(s for s in report.scopes if s.scope == "whole_step")
    one_l = next(s for s in report.scopes if s.scope == "single_layer")
    assert tiled.n_splits > 1
    assert tiled.feasible
    assert tiled.w_step_gb == one_l.w_step_gb
    assert tiled.step_bound_us < 1.5 * one_l.step_bound_us


def test_fused_scopes_price_the_recommended_moe_mode() -> None:
    """Verify a full row tile per EP rank prices on-chip grouping either way."""
    model_profile = advisor.resolve_model_profile("qwen3.5-397b")
    steps = []
    for mode in ("masked_on_chip", "grouped_on_chip"):
        report = advisor.advise(
            model_profile,
            advisor.WorkloadSpec(phase="decode", batch=64, tp=8, ep=8, moe_mode=mode),
            "tpu7x",
        )
        steps.append(next(s for s in report.scopes if s.scope == "whole_step"))
        assert report.recommended_moe_mode == "grouped_on_chip"
    assert steps[0].moe_flop_factor == steps[1].moe_flop_factor


def test_intra_host_dp_grouped_moe_consistency() -> None:
    """Verify intra-host DPExchange groups MoE while multi-host DCN stays ungrouped."""
    grouped_phi_ceiling = 5.0
    ungrouped_phi_floor = 10.0
    model_profile = dataclasses.replace(
        advisor.resolve_model_profile("qwen3.5-397b"), weight_block_k=0
    )
    rep_grp = advisor.advise(
        model_profile,
        advisor.WorkloadSpec(
            phase="decode",
            batch=600,
            context=32768,
            tp=2,
            dp=4,
            ep=8,
            moe_mode="grouped_on_chip",
        ),
        "tpu7x",
    )
    ws_grp = next(s for s in rep_grp.scopes if s.scope == "whole_step")
    assert ws_grp.moe_flop_factor < grouped_phi_ceiling
    assert rep_grp.fastest_scope == "whole_step"
    assert rep_grp.recommended_scope == "per_op"
    assert rep_grp.recommended_moe_mode == "grouped_on_chip"
    assert float(rep_grp.metrics["achievable_gain_pct"]["value"]) > 0.0

    rep_msk = advisor.advise(
        model_profile,
        advisor.WorkloadSpec(
            phase="decode",
            batch=600,
            context=32768,
            tp=2,
            dp=4,
            ep=8,
            moe_mode="masked_on_chip",
        ),
        "tpu7x",
    )
    ws_msk = next(s for s in rep_msk.scopes if s.scope == "whole_step")
    assert ws_msk.moe_flop_factor > ungrouped_phi_floor
    assert rep_msk.recommended_moe_mode != "grouped_on_chip"

    rep_dcn = advisor.advise(
        model_profile,
        advisor.WorkloadSpec(
            phase="decode",
            batch=600,
            context=32768,
            tp=2,
            dp=8,
            ep=16,
            moe_mode="grouped_on_chip",
        ),
        "tpu7x",
    )
    ws_dcn = next(s for s in rep_dcn.scopes if s.scope == "whole_step")
    assert ws_dcn.moe_flop_factor > ungrouped_phi_floor
    assert rep_dcn.recommended_moe_mode != "grouped_on_chip"


def test_block_scaled_ungrouped_dp_prefers_grouped_single_layer() -> None:
    """Verify block-scaled DP stays ungrouped whole-step and picks single-layer."""
    grouped_phi_ceiling = 5.0
    ungrouped_phi_floor = 10.0
    model_profile = advisor.resolve_model_profile("qwen3.5-397b")
    assert model_profile.weight_block_k > 0
    rep_grp = advisor.advise(
        model_profile,
        advisor.WorkloadSpec(
            phase="decode",
            batch=600,
            context=32768,
            tp=2,
            dp=4,
            ep=8,
            moe_mode="grouped_on_chip",
        ),
        "tpu7x",
    )
    ws_grp = next(s for s in rep_grp.scopes if s.scope == "whole_step")
    assert ws_grp.moe_flop_factor < grouped_phi_ceiling
    assert rep_grp.fastest_scope == "whole_step"
    assert rep_grp.recommended_scope == "per_op"
    assert rep_grp.recommended_moe_mode == "grouped_on_chip"
    assert float(rep_grp.metrics["achievable_gain_pct"]["value"]) > 0.0

    rep_msk = advisor.advise(
        model_profile,
        advisor.WorkloadSpec(
            phase="decode",
            batch=600,
            context=32768,
            tp=2,
            dp=4,
            ep=8,
            moe_mode="masked_on_chip",
        ),
        "tpu7x",
    )
    ws_msk = next(s for s in rep_msk.scopes if s.scope == "whole_step")
    assert ws_msk.moe_flop_factor > ungrouped_phi_floor
    # An ungrouped whole step pays block-scale work on every padded row, so
    # the grouped single-layer scope is faster.
    assert rep_msk.fastest_scope == "single_layer"
    assert rep_msk.recommended_moe_mode == "grouped_on_chip"

    rep_dcn = advisor.advise(
        model_profile,
        advisor.WorkloadSpec(
            phase="decode",
            batch=600,
            context=32768,
            tp=2,
            dp=8,
            ep=16,
            moe_mode="grouped_on_chip",
        ),
        "tpu7x",
    )
    ws_dcn = next(s for s in rep_dcn.scopes if s.scope == "whole_step")
    assert ws_dcn.moe_flop_factor > ungrouped_phi_floor
    # An ungrouped whole step pays block-scale work on every padded row, so
    # the grouped single-layer scope is faster.
    assert rep_dcn.fastest_scope == "single_layer"
    assert rep_dcn.recommended_moe_mode == "grouped_on_chip"


def test_unchunked_pp_prefill_preserves_window() -> None:
    """Verify unchunked B=1 prefill under PP>1 preserves its full stage window."""
    model_profile = advisor.resolve_model_profile("qwen3.5-397b")
    isl_tokens = 4096
    rep_single = advisor.advise(
        model_profile,
        advisor.WorkloadSpec(
            phase="prefill",
            batch=1,
            isl=isl_tokens,
            tp=1,
            pp=8,
            ep=1,
            chunk_size=0,
            microbatches=0,
        ),
        "tpu7x",
    )
    assert rep_single.active_tokens == isl_tokens
    assert rep_single.recommended_scope == "per_op"
    # Block-scaled FP8 weights make one full-window single-layer call slower
    # than per_op, so no fused scope has a positive compiler-proof gain.
    assert rep_single.fastest_scope == "per_op"
    assert rep_single.recommended_moe_mode == "grouped_in_hbm"
    per_op_sc = next(s for s in rep_single.scopes if s.scope == "per_op")
    single_sc = next(s for s in rep_single.scopes if s.scope == "single_layer")
    assert single_sc.step_bound_us > per_op_sc.step_bound_us
    assert float(rep_single.metrics["compiler_proof_gain_pct"]["value"]) == 0.0

    rep_half_window = advisor.advise(
        model_profile,
        advisor.WorkloadSpec(
            phase="prefill",
            batch=1,
            isl=isl_tokens // 2,
            tp=1,
            pp=8,
            ep=1,
            chunk_size=0,
            microbatches=0,
        ),
        "tpu7x",
    )
    assert rep_half_window.active_tokens == isl_tokens // 2
    assert rep_half_window.fastest_scope == "single_layer"
    per_op_half = next(
        scope for scope in rep_half_window.scopes if scope.scope == "per_op"
    )
    single_half = next(
        scope for scope in rep_half_window.scopes if scope.scope == "single_layer"
    )
    pure_diff_us = per_op_half.step_bound_us - single_half.step_bound_us
    expected_gain_pct = round(
        100.0 * pure_diff_us / max(1e-6, per_op_half.step_bound_us), 2
    )
    assert (
        float(rep_half_window.metrics["compiler_proof_gain_pct"]["value"])
        == expected_gain_pct
    )
    assert single_half.step_upper_us < per_op_half.step_upper_us

    rep_microbatched = advisor.advise(
        model_profile,
        advisor.WorkloadSpec(
            phase="prefill",
            batch=8,
            isl=isl_tokens,
            tp=1,
            pp=8,
            ep=1,
            chunk_size=0,
            microbatches=8,
        ),
        "tpu7x",
    )
    assert rep_microbatched.active_tokens == isl_tokens


def test_core_limits_literals_match_megakernel_core() -> None:
    """Verify CORE_LIMITS matches megakernel/core/exec_spec.py and profile.py."""
    tiles_default = Tiles()
    lim = advisor.CORE_LIMITS
    assert lim.max_batch_rows == MAX_BATCH_ROWS
    assert lim.verified_ranks == _NUMERICS_GRID_R

    assert lim.max_batch_rows == constants.CORE_LIMITS.max_batch_rows
    assert lim.verified_ranks == constants.CORE_LIMITS.verified_ranks
    assert lim.chip_kinds == constants.CORE_LIMITS.chip_kinds
    assert lim.max_ar_slots_per_layer == constants.CORE_LIMITS.max_ar_slots_per_layer
    assert (lim.nbd, lim.nbe, lim.bank, lim.ebank, lim.vmem_mib) == (
        tiles_default.nbd,
        tiles_default.nbe,
        tiles_default.bank,
        tiles_default.ebank,
        tiles_default.vmem_mib,
    )


@pytest.mark.usefixtures("without_declared_megakernels")
@pytest.mark.parametrize(
    ("workload", "chip", "expected_blocker"),
    [
        (
            advisor.WorkloadSpec(phase="decode", batch=256, tp=8, ep=8, dp=1),
            "tpu7x",
            "rows_exceed_core_max_batch_rows",
        ),
        (
            advisor.WorkloadSpec(phase="decode", batch=64, tp=2, ep=8, dp=4),
            "tpu7x",
            "all_to_all_not_in_core",
        ),
        (
            advisor.WorkloadSpec(phase="decode", batch=16, tp=32, ep=32, dp=1),
            "tpu7x",
            "rank_count_outside_core_grid",
        ),
        (
            advisor.WorkloadSpec(phase="decode", batch=16, tp=16, ep=16, dp=1),
            "v6e",
            "chip_without_core_budget",
        ),
    ],
)
def test_no_fused_recommendation_outside_core_limits(
    workload: advisor.WorkloadSpec,
    chip: str,
    expected_blocker: str,
) -> None:
    """Verify >128 rows, ep_dp>1, 32 ranks, or non-TPU7x chips gate to per_op."""
    model_profile = advisor.resolve_model_profile("qwen3.5-397b")
    report = advisor.advise(model_profile, workload, chip)
    whole = next(s for s in report.scopes if s.scope == "whole_step")
    assert whole.support == "hypothetical"
    assert expected_blocker in whole.blockers
    assert report.recommended_scope == "per_op"
    assert report.build_megakernel is False
    assert report.fastest_scope in ("whole_step", "single_layer", "layer_group")
    assert report.hypothetical_best is not None


def test_in_slice_all_reduce_uses_ici_for_every_rank_count() -> None:
    """Verify TP32 All-Reduce uses single-slice ICI rather than DCN (Ledger Row 2)."""
    chip = advisor.resolve_chip_spec("tpu7x")
    msg_bytes = 64 * 8192 * 2
    t_trans, t_lat, hops, _ = advisor.compute_all_reduce_transfer_cost(
        msg_bytes, 32, chip
    )
    expected_hops = 1 + 2 + 2  # 5 hops across 2x2x4 3D torus for 32 cores (16 chips)
    assert hops == expected_hops
    # 2 * (31 / 32) * 1 MiB over one 100 GB/s link is 20.3162 us, plus 5 hops of 1 us.
    expected_ar_cost_us = round(
        (
            2.0 * (31.0 / 32.0) * msg_bytes / chip.ici_link_bw_bytes_per_s
            + hops * chip.collective_latency_s
        )
        * 1e6,
        4,
    )
    assert round((t_trans + t_lat) * 1e6, 4) == expected_ar_cost_us


def test_all_reduce_transfer_uses_single_link() -> None:
    """Verify all-reduce transfer is priced on one ICI link, not the egress sum."""
    chip = advisor.resolve_chip_spec("tpu7x")
    msg_bytes = 64 * 8192 * 2
    t_trans, _, _, volume = advisor.compute_all_reduce_transfer_cost(msg_bytes, 8, chip)
    assert t_trans == pytest.approx(volume / chip.ici_link_bw_bytes_per_s)
    assert t_trans > volume / chip.ici_egress_bw_bytes_per_s


@pytest.mark.parametrize("batch", [16, 512])
def test_strict_floor_prices_all_reduce_on_egress(batch: int) -> None:
    """Verify the strict floor ignores link bandwidth while the floor pays it."""
    model_profile = advisor.resolve_model_profile("qwen3.5-397b")
    workload = advisor.WorkloadSpec(phase="decode", batch=batch, tp=8, ep=8)
    chip = advisor.resolve_chip_spec("tpu7x")
    slow_chip = dataclasses.replace(
        chip, ici_link_bw_bytes_per_s=chip.ici_link_bw_bytes_per_s / 2
    )
    fast, slow = (
        next(
            s
            for s in advisor.advise(model_profile, workload, spec).scopes
            if s.scope == "whole_step"
        )
        for spec in (chip, slow_chip)
    )
    multi_pass_batch = 512
    assert (fast.n_splits > 1) == (batch == multi_pass_batch)
    assert slow.step_floor_strict_us == pytest.approx(fast.step_floor_strict_us)
    assert slow.step_bound_us > fast.step_bound_us


def test_all_to_all_fanout_capped_by_destination_groups() -> None:
    """Verify MoE All-to-All fanout caps at min(top_k, ep_dp) (Ledger Row 3)."""
    model_profile = advisor.resolve_model_profile("qwen3.5-397b")
    chip = advisor.resolve_chip_spec("tpu7x")
    workload = advisor.WorkloadSpec(
        phase="decode", batch=2400, tp=2, dp=4, ep=8, moe_mode="grouped_on_chip"
    )
    scope_context = _build_scope_context(model_profile, workload, chip)
    assert scope_context.ep_dp == workload.ep // workload.tp
    # A2A transfer priced at egress bandwidth across destination groups.
    expected_a2a_us = (
        2.0
        * (scope_context.ep_dp - 1)
        / scope_context.ep_dp
        * min(max(1, model_profile.top_k), scope_context.ep_dp)
        * scope_context.token_residual_bytes
        / chip.ici_egress_bw_bytes_per_s
    ) * 1e6
    assert scope_context.a2a_time * 1e6 == pytest.approx(expected_a2a_us, abs=0.01)


def test_strict_floor_not_above_uniform_estimate() -> None:
    """Verify strict balanced routing floor vs uniform coupon-collector (Row 4)."""
    model_profile = advisor.resolve_model_profile("qwen3.5-397b")
    workload = advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8)
    report = advisor.advise(
        model_profile,
        workload,
        "tpu7x",
    )
    whole = next(s for s in report.scopes if s.scope == "whole_step")
    chip = advisor.resolve_chip_spec("tpu7x")
    # Strict floor weights for Qwen3.5-397B are non-zero and below total weights.
    assert 0.0 < whole.w_step_strict_gb < whole.w_step_gb
    # The strict floor incorporates routed compute chain, egress collectives,
    # and pipeline fill/drain. It strictly exceeds the pure HBM stream time
    # of strict weights (586.64 us) and sits below step_bound_us (4192.01 us).
    raw_hbm_floor_us = round(
        ((whole.w_step_strict_gb * 1e9) / chip.hbm_bw_bytes_per_s) * 1e6, 2
    )
    assert whole.step_floor_strict_us > raw_hbm_floor_us
    assert 0.0 < whole.step_floor_strict_us < whole.step_bound_us


def test_fused_core_is_max_of_hbm_and_compute_plus_comm() -> None:
    """Verify fused critical path uses max(hbm_time_s, t_matmul + comm_s)."""
    model_profile = advisor.resolve_model_profile("kimi-k3")
    report = advisor.advise(
        model_profile,
        advisor.WorkloadSpec(phase="decode", batch=256, tp=32, ep=32),
        "tpu7x",
    )
    whole = next(s for s in report.scopes if s.scope == "whole_step")
    assert whole.compute_us + whole.comm_us > whole.hbm_us
    assert whole.comm_us > 0.0
    assert whole.roof_floor_us == pytest.approx(
        max(whole.hbm_us, whole.compute_us + whole.comm_us) + whole.fill_drain_us,
        abs=0.05,
    )
    assert (
        whole.roof_floor_us
        > max(whole.compute_us, whole.hbm_us, whole.comm_us) + whole.fill_drain_us
    )


def test_core_time_routed_stream_cannot_hide_collectives() -> None:
    """Verify collectives add to the routed-expert chain, not under the HBM stream."""
    t_dense_s = 0.05
    t_moe_s = 0.10
    routed_hbm_s = 0.80
    hbm_s = 1.00
    comm_s = 0.30
    chain_s = _routed_chain(t_dense_s, routed_hbm_s, t_moe_s)
    assert chain_s == pytest.approx(t_dense_s + routed_hbm_s)
    fused_s = _core_time(hbm_s, chain_s, comm_s, fused=True)
    assert fused_s == pytest.approx(chain_s + comm_s)
    assert fused_s > max(hbm_s, t_dense_s + t_moe_s + comm_s)
    split_s = _core_time(hbm_s, chain_s, comm_s, fused=False)
    assert split_s == pytest.approx(hbm_s + comm_s)
    assert _core_time(hbm_s, chain_s, 0.0, fused=True) == hbm_s


def test_whole_step_bound_pays_collectives_after_router() -> None:
    """Verify a slower collective raises the fused MoE floor one for one."""
    model_profile = advisor.resolve_model_profile("qwen3.5-397b")
    workload = advisor.WorkloadSpec(
        phase="verify", batch=1, spec_tokens=3, context=4096, tp=8, dcp=4, ep=8
    )
    chip = advisor.resolve_chip_spec("tpu7x")
    slow_collective_us = 3.0
    slow_chip = dataclasses.replace(
        chip,
        collective_latency_s=slow_collective_us * 1e-6,
        collective_latency_us_bracket=(slow_collective_us, slow_collective_us),
    )
    wholes = [
        next(
            s
            for s in advisor.advise(model_profile, workload, spec).scopes
            if s.scope == "whole_step"
        )
        for spec in (chip, slow_chip)
    ]
    fast, slow = wholes
    fast_core_us = fast.roof_floor_us - fast.fill_drain_us
    slow_core_us = slow.roof_floor_us - slow.fill_drain_us
    assert fast.n_splits == slow.n_splits == 1
    assert fast_core_us > fast.hbm_us
    assert slow.comm_us > fast.comm_us
    assert slow_core_us - fast_core_us == pytest.approx(
        slow.comm_us - fast.comm_us, abs=0.05
    )


def test_achievable_upper_band_never_inverts_below_achievable_or_floor() -> None:
    """Verify step_achievable_upper_us never inverts below achievable or floor."""
    model_profile = advisor.resolve_model_profile("qwen3.5-397b")
    report = advisor.advise(
        model_profile,
        advisor.WorkloadSpec(
            phase="verify", batch=16, spec_tokens=3, context=4096, tp=8, dcp=4, ep=8
        ),
        "tpu7x",
    )
    whole = next(s for s in report.scopes if s.scope == "whole_step")
    assert whole.step_achievable_upper_us >= whole.step_achievable_us
    for scope in report.scopes:
        assert scope.step_achievable_upper_us >= max(
            scope.step_achievable_us, scope.step_upper_us
        )


def test_gain_split_sums_to_total_and_fact_part_ignores_priors() -> None:
    """Verify gain_fact_us and gain_assumed_us split and ASSUMED tag (Row 5)."""
    model_profile = advisor.resolve_model_profile("qwen3.5-397b")
    report = advisor.advise(
        model_profile,
        advisor.WorkloadSpec(
            phase="decode",
            batch=16,
            tp=8,
            dcp=4,
            ep=8,
        ),
        "tpu7x",
    )
    gain_fact_us = float(report.metrics["gain_fact_us"]["value"])
    gain_assumed_us = float(report.metrics["gain_assumed_us"]["value"])
    total_gain = float(report.metrics["compiler_proof_gain_us"]["value"])
    assert gain_fact_us > 0.0
    assert gain_assumed_us > 0.0
    assert total_gain == pytest.approx(gain_fact_us + gain_assumed_us, abs=0.01)
    assert report.metrics["compiler_proof_gain_us"]["tag"] == "ASSUMED"


def test_invalid_sharding_returns_invalid_config() -> None:
    """Verify non-power-of-two TP produces invalid_config verdict and per_op."""
    model_profile = advisor.resolve_model_profile("qwen3.5-397b")
    report = advisor.advise(
        model_profile,
        advisor.WorkloadSpec(phase="decode", batch=16, tp=3, ep=1, objective="both"),
        "tpu7x",
    )
    assert report.verdict.startswith("invalid_config(")
    assert report.fastest_verdict == report.verdict
    assert report.recommended_scope == "per_op"
    assert report.build_megakernel is False


def test_bundled_priors_keep_prior_only_confidence() -> None:
    """Verify architectural_prior bundled measurements keep prior_only confidence."""
    model_profile = advisor.resolve_model_profile("qwen3.5-397b")
    meas: str | dict[str, object] = (
        "bundled"
        if advisor.chips.public_calibration_path("tpu7x") is not None
        else {
            "schema": "advisor.measurements.v1",
            "chip": "tpu7x",
            "source": {"kind": "architectural_prior"},
        }
    )
    report = advisor.advise(
        model_profile,
        advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8),
        "tpu7x",
        measurements=meas,
    )
    assert report.confidence == "prior_only"


def _verify_whole_step(
    batch: int, max_rows_per_call: int, op_overhead_us: float = 0.0
) -> tuple[advisor.AdvisorReport, advisor.ScopeEstimate]:
    """Advise a Qwen3.5-397B verify step; return the report and its whole_step."""
    report = advisor.advise(
        advisor.resolve_model_profile("qwen3.5-397b"),
        advisor.WorkloadSpec(
            phase="verify",
            batch=batch,
            spec_tokens=3,  # rows_per_request = 1 prompt + 3 spec = 4
            context=4096,
            tp=8,
            dcp=4,
            ep=8,
            max_rows_per_call=max_rows_per_call,
            op_overhead_us=op_overhead_us,
        ),
        "tpu7x",
    )
    return report, next(s for s in report.scopes if s.scope == "whole_step")


def test_max_rows_per_call_clamps_single_call_rows() -> None:
    """Verify a declared row limit caps rows per call without changing the price."""
    _, ideal = _verify_whole_step(16, 0)
    _, capped = _verify_whole_step(16, constants.DEFAULT_ROW_LIMIT)
    assert ideal.t_vmem_max > constants.DEFAULT_ROW_LIMIT
    assert capped.t_vmem_max == constants.DEFAULT_ROW_LIMIT
    assert capped.n_splits == 1
    assert capped.step_bound_us == ideal.step_bound_us


def test_rows_above_single_call_limit_priced_as_microbatches() -> None:
    """Verify rows past the limit cost a full call plus the remainder call."""
    rows_per_req = 4
    full_requests = constants.DEFAULT_ROW_LIMIT // rows_per_req
    split_batch = 28
    _, split = _verify_whole_step(split_batch, constants.DEFAULT_ROW_LIMIT)
    _, full = _verify_whole_step(full_requests, constants.DEFAULT_ROW_LIMIT)
    _, rest = _verify_whole_step(
        split_batch - full_requests, constants.DEFAULT_ROW_LIMIT
    )
    _, tiled = _verify_whole_step(28, 0)
    assert split.n_splits == full.n_splits + rest.n_splits
    assert split.ops_per_step == full.ops_per_step + rest.ops_per_step
    assert split.step_bound_us == pytest.approx(
        full.step_bound_us + rest.step_bound_us, abs=0.02
    )
    assert split.step_achievable_us == pytest.approx(
        full.step_achievable_us + rest.step_achievable_us, abs=0.02
    )
    assert split.w_step_gb == pytest.approx(full.w_step_gb + rest.w_step_gb, abs=1e-3)
    assert split.step_bound_us > tiled.step_bound_us


def test_max_rows_per_call_splits_whole_requests() -> None:
    """Verify a limit between request boundaries rounds down to whole requests."""
    _, exact = _verify_whole_step(28, constants.DEFAULT_ROW_LIMIT)
    _, ragged = _verify_whole_step(28, constants.DEFAULT_ROW_LIMIT + 1)
    assert ragged.t_vmem_max == constants.DEFAULT_ROW_LIMIT
    assert ragged == exact


def test_max_rows_per_call_below_one_request_is_config_error() -> None:
    """Verify a limit smaller than one verify request is rejected."""
    with pytest.raises(MegakernelConfigError, match="below one verify request"):
        _verify_whole_step(16, 4 - 1)


def test_step_above_row_override_is_hypothetical_with_row_blocker() -> None:
    """Verify a split step above a row override is hypothetical with row blocker."""
    report, split = _verify_whole_step(28, constants.DEFAULT_ROW_LIMIT)
    assert split.support == "hypothetical"
    assert "rows_exceed_core_max_batch_rows" in split.blockers
    assert "microbatch_rerun" in report.flags
    assert "multi_call_unmodeled" not in report.flags
    rep_one, _ = _verify_whole_step(16, constants.DEFAULT_ROW_LIMIT)
    assert "microbatch_rerun" not in rep_one.flags


def test_microbatched_step_intensity_counts_weights_once() -> None:
    """Verify two equal full calls keep the arithmetic intensity of one call."""
    favoring_gap_us = 500.0
    rows_per_req = 4
    full_requests = constants.DEFAULT_ROW_LIMIT // rows_per_req
    two_calls = 2
    one, _ = _verify_whole_step(
        full_requests, constants.DEFAULT_ROW_LIMIT, favoring_gap_us
    )
    two, split = _verify_whole_step(
        two_calls * full_requests,
        constants.DEFAULT_ROW_LIMIT,
        favoring_gap_us,
    )
    assert one.fastest_scope == two.fastest_scope == "whole_step"
    assert split.n_splits == two_calls
    # Doubling the batch doubles both the FLOPs and the bytes of the step.
    assert two.step_arithmetic_intensity == pytest.approx(
        one.step_arithmetic_intensity, abs=0.02
    )


def test_in_call_passes_stay_flagged_under_row_limit() -> None:
    """Verify a call larger than the VMEM ceiling keeps the unmodeled-pass flag."""
    favoring_gap_us = 500.0
    rows_per_req = 4
    two_calls = 2
    qwen35_ideal_vmem_rows = 448
    limit = two_calls * qwen35_ideal_vmem_rows
    requests = (limit + qwen35_ideal_vmem_rows) // rows_per_req
    report, split = _verify_whole_step(requests, limit, favoring_gap_us)
    assert report.fastest_scope == "single_layer"
    # One full call of two ideal-ceiling passes, then a remainder call of one.
    assert split.n_splits == two_calls + 1
    assert {"microbatch_rerun", "multi_call_unmodeled"} <= set(report.flags)
    single_pass_calls, _ = _verify_whole_step(
        two_calls * (constants.DEFAULT_ROW_LIMIT // rows_per_req),
        constants.DEFAULT_ROW_LIMIT,
        favoring_gap_us,
    )
    assert single_pass_calls.fastest_scope == "whole_step"
    assert "multi_call_unmodeled" not in single_pass_calls.flags


def test_microbatch_zero_map_is_sum_of_calls() -> None:
    """Verify overridden latency priors reach every call of a split whole_step."""
    model_profile = advisor.resolve_model_profile("qwen3.5-397b")
    chip = advisor.resolve_chip_spec("tpu7x")

    def zero_latency_bound(batch: int) -> float:
        """Return the whole_step bound with every latency prior set to zero."""
        workload = advisor.WorkloadSpec(
            phase="verify",
            batch=batch,
            spec_tokens=3,  # rows_per_request = 1 prompt + 3 spec = 4
            context=4096,
            tp=8,
            dcp=4,
            ep=8,
            max_rows_per_call=constants.DEFAULT_ROW_LIMIT,
        )
        scope_context = dataclasses.replace(
            _build_scope_context(model_profile, workload, chip),
            lat_coll=0.0,
            lat_coll_upper=0.0,
            ar_base_lat_s=0.0,
            ar_inter_latency_s=0.0,
            launch_gap_upper_us=0.0,
        )
        estimates, _ = _evaluate_scopes(scope_context, model_profile, workload)
        return next(e for e in estimates if e.scope == "whole_step").step_bound_us

    rows_per_req = 4
    full_requests = constants.DEFAULT_ROW_LIMIT // rows_per_req
    split_batch = 28
    split_us = zero_latency_bound(split_batch)
    assert split_us == pytest.approx(
        zero_latency_bound(full_requests)
        + zero_latency_bound(split_batch - full_requests),
        abs=0.02,
    )
    _, priced = _verify_whole_step(split_batch, constants.DEFAULT_ROW_LIMIT)
    assert split_us < priced.step_bound_us


def test_declared_preset_status_reachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify a deployments.json megakernel declaration yields declared support."""
    dep = advisor.DEPLOYMENT_PRESETS["qwen3.5-397b"]
    declaration = {
        "source": "unit test",
        "scopes": ["whole_step"],
        "max_rows": constants.DEFAULT_ROW_LIMIT,
        "max_total_rows": 2 * constants.DEFAULT_ROW_LIMIT,
        "ranks": dep.decode.tp,
    }
    declared = dataclasses.replace(
        dep, decode=dataclasses.replace(dep.decode, megakernel=declaration)
    )
    monkeypatch.setattr(support, "DEPLOYMENT_PRESETS", {"qwen3.5-397b": declared})
    _, one_call = _verify_whole_step(16, 0)
    assert one_call.support == "declared"
    assert one_call.t_vmem_max == constants.DEFAULT_ROW_LIMIT
    report, split = _verify_whole_step(28, 0)
    assert split.support == "declared"
    assert split.n_splits > 1
    assert "microbatch_rerun" in report.flags


def test_preset_row_limit_below_one_request_is_config_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify a preset row limit below one verify request is rejected, not ignored."""
    dep = advisor.DEPLOYMENT_PRESETS["qwen3.5-397b"]
    declaration = {
        "source": "unit test",
        "scopes": ["whole_step"],
        "max_rows": 4 - 1,
        "ranks": dep.decode.tp,
    }
    declared = dataclasses.replace(
        dep, decode=dataclasses.replace(dep.decode, megakernel=declaration)
    )
    monkeypatch.setattr(support, "DEPLOYMENT_PRESETS", {"qwen3.5-397b": declared})
    with pytest.raises(MegakernelConfigError, match="below one verify request"):
        _verify_whole_step(16, 0)


def test_validate_workload_names_row_limit_below_one_request() -> None:
    """Verify the non-raising validator reports a row limit below one request."""
    rows_per_req = 4
    workload = advisor.WorkloadSpec(
        phase="verify",
        batch=16,
        spec_tokens=rows_per_req - 1,
        context=4096,
        tp=8,
        dcp=4,
        ep=8,
        max_rows_per_call=rows_per_req - 1,
    )
    reasons = advisor.validate_workload("qwen3.5-397b", "tpu7x", workload)
    assert reasons == ("row_limit_below_request",)
    one_request = dataclasses.replace(workload, max_rows_per_call=rows_per_req)
    assert advisor.validate_workload("qwen3.5-397b", "tpu7x", one_request) == ()


def test_rows_above_default_buckets_flagged() -> None:
    """Verify CORE_LIMITS mirrors the core bucket defaults and flags larger calls."""
    tops = [
        max(core_runtime.DEFAULT_BUCKETS),
        max(core_runtime.DEFAULT_VERIFY_BUCKETS),
    ]
    lim = advisor.CORE_LIMITS
    assert [lim.default_decode_bucket_top, lim.default_verify_bucket_top] == tops

    model_profile = advisor.resolve_model_profile("qwen3.5-397b")
    for batch, flagged in (
        (lim.default_decode_bucket_top, False),
        (lim.default_decode_bucket_top + 1, True),
    ):
        report = advisor.advise(
            model_profile,
            advisor.WorkloadSpec(phase="decode", batch=batch, tp=8, ep=8),
            "tpu7x",
        )
        assert ("above_default_buckets" in report.flags) is flagged
    verify_top_requests = lim.default_verify_bucket_top // 4
    rep_at_top, _ = _verify_whole_step(verify_top_requests, 0)
    rep_above, _ = _verify_whole_step(verify_top_requests + 1, 0)
    assert "above_default_buckets" not in rep_at_top.flags
    assert "above_default_buckets" in rep_above.flags


@pytest.mark.usefixtures("without_declared_megakernels")
def test_ffn_scratch_includes_fp32_bytes() -> None:
    """Verify FFN scratch uses FP32 bytes so Qwen3.5-397B VMEM cap is 448 rows."""
    model_profile = advisor.resolve_model_profile("qwen3.5-397b")
    workload = advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8)
    chip = advisor.resolve_chip_spec("tpu7x")
    scope_context = _build_scope_context(model_profile, workload, chip)
    assert scope_context.ffn_scratch == (
        constants.DOUBLE_BUFFER_SLOTS
        * constants.ATTN_TILE_TOKENS
        * model_profile.moe_intermediate_size
        * constants.FP32_BYTES
    )
    report = advisor.advise(model_profile, workload, chip)
    whole = next(s for s in report.scopes if s.scope == "whole_step")
    qwen35_ideal_vmem_rows = 448
    assert whole.t_vmem_max == qwen35_ideal_vmem_rows


def test_physical_config_rejects_undersized_slice_and_invalid_combos() -> None:
    """Verify undersized slices are flagged and unphysical combos raise cleanly."""
    undersized = advisor.validate_workload(
        "qwen3.5-397b",
        "tpu7x",
        advisor.WorkloadSpec(phase="decode", batch=4, tp=2, ep=2, dp=1),
    )
    assert "slice_topology_unsupported" in undersized
    valid_slice = advisor.validate_workload(
        "qwen3.5-397b",
        "tpu7x",
        advisor.WorkloadSpec(phase="decode", batch=4, tp=2, ep=2, dp=4),
    )
    assert "slice_topology_unsupported" not in valid_slice

    model_profile = advisor.resolve_model_profile("qwen3.5-397b")
    tpu7x_vmem_mib = 62.0
    invalid_workloads = (
        advisor.WorkloadSpec(phase="decode", batch=2, tp=2, dp=4, ep=8),
        advisor.WorkloadSpec(phase="verify", batch=4, tp=8, ep=8, spec_tokens=0),
        advisor.WorkloadSpec(
            phase="decode", batch=4, tp=8, ep=8, hbm_utilization_cap=1.5
        ),
        advisor.WorkloadSpec(phase="decode", batch=4, tp=8, ep=8, dma_buffers=4),
        advisor.WorkloadSpec(
            phase="decode",
            batch=4,
            tp=8,
            ep=8,
            vmem_budget_mib=tpu7x_vmem_mib + 1.0,
        ),
    )
    for bad_wl in invalid_workloads:
        with pytest.raises(MegakernelConfigError):
            advisor.advise(model_profile, bad_wl, "tpu7x")

    kvless = dataclasses.replace(model_profile, num_kv_heads=0)
    kv_reasons = advisor.validate_workload(
        kvless,
        "tpu7x",
        advisor.WorkloadSpec(phase="decode", batch=8, tp=8, kv_split=2),
    )
    assert "kv_heads_not_divisible_by_tp" in kv_reasons


def test_advisor_core_limits_match_core_package() -> None:
    """Verify spec_bridge AST drift check reports zero differences vs core/."""
    assert spec_bridge.verify_core_limits_sync() == []


def test_per_op_vmem_overflow_marks_infeasible() -> None:
    """Verify static VMEM overflow on per_op marks the baseline infeasible."""
    model_profile = advisor.resolve_model_profile("qwen3.5-397b")
    report = advisor.advise(
        model_profile,
        advisor.WorkloadSpec(
            phase="decode",
            tp=8,
            ep=8,
            batch=4,
            vmem_budget_mib=float(constants.MIN_VMEM_MIB),
        ),
        "tpu7x",
    )
    per_op = next(s for s in report.scopes if s.scope == "per_op")
    assert (per_op.feasible, per_op.t_vmem_max, per_op.reason) == (
        False,
        0,
        "per_op_vmem_overflow",
    )


@pytest.mark.usefixtures("without_declared_megakernels")
def test_recommended_scope_metrics_and_hbm_oom_zero_tps() -> None:
    """Verify summary metrics follow recommended scope and zero out on HBM OOM."""
    prof_bad = advisor.resolve_model_profile("qwen3.5-397b")
    rep_bad = advisor.advise(
        prof_bad, advisor.WorkloadSpec(phase="decode", batch=16, tp=2), "tpu7x"
    )
    assert rep_bad.verdict == "invalid_config(slice_topology_unsupported)"
    per_op_bad = next(s for s in rep_bad.scopes if s.scope == "per_op")
    assert float(rep_bad.metrics["step_floor_us"]["value"]) == pytest.approx(
        per_op_bad.step_bound_us, abs=0.01
    )
    assert float(rep_bad.metrics["throughput_tok_s_per_chip"]["value"]) == 0.0
    cmp_res = advisor.verdict.compare_parallelism_candidates(
        "qwen3.5-397b",
        (advisor.ParallelLayout(tp=2, ep=1), advisor.ParallelLayout(tp=8, ep=8)),
        chip="tpu7x",
    )
    assert cmp_res["best_step_layout"] == "TP8 x EP8"
    assert cmp_res["best_throughput_layout"] == "TP8 x EP8"
    assert cmp_res["verdict"] == "TP8 x EP8"

    pk3 = advisor.resolve_model_profile("kimi-k3")
    rep_k3 = advisor.advise(
        pk3,
        advisor.WorkloadSpec(phase="decode", batch=4, tp=32, ep=32),
        "tpu7x",
    )
    assert rep_k3.fastest_scope == "whole_step"
    assert rep_k3.recommended_scope == "per_op"
    per_op_k3 = next(s for s in rep_k3.scopes if s.scope == "per_op")
    assert float(rep_k3.metrics["step_floor_us"]["value"]) == pytest.approx(
        per_op_k3.step_bound_us, abs=0.01
    )
    assert float(rep_k3.metrics["throughput_tok_s_per_chip"]["value"]) > 0.0

    p397 = advisor.resolve_model_profile("qwen3.5-397b")
    rep_oom = advisor.advise(
        p397,
        advisor.WorkloadSpec(phase="decode", batch=4, tp=2, ep=2, dp=2),
        "tpu7x",
    )
    assert rep_oom.hbm_fits is False
    assert float(rep_oom.metrics["throughput_tok_s_per_chip"]["value"]) == 0.0


@pytest.mark.parametrize(
    ("model", "workload"),
    [
        (
            "qwen3.5-397b",
            advisor.WorkloadSpec(
                phase="verify", batch=4, spec_tokens=3, context=4096, tp=8, dcp=4, ep=8
            ),
        ),
        ("qwen3.5-397b", advisor.WorkloadSpec(phase="decode", batch=64, tp=8, ep=8)),
        (
            "glm-5.2",
            advisor.WorkloadSpec(
                phase="prefill", batch=1, isl=512, chunk_size=512, tp=16
            ),
        ),
    ],
)
def test_scope_subterms_reconstruct_roof_floor(
    model: str, workload: advisor.WorkloadSpec
) -> None:
    """Verify chain_us with hbm_us and comm_us rebuilds every scope's floor."""
    report = advisor.advise(advisor.resolve_model_profile(model), workload, "tpu7x")
    for scope in report.scopes:
        core_us = (
            max(scope.hbm_us, scope.chain_us + scope.comm_us)
            if scope.scope in ("whole_step", "layer_group")
            else max(scope.hbm_us, scope.chain_us) + scope.comm_us
        )
        assert scope.chain_us >= scope.compute_us - 0.02
        assert core_us + scope.fill_drain_us == pytest.approx(
            scope.roof_floor_us, abs=0.02
        )


def test_vector_unit_fp4_dequant_floor() -> None:
    """Verify vector unit floor on mxfp4 MoE with high FLOP and bandwidth rates."""
    base_prof = advisor.resolve_model_profile("qwen3.5-397b")
    model_profile = dataclasses.replace(base_prof, expert_dtype="mxfp4")
    workload = advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8)
    base_chip = advisor.resolve_chip_spec("tpu7x")
    high_flops = 1e20
    high_bw = 1e20
    target_vec_rate = 1e12
    chip = dataclasses.replace(
        base_chip,
        fp8_flops_per_s=high_flops,
        bf16_flops_per_s=high_flops,
        hbm_bw_bytes_per_s=high_bw,
        vector_f32_ops_per_s=target_vec_rate,
    )
    report = advisor.advise(model_profile, workload, chip)
    scope = next(s for s in report.scopes if s.scope == "whole_step")
    min_vector_floor_us = 20000.0
    assert scope.step_bound_us >= min_vector_floor_us

    # B1: verify moe_tokens_local scaling in _compute_scope_matmuls
    scope_context = _build_scope_context(model_profile, workload, chip)
    res_1x = _compute_scope_matmuls(
        scope_context,
        model_profile,
        workload,
        (1.0, 10.0 * 1e9, False),
    )
    # B2: verify per_op / single_layer scaling of streamed weight params with n_splits
    res_2x = _compute_scope_matmuls(
        scope_context,
        model_profile,
        workload,
        (1.0, 20.0 * 1e9, False),
    )
    # Doubling weight bytes from 10 GB to 20 GB doubles streamed weight time
    expected_ratio = (20.0 * 1e9) / (10.0 * 1e9)
    assert res_2x[3] / res_1x[3] == pytest.approx(expected_ratio, rel=1e-2)

    # B3: verify strict floor incorporates vector unit terms and is bounded
    routed_bytes = scope.w_step_strict_gb * 1e9 - scope_context.w_static
    strict_ops = _compute_scope_matmuls(
        scope_context,
        model_profile,
        workload,
        (1.0, routed_bytes, False),
    )
    unpack_ops = 2 * routed_bytes / constants.DTYPE_BYTES["mxfp4"]
    expected_vector_us = (
        (strict_ops[2] + strict_ops[3] - unpack_ops)
        / (constants.VECTOR_MAX_ELEMENTS_PER_WORD * target_vec_rate)
        + unpack_ops / (constants.VECTOR_SUB_BYTE_ELEMENTS_PER_WORD * target_vec_rate)
    ) * 1e6
    assert scope.step_floor_strict_us >= expected_vector_us
    chip_half_rate = dataclasses.replace(chip, vector_f32_ops_per_s=target_vec_rate / 2)
    rep_half_rate = advisor.advise(model_profile, workload, chip_half_rate)
    sc_half_rate = next(s for s in rep_half_rate.scopes if s.scope == "whole_step")
    assert (
        sc_half_rate.step_floor_strict_us - scope.step_floor_strict_us
        == pytest.approx(expected_vector_us, abs=STRICT_VECTOR_DIFF_TOLERANCE_US)
    )
    assert scope.step_floor_strict_us <= scope.step_bound_us

    # D1: verify step_achievable_us derates vector ops by eta_dense
    wl_derated = advisor.WorkloadSpec(
        phase="decode",
        batch=16,
        tp=8,
        ep=8,
        efficiency=advisor.specs.EfficiencySpec(
            eta_hbm=1.0, eta_dense=0.5, eta_moe_grouped=1.0, eta_ici=1.0
        ),
    )
    rep_derated = advisor.advise(model_profile, wl_derated, chip)
    sc_derated = next(s for s in rep_derated.scopes if s.scope == "whole_step")
    assert sc_derated.step_achievable_us > scope.step_bound_us
    assert scope.compute_us >= min_vector_floor_us


def test_gdn_linear_head_divisibility_reasons() -> None:
    """Verify linear attention head divisibility rejection and replication rules."""
    base_prof = advisor.resolve_model_profile("qwen3.5-397b")
    bad_v_prof = dataclasses.replace(
        base_prof, linear_value_heads=18, linear_key_heads=16
    )
    wl_tp16 = advisor.WorkloadSpec(phase="decode", batch=1, tp=16)
    c7x = advisor.resolve_chip_spec("tpu7x")
    reasons_v = _mesh_divisibility_reasons(bad_v_prof, wl_tp16, c7x)
    expected_v_reason = "linear_v_heads_not_divisible_by_tp"
    assert expected_v_reason in reasons_v

    bad_k_prof = dataclasses.replace(
        base_prof, linear_value_heads=16, linear_key_heads=12
    )
    reasons_k = _mesh_divisibility_reasons(bad_k_prof, wl_tp16, c7x)
    expected_k_reason = "linear_k_heads_not_divisible_by_tp"
    assert expected_k_reason in reasons_k

    rep_prof = dataclasses.replace(
        base_prof, linear_value_heads=128, linear_key_heads=16
    )
    wl_tp32 = advisor.WorkloadSpec(phase="decode", batch=1, tp=32)
    reasons_rep = _mesh_divisibility_reasons(rep_prof, wl_tp32, c7x)
    assert expected_v_reason not in reasons_rep
    assert expected_k_reason not in reasons_rep


def test_pp_with_pcp_disagg_prefill_supported() -> None:
    """Verify disaggregated prefill with simultaneous PP and PCP evaluates cleanly."""
    model_profile = advisor.resolve_model_profile("qwen3.5-397b")
    workload = advisor.WorkloadSpec(
        phase="prefill",
        batch=4,
        isl=4096,
        tp=1,
        pp=4,
        pcp=8,
        ep=8,
    )
    c7x = advisor.resolve_chip_spec("tpu7x")
    report = advisor.advise(model_profile, workload, c7x)
    unsupported_reason = "pp_with_pcp_unsupported"
    assert report.hbm_fits is True
    for scope in report.scopes:
        assert unsupported_reason not in scope.blockers


def test_fastest_tiled_scope_arithmetic_intensity_streams_weights_once() -> None:
    """Verify fastest tiled scope arithmetic intensity streams weights once."""
    model_profile = advisor.resolve_model_profile("qwen3.5-397b")
    workload = advisor.WorkloadSpec(
        phase="decode",
        batch=128,
        tp=8,
        ep=8,
        vmem_budget_mib=20,
        dispatch_regime="unfused_eager",
    )
    c7x = advisor.resolve_chip_spec("tpu7x")
    report = advisor.advise(model_profile, workload, c7x)
    assert report.fastest_scope == "single_layer"
    best = next(s for s in report.scopes if s.scope == report.fastest_scope)
    min_splits = 1
    assert best.n_splits > min_splits
    scope_context = _build_scope_context(model_profile, workload, c7x)
    state_bytes = (
        scope_context.kv_cache_read_bytes + scope_context.recurrent_state_rw_bytes
    )
    weight_reads = 1.0
    b_wa = best.w_step_gb * 1e9 * weight_reads + best.act_hbm_mb * 1e6
    tot_bytes = max(1.0, b_wa + state_bytes)
    total_flops = (
        scope_context.flops_dense + best.moe_flop_factor * scope_context.flops_moe
    )
    expected_ai = round(total_flops / tot_bytes, 2)
    assert report.step_arithmetic_intensity == pytest.approx(expected_ai, abs=0.05)


# 16 Ki KV tokens per DCP shard across 8 shards plus 4 verify query rows keeps
# HBM stream above chain_us so floor collectives partially hide under stream.
_GLM_VERIFY_DCP_DEGREE = 8
_GLM_VERIFY_QUERY_ROWS = 4
_GLM_VERIFY_KV_TOKENS_PER_SHARD = 16384
_GLM_VERIFY_CONTEXT_TOKENS = (
    _GLM_VERIFY_KV_TOKENS_PER_SHARD * _GLM_VERIFY_DCP_DEGREE + _GLM_VERIFY_QUERY_ROWS
)


@pytest.mark.parametrize(
    ("model_slug", "workload"),
    [
        ("qwen3.5-397b", advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8)),
        (
            "glm-5.2",
            advisor.WorkloadSpec(
                phase="verify",
                spec_tokens=_GLM_VERIFY_QUERY_ROWS - 1,
                batch=1,
                context=_GLM_VERIFY_CONTEXT_TOKENS,
                osl=1,
                tp=16,
                ep=1,
                dcp=_GLM_VERIFY_DCP_DEGREE,
            ),
        ),
    ],
)
def test_evaluate_corners_matches_step_upper_us(
    model_slug: str, workload: advisor.WorkloadSpec
) -> None:
    """Verify _evaluate_corners low and high corners match pure floor bounds."""
    model_profile = advisor.resolve_model_profile(model_slug)
    c7x = advisor.resolve_chip_spec("tpu7x")
    report = advisor.advise(model_profile, workload, c7x)
    scope_context = _build_scope_context(model_profile, workload, c7x)
    priors = _resolve_calibrated_priors(workload, c7x, scope_context, None)
    box = _assumed_box(scope_context, workload, priors)
    ctx_box, wl_box = _apply_box(scope_context, workload, box)
    corners = _evaluate_corners(ctx_box, model_profile, wl_box, box)
    low_corner = (
        box.launch_gap_us[0],
        round(box.collective_s[0] * 1e6, 3),
        box.attn_dots[0],
        "low",
        "low",
    )
    high_corner = (
        box.launch_gap_us[1],
        round(box.collective_s[1] * 1e6, 3),
        box.attn_dots[-1],
        "high",
        "high",
    )
    scope = next(s for s in report.scopes if s.scope == "whole_step")
    corner_tol_us = 0.05
    assert corners[low_corner]["whole_step"] == pytest.approx(
        scope.step_bound_us, abs=corner_tol_us
    )
    assert corners[high_corner]["whole_step"] == pytest.approx(
        scope.step_upper_us, abs=corner_tol_us
    )


def test_achievable_upper_scales_a2a_extra_by_eta_ici() -> None:
    """Verify achievable upper composes exposed comm + a2a_extra_s / eta_ici."""
    model_profile = advisor.resolve_model_profile("qwen3.5-397b")
    wl_base = advisor.WorkloadSpec(
        phase="decode",
        batch=16,
        tp=2,
        dp=4,
        ep=8,
        efficiency=advisor.specs.EfficiencySpec(1.0, 1.0, 1.0, 1.0),
    )
    c7x = advisor.resolve_chip_spec("tpu7x")
    rep_base = advisor.advise(model_profile, wl_base, c7x)
    sc_base = next(s for s in rep_base.scopes if s.scope == "whole_step")
    scope_context = _build_scope_context(model_profile, wl_base, c7x)
    priors = _resolve_calibrated_priors(wl_base, c7x, scope_context, None)
    box = _assumed_box(scope_context, wl_base, priors)
    _, wl_box = _apply_box(scope_context, wl_base, box)
    hop_delta_s = box.collective_s[1] - box.collective_s[0]
    # TP2 on single host = 1 intra-ICI hop per all-reduce across 2 * num_layers ARs;
    # EP8 x DP4 > 1 = 2 sync hops per MoE layer for all-to-all.
    num_all_reduces = constants.SUBLAYER_BOUNDARIES_PER_LAYER * model_profile.num_layers
    ar_extra_s = num_all_reduces * hop_delta_s
    a2a_extra_s = model_profile.num_moe_layers * 2 * hop_delta_s
    expected_base_ach_upper_us = (
        sc_base.step_achievable_us
        + (ar_extra_s + a2a_extra_s) * 1e6
        + sc_base.ops_per_step * (box.launch_gap_us[1] - wl_box.op_overhead_us)
    )
    # 4 rounded ScopeEstimate terms at 0.01 us precision -> <= 4 * 0.005 = 0.02 us.
    rounding_tol_us = 4 * 0.005
    assert sc_base.step_achievable_upper_us > sc_base.step_upper_us
    assert sc_base.step_achievable_upper_us == pytest.approx(
        expected_base_ach_upper_us, abs=rounding_tol_us
    )

    low_eta_ici = 0.7
    wl_low_ici = dataclasses.replace(
        wl_base, efficiency=advisor.specs.EfficiencySpec(1.0, 1.0, 1.0, low_eta_ici)
    )
    rep_low_ici = advisor.advise(model_profile, wl_low_ici, c7x)
    sc_low_ici = next(s for s in rep_low_ici.scopes if s.scope == "whole_step")
    expected_a2a_ici_delta_us = a2a_extra_s * (1.0 / low_eta_ici - 1.0) * 1e6
    assert (
        sc_low_ici.step_achievable_upper_us - sc_base.step_achievable_upper_us
    ) >= expected_a2a_ici_delta_us


def test_qwen35_tp16_outside_declared_8rank_group_is_hypothetical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TP16 outside declared 8-rank group makes whole_step hypothetical."""
    dep = advisor.DEPLOYMENT_PRESETS["qwen3.5-397b"]
    declaration = {
        "source": "unit test",
        "scopes": ["whole_step"],
        "ranks": 8,
    }
    declared = dataclasses.replace(
        dep, decode=dataclasses.replace(dep.decode, megakernel=declaration)
    )
    monkeypatch.setattr(support, "DEPLOYMENT_PRESETS", {"qwen3.5-397b": declared})
    model_profile = advisor.resolve_model_profile("qwen3.5-397b")
    r = advisor.advise(
        model_profile, advisor.WorkloadSpec(phase="decode", batch=16, tp=16), "tpu7x"
    )
    assert r.recommended_scope == "per_op"
    assert r.fastest_scope == "whole_step"
    ws = next(s for s in r.scopes if s.scope == "whole_step")
    assert "rank_count_outside_core_grid" in ws.blockers
    assert ws.support == "hypothetical"


def test_quantization_ridge_tokens_and_tp_invariance() -> None:
    """Verify W8A16/W8A8/BF16 ridge tokens on tpu7x/v6e and TP invariance."""
    c7x = advisor.chips.resolve_chip_spec("tpu7x")
    cv6e = advisor.chips.resolve_chip_spec("v6e")

    expected_ridge_w8a16_7x = 312.16
    expected_tokens_w8a16_7x = 157
    expected_ridge_w8a8_7x = 621.62
    expected_tokens_w8a8_7x = (311, 312, 313)
    expected_tokens_bf16_7x = 313
    expected_tokens_w8a16_v6 = 281
    expected_tokens_bf16_v6 = 561

    r_w8a16_7x = c7x.peak_flops("fp8", "bf16") / c7x.hbm_bw_bytes_per_s
    assert round(r_w8a16_7x, 2) == expected_ridge_w8a16_7x
    assert math.ceil(r_w8a16_7x / 2.0) == expected_tokens_w8a16_7x

    r_w8a8_7x = c7x.peak_flops("fp8", "fp8") / c7x.hbm_bw_bytes_per_s
    assert round(r_w8a8_7x, 2) == expected_ridge_w8a8_7x
    assert math.ceil(r_w8a8_7x / 2.0) in expected_tokens_w8a8_7x

    r_bf16_7x = c7x.peak_flops("bf16", "bf16") / c7x.hbm_bw_bytes_per_s
    assert math.ceil(r_bf16_7x / 1.0) == expected_tokens_bf16_7x

    r_w8a16_v6 = cv6e.peak_flops("fp8", "bf16") / cv6e.hbm_bw_bytes_per_s
    assert math.ceil(r_w8a16_v6 / 2.0) == expected_tokens_w8a16_v6

    r_bf16_v6 = cv6e.peak_flops("bf16", "bf16") / cv6e.hbm_bw_bytes_per_s
    assert math.ceil(r_bf16_v6 / 1.0) == expected_tokens_bf16_v6

    prof_div = models.finalize_model_profile(
        advisor.ModelProfile(
            name="dense-tp-invariant",
            num_layers=8,
            hidden_size=4096,
            intermediate_size=11008,
            num_attention_heads=32,
            num_kv_heads=16,
            head_dim=128,
            weight_dtype="fp8",
            expert_dtype="fp8",
        )
    )
    wl8 = advisor.specs.WorkloadSpec(phase="decode", batch=16, tp=8, context=0)
    wl16 = advisor.specs.WorkloadSpec(phase="decode", batch=16, tp=16, context=0)
    rep8 = advisor.advise(prof_div, wl8, c7x)
    rep16 = advisor.advise(prof_div, wl16, c7x)
    assert rep8.ridge_tokens == rep16.ridge_tokens


def test_mixed_precision_attention_qk_pv_executes_at_bf16_peak() -> None:
    """Verify mixed-precision attention dot products execute at bf16_peak."""
    c7x = advisor.chips.resolve_chip_spec("tpu7x")
    wl_prefill = advisor.specs.WorkloadSpec(
        phase="prefill", batch=1, isl=2048, osl=0, tp=8
    )
    prof_mixed = advisor.models.resolve_model_profile(
        "qwen3.5-397b", weight_dtype="fp8", act_dtype="bf16", kv_dtype="fp8"
    )
    ctx_mixed = _build_scope_context(prof_mixed, wl_prefill, c7x)
    assert ctx_mixed.attn_is_bf16 is True

    t_matmul, t_matmul_upper, _, _, _ = _compute_scope_matmuls(
        ctx_mixed, prof_mixed, wl_prefill, (1.0, 0.0, False)
    )
    expected_dense_matmul = (
        ctx_mixed.flops_dense - ctx_mixed.flops_kv
    ) / ctx_mixed.peak_w + ctx_mixed.flops_kv / ctx_mixed.bf16_peak
    t_moe = ctx_mixed.flops_moe / ctx_mixed.peak_e
    assert t_matmul == pytest.approx(expected_dense_matmul + t_moe)
    assert t_matmul_upper == pytest.approx(t_matmul)


def test_subchannel_quantization_vpu_bottleneck() -> None:
    """Verify subchannel block_k < 512 adds VPU ops and shifts compute bound."""
    c7x = advisor.chips.resolve_chip_spec("tpu7x")
    wl_prefill = advisor.specs.WorkloadSpec(
        phase="prefill", batch=1, isl=2048, osl=0, tp=8
    )
    prof_w128 = advisor.resolve_model_profile("qwen3.5-397b", act_dtype="bf16")
    expected_block_k = 128
    assert prof_w128.weight_block_k == expected_block_k
    ctx_w128 = _build_scope_context(prof_w128, wl_prefill, c7x)
    _, _, v_dense_128, _, _ = _compute_scope_matmuls(
        ctx_w128, prof_w128, wl_prefill, (1.0, 0.0, False)
    )
    expected_subchannel_ops = (
        constants.SUBCHANNEL_VPU_OPS_PER_BLOCK
        * ctx_w128.active_tokens
        * ctx_w128.base_dense_params
        / expected_block_k
    )
    assert v_dense_128 >= expected_subchannel_ops


def _whole_step_of(report: advisor.AdvisorReport) -> advisor.ScopeEstimate:
    """Return the whole_step estimate of an advisor report."""
    return next(scope for scope in report.scopes if scope.scope == "whole_step")


def test_sourced_declaration_satisfies_rank_family_and_attention_dp_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify sourced Kimi-K3 declarations pass the rank, family and DP checks."""
    deployment = advisor.DEPLOYMENT_PRESETS["kimi-k3"]
    kimi_profile = advisor.resolve_model_profile("kimi-k3")
    chip_spec = advisor.resolve_chip_spec("tpu7x")
    assert deployment.decode.megakernel is not None
    assert deployment.saturation is not None
    assert deployment.saturation.megakernel is not None
    declared_rows = support.single_call_row_limit(
        "kimi-k3",
        advisor.WorkloadSpec(phase="decode", tp=32, ep=32, dcp=32),
        deployment.decode.tp,
    )
    assert declared_rows > 0

    decode_workload = advisor.WorkloadSpec(
        phase="decode", batch=declared_rows, tp=32, ep=32, dcp=32
    )
    decode_report = advisor.advise(kimi_profile, decode_workload, chip_spec)
    assert decode_report.active_tokens == declared_rows
    assert _whole_step_of(decode_report).support == "declared"
    assert _whole_step_of(decode_report).blockers == ()

    # Two attention replicas run inside one call: the declaration carries the flag.
    replicated_workload = advisor.WorkloadSpec(
        phase="decode", batch=declared_rows, tp=16, dp=2, ep=32, dcp=16
    )
    replicated_context = _build_scope_context(
        kimi_profile, replicated_workload, chip_spec
    )
    replicated_report = advisor.advise(kimi_profile, replicated_workload, chip_spec)
    assert _whole_step_of(replicated_report).support == "declared"
    assert _whole_step_of(replicated_report).blockers == ()

    # A model without a declaration keeps the layer-family check.
    glm_report = advisor.advise(
        advisor.resolve_model_profile("glm-5.2"),
        advisor.WorkloadSpec(phase="decode", batch=32, tp=16, dcp=8),
        chip_spec,
    )
    assert _whole_step_of(glm_report).support == "hypothetical"
    assert "family_not_in_core" in _whole_step_of(glm_report).blockers

    # The same block without the flag brings the all-to-all blocker back.
    block_without_flag = {
        **deployment.saturation.megakernel,
        "attention_dp_in_call": False,
    }
    without_flag = dataclasses.replace(
        deployment,
        saturation=dataclasses.replace(
            deployment.saturation, megakernel=block_without_flag
        ),
    )
    monkeypatch.setattr(support, "DEPLOYMENT_PRESETS", {"kimi-k3": without_flag})
    status, blockers = support.evaluate_support(
        "whole_step", kimi_profile, replicated_workload, chip_spec, replicated_context
    )
    assert status == "hypothetical"
    assert "all_to_all_not_in_core" in blockers


def test_split_step_support_uses_whole_step_rows() -> None:
    """Verify a split verify step is checked against the declared total rows."""
    # 192 rows exceed one call and stay within the declared total: split, declared.
    split_report, split_step = _verify_whole_step(
        SPLIT_STEP_ROWS // ROWS_PER_VERIFY_REQUEST, 0
    )
    min_splits = 1
    assert split_report.active_tokens == SPLIT_STEP_ROWS
    assert split_step.support == "declared"
    assert split_step.blockers == ()
    assert split_step.n_splits > min_splits

    # The boundary comes from the preset: the declared total passes, one more
    # request does not.
    deployment = advisor.DEPLOYMENT_PRESETS["qwen3.5-397b"]
    no_override = advisor.WorkloadSpec()
    total_rows = support.whole_step_row_limit(deployment.decode.megakernel, no_override)
    assert SPLIT_STEP_ROWS <= total_rows < ROWS_ABOVE_DECLARED_TOTAL
    assert total_rows % ROWS_PER_VERIFY_REQUEST == 0
    requests_at_total = total_rows // ROWS_PER_VERIFY_REQUEST
    at_total_report, at_total = _verify_whole_step(requests_at_total, 0)
    assert at_total_report.active_tokens == total_rows
    assert at_total.support == "declared"
    _, above_total = _verify_whole_step(requests_at_total + 1, 0)
    assert above_total.support == "hypothetical"
    assert ROW_BLOCKER in above_total.blockers

    far_report, far_above_total = _verify_whole_step(
        ROWS_ABOVE_DECLARED_TOTAL // ROWS_PER_VERIFY_REQUEST, 0
    )
    assert far_report.active_tokens == ROWS_ABOVE_DECLARED_TOTAL
    assert far_above_total.support == "hypothetical"
    assert ROW_BLOCKER in far_above_total.blockers

    # A workload row override is a kernel that stops there and does not split.
    override_report, above_override = _verify_whole_step(
        ROWS_ABOVE_ROW_OVERRIDE // ROWS_PER_VERIFY_REQUEST, ROW_OVERRIDE
    )
    assert override_report.active_tokens == ROWS_ABOVE_ROW_OVERRIDE
    assert above_override.support == "hypothetical"
    assert ROW_BLOCKER in above_override.blockers
    _, at_override = _verify_whole_step(
        ROW_OVERRIDE // ROWS_PER_VERIFY_REQUEST, ROW_OVERRIDE
    )
    assert at_override.support == "declared"

    # Precedence of the whole-step limit: workload override, declared total,
    # declared single call, core limit.
    single_call_only = {"max_rows": ROW_OVERRIDE}
    with_total = {"max_rows": ROW_OVERRIDE, "max_total_rows": SPLIT_STEP_ROWS}
    overridden = advisor.WorkloadSpec(max_rows_per_call=ROWS_ABOVE_ROW_OVERRIDE)
    core_rows = constants.CORE_LIMITS.max_batch_rows
    assert support.whole_step_row_limit(None, no_override) == core_rows
    assert support.whole_step_row_limit(single_call_only, no_override) == ROW_OVERRIDE
    assert support.whole_step_row_limit(with_total, no_override) == SPLIT_STEP_ROWS
    assert (
        support.whole_step_row_limit(with_total, overridden) == ROWS_ABOVE_ROW_OVERRIDE
    )


def test_declaration_phase_filter_keeps_prefill_undeclared() -> None:
    """Verify a decode declaration never makes a prefill step declared."""
    kimi_profile = advisor.resolve_model_profile("kimi-k3")
    chip_spec = advisor.resolve_chip_spec("tpu7x")
    # The prefill layout has the rank count of the decode declaration, so only
    # the phase separates the two lookups.
    prefill_workload = advisor.WorkloadSpec(
        phase="prefill", batch=1, isl=SHORT_PREFILL_TOKENS, tp=32, ep=16, pcp=32
    )
    prefill_report = advisor.advise(kimi_profile, prefill_workload, chip_spec)
    assert _whole_step_of(prefill_report).support == "hypothetical"
    ranks = prefill_workload.tp
    prefill_step = advisor.WorkloadSpec(phase="prefill", tp=32, ep=32, dcp=32)
    decode_step = advisor.WorkloadSpec(phase="decode", tp=32, ep=32, dcp=32)
    assert (
        support.find_declaration("kimi-k3", "whole_step", ranks, prefill_step) is None
    )
    assert (
        support.find_declaration("kimi-k3", "whole_step", ranks, decode_step)
        is not None
    )


def test_single_layer_floor_streams_weights_once() -> None:
    """Verify a tiled single_layer step is charged one weight pass, not one per tile."""
    kimi_profile = advisor.resolve_model_profile("kimi-k3")
    chip_spec = advisor.resolve_chip_spec("tpu7x")
    workload = advisor.WorkloadSpec(
        phase="prefill", batch=1, isl=LONG_PREFILL_TOKENS, tp=32, ep=32, dcp=32
    )
    report = advisor.advise(kimi_profile, workload, chip_spec)
    scope_context = _build_scope_context(kimi_profile, workload, chip_spec)
    single_layer = next(
        scope for scope in report.scopes if scope.scope == "single_layer"
    )
    min_row_tiles = 2
    assert report.active_tokens == LONG_PREFILL_TOKENS
    assert single_layer.n_splits >= min_row_tiles
    one_weight_pass_us = single_layer.w_step_gb * 1e9 / scope_context.bw_hbm * 1e6
    assert single_layer.hbm_us >= one_weight_pass_us
    assert single_layer.hbm_us < WEIGHT_PASSES_UPPER_LIMIT * one_weight_pass_us
    # Expert coverage is counted over the whole step, as for whole_step.
    assert single_layer.w_step_gb == _whole_step_of(report).w_step_gb


def test_strict_vector_floor_uses_packed_element_rate() -> None:
    """Verify the strict floor divides vector elements by the packed element rate."""
    profile = advisor.resolve_model_profile("qwen3.5-397b")
    # Expert weights of one byte or more: no sub-byte unpack term.
    assert profile.expert_dtype not in constants.SUB_BYTE_EXPERT_DTYPES
    workload = advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8)
    chip_spec = dataclasses.replace(
        advisor.resolve_chip_spec("tpu7x"),
        fp8_flops_per_s=UNBOUNDED_RATE,
        bf16_flops_per_s=UNBOUNDED_RATE,
        hbm_bw_bytes_per_s=UNBOUNDED_RATE,
        vector_f32_ops_per_s=SLOW_VECTOR_WORD_RATE_PER_S,
    )
    half_rate_chip_spec = dataclasses.replace(
        chip_spec, vector_f32_ops_per_s=SLOW_VECTOR_WORD_RATE_PER_S / 2
    )
    whole_step = _whole_step_of(advisor.advise(profile, workload, chip_spec))
    half_rate_whole_step = _whole_step_of(
        advisor.advise(profile, workload, half_rate_chip_spec)
    )
    scope_context = _build_scope_context(profile, workload, chip_spec)
    routed_bytes = whole_step.w_step_strict_gb * 1e9 - scope_context.w_static
    vector_counts = _compute_scope_matmuls(
        scope_context, profile, workload, (1.0, routed_bytes, False)
    )
    vector_elements = vector_counts[2] + vector_counts[3]
    packed_element_rate = (
        constants.VECTOR_MAX_ELEMENTS_PER_WORD * SLOW_VECTOR_WORD_RATE_PER_S
    )
    expected_vector_us = vector_elements / packed_element_rate * 1e6
    assert expected_vector_us > MIN_RESOLVABLE_VECTOR_TERM_US
    # Halving the vector rate doubles the vector term and nothing else.
    assert (
        half_rate_whole_step.step_floor_strict_us - whole_step.step_floor_strict_us
        == pytest.approx(expected_vector_us, abs=STRICT_VECTOR_DIFF_TOLERANCE_US)
    )


def test_floor_moe_padding_is_min_of_masked_and_grouped() -> None:
    """Verify per_op floor MoE padding at 1 row equals the masked factor."""
    model_profile = advisor.resolve_model_profile("kimi-k3")
    chip_spec = advisor.resolve_chip_spec("tpu7x")
    workload = advisor.WorkloadSpec(
        phase="decode",
        batch=1,
        tp=32,
        ep=32,
        moe_mode="grouped_on_chip",
    )
    report = advisor.advise(model_profile, workload, chip_spec)
    per_op = next(est for est in report.scopes if est.scope == "per_op")
    masked_tile_rows = 16.0
    grouped_row_floor = float(chip_spec.lhs_row_floor(model_profile.expert_dtype))
    assert masked_tile_rows < grouped_row_floor
    assert per_op.moe_flop_factor == pytest.approx(masked_tile_rows)


@pytest.mark.usefixtures("without_declared_megakernels")
def test_fused_floor_never_exceeds_either_loop_order() -> None:
    """Verify tiled whole_step HBM bytes stay at most min(Order A, Order B)."""
    model_profile = advisor.resolve_model_profile("qwen3.5-397b")
    chip_spec = advisor.resolve_chip_spec("tpu7x")
    active_rows = 512
    tp_deg = 8
    context_tokens = 4096
    min_splits = 2
    workload = advisor.WorkloadSpec(
        phase="decode",
        batch=active_rows,
        isl=context_tokens,
        context=context_tokens,
        tp=tp_deg,
        ep=tp_deg,
    )
    report = advisor.advise(model_profile, workload, chip_spec)
    whole = next(est for est in report.scopes if est.scope == "whole_step")
    assert whole.n_splits >= min_splits

    w_step_one_pass = whole.w_step_gb * 1e9
    tile_rows = (active_rows + whole.n_splits - 1) // whole.n_splits
    tile_report = advisor.advise(
        model_profile, dataclasses.replace(workload, batch=tile_rows), chip_spec
    )
    tile_whole = next(est for est in tile_report.scopes if est.scope == "whole_step")
    w_tile = tile_whole.w_step_gb * 1e9

    kvh_local = max(1, (model_profile.num_kv_heads + tp_deg - 1) // tp_deg)
    kv_bytes_per_tok = (
        2.0
        * kvh_local
        * model_profile.head_dim
        * constants.DTYPE_BYTES[model_profile.kv_dtype]
        * model_profile.kv_compress_factor
    )
    kv_read_bytes = (
        model_profile.num_attention_layers
        * active_rows
        * context_tokens
        * kv_bytes_per_tok
    )
    kh_local = (model_profile.linear_key_heads + tp_deg - 1) // tp_deg
    vh_local = (model_profile.linear_value_heads + tp_deg - 1) // tp_deg
    kd, vd = model_profile.linear_key_dim, model_profile.linear_value_dim
    z_dim = vh_local * vd
    qkv_lin = 2 * kh_local * kd + z_dim
    recurrent_rw_bytes = (
        2.0
        * model_profile.num_linear_layers
        * active_rows
        * (z_dim * kd + qkv_lin * model_profile.linear_conv_kernel)
        * constants.DTYPE_BYTES[model_profile.state_dtype]
    )
    eff_state_bytes = kv_read_bytes + recurrent_rw_bytes
    residual_bytes = (
        2.0
        * float(model_profile.residual_streams)
        * float(active_rows)
        * float(model_profile.hidden_size)
        * 2.0
    )
    order_a_bytes = whole.n_splits * w_tile + eff_state_bytes + residual_bytes
    order_b_bytes = (
        w_step_one_pass
        + eff_state_bytes
        + float(model_profile.num_layers) * residual_bytes
    )
    charged_hbm_bytes = (whole.hbm_us * 1e-6) * chip_spec.hbm_bw_bytes_per_s
    rounding_slack_bytes = 0.01e-6 * chip_spec.hbm_bw_bytes_per_s
    assert charged_hbm_bytes <= min(order_a_bytes, order_b_bytes) + rounding_slack_bytes


def test_layer_group_search_returns_lowest_bound_feasible_partition() -> None:
    """Verify search_layer_groups ranks 1 <= listed <= 1-per-layer < per_op."""
    prefill_role = advisor.DEPLOYMENT_PRESETS["kimi-k3"].prefill
    prefill_window_tokens = 512
    workload = advisor.WorkloadSpec(
        phase="prefill",
        batch=1,
        isl=prefill_window_tokens,
        osl=0,
        context=prefill_window_tokens,
        tp=prefill_role.tp,
        ep=prefill_role.ep,
        dp=prefill_role.dp,
        pp=prefill_role.pp,
        dcp=prefill_role.dcp,
        pcp=prefill_role.pcp,
        layer_groups=prefill_role.layer_groups,
    )
    search_res = advisor.search_layer_groups("kimi-k3", workload, "tpu7x")
    assert search_res.best is not None
    assert search_res.best.num_groups == 1
    assert search_res.listed_step_bound_us is not None
    kimi_num_layers = 93
    per_layer = next(
        cand.step_bound_us
        for cand in search_res.candidates
        if cand.num_groups == kimi_num_layers
    )
    assert (
        search_res.best.step_bound_us
        <= search_res.listed_step_bound_us
        <= per_layer
        < search_res.per_op_step_bound_us
    )
    assert search_res.gap_vs_per_op_us > 0.0
    assert "one fused kernel per group" in search_res.note
    assert "per_op" in search_res.note


def test_declaration_matches_full_layout() -> None:
    """Verify declarations match on full mesh layout."""
    model_profile = models.resolve_model_profile("qwen3.5-397b")
    chip_spec = advisor.resolve_chip_spec("tpu7x")
    expected_dcp4_rows = 112
    expected_coloc_rows = 64

    wl_dcp = advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8, dcp=4)
    assert (
        support.single_call_row_limit("qwen3.5-397b", wl_dcp, 8) == expected_dcp4_rows
    )
    rep_dcp = advisor.advise(model_profile, wl_dcp, chip_spec)
    ws_dcp = next(s for s in rep_dcp.scopes if s.scope == "whole_step")
    assert ws_dcp.support == "declared"
    assert ws_dcp.t_vmem_max == expected_dcp4_rows

    wl_coloc = advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8, dcp=1)
    assert (
        support.single_call_row_limit("qwen3.5-397b", wl_coloc, 8)
        == expected_coloc_rows
    )
    rep_coloc = advisor.advise(model_profile, wl_coloc, chip_spec)
    ws_coloc = next(s for s in rep_coloc.scopes if s.scope == "whole_step")
    assert ws_coloc.support == "declared"
    assert ws_coloc.t_vmem_max == expected_coloc_rows

    wl_verify_16 = advisor.WorkloadSpec(
        phase="verify", spec_tokens=3, tp=8, ep=8, dcp=1, batch=16
    )
    rep_v16 = advisor.advise(model_profile, wl_verify_16, chip_spec)
    ws_v16 = next(s for s in rep_v16.scopes if s.scope == "whole_step")
    assert ws_v16.support == "declared"

    expected_v20_splits = 2
    wl_verify_20 = advisor.WorkloadSpec(
        phase="verify", spec_tokens=3, tp=8, ep=8, dcp=1, batch=20
    )
    rep_v20 = advisor.advise(model_profile, wl_verify_20, chip_spec)
    ws_v20 = next(s for s in rep_v20.scopes if s.scope == "whole_step")
    assert ws_v20.support == "declared"
    assert ws_v20.n_splits == expected_v20_splits

    wl_verify_41 = advisor.WorkloadSpec(
        phase="verify", spec_tokens=3, tp=8, ep=8, dcp=1, batch=41
    )
    rep_v41 = advisor.advise(model_profile, wl_verify_41, chip_spec)
    ws_v41 = next(s for s in rep_v41.scopes if s.scope == "whole_step")
    assert ws_v41.support == "hypothetical"
    assert "rows_exceed_core_max_batch_rows" in ws_v41.blockers

    wl_v20_dcp = advisor.WorkloadSpec(
        phase="verify", spec_tokens=3, tp=8, ep=8, dcp=4, batch=20
    )
    rep_v20_dcp = advisor.advise(model_profile, wl_v20_dcp, chip_spec)
    ws_v20_dcp = next(s for s in rep_v20_dcp.scopes if s.scope == "whole_step")
    assert ws_v20_dcp.support == "declared"


def test_layout_mismatch_reports_layout_not_declared() -> None:
    """Verify mismatched layout with matching ranks reports layout_not_declared."""
    model_profile = models.resolve_model_profile("qwen3.5-397b")
    chip_spec = advisor.resolve_chip_spec("tpu7x")

    wl_tp8 = advisor.WorkloadSpec(
        phase="decode", batch=16, tp=8, ep=1, dp=1, pp=1, dcp=1
    )
    rep_tp8 = advisor.advise(model_profile, wl_tp8, chip_spec)
    ws_tp8 = next(s for s in rep_tp8.scopes if s.scope == "whole_step")
    assert ws_tp8.support == "hypothetical"
    assert "layout_not_declared" in ws_tp8.blockers
    assert "rank_count_outside_core_grid" not in ws_tp8.blockers

    wl_tp4 = advisor.WorkloadSpec(
        phase="decode", batch=16, tp=4, ep=1, dp=1, pp=1, dcp=1
    )
    rep_tp4 = advisor.advise(model_profile, wl_tp4, chip_spec)
    ws_tp4 = next(s for s in rep_tp4.scopes if s.scope == "whole_step")
    assert ws_tp4.support == "hypothetical"
    assert "rank_count_outside_core_grid" in ws_tp4.blockers
    assert "layout_not_declared" not in ws_tp4.blockers


def test_tied_declarations_are_rejected_at_load() -> None:
    """Verify conflicting declarations with identical layouts raise an error."""
    decode_layout = models.ParallelLayout(
        tp=8,
        ep=8,
        dp=1,
        pp=1,
        dcp=4,
        megakernel={
            "source": "Decode megakernel on TP8 x EP8 x DCP4",
            "scopes": ["whole_step"],
            "ranks": 8,
            "max_rows": 112,
            "phases": ["decode", "verify"],
        },
    )
    prefill_layout = models.ParallelLayout(tp=1, ep=1, dp=1, pp=8, dcp=1)
    tied_layout = models.ParallelLayout(
        tp=8,
        ep=8,
        dp=1,
        pp=1,
        dcp=4,
        megakernel={
            "source": "Duplicate decode megakernel on TP8 x EP8 x DCP4",
            "scopes": ["whole_step"],
            "ranks": 8,
            "max_rows": 64,
            "phases": ["decode"],
        },
    )
    tied_preset = models.DeploymentPreset(
        decode=decode_layout,
        prefill=prefill_layout,
        megakernel_layouts=(tied_layout,),
    )
    with pytest.raises(MegakernelConfigError):
        models.validate_declared_layouts("test-model", tied_preset)

    disjoint_dcp_layout = models.ParallelLayout(
        tp=8,
        ep=8,
        dp=1,
        pp=1,
        dcp=1,
        megakernel={
            "source": "Colocated decode megakernel on TP8 x EP8",
            "scopes": ["whole_step"],
            "ranks": 8,
            "max_rows": 64,
            "phases": ["decode"],
        },
    )
    valid_dcp_preset = models.DeploymentPreset(
        decode=decode_layout,
        prefill=prefill_layout,
        megakernel_layouts=(disjoint_dcp_layout,),
    )
    models.validate_declared_layouts("test-model", valid_dcp_preset)

    disjoint_phase_layout = models.ParallelLayout(
        tp=8,
        ep=8,
        dp=1,
        pp=1,
        dcp=4,
        megakernel={
            "source": "Verify-only megakernel on TP8 x EP8 x DCP4",
            "scopes": ["whole_step"],
            "ranks": 8,
            "max_rows": 64,
            "phases": ["verify"],
        },
    )
    decode_only_layout = models.ParallelLayout(
        tp=8,
        ep=8,
        dp=1,
        pp=1,
        dcp=4,
        megakernel={
            "source": "Decode-only megakernel on TP8 x EP8 x DCP4",
            "scopes": ["whole_step"],
            "ranks": 8,
            "max_rows": 112,
            "phases": ["decode"],
        },
    )
    valid_phase_preset = models.DeploymentPreset(
        decode=decode_only_layout,
        prefill=prefill_layout,
        megakernel_layouts=(disjoint_phase_layout,),
    )
    models.validate_declared_layouts("test-model", valid_phase_preset)


@pytest.mark.usefixtures("without_declared_megakernels")
def test_undeclared_fused_call_above_default_bucket_is_hypothetical() -> None:
    """Verify undeclared fused calls above default buckets require stated limits."""
    model_profile = advisor.resolve_model_profile("qwen3.5-397b")
    chip_spec = advisor.resolve_chip_spec("tpu7x")

    def eval_whole_step(
        phase: Literal["decode", "verify"],
        batch: int,
        spec_tokens: int = 0,
        max_rows_per_call: int = 0,
    ) -> tuple[advisor.AdvisorReport, advisor.ScopeEstimate]:
        """Evaluate an undeclared Qwen3.5 step and return (report, whole_step)."""
        wl = advisor.WorkloadSpec(
            phase=phase,
            batch=batch,
            spec_tokens=spec_tokens,
            max_rows_per_call=max_rows_per_call,
            tp=8,
            ep=8,
            context=4096,
        )
        rep = advisor.advise(model_profile, wl, chip_spec)
        ws = next(s for s in rep.scopes if s.scope == "whole_step")
        return rep, ws

    # Decode 32 rows: within core limits, no row limit blocker, no assumption.
    rep_dec_32, ws_dec_32 = eval_whole_step(phase="decode", batch=32)
    assert ws_dec_32.support == "within_core_limits"
    assert "row_limit_not_stated" not in ws_dec_32.blockers
    assert not any(
        u["name"] == "max_rows_per_call" for u in rep_dec_32.unresolved_assumptions
    )

    # Decode 33 rows: exceeds default decode bucket, requires stated limit.
    rep_dec_33, ws_dec_33 = eval_whole_step(phase="decode", batch=33)
    assert ws_dec_33.support == "hypothetical"
    assert "row_limit_not_stated" in ws_dec_33.blockers
    assert rep_dec_33.recommended_scope == "per_op"
    unres_dec_33 = [
        u for u in rep_dec_33.unresolved_assumptions if u["name"] == "max_rows_per_call"
    ]
    assert len(unres_dec_33) == 1
    item_dec_33 = unres_dec_33[0]
    assert item_dec_33["affects"] == "support"
    assert item_dec_33["unit"] == "rows"
    assert item_dec_33["sensitivity_us"] == 0.0
    assert item_dec_33["band"] == [
        constants.CORE_LIMITS.default_decode_bucket_top,
        constants.CORE_LIMITS.max_batch_rows,
    ]
    assert "--max-rows-per-call" in str(item_dec_33["how_to_settle"])

    # Verify 64 rows: within default verify bucket (16 * 4 = 64).
    _, ws_ver_64 = eval_whole_step(phase="verify", batch=16, spec_tokens=3)
    assert ws_ver_64.support == "within_core_limits"
    assert "row_limit_not_stated" not in ws_ver_64.blockers

    # Verify 68 rows: exceeds default verify bucket (17 * 4 = 68).
    rep_ver_68, ws_ver_68 = eval_whole_step(phase="verify", batch=17, spec_tokens=3)
    assert ws_ver_68.support == "hypothetical"
    assert "row_limit_not_stated" in ws_ver_68.blockers
    unres_ver_68 = [
        u for u in rep_ver_68.unresolved_assumptions if u["name"] == "max_rows_per_call"
    ]
    assert len(unres_ver_68) == 1
    item_ver_68 = unres_ver_68[0]
    assert item_ver_68["affects"] == "support"
    assert item_ver_68["band"] == [
        constants.CORE_LIMITS.default_verify_bucket_top,
        constants.CORE_LIMITS.max_batch_rows,
    ]

    # Explicit max_rows_per_call clears the blocker and assumption.
    rep_dec_33_exp, ws_dec_33_exp = eval_whole_step(
        phase="decode", batch=33, max_rows_per_call=64
    )
    assert "row_limit_not_stated" not in ws_dec_33_exp.blockers
    assert ws_dec_33_exp.support == "within_core_limits"
    assert not any(
        u["name"] == "max_rows_per_call" for u in rep_dec_33_exp.unresolved_assumptions
    )

    rep_ver_68_exp, ws_ver_68_exp = eval_whole_step(
        phase="verify", batch=17, spec_tokens=3, max_rows_per_call=128
    )
    assert "row_limit_not_stated" not in ws_ver_68_exp.blockers
    assert ws_ver_68_exp.support == "within_core_limits"
    assert not any(
        u["name"] == "max_rows_per_call" for u in rep_ver_68_exp.unresolved_assumptions
    )


def test_achievable_adds_collectives_while_floor_and_corners_stay_overlapped() -> None:
    """Verify achievable adds collectives to max(stream, chain) while floor overlaps."""
    glm_profile = advisor.resolve_model_profile("glm-5.2")
    unit_efficiency = advisor.specs.EfficiencySpec(1.0, 1.0, 1.0, 1.0)
    workload = advisor.WorkloadSpec(
        phase="verify",
        spec_tokens=_GLM_VERIFY_QUERY_ROWS - 1,
        batch=1,
        context=_GLM_VERIFY_CONTEXT_TOKENS,
        osl=1,
        tp=16,
        ep=1,
        dcp=_GLM_VERIFY_DCP_DEGREE,
        dispatch_regime="zero_floor",
        efficiency=unit_efficiency,
    )
    chip_spec = advisor.resolve_chip_spec("tpu7x")
    report = advisor.advise(glm_profile, workload, chip_spec)
    whole_step = next(s for s in report.scopes if s.scope == "whole_step")
    # Each ScopeEstimate latency rounds to 0.01 us (0.005 us half-ULP); summing 6
    # rounded fields + step output errs by at most (6 + 1) * 0.005 = 0.035 us.
    rounding_tol_us = 7 * 0.005
    min_sublayer_comm_us = (
        glm_profile.num_layers
        * constants.SUBLAYER_BOUNDARIES_PER_LAYER
        * chip_spec.collective_latency_s
        * 1e6
    )
    expected_floor_us = (
        max(whole_step.hbm_us, whole_step.chain_us + whole_step.comm_us)
        + whole_step.fill_drain_us
        + whole_step.repass_spill_us
        + whole_step.launch_us
    )
    extra_indexer_radix_us = (
        (constants.INDEXER_RADIX_PASSES - 1)
        * glm_profile.num_indexer_layers
        * round(math.log2(_GLM_VERIFY_DCP_DEGREE))
        * chip_spec.collective_latency_s
        * 1e6
    )
    ach_boundary_fill_us = (
        (whole_step.ops_per_step - 1)
        * (
            constants.DENSE_SLOT_DEFAULT_BYTES / chip_spec.hbm_bw_bytes_per_s
            + constants.BOUNDARY_SYNC_HOP_FACTOR * chip_spec.collective_latency_s
        )
        * 1e6
    )
    expected_ach_us = (
        max(whole_step.hbm_us, whole_step.chain_us)
        + whole_step.comm_us
        + extra_indexer_radix_us
        + whole_step.fill_drain_us
        + ach_boundary_fill_us
        + whole_step.repass_spill_us
        + whole_step.launch_us
    )
    assert whole_step.comm_us >= min_sublayer_comm_us
    assert whole_step.hbm_us > whole_step.chain_us
    assert whole_step.step_bound_us == pytest.approx(
        expected_floor_us, abs=rounding_tol_us
    )
    assert whole_step.step_achievable_us == pytest.approx(
        expected_ach_us, abs=rounding_tol_us
    )


def _exact_binomial_busiest_reference(
    tokens_ep: int,
    top_k: int,
    num_experts: int,
    experts_local: int,
    owner_groups: int,
) -> float:
    """Compute exact independent-binomial busiest-group expectation via math.comb."""
    hit = 1.0 - (1.0 - min(1.0, top_k / num_experts)) ** tokens_ep
    if owner_groups <= 1:
        return max(1.0, experts_local * hit)
    experts_per_group = num_experts // owner_groups
    if hit >= 1.0:
        return max(1.0, float(experts_local))
    cdf = 0.0
    expected_group_max = 0.0
    for count in range(experts_per_group):
        pmf = (
            math.comb(experts_per_group, count)
            * (hit**count)
            * ((1.0 - hit) ** (experts_per_group - count))
        )
        cdf = min(1.0, cdf + pmf)
        expected_group_max += 1.0 - cdf**owner_groups
    return max(1.0, expected_group_max * (experts_local / experts_per_group))


def _lgamma_binomial_rel_tol(num_experts: int, owner_groups: int) -> float:
    """Bound IEEE-754 float64 relative error across E_g lgamma log-PMF terms."""
    # 3 lgamma evaluations + 1 exp per binomial log-PMF term = 4 transcendental ops.
    transcendental_ops_per_term = 4.0
    experts_per_group = max(1, num_experts // max(1, owner_groups))
    return (
        transcendental_ops_per_term
        * experts_per_group
        * max(1.0, math.lgamma(experts_per_group + 1))
        * math.ulp(1.0)
    )


@pytest.mark.parametrize(
    ("model_slug", "tokens_ep", "ep_size", "owner_groups", "saturate_hit"),
    [
        ("qwen3.5-397b", 16, 8, 1, False),
        ("qwen3.5-397b", 16, 8, 8, False),
        ("kimi-k3", 4, 32, 8, False),
        ("qwen3.5-397b", 4, 8, 8, True),
    ],
)
def test_experts_hit_busiest_matches_exact_binomial_order_statistic(
    model_slug: str,
    tokens_ep: int,
    ep_size: int,
    owner_groups: int,
    *,
    saturate_hit: bool,
) -> None:
    """Verify experts_hit_busiest against exact math.comb sum and per-rank ratio."""
    profile = advisor.resolve_model_profile(model_slug)
    top_k = profile.num_experts if saturate_hit else profile.top_k
    experts_local = profile.num_experts // ep_size
    expected = _exact_binomial_busiest_reference(
        tokens_ep, top_k, profile.num_experts, experts_local, owner_groups
    )
    actual = roofline.experts_hit_busiest(
        tokens_ep, top_k, profile.num_experts, experts_local, owner_groups
    )
    rel_tol = _lgamma_binomial_rel_tol(profile.num_experts, owner_groups)
    assert actual == pytest.approx(expected, rel=rel_tol)


def test_busiest_owner_group_scales_achievable_and_handles_large_expert_counts() -> (
    None
):
    """Verify busiest-group order statistics scale MoE without float overflow."""
    qwen_profile = advisor.resolve_model_profile("qwen3.5-397b")
    qwen_ep = 8
    qwen_local = qwen_profile.num_experts // qwen_ep
    single_group_qwen = roofline.experts_hit_busiest(
        16, qwen_profile.top_k, qwen_profile.num_experts, qwen_local, 1
    )
    busiest_qwen = roofline.experts_hit_busiest(
        16, qwen_profile.top_k, qwen_profile.num_experts, qwen_local, qwen_ep
    )
    assert busiest_qwen > single_group_qwen

    kimi_profile = advisor.resolve_model_profile("kimi-k3")
    kimi_preset = advisor.DEPLOYMENT_PRESETS["kimi-k3"]
    assert kimi_preset.decode is not None
    assert kimi_preset.decode.megakernel is not None
    expected_kimi_groups = int(
        str(kimi_preset.decode.megakernel["expert_owner_groups"])
    )
    kimi_workload = advisor.WorkloadSpec(
        phase="decode", batch=1, context=32768, osl=1, tp=32, ep=32, dcp=32
    )
    kimi_whole_step_declaration = support.find_declaration(
        "kimi-k3", "whole_step", 32, kimi_workload
    )
    kimi_per_op_declaration = support.find_declaration(
        "kimi-k3", "per_op", 32, kimi_workload
    )
    assert (
        support.effective_expert_owner_groups(
            kimi_whole_step_declaration, kimi_workload
        )
        == expected_kimi_groups
    )
    assert (
        support.effective_expert_owner_groups(kimi_per_op_declaration, kimi_workload)
        == kimi_workload.ep
    )
    tp_override_workload = dataclasses.replace(kimi_workload, ep=1)
    tp_override_declaration = support.find_declaration(
        "kimi-k3", "whole_step", 32, tp_override_workload
    )
    assert (
        support.effective_expert_owner_groups(
            tp_override_declaration, tp_override_workload
        )
        == 1
    )

    kimi_local = kimi_profile.num_experts // kimi_workload.ep
    kimi_one_row = roofline.experts_hit_busiest(
        1,
        kimi_profile.top_k,
        kimi_profile.num_experts,
        kimi_local,
        expected_kimi_groups,
    )
    assert kimi_one_row == pytest.approx(
        _exact_binomial_busiest_reference(
            1,
            kimi_profile.top_k,
            kimi_profile.num_experts,
            kimi_local,
            expected_kimi_groups,
        ),
        rel=_lgamma_binomial_rel_tol(kimi_profile.num_experts, expected_kimi_groups),
    )
    kimi_eight_groups = advisor.advise("kimi-k3", kimi_workload, "tpu7x")
    kimi_whole_step = next(
        s for s in kimi_eight_groups.scopes if s.scope == "whole_step"
    )
    assert (
        kimi_whole_step.step_achievable_us
        > kimi_whole_step.step_bound_us + kimi_whole_step.comm_us
    )

    large_experts = 16384
    large_groups = 8
    large_local = large_experts // large_groups
    large_single = roofline.experts_hit_busiest(
        16, kimi_profile.top_k, large_experts, large_local, 1
    )
    large_busiest = roofline.experts_hit_busiest(
        16, kimi_profile.top_k, large_experts, large_local, large_groups
    )
    assert math.isfinite(large_busiest)
    assert large_busiest > large_single

    with pytest.raises(ValueError, match="experts_hit_busiest"):
        roofline.experts_hit_busiest(4, 8, 0, 8, 2)
    with pytest.raises(ValueError, match="experts_hit_busiest"):
        roofline.experts_hit_busiest(4, 8, 16, 4, 32)


def test_per_op_achievable_prices_operator_kernel_boundaries() -> None:
    """Verify per_op achievable charges operator calls while keeping ops_per_step."""
    qwen_profile = advisor.resolve_model_profile("qwen3.5-397b")
    chip_spec = advisor.resolve_chip_spec("tpu7x")
    unit_efficiency = advisor.specs.EfficiencySpec(1.0, 1.0, 1.0, 1.0)
    workload = advisor.WorkloadSpec(
        phase="decode",
        batch=4,
        context=4096,
        osl=1,
        tp=8,
        ep=1,
        dispatch_regime="zero_floor",
        efficiency=unit_efficiency,
    )
    report = advisor.advise(qwen_profile, workload, chip_spec)
    per_op_scope = next(s for s in report.scopes if s.scope == "per_op")
    expected_sublayer_ops = (
        constants.SUBLAYER_BOUNDARIES_PER_LAYER * qwen_profile.num_layers
    )
    expected_operator_calls = (
        constants.PER_OP_MIXER_KERNELS_PER_LAYER
        + constants.PER_OP_BASE_FFN_KERNELS_PER_LAYER
        + constants.PER_OP_SHARED_EXPERT_KERNELS_PER_LAYER
    ) * qwen_profile.num_layers
    per_boundary_us = (
        constants.DENSE_SLOT_DEFAULT_BYTES / chip_spec.hbm_bw_bytes_per_s
        + constants.BOUNDARY_SYNC_HOP_FACTOR * chip_spec.collective_latency_s
    ) * 1e6
    expected_xla_ar_delta_us = (
        qwen_profile.ar_slots_per_layer
        * qwen_profile.num_layers
        * (constants.PER_OP_COLLECTIVE_CALL_FACTOR - 1)
        * round(math.log2(workload.tp))
        * chip_spec.collective_latency_s
        * 1e6
    )
    expected_delta_us = (
        expected_operator_calls - 1
    ) * per_boundary_us + expected_xla_ar_delta_us
    # Difference of two 0.01 us-rounded step latencies -> <= 2 * 0.005 = 0.01 us.
    rounding_tol_us = 2 * 0.005
    assert per_op_scope.ops_per_step == expected_sublayer_ops
    assert (
        per_op_scope.step_achievable_us - per_op_scope.step_bound_us
        == pytest.approx(expected_delta_us, abs=rounding_tol_us)
    )


def compute_expected_kimi_k3_decode_comm_us(
    kimi_profile: models.ModelProfile,
    chip_spec: advisor.ChipSpec,
    active_tokens: int,
) -> float:
    """Derive Kimi-K3 TP32 x EP32 x DCP32 decode whole_step collective floor."""
    tp_size = 32
    routed_group_size = 8
    dcp_group_size = 32
    declared_all_reduces = 94
    calls_per_layer = constants.DEFAULT_DECLARED_CALLS_PER_LAYER
    lat_s = chip_spec.collective_latency_s
    link_bw = chip_spec.ici_link_bw_bytes_per_s
    # Single-slice TP=32 ring: round(log2(tp_size)) = 5 hops per all-reduce.
    residual_bytes = active_tokens * kimi_profile.hidden_size * constants.BF16_BYTES
    ar_xfer_s = 2.0 * ((tp_size - 1) / tp_size) * residual_bytes / link_bw
    ar_hops = round(math.log2(tp_size))
    ar_total_s = declared_all_reduces * (ar_xfer_s + ar_hops * lat_s)
    # 92 MoE layers: group-8 routed dispatch + combine over intra-group TP=32//8=4.
    tp_in_group = tp_size // routed_group_size
    routed_hidden = kimi_profile.routed_expert_hidden_size or kimi_profile.hidden_size
    routed_fanout = min(kimi_profile.top_k, routed_group_size)
    routed_wire_bytes = (
        kimi_profile.num_moe_layers
        * 2.0
        * ((routed_group_size - 1) / routed_group_size)
        * active_tokens
        * routed_fanout
        * routed_hidden
        * constants.BF16_BYTES
        / tp_in_group
    )
    routed_hops = (
        kimi_profile.num_moe_layers
        * calls_per_layer
        * round(math.log2(routed_group_size))
    )
    # 24 MLA full-attention layers: group-32 query gather + FP32 output/LSE combine.
    local_heads = kimi_profile.num_attention_heads // tp_size
    q_head_dim = kimi_profile.qk_nope_head_dim + kimi_profile.qk_rope_head_dim
    dcp_bytes_per_head = constants.BF16_BYTES * q_head_dim + constants.FP32_BYTES * (
        kimi_profile.head_dim + constants.LSE_SCALARS_PER_HEAD
    )
    dcp_wire_bytes = (
        kimi_profile.num_attention_layers
        * (dcp_group_size - 1)
        * active_tokens
        * local_heads
        * dcp_bytes_per_head
    )
    dcp_hops = (
        kimi_profile.num_attention_layers
        * calls_per_layer
        * round(math.log2(dcp_group_size))
    )
    total_comm_s = (
        ar_total_s
        + (routed_wire_bytes + dcp_wire_bytes) / link_bw
        + (routed_hops + dcp_hops) * lat_s
    )
    return round(total_comm_s * 1e6, 2)


def test_declared_collective_steps_price_kimi_k3_and_qwen35_floors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify declared collective floor math on Kimi-K3 decode and Qwen3.5 verify."""
    chip_spec = advisor.resolve_chip_spec("tpu7x")
    kimi_profile = advisor.resolve_model_profile("kimi-k3")
    kimi_batch = 4
    kimi_workload = advisor.WorkloadSpec(
        phase="decode",
        batch=kimi_batch,
        context=4096,
        osl=1,
        tp=32,
        ep=32,
        dcp=32,
    )
    kimi_report = advisor.advise(kimi_profile, kimi_workload, chip_spec)
    kimi_whole_step = next(
        scope for scope in kimi_report.scopes if scope.scope == "whole_step"
    )
    expected_kimi_comm_floor_us = compute_expected_kimi_k3_decode_comm_us(
        kimi_profile, chip_spec, kimi_batch
    )
    rounding_tol_us = 0.05
    kimi_preset_check = advisor.DEPLOYMENT_PRESETS["kimi-k3"]
    assert kimi_preset_check.decode is not None
    assert advisor.declared_collectives(kimi_preset_check.decode.megakernel) is not None
    assert kimi_whole_step.comm_us == pytest.approx(
        expected_kimi_comm_floor_us, abs=rounding_tol_us
    )
    assert kimi_whole_step.support == "declared"

    pcp_mismatch_workload = dataclasses.replace(kimi_workload, pcp=2)
    assert (
        support.find_declaration("kimi-k3", "whole_step", 32, pcp_mismatch_workload)
        is None
    )
    pcp_mismatch_report = advisor.advise(kimi_profile, pcp_mismatch_workload, chip_spec)
    pcp_mismatch_whole_step = next(
        scope for scope in pcp_mismatch_report.scopes if scope.scope == "whole_step"
    )
    assert "layout_not_declared" in pcp_mismatch_whole_step.blockers

    qwen_profile = advisor.resolve_model_profile("qwen3.5-397b")
    qwen_verify_workload = advisor.WorkloadSpec(
        phase="verify",
        batch=4,
        spec_tokens=3,
        context=4096,
        tp=8,
        ep=8,
        dcp=4,
    )
    qwen_with_split_kv = advisor.advise(qwen_profile, qwen_verify_workload, chip_spec)
    whole_step_with_split = next(
        scope for scope in qwen_with_split_kv.scopes if scope.scope == "whole_step"
    )

    qwen_preset = advisor.DEPLOYMENT_PRESETS["qwen3.5-397b"]
    assert qwen_preset.decode is not None
    assert qwen_preset.decode.megakernel is not None
    no_step_mk = dict(qwen_preset.decode.megakernel)
    no_step_mk["collectives"] = {"steps": []}
    monkeypatch.setitem(
        advisor.DEPLOYMENT_PRESETS,
        "qwen3.5-397b",
        dataclasses.replace(
            qwen_preset,
            decode=dataclasses.replace(qwen_preset.decode, megakernel=no_step_mk),
        ),
    )
    qwen_without_split_kv = advisor.advise(
        qwen_profile, qwen_verify_workload, chip_spec
    )
    whole_step_without_split = next(
        scope for scope in qwen_without_split_kv.scopes if scope.scope == "whole_step"
    )
    verify_rows = qwen_verify_workload.batch * (qwen_verify_workload.spec_tokens + 1)
    split_kv_group_size = 4
    split_kv_hops_per_call = 2
    local_qwen_heads = qwen_profile.num_attention_heads // qwen_verify_workload.tp
    split_kv_wire_bytes = (
        qwen_profile.num_attention_layers
        * (split_kv_group_size - 1)
        * verify_rows
        * local_qwen_heads
        * qwen_profile.head_dim
        * (constants.BF16_BYTES + constants.FP32_BYTES)
    )
    split_kv_hop_count = (
        qwen_profile.num_attention_layers
        * constants.DEFAULT_DECLARED_CALLS_PER_LAYER
        * split_kv_hops_per_call
    )
    expected_split_kv_floor_us = (
        split_kv_wire_bytes / chip_spec.ici_link_bw_bytes_per_s
        + split_kv_hop_count * chip_spec.collective_latency_s
    ) * 1e6
    assert (
        whole_step_with_split.comm_us - whole_step_without_split.comm_us
        == pytest.approx(expected_split_kv_floor_us, abs=rounding_tol_us)
    )
    # Non-power-of-two DCP must return invalid_config without raising inside advise().
    non_pow2_dcp_report = advisor.advise(
        qwen_profile,
        advisor.WorkloadSpec(phase="decode", batch=4, tp=6, dcp=3),
        chip_spec,
    )
    assert non_pow2_dcp_report.verdict.startswith("invalid_config(")


def test_row_threshold_expert_mode_switches_tp_below_and_ep_above_limit() -> None:
    """Verify Qwen3.5-397B switches expert_mode from tp (<=12) to ep (>12)."""
    for batch_size, expected_mode in ((1, "tp"), (3, "tp"), (4, "ep"), (8, "ep")):
        workload = advisor.WorkloadSpec(
            phase="verify",
            batch=batch_size,
            spec_tokens=3,
            context=4096,
            tp=8,
            ep=8,
            dcp=4,
        )
        report = advisor.advise("qwen3.5-397b", workload, "tpu7x")
        whole_step_scope = next(
            scope for scope in report.scopes if scope.scope == "whole_step"
        )
        per_op_scope = next(scope for scope in report.scopes if scope.scope == "per_op")
        assert whole_step_scope.expert_mode == expected_mode
        assert per_op_scope.expert_mode == "ep"


def test_host_head_defaults_zero_and_computes_break_even_delta() -> None:
    """Verify default 0 host head, single-add split behavior, and break-even delta."""
    whole_step_device_us = 14250.0
    per_op_device_us = 12800.0
    # Break-even host-head delta equals max(0, fused_device_us - per_op_device_us).
    expected_synthetic_delta_us = whole_step_device_us - per_op_device_us
    assert (
        roofline.compute_break_even_host_head_delta_us(
            whole_step_device_us, per_op_device_us
        )
        == expected_synthetic_delta_us
    )

    # 56 verify requests * 4 tokens/req = 224 rows -> 2 calls of 112 rows on Qwen3.5.
    split_workload = advisor.WorkloadSpec(
        phase="verify",
        batch=56,
        spec_tokens=3,
        context=4096,
        tp=8,
        ep=8,
        dcp=4,
        dispatch_regime="zero_floor",
    )
    default_report = advisor.advise("qwen3.5-397b", split_workload, "tpu7x")
    per_op_default = next(
        scope for scope in default_report.scopes if scope.scope == "per_op"
    )
    whole_step_default = next(
        scope for scope in default_report.scopes if scope.scope == "whole_step"
    )
    assert per_op_default.host_head_us == 0.0
    assert whole_step_default.host_head_us == 0.0
    assert per_op_default.break_even_host_head_delta_us == 0.0
    assert whole_step_default.step_bound_us > per_op_default.step_bound_us
    break_even_delta_us = whole_step_default.break_even_host_head_delta_us
    rounding_tol_us = 0.02
    assert break_even_delta_us == pytest.approx(
        whole_step_default.step_bound_us - per_op_default.step_bound_us,
        abs=rounding_tol_us,
    )

    fused_head_us = 550.0
    per_op_head_us = fused_head_us + break_even_delta_us
    equalized_workload = dataclasses.replace(
        split_workload,
        host_head_per_op_us=per_op_head_us,
        host_head_fused_us=fused_head_us,
        host_head_explicit=True,
    )
    equalized_report = advisor.advise("qwen3.5-397b", equalized_workload, "tpu7x")
    per_op_eq = next(
        scope for scope in equalized_report.scopes if scope.scope == "per_op"
    )
    whole_step_eq = next(
        scope for scope in equalized_report.scopes if scope.scope == "whole_step"
    )
    # Host head adds once per step even when whole_step splits across 2 calls.
    assert whole_step_eq.host_head_us == pytest.approx(fused_head_us, abs=1e-6)
    assert whole_step_eq.step_bound_us == pytest.approx(
        whole_step_default.step_bound_us + fused_head_us,
        abs=rounding_tol_us,
    )
    assert whole_step_eq.step_bound_us == pytest.approx(
        per_op_eq.step_bound_us,
        abs=rounding_tol_us,
    )
    assert whole_step_eq.roof_floor_us == whole_step_default.roof_floor_us
    assert whole_step_eq.step_floor_strict_us == whole_step_default.step_floor_strict_us
    assert (
        equalized_report.metrics["compiler_proof_gain_us"]["value"]
        == default_report.metrics["compiler_proof_gain_us"]["value"]
    )
    host_head_metric = equalized_report.metrics["host_head_us"]
    host_head_band = host_head_metric["band"]
    assert host_head_band[0] <= float(host_head_metric["value"]) <= host_head_band[1]


def test_measured_per_op_step_and_coverage_set_whole_server_gain_basis() -> None:
    """Verify gain_basis requires per_op_step_us and step_coverage to be MEASURED."""
    base_workload = advisor.WorkloadSpec(
        phase="decode",
        batch=4,
        context=4096,
        tp=8,
        ep=8,
        dcp=4,
    )
    uncalibrated_report = advisor.advise("qwen3.5-397b", base_workload, "tpu7x")
    assert uncalibrated_report.gain_basis == constants.GAIN_BASIS_ESTIMATE
    assert (
        uncalibrated_report.metrics["gain_basis"]["value"]
        == constants.GAIN_BASIS_ESTIMATE
    )
    assert uncalibrated_report.metrics["per_op_step_us"]["tag"] == "ASSUMED"

    scope_affecting = {
        str(item["name"])
        for item in uncalibrated_report.unresolved_assumptions
        if item["affects"] == "scope"
    }
    assert scope_affecting <= set(constants.CORNER_AXES.keys())
    unresolved_names = {
        str(item["name"]) for item in uncalibrated_report.unresolved_assumptions
    }
    for expected_name in (
        "per_op_step_us",
        "step_coverage",
        "host_head_us",
        "eta_hbm",
        "eta_dense",
        "decoding_sequences",
    ):
        assert expected_name in unresolved_names
    for item in uncalibrated_report.unresolved_assumptions:
        for required_key in (
            "name",
            "value",
            "band",
            "unit",
            "sensitivity_us",
            "affects",
            "how_to_settle",
        ):
            assert required_key in item

    only_step_workload = dataclasses.replace(base_workload, per_op_step_us=9500.0)
    only_step_report = advisor.advise("qwen3.5-397b", only_step_workload, "tpu7x")
    assert only_step_report.gain_basis == constants.GAIN_BASIS_ESTIMATE
    assert only_step_report.metrics["per_op_step_us"]["tag"] == "MEASURED"

    calibrated_measurements = {
        "schema": "advisor.measurements.v1",
        "source": {
            "kind": "xprof_xplane",
            "file": "baseline_trace.xplane.pb",
            "image": "vllm-tpu:baseline-v1",
            "scope": "per_op",
        },
        "step_time_us": {"p50": 9500.0},
        "step_coverage": {"fused_eligible_fraction": 0.85},
        "host_head_us": {"p50": 4800.0, "p90": 5100.0, "n": 12},
    }
    calibrated_report = advisor.advise(
        "qwen3.5-397b",
        base_workload,
        "tpu7x",
        measurements=calibrated_measurements,
    )
    assert calibrated_report.gain_basis == constants.GAIN_BASIS_E2E
    assert calibrated_report.metrics["gain_basis"]["value"] == constants.GAIN_BASIS_E2E
    assert calibrated_report.metrics["per_op_step_us"]["tag"] == "MEASURED"
    assert (
        "vllm-tpu:baseline-v1" in calibrated_report.metrics["per_op_step_us"]["source"]
    )
    assert calibrated_report.metrics["calibrated_host_head_us"][
        "value"
    ] == pytest.approx(4800.0)


def test_glm_moe_dsa_sparse_attention_indexer_and_dcp_bounds() -> None:
    """Verify GLM-5.2 DSA sparse KV gather, 21-layer indexer, VMEM, and DCP bounds."""
    glm_profile = advisor.resolve_model_profile("glm-5.2")
    chip_spec = advisor.resolve_chip_spec("tpu7x")
    # Closed-form first-principles predictions for GLM-5.2 TP16 x DCP8 verify
    # (spec_tokens=3, context=131076) on TPU v7x: (floor_ms, achievable_ms).
    # Pure hardware floors exclude the 5.13 us cold-slot fill/drain term removed
    # from step_bound_us (2.888, 7.123, 11.210 ms).
    expected_ms_by_batch = {
        1: (2.89, 4.74),
        4: (7.12, 9.70),
        8: (11.21, 14.71),
    }
    # 0.005 ms half-ULP of 2-decimal prediction + 0.005 ms half-ULP of built output.
    ledger_tol_ms = 2 * 0.005
    for batch_size, (
        expected_floor_ms,
        expected_achievable_ms,
    ) in expected_ms_by_batch.items():
        workload = advisor.WorkloadSpec(
            phase="verify",
            spec_tokens=_GLM_VERIFY_QUERY_ROWS - 1,
            batch=batch_size,
            context=_GLM_VERIFY_CONTEXT_TOKENS,
            osl=1,
            tp=16,
            ep=1,
            dcp=_GLM_VERIFY_DCP_DEGREE,
        )
        report = advisor.advise(glm_profile, workload, chip_spec)
        whole_step = next(s for s in report.scopes if s.scope == "whole_step")
        per_op = next(s for s in report.scopes if s.scope == "per_op")
        floor_ms = whole_step.step_bound_us / 1e3
        achievable_ms = whole_step.step_achievable_us / 1e3
        assert floor_ms == pytest.approx(expected_floor_ms, abs=ledger_tol_ms)
        assert achievable_ms == pytest.approx(expected_achievable_ms, abs=ledger_tol_ms)
        assert whole_step.step_floor_strict_us <= whole_step.step_bound_us
        assert (
            whole_step.step_bound_us
            <= whole_step.step_achievable_us
            <= whole_step.step_achievable_upper_us
        )
        assert per_op.act_hbm_mb > whole_step.act_hbm_mb
        assert report.recommended_scope == "per_op"
        assert whole_step.support == "hypothetical"

    # Resident HBM includes both full MLA KV and packed 21-layer indexer K-cache.
    workload_dcp8 = advisor.WorkloadSpec(
        phase="verify",
        spec_tokens=_GLM_VERIFY_QUERY_ROWS - 1,
        batch=1,
        context=_GLM_VERIFY_CONTEXT_TOKENS,
        tp=16,
        ep=1,
        dcp=_GLM_VERIFY_DCP_DEGREE,
    )
    workload_dcp1 = dataclasses.replace(workload_dcp8, dcp=1)
    scope_context_batch1 = _build_scope_context(glm_profile, workload_dcp8, chip_spec)
    assert scope_context_batch1.has_dsa
    assert scope_context_batch1.is_sparse_gather
    assert (
        scope_context_batch1.hbm_used
        > scope_context_batch1.w_static + scope_context_batch1.kv_cache_read_bytes
    )

    # Shared indexer spine (21 < 78) increases residual_buffer_bytes, and DCP=8
    # adds cross-DCP gathered query scratch relative to DCP=1.
    vmem_dcp8 = advisor.compute_per_token_vmem_bytes(glm_profile, workload_dcp8)
    vmem_dcp1 = advisor.compute_per_token_vmem_bytes(glm_profile, workload_dcp1)
    expected_shared_spine_dcp8 = constants.WORD_ALIGN_BYTES * math.ceil(
        glm_profile.index_topk / _GLM_VERIFY_DCP_DEGREE
    )
    base_residual_bytes = (
        4 * max(1, glm_profile.residual_streams) + 2 + 6
    ) * glm_profile.hidden_size
    assert vmem_dcp8["residual_buffer_bytes"] == (
        base_residual_bytes + expected_shared_spine_dcp8
    )
    nh_local = math.ceil(glm_profile.num_attention_heads / workload_dcp8.tp)
    expected_dcp_q_scratch = (
        constants.BF16_BYTES
        * (_GLM_VERIFY_DCP_DEGREE - 1)
        * nh_local
        * models.dsa_query_gather_dim(glm_profile)
    )
    assert (
        vmem_dcp8["attn_scratch_bytes"] - vmem_dcp1["attn_scratch_bytes"]
        == expected_dcp_q_scratch
    )
    # With vmem_alias=True, moe_scratch dominates attn_scratch while DCP=8 shards
    # shared_indexer_spine_bytes (8192 -> 1024 B/tok), raising t_vmem_max.
    assert vmem_dcp8["total_per_token_bytes"] < vmem_dcp1["total_per_token_bytes"]
    report_dcp8 = advisor.advise(glm_profile, workload_dcp8, chip_spec)
    report_dcp1 = advisor.advise(glm_profile, workload_dcp1, chip_spec)
    whole_dcp8 = next(s for s in report_dcp8.scopes if s.scope == "whole_step")
    whole_dcp1 = next(s for s in report_dcp1.scopes if s.scope == "whole_step")
    assert whole_dcp8.t_vmem_max > whole_dcp1.t_vmem_max


def test_glm_moe_dsa_evaluate_corners_matches_step_upper_us() -> None:
    """Verify cross-DCP Q all-gather and indexer hops preserve _evaluate_corners."""
    glm_profile = advisor.resolve_model_profile("glm-5.2")
    chip_spec = advisor.resolve_chip_spec("tpu7x")
    workload = advisor.WorkloadSpec(
        phase="verify",
        spec_tokens=_GLM_VERIFY_QUERY_ROWS - 1,
        batch=4,
        context=_GLM_VERIFY_CONTEXT_TOKENS,
        osl=1,
        tp=16,
        ep=1,
        dcp=_GLM_VERIFY_DCP_DEGREE,
    )
    report = advisor.advise(glm_profile, workload, chip_spec)
    scope_context = _build_scope_context(glm_profile, workload, chip_spec)
    assert scope_context.has_dsa
    assert scope_context.dcp_query_all_gather_hops > 0
    assert scope_context.dcp_post_attn_hops > 0
    priors = _resolve_calibrated_priors(workload, chip_spec, scope_context, None)
    box = _assumed_box(scope_context, workload, priors)
    ctx_box, wl_box = _apply_box(scope_context, workload, box)
    corners = _evaluate_corners(ctx_box, glm_profile, wl_box, box)
    low_corner = (
        box.launch_gap_us[0],
        round(box.collective_s[0] * 1e6, 3),
        box.attn_dots[0],
        "low",
        "low",
    )
    high_corner = (
        box.launch_gap_us[1],
        round(box.collective_s[1] * 1e6, 3),
        box.attn_dots[-1],
        "high",
        "high",
    )
    # 0.01 us 2-decimal display quantum + 0.005 us float-rounding slack.
    display_round_tol_us = 0.015
    for scope in report.scopes:
        if not scope.feasible:
            continue
        assert corners[low_corner][scope.scope] == pytest.approx(
            scope.step_bound_us, abs=display_round_tol_us
        )
        assert corners[high_corner][scope.scope] == pytest.approx(
            scope.step_upper_us, abs=display_round_tol_us
        )


def test_homogeneous_dsa_profile_gated_without_core_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify all-full DSA VMEM/DCP layout, core gating, and dcp_gather pricing."""
    profile = models.finalize_model_profile(
        advisor.ModelProfile(
            name="homo-dsa",
            model_type="glm_moe_dsa",
            num_layers=32,
            num_attention_layers=32,
            num_moe_layers=32,
            hidden_size=4096,
            intermediate_size=8192,
            moe_intermediate_size=1024,
            shared_expert_intermediate_size=1024,
            num_attention_heads=32,
            num_kv_heads=32,
            head_dim=128,
            q_lora_rank=1024,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            num_experts=64,
            top_k=8,
            index_topk=2048,
            index_head_dim=128,
            index_n_heads=16,
            num_indexer_layers=32,
            weight_dtype="bf16",
            expert_dtype="bf16",
        )
    )
    assert profile.is_periodic
    assert profile.layer_families == ("dsa", "moe")
    workload = advisor.WorkloadSpec(
        phase="decode", batch=8, context=8192, tp=8, ep=1, dcp=4
    )
    # All-full indexer stack keeps indices in per-layer attn_scratch, not spine.
    vmem_parts = advisor.compute_per_token_vmem_bytes(profile, workload)
    shared_profile = dataclasses.replace(
        profile, num_indexer_layers=profile.num_attention_layers // 2
    )
    shared_vmem_parts = advisor.compute_per_token_vmem_bytes(shared_profile, workload)
    base_residual_bytes = (
        4 * max(1, profile.residual_streams) + 2 + 6
    ) * profile.hidden_size
    assert vmem_parts["residual_buffer_bytes"] == base_residual_bytes
    expected_indices_bytes = constants.WORD_ALIGN_BYTES * math.ceil(
        profile.index_topk / workload.dcp
    )
    assert (
        vmem_parts["attn_scratch_bytes"] - shared_vmem_parts["attn_scratch_bytes"]
        == expected_indices_bytes
    )
    assert (
        shared_vmem_parts["residual_buffer_bytes"] - vmem_parts["residual_buffer_bytes"]
        == expected_indices_bytes
    )

    chip_spec = advisor.resolve_chip_spec("tpu7x")
    report = advisor.advise(profile, workload, chip_spec)
    whole_step = next(s for s in report.scopes if s.scope == "whole_step")
    assert whole_step.support == "hypothetical"
    assert whole_step.blockers == ("family_not_in_core",)

    # Declared dcp_gather step on a DSA model suppresses automatic DSA DCP pricing
    # so cross-DCP collectives are never double-counted.
    declared_step = models.DeclaredCollectiveStep(
        kind="dcp_gather", group_size=4, calls_per_layer=2
    )
    declared_deployment = models.DeploymentPreset(
        decode=models.ParallelLayout(
            tp=8,
            ep=1,
            dp=1,
            pp=1,
            dcp=4,
            megakernel={
                "scopes": ["whole_step"],
                "max_rows": 64,
                "ranks": 8,
                "phases": ["decode"],
                "collectives": {
                    "steps": [
                        {
                            "kind": "dcp_gather",
                            "group_size": 4,
                            "calls_per_layer": 2,
                        }
                    ]
                },
                "source": "declared DSA DCP contract test",
            },
        ),
        prefill=models.ParallelLayout(tp=8, ep=1),
    )
    monkeypatch.setitem(support.DEPLOYMENT_PRESETS, "homo-dsa", declared_deployment)
    declared_report = advisor.advise(profile, workload, chip_spec)
    declared_whole = next(s for s in declared_report.scopes if s.scope == "whole_step")
    assert declared_whole.support == "declared"

    scope_context = _build_scope_context(profile, workload, chip_spec)
    all_reduce_us = (
        scope_context.num_ar
        * (
            scope_context.ar_transfer_time_s
            + scope_context.ar_hops * chip_spec.collective_latency_s
        )
        * 1e6
    )
    declared_transfer_s, _, declared_hops = roofline.compute_collective_step_terms(
        declared_step, profile, scope_context, (1, 1, "tp")
    )
    declared_dcp_us = (
        declared_transfer_s + declared_hops * chip_spec.collective_latency_s
    ) * 1e6
    auto_dsa_dcp_us = (
        scope_context.dcp_post_attn_transfer_s
        + scope_context.dcp_post_attn_hops * chip_spec.collective_latency_s
    ) * 1e6
    extra_radix_us = (
        (constants.INDEXER_RADIX_PASSES - 1)
        * scope_context.l_indexer
        * scope_context.dcp_hops
        * chip_spec.collective_latency_s
        / workload.efficiency.eta_ici
        * 1e6
    )
    # 0.01 us 2-decimal display quantum + 0.005 us float-rounding slack.
    display_round_tol_us = 0.015
    assert declared_whole.comm_us == pytest.approx(
        all_reduce_us + declared_dcp_us, abs=display_round_tol_us
    )
    assert declared_whole.chain_us < whole_step.chain_us
    expected_achievable_drop_us = (
        auto_dsa_dcp_us - declared_dcp_us
    ) / workload.efficiency.eta_ici + extra_radix_us
    assert whole_step.step_achievable_us - declared_whole.step_achievable_us == (
        pytest.approx(expected_achievable_drop_us, abs=display_round_tol_us)
    )


def test_support_separates_core_module_families_from_physics_modeled() -> None:
    """Verify whole_step support separates declared models from priced-only models."""
    default_workload = advisor.WorkloadSpec(phase="decode", batch=4, context=4096, tp=8)

    glm_profile = advisor.resolve_model_profile("glm-5.2")
    assert "dsa" in glm_profile.layer_families
    assert "dense_prefix_moe" in glm_profile.layer_families
    assert glm_profile.weight_dtype in specs.CORE_SUPPORTED_DTYPES
    glm_report = advisor.advise("glm-5.2", default_workload, "tpu7x")
    glm_whole_step = _whole_step_of(glm_report)
    assert "family_not_in_core" in glm_whole_step.blockers
    assert "dtype_not_in_core" not in glm_whole_step.blockers
    assert "planner_rejected" in glm_whole_step.blockers

    kimi_profile = advisor.resolve_model_profile("kimi-k3")
    assert kimi_profile.expert_dtype not in specs.CORE_SUPPORTED_DTYPES
    kimi_undeclared_whole = _whole_step_of(
        advisor.advise("kimi-k3", default_workload, "tpu7x")
    )
    assert "dtype_not_in_core" in kimi_undeclared_whole.blockers
    assert "family_not_in_core" in kimi_undeclared_whole.blockers

    qwen_profile = advisor.resolve_model_profile("qwen3.5-397b")
    assert all(
        family in specs.CORE_MODULE_FAMILIES for family in qwen_profile.layer_families
    )
    assert qwen_profile.weight_dtype in specs.CORE_SUPPORTED_DTYPES
    assert qwen_profile.expert_dtype in specs.CORE_SUPPORTED_DTYPES
    qwen_prefill_whole = _whole_step_of(
        advisor.advise(
            "qwen3.5-397b",
            advisor.WorkloadSpec(phase="prefill", batch=1, isl=64, tp=8, ep=8),
            "tpu7x",
        )
    )
    assert "family_not_in_core" not in qwen_prefill_whole.blockers
    assert "dtype_not_in_core" not in qwen_prefill_whole.blockers

    decode_layout = advisor.DEPLOYMENT_PRESETS["kimi-k3"].decode
    assert decode_layout is not None
    kimi_workload = advisor.WorkloadSpec(
        phase="decode",
        batch=4,
        context=4096,
        tp=decode_layout.tp,
        ep=decode_layout.ep,
        dcp=decode_layout.dcp,
    )
    kimi_report = advisor.advise("kimi-k3", kimi_workload, "tpu7x")
    kimi_whole_step = _whole_step_of(kimi_report)
    assert kimi_whole_step.support == "declared"
    assert kimi_whole_step.blockers == ()

    expected_tiers: tuple[tuple[str, specs.ModelSupportTier], ...] = (
        ("kimi-k3", "declared"),
        ("qwen3.5-397b", "declared"),
        ("glm-5.2", "priced_only"),
    )
    classified_tiers: dict[str, tuple[specs.ModelSupportTier, tuple[str, ...]]] = {}
    for model_name, expected_tier in expected_tiers:
        actual_tier, actual_notes = support.classify_model_support_tier(
            advisor.resolve_model_profile(model_name)
        )
        assert actual_tier == expected_tier
        classified_tiers[model_name] = (actual_tier, actual_notes)

    assert (
        classified_tiers["kimi-k3"][1]
        == advisor.DEPLOYMENT_PRESETS["kimi-k3"].unmodeled_notes
    )
    assert classified_tiers["kimi-k3"][1] != ()
    assert classified_tiers["glm-5.2"][1] == ()
    expected_kimi_tier, expected_kimi_notes = classified_tiers["kimi-k3"]
    assert kimi_report.support_tier == expected_kimi_tier
    assert kimi_report.unmodeled_notes == expected_kimi_notes
    expected_glm_tier, expected_glm_notes = classified_tiers["glm-5.2"]
    assert glm_report.support_tier == expected_glm_tier
    assert glm_report.unmodeled_notes == expected_glm_notes


def test_core_sync_checks_bucket_cap_and_module_families(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify core sync checks catch module family drift and bucket cap overflows."""
    assert spec_bridge.verify_core_limits_sync() == []

    monkeypatch.setattr(spec_bridge, "CORE_MODULE_FAMILIES", frozenset({"gqa"}))
    family_messages = spec_bridge.verify_core_limits_sync()
    family_mismatches = [
        message
        for message in family_messages
        if message.startswith("core_module_families:")
    ]
    assert len(family_mismatches) == 1

    monkeypatch.undo()

    monkeypatch.setattr(
        spec_bridge,
        "CORE_LIMITS",
        dataclasses.replace(
            advisor.CORE_LIMITS,
            default_verify_bucket_top=core_runtime.MAX_BUCKET_ROWS + 1,
        ),
    )
    bucket_messages = spec_bridge.verify_core_limits_sync()
    bucket_cap_mismatches = [
        message
        for message in bucket_messages
        if message.startswith("default_bucket_tops_within_max_bucket_rows:")
    ]
    bucket_top_mismatches = [
        message
        for message in bucket_messages
        if message.startswith("default_verify_bucket_top:")
    ]
    assert len(bucket_cap_mismatches) == 1
    assert len(bucket_top_mismatches) == 1


def test_floors_use_low_band_factors_and_exclude_first_slot_hbm() -> None:
    """Verify floors use low factor band ends without first-slot HBM double-count."""
    qwen_profile = advisor.resolve_model_profile("qwen3.5-397b")
    chip_spec = advisor.resolve_chip_spec("tpu7x")
    workload = advisor.WorkloadSpec(
        phase="decode",
        batch=4,
        context=4096,
        osl=1,
        tp=8,
        ep=1,
        dispatch_regime="zero_floor",
    )
    report = advisor.advise(qwen_profile, workload, chip_spec)
    per_op = next(s for s in report.scopes if s.scope == "per_op")
    assert per_op.fill_drain_us == 0.0
    assert per_op.step_floor_strict_us <= per_op.step_bound_us < per_op.step_upper_us

    scope_context = _build_scope_context(qwen_profile, workload, chip_spec)
    expected_floor_comm_us = (
        scope_context.num_ar
        * (scope_context.ar_transfer_time_s + scope_context.ar_base_lat_s)
        + 2.0
        * scope_context.num_ar
        * scope_context.token_residual_bytes
        / scope_context.bw_hbm
    ) * 1e6
    # 0.005 us half-ULP display rounding on comm_us + 0.005 us float sum slack.
    comm_tol_us = 2 * 0.005
    assert per_op.comm_us == pytest.approx(expected_floor_comm_us, abs=comm_tol_us)
    # Regression check for _resolve_collective_call_prior: uncalibrated XLA band
    # starts at 1x ar_base_lat_s and tops out at PER_OP_COLLECTIVE_CALL_FACTOR.
    xla_call_band = report.metrics["collective_call_us_xla"]["band"]
    assert isinstance(xla_call_band, list)
    hbm_staging_s = 2.0 * scope_context.token_residual_bytes / scope_context.bw_hbm
    expected_xla_lo_us = (
        scope_context.ar_transfer_time_s + scope_context.ar_base_lat_s + hbm_staging_s
    ) * 1e6
    extra_hop_s = max(0.0, scope_context.lat_coll_upper - scope_context.lat_coll)
    expected_xla_hi_us = (
        scope_context.ar_transfer_time_s
        + constants.PER_OP_COLLECTIVE_CALL_FACTOR
        * (scope_context.ar_base_lat_s + scope_context.ar_intra_hops * extra_hop_s)
        + hbm_staging_s
    ) * 1e6
    # 5e-5 us half-ULP at QUANTILE_DECIMALS=4 + 5e-5 us float slack.
    call_band_tol_us = 1e-4
    assert float(xla_call_band[0]) == pytest.approx(
        expected_xla_lo_us, abs=call_band_tol_us
    )
    assert float(xla_call_band[1]) == pytest.approx(
        expected_xla_hi_us, abs=call_band_tol_us
    )


def test_core_sync_checks_max_ar_slots_per_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify verify_core_limits_sync catches max_ar_slots_per_layer drift."""
    assert spec_bridge.verify_core_limits_sync() == []
    drifted_values = (
        advisor.CORE_LIMITS.max_ar_slots_per_layer - 1,
        advisor.CORE_LIMITS.max_ar_slots_per_layer + 1,
    )
    for drifted_slots in drifted_values:
        monkeypatch.setattr(
            spec_bridge,
            "CORE_LIMITS",
            dataclasses.replace(
                advisor.CORE_LIMITS,
                max_ar_slots_per_layer=drifted_slots,
            ),
        )
        messages = spec_bridge.verify_core_limits_sync()
        slot_mismatches = [
            msg for msg in messages if msg.startswith("max_ar_slots_per_layer:")
        ]
        assert len(slot_mismatches) >= 1


def test_declared_layout_collectives_shared_symmetrically_across_scopes() -> None:
    """Verify undeclared scopes share the layout's declared collective schedule."""
    kimi_batch = 16
    kimi_mesh = 32
    workload = advisor.WorkloadSpec(
        phase="decode",
        batch=kimi_batch,
        tp=kimi_mesh,
        ep=kimi_mesh,
        dcp=kimi_mesh,
    )
    report = advisor.advise("kimi-k3", workload, "tpu7x")
    by_scope = {scope.scope: scope for scope in report.scopes}
    whole_step = by_scope["whole_step"]
    single_layer = by_scope["single_layer"]
    per_op = by_scope["per_op"]
    assert single_layer.comm_us == whole_step.comm_us
    assert per_op.comm_us >= whole_step.comm_us
    assert whole_step.step_bound_us < per_op.step_bound_us
