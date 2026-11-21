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
"""Unit and integration tests for the framework-free `megakernel.core.runtime`."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.megakernel.core.megakernel_synth import TINY_TILES, synth_hybrid_spec
from vllm_torchtpu.megakernel.core import model_def, planner, profile
from vllm_torchtpu.megakernel.core import runtime as mk_rt
from vllm_torchtpu.megakernel.core.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.core.geometry import ModelGeometry
from vllm_torchtpu.megakernel.core.model_def import (
    Every,
    FromHF,
    FromHFLayerTypes,
    LMHead,
    ModelDef,
    Schedule,
    find_model,
    register_model,
)
from vllm_torchtpu.megakernel.core.modules import GDN, GQA, Embedding, MoE, Norm
from vllm_torchtpu.megakernel.core.pool.config import CacheExec, with_cache
from vllm_torchtpu.megakernel.core.profile import DEFAULT_PROFILE, Profile
from vllm_torchtpu.megakernel.core.runtime import (
    BucketPolicy,
    JaxStepMetaOps,
    align_cache_exec,
    bucket_exec_spec,
    explicit_profile_settings,
    pack_step_meta,
    parse_runtime_options,
    resolve_driver_profile,
    resolve_step_positions,
)


@pytest.fixture()
def isolated_model_registry(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(model_def, "_MODEL_REGISTRY", dict(model_def._MODEL_REGISTRY))
    monkeypatch.setattr(model_def, "_ARCH_REGISTRY", dict(model_def._ARCH_REGISTRY))
    monkeypatch.setattr(
        model_def, "_MODEL_TYPE_REGISTRY", dict(model_def._MODEL_TYPE_REGISTRY)
    )
    monkeypatch.setattr(profile, "_REGISTRY", dict(profile._REGISTRY))
    monkeypatch.setattr(profile, "_MODEL_DEFAULTS", dict(profile._MODEL_DEFAULTS))


def test_runtime_config_and_gpr_bucket_exec_spec():
    """Test `parse_runtime_options`, `resolve_driver_profile` with `BucketPolicy`,
    and `bucket_exec_spec` GPR rules (bucket parsing and selection:
    `test_bucket_policy.py`)."""
    prof = resolve_driver_profile(
        DEFAULT_PROFILE,
        fused_pool_io=False,
        fused_pool_io_max_b=16,
        fused_pool_io_min_b_hi=128,
        pool_gather=False,
    )
    assert prof.fused_pool_io is False
    assert prof.fused_pool_io_max_b == 16
    assert prof.fused_pool_io_min_b_hi == 128
    assert prof.pool_gather is False

    pol = BucketPolicy.resolve(
        prof,
        buckets=(8, 16),
        verify_buckets=(16, 32),
        speculative_tokens=0,
        buckets_explicit=True,
        verify_buckets_explicit=False,
    )
    assert pol.buckets == (8, 16)
    assert pol.verify_buckets == ()
    assert list(pol.iter_buckets()) == [(1, 8), (1, 16)]
    assert (pol.max_bucket, pol.max_verify_bucket) == (16, 0)
    assert pol.select(5, 1) == 8

    cfg = parse_runtime_options(
        {
            "buckets": [4, 8],
            "verify_buckets": [16],
            "fused_pool_io": "auto",
            "pool_gather": True,
            "mode": None,
        }
    )
    assert "enabled" not in cfg and cfg["mode"] == ""
    assert cfg["buckets"] == (4, 8)
    assert cfg["verify_buckets"] == (16,)
    owned = explicit_profile_settings(cfg)
    assert owned["buckets"] == (4, 8)
    assert owned["verify_buckets"] == (16,)
    assert owned["fused_pool_io"] == "auto"
    assert owned["pool_gather"] is True
    # `enabled` and the other runtime keys belong to the adapter's config
    # parser; a typo or a non-option key raises.
    for bad in ({"enabled": True}, {"profiel": "default"}, {"ab": True}):
        with pytest.raises(ValueError, match="unknown runtime option"):
            parse_runtime_options(bad)

    # A `Profile(pool_gather=True)` normalizes True -> "eligible" via
    # `Profile.exec_for` and never re-applies a raw True on T=1.
    g = ModelGeometry(
        L=4,
        att_every=4,
        H=256,
        NH=4,
        NKV=1,
        HD=256,
        ROT=64,
        KH=2,
        VH=4,
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=256,
        IS=256,
        moe_tp=False,
        e_slots=0,
        theta=10_000_000.0,
        zero_centered_norm=True,
    )
    ms = synth_hybrid_spec(g, R=1)
    base_ex = ExecSpec(
        R=1,
        B=4,
        T=1,
        ctx=256,
        interpret=True,
        stack_e=False,
        tiles=dataclasses.replace(TINY_TILES, bank=512 << 10),
        cache=CacheExec(paged_kv=True, num_pages=96, max_pages_per_seq=2),
    )
    gpr_prof = Profile(name="test-gpr-true", numerics="approx", pool_gather=True)
    assert gpr_prof.exec_for(base_ex, B=4, T=1, g=g).cache.pool_gather == "eligible"
    ex_t1, use_fused_t1, use_gpr_t1 = bucket_exec_spec(
        ms,
        base_ex,
        gpr_prof,
        bucket_t=4,
        tokens_per_seq=1,
        fused_pool_io=False,
        fused_pool_io_max_b=gpr_prof.fused_pool_io_max_b,
        fused_pool_io_min_b_hi=gpr_prof.fused_pool_io_min_b_hi,
    )
    assert ex_t1.cache.pool_gather is False
    assert (use_fused_t1, use_gpr_t1) == (False, False)
    # planner.plan on T=1 must succeed (would raise if pool_gather=True)
    planner.plan(ms, ex_t1)

    ex_t4, use_fused_t4, use_gpr_t4 = bucket_exec_spec(
        ms,
        base_ex,
        gpr_prof,
        bucket_t=4,
        tokens_per_seq=4,
        fused_pool_io=False,
        fused_pool_io_max_b=gpr_prof.fused_pool_io_max_b,
        fused_pool_io_min_b_hi=gpr_prof.fused_pool_io_min_b_hi,
    )
    assert ex_t4.cache.pool_gather is True
    assert (use_fused_t4, use_gpr_t4) == (False, True)
    planner.plan(ms, ex_t4)

    # Explicit base_exec pool_gather="split" is preserved when profile pool_gather=None
    base_split = with_cache(base_ex, pool_gather="split")
    ex_split_t4, _, use_gpr_split = bucket_exec_spec(
        ms,
        base_split,
        DEFAULT_PROFILE,
        bucket_t=4,
        tokens_per_seq=4,
        fused_pool_io=False,
        fused_pool_io_max_b=DEFAULT_PROFILE.fused_pool_io_max_b,
        fused_pool_io_min_b_hi=DEFAULT_PROFILE.fused_pool_io_min_b_hi,
    )
    assert ex_split_t4.cache.pool_gather is False
    assert not use_gpr_split


def test_pack_step_meta_and_resolve_step_positions_jax():
    """Test `pack_step_meta` and `resolve_step_positions` over `JaxStepMetaOps`."""
    ops = JaxStepMetaOps()
    seq_lens = jnp.asarray([5, 10], dtype=jnp.int32)
    pos_t1 = resolve_step_positions(
        None, seq_lens, num_reqs=2, tokens_per_seq=1, like=seq_lens, ops=ops
    )
    np.testing.assert_array_equal(np.asarray(pos_t1), np.array([4, 9], dtype=np.int32))

    # `seq_lens` is the length after this step (`num_computed_tokens + T`), so a
    # sequence's `T` tokens sit at `seq_len - T .. seq_len - 1`.
    pos_t4 = resolve_step_positions(
        None, seq_lens, num_reqs=2, tokens_per_seq=4, like=seq_lens, ops=ops
    )
    np.testing.assert_array_equal(
        np.asarray(pos_t4), np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=np.int32)
    )
    pos_t2_short = resolve_step_positions(
        None,
        jnp.asarray([1, 3], dtype=jnp.int32),
        num_reqs=2,
        tokens_per_seq=2,
        like=seq_lens,
        ops=ops,
    )
    np.testing.assert_array_equal(
        np.asarray(pos_t2_short), np.array([0, 0, 1, 2], dtype=np.int32)
    )

    # Short positions triggers fallback to seq_lens; M-RoPE 2D positions are preserved
    pos_short = jnp.asarray([4], dtype=jnp.int32)
    pos_from_short = resolve_step_positions(
        pos_short, seq_lens, num_reqs=2, tokens_per_seq=1, like=seq_lens, ops=ops
    )
    np.testing.assert_array_equal(
        np.asarray(pos_from_short), np.array([4, 9], dtype=np.int32)
    )
    pos_mrope = jnp.stack([pos_t4, pos_t4, pos_t4], axis=0)
    pos_from_mrope = resolve_step_positions(
        pos_mrope, seq_lens, num_reqs=2, tokens_per_seq=4, like=seq_lens, ops=ops
    )
    assert pos_from_mrope.shape == (8,)
    np.testing.assert_array_equal(np.asarray(pos_from_mrope), np.asarray(pos_t4))

    ids = jnp.asarray([11, 12, 13, 14, 21, 22, 23, 24], dtype=jnp.int32)
    bt_1d = jnp.asarray([100, 101, 200, 201], dtype=jnp.int32)
    s0 = jnp.asarray([1, 0], dtype=jnp.int32)  # seq 1 inactive (slot 0)
    c0 = jnp.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=jnp.int32)
    r0 = jnp.asarray([0, -1, 0, 0, 0, 0], dtype=jnp.int32)
    meta = pack_step_meta(
        ids,
        pos_mrope,
        bt_1d,
        ((s0, c0, r0), (s0, c0, r0), (s0, c0, r0)),
        bucket_t=8,
        tokens_per_seq=4,
        max_pages_per_seq=2,
        has_roff=True,
        use_ckpt_2d=True,
        ops=ops,
    )
    assert meta.ids_b.shape == (8,)
    assert meta.pos_b.shape == (8,)
    assert meta.pt_b.shape == (2, 2)
    assert meta.ckpt_b.shape == (3, 2, 4)
    assert meta.roff_b.shape == (3, 2)
    np.testing.assert_array_equal(
        np.asarray(meta.ids_b), np.array([11, 12, 13, 14, 0, 0, 0, 0], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        np.asarray(meta.roff_b[:, 0]), np.array([-1, -1, -1], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        np.asarray(meta.roff_b[:, 1]), np.array([0, 0, 0], dtype=np.int32)
    )

    with pytest.raises(ValueError, match="must be divisible by tokens_per_seq"):
        pack_step_meta(
            ids,
            pos_t4,
            bt_1d,
            ((s0, c0, r0), (s0, c0, r0), (s0, c0, r0)),
            bucket_t=6,
            tokens_per_seq=4,
            max_pages_per_seq=2,
            has_roff=False,
            use_ckpt_2d=False,
            ops=ops,
        )


def test_pack_step_meta_t1_read_column_and_1d_table_fallback():
    """`T=1` with a 2-D `c_raw` and `has_roff`: column 0 reads `c[s, roff[s]]`
    (`roff` clipped to the window), column 1 writes the base slot, and only
    negative read offsets are kept. A 1-D table that `s0.shape[0]` doesn't
    divide is reshaped by `max_pages_per_seq`."""
    ops = JaxStepMetaOps()
    s0 = jnp.asarray([1, 5, 7, 0], dtype=jnp.int32)
    c0 = jnp.asarray(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [0, 0, 0, 0]], dtype=jnp.int32
    )
    r0 = jnp.zeros((8,), dtype=jnp.int32).at[1].set(2).at[5].set(9).at[7].set(-1)
    bt_1d = jnp.asarray([10, 11, 12, 20, 21, 22, 30, 31, 32], dtype=jnp.int32)
    kw: dict[str, Any] = dict(
        bucket_t=4, tokens_per_seq=1, max_pages_per_seq=3, has_roff=True, ops=ops
    )
    ids, pos = jnp.arange(1, 5, dtype=jnp.int32), jnp.arange(10, 14, dtype=jnp.int32)
    meta = pack_step_meta(ids, pos, bt_1d, [(s0, c0, r0)], use_ckpt_2d=True, **kw)
    np.testing.assert_array_equal(
        np.asarray(meta.ckpt_b[0]), [[3, 1], [8, 5], [9, 7], [0, 0]]
    )
    np.testing.assert_array_equal(np.asarray(meta.roff_b[0]), [0, 0, -1, 0])
    np.testing.assert_array_equal(
        np.asarray(meta.pt_b), [[10, 11, 12], [20, 21, 22], [30, 31, 32], [0, 0, 0]]
    )
    np.testing.assert_array_equal(np.asarray(meta.ids_b), [1, 2, 3, 0])
    np.testing.assert_array_equal(np.asarray(meta.pos_b), [10, 11, 12, 0])
    # Without the 2-D window the read column is the base slot.
    meta_1d = pack_step_meta(ids, pos, bt_1d, [(s0, c0, r0)], use_ckpt_2d=False, **kw)
    np.testing.assert_array_equal(np.asarray(meta_1d.ckpt_b[0, :, 0]), [1, 5, 7, 0])
    with pytest.raises(ValueError, match="is not divisible by"):
        pack_step_meta(ids, pos, bt_1d[:7], [(s0, c0, r0)], use_ckpt_2d=True, **kw)


def test_pack_step_meta_validate_rejects_duplicate_write_slots():
    """`validate=True` rejects an active `T>1` row without `T` distinct positive
    write slots (concrete inputs only); the check doesn't run otherwise."""
    s_raw = jnp.asarray([1, 2], dtype=jnp.int32)
    c_dup = jnp.asarray([[3, 3], [4, 5]], dtype=jnp.int32)
    r_raw = jnp.zeros((4,), dtype=jnp.int32)

    def pack(c_raw: Any, *, validate: bool) -> Any:
        return pack_step_meta(
            jnp.arange(4, dtype=jnp.int32),
            jnp.arange(4, dtype=jnp.int32),
            jnp.asarray([[7], [8]], dtype=jnp.int32),
            [(s_raw, c_raw, r_raw)],
            bucket_t=4,
            tokens_per_seq=2,
            max_pages_per_seq=1,
            has_roff=True,
            use_ckpt_2d=True,
            validate=validate,
        )

    with pytest.raises(ValueError, match="2 distinct positive checkpoint slots"):
        pack(c_dup, validate=True)
    pack(jnp.asarray([[3, 6], [4, 5]], dtype=jnp.int32), validate=True)
    np.testing.assert_array_equal(
        np.asarray(pack(c_dup, validate=False).ckpt_b[0]), [[3, 3], [4, 5]]
    )
    traced = jax.jit(lambda c: pack(c, validate=True).ckpt_b)(c_dup)
    np.testing.assert_array_equal(np.asarray(traced[0]), [[3, 3], [4, 5]])
    assert isinstance(JaxStepMetaOps(), mk_rt.StepMetaOps)


def test_align_cache_exec_aligns_ctx_to_kv_blk():
    """`align_cache_exec` sets `ctx` to the largest `kv_blk` multiple the page
    table holds (never below `base_exec.ctx`) and binds the paged KV cache."""
    g = ModelGeometry(
        L=4,
        att_every=4,
        H=256,
        NH=4,
        NKV=1,
        HD=256,
        ROT=64,
        KH=2,
        VH=4,
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=256,
        IS=256,
        moe_tp=False,
        theta=10_000_000.0,
    )
    tiles = dataclasses.replace(TINY_TILES, bank=512 << 10)
    base = ExecSpec(R=1, B=4, T=1, ctx=256, interpret=True, stack_e=False)
    for kv_blk, mpps, want_ctx in ((128, 3, 384), (256, 3, 256), (256, 5, 512)):
        ex0 = dataclasses.replace(
            base, tiles=tiles.with_modules({"gqa": {"kv_blk": kv_blk}})
        )
        ex = align_cache_exec(
            ex0, DEFAULT_PROFILE, g, num_pages=96, page_size=128, max_pages_per_seq=mpps
        )
        assert ex.ctx == want_ctx, (kv_blk, mpps)
        cx = ex.cache
        assert (cx.paged_kv, cx.num_pages, cx.page_size) == (True, 96, 128)
        assert cx.max_pages_per_seq == mpps
    # A page table shorter than `base_exec.ctx` keeps `ctx`; GQA's
    # `derive_params` then rejects it (`max_pages_per_seq < ctx // page_size`).
    ex_short = align_cache_exec(
        dataclasses.replace(base, tiles=tiles.with_modules({"gqa": {"kv_blk": 128}})),
        DEFAULT_PROFILE,
        g,
        num_pages=96,
        page_size=128,
        max_pages_per_seq=1,
    )
    assert ex_short.ctx == 256


_SYNTH_ROUTING_DEFAULT_HF_CONFIG: dict[str, Any] = {
    "architectures": ["SynthMoeForCausalLM"],
    "model_type": "synth_moe",
    "vocab_size": 256,
    "hidden_size": 4096,
    "num_hidden_layers": 60,
    "num_attention_heads": 32,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "partial_rotary_factor": 0.25,
    "rope_theta": 10_000_000.0,
    "rms_norm_eps": 1e-6,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 64,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "moe_intermediate_size": 1024,
    "shared_expert_intermediate_size": 1024,
    "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 15,
}


class _SynthRoutingModel(ModelDef):
    """Synthetic hybrid `ModelDef` registering 7 HF architecture / model_type keys."""

    name = "synth_routing"
    hf_architectures = (
        "SynthMoeForConditionalGeneration",
        "SynthMoeForCausalLM",
        "SynthForConditionalGeneration",
        "SynthForCausalLM",
    )
    hf_model_type = ("synth_moe", "synth_moe_text", "synth_family")
    supported_tp_sizes = (8,)
    default_hf_config = _SYNTH_ROUTING_DEFAULT_HF_CONFIG
    geometry = FromHF(
        hidden="hidden_size",
        layers="num_hidden_layers",
        vocab="vocab_size",
        eps="rms_norm_eps",
        rope_theta="rope_theta",
        rope_partial=("partial_rotary_factor", 0.25),
    )
    attn = GQA.bind(
        prefix="self_attn",
        heads="num_attention_heads",
        kv_heads="num_key_value_heads",
        head_dim="head_dim",
        qk_norm=True,
        output_gate=True,
    )
    linear = GDN.bind(
        prefix="linear_attn",
        k_heads="linear_num_key_heads",
        v_heads="linear_num_value_heads",
        k_dim="linear_key_head_dim",
        v_dim="linear_value_head_dim",
        conv_kernel="linear_conv_kernel_dim",
    )
    moe = MoE.bind(
        prefix="mlp",
        experts="num_experts",
        top_k="num_experts_per_tok",
        inter="moe_intermediate_size",
        shared_inter="shared_expert_intermediate_size",
        shared_gate=True,
        moe_tp=False,
    )
    schedule = Schedule(
        mixer=FromHFLayerTypes(
            "layer_types",
            {"full_attention": attn, "linear_attention": linear},
        ),
        ffn=moe,
        pre_norm=Norm.bind("input_layernorm", zero_centered=True),
        post_norm=Norm.bind("post_attention_layernorm", zero_centered=True),
    )
    embed = Embedding("model.embed_tokens")
    final_norm = Norm.bind("model.norm", zero_centered=True)
    lm_head = LMHead("lm_head")


@pytest.mark.parametrize(
    ("arch", "model_type", "num_experts", "exp_found", "exp_eligible", "exp_reason"),
    [
        # 1-2. Canonical MoE model_types
        ("UnknownArch", "synth_moe", 512, True, True, "ok"),
        ("UnknownArch", "synth_moe_text", 512, True, True, "ok"),
        # 3. Shared family model_type (MoE vs dense disambiguated by validate_hf_config)
        ("UnknownArch", "synth_family", 512, True, True, "ok"),
        ("UnknownArch", "synth_family", 0, True, False, "non_moe_checkpoint"),
        # 4-5. Canonical MoE HF architectures
        ("SynthMoeForConditionalGeneration", "unknown", 512, True, True, "ok"),
        ("SynthMoeForCausalLM", "unknown", 512, True, True, "ok"),
        # 6-7. Shared family HF architectures (MoE vs dense)
        ("SynthForConditionalGeneration", "unknown", 512, True, True, "ok"),
        (
            "SynthForConditionalGeneration",
            "unknown",
            0,
            True,
            False,
            "non_moe_checkpoint",
        ),
        ("SynthForCausalLM", "unknown", 512, True, True, "ok"),
        ("SynthForCausalLM", "unknown", None, True, False, "non_moe_checkpoint"),
        # Unsupported model keys return None from find_model
        ("OtherForCausalLM", "other_arch", 512, False, False, ""),
        ("AltForCausalLM", "alt_arch", 512, False, False, ""),
    ],
)
def test_multi_key_routing_and_dense_rejection(
    isolated_model_registry,
    arch: str,
    model_type: str,
    num_experts: int | None,
    exp_found: bool,
    exp_eligible: bool,
    exp_reason: str,
):
    """Verify all 7 routing keys resolve to `_SynthRoutingModel` and reject dense."""
    register_model(
        name=_SynthRoutingModel.name,
        hf_architectures=_SynthRoutingModel.hf_architectures,
        hf_model_type=_SynthRoutingModel.hf_model_type,
        replace=True,
    )(_SynthRoutingModel)
    cfg = dict(_SynthRoutingModel.default_hf_config)
    cfg["architectures"] = [arch]
    cfg["model_type"] = model_type
    if num_experts is None:
        cfg.pop("num_experts", None)
    else:
        cfg["num_experts"] = num_experts

    found = find_model(cfg)
    if not exp_found:
        assert found is None
        return
    assert found is _SynthRoutingModel
    el = found.validate_hf_config(cfg)
    assert el.eligible is exp_eligible
    assert el.reason == exp_reason


def test_schedule_driven_validate_hf_config(isolated_model_registry):
    """`ModelDef.validate_hf_config` resolves `Schedule.ffn` across schedule forms."""
    valid_cfg = dict(_SynthRoutingModel.default_hf_config)
    assert _SynthRoutingModel.validate_hf_config(valid_cfg).eligible

    dense_cfg = {k: v for k, v in valid_cfg.items() if k != "num_experts"}
    res_dense = _SynthRoutingModel.validate_hf_config(dense_cfg)
    assert not res_dense.eligible and res_dense.reason == "non_moe_checkpoint"
    assert "_SynthRoutingModel supports MoE checkpoints only" in res_dense.message

    # Non-numeric num_experts (e.g. string "512") returns non_moe_checkpoint cleanly
    str_experts_cfg = {**valid_cfg, "num_experts": "512"}
    res_str = _SynthRoutingModel.validate_hf_config(str_experts_cfg)
    assert not res_str.eligible and res_str.reason == "non_moe_checkpoint"

    # Missing hidden_size still checks present geometry keys (e.g. head_dim)
    no_hidden_bad_hd = {"num_experts": 512, "head_dim": 64}
    res_hd = _SynthRoutingModel.validate_hf_config(no_hidden_bad_hd)
    assert not res_hd.eligible and res_hd.reason == "geometry_mismatch"
    assert "HD=64 (expected 256)" in res_hd.message

    custom_moe = MoE.bind(
        prefix="mlp",
        experts="routed_expert_count",
        top_k="num_experts_per_tok",
        inter="moe_intermediate_size",
        shared_inter="shared_expert_intermediate_size",
        shared_gate=True,
    )
    other_moe = MoE.bind(
        prefix="mlp",
        experts="alt_expert_count",
        top_k="num_experts_per_tok",
        inter="moe_intermediate_size",
        shared_inter="shared_expert_intermediate_size",
        shared_gate=True,
    )
    attn = GQA.bind(
        prefix="self_attn",
        heads="num_attention_heads",
        kv_heads="num_key_value_heads",
        head_dim="head_dim",
    )
    gdn = GDN.bind(
        prefix="linear_attn",
        k_heads="linear_num_key_heads",
        v_heads="linear_num_value_heads",
        k_dim="linear_key_head_dim",
        v_dim="linear_value_head_dim",
        conv_kernel="linear_conv_kernel_dim",
    )
    dummy_non_moe = Norm.bind("mlp_norm", zero_centered=True)

    base_synth_cfg = {
        **dense_cfg,
        "architectures": [],
        "num_hidden_layers": 4,
        "routed_expert_count": 128,
        "alt_expert_count": 64,
        "layer_types": ["linear_attention", "full_attention"] * 2,
        "ffn_types": ["dense", "moe", "dense", "moe"],
    }
    geom_spec = FromHF(
        hidden="hidden_size",
        layers="num_hidden_layers",
        eps="rms_norm_eps",
        rope_theta="rope_theta",
    )

    # 1. Schedule.ffn wrapped in Every(...)
    @register_model(name="synth_every_ffn", hf_model_type="synth_every_ffn")
    class SynthEveryFFN(ModelDef):
        default_hf_config = {**base_synth_cfg, "model_type": "synth_every_ffn"}
        geometry = geom_spec
        schedule = Schedule(
            mixer=Every(n=2, last=attn, rest=gdn),
            ffn=Every(n=2, last=custom_moe, rest=dummy_non_moe),
            pre_norm=Norm.bind("input_layernorm", zero_centered=True),
            post_norm=Norm.bind("post_attention_layernorm", zero_centered=True),
        )
        final_norm = Norm.bind("model.norm", zero_centered=True)

    assert SynthEveryFFN.validate_hf_config(SynthEveryFFN.default_hf_config).eligible
    assert not SynthEveryFFN.validate_hf_config(
        {**SynthEveryFFN.default_hf_config, "routed_expert_count": 0}
    ).eligible

    # 2. Schedule.ffn wrapped in FromHFLayerTypes(...)
    class SynthLayerTypesFFN(ModelDef):
        default_hf_config = {**base_synth_cfg, "model_type": "synth_lh_ffn"}
        geometry = geom_spec
        schedule = Schedule(
            mixer=Every(n=2, last=attn, rest=gdn),
            ffn=FromHFLayerTypes(
                "ffn_types",
                {"dense": dummy_non_moe, "moe": custom_moe},
            ),
            pre_norm=Norm.bind("input_layernorm", zero_centered=True),
            post_norm=Norm.bind("post_attention_layernorm", zero_centered=True),
        )
        final_norm = Norm.bind("model.norm", zero_centered=True)

    assert SynthLayerTypesFFN.validate_hf_config(
        SynthLayerTypesFFN.default_hf_config
    ).eligible
    assert not SynthLayerTypesFFN.validate_hf_config(
        {**SynthLayerTypesFFN.default_hf_config, "routed_expert_count": None}
    ).eligible

    # 3. Schedule.ffn as a per-layer sequence + Schedule.overrides with a second MoE key
    @register_model(name="synth_seq_ov_ffn", hf_model_type="synth_seq_ov_ffn")
    class SynthSeqOverridesFFN(ModelDef):
        default_hf_config = {**base_synth_cfg, "model_type": "synth_seq_ov_ffn"}
        geometry = geom_spec
        schedule = Schedule(
            mixer=Every(n=2, last=attn, rest=gdn),
            ffn=[dummy_non_moe, custom_moe, dummy_non_moe, dummy_non_moe],
            pre_norm=Norm.bind("input_layernorm", zero_centered=True),
            post_norm=Norm.bind("post_attention_layernorm", zero_centered=True),
            overrides={3: {"ffn": other_moe}},
        )
        final_norm = Norm.bind("model.norm", zero_centered=True)

    assert SynthSeqOverridesFFN.validate_hf_config(
        SynthSeqOverridesFFN.default_hf_config
    ).eligible
    assert not SynthSeqOverridesFFN.validate_hf_config(
        {**SynthSeqOverridesFFN.default_hf_config, "routed_expert_count": 0}
    ).eligible
    assert not SynthSeqOverridesFFN.validate_hf_config(
        {**SynthSeqOverridesFFN.default_hf_config, "alt_expert_count": 0}
    ).eligible
