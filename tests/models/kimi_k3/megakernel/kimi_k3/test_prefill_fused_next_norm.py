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

from __future__ import annotations

import inspect
import os
import subprocess
import sys
from types import SimpleNamespace
from typing import Any
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.pallas.mosaic import tpu_info
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3 import config as k3_config
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.config import KernelConfig
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels import (
    kda_a2a,
    mla_a2a,
    rank_loader,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels import (
    prefill_megakernel as pm,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels.prefill_moe import (
    fused,
    routed_16k,
    w4a8_16k,
)


def _rel_l2(got: np.ndarray, expected: np.ndarray) -> float:
    diff = got.astype(np.float64) - expected.astype(np.float64)
    denom = np.linalg.norm(expected.astype(np.float64)) + 1e-30
    return float(np.linalg.norm(diff) / denom)


def _reference_epilogue_and_next_mixture(
    reduced_bf16: np.ndarray,
    shared_bf16: np.ndarray,
    norm_w_f32: np.ndarray,
    up_w_bf16: np.ndarray,
    prefix_bf16: np.ndarray,
    blocks_bf16: np.ndarray,
    next_folded_f32: np.ndarray,
    next_norm_bf16: np.ndarray,
    *,
    live_tokens: int,
    rank: int,
    next_active: int,
    eps: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    rows, width = reduced_bf16.shape
    stripe = pm.window_stripe(rows)
    local_idx = np.arange(rows, dtype=np.int32)
    global_tok = (
        (local_idx // stripe) * (32 * stripe) + rank * stripe + (local_idx % stripe)
    )
    valid = (global_tok < live_tokens)[:, None]

    routed_f32 = reduced_bf16.astype(np.float32)
    r_inv = 1.0 / np.sqrt(np.mean(routed_f32 * routed_f32, axis=1, keepdims=True) + eps)
    normed_bf16 = jnp.asarray(
        (routed_f32 * r_inv) * norm_w_f32[0, :, :width], jnp.bfloat16
    )
    up_w_jax = jnp.asarray(up_w_bf16[0], jnp.bfloat16)
    shared_jax = jnp.asarray(shared_bf16, jnp.bfloat16)
    prefix_jax = jnp.asarray(prefix_bf16, jnp.bfloat16)
    valid_jax = jnp.asarray(valid)
    panel_outs = []
    for panel in range(14):
        c0, c1 = panel * 512, (panel + 1) * 512
        prod_p = jnp.matmul(
            normed_bf16,
            up_w_jax[:, c0:c1],
            preferred_element_type=jnp.float32,
        ).astype(jnp.bfloat16)
        val_p = jnp.where(valid_jax, prod_p + shared_jax[:, c0:c1], jnp.bfloat16(0.0))
        panel_outs.append(prefix_jax[:, c0:c1] + val_p)
    moe_out_bf16 = np.asarray(jnp.concatenate(panel_outs, axis=1))

    px_f32 = moe_out_bf16.astype(np.float32)
    fold = next_folded_f32.astype(np.float32)
    scores = [
        np.sum(px_f32 * fold, axis=1, keepdims=True)
        / np.sqrt(np.mean(px_f32 * px_f32, axis=1, keepdims=True) + eps)
    ]
    terms = [px_f32]
    for blk in range(next_active):
        b_f32 = blocks_bf16[blk].astype(np.float32)
        b_sc = np.sum(b_f32 * fold, axis=1, keepdims=True) / np.sqrt(
            np.mean(b_f32 * b_f32, axis=1, keepdims=True) + eps
        )
        scores.append(b_sc)
        terms.append(b_f32)
    score_stack = np.stack(scores, axis=0)
    max_sc = np.max(score_stack, axis=0)
    weights = np.exp(score_stack - max_sc)
    mixed_f32 = np.sum(weights * np.stack(terms, axis=0), axis=0) / np.sum(
        weights, axis=0
    )
    mixed_bf16_f32 = np.asarray(jnp.asarray(mixed_f32, jnp.bfloat16), np.float32)
    m_inv = 1.0 / np.sqrt(
        np.mean(mixed_bf16_f32 * mixed_bf16_f32, axis=1, keepdims=True) + eps
    )
    next_normed = np.asarray(
        jnp.asarray(
            mixed_bf16_f32 * m_inv * next_norm_bf16.astype(np.float32),
            jnp.bfloat16,
        )
    )
    return moe_out_bf16, next_normed


def _fake_mesh_coord_devices() -> list[SimpleNamespace]:
    devs: list[SimpleNamespace] = []
    dev_id = 0
    for z in range(4):
        for y in range(2):
            for x in range(2):
                for core in range(2):
                    devs.append(
                        SimpleNamespace(id=dev_id, coords=(x, y, z), core_on_chip=core)
                    )
                    dev_id += 1
    return devs


@pytest.mark.parametrize(
    ("seq_len", "monolithic", "dense_fp8", "fused_moe", "expected"),
    [
        (16384, True, False, True, True),
        (8192, True, False, True, False),
        (10240, True, False, True, False),
        (16384, False, False, True, False),
        (16384, True, True, True, False),
        (16384, True, False, False, False),
    ],
)
def test_should_enable_fused_next_mixture_only_when_16k_monolithic_and_bf16_dense(
    monkeypatch, seq_len, monolithic, dense_fp8, fused_moe, expected
):
    cfg = KernelConfig(
        prefill_expert_parallel=16,
        prefill_moe_16k_monolithic=monolithic,
        prefill_moe_dense_fp8=dense_fp8,
        prefill_fused_moe=fused_moe,
    )
    monkeypatch.setattr(k3_config, "active", lambda: cfg)

    assert fused.supports_fused_next_mixture(seq_len=seq_len) is expected


def test_should_reject_invalid_with_next_mixture_configurations_fail_fast(monkeypatch):
    one_dev_mesh = Mesh(np.asarray(jax.devices()[:1]), ("k3_sp",))

    monkeypatch.setattr(
        k3_config,
        "active",
        lambda: KernelConfig(
            prefill_expert_parallel=16,
            prefill_moe_16k_monolithic=True,
            prefill_moe_dense_fp8=True,
        ),
    )
    with pytest.raises(ValueError, match="prefill_moe_dense_fp8"):
        fused.make_indexed_mixture_moe_sp_ring(
            one_dev_mesh, seq_len=16384, with_next_mixture=True
        )

    monkeypatch.setattr(
        k3_config,
        "active",
        lambda: KernelConfig(
            prefill_expert_parallel=16,
            prefill_moe_16k_monolithic=True,
        ),
    )
    with pytest.raises(ValueError, match="monolithic"):
        fused.make_indexed_mixture_moe_sp_ring(
            one_dev_mesh, seq_len=10240, with_next_mixture=True
        )


@pytest.mark.parametrize(
    ("rows", "staged_hbm", "next_active"),
    [
        (128, True, 0),
        (128, True, 1),
        (256, True, 2),
        (512, True, 5),
        (128, True, 8),
        (32, False, 3),
    ],
)
def test_should_match_standalone_epilogue_and_mixture_norm_with_poisoned_blocks(
    rows: int, staged_hbm: bool, next_active: int
):
    width, hidden = 3584, 7168
    stripe = pm.window_stripe(rows)
    live_tokens = ((rows // stripe) - 1) * (32 * stripe) + (stripe - 5)
    rng = np.random.default_rng(20261114 + rows + next_active)

    reduced_np = rng.standard_normal((rows, width)).astype(np.float32) * 0.25
    shared_np = rng.standard_normal((rows, hidden)).astype(np.float32) * 0.15
    reduced_np[-5:, :] = 777.0
    shared_np[-5:, :] = -888.0

    blocks_np = rng.standard_normal((8, rows, hidden)).astype(np.float32) * 0.35
    if next_active < 8:
        blocks_np[next_active:, :, :] = 999.0

    reduced_bf16 = jnp.asarray(reduced_np, jnp.bfloat16)
    shared_bf16 = jnp.asarray(shared_np, jnp.bfloat16)
    blocks_bf16 = jnp.asarray(blocks_np, jnp.bfloat16)
    norm_w = jnp.asarray(0.8 + 0.4 * rng.random((1, 1, width)), jnp.float32)
    up_w = jnp.asarray(rng.standard_normal((1, width, hidden)) * 0.02, jnp.bfloat16)
    prefix_bf16 = jnp.asarray(rng.standard_normal((rows, hidden)) * 0.3, jnp.bfloat16)
    next_folded = jnp.asarray(rng.standard_normal((1, hidden)) * 0.2, jnp.float32)
    next_norm = jnp.asarray(0.75 + 0.5 * rng.random((1, hidden)), jnp.bfloat16)
    control = jnp.array([live_tokens, 0, 1, next_active], dtype=jnp.int32)

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    rs_spec = hbm if staged_hbm else vmem
    poll_counter = [0]

    def kernel_body(
        ctrl_r,
        red_r,
        shr_r,
        nrm_r,
        up_r,
        pfx_r,
        blk_r,
        nf_r,
        nn_r,
        moe_out_r,
        next_out_r,
        b_num_r,
        b_stats_r,
        dyn_polls_r,
    ):
        dyn_polls_r[0, 0] = 0

        def count_progress() -> None:
            poll_counter[0] += 1
            dyn_polls_r[0, 0] = dyn_polls_r[0, 0] + 1

        fused.stage_b_last_ring_overlap(
            ctrl_r[3], blk_r, nf_r, b_num_r, b_stats_r, count_progress
        )
        fused.epilogue_merge_next_norm(
            ctrl_r,
            red_r,
            shr_r,
            nrm_r,
            up_r,
            pfx_r,
            moe_out_r,
            staged_hbm=staged_hbm,
            b_num_hbm=b_num_r,
            b_stats_hbm=b_stats_r,
            next_folded_hbm=nf_r,
            next_norm_hbm=nn_r,
            next_normed_out=next_out_r,
        )

    pallas_fn = pl.pallas_call(
        kernel_body,
        out_shape=(
            jax.ShapeDtypeStruct((rows, hidden), jnp.bfloat16),
            jax.ShapeDtypeStruct((rows, hidden), jnp.bfloat16),
            jax.ShapeDtypeStruct((rows, hidden), jnp.float32),
            jax.ShapeDtypeStruct((rows, 128), jnp.float32),
            jax.ShapeDtypeStruct((1, 1), jnp.int32),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=(rs_spec, rs_spec, hbm, hbm, hbm, hbm, hbm, hbm),
            out_specs=(hbm, hbm, hbm, hbm, smem),
        ),
        interpret=True,
    )
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("k3_sp",))
    rep = P()
    run_1r = jax.shard_map(
        pallas_fn,
        mesh=mesh,
        in_specs=(rep,) * 9,
        out_specs=(rep,) * 5,
        check_vma=False,
    )
    moe_out, next_out, _, _, dyn_polls = run_1r(
        control,
        reduced_bf16,
        shared_bf16,
        norm_w,
        up_w,
        prefix_bf16,
        blocks_bf16,
        next_folded,
        next_norm,
    )

    exp_moe, exp_next = _reference_epilogue_and_next_mixture(
        np.asarray(reduced_bf16),
        np.asarray(shared_bf16),
        np.asarray(norm_w),
        np.asarray(up_w),
        np.asarray(prefix_bf16),
        np.asarray(blocks_bf16),
        np.asarray(next_folded),
        np.asarray(next_norm),
        live_tokens=live_tokens,
        rank=0,
        next_active=next_active,
    )
    standalone_mix = pm.make_mixture_norm_tiled(
        seq_len=rows, tile=min(128, rows), eps=1e-5, interpret=True
    )
    exp_standalone_next = np.asarray(
        standalone_mix(
            moe_out,
            blocks_bf16,
            next_folded,
            next_norm,
            jnp.array([next_active], dtype=jnp.int32),
        )
    )

    stripe = pm.window_stripe(rows)
    local_idx = np.arange(rows, dtype=np.int32)
    global_tok = (local_idx // stripe) * (32 * stripe) + (local_idx % stripe)
    invalid_mask = global_tok >= live_tokens
    assert int(np.sum(invalid_mask)) > 0
    np.testing.assert_array_equal(
        np.asarray(moe_out, np.float32)[invalid_mask],
        np.asarray(prefix_bf16, np.float32)[invalid_mask],
    )
    assert _rel_l2(np.asarray(moe_out, np.float32), exp_moe.astype(np.float32)) < 5e-4
    assert _rel_l2(np.asarray(next_out, np.float32), exp_next.astype(np.float32)) < 5e-4
    assert (
        _rel_l2(
            np.asarray(next_out, np.float32),
            exp_standalone_next.astype(np.float32),
        )
        < 5e-4
    )
    if rows == 512:
        assert poll_counter[0] == 5
        assert int(np.asarray(dyn_polls)[0, 0]) >= 1 + 4 * max(1, next_active * 2)

    wrong_active = 0 if next_active > 0 else 1
    _, wrong_next = _reference_epilogue_and_next_mixture(
        np.asarray(reduced_bf16),
        np.asarray(shared_bf16),
        np.asarray(norm_w),
        np.asarray(up_w),
        np.asarray(prefix_bf16),
        np.asarray(blocks_bf16),
        np.asarray(next_folded),
        np.asarray(next_norm),
        live_tokens=live_tokens,
        rank=0,
        next_active=wrong_active,
    )
    assert (
        _rel_l2(np.asarray(next_out, np.float32), wrong_next.astype(np.float32)) > 0.1
    )


def test_should_expose_optional_last_ring_overlap_fn_on_routed_16k():
    sig = inspect.signature(routed_16k.routed_experts_16k)
    param = sig.parameters["last_ring_overlap_fn"]
    assert param.kind == inspect.Parameter.KEYWORD_ONLY
    assert param.default is None


def test_should_bypass_mla_mixture_norm_when_pre_normalized_supplied(monkeypatch):
    norm_calls: list[int] = []
    mla_inputs: list[jax.Array] = []

    def fake_mla_builder(_mesh: Any, **_kwargs: Any):
        def run_mla(local_hidden, prefix, *_rest):
            mla_inputs.append(local_hidden)
            return local_hidden + prefix, jnp.zeros((32, 16, 576), jnp.bfloat16)

        return run_mla

    def fake_mixture(prefix, _blocks, _res, _norm, active):
        norm_calls.append(int(np.asarray(active).reshape(-1)[0]))
        return prefix * jnp.bfloat16(2.0)

    monkeypatch.setattr(pm, "_config", lambda: KernelConfig(prefill_mla_a2a=True))
    monkeypatch.setattr(mla_a2a, "make_indexed_mla_layer_sp_a2a", fake_mla_builder)

    mesh = Mesh(np.asarray(jax.devices()[:1]), ("k3_sp",))
    step_fn = pm.make_indexed_mla_step(
        mesh,
        seq_len=512,
        segments=1,
        prefix_capacity=16,
        flash_tile=256,
        mixture=fake_mixture,
    )
    prefix = jnp.ones((1, 16, 8), jnp.bfloat16)
    pre_norm = jnp.full((1, 16, 8), 3.5, jnp.bfloat16)

    out_unfused, _ = step_fn(
        prefix,
        None,
        None,
        None,
        jnp.array([2], jnp.int32),
        None,
        None,
        None,
        None,
        None,
    )
    assert norm_calls == [2]
    np.testing.assert_allclose(np.asarray(out_unfused, np.float32), 3.0)

    out_fused, _ = step_fn(
        prefix,
        None,
        None,
        None,
        jnp.array([2], jnp.int32),
        None,
        None,
        None,
        None,
        None,
        pre_normalized=pre_norm,
    )
    assert norm_calls == [2]
    np.testing.assert_allclose(np.asarray(out_fused, np.float32), 4.5)


@pytest.mark.parametrize("route_counts", [False, True])
@pytest.mark.parametrize("pair_idx", [0, 1, 2])
def test_should_preserve_commit_block_contract_across_group8_and_stack_transitions(
    monkeypatch, route_counts: bool, pair_idx: int
):
    cfg = KernelConfig(
        prefill_expert_parallel=16,
        prefill_moe_16k_monolithic=True,
        prefill_fused_moe=True,
        prefill_kda_a2a=True,
        prefill_mla_a2a=True,
        prefill_route_counts=route_counts,
    )
    monkeypatch.setattr(k3_config, "active", lambda: cfg)
    monkeypatch.setattr(pm, "_config", lambda: cfg)

    rows, hidden = 4, 7168
    seq_len = 32 * rows
    counters = {"kda_norm": 0, "mla_norm": 0, "moe_next": 0, "moe_plain": 0}

    def ref_mix(
        prefix: jax.Array,
        blocks: jax.Array,
        res: jax.Array,
        nrm: jax.Array,
        act: jax.Array,
    ) -> jax.Array:
        act_cnt = act.reshape(())
        px_f = prefix.astype(jnp.float32)
        fold = res.astype(jnp.float32)
        rm = jnp.sum(px_f * fold, axis=-1, keepdims=True) * jax.lax.rsqrt(
            jnp.mean(px_f * px_f, axis=-1, keepdims=True) + 1e-5
        )
        num = px_f
        den = jnp.ones_like(rm)
        for b in range(8):
            use = (b < act_cnt).astype(jnp.float32)
            bv = blocks[:, b].astype(jnp.float32)
            sc = jnp.sum(bv * fold, axis=-1, keepdims=True) * jax.lax.rsqrt(
                jnp.mean(bv * bv, axis=-1, keepdims=True) + 1e-5
            )
            nm = jnp.where(b < act_cnt, jnp.maximum(rm, sc), rm)
            os_ = jnp.exp(rm - nm)
            ns_ = jnp.exp(sc - nm) * use
            num = num * os_ + bv * ns_
            den = den * os_ + ns_
            rm = nm
        mixed = (num / den).astype(jnp.bfloat16).astype(jnp.float32)
        rvar = jax.lax.rsqrt(jnp.mean(mixed * mixed, axis=-1, keepdims=True) + 1e-5)
        return (mixed * rvar * nrm.astype(jnp.float32)).astype(jnp.bfloat16)

    def contract_kda_builder(_mesh: Any, **_kw: Any):
        def run_kda(
            prefix,
            blocks,
            folded,
            norm,
            weights,
            rec,
            hist,
            active,
            _length,
            slot,
            _seg,
            *,
            pre_normalized=None,
        ):
            if pre_normalized is None:
                counters["kda_norm"] += 1
                normed = ref_mix(prefix, blocks, folded, norm, active)
            else:
                normed = pre_normalized
            w_slot = jax.lax.dynamic_index_in_dim(
                weights["scale"], slot[0], axis=1, keepdims=False
            )
            out = (
                prefix.astype(jnp.float32) + normed.astype(jnp.float32) * w_slot
            ).astype(jnp.bfloat16)
            stub = jnp.zeros((32, 0, 384), jnp.bfloat16)
            return (
                normed,
                normed,
                stub,
                rec + 1.0,
                hist + jnp.bfloat16(1.0),
                out,
                rec + 2.0,
                hist + jnp.bfloat16(2.0),
            )

        return run_kda

    def contract_mla_step_builder(_mesh: Any, **_kw: Any):
        def run_mla(
            prefix,
            blocks,
            attn_res,
            attn_norm,
            attn_active,
            mla_families,
            _seg,
            _len,
            slot,
            _arena,
            *,
            pre_normalized=None,
        ):
            if pre_normalized is None:
                counters["mla_norm"] += 1
                normed = ref_mix(prefix, blocks, attn_res, attn_norm, attn_active)
            else:
                normed = pre_normalized
            w_slot = jax.lax.dynamic_index_in_dim(
                mla_families["scale"], slot[0], axis=1, keepdims=False
            )
            out = (
                prefix.astype(jnp.float32) + normed.astype(jnp.float32) * w_slot
            ).astype(jnp.bfloat16)
            lat = normed[:, :, :576]
            return out, lat

        return run_mla

    def contract_moe_builder(
        _mesh: Any, *, seq_len: int = 8192, with_next_mixture: bool = False
    ):
        del seq_len

        def run_plain(prefix, blocks, mlp_res, ffn_norm, active, families, _len, slot):
            counters["moe_plain"] += 1
            normed = ref_mix(prefix, blocks, mlp_res, ffn_norm, active)
            w_slot = jax.lax.dynamic_index_in_dim(
                families["scale"], slot[0], axis=1, keepdims=False
            )
            out = (
                prefix.astype(jnp.float32) + normed.astype(jnp.float32) * w_slot
            ).astype(jnp.bfloat16)
            if route_counts:
                return out, jnp.full((32, 56), slot[0] + 1, jnp.int32)
            return out

        def run_next(
            prefix,
            blocks,
            mlp_res,
            ffn_norm,
            active,
            families,
            length,
            slot,
            next_folded,
            next_norm,
            next_active,
        ):
            counters["moe_next"] += 1
            moe_res = run_plain(
                prefix, blocks, mlp_res, ffn_norm, active, families, length, slot
            )
            counters["moe_plain"] -= 1
            moe_out = moe_res[0] if route_counts else moe_res
            next_normed = ref_mix(moe_out, blocks, next_folded, next_norm, next_active)
            return moe_res, next_normed

        return run_next if with_next_mixture else run_plain

    def contract_dense_builder(_mesh: Any, **_kw: Any):
        return lambda ffn_in, after_attn, _gu, _dn, _len: (
            after_attn.astype(jnp.float32) + ffn_in.astype(jnp.float32) * 0.1
        ).astype(jnp.bfloat16)

    def contract_mix_sp(_mesh: Any, **_kw: Any):
        return ref_mix

    monkeypatch.setattr(pm, "make_norm_gather_kda_sp", contract_kda_builder)
    monkeypatch.setattr(pm, "make_indexed_mla_step", contract_mla_step_builder)
    monkeypatch.setattr(pm, "make_indexed_mixture_moe_sp_ring", contract_moe_builder)
    monkeypatch.setattr(pm, "make_dense_ffn_sp", contract_dense_builder)
    monkeypatch.setattr(pm, "make_mixture_norm_sp", contract_mix_sp)
    monkeypatch.setattr(pm, "NamedSharding", lambda _m, _p: None)
    monkeypatch.setattr(pm, "check_window", lambda *_a, **_k: None)

    fake_mesh = SimpleNamespace(size=32, axis_names=("k3_sp",), devices=np.zeros((32,)))
    rng = np.random.default_rng(900 + pair_idx + int(route_counts))
    prefix = jnp.asarray(rng.standard_normal((32, rows, hidden)) * 0.2, jnp.bfloat16)
    blocks = jnp.asarray(rng.standard_normal((32, 8, rows, hidden)) * 0.2, jnp.bfloat16)
    common = {
        "attn_res": jnp.asarray(
            rng.standard_normal((32, 93, 1, hidden)) * 0.2, jnp.float32
        ),
        "attn_norm": jnp.asarray(
            0.8 + 0.4 * rng.random((32, 93, 1, hidden)), jnp.bfloat16
        ),
        "mlp_res": jnp.asarray(
            rng.standard_normal((32, 93, 1, hidden)) * 0.2, jnp.float32
        ),
        "ffn_norm": jnp.asarray(
            0.8 + 0.4 * rng.random((32, 93, 1, hidden)), jnp.bfloat16
        ),
        "dense_gu": jnp.zeros((32, 1, 1), jnp.bfloat16),
        "dense_down": jnp.zeros((32, 1, 1), jnp.bfloat16),
    }
    kda_fam = {
        "scale": jnp.asarray(0.05 + 0.05 * rng.random((32, 69, 1, 1)), jnp.float32)
    }
    mla_fam = {
        "scale": jnp.asarray(0.05 + 0.05 * rng.random((32, 24, 1, 1)), jnp.float32)
    }
    moe_fam = {
        "scale": jnp.asarray(0.05 + 0.05 * rng.random((32, 92, 1, 1)), jnp.float32)
    }
    kda_state = (
        jnp.zeros((32, 69, 2, 3, 4, 4), jnp.float32),
        jnp.zeros((32, 69, 2, 3, 3, 3, 4), jnp.bfloat16),
        jnp.zeros((32, 69, 2, 3, 4, 4), jnp.float32),
        jnp.zeros((32, 69, 2, 3, 3, 3, 4), jnp.bfloat16),
    )
    mla_state = (jnp.zeros((32, 24, rows, 576), jnp.bfloat16),)
    length = jnp.array([seq_len], jnp.int32)
    seg_desc = jnp.zeros((6,), jnp.int32)
    arena = pm.CpPrefix(
        pools=(jnp.zeros((1,)), jnp.zeros((1,))),
        ids=jnp.zeros((1,), jnp.int32),
        local=jnp.zeros((1,), jnp.int32),
        counts=jnp.zeros((1,), jnp.int32),
        order=jnp.zeros((1,), jnp.int32),
    )
    route_init = jnp.zeros((32, 92, 56), jnp.int32) if route_counts else None

    counters.update(kda_norm=0, mla_norm=0, moe_next=0, moe_plain=0)
    g8_base = pm.make_regular_group8_sp_indexed(
        fake_mesh,
        seq_len=seq_len,
        segments=2,
        prefix_capacity=16,
        fuse_next_mixture=False,
    )
    out_g8_base = g8_base(
        prefix,
        blocks,
        common,
        kda_fam,
        mla_fam,
        moe_fam,
        kda_state,
        mla_state,
        length,
        jnp.array([pair_idx], jnp.int32),
        seg_desc,
        arena,
        route_init,
    )
    assert counters == {"kda_norm": 3, "mla_norm": 1, "moe_next": 0, "moe_plain": 4}

    counters.update(kda_norm=0, mla_norm=0, moe_next=0, moe_plain=0)
    g8_fused = pm.make_regular_group8_sp_indexed(
        fake_mesh,
        seq_len=seq_len,
        segments=2,
        prefix_capacity=16,
        fuse_next_mixture=True,
    )
    out_g8_fused = g8_fused(
        prefix,
        blocks,
        common,
        kda_fam,
        mla_fam,
        moe_fam,
        kda_state,
        mla_state,
        length,
        jnp.array([pair_idx], jnp.int32),
        seg_desc,
        arena,
        route_init,
    )
    assert counters == {"kda_norm": 1, "mla_norm": 0, "moe_next": 7, "moe_plain": 1}

    for got_leaf, exp_leaf in zip(
        jax.tree.leaves(out_g8_fused), jax.tree.leaves(out_g8_base), strict=True
    ):
        np.testing.assert_array_equal(
            np.asarray(got_leaf, np.float32), np.asarray(exp_leaf, np.float32)
        )

    if pair_idx == 0:
        counters.update(kda_norm=0, mla_norm=0, moe_next=0, moe_plain=0)
        st_base = pm.make_prefill_stack_sp_indexed(
            fake_mesh,
            seq_len=seq_len,
            depth=4,
            segments=2,
            prefix_capacity=16,
            unroll=True,
            tap_layer=3,
            fuse_next_mixture=False,
        )
        out_st_base = st_base(
            prefix,
            common,
            kda_fam,
            mla_fam,
            moe_fam,
            kda_state,
            mla_state,
            length,
            seg_desc,
            arena,
            route_init,
        )
        assert counters == {"kda_norm": 3, "mla_norm": 1, "moe_next": 0, "moe_plain": 3}

        counters.update(kda_norm=0, mla_norm=0, moe_next=0, moe_plain=0)
        st_fused = pm.make_prefill_stack_sp_indexed(
            fake_mesh,
            seq_len=seq_len,
            depth=4,
            segments=2,
            prefix_capacity=16,
            unroll=True,
            tap_layer=3,
            fuse_next_mixture=True,
        )
        out_st_fused = st_fused(
            prefix,
            common,
            kda_fam,
            mla_fam,
            moe_fam,
            kda_state,
            mla_state,
            length,
            seg_desc,
            arena,
            route_init,
        )
        assert counters == {"kda_norm": 2, "mla_norm": 0, "moe_next": 2, "moe_plain": 1}
        for got_leaf, exp_leaf in zip(
            jax.tree.leaves(out_st_fused), jax.tree.leaves(out_st_base), strict=True
        ):
            np.testing.assert_array_equal(
                np.asarray(got_leaf, np.float32), np.asarray(exp_leaf, np.float32)
            )


def test_should_lower_32rank_16k_moe_with_next_mixture_and_kda_prenorm_to_mlir(
    monkeypatch,
):
    cfg = KernelConfig(
        decode_expert_parallel=16,
        prefill_expert_parallel=16,
        prefill_expert_plan="plan_ep16_real_hotfirst.npz",
        prefill_replica_plan="replica_plan_ep16_real_plain4.npz",
        kd_replica_plan="replica_plan_ep16_real_plain4.npz",
        decode_expert_banks=False,
        prefill_prefix_fp8=True,
        prefill_fused_moe=True,
        prefill_moe_16k_monolithic=True,
        prefill_moe_w4a8=True,
        prefill_kda_a2a=True,
        prefill_kda_ring_set=True,
        prefill_mla_a2a=True,
        prefill_a2a_chip_shard=True,
    )
    monkeypatch.setattr(k3_config, "active", lambda: cfg)
    monkeypatch.setattr(pm, "_config", lambda: cfg)
    monkeypatch.setattr(kda_a2a, "_config", lambda: cfg)
    monkeypatch.setattr(rank_loader, "_config", lambda: cfg)

    cpu_devs = jax.devices("cpu")
    if len(cpu_devs) < 32:
        env = dict(os.environ)
        env["JAX_PLATFORMS"] = "cpu"
        env["XLA_FLAGS"] = (
            env.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=32"
        ).strip()
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--no-header",
                f"{__file__}::test_should_lower_32rank_16k_moe_with_next_mixture_and_kda_prenorm_to_mlir",
            ],
            env=env,
            check=True,
        )
        return

    mesh = Mesh(np.asarray(cpu_devs[:32]), ("k3_sp",))
    coord_devs = _fake_mesh_coord_devices()
    orig_ring_tables = pm.ring_tables
    monkeypatch.setattr(
        pm,
        "ring_tables",
        lambda _devs, token_ranks, *args, **kwargs: orig_ring_tables(
            coord_devs, token_ranks, *args, **kwargs
        ),
    )
    tpu_7x = tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_7X, 1)
    sharded = jax.sharding.NamedSharding(mesh, P("k3_sp"))
    replicated = jax.sharding.NamedSharding(mesh, P())

    def sds(shape: tuple[int, ...], dtype: Any, shd: Any) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct(shape, dtype, sharding=shd)

    rep_slots = rank_loader.prefill_replica_slots()
    le = 56 + rep_slots
    moe_next = fused.make_indexed_mixture_moe_sp_ring(
        mesh, seq_len=16384, with_next_mixture=True
    )
    moe_args = (
        sds((32, 512, 7168), jnp.bfloat16, sharded),
        sds((32, 8, 512, 7168), jnp.bfloat16, sharded),
        sds((32, 1, 7168), jnp.float32, sharded),
        sds((32, 1, 7168), jnp.bfloat16, sharded),
        sds((1,), jnp.int32, replicated),
        {
            "router": sds((32, 8, 7168, 896), jnp.bfloat16, sharded),
            "router_bias": sds((32, 8, 1, 896), jnp.float32, sharded),
            "latent_down": sds((32, 8, 7168, 4096), jnp.bfloat16, sharded),
            "shared_gu": sds((32, 8, 7168, 6144), jnp.bfloat16, sharded),
            "shared_down": sds((32, 8, 3072, 7168), jnp.bfloat16, sharded),
            "expert_gu": sds((32, 8, le, 448, 3072), jnp.uint32, sharded),
            "expert_gus": sds((32, 8, le, 112, 3072), jnp.uint8, sharded),
            "expert_down": sds((32, 8, le, 192, 3584), jnp.uint32, sharded),
            "expert_ds": sds((32, 8, le, 48, 3584), jnp.uint8, sharded),
            "latent_norm": sds((32, 8, 1, 4096), jnp.float32, sharded),
            "latent_up": sds((32, 8, 3584, 7168), jnp.bfloat16, sharded),
            "prefill_replica_table": sds((32, 8, 256), jnp.int32, sharded),
        },
        sds((1,), jnp.int32, replicated),
        sds((1,), jnp.int32, replicated),
        sds((32, 1, 7168), jnp.float32, sharded),
        sds((32, 1, 7168), jnp.bfloat16, sharded),
        sds((1,), jnp.int32, replicated),
    )

    kda_fn = kda_a2a.make_norm_gather_kda_sp_a2a(mesh, seq_len=16384, segments=4)
    kda_weights = {
        "k_gate": sds((32, 6, 7168, 128), jnp.bfloat16, sharded),
        "k_fb": sds((32, 6, 128, 384), jnp.bfloat16, sharded),
        "k_projection": sds((32, 6, 7168, 1664), jnp.bfloat16, sharded),
        "k_conv": sds((32, 6, 4, 3, 3, 128), jnp.float32, sharded),
        "k_a_log": sds((32, 6, 3, 1), jnp.float32, sharded),
        "k_dt": sds((32, 6, 3, 128), jnp.float32, sharded),
        "k_norm": sds((32, 6, 8, 128), jnp.float32, sharded),
        "k_o": sds((32, 6, 12288, 3584), jnp.bfloat16, sharded),
    }

    with (
        mock.patch.object(tpu_info, "get_tpu_info", return_value=tpu_7x),
        mock.patch.object(pltpu, "get_tpu_info", return_value=tpu_7x),
        mock.patch("jax._src.tpu_info.get_tpu_info", return_value=tpu_7x),
    ):
        moe_mlir = str(
            moe_next.trace(*moe_args)
            .lower(lowering_platforms=("tpu",))
            .compiler_ir("stablehlo")
        )
        assert "fused_mixture_next_moe_pallas_16384" in moe_mlir
        assert "mixture_norm_tiled" in moe_mlir
        assert jnp.float8_e4m3fn == w4a8_16k.LATENT_E4M3_DTYPE

        kda_pre_fn = jax.jit(
            lambda px, pn, w, r, h, ln, sl, sd: kda_fn(
                px,
                None,
                None,
                None,
                w,
                r,
                h,
                None,
                ln,
                sl,
                sd,
                pre_normalized=pn,
            )
        )
        kda_mlir = str(
            kda_pre_fn.trace(
                sds((32, 512, 7168), jnp.bfloat16, sharded),
                sds((32, 512, 7168), jnp.bfloat16, sharded),
                kda_weights,
                sds((32, 4, 3, 128, 128), jnp.float32, sharded),
                sds((32, 4, 3, 3, 3, 128), jnp.bfloat16, sharded),
                sds((1,), jnp.int32, replicated),
                sds((1,), jnp.int32, replicated),
                sds((12,), jnp.int32, replicated),
            )
            .lower(lowering_platforms=("tpu",))
            .compiler_ir("stablehlo")
        )
        assert "kimi_prefill_gather_kda_a2a_prenorm" in kda_mlir
        assert "kimi_prefill_norm_gather_kda_a2a" not in kda_mlir


def test_should_respect_dynamic_config_active_and_legacy_moe_stub_when_default_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fused,
        "make_indexed_mixture_moe_sp_ring",
        lambda mesh, *, seq_len: ("legacy_stub", seq_len),
    )
    monkeypatch.setattr(
        pm,
        "make_mixture_norm_sp",
        lambda mesh, *, seq_len: lambda *a: jnp.int32(seq_len),
    )
    monkeypatch.setattr(
        pm,
        "make_indexed_moe_sp_ring",
        lambda mesh, *, seq_len: lambda h, *a: jnp.int32(100000) + h,
    )
    monkeypatch.setattr(
        k3_config,
        "active",
        lambda: KernelConfig(prefill_expert_parallel=16, prefill_fused_moe=False),
    )
    disabled = pm.make_indexed_mixture_moe_sp_ring(None, seq_len=16384)
    z = jnp.int32(0)
    assert int(disabled(z, z, z, z, z, z, z, z)) == 116384

    monkeypatch.setattr(
        k3_config,
        "active",
        lambda: KernelConfig(prefill_expert_parallel=16, prefill_fused_moe=True),
    )
    assert pm.make_indexed_mixture_moe_sp_ring(None, seq_len=16384) == (
        "legacy_stub",
        16384,
    )
