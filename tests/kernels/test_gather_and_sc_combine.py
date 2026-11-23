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
"""Pallas interpret and TPU tests for gather_gmm V2, dense expert, and SC v4."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental import moe_dense_expert
from vllm_torchtpu.kernels.megablox import gather_gmm
from vllm_torchtpu.kernels.megablox.gmm_v2 import apply_gated_act
from vllm_torchtpu.kernels.quantized_matmul import util
from vllm_torchtpu.kernels.sparse_core import ragged_gather_reduce_v4


def _quantize_experts(w1_f: jax.Array, w2_f: jax.Array):
    w1_l, s1_l, w2_l, s2_l = [], [], [], []
    for e in range(w1_f.shape[0]):
        q1, s1 = util.quantize_tensor(w1_f[e], jnp.float8_e4m3fn, dim=1)
        q2, s2 = util.quantize_tensor(w2_f[e], jnp.float8_e4m3fn, dim=1)
        w1_l.append(q1.T)
        s1_l.append(s1.squeeze(-1).astype(jnp.float32))
        w2_l.append(q2.T)
        s2_l.append(s2.squeeze(-1).astype(jnp.float32))
    return jnp.stack(w1_l), jnp.stack(s1_l), jnp.stack(w2_l), jnp.stack(s2_l)


def _reference_per_token_fp8_moe(hidden, w1, w1_s, w2, w2_s, topk_ids, topk_w, act):
    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    n, h = hidden.shape
    inter = w2.shape[1]
    x32 = hidden.astype(jnp.float32)
    xs = jnp.max(jnp.abs(x32), axis=1, keepdims=True) / fp8_max
    xq = (
        jnp.clip(x32 * jnp.where(xs == 0, 0.0, 1.0 / xs), -fp8_max, fp8_max)
        .astype(jnp.bfloat16)
        .astype(jnp.float8_e4m3fn)
    )
    out = jnp.zeros((n, h), dtype=jnp.float32)
    for t in range(n):
        for k in range(topk_ids.shape[1]):
            e, wt = int(topk_ids[t, k]), float(topk_w[t, k])
            if e < 0 or wt == 0.0:
                continue
            gu = (
                jnp.dot(xq[t : t + 1], w1[e], preferred_element_type=jnp.float32)
                * xs[t : t + 1]
                * w1_s[e].reshape(1, -1)
            ).astype(jnp.bfloat16)
            mid = apply_gated_act(gu[:, :inter], gu[:, inter:], act)
            ms = (
                jnp.max(jnp.abs(mid), axis=1, keepdims=True).astype(jnp.float32)
                / fp8_max
            )
            mq = jnp.clip(
                mid.astype(jnp.float32) * jnp.where(ms == 0, 0.0, 1.0 / ms),
                -fp8_max,
                fp8_max,
            ).astype(jnp.float8_e4m3fn)
            down = (
                jnp.dot(mq, w2[e], preferred_element_type=jnp.float32)
                * ms
                * w2_s[e].reshape(1, -1)
            ).astype(jnp.bfloat16)
            out = out.at[t].add(down[0].astype(jnp.float32) * wt)
    return out.astype(jnp.bfloat16)


def test_gather_gmm_calibrated_vmem_and_smem_caps() -> None:
    vmem = gather_gmm.gather_fused_moe_v2_vmem_bytes(2816, 768)
    assert 0 < vmem < int(64 * 1024 * 1024 * gather_gmm.VMEM_LIMIT_FRACTION)
    assert (gather_gmm.MAX_COMBINE_SMEM_BYTES, gather_gmm.MAX_COMBINE_ROUTES) == (
        256 * 1024,
        64 * 1024,
    )


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("act", ["silu", "gelu_tanh"])
def test_gather_gmm_v2_and_chunked_combine_match_reference(
    masked: bool, act: str
) -> None:
    tokens, hidden, inter, num_experts, topk = 32, 256, 128, 4, 2
    keys = jax.random.split(jax.random.key(808 + int(masked)), 5)
    x = jax.random.normal(keys[0], (tokens, hidden), dtype=jnp.bfloat16)
    w1, w1s, w2, w2s = _quantize_experts(
        jax.random.normal(keys[1], (num_experts, 2 * inter, hidden)) * 0.2,
        jax.random.normal(keys[2], (num_experts, hidden, inter)) * 0.2,
    )
    w1_s, w2_s = w1s[:, None, :], w2s[:, None, :]
    topk_ids = jax.random.randint(keys[3], (tokens, topk), 0, num_experts, jnp.int32)
    topk_w = jax.nn.softmax(jax.random.normal(keys[4], (tokens, topk)), -1).astype(
        jnp.bfloat16
    )
    if masked:
        drop = (jnp.arange(tokens * topk).reshape(tokens, topk) % 4) == 1
        topk_ids = jnp.where(drop, -1, topk_ids)
        topk_w = jnp.where(drop, jnp.bfloat16(0.0), topk_w)
    flat_ids = topk_ids.reshape(-1)
    sort_keys = jnp.where(flat_ids >= 0, flat_ids, num_experts)
    order = jnp.argsort(sort_keys, stable=True)
    positions = jnp.argsort(order).reshape(tokens, topk)
    if masked:
        positions = jnp.where(topk_ids >= 0, positions, -1)
    tok_sorted = (order // topk).astype(jnp.int32)
    group_sizes = jnp.bincount(sort_keys, length=num_experts + 1)[:num_experts]
    if masked:
        tok_sorted = tok_sorted.at[int(jnp.sum(group_sizes)) :].set(-999)

    packed_x = gather_gmm.quantize_pack_rows(x, tile_tokens=16, interpret=True)
    y_packed = gather_gmm.gather_fused_moe_v2(
        packed_x,
        tok_sorted,
        group_sizes,
        w1,
        w1_s,
        w2,
        w2_s,
        act=act,
        tile_m=32,
        height_step=16,
        issue_group=8,
        interpret=True,
    )
    out = gather_gmm.gather_combine_in_chunks(
        y_packed,
        positions,
        topk_w,
        hidden_size=hidden,
        max_routes=32,
        tile_tokens=16,
        masked=masked,
        interpret=True,
    )
    ref = _reference_per_token_fp8_moe(x, w1, w1_s, w2, w2_s, topk_ids, topk_w, act)
    out_f32, ref_f32 = np.asarray(out, np.float32), np.asarray(ref, np.float32)
    rel_l2 = float(np.linalg.norm(out_f32 - ref_f32) / np.linalg.norm(ref_f32))
    cos = float(
        np.sum(out_f32 * ref_f32) / (np.linalg.norm(out_f32) * np.linalg.norm(ref_f32))
    )
    assert rel_l2 < 3.5e-2 and cos > 0.999
    np.testing.assert_allclose(
        out_f32, ref_f32, rtol=5e-2, atol=6e-2 * float(np.max(np.abs(ref_f32)))
    )


def test_dense_expert_moe_gelu_tanh_ep_slice_interpret(monkeypatch) -> None:
    tokens, hidden, inter, local_e, topk = 16, 256, 128, 2, 2
    keys = jax.random.split(jax.random.key(919), 4)
    x = jax.random.normal(keys[0], (tokens, hidden), dtype=jnp.bfloat16)
    w1, s1_raw, w2, s2_raw = _quantize_experts(
        jax.random.normal(keys[1], (local_e, 2 * inter, hidden)) * 0.1,
        jax.random.normal(keys[2], (local_e, hidden, inter)) * 0.1,
    )
    s1, s2 = s1_raw[:, None, None, :], s2_raw[:, None, None, :]
    base_ids = jnp.array(
        [[0, 2], [2, 3], [1, 0], [3, 1], [2, 0], [1, 3], [0, 1], [2, 3]], jnp.int32
    )
    global_ids = jnp.concatenate([base_ids, base_ids], axis=0)
    weights = jnp.full((tokens, topk), 0.5, dtype=jnp.bfloat16)
    can_kw = dict(
        activation="gelu_tanh", rhs_quant_dtype=None, skip_padded_tokens=False
    )
    if jax.default_backend() != "tpu":
        assert not moe_dense_expert.can_use_dense_expert(
            x, w1, w2, s1, s2, None, None, **can_kw
        )
    monkeypatch.setattr(
        moe_dense_expert.pltpu,
        "get_tpu_info",
        lambda: SimpleNamespace(
            fp8_ops_per_second=1, mxu_column_size=128, vmem_capacity_bytes=64 << 20
        ),
    )
    assert moe_dense_expert.can_use_dense_expert(
        x, w1, w2, s1, s2, None, None, **can_kw
    )
    out = moe_dense_expert.dense_expert_moe(
        x,
        w1,
        w2,
        s1,
        s2,
        weights,
        global_ids,
        activation="gelu_tanh",
        experts_start=jnp.int32(2),
        interpret=True,
    )
    local_ids = global_ids - 2
    valid = (local_ids >= 0) & (local_ids < local_e)
    ref = _reference_per_token_fp8_moe(
        x,
        w1,
        s1.reshape(local_e, 1, 2 * inter),
        w2,
        s2.reshape(local_e, 1, hidden),
        jnp.where(valid, local_ids, -1),
        jnp.where(valid, weights, jnp.bfloat16(0.0)),
        "gelu_tanh",
    )
    for row in (2, 6):
        np.testing.assert_array_equal(
            np.asarray(out[row], np.float32), np.zeros((hidden,), np.float32)
        )
    out_f32, ref_f32 = np.asarray(out, np.float32), np.asarray(ref, np.float32)
    assert float(np.linalg.norm(out_f32 - ref_f32) / np.linalg.norm(ref_f32)) < 3e-2
    np.testing.assert_allclose(
        out_f32, ref_f32, rtol=5e-2, atol=6e-2 * float(np.max(np.abs(ref_f32)))
    )


def test_sparsecore_combine_v4_unified_modes_and_dma_clamp(monkeypatch) -> None:
    for n, want in ((2048, 64), (1024, 32), (256, 16)):
        assert ragged_gather_reduce_v4.pick_tile_tokens(n, reduce_group_size=8) == want
    num_tokens, k, x_rows = 64, 4, 32
    raw_idx = jnp.array(
        [
            -1 if i % 3 == 0 else (999 if i % 5 == 0 else i % x_rows)
            for i in range(num_tokens * k)
        ],
        jnp.int32,
    )
    valid_mask, weights = raw_idx >= 0, jnp.ones((num_tokens * k,), jnp.float32)
    gidx_seg, *_ = ragged_gather_reduce_v4._prepare_segmented(
        raw_idx, weights, valid_mask, k, 16, x_rows
    )
    gidx_grp, w_grp, *_ = ragged_gather_reduce_v4._prepare_grouped(
        raw_idx, weights, valid_mask, k, 64, x_rows
    )
    assert int(jnp.min(gidx_seg)) >= 0 and int(jnp.max(gidx_seg)) < x_rows // 2
    assert int(jnp.min(gidx_grp)) >= 0 and int(jnp.max(gidx_grp)) < x_rows // 2
    assert np.all(np.isfinite(np.asarray(w_grp)))

    dummy_x, dummy_idx = (
        jnp.zeros((128, 256), jnp.bfloat16),
        jnp.zeros((512,), jnp.int32),
    )
    if jax.default_backend() != "tpu":
        assert not ragged_gather_reduce_v4.supported(dummy_x, dummy_idx, 8, 16)
    monkeypatch.setattr(
        ragged_gather_reduce_v4.pltpu,
        "get_tpu_info",
        lambda: SimpleNamespace(
            num_lanes=128,
            sparse_core=SimpleNamespace(
                num_cores=2, num_subcores=16, num_lanes=16, vmem_capacity_bytes=4096
            ),
        ),
    )
    assert not ragged_gather_reduce_v4.supported(dummy_x, dummy_idx, 8, 64)


@pytest.mark.tpu_test
def test_gather_gmm_dense_expert_and_sc_combine_v4_on_tpu() -> None:
    if jax.default_backend() != "tpu":
        pytest.skip("Requires TPU backend")
    k1, k2 = jax.random.split(jax.random.key(404))
    x_sc = jax.random.normal(k1, (512, 256), dtype=jnp.bfloat16)
    for num_tokens, grouped in ((256, False), (2048, True)):
        idx = jax.random.randint(k2, (num_tokens * 4,), -8, 512, dtype=jnp.int32)
        valid = idx >= 0
        w = jnp.where(valid, jnp.float32(0.25), jnp.float32(0.0))
        tile = 64 if grouped else 16
        if not ragged_gather_reduce_v4.supported(x_sc, idx, 4, tile, grouped=grouped):
            continue
        out = ragged_gather_reduce_v4.ragged_gather_reduce_v4(
            x_sc, idx, w, valid, reduce_group_size=4, tile_tokens=tile, grouped=grouped
        )
        ref = (
            (x_sc[jnp.where(valid, idx, 0)].astype(jnp.float32) * w[:, None])
            .reshape(num_tokens, 4, 256)
            .sum(axis=1)
            .astype(jnp.bfloat16)
        )
        np.testing.assert_allclose(
            np.asarray(out, np.float32),
            np.asarray(ref, np.float32),
            rtol=1e-2,
            atol=2e-2,
        )

    tokens, hidden, inter, num_experts, topk = 128, 512, 512, 4, 2
    keys = jax.random.split(jax.random.key(505), 5)
    x = jax.random.normal(keys[0], (tokens, hidden), dtype=jnp.bfloat16) * 0.2
    w1, w1s, w2, w2s = _quantize_experts(
        jax.random.normal(keys[1], (num_experts, 2 * inter, hidden)) * 0.04,
        jax.random.normal(keys[2], (num_experts, hidden, inter)) * 0.04,
    )
    topk_ids = jax.random.randint(keys[3], (tokens, topk), 0, num_experts, jnp.int32)
    topk_w = jax.nn.softmax(jax.random.normal(keys[4], (tokens, topk)), -1).astype(
        jnp.bfloat16
    )
    order = jnp.argsort(topk_ids.reshape(-1), stable=True)
    positions = jnp.argsort(order).reshape(tokens, topk)
    tok_sorted = (order // topk).astype(jnp.int32)
    group_sizes = jnp.bincount(topk_ids.reshape(-1), length=num_experts)
    packed_x = gather_gmm.quantize_pack_rows(x, tile_tokens=16)
    y_packed = gather_gmm.gather_fused_moe_v2(
        packed_x,
        tok_sorted,
        group_sizes,
        w1,
        w1s[:, None, :],
        w2,
        w2s[:, None, :],
        act="silu",
    )
    out_gmm = gather_gmm.gather_combine_in_chunks(
        y_packed, positions, topk_w, hidden_size=hidden, max_routes=256, tile_tokens=16
    )
    ref_gmm = _reference_per_token_fp8_moe(
        x, w1, w1s[:, None, :], w2, w2s[:, None, :], topk_ids, topk_w, "silu"
    )
    np.testing.assert_allclose(
        np.asarray(out_gmm, np.float32),
        np.asarray(ref_gmm, np.float32),
        rtol=5e-2,
        atol=2e-1,
    )
    de_kw = dict(
        activation="silu", rhs_quant_dtype=jnp.float8_e4m3fn, skip_padded_tokens=False
    )
    if moe_dense_expert.can_use_dense_expert(
        x[:16], w1, w2, w1s, w2s, None, None, **de_kw
    ):
        out_de = moe_dense_expert.dense_expert_moe(
            x[:16], w1, w2, w1s, w2s, topk_w[:16], topk_ids[:16], activation="silu"
        )
        np.testing.assert_allclose(
            np.asarray(out_de, np.float32),
            np.asarray(ref_gmm[:16], np.float32),
            rtol=5e-2,
            atol=2e-1,
        )
