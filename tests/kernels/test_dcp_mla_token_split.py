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
"""TPU_DCP_MLA_TOKEN_SPLIT: the token-split sparse MLA equals head split.

(2 lanes, DCP4) on 8 devices, GLM-5.2 head shapes (8 heads a rank, 192 + 64
q head, 512 latent, 256 V). One step through the JAX half of the layer, both
ways, on the same inputs and caches:

* head split (production today, indexer lane split on): lane l projects
  every token with its 32 q_b heads, absorbs with W_UK_T_dcp_qrep, attends,
  and the W_UV DCP merge leaves rank (l, d) its 8 o_proj heads.
* token split: lane l projects its half of the tokens with the 64-head q_b
  (columns from `dcp_token_split.q_column_index` over the TP-gathered q_b),
  absorbs with W_UK_T in `head_order`, takes the indexer's lane-local top-k,
  attends with the lane-sliced cu_q_lens, merges with W_UV in `head_order`
  and runs the lane all-to-all.

Every rank's o_proj input `[T, 8, 256]` and the written KV caches must be bit
identical, on a full bucket of one request, a request cut by the lane
boundary, a skewed layout (lane 0 short context, lane 1 long), a mixed
decode + prefill bucket and a partly filled one. Positive controls: the
standard head order instead of `head_order`, and swapped lane ids.
"""

import os
import sys
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

sys.path.insert(0, os.path.dirname(__file__))

import test_mla_decode_prologue as prologue  # noqa: E402

pytestmark = pytest.mark.tpu_test

DCP, LANES, HPR = 4, 2, 8
NOPE, ROPE, LKV, V = 192, 64, 512, 256
HEAD = NOPE + ROPE
GROUP = DCP * HPR  # q heads a lane holds under query replication
QL = 256  # q_lora_rank (any width: only the column order is under test)
T = 1024
PAGE_SIZE, INTERLEAVE, PAGES_PER_RANK, PAGES_PER_SEQ = 256, 4, 48, 16
TOPK, H_IDX, D_IDX, WIDTH = 2048, 32, 128, 256

ENV = dict(
    prologue.PROD_ENV,
    TPU_DCP_MLA_TOPK_CAP="1024",
    TPU_DCP_MLA_MIXED_LIVE_WIDTHS="512,768",
)

# (q_lens, seq_lens, request_distribution)
LAYOUTS = {
    "one": ([T], [16000], [0, 0, 1]),
    "cut": ([300, T - 300], [9000, 16384], [0, 0, 2]),
    "skew": ([T // 2, T // 2], [T // 2, 16384], [0, 0, 2]),
    "mixed": ([4, 4, T - 8], [7000, 12000, 15000], [2, 2, 3]),
    "partial": ([200, 100], [5000, 3000], [0, 0, 2]),
}


def _mesh():
    from vllm_torchtpu.distributed.dcp import DCP_AXIS_NAME, DCP_LANE_AXIS_NAME

    devices = np.array(jax.devices()[: LANES * DCP]).reshape(LANES, DCP)
    return jax.sharding.Mesh(devices, (DCP_LANE_AXIS_NAME, DCP_AXIS_NAME))


def _per_lane(mesh, arrays):
    """A P() array whose devices on lane l hold `arrays[l]` (the torch bridge
    hands each rank its own tensor under a replicated spec)."""
    bufs = [
        jax.device_put(jnp.asarray(arrays[lane]), mesh.devices[lane, d])
        for lane in range(LANES)
        for d in range(DCP)
    ]
    shape = np.shape(arrays[0])
    return jax.make_array_from_single_device_arrays(
        shape, NamedSharding(mesh, P()), bufs
    )


def _per_device_ids(mesh, values):
    """i32 [DCP] under P(dcp), device (l, d) holding `values[l][d]`."""
    from vllm_torchtpu.distributed.dcp import DCP_AXIS_NAME

    bufs = [
        jax.device_put(jnp.asarray([values[lane][d]], jnp.int32), mesh.devices[lane, d])
        for lane in range(LANES)
        for d in range(DCP)
    ]
    return jax.make_array_from_single_device_arrays(
        (DCP,), NamedSharding(mesh, P(DCP_AXIS_NAME)), bufs
    )


def _q_cols(order):
    """q columns of `order` in the TP-gathered q_b, as `q_column_index`."""
    cols = []
    for g in order:
        lane, within = divmod(g, GROUP)
        start = lane * DCP * GROUP * HEAD + within * HEAD
        cols.append(np.arange(start, start + HEAD))
    return np.concatenate(cols)


def _case(layout, seed=11):
    from vllm_torchtpu.kernels.mla.kv_cache_utils import (
        KVCacheLayout,
        KVCacheType,
        SparseMLAKVCacheSpec,
    )

    q_lens, seq_lens, dist = LAYOUTS[layout]
    n = len(q_lens)
    rng = np.random.default_rng(seed)
    mesh = _mesh()
    from vllm_torchtpu.distributed.dcp import DCP_AXIS_NAME

    shard = NamedSharding(mesh, P(DCP_AXIS_NAME))

    def bf16(*shape, scale=1.0):
        return jnp.asarray(rng.normal(size=shape).astype(np.float32) * scale).astype(
            jnp.bfloat16
        )

    def fp8(*shape, scale=1.0):
        return jnp.asarray(rng.normal(size=shape).astype(np.float32) * scale).astype(
            jnp.float8_e4m3fn
        )

    def finite_words(shape):
        b = rng.integers(0, 0x7F, size=(*shape, 4), dtype=np.uint8)
        b |= rng.integers(0, 2, size=b.shape, dtype=np.uint8) << 7
        return b.view(np.uint32)[..., 0]

    packing = prologue.sparse_mla.get_dtype_packing(jnp.float8_e4m3fn)
    specs = tuple(
        SparseMLAKVCacheSpec.create(
            t, KVCacheLayout.SPARSECORE, PAGES_PER_RANK, PAGE_SIZE, d, packing
        )
        for t, d in ((KVCacheType.NOPE, LKV), (KVCacheType.ROPE, ROPE))
    )
    caches = tuple(
        jax.device_put(
            jnp.asarray(finite_words((DCP * s.shape[0], *s.shape[1:]))), shard
        )
        for s in specs
    )
    # Indexer K records: fp8 values, a 1.0 e8m0 scale byte, zero padding.
    vals = rng.normal(size=(DCP * PAGES_PER_RANK * PAGE_SIZE, D_IDX)).astype(np.float32)
    rec = np.zeros((vals.shape[0], WIDTH), np.uint8)
    rec[:, :D_IDX] = np.asarray(
        jax.lax.bitcast_convert_type(
            jnp.asarray(vals).astype(jnp.float8_e4m3fn), jnp.uint8
        )
    )
    rec[:, D_IDX] = 127
    idx_cache = jax.device_put(
        jnp.asarray(rec.reshape(DCP * PAGES_PER_RANK, PAGE_SIZE // 4, 4, WIDTH)), shard
    )
    block_tables = rng.permutation(PAGES_PER_RANK)[: n * PAGES_PER_SEQ].astype(np.int32)
    qsl = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    weights = dict(
        q_b=[bf16(QL, GROUP * HEAD, scale=0.06) for _ in range(LANES)],
        w_uk_t=bf16(LANES * GROUP, NOPE, LKV, scale=0.07),  # rank-major heads
        w_uv=bf16(LANES * GROUP, LKV, V, scale=0.05),
    )
    return dict(
        mesh=mesh,
        specs=specs,
        caches=caches,
        idx_cache=idx_cache,
        x=bf16(T, QL),
        q_idx=fp8(T, H_IDX, D_IDX, scale=0.5),
        w_idx=bf16(T, H_IDX),
        kv_c=fp8(T, LKV, scale=1 / prologue.K_SCALE),
        k_pe=fp8(T, ROPE, scale=1 / prologue.K_SCALE),
        seq_lens=jnp.asarray(seq_lens, jnp.int32),
        block_tables=jnp.asarray(block_tables),
        qsl=jnp.asarray(qsl),
        dist=jnp.asarray(dist, jnp.int32),
        ranks=_per_device_ids(mesh, [list(range(DCP))] * LANES),
        **weights,
    )


@jax.jit
def _q_proj(x, w):
    q = jnp.dot(x, w, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
    return q.reshape(x.shape[0], -1, HEAD)


def _absorb(q, w_uk_t):
    from vllm_torchtpu.kernels.mla.absorb_q import absorb_q

    return jax.jit(absorb_q)(q[..., :NOPE], w_uk_t), q[..., NOPE:]


def _indexer(c, lane_ids, lane_local):
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import streamindex_topk_dcp

    d = c["dist"]
    return streamindex_topk_dcp(
        c["q_idx"],
        c["w_idx"],
        c["idx_cache"],
        c["seq_lens"],
        c["block_tables"],
        c["qsl"],
        jnp.stack([d[0], d[0], d[2]]),
        mesh=c["mesh"],
        k=TOPK,
        compression_ratio=1,
        dcp_size=DCP,
        interleave_size=INTERLEAVE,
        num_kv_pages_per_block=(2, 2, 2),
        num_queries_per_block=(1, 128, 128),
        hist_merge=True,
        dcp_rank_ids=c["ranks"],
        dcp_lane_ids=_per_device_ids(c["mesh"], [[lane] * DCP for lane in lane_ids]),
        lane_split=True,
        lane_local=lane_local,
    )


def _attention(c, ql, q_pe, topk, lane_ids=None):
    from vllm_torchtpu.layers.core import attention_interface

    extra = {}
    if lane_ids is not None:
        extra = dict(
            lane_split=True,
            dcp_lane_ids=_per_device_ids(
                c["mesh"], [[lane] * DCP for lane in lane_ids]
            ),
        )
    return attention_interface.sparse_mla_attention_dcp(
        ql,
        q_pe,
        c["kv_c"],
        c["k_pe"],
        *c["caches"],
        topk,
        c["seq_lens"],
        c["block_tables"],
        c["qsl"],
        c["dist"],
        c["mesh"],
        *c["specs"],
        sm_scale=prologue.SM_SCALE,
        k_scale=prologue.K_SCALE,
        dcp_size=DCP,
        interleave_size=INTERLEAVE,
        dcp_rank_ids=c["ranks"],
        **extra,
    )


def _merge(mesh, partial, lse, w_uv, token_split):
    """The backbone's V-space W_UV merge (`_weighted_reduce_scatter_vspace`:
    lse all-gather, weights, head-batched W_UV, f32 reduce-scatter of the head
    axis over DCP), then under token split the lane all-to-all. Returns
    [ranks, T, HPR, V], rank l * DCP + d."""
    from vllm_torchtpu.distributed.dcp import DCP_AXIS_NAME, DCP_LANE_AXIS_NAME
    from vllm_torchtpu.layers.adapter import dcp_token_split
    from vllm_torchtpu.layers.adapter.dcp_vspace import vspace_weighted_heads_major_jax

    def body(o, lse_, w):
        lses = jax.lax.all_gather(lse_, DCP_AXIS_NAME, axis=0)
        weight = jnp.exp(lse_ - jax.nn.logsumexp(lses, axis=0))
        x = vspace_weighted_heads_major_jax(o, weight, w, None, out_dtype=jnp.float32)
        out = jax.lax.psum_scatter(x, DCP_AXIS_NAME, scatter_dimension=0, tiled=True)
        out = jnp.transpose(out, (1, 0, 2))
        if token_split:
            out = dcp_token_split.lane_all_to_all_local(out, DCP_LANE_AXIS_NAME)
        return out[None]

    return jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(P(DCP_AXIS_NAME), P(DCP_AXIS_NAME), P()),
            out_specs=P((DCP_LANE_AXIS_NAME, DCP_AXIS_NAME)),
            check_vma=False,
        )
    )(partial, lse, w_uv)


def _head_split(c):
    mesh = c["mesh"]
    qs = [_q_proj(c["x"], c["q_b"][lane]) for lane in range(LANES)]
    w_uk = [c["w_uk_t"][lane * GROUP : (lane + 1) * GROUP] for lane in range(LANES)]
    w_uv = [c["w_uv"][lane * GROUP : (lane + 1) * GROUP] for lane in range(LANES)]
    absorbed = [_absorb(q, w) for q, w in zip(qs, w_uk)]
    topk = _indexer(c, (0, 1), lane_local=False)
    nope, rope, partial, lse = _attention(
        c,
        _per_lane(mesh, [a[0] for a in absorbed]),
        _per_lane(mesh, [a[1] for a in absorbed]),
        topk,
    )
    merged = _merge(mesh, partial, lse, _per_lane(mesh, w_uv), token_split=False)
    return dict(topk=topk, nope=nope, rope=rope, out=merged)


def _token_split(c, order=None, lane_ids=(0, 1)):
    from vllm_torchtpu.layers.adapter import dcp_token_split

    mesh = c["mesh"]
    if order is None:
        order = dcp_token_split.head_order(DCP, LANES, HPR)
        cols = dcp_token_split.q_column_index(DCP, LANES, HPR, HEAD).numpy()
        np.testing.assert_array_equal(cols, _q_cols(order))
    else:
        cols = _q_cols(order)
    order = np.asarray(order)
    # q_b all-gathered over the TP group: rank r holds lane r // DCP's heads.
    gathered = jnp.concatenate([c["q_b"][r // DCP] for r in range(LANES * DCP)], 1)
    q_b_ts = gathered[:, cols]
    half = T // LANES
    absorbed = []
    for lane in range(LANES):
        q = _q_proj(c["x"][lane * half : (lane + 1) * half], q_b_ts)
        absorbed.append(_absorb(q, c["w_uk_t"][order]))
    topk = _indexer(c, lane_ids, lane_local=True)
    nope, rope, partial, lse = _attention(
        c,
        _per_lane(mesh, [a[0] for a in absorbed]),
        _per_lane(mesh, [a[1] for a in absorbed]),
        topk,
        lane_ids=lane_ids,
    )
    merged = _merge(mesh, partial, lse, c["w_uv"][order], token_split=True)
    return dict(topk=topk, nope=nope, rope=rope, out=merged)


def _shards(arr, mesh):
    """{(lane, dcp): host copy of that device's shard}."""
    where = {dev: (lane, d) for (lane, d), dev in np.ndenumerate(mesh.devices)}
    return {where[s.device]: np.asarray(s.data) for s in arr.addressable_shards}


def _require_devices():
    prologue._require_tpu()
    if len(jax.devices()) < LANES * DCP:
        pytest.skip(f"needs {LANES * DCP} devices")


@pytest.fixture
def env():
    with mock.patch.dict(os.environ, ENV):
        jax.clear_caches()
        yield


def _diff(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    finite = np.isfinite(a) & np.isfinite(b)
    err = np.abs(a - b)[finite]
    return int((~same).sum()), float(err.max()) if err.size else 0.0


@pytest.mark.multichip
@pytest.mark.usefixtures("env")
@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_token_split_matches_head_split(layout):
    _require_devices()
    c = _case(layout)
    want = _head_split(c)
    got = _token_split(c)
    mesh = c["mesh"]
    real = int(np.asarray(c["qsl"])[-1])

    # The indexer's lane-local rows are this lane's half of the gathered rows.
    full, local = _shards(want["topk"], mesh), _shards(got["topk"], mesh)
    half = T // LANES
    for (lane, d), rows in local.items():
        np.testing.assert_array_equal(
            rows, full[(lane, d)][lane * half : (lane + 1) * half], f"topk {lane, d}"
        )
    for k in ("nope", "rope"):
        np.testing.assert_array_equal(
            np.asarray(got[k]), np.asarray(want[k]), err_msg=f"{k} cache"
        )
    a, b = np.asarray(got["out"]), np.asarray(want["out"])
    n, err = _diff(a, b)
    print(
        f"{layout}: o_proj input {a.shape}, {n} of {a.size} elements differ"
        f" (max |diff| {err:.3g}); real rows {real}"
    )
    assert np.isfinite(b[:, :real]).all()
    assert np.abs(b[:, :real]).max() > 0
    np.testing.assert_array_equal(a, b)


@pytest.mark.multichip
@pytest.mark.usefixtures("env")
@pytest.mark.parametrize("control", ["standard_head_order", "swapped_lanes"])
def test_token_split_positive_controls_fail(control):
    _require_devices()
    c = _case("cut")
    want = np.asarray(_head_split(c)["out"])
    if control == "standard_head_order":
        got = _token_split(c, order=list(range(LANES * GROUP)))
    else:
        got = _token_split(c, lane_ids=(1, 0))
    n, _ = _diff(np.asarray(got["out"]), want)
    print(f"{control}: {n} of {want.size} elements differ")
    assert n > 0
