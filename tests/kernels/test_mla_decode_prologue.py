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
"""TPU_MLA_DECODE_PROLOGUE: in-kernel q rope and kernel-owned lse.

The claim is bit-identity with the incumbent: q_pe roped by XLA (the bf16
NeoX expression the wrapper runs) and handed to the split-q kernel, plus an
XLA-initialised aliased lse. The candidate hands the raw q head view and the
per-token rope table and lets the kernel do both. Checked

* per kernel call (one device): every output and lse row, padding rows
  included, decode buckets 16/32 (incl. a zero-real-token step) and a
  two-chunk mixed bucket (q rope only; lse stays aliased there);
* through `sparse_mla_attention_dcp` on DCP ranks (prefill then decode with
  the production env): each rank's caches, partial outputs and lse;

with positive controls the comparison must catch: skipped rope, swapped
rope halves, a wrong lse sentinel.
"""

import os
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest, parameterized

from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla
from vllm_torchtpu.kernels.mla.sparse import native_sc_cache_layout

pytestmark = pytest.mark.tpu_test

NOPE_DIM = 512
ROPE_DIM = 64
RAW_HEAD = 256  # q_b head: nope 192 | rope 64
HEADS = 32  # GLM-5.2 per-rank attention heads under qrep (TP16 x DCP16)
K_SCALE = 0.75
SM_SCALE = 0.125
MAX_POS = 8192


def _require_tpu():
    if jax.devices()[0].platform != "tpu":
        pytest.skip("SparseCore-layout sparse MLA needs a TPU")


def _bits16(x):
    return np.asarray(jax.lax.bitcast_convert_type(x, jnp.uint16))


def _bits32(x):
    return np.asarray(x).view(np.uint32)


def _cos_sin_cache():
    """A real NeoX rope table, bf16, [MAX_POS, 64] = (cos | sin)."""
    inv = 1.0 / (10000 ** (np.arange(0, ROPE_DIM, 2) / ROPE_DIM))
    ang = np.arange(MAX_POS)[:, None] * inv[None, :]
    cs = np.concatenate([np.cos(ang), np.sin(ang)], -1).astype(np.float32)
    return jnp.asarray(cs, jnp.bfloat16)


def rope_reference(q_raw, positions, cache):
    """The wrapper's q rope, as XLA runs it: bf16 o1 = x1*cos - x2*sin,
    o2 = x2*cos + x1*sin on q_pe = q_raw[..., 192:], concatenated."""
    cs = cache[positions]
    cos, sin = cs[:, None, :32], cs[:, None, 32:]
    x = q_raw[..., RAW_HEAD - ROPE_DIM :]
    x1, x2 = x[..., :32], x[..., 32:]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)


def rope_table(positions, cache):
    """`mla_q_rope_table` in jnp: f32 [T, 2, 128]."""
    cs = cache[positions]
    cos, sin = cs[:, :32], cs[:, 32:]
    c = jnp.concatenate([cos, cos, cos, cos], -1).astype(jnp.float32)
    s = jnp.concatenate([-sin, sin, -sin, sin], -1).astype(jnp.float32)
    return jnp.stack([c, s], 1)


def _problem(rng, query_tokens, seq_q, topk=2048):
    """A DCP-rank-shaped call: SparseCore caches, raw q, per-row live counts
    incl. empty and overflowing rows (the `_split_q_lse_case` shape)."""
    page_size = 256
    cache_tokens = 8192
    pages = cache_tokens // page_size

    def quantized_bytes(dim):
        values = rng.random((pages, page_size, dim), dtype=np.float32) * 2 - 1
        fp8 = jnp.asarray(values / K_SCALE).astype(jnp.float8_e4m3fn)
        return np.asarray(jax.lax.bitcast_convert_type(fp8, jnp.uint8))

    nope = quantized_bytes(NOPE_DIM).reshape(pages, page_size, 4, 128)
    rope = np.pad(quantized_bytes(ROPE_DIM), ((0, 0), (0, 0), (0, ROPE_DIM)))
    rope = rope.reshape(pages, page_size // 4, 4, 128)
    page_table = rng.permutation(pages).astype(np.int32)
    inv = np.argsort(page_table)
    physical = np.stack(
        [
            rng.choice(cache_tokens, size=topk, replace=False)
            for _ in range(query_tokens)
        ]
    )
    logical = (inv[physical // page_size] * page_size + physical % page_size).astype(
        np.int32
    )
    counts = np.resize(
        np.array([0, 1, 63, 64, 127, 129, 255, 257, 511, 512, 513, 700, 1025, 2048]),
        query_tokens,
    )
    rng.shuffle(counts)
    counts[0], counts[min(1, query_tokens - 1)] = 0, 1025
    live = np.arange(topk)[None, :] < counts[:, None]
    topk_idx = np.where(live, logical, -1).astype(np.int32)
    num_seqs = len(seq_q)
    tables = [page_table] + [
        rng.permutation(pages) for _ in range(max(num_seqs - 1, 0))
    ]
    num_decode = sum(1 for n in seq_q if n == 4)
    q_raw = jnp.asarray(
        rng.normal(size=(query_tokens, HEADS, RAW_HEAD)).astype(np.float32),
        jnp.bfloat16,
    )
    ql_nope = jnp.asarray(
        rng.normal(size=(query_tokens, HEADS, NOPE_DIM)).astype(np.float32),
        jnp.bfloat16,
    )
    return dict(
        nope=jnp.asarray(native_sc_cache_layout.pack_nope(nope)),
        rope=jnp.asarray(native_sc_cache_layout.pack_rope_banded(rope)),
        topk=jnp.asarray(topk_idx),
        page_table=jnp.asarray(np.concatenate(tables).astype(np.int32)),
        cu_q_lens=jnp.asarray(np.concatenate([[0], np.cumsum(seq_q)]), jnp.int32),
        distribution=jnp.array([num_decode, num_decode, num_seqs], jnp.int32),
        q_raw=q_raw,
        ql_nope=ql_nope,
        positions=jnp.asarray(rng.integers(0, MAX_POS, size=query_tokens), jnp.int32),
        counts=counts,
    )


def _kernel(p, *, candidate, fault=None, q_pe=None):
    cache = _cos_sin_cache()
    kw = {}
    if candidate:
        kw = dict(
            q_pe=p["q_raw"],
            q_rope_table=rope_table(p["positions"], cache),
            lse_in_kernel=True,
            prologue_fault=fault,
        )
    else:
        kw = dict(
            q_pe=(
                q_pe
                if q_pe is not None
                else rope_reference(p["q_raw"], p["positions"], cache)
            )
        )
    out, lse = sparse_mla.sparse_ragged_paged_attention(
        p["ql_nope"],
        p["nope"],
        p["rope"],
        p["topk"],
        p["page_table"],
        p["cu_q_lens"],
        p["distribution"],
        sm_scale=SM_SCALE,
        k_scale=K_SCALE,
        return_lse=True,
        cache_layout=sparse_mla.SPARSECORE_CACHE_LAYOUT,
        gather_and_attention_chunk_size=256,
        topk_cap=512,
        small_chunk_batch_size=4,
        small_chunk_tokens=32,
        lse_lane_dense=True,
        lse_mask_empty=True,
        fused_window_prep=True,
        mixed_live_gather=True,
        mixed_live_widths=(256, 384),
        mixed_body_levers=sparse_mla.MIXED_BODY_DEFAULT,
        **kw,
    )
    return jax.block_until_ready((out[..., :NOPE_DIM], lse))


CELLS = (
    # Decode buckets (MTP 3: 4 tokens per request), as served.
    (16, (4,)),
    (16, (4, 4)),
    (16, (4, 4, 4, 1)),
    (32, (4, 4, 4, 4, 4)),
    (32, (4,) * 8),
    # Mixed bucket: two chunks of 256; q rope in-kernel, lse stays aliased.
    (512, (4, 4, 492)),
)


class DecodePrologueKernelTest(parameterized.TestCase):
    @parameterized.parameters(*CELLS)
    def test_bit_identical(self, query_tokens, seq_q):
        _require_tpu()
        p = _problem(np.random.default_rng(1012 + query_tokens), query_tokens, seq_q)
        ref, ref_lse = _kernel(p, candidate=False)
        out, lse = _kernel(p, candidate=True)
        np.testing.assert_array_equal(_bits16(out), _bits16(ref))
        np.testing.assert_array_equal(_bits32(lse), _bits32(ref_lse))
        real = int(sum(seq_q))
        # Non-vacuous: real rows attend (not the aliased q), empty rows are
        # -inf, padding rows keep the sentinel.
        assert np.isfinite(np.asarray(out, np.float32)).all()
        empty = p["counts"][:real] == 0
        assert empty.any() and np.all(np.asarray(lse)[:real][empty] == -np.inf)
        # Rows past the last grid step (real rounded up to the kernel batch,
        # at most 16) keep the sentinel the kernel itself laid down.
        rup = -(-real // 16) * 16
        assert np.all(np.asarray(lse)[rup:] == -np.inf)

    def test_zero_real_tokens(self):
        """A bucket with no live row: the grid's one step only lays down the
        sentinel; both arms return -inf lse on every row."""
        _require_tpu()
        p = _problem(np.random.default_rng(7), 16, (4,))
        p["cu_q_lens"] = jnp.array([0, 0], jnp.int32)
        p["distribution"] = jnp.array([0, 0, 0], jnp.int32)
        ref, ref_lse = _kernel(p, candidate=False)
        out, lse = _kernel(p, candidate=True)
        assert np.all(np.asarray(ref_lse) == -np.inf)
        np.testing.assert_array_equal(_bits32(lse), _bits32(ref_lse))
        np.testing.assert_array_equal(_bits16(out), _bits16(ref))

    @parameterized.parameters(
        ("skip_q_rope", (16, (4, 4))),
        ("rope_swap", (16, (4, 4))),
        # One rounding of the f32 sum instead of XLA's per-product rounding.
        ("rope_f32", (16, (4, 4))),
        ("skip_q_rope", (512, (4, 4, 492))),
        ("skip_lse_fill", (16, (4,))),
    )
    def test_positive_control_fails(self, fault, cell):
        _require_tpu()
        query_tokens, seq_q = cell
        p = _problem(np.random.default_rng(1012 + query_tokens), query_tokens, seq_q)
        ref, ref_lse = _kernel(p, candidate=False)
        out, lse = _kernel(p, candidate=True, fault=fault)
        same = np.array_equal(_bits16(out), _bits16(ref)) and np.array_equal(
            _bits32(lse), _bits32(ref_lse)
        )
        assert not same, f"fault {fault} not caught"


# --- DCP: sparse_mla_attention_dcp, caches + partials on every rank --------

PAGE_SIZE = 256
INTERLEAVE = 4
PAGES_PER_RANK = 24
PAGES_PER_SEQ = 2  # virtual pages of PAGE_SIZE * dcp positions
PROD_ENV = {
    "TPU_DCP_MLA_SPLIT_Q_LSE": "1",
    "TPU_DCP_MLA_MASK_IN_KERNEL": "1",
    "TPU_DCP_MLA_FUSED_PREP": "1",
    "TPU_DCP_MLA_TOPK_CAP": "512",
    "TPU_DCP_MLA_TOPK_CAP_CHUNK": "256",
    "TPU_DCP_MLA_SMALL_CHUNK_BATCH": "4",
    "TPU_DCP_MLA_SMALL_CHUNK_TOKENS": "32",
    "TPU_DCP_MLA_MIXED_LIVE": "1",
    "TPU_DCP_MLA_MIXED_LIVE_WIDTHS": "256,384",
    "TPU_DCP_MLA_MIXED_BODY": "1",
    "TPU_DCP_MLA_MIXED_MASK_IN_KERNEL": "1",
    "TPU_MLA_TC_KV_WRITER": "1",
}


def _local_topk(global_rows, dcp, interleave):
    """Global top-k rows -> per-rank local cache indices (streamindex contract)."""
    t, k = global_rows.shape
    out = np.full((dcp, t, k), -1, np.int32)
    vps, cycle = PAGE_SIZE * dcp, dcp * interleave
    for i in range(t):
        row = global_rows[i][global_rows[i] >= 0]
        off = row % vps
        owner = (off % cycle) // interleave
        local = (
            (row // vps) * PAGE_SIZE + (off // cycle) * interleave + off % interleave
        )
        for r in range(dcp):
            sel = local[owner == r]
            out[r, i, : sel.size] = sel
    return out.reshape(dcp * t, k)


def _causal_rows(positions, topk, rng):
    rows = np.full((len(positions), topk), -1, np.int32)
    for i, pos in enumerate(positions):
        n = min(pos + 1, topk)
        rows[i, :n] = np.sort(rng.choice(pos + 1, size=n, replace=False))
    return rows


def _dcp_run(dcp, *, candidate, fault=None, big_mixed=False, **dcp_kwargs):
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk
    from vllm_torchtpu.kernels.mla.kv_cache_utils import (
        KVCacheLayout,
        KVCacheType,
        SparseMLAKVCacheSpec,
    )
    from vllm_torchtpu.layers.core import attention_interface

    axis = streamindex_topk.DCP_AXIS_NAME
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:dcp]), (axis,))
    packing = sparse_mla.get_dtype_packing(jnp.float8_e4m3fn)
    specs = tuple(
        SparseMLAKVCacheSpec.create(
            t, KVCacheLayout.SPARSECORE, PAGES_PER_RANK, PAGE_SIZE, d, packing
        )
        for t, d in ((KVCacheType.NOPE, NOPE_DIM), (KVCacheType.ROPE, ROPE_DIM))
    )
    rng = np.random.default_rng(31)
    cache = _cos_sin_cache()
    prior = [int(x) for x in rng.integers(300, 2400, size=4)]
    num_seqs = len(prior)
    block_tables = jnp.asarray(
        rng.permutation(PAGES_PER_RANK)[: num_seqs * PAGES_PER_SEQ], jnp.int32
    )
    topk = 2048
    shard = NamedSharding(mesh, P(axis))

    def fp8(n, d):
        v = rng.normal(size=(n, d)).astype(np.float32)
        return jnp.asarray(v / K_SCALE).astype(jnp.float8_e4m3fn)

    def call(caches, positions, seq_lens, qsl, dist, n, step_rng):
        t = len(positions)
        pos = np.zeros(n, np.int32)
        pos[:t] = positions
        rows = np.concatenate(
            [_causal_rows(positions, topk, step_rng), np.full((n - t, topk), -1)]
        ).astype(np.int32)
        q_raw = jnp.asarray(
            step_rng.normal(size=(n, HEADS, RAW_HEAD)).astype(np.float32), jnp.bfloat16
        )
        ql_nope = jnp.asarray(
            step_rng.normal(size=(n, HEADS, NOPE_DIM)).astype(np.float32),
            jnp.bfloat16,
        )
        kv_c, k_pe = fp8(n, NOPE_DIM), fp8(n, ROPE_DIM)
        posj = jnp.asarray(pos)
        if candidate:
            q_pe, extra = (
                q_raw,
                dict(q_rope_table=rope_table(posj, cache), prologue_fault=fault),
            )
        else:
            q_pe, extra = rope_reference(q_raw, posj, cache), {}
        return attention_interface.sparse_mla_attention_dcp(
            ql_nope,
            q_pe,
            kv_c,
            k_pe,
            caches[0],
            caches[1],
            jnp.asarray(_local_topk(rows, dcp, INTERLEAVE)),
            jnp.asarray(seq_lens, jnp.int32),
            block_tables,
            jnp.asarray(qsl, jnp.int32),
            jnp.asarray(dist, jnp.int32),
            mesh,
            *specs,
            sm_scale=SM_SCALE,
            k_scale=K_SCALE,
            dcp_size=dcp,
            interleave_size=INTERLEAVE,
            **extra,
            **dcp_kwargs,
        )

    def finite_words(shape):
        # Every byte a finite e4m3fn value (0x7f / 0xff are NaN), so attention
        # over never-written positions stays finite and comparable.
        b = rng.integers(0, 0x7F, size=(*shape, 4), dtype=np.uint8)
        b |= rng.integers(0, 2, size=b.shape, dtype=np.uint8) << 7
        return b.view(np.uint32)[..., 0]

    caches = tuple(
        jax.device_put(
            jnp.asarray(finite_words((dcp * s.shape[0], *s.shape[1:]))), shard
        )
        for s in specs
    )
    results = []
    # A mixed step (2 decode requests + a 248-token prefill chunk of seq 2),
    # then decode steps in the 16 and 32 buckets on top of it.
    chunk = 248
    p = prior
    steps = [
        (
            [p[0] + j for j in range(4)]
            + [p[1] + j for j in range(4)]
            + list(range(p[2], p[2] + chunk)),
            [p[0] + 4, p[1] + 4, p[2] + chunk, p[3]],
            [0, 4, 8, 8 + chunk, 8 + chunk],
            [2, 3, 3],
            256,
        )
    ]
    q = [p[0] + 4, p[1] + 4, p[2] + chunk, p[3]]
    steps.append(
        (
            [q[i] + j for i in range(2) for j in range(4)],
            [q[0] + 4, q[1] + 4, q[2], q[3]],
            [0, 4, 8, 8, 8],
            [2, 2, 2],
            16,
        )
    )
    q = [q[0] + 4, q[1] + 4, q[2], q[3]]
    steps.append(
        (
            [q[i] + j for i in range(4) for j in range(4)],
            [x + 4 for x in q],
            [0, 4, 8, 12, 16],
            [4, 4, 4],
            32,
        )
    )
    if big_mixed:
        # A 2048-token mixed bucket (MTP decode of seqs 0-1 + 1020-token
        # prefill chunks of seqs 2 and 3): the >= 2048-token programs.
        q = [x + 4 for x in q]
        c = 1020
        steps.append(
            (
                [q[i] + j for i in range(2) for j in range(4)]
                + list(range(q[2], q[2] + c))
                + list(range(q[3], q[3] + c)),
                [q[0] + 4, q[1] + 4, q[2] + c, q[3] + c],
                [0, 4, 8, 8 + c, 8 + 2 * c],
                [2, 4, 4],
                2048,
            )
        )
    for i, (positions, seq_lens, qsl, dist, n) in enumerate(steps):
        nope, rope, out, lse = call(
            caches, positions, seq_lens, qsl, dist, n, np.random.default_rng(100 + i)
        )
        caches = (nope, rope)
        results.append(
            dict(
                nope=np.asarray(nope),
                rope=np.asarray(rope),
                out=_bits16(out),
                lse=_bits32(lse),
                real=len(positions),
            )
        )
    return results


@pytest.fixture
def prod_env():
    with mock.patch.dict(os.environ, PROD_ENV):
        yield


def _dcp_size():
    return min(8, len(jax.devices()))


@pytest.mark.usefixtures("prod_env")
def test_dcp_caches_partials_and_lse_bit_identical():
    _require_tpu()
    dcp = _dcp_size()
    if dcp < 2:
        pytest.skip("DCP needs at least 2 devices")
    ref = _dcp_run(dcp, candidate=False)
    got = _dcp_run(dcp, candidate=True)
    for i, (a, b) in enumerate(zip(ref, got)):
        for k in ("nope", "rope", "out", "lse"):
            np.testing.assert_array_equal(b[k], a[k], err_msg=f"step {i} {k}")
        # Non-vacuous: some rank owns rows of every real token, outputs move.
        lse = a["lse"].view(np.float32).reshape(dcp, -1, HEADS)
        assert np.isfinite(lse).any(axis=0)[: a["real"]].all(), f"step {i}"


@pytest.mark.usefixtures("prod_env")
@pytest.mark.parametrize(
    "fault", ["skip_q_rope", "rope_swap", "rope_f32", "skip_lse_fill"]
)
def test_dcp_positive_control_fails(fault):
    _require_tpu()
    dcp = _dcp_size()
    if dcp < 2:
        pytest.skip("DCP needs at least 2 devices")
    ref = _dcp_run(dcp, candidate=False)
    got = _dcp_run(dcp, candidate=True, fault=fault)
    same = all(
        np.array_equal(a[k], b[k]) for a, b in zip(ref, got) for k in ("out", "lse")
    )
    assert not same, f"fault {fault} not caught"


if __name__ == "__main__":
    absltest.main()


# --- torch side: the wrapper's q rope on TPU == the rope the kernel rebuilds --


def test_torch_wrapper_rope_matches_kernel_rope_semantics():
    """The model ropes q_pe in torch (`rope_half_split`, lowered to XLA by
    torch_tpu). The kernel reproduces XLA's per-op bf16 rope
    (`rope_reference`, checked against the kernel above). Here: torch on TPU
    == `rope_reference` bit for bit, and `mla_q_rope_table` == `rope_table`,
    on the same q head / positions."""
    _require_tpu()
    try:
        import torch
        import torch_tpu  # noqa: F401
        import vllm.model_executor.custom_op as custom_op_mod
        from vllm.config.compilation import CompilationConfig, CompilationMode
        from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding

        from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op as op
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"torch_tpu unavailable: {e}")
    cfg = CompilationConfig(mode=CompilationMode.NONE, custom_ops=["none"])
    with mock.patch.object(custom_op_mod, "get_cached_compilation_config", lambda: cfg):
        rope = RotaryEmbedding(
            head_size=ROPE_DIM,
            rotary_dim=ROPE_DIM,
            max_position_embeddings=MAX_POS,
            base=10000.0,
            is_neox_style=False,
            dtype=torch.bfloat16,
        )
    dev = torch.device("tpu")
    rope.cos_sin_cache = rope.cos_sin_cache.to(dev)
    rng = np.random.default_rng(3)
    t = 32
    q_np = rng.normal(size=(t, HEADS, RAW_HEAD)).astype(np.float32)
    k_np = rng.normal(size=(t, 1, ROPE_DIM)).astype(np.float32)
    pos_np = rng.integers(0, MAX_POS, size=t).astype(np.int64)
    q = torch.from_numpy(q_np).bfloat16().to(dev)
    k_pe = torch.from_numpy(k_np).bfloat16().to(dev)
    positions = torch.from_numpy(pos_np).to(dev)
    q_pe = q.split([RAW_HEAD - ROPE_DIM, ROPE_DIM], dim=-1)[1]
    q_rot, k_rot = op.rope_half_split(rope, positions, q_pe, k_pe)
    _, k_rot_fold = op.rope_half_split(rope, positions, q_pe, k_pe)
    tab = op.mla_q_rope_table(rope, positions, q)
    got = q_rot.to(torch.bfloat16).cpu().view(torch.int16).numpy().view(np.uint16)
    k_a = k_rot.cpu().view(torch.int16).numpy()
    k_b = k_rot_fold.cpu().view(torch.int16).numpy()
    tab_np = tab.cpu().numpy()
    cache = jnp.asarray(rope.cos_sin_cache.cpu().float().numpy(), jnp.float32).astype(
        jnp.bfloat16
    )
    q_j = jnp.asarray(q_np, jnp.bfloat16)
    pos_j = jnp.asarray(pos_np.astype(np.int32))
    want = _bits16(jax.jit(rope_reference)(q_j, pos_j, cache))
    np.testing.assert_array_equal(got.reshape(want.shape), want)
    np.testing.assert_array_equal(k_a, k_b)
    np.testing.assert_array_equal(
        tab_np.view(np.uint32), _bits32(rope_table(pos_j, cache))
    )


# --- composition: prologue partials/lse through the packed a2a merge ---------


def _merge_every_rank(outputs, lses, pack):
    """`merge_lse_partials_all_to_all` on every DCP rank (CPU, lockstep threads
    standing in for the exchange); returns the [T, H, V] merge over all heads."""
    import concurrent.futures
    import threading

    import torch

    from vllm_torchtpu.layers.adapter import cp_mla_attention as cp_mla

    world = outputs.shape[0]
    slots = [None] * world
    barrier = threading.Barrier(world)

    class _Group:
        def __init__(self, rank):
            self.world_size, self.rank_in_group, self.device_group = world, rank, None

    def exchange(x, group):
        slots[group.rank_in_group] = x
        barrier.wait()
        recv = torch.stack([slots[s][group.rank_in_group] for s in range(world)])
        barrier.wait()
        return recv

    def run(rank):
        return cp_mla.merge_lse_partials_all_to_all(
            outputs[rank], lses[rank], _Group(rank)
        )

    with (
        mock.patch.object(cp_mla, "_all_to_all", exchange),
        mock.patch.object(cp_mla.envs, "TPU_DCP_LSE_MERGE_A2A_PACK", pack, create=True),
        concurrent.futures.ThreadPoolExecutor(world) as pool,
    ):
        return torch.cat(list(pool.map(run, range(world))), dim=1)


def _rank_partials(step, dcp):
    import torch

    n = step["out"].shape[0] // dcp
    out = torch.from_numpy(step["out"].view(np.int16).copy()).view(torch.bfloat16)
    lse = torch.from_numpy(step["lse"].view(np.float32).copy())
    return out.reshape(dcp, n, HEADS, -1), lse.reshape(dcp, n, HEADS)


@pytest.mark.usefixtures("prod_env")
def test_dcp_prologue_partials_through_packed_merge_bit_identical():
    """TPU_MLA_DECODE_PROLOGUE + TPU_DCP_LSE_MERGE_A2A_PACK compose: the
    kernel-owned lse (its in-kernel sentinel included) and partials, merged
    through the one-exchange packed a2a, equal bit for bit the incumbent's
    (XLA rope, aliased lse) merged through the two-exchange a2a -- every rank's
    heads, every row, mixed 256 and decode 16 / 32 buckets."""
    _require_tpu()
    dcp = _dcp_size()
    if dcp < 2:
        pytest.skip("DCP needs at least 2 devices")
    import torch

    ref = _dcp_run(dcp, candidate=False)
    got = _dcp_run(dcp, candidate=True)
    for i, (a, b) in enumerate(zip(ref, got)):
        want = _merge_every_rank(*_rank_partials(a, dcp), pack=False)
        o, lse = _rank_partials(b, dcp)
        one = _merge_every_rank(o, lse, pack=True)
        assert torch.equal(one.view(torch.int16), want.view(torch.int16)), f"step {i}"
        # Non-vacuous: on the padded decode buckets the kernel's sentinel rows
        # ride the pack, and real rows merge > 1 rank.
        sentinel = ~torch.isfinite(lse) | (lse == torch.finfo(torch.float32).min)
        if b["real"] < lse.shape[1]:
            assert sentinel[:, b["real"] :].all(), f"step {i}"
        assert ((~sentinel).sum(0)[: b["real"]] > 1).any(), f"step {i}"
        assert torch.isfinite(one.float()).all(), f"step {i}"


@pytest.mark.usefixtures("prod_env")
def test_dcp_prologue_packed_merge_positive_control_fails():
    """The same comparison must fail when the kernel lays a wrong lse sentinel
    (0.0) down: the packed merge then weights rows no rank owns."""
    _require_tpu()
    dcp = _dcp_size()
    if dcp < 2:
        pytest.skip("DCP needs at least 2 devices")
    import torch

    ref = _dcp_run(dcp, candidate=False)
    bad = _dcp_run(dcp, candidate=True, fault="skip_lse_fill")
    same = all(
        torch.equal(
            _merge_every_rank(*_rank_partials(b, dcp), pack=True).view(torch.int16),
            _merge_every_rank(*_rank_partials(a, dcp), pack=False).view(torch.int16),
        )
        for a, b in zip(ref, bad)
    )
    assert not same, "wrong lse sentinel not caught through the packed merge"
