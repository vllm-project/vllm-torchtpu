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
"""GLM5_MKD_STACK_C on TPU: stacked == per-layer, bitwise, for the DCP ops.

A small multi-layer GLM-5.2-shaped setup (page 256, interleave 4, SparseCore
MLA caches, SAL indexer cache, top-k 2048, 64 query heads) runs three steps --
a prefill, a mixed MTP-verify-sized step and a decode step -- through the
real DCP indexer step (`streamidx_topk_dcp_step`: SAL Pallas append +
`streamindex_topk_dcp`) feeding the real DCP sparse attention
(`sparse_mla_attention_dcp`: SparseCore writer + gather kernel), once on
per-layer caches and once on [n_layers, num_blocks, ...] stacks shared by every
layer (`kv_stack=(row, num_blocks)`). Every output and every cache must match
bit for bit.

Two knob sets: the code defaults (the capped plain-window prep at DCP 8), and
the best served config's sparse-MLA knobs (`BEST_ENV`, j-e0ae7f9f): the
live-group mixed prep with the mixed fused prep (TPU_DCP_MLA_MIXED_LIVE /
_MIXED_FUSED_PREP), the fused decode prep (TPU_DCP_MLA_FUSED_PREP) with the
KV write in the gather (TPU_MLA_KV_WRITE_IN_GATHER), merge pieces, the rank
operand. Under the stack every one of those preps runs on the per-layer
geometry and the SparseCore gather adds the layer's row offset in-kernel (to
its gathered rows and its write rows); the test asserts those paths ran.
Positive controls: every stacked layer on row 0, and a wrong in-gather row
offset (the gather's `row_offset` shifted one layer) -- both must be caught.
"""

import os
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    DCP_AXIS_NAME,
    KVLayout,
    get_kv_cache_shape,
)
from vllm_torchtpu.kernels.mla.kv_cache_utils import (
    KVCacheLayout,
    KVCacheType,
    SparseMLAKVCacheSpec,
)

pytestmark = [pytest.mark.tpu_test, pytest.mark.multichip]

P = jax.sharding.PartitionSpec

N_LAYERS = 3
NUM_BLOCKS = 10  # per rank
PAGE = 256
INTERLEAVE = 4
PAGES_PER_SEQ = 2
LKV, ROPE, HEADS = 512, 64, 64
H_IDX, D_IDX, TOPK = 32, 128, 2048
SM_SCALE = 1.0 / 16

# (q_lens, seq_lens, request_distribution, padded token count)
STEPS = (
    ([3000, 700, 300, 0], [3000, 700, 300, 0], [0, 0, 3], 4096),  # prefill
    ([4, 64, 4, 0], [3004, 764, 304, 0], [0, 0, 3], 128),  # mixed / verify
    ([1, 1, 1, 0], [3005, 765, 305, 0], [3, 3, 3], 16),  # decode
)


# The best config's (j-e0ae7f9f) knobs read by the DCP sparse-MLA op and its
# kernel (`scratch/cmp_best/recipe_best.yaml`); TPU_DCP_MERGE_CHUNKED* size the
# merge pieces. Adapter-level knobs (prologue, send layout, small a2a) need
# operands this op-level test does not build.
BEST_ENV = {
    "TPU_DCP_MLA_FUSED_PREP": "1",
    "TPU_DCP_MLA_MASK_IN_KERNEL": "1",
    "TPU_DCP_MLA_MERGE_PIECES": "2",
    "TPU_DCP_MLA_MERGE_PIECES_MIN_TOKENS": "2048",
    "TPU_DCP_MLA_MIXED_BODY": "1",
    "TPU_DCP_MLA_MIXED_FUSED_PREP": "1",
    "TPU_DCP_MLA_MIXED_KERNEL_V2": "1",
    "TPU_DCP_MLA_MIXED_LIVE": "1",
    "TPU_DCP_MLA_MIXED_LIVE_WIDTHS": "256,384",
    "TPU_DCP_MLA_MIXED_MASK_IN_KERNEL": "1",
    "TPU_DCP_MLA_SMALL_CHUNK_BATCH": "4",
    "TPU_DCP_MLA_SMALL_CHUNK_TOKENS": "32",
    "TPU_DCP_MLA_TOPK_CAP": "512",
    "TPU_DCP_MLA_TOPK_CAP_CHUNK": "512",
    "TPU_DCP_RANK_OPERAND": "1",
    "TPU_MLA_DECODE_BODY_LEVERS": "1",
    "TPU_MLA_DECODE_ROW_WIDTHS": "128,256,384",
    "TPU_MLA_KV_WRITE_IN_GATHER": "1",
    "TPU_MLA_SPARSE_SPLIT_Q": "1",
    "TPU_MLA_TC_KV_WRITER": "1",
    "TPU_SPARSE_MLA_CHUNK_TOKENS": "256",
    "TPU_SPARSE_MLA_LIVE_GATHER": "0",
    "TPU_SPARSE_MLA_PACKED_ROPE": "0",
    "TPU_DCP_MERGE_CHUNKED": "8",
    "TPU_DCP_MERGE_CHUNKED_MIN_TOKENS": "1024",
    "TPU_DCP_MERGE_CHUNKED_SMALL": "2",
    "TPU_DCP_MERGE_CHUNKED_LSE_AG": "1",
    "TPU_DCP_MERGE_CHUNKED_LSE_AG_MAX_TOKENS": "2048",
}

# Best-config steps: a 4096 prefill, a served-shaped 4096 mixed step (three
# MTP-3 verify windows + a fresh 2000-token chunk), a 16-row verify step
# (4 x 4) and a decode step -- the last two take the fused decode prep with the
# KV write in the gather.
STEPS_BEST = (
    ([3000, 700, 300, 0], [3000, 700, 300, 0], [0, 0, 3], 4096),  # prefill
    ([4, 4, 4, 2000], [3004, 704, 304, 2000], [0, 0, 4], 4096),  # mixed
    ([4, 4, 4, 4], [3008, 708, 308, 2004], [0, 0, 4], 16),  # verify
    ([1, 1, 1, 1], [3009, 709, 309, 2005], [4, 4, 4], 16),  # decode
)


def _dcp_size():
    n = len(jax.devices())
    d = 8 if n >= 8 else (4 if n >= 4 else 2)
    if n < 2:
        pytest.skip(f"needs >= 2 devices, have {n}")
    return d


def _specs():
    nope = SparseMLAKVCacheSpec.create(
        KVCacheType.NOPE, KVCacheLayout.SPARSECORE, NUM_BLOCKS, PAGE, LKV, 4
    )
    rope = SparseMLAKVCacheSpec.create(
        KVCacheType.ROPE, KVCacheLayout.SPARSECORE, NUM_BLOCKS, PAGE, ROPE, 4
    )
    return nope, rope


def _step_inputs(step, layer, q_lens, num_tokens):
    rng = np.random.default_rng(1000 * step + layer)
    f = lambda *s: rng.standard_normal(s).astype(np.float32)  # noqa: E731
    fp8 = lambda x: jnp.asarray(x).astype(jnp.float8_e4m3fn)  # noqa: E731
    kq = jax.lax.bitcast_convert_type(fp8(f(num_tokens, D_IDX)), jnp.uint8)
    k_packed = jnp.concatenate([kq, jnp.full((num_tokens, 1), 127, jnp.uint8)], axis=-1)
    return dict(
        q_bytes=jax.lax.bitcast_convert_type(
            fp8(f(num_tokens, H_IDX, D_IDX) * 0.5), jnp.uint8
        ),
        weights=jnp.asarray(f(num_tokens, H_IDX)).astype(jnp.bfloat16),
        k_packed=k_packed,
        ql_nope=jnp.asarray(f(num_tokens, HEADS, LKV)).astype(jnp.bfloat16),
        q_pe=jnp.asarray(f(num_tokens, HEADS, ROPE)).astype(jnp.bfloat16),
        kv_c=fp8(f(num_tokens, LKV)),
        k_pe=fp8(f(num_tokens, ROPE)),
    )


def test_stacked_caches_match_per_layer_bitwise():
    assert _run_stacked_vs_per_layer(fault=None) == []


def test_wrong_stack_row_is_caught():
    """Positive control: every stacked layer addressing row 0 (layers collide in
    one row) must break the bitwise match -- outputs and caches."""
    bad = _run_stacked_vs_per_layer(fault="row0")
    assert any(b.startswith("out ") for b in bad), bad
    assert any(b.startswith("cache ") for b in bad), bad


def test_best_env_stacked_caches_match_per_layer_bitwise():
    """The best config's knobs: live-group + mixed fused prep, fused decode
    prep, KV write in the gather -- every one on the per-layer geometry with
    the row offset in the SparseCore gather -- stacked == per-layer, bitwise.
    The run asserts those paths fired (`_check_best_paths`)."""
    with mock.patch.dict(os.environ, BEST_ENV):
        assert _run_stacked_vs_per_layer(fault=None, steps=STEPS_BEST) == []


def test_best_env_whole_output_stacked_caches_match_per_layer_bitwise():
    """The best config's knobs with the op's output read whole
    (`merge_pieces=False`, the call TPU_DCP_MLA_OPROJ_ALL_REDUCE makes on every
    bucket): stacked == per-layer, bitwise, on the same paths."""
    with mock.patch.dict(os.environ, BEST_ENV):
        assert (
            _run_stacked_vs_per_layer(fault=None, steps=STEPS_BEST, merge_pieces=False)
            == []
        )


def test_best_env_whole_output_wrong_stack_row_is_caught():
    with mock.patch.dict(os.environ, BEST_ENV):
        bad = _run_stacked_vs_per_layer(
            fault="row0", steps=STEPS_BEST, merge_pieces=False
        )
    assert any(b.startswith("out ") for b in bad), bad
    assert any(b.startswith("cache ") for b in bad), bad


def test_best_env_wrong_stack_row_is_caught():
    with mock.patch.dict(os.environ, BEST_ENV):
        bad = _run_stacked_vs_per_layer(fault="row0", steps=STEPS_BEST)
    assert any(b.startswith("out ") for b in bad), bad
    assert any(b.startswith("cache ") for b in bad), bad


def test_best_env_wrong_gather_offset_is_caught():
    """Positive control for the in-kernel offset: the stacked arm's SparseCore
    gather gets the NEXT layer's row offset (row 0 too). The gathered rows and
    the in-gather KV writes then address another layer, so outputs (and the
    caches the gather writes on the verify / decode steps) must differ -- a
    gather ignoring `row_offset` would leave this undetected."""
    with mock.patch.dict(os.environ, BEST_ENV):
        bad = _run_stacked_vs_per_layer(fault="wrong_offset", steps=STEPS_BEST)
    assert any(b.startswith("out ") for b in bad), bad
    assert any(b.startswith("cache ") for b in bad), bad


def _spied(fault):
    """Spies on the window preps and the SparseCore gather (calls are traces);
    fault="wrong_offset" shifts every stacked-cache gather's row offset by one
    layer."""
    from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla
    from vllm_torchtpu.kernels.mla.sparse import native_sc_gather

    real_gather = native_sc_gather.dsa_gather_native_sc
    layer_rows = NUM_BLOCKS * PAGE
    gathers = []

    def gather(nope, rope, indices, **kw):
        stacked = nope.shape[0] == N_LAYERS * NUM_BLOCKS
        if fault == "wrong_offset" and stacked:
            row = (kw.get("row_offset") or 0) // layer_rows
            kw["row_offset"] = ((row + 1) % N_LAYERS) * layer_rows or None
        gathers.append(
            dict(
                stacked=stacked,
                row_offset=kw.get("row_offset"),
                kv_write=kw.get("kv_write") is not None,
            )
        )
        return real_gather(nope, rope, indices, **kw)

    preps = {}
    patches = [mock.patch.object(native_sc_gather, "dsa_gather_native_sc", gather)]
    for name in (
        "prepare_capped_window_indices",
        "prepare_capped_window_live",
        "prepare_capped_window_live_fused",
        "prepare_capped_window_fused",
    ):
        real = getattr(sparse_mla, name)

        def spy(*a, _real=real, _name=name, **kw):
            preps.setdefault(_name, []).append(kw.get("total_num_pages"))
            return _real(*a, **kw)

        patches.append(mock.patch.object(sparse_mla, name, spy))
    return patches, preps, gathers


def _check_best_paths(preps, gathers):
    """The best-env run took the live-group fused prep (4096 buckets) and the
    fused decode prep (16-row buckets), every prep on the per-layer geometry;
    the gather wrote KV and carried the stack row offsets (rows 1 and 2)."""
    assert preps.get("prepare_capped_window_live_fused"), preps.keys()
    assert preps.get("prepare_capped_window_fused"), preps.keys()
    assert not preps.get("prepare_capped_window_live"), preps.keys()
    assert all(n == NUM_BLOCKS for v in preps.values() for n in v), preps
    assert any(g["kv_write"] for g in gathers)
    layer_rows = NUM_BLOCKS * PAGE
    offs = {g["row_offset"] for g in gathers if g["stacked"]}
    assert offs == {None, layer_rows, 2 * layer_rows}, offs
    assert all(g["row_offset"] is None for g in gathers if not g["stacked"])
    assert any(g["kv_write"] and g["row_offset"] for g in gathers)


def _run_stacked_vs_per_layer(fault=None, steps=STEPS, merge_pieces=True):
    """Runs the steps on per-layer and stacked caches. fault=None: asserts every
    output and cache bitwise equal (returns []). fault="row0": the stacked arm's
    kv_stack is (0, NUM_BLOCKS) for every layer; fault="wrong_offset": the
    stacked arm's gathers get the next layer's row offset. merge_pieces=False
    passes the op's whole-output flag. Returns the mismatches."""
    patches, preps, gathers = _spied(fault)
    jax.clear_caches()  # the spies count traces
    for p in patches:
        p.start()
    try:
        bad = _run(fault, steps, merge_pieces)
    finally:
        for p in patches:
            p.stop()
    if fault is None and steps is STEPS_BEST:
        _check_best_paths(preps, gathers)
    return bad


def _run(fault, steps, merge_pieces=True):
    from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
        streamidx_topk_dcp_step,
    )

    bad = []
    from vllm_torchtpu.layers.core.attention_interface import (
        sparse_mla_attention_dcp,
    )

    dcp = _dcp_size()
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:dcp]), (DCP_AXIS_NAME,))
    sharded = jax.sharding.NamedSharding(mesh, P(DCP_AXIS_NAME))
    nope_spec, rope_spec = _specs()
    idx_shape = get_kv_cache_shape(NUM_BLOCKS, PAGE, D_IDX, KVLayout.SEQ_ALONG_LANE)

    # Virtual block table, replicated; one id past the pool (resolved by `%`).
    table = np.random.default_rng(7).permutation(NUM_BLOCKS)[: 4 * PAGES_PER_SEQ]
    table = table.astype(np.int32).reshape(4, PAGES_PER_SEQ)
    table[1, 1] += NUM_BLOCKS
    block_tables = jnp.asarray(table.reshape(-1))

    def zeros(per_layer_shape, dtype, stacked):
        lead = (dcp * N_LAYERS,) if stacked else (dcp * per_layer_shape[0],)
        rest = per_layer_shape if stacked else per_layer_shape[1:]
        return jax.device_put(jnp.zeros(lead + tuple(rest), dtype), sharded)

    caches = {}
    for stacked in (False, True):
        if stacked:
            caches[stacked] = [
                zeros(nope_spec.shape, nope_spec.jax_dtype, True),
                zeros(rope_spec.shape, rope_spec.jax_dtype, True),
                zeros(idx_shape, jnp.uint8, True),
            ]
        else:
            caches[stacked] = [
                [
                    zeros(nope_spec.shape, nope_spec.jax_dtype, False),
                    zeros(rope_spec.shape, rope_spec.jax_dtype, False),
                    zeros(idx_shape, jnp.uint8, False),
                ]
                for _ in range(N_LAYERS)
            ]

    # TPU_DCP_RANK_OPERAND: the shard's rank as data (as the adapter passes it).
    rank_kw = {}
    if not merge_pieces:
        rank_kw["merge_pieces"] = False
    if os.environ.get("TPU_DCP_RANK_OPERAND") == "1":
        rank_kw["dcp_rank_ids"] = jax.device_put(
            jnp.arange(dcp, dtype=jnp.int32), sharded
        )
    nontrivial_topk = False
    for step, (q_lens, seq_lens, rd, num_tokens) in enumerate(steps):
        qsl = jnp.asarray(np.concatenate([[0], np.cumsum(q_lens)]), jnp.int32)
        seq_lens_a = jnp.asarray(seq_lens, jnp.int32)
        dist_attn = jnp.asarray(rd, jnp.int32)
        dist_idx = jnp.asarray([rd[0], rd[0], rd[2]], jnp.int32)
        for layer in range(N_LAYERS):
            x = _step_inputs(step, layer, q_lens, num_tokens)
            outs = {}
            for stacked in (False, True):
                if stacked:
                    nope, rope, idx = caches[True]
                    kv_stack = (0 if fault == "row0" else layer, NUM_BLOCKS)
                else:
                    nope, rope, idx = caches[False][layer]
                    kv_stack = None
                idx, local_topk = streamidx_topk_dcp_step(
                    x["q_bytes"],
                    x["weights"],
                    idx,
                    x["k_packed"],
                    seq_lens_a,
                    block_tables,
                    qsl,
                    dist_idx,
                    mesh=mesh,
                    topk=TOPK,
                    dcp_size=dcp,
                    interleave_size=INTERLEAVE,
                    chunk_tokens=None,
                    kv_cache_mode="sal",
                    kv_stack=kv_stack,
                )
                nope, rope, out, lse = sparse_mla_attention_dcp(
                    x["ql_nope"],
                    x["q_pe"],
                    x["kv_c"],
                    x["k_pe"],
                    nope,
                    rope,
                    local_topk,
                    seq_lens_a,
                    block_tables,
                    qsl,
                    dist_attn,
                    mesh,
                    nope_spec,
                    rope_spec,
                    sm_scale=SM_SCALE,
                    k_scale=1.0,
                    dcp_size=dcp,
                    interleave_size=INTERLEAVE,
                    kv_stack=kv_stack,
                    **rank_kw,
                )
                if stacked:
                    caches[True] = [nope, rope, idx]
                else:
                    caches[False][layer] = [nope, rope, idx]
                # Valid token rows only: rows past the batch are padding the
                # kernel leaves unspecified.
                total = int(sum(q_lens))
                outs[stacked] = [
                    np.asarray(a).reshape(dcp, num_tokens, *a.shape[1:])[:, :total]
                    for a in (local_topk, out, lse)
                ]
            for name, a, b in zip(
                ("local_topk", "out", "lse"), outs[False], outs[True]
            ):
                if fault is not None:
                    if not np.array_equal(a, b):
                        bad.append(f"{name} step {step} layer {layer}")
                    continue
                np.testing.assert_array_equal(
                    b, a, err_msg=f"step {step} layer {layer}: {name}"
                )
            topk = outs[False][0]
            assert (topk >= 0).any(), f"step {step} layer {layer}: empty top-k"
            assert np.isfinite(outs[False][1].astype(np.float32)).all()
            if step == 0:
                # The 3000-token row has more context than k: selection is real.
                per_tok = (topk >= 0).sum(axis=(0, 2))
                nontrivial_topk |= bool((per_tok[: q_lens[0]] == TOPK).any())
    assert nontrivial_topk, "no row selected a strict top-k subset"

    # Final caches: the stack's row l on every rank == layer l's cache.
    for kind in range(3):
        stack = np.asarray(caches[True][kind])
        stack = stack.reshape(dcp, N_LAYERS, *stack.shape[1:])
        for layer in range(N_LAYERS):
            per = np.asarray(caches[False][layer][kind])
            per = per.reshape(dcp, *stack.shape[2:])
            assert per.any(), f"kind {kind} layer {layer}: nothing written"
            if fault is not None:
                if not np.array_equal(stack[:, layer], per):
                    bad.append(f"cache kind {kind} layer {layer}")
                continue
            np.testing.assert_array_equal(
                stack[:, layer], per, err_msg=f"cache kind {kind} layer {layer}"
            )
    return bad
