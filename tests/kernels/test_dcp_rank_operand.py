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
"""`TPU_DCP_RANK_OPERAND`: the DCP rank as a sharded i32 operand.

Each DCP op learns its shard's rank from `cp_rank_as_data` (an `all_to_all`
of an iota) unless it is handed `rank_ids`, i32[dcp] sharded over the DCP
axis with entry r = r. For each of the three sites -- the indexer K write,
`streamindex_topk_dcp`, `sparse_mla_attention_dcp` -- the two must give
bit-identical outputs, the operand path must lower with one all_to_all fewer,
and a positive control (the ranks rotated by one) must change the output, so
the equality cannot come from an operand the op ignores.
"""

import functools
import importlib.util
import os
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    DCP_AXIS_NAME,
    KVLayout,
    streamindex_topk_dcp,
)

pytestmark = pytest.mark.tpu_test

P = jax.sharding.PartitionSpec
NS = jax.sharding.NamedSharding


def _load(rel):
    """A sibling test module by path (tests/ is not a package)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", rel)
    spec = importlib.util.spec_from_file_location(
        "_dcp_rank_" + os.path.basename(rel)[:-3], path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _need(n):
    if len(jax.devices()) < n:
        pytest.skip(f"needs {n} devices, have {len(jax.devices())}")


def _mesh(n):
    return jax.sharding.Mesh(np.array(jax.devices()[:n]), (DCP_AXIS_NAME,))


def _rank_ids(mesh, n, shift=0):
    ids = (jnp.arange(n, dtype=jnp.int32) + shift) % n
    return jax.device_put(ids, NS(mesh, P(DCP_AXIS_NAME)))


def _num_a2a(text):
    return len(re.findall(r"all[_-]to[_-]all", text))


def _assert_tree_equal(a, b):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def _tree_differs(a, b):
    return any(
        not np.array_equal(np.asarray(x), np.asarray(y))
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b))
    )


@pytest.mark.multichip
@pytest.mark.parametrize("dcp_size", [4, 8])
def test_indexer_k_scatter_rank_operand(dcp_size):
    from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op

    _need(dcp_size)
    mesh = _mesh(dcp_size)
    rng = np.random.default_rng(3)
    pages_per_rank, page_size, head_dim, lkv_dim = 4, 128, 128, 256
    num_tokens = 16
    init = jax.device_put(
        jnp.asarray(
            rng.integers(
                0,
                256,
                size=(dcp_size * pages_per_rank, page_size // 4, 4, lkv_dim),
                dtype=np.uint8,
            )
        ),
        NS(mesh, P(DCP_AXIS_NAME)),
    )
    k_packed = jnp.asarray(
        rng.integers(0, 256, size=(num_tokens, head_dim + 1), dtype=np.uint8)
    )
    # Two sequences long enough that every rank owns some of the new rows.
    seq_lens = jnp.array([300, 1000], dtype=jnp.int32)
    page_indices = jnp.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 7, 16], dtype=jnp.int32)
    run = functools.partial(
        mla_attention_op._cp_scatter_indexer_k,
        mesh=mesh,
        cp_size=dcp_size,
        interleave_size=1,
        axis_name=DCP_AXIS_NAME,
        writer="xla",
        kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
    )
    args = (k_packed, seq_lens, page_indices, cu_q_lens)

    base = run(jnp.copy(init), *args)
    got = run(jnp.copy(init), *args, rank_ids=_rank_ids(mesh, dcp_size))
    _assert_tree_equal(got, base)
    assert _tree_differs(base, init), "nothing was written; the test is blind"
    wrong = run(jnp.copy(init), *args, rank_ids=_rank_ids(mesh, dcp_size, 1))
    assert _tree_differs(wrong, base), "rank_ids is ignored"

    def lower(**kw):
        return jax.jit(functools.partial(run, **kw)).lower(init, *args)

    n0 = _num_a2a(lower().as_text())
    n1 = _num_a2a(lower(rank_ids=_rank_ids(mesh, dcp_size)).as_text())
    assert n1 == n0 - 1, (n0, n1)


@pytest.mark.multichip
@pytest.mark.parametrize("dcp_size", [4, 8])
def test_streamindex_topk_dcp_rank_operand(dcp_size):
    # The DCP case of test_dsv4_streamindex_topk_dcp: same records, same call.
    _dcp_case = _load("kernels/test_dsv4_streamindex_topk_dcp.py")._dcp_case

    _need(dcp_size)
    _, args, kwargs = _dcp_case(dcp_size, 11)
    mesh = kwargs["mesh"]
    base = streamindex_topk_dcp(*args, **kwargs)
    got = streamindex_topk_dcp(*args, **kwargs, dcp_rank_ids=_rank_ids(mesh, dcp_size))
    _assert_tree_equal(got, base)
    wrong = streamindex_topk_dcp(
        *args, **kwargs, dcp_rank_ids=_rank_ids(mesh, dcp_size, 1)
    )
    assert _tree_differs(wrong, base), "dcp_rank_ids is ignored"

    n0 = _num_a2a(streamindex_topk_dcp.lower(*args, **kwargs).as_text())
    n1 = _num_a2a(
        streamindex_topk_dcp.lower(
            *args, **kwargs, dcp_rank_ids=_rank_ids(mesh, dcp_size)
        ).as_text()
    )
    assert n1 == n0 - 1, (n0, n1)


@pytest.mark.multichip
@pytest.mark.parametrize("dcp_size", [4, 8])
def test_sparse_mla_attention_dcp_rank_operand(dcp_size):
    t = _load("layers/core/test_sparse_mla_attention.py")
    from vllm_torchtpu.kernels.mla.kv_cache_utils import (
        KVCacheLayout,
        KVCacheType,
        SparseMLAKVCacheSpec,
    )
    from vllm_torchtpu.layers.core import attention_interface

    _need(dcp_size)
    mesh = _mesh(dcp_size)
    rng = np.random.default_rng(5)
    nope_spec = SparseMLAKVCacheSpec.create(
        KVCacheType.NOPE,
        KVCacheLayout.TENSORCORE,
        t.TOTAL_PAGES,
        t.PAGE_SIZE,
        t.LKV_DIM,
        t.KV_PACKING,
    )
    rope_spec = SparseMLAKVCacheSpec.create(
        KVCacheType.ROPE,
        KVCacheLayout.TENSORCORE,
        t.TOTAL_PAGES,
        t.PAGE_SIZE,
        t.ROPE_DIM,
        t.KV_PACKING,
    )

    def cache(spec):
        shape = (spec.shape[0] * dcp_size, *spec.shape[1:])
        bits = rng.integers(0, 120, size=shape, dtype=np.uint8)
        arr = jax.lax.bitcast_convert_type(jnp.asarray(bits), spec.jax_dtype)
        return jax.device_put(arr, NS(mesh, P(DCP_AXIS_NAME)))

    nope0, rope0 = cache(nope_spec), cache(rope_spec)
    num_tokens, topk, num_seqs = t.TOKEN_PAD, 128, 2
    ql_nope = jnp.asarray(
        rng.standard_normal((num_tokens, t.NUM_HEADS, t.LKV_DIM)), jnp.bfloat16
    )
    q_pe = jnp.asarray(
        rng.standard_normal((num_tokens, t.NUM_HEADS, t.ROPE_DIM)), jnp.bfloat16
    )
    kv_c = jnp.asarray(rng.standard_normal((num_tokens, t.LKV_DIM)), jnp.float8_e4m3fn)
    k_pe = jnp.asarray(rng.standard_normal((num_tokens, t.ROPE_DIM)), jnp.float8_e4m3fn)
    # Every rank attends over its first 40 local positions for every token.
    local_topk = np.full((dcp_size * num_tokens, topk), -1, np.int32)
    local_topk[:, :40] = np.arange(40)
    local_topk = jax.device_put(
        jnp.asarray(local_topk), NS(mesh, P(DCP_AXIS_NAME, None))
    )
    # Decode step at positions 60 and 90: rows land on ranks (60 // 4) % n and
    # (90 // 4) % n, at local slots inside every rank's first 40.
    seq_lens = jnp.array([61, 91], jnp.int32)
    block_tables = jnp.arange(num_seqs * t.PAGES_PER_SEQ, dtype=jnp.int32)
    query_start_loc = jnp.array([0, 1, 2], jnp.int32)
    distribution = jnp.array([num_seqs, num_seqs, num_seqs], jnp.int32)

    def run(rank_ids=None):
        return attention_interface.sparse_mla_attention_dcp(
            ql_nope,
            q_pe,
            kv_c,
            k_pe,
            jnp.copy(nope0),
            jnp.copy(rope0),
            local_topk,
            seq_lens,
            block_tables,
            query_start_loc,
            distribution,
            mesh,
            nope_spec,
            rope_spec,
            sm_scale=0.05,
            k_scale=1.0,
            dcp_size=dcp_size,
            # A rope tile packs WORD_BYTES (4) tokens; the interleave must not
            # split one across ranks.
            interleave_size=4,
            dcp_rank_ids=rank_ids,
        )

    base = run()
    _assert_tree_equal(run(_rank_ids(mesh, dcp_size)), base)
    assert _tree_differs(run(_rank_ids(mesh, dcp_size, 1)), base), (
        "dcp_rank_ids is ignored"
    )
