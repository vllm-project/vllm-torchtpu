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
"""Composition of the GLM-5.2 DCP decode flags, all on vs all off.

Flags: TPU_MLA_DECODE_PROLOGUE, TPU_MLA_KV_WRITE_IN_GATHER,
TPU_DCP_RANK_OPERAND, TPU_INDEXER_QB_OVERLAP, TPU_DCP_LSE_MERGE_A2A_PACK,
TPU_DCP_LSE_MERGE_A2A_REAL_ROWS,
TPU_DCP_MLA_MIXED_FUSED_PREP (TPU_MOE_SC_COMBINE_V5 is MoE-only, set for
completeness).

* Attention (`sparse_mla_attention_dcp`, DCP-8, test_mla_decode_prologue's
  mixed-256 + decode-16 + decode-32 steps plus a 2048-token mixed step,
  production env): with prologue +
  kv-in-gather + rank operand on, every rank's caches, partial outputs and lse
  equal the all-off run bit for bit; the merged output (all-to-all merge,
  threaded stand-in) with the pack on equals the two-exchange merge of the
  all-off partials bit for bit. Wiring: the gather does the KV writes, the
  iota all_to_all rank is never traced. `T4_COMB_DUMP=<npz>` saves both
  runs' partials for test_t4_combined_merge_multichip (compiled, real
  collectives).
* Indexer (`streamindex_topk_dcp`, DCP-8 decode bucket): rank operand + q_b
  side work on == plain call, top-k and q bit for bit; rotated rank ids are
  caught.
* Torch op schemas: the sparse MLA DCP op builds for every (q_rope, rank
  operand) combination with exactly 11 + q_rope + rank operands.
"""

import os
import sys
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

import test_mla_decode_prologue as prologue  # noqa: E402

pytestmark = pytest.mark.tpu_test

FLAGS_ON = {
    "TPU_MLA_DECODE_PROLOGUE": "1",
    "TPU_MLA_KV_WRITE_IN_GATHER": "1",
    "TPU_DCP_RANK_OPERAND": "1",
    "TPU_INDEXER_QB_OVERLAP": "1",
    "TPU_DCP_LSE_MERGE_A2A_REAL_ROWS": "1",
    "TPU_DCP_LSE_MERGE_A2A_PACK": "1",
    "TPU_DCP_MLA_MIXED_FUSED_PREP": "1",
    # Read at import (fused_moe_gmm._SC_COMBINE_V5) and MoE-only: no effect
    # here; test_ragged_gather_reduce_v5 holds its own on/off check.
    "TPU_MOE_SC_COMBINE_V5": "1",
}


def _rank_ids(dcp, shift=0):
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk

    axis = streamindex_topk.DCP_AXIS_NAME
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:dcp]), (axis,))
    ids = (jnp.arange(dcp, dtype=jnp.int32) + shift) % dcp
    return jax.device_put(ids, NamedSharding(mesh, P(axis)))


def _attention_run(dcp, on, rank_shift=0):
    from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk
    from vllm_torchtpu.kernels.mla import tc_scatter
    from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_kernel
    from vllm_torchtpu.kernels.mla.sparse import native_sc_gather

    env = dict(prologue.PROD_ENV, **(FLAGS_ON if on else {}))
    jax.clear_caches()  # the spies count traces
    with (
        mock.patch.dict(os.environ, env),
        mock.patch.object(
            tc_scatter, "scatter_caches", wraps=tc_scatter.scatter_caches
        ) as tc_spy,
        mock.patch.object(
            native_sc_gather,
            "dsa_gather_native_sc",
            wraps=native_sc_gather.dsa_gather_native_sc,
        ) as gather_spy,
        mock.patch.object(
            streamindex_topk,
            "cp_rank_as_data",
            wraps=streamindex_topk.cp_rank_as_data,
        ) as iota_spy,
        mock.patch.object(
            sparse_kernel,
            "prepare_capped_window_live_fused",
            wraps=sparse_kernel.prepare_capped_window_live_fused,
        ) as prep_spy,
    ):
        if not on:
            for k in FLAGS_ON:
                os.environ.pop(k, None)
        extra = {"dcp_rank_ids": _rank_ids(dcp, rank_shift)} if on else {}
        res = prologue._dcp_run(dcp, candidate=on, big_mixed=True, **extra)
        kvw = sum(
            c.kwargs.get("kv_write") is not None for c in gather_spy.call_args_list
        )
    return res, dict(
        tc=tc_spy.call_count,
        kvw=kvw,
        iota=iota_spy.call_count,
        prep=prep_spy.call_count,
    )


def _dcp_size():
    return min(8, len(jax.devices()))


def _ulp_diff(a, b):
    """|a - b| in bf16 ulps (bit patterns as a monotone integer line)."""

    def line(x):
        x = x.astype(np.int32)
        return np.where(x < 0, -(x & 0x7FFF), x)

    return np.abs(line(a.view(np.int16)) - line(b.view(np.int16)))


def test_attention_all_flags_on_vs_off():
    prologue._require_tpu()
    dcp = _dcp_size()
    if dcp < 2:
        pytest.skip("DCP needs at least 2 devices")
    import torch

    ref, ref_w = _attention_run(dcp, on=False)
    got, got_w = _attention_run(dcp, on=True)
    print("wiring off", ref_w, "on", got_w)
    # Off: tc_scatter writes the two decode steps, the rank comes from the iota
    # all_to_all. On: the gather writes them, the iota is never traced.
    # Mixed buckets (256, 2048): the fused prep only with the flag.
    assert ref_w["tc"] == 2 and ref_w["kvw"] == 0 and ref_w["iota"] > 0, ref_w
    assert ref_w["prep"] == 0, ref_w
    assert got_w["tc"] == 0 and got_w["kvw"] == 2 and got_w["iota"] == 0, got_w
    assert got_w["prep"] > 0, got_w
    dump = {}
    for i, (a, b) in enumerate(zip(ref, got)):
        for k in ("nope", "rope", "out", "lse"):
            n = int((np.asarray(a[k]) != np.asarray(b[k])).sum())
            print(f"step {i} {k}: {n} differing elements")
            np.testing.assert_array_equal(b[k], a[k], err_msg=f"step {i} {k}")
        lse = a["lse"].view(np.float32).reshape(dcp, -1, prologue.HEADS)
        assert np.isfinite(lse).any(axis=0)[: a["real"]].all(), f"step {i}"
        real = a["real"]
        want = _merge_every_rank(*prologue._rank_partials(a, dcp), on=False)
        one = _merge_every_rank(*prologue._rank_partials(b, dcp), on=True, real=real)
        diff = int(
            (one[:real].view(torch.int16) != want[:real].view(torch.int16)).sum()
        )
        print(
            f"step {i} merged (threaded a2a, pack + real rows vs two-exchange):"
            f" {diff} differ on {real} real rows; padding rows zero:"
            f" {bool((one[real:] == 0).all())}"
        )
        assert diff == 0, f"step {i}"
        assert (one[real:] == 0).all(), f"step {i}"
        assert torch.isfinite(one.float()).all(), f"step {i}"
        for tag, r in (("off", a), ("on", b)):
            dump[f"s{i}_{tag}_out"] = r["out"]
            dump[f"s{i}_{tag}_lse"] = r["lse"]
        dump[f"s{i}_real"] = np.int32(a["real"])
    path = os.environ.get("T4_COMB_DUMP")
    if path:
        np.savez(path, dcp=np.int32(dcp), steps=np.int32(len(ref)), **dump)
        print("saved", path)


def _merge_every_rank(outputs, lses, on, real=None):
    """`merge_lse_partials_all_to_all` on every DCP rank (CPU, lockstep threads
    standing in for the exchanges). `on`: pack + real rows with a step of `real`
    real tokens; off: the two-exchange all-to-all."""
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

    def ragged(x, rows, group):
        recv = exchange(x, group)
        n = int(rows.reshape(-1)[0])
        out = torch.zeros_like(recv)
        out[:, :n] = recv[:, :n]
        return out

    qsl = None if real is None else torch.tensor([0, real], dtype=torch.int32)

    def run(rank):
        return cp_mla.merge_lse_partials_all_to_all(
            outputs[rank], lses[rank], _Group(rank), query_start_loc=qsl
        )

    with (
        mock.patch.object(cp_mla, "_all_to_all", exchange),
        mock.patch.object(cp_mla, "_ragged_all_to_all", ragged),
        mock.patch.object(cp_mla.envs, "TPU_DCP_LSE_MERGE_A2A_PACK", on, create=True),
        mock.patch.object(
            cp_mla.envs, "TPU_DCP_LSE_MERGE_A2A_REAL_ROWS", on, create=True
        ),
        concurrent.futures.ThreadPoolExecutor(world) as pool,
    ):
        return torch.cat(list(pool.map(run, range(world))), dim=1)


def test_attention_all_flags_on_rank_control_fails():
    """All flags on with the rank ids rotated by one: caches or partials move."""
    prologue._require_tpu()
    dcp = _dcp_size()
    if dcp < 2:
        pytest.skip("DCP needs at least 2 devices")
    ref, _ = _attention_run(dcp, on=False)
    bad, _ = _attention_run(dcp, on=True, rank_shift=1)
    assert any(
        not np.array_equal(a[k], b[k])
        for a, b in zip(ref, bad)
        for k in ("nope", "rope", "out", "lse")
    ), "rotated rank ids not caught with every flag on"


@pytest.mark.multichip
@pytest.mark.parametrize("q_len", [16, 256], ids=["decode16", "mixed256"])
def test_indexer_rank_operand_and_q_b_overlap(q_len):
    """streamindex_topk_dcp with the rank operand and the pinned q_b overlap
    (the served setting: pin decode buckets only) == the plain call, top-k and
    q bit for bit; rotated rank ids are caught."""
    import importlib.util

    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
        streamindex_topk_dcp,
    )
    from vllm_torchtpu.kernels.quantized_matmul import fp8_skinny
    from vllm_torchtpu.layers.adapter.linear_common import _fp8_linear_pallas_jax

    dcp = 8
    if len(jax.devices()) < dcp:
        pytest.skip(f"needs {dcp} devices")
    path = os.path.join(os.path.dirname(__file__), "test_dsv4_streamindex_topk_dcp.py")
    spec = importlib.util.spec_from_file_location("_t4c_sidx", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _, args, kwargs = mod._dcp_case(dcp, 31, q_len=q_len)
    x, w, s = mod._qb_overlap_operands(m=q_len)
    base = np.asarray(streamindex_topk_dcp(*args, **kwargs))
    want_q = np.asarray(jax.jit(_fp8_linear_pallas_jax)(x, w, s)).view(np.uint16)
    ov = dict(
        overlap_fn=_fp8_linear_pallas_jax,
        overlap_args=(x, w, s),
        overlap_pin_max_tokens=fp8_skinny.MAX_M,
    )
    topk, q = streamindex_topk_dcp(*args, **kwargs, **ov, dcp_rank_ids=_rank_ids(dcp))
    np.testing.assert_array_equal(np.asarray(topk), base)
    np.testing.assert_array_equal(np.asarray(q).view(np.uint16), want_q)
    bad, _ = streamindex_topk_dcp(*args, **kwargs, **ov, dcp_rank_ids=_rank_ids(dcp, 1))
    assert not np.array_equal(np.asarray(bad), base), "rotated rank ids not caught"


@pytest.mark.parametrize("q_rope", [False, True])
@pytest.mark.parametrize("rank", [False, True])
def test_sparse_mla_dcp_op_schema(q_rope, rank):
    """The torch op for each (q_rope, rank operand) variant registers (no
    varargs in the inferred schema) with exactly its operands."""
    import torch

    from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk
    from vllm_torchtpu.layers.adapter import attention

    mesh = jax.sharding.Mesh(
        np.array(jax.devices("cpu")[:1]), (streamindex_topk.DCP_AXIS_NAME,)
    )
    name = f"t4c_schema_{int(q_rope)}{int(rank)}"
    layer = mock.Mock(
        kv_cache_quantized_dtype=None, layer_name=name, scale=0.1, mla_kv_spec=(1, 2)
    )
    impl = mock.Mock(_dcp_mesh=mesh)
    env = {"TPU_DCP_RANK_OPERAND": "1"} if rank else {}
    with (
        mock.patch.dict(os.environ, env),
        mock.patch.object(
            attention,
            "dcp_rank_operand",
            return_value=torch.zeros(1, dtype=torch.int32),
        ),
    ):
        if not rank:
            os.environ.pop("TPU_DCP_RANK_OPERAND", None)
        attention.PallasMLAttentionBackendImpl._build_sparse_mla_dcp_op(
            impl, layer, 8, 4, q_rope=q_rope
        )
    op = f"sparse_mla_attention_dcp_{name}" + ("_qrope" if q_rope else "")
    op += "_rank" if rank else ""
    schema = getattr(torch.ops.pallas, op).default._schema
    print(schema)
    assert len(schema.arguments) == 11 + int(q_rope) + int(rank), schema


@pytest.mark.parametrize("qb", [False, True])
@pytest.mark.parametrize("rank", [False, True])
def test_indexer_dcp_op_schema(qb, rank):
    """The indexer DCP op for each (q_b overlap, rank operand) variant
    registers with exactly 8 + 3 * qb + rank operands."""
    import torch

    from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op as mop

    name = f"t4c_idx_{int(qb)}{int(rank)}"
    stub = mock.Mock(topk_tokens=2048)
    stub.k_cache.prefix = name
    env = {"TPU_DCP_RANK_OPERAND": "1"} if rank else {}
    with (
        mock.patch.dict(os.environ, env),
        mock.patch.object(
            mop, "dcp_rank_operand", return_value=torch.zeros(1, dtype=torch.int32)
        ),
    ):
        if not rank:
            os.environ.pop("TPU_DCP_RANK_OPERAND", None)
        mop.VllmTPUSparseAttnIndexer._build_streamidx_dcp_op(stub, 8, 4, qb_overlap=qb)
    op = f"streamidx_topk_dcp_{name}" + ("_qb" if qb else "")
    schema = getattr(torch.ops.pallas, op).default._schema
    print(schema)
    assert len(schema.arguments) == 8 + 3 * int(qb) + int(rank), schema
