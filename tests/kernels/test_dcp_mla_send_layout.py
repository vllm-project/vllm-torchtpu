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
"""TPU_DCP_MLA_SEND_LAYOUT: the decode MLA kernel writes the packed real-rows
DCP merge send buffer itself.

The claim is bit-identity with the incumbent (TPU_MLA_DECODE_PROLOGUE +
TPU_DCP_LSE_MERGE_A2A_PACK + _REAL_ROWS): every row the ragged exchange sends
-- each destination's lse word row and the real tokens' partial rows -- equals
what `cp_mla_attention._all_to_all_packed` builds from the incumbent kernel's
(output, lse). Checked

* per kernel call (one device) at the served geometries -- W=16 ranks x 64
  heads (2x2x2 mirror), W=8 x 64 (2x2x1 twin), W=8 x 32 -- on decode buckets
  16 / 32 with 1..32 real tokens (incl. capped-overflow and empty rows) and a
  zero-real-token step;
* through `sparse_mla_attention_dcp` on 8 DCP ranks (production env, cache
  writes in the gather): caches bit for bit, sent rows bit for bit on every
  rank, and the merge (`merge_send_layout`, threaded stand-in exchange) equal
  to the packed real-rows merge of the incumbent partials on every row;
  `T5SL_DUMP=<npz>` saves both for the compiled real-collective check
  (tests/layers/adapter/test_dcp_send_layout_merge_multichip.py);

with positive controls the comparison must catch: partial slabs sent to the
next destination, lse lo halves dropped.
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

from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla  # noqa: E402

pytestmark = pytest.mark.tpu_test

PAD = sparse_mla.SEND_PAD_ROWS
SEND_ENV = {
    "TPU_MLA_DECODE_PROLOGUE": "1",
    "TPU_MLA_KV_WRITE_IN_GATHER": "1",
    "TPU_DCP_RANK_OPERAND": "1",
    "TPU_DCP_LSE_MERGE_A2A": "1",
    "TPU_DCP_LSE_MERGE_A2A_PACK": "1",
    "TPU_DCP_LSE_MERGE_A2A_REAL_ROWS": "1",
}


def packed_rows(out16, lse32, world, real):
    """The rows the packed real-rows exchange sends, from one rank's
    incumbent (output, lse): int16 lse words [W, V] (hi halves, lo halves,
    zeros) and int16 partial rows [W, real * L, V] (destination-major)."""
    out16 = np.asarray(out16).view(np.int16)
    tokens, heads, vdim = out16.shape
    per = heads // world
    rows = tokens * per
    o = out16.reshape(tokens, world, per, vdim).transpose(1, 0, 2, 3)
    o = o.reshape(world, rows, vdim)
    bits = np.asarray(lse32).view(np.int32).reshape(tokens, world, per)
    bits = bits.transpose(1, 0, 2).reshape(world, rows)
    words = np.zeros((world, vdim), np.int16)
    words[:, :rows] = (bits >> 16).astype(np.int16)
    words[:, rows : 2 * rows] = (((bits & 0xFFFF) ^ 0x8000) - 0x8000).astype(np.int16)
    return words, o[:, : real * per]


def sent_rows(send16, real, per):
    send16 = np.asarray(send16).view(np.int16)
    return send16[:, PAD - 1], send16[:, PAD : PAD + real * per]


def _diffs(send16, out16, lse32, world, real):
    per = out16.shape[1] // world
    want_w, want_o = packed_rows(out16, lse32, world, real)
    got_w, got_o = sent_rows(send16, real, per)
    return int((got_w != want_w).sum()), int((got_o != want_o).sum())


# --- one kernel call ---------------------------------------------------------


def _kernel_send(p, world, fault=None):
    cache = prologue._cos_sin_cache()
    _, send = sparse_mla.sparse_ragged_paged_attention(
        p["ql_nope"],
        p["nope"],
        p["rope"],
        p["topk"],
        p["page_table"],
        p["cu_q_lens"],
        p["distribution"],
        sm_scale=prologue.SM_SCALE,
        k_scale=prologue.K_SCALE,
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
        q_pe=p["q_raw"],
        q_rope_table=prologue.rope_table(p["positions"], cache),
        lse_in_kernel=True,
        prologue_fault=fault,
        send_world=world,
    )
    return np.asarray(jax.block_until_ready(send))


GEOMETRIES = ((16, 64), (8, 64), (8, 32))  # (ranks, heads): mirror, twin, test
DECODE_CELLS = ((4, (3,)), (8, (4, 3))) + tuple(c for c in prologue.CELLS if c[0] <= 32)


def _problem(query_tokens, seq_q, heads, zero=False):
    with mock.patch.object(prologue, "HEADS", heads):
        p = prologue._problem(
            np.random.default_rng(1012 + query_tokens + heads), query_tokens, seq_q
        )
    if zero:
        p["cu_q_lens"] = jnp.array([0, 0], jnp.int32)
        p["distribution"] = jnp.array([0, 0, 0], jnp.int32)
    return p


@pytest.mark.parametrize("world,heads", GEOMETRIES)
@pytest.mark.parametrize("cell", DECODE_CELLS + ((16, ()),), ids=str)
def test_kernel_send_rows_bit_identical(world, heads, cell):
    prologue._require_tpu()
    query_tokens, seq_q = cell
    p = _problem(query_tokens, seq_q, heads, zero=not seq_q)
    real = int(sum(seq_q))
    out, lse = prologue._kernel(p, candidate=True)
    send = _kernel_send(p, world)
    per = heads // world
    assert send.shape == (world, PAD + query_tokens * per, out.shape[-1]), send.shape
    nw, no = _diffs(send, prologue._bits16(out), np.asarray(lse), world, real)
    print(f"W={world} H={heads} cell={cell}: lse-word diffs {nw}, partial diffs {no}")
    assert nw == 0 and no == 0
    # Non-vacuous: the sent lse holds -inf (empty / padding rows) and finite
    # values, and the head rows other than the word row are zero.
    lse_np = np.asarray(lse)
    assert np.isneginf(lse_np).any()
    if real:
        assert np.isfinite(lse_np[:real]).any()
    assert not np.asarray(send).view(np.int16)[:, : PAD - 1].any()


@pytest.mark.parametrize("fault", ["send_dest_rotate", "send_lse_lo_skip"])
@pytest.mark.parametrize("world,heads", ((16, 64), (8, 64)))
def test_kernel_send_positive_control_fails(fault, world, heads):
    prologue._require_tpu()
    query_tokens, seq_q = 16, (4, 4, 4, 1)
    p = _problem(query_tokens, seq_q, heads)
    out, lse = prologue._kernel(p, candidate=True)
    send = _kernel_send(p, world, fault=fault)
    nw, no = _diffs(send, prologue._bits16(out), np.asarray(lse), world, sum(seq_q))
    print(f"fault {fault}: lse-word diffs {nw}, partial diffs {no}")
    assert nw or no, f"fault {fault} not caught"


# --- sparse_mla_attention_dcp on DCP ranks -----------------------------------


def _rank_sends(step, dcp):
    import torch

    send = torch.from_numpy(step["out"].view(np.int16).copy())
    return send.reshape(dcp, dcp, *send.shape[1:])


def _merge_send_every_rank(sends, real, tokens):
    """`merge_send_layout` on every DCP rank (CPU, lockstep threads standing
    in for the ragged exchange)."""
    import concurrent.futures
    import threading

    import torch

    from vllm_torchtpu.layers.adapter import cp_mla_attention as cp_mla

    world = sends.shape[0]
    slots = [None] * world
    barrier = threading.Barrier(world)
    local = threading.local()

    class _Group:
        def __init__(self, rank):
            self.world_size, self.rank_in_group, self.device_group = world, rank, None

    def ragged(x, rows, group, skip=0):
        slots[group.rank_in_group] = x
        barrier.wait()
        recv = torch.stack([slots[s][group.rank_in_group] for s in range(world)])
        barrier.wait()
        n = int(rows.reshape(-1)[0])
        out = torch.zeros_like(recv)
        out[:, skip : skip + n] = recv[:, skip : skip + n]
        return out

    qsl = torch.tensor([0, real], dtype=torch.int32)

    def run(rank):
        local.group = _Group(rank)
        return cp_mla.merge_send_layout(sends[rank], qsl, tokens)

    # The stand-in replaces the ragged exchange only: the other exchanges of
    # the send buffer (on by default since trials 4, 5 and 6) are pinned off.
    with (
        mock.patch.object(cp_mla, "_ragged_all_to_all", ragged),
        mock.patch.object(cp_mla, "get_dcp_group", lambda: local.group),
        mock.patch.object(cp_mla.envs, "TPU_DCP_MLA_SEND_SMALL_A2A", False),
        mock.patch.object(cp_mla.envs, "TPU_DCP_MLA_SEND_GROUP_ROWS", 0),
        mock.patch.object(cp_mla.envs, "TPU_DCP_MLA_SEND_LAYOUT_UNPACK", False),
        concurrent.futures.ThreadPoolExecutor(world) as pool,
    ):
        return torch.cat(list(pool.map(run, range(world))), dim=1)


def _dcp_size():
    return min(8, len(jax.devices()))


def _require_dcp8(dcp):
    """The decode buckets these steps send (16 and 32 tokens of 32 heads) fit
    the send layout only from DCP8: at DCP2, 32 tokens x 16 local heads is
    512 partial rows, past the V/2 = 256 the lse word row allows."""
    if dcp < 8:
        pytest.skip(f"send layout steps need DCP8 (8 devices), have {dcp}")


@pytest.mark.multichip
def test_dcp_send_layout_bit_identical_and_merges():
    prologue._require_tpu()
    dcp = _dcp_size()
    _require_dcp8(dcp)
    import test_t4_combined_composition as comb  # noqa: PLC0415
    import torch

    env = dict(prologue.PROD_ENV, **SEND_ENV)
    with mock.patch.dict(os.environ, env):
        jax.clear_caches()
        ref = prologue._dcp_run(dcp, candidate=True, dcp_rank_ids=comb._rank_ids(dcp))
        jax.clear_caches()
        got = _send_dcp_run_with_rank(dcp)
    dump = {}
    checked = 0
    for i, (a, b) in enumerate(zip(ref, got)):
        for k in ("nope", "rope"):
            np.testing.assert_array_equal(b[k], a[k], err_msg=f"step {i} {k}")
        tokens = a["out"].shape[0] // dcp
        if tokens > sparse_mla.FUSED_PREP_MAX_TOKENS:
            continue
        real = a["real"]
        outs, lses = prologue._rank_partials(a, dcp)
        sends = _rank_sends(b, dcp)
        for r in range(dcp):
            nw, no = _diffs(
                sends[r].numpy(),
                outs[r].view(torch.int16).numpy(),
                lses[r].numpy(),
                dcp,
                real,
            )
            assert nw == 0 and no == 0, f"step {i} rank {r}: {nw} / {no}"
        want = comb._merge_every_rank(outs, lses, on=True, real=real)
        one = _merge_send_every_rank(sends, real, tokens)
        diff = int((one.view(torch.int16) != want.view(torch.int16)).sum())
        print(
            f"step {i} ({tokens} tokens, {real} real): sent rows identical on "
            f"{dcp} ranks; merge diffs {diff} over {want.numel()} elements"
        )
        assert diff == 0, f"step {i}"
        assert torch.isfinite(one.float()).all()
        assert (one[:real] != 0).any()
        checked += 1
        dump[f"s{checked - 1}_on_out"] = a["out"]
        dump[f"s{checked - 1}_on_lse"] = a["lse"]
        dump[f"s{checked - 1}_send"] = b["out"]
        dump[f"s{checked - 1}_real"] = np.int32(real)
    assert checked == 2, checked  # decode 16 and decode 32
    path = os.environ.get("T5SL_DUMP")
    if path:
        np.savez(path, dcp=np.int32(dcp), steps=np.int32(checked), **dump)
        print("saved", path)


def _send_dcp_run_with_rank(dcp, fault=None):
    import test_t4_combined_composition as comb  # noqa: PLC0415

    from vllm_torchtpu.layers.core import attention_interface

    real_fn = attention_interface.sparse_mla_attention_dcp

    def wrapped(ql_nope, *args, **kwargs):
        if ql_nope.shape[0] > sparse_mla.FUSED_PREP_MAX_TOKENS:
            return real_fn(ql_nope, *args, **kwargs)
        kwargs["prologue_fault"] = fault
        nope, rope, send = real_fn(ql_nope, *args, send_layout=True, **kwargs)
        lse = jnp.zeros((dcp * ql_nope.shape[0], ql_nope.shape[1]), jnp.float32)
        return nope, rope, send, lse

    with mock.patch.object(attention_interface, "sparse_mla_attention_dcp", wrapped):
        return prologue._dcp_run(dcp, candidate=True, dcp_rank_ids=comb._rank_ids(dcp))


@pytest.mark.multichip
@pytest.mark.parametrize("fault", ["send_dest_rotate", "send_lse_lo_skip"])
def test_dcp_send_layout_positive_control_fails(fault):
    prologue._require_tpu()
    dcp = _dcp_size()
    _require_dcp8(dcp)
    import test_t4_combined_composition as comb  # noqa: PLC0415
    import torch

    env = dict(prologue.PROD_ENV, **SEND_ENV)
    with mock.patch.dict(os.environ, env):
        jax.clear_caches()
        ref = prologue._dcp_run(dcp, candidate=True, dcp_rank_ids=comb._rank_ids(dcp))
        jax.clear_caches()
        bad = _send_dcp_run_with_rank(dcp, fault=fault)
    caught = False
    for a, b in zip(ref, bad):
        tokens = a["out"].shape[0] // dcp
        if tokens > sparse_mla.FUSED_PREP_MAX_TOKENS:
            continue
        outs, lses = prologue._rank_partials(a, dcp)
        want = comb._merge_every_rank(outs, lses, on=True, real=a["real"])
        one = _merge_send_every_rank(_rank_sends(b, dcp), a["real"], tokens)
        caught |= not torch.equal(one.view(torch.int16), want.view(torch.int16))
    assert caught, f"fault {fault} not caught through the merge"


@pytest.mark.parametrize("rank", [False, True])
def test_sparse_mla_dcp_send_op_schema(rank):
    """The send-layout torch op registers with the prologue op's operands."""
    import torch

    from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk
    from vllm_torchtpu.layers.adapter import attention

    mesh = jax.sharding.Mesh(
        np.array(jax.devices("cpu")[:1]), (streamindex_topk.DCP_AXIS_NAME,)
    )
    name = f"t5sl_schema_{int(rank)}"
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
            impl, layer, 8, 4, q_rope=True, send_layout=True
        )
    op = f"sparse_mla_attention_dcp_{name}_qrope" + ("_rank" if rank else "") + "_send"
    schema = getattr(torch.ops.pallas, op).default._schema
    print(schema)
    assert len(schema.arguments) == 12 + int(rank), schema
    # Caches + the send buffer: a schema with the partial + lse pair's four
    # returns segfaults torch's functionalization on the three real outputs.
    assert len(schema.returns) == 3, schema


@pytest.mark.parametrize(
    "world,tokens,per", [(16, 16, 4), (16, 32, 4), (8, 16, 8), (8, 32, 8)]
)
def test_jax_merge_matches_torch_merge(world, tokens, per):
    """`merge_received_jax` (the fused small all-to-all's merge) ==
    `_merge_received` compiled by torch on one device, at the mirror (W=16) and
    twin (W=8) geometries."""
    prologue._require_tpu()
    import torch
    import torch_tpu  # noqa: F401
    from torch_tpu._internal import compile as tpu_compile

    from vllm_torchtpu.layers.adapter import cp_mla_attention as cp_mla

    rng = np.random.default_rng(world * 1000 + tokens)
    n = tokens * per
    o = rng.normal(size=(world, n, 512)).astype(np.float32)
    o *= rng.choice([1e-3, 1.0, 30.0], size=(world, n, 1))
    lse = (rng.normal(size=(world, n)) * 5).astype(np.float32)
    lse[rng.random((world, n)) < 0.3] = -np.inf
    lse[:, 3] = -np.inf  # a row no rank owns
    o[1, :, :7] = np.nan  # a rank's NaN partial (weight 0 where lse is -inf)
    lse[1, :] = -np.inf
    o_bf = jnp.asarray(o, jnp.bfloat16)
    ot = torch.from_numpy(np.asarray(o_bf).view(np.int16).copy()).view(torch.bfloat16)
    lt = torch.from_numpy(lse.copy())
    fn = torch.compile(
        lambda a, b: cp_mla._merge_received(a.unsqueeze(2), b.unsqueeze(2))
        .squeeze(1)
        .to(torch.bfloat16),
        fullgraph=True,
        dynamic=False,
        backend=tpu_compile.TpuBackend(),
    )
    got_t = fn(ot.to("tpu"), lt.to("tpu")).cpu().view(torch.int16).numpy()
    diffs = {}
    for name in cp_mla.MERGE_JAX_VARIANTS:
        f = jax.jit(
            lambda a, b, name=name: cp_mla.merge_received_jax(a, b, name).astype(
                jnp.bfloat16
            )
        )
        want_j = np.asarray(f(o_bf, jnp.asarray(lse))).view(np.int16)
        diffs[name] = int((got_t != want_j).sum())
    print(
        f"W={world} T={tokens} L={per}: jax vs torch merge diffs {diffs} / {got_t.size}"
    )
    # Known not bitwise: report, and only bound the drift.
    assert diffs[cp_mla.MERGE_JAX_VARIANTS[0]] <= max(8, got_t.size // 10000), diffs
