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
"""TPU_MLA_KV_WRITE_IN_GATHER: the SparseCore gather writes the KV rows.

The claim is bit-identity with the incumbent: the step's rows written by the
cache writer (golden `update_sparse_mla_kv_cache_dcp_jax`, which
`tc_scatter` matches), then the plain native-SC gather. Checked

* per gather call, every rank of a DCP16/il4 group, the `test_mla_tc_kv_writer`
  cells (decode, padded bucket, straddling MTP steps, page boundaries, all
  rows on one rank, all padding): both whole caches and every gathered row,
  with the step's own rows placed in the gather (first row of every owner's
  first batch, so a missing write or fence reads stale bytes), on the live
  gather and on an idle rank's zero-trip gather;
* through `sparse_mla_attention_dcp` on DCP ranks (test_mla_decode_prologue's
  prefill + decode steps, production env incl. TPU_MLA_DECODE_PROLOGUE):
  each rank's caches, partial outputs and lse;

with positive controls the comparison must catch: wrong row, write ignoring
the owner mask, wrong owner, skipped write, no fence.
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
import test_mla_tc_kv_writer as tcw  # noqa: E402

from vllm_torchtpu.kernels.mla import kv_cache_utils  # noqa: E402
from vllm_torchtpu.kernels.mla.sparse import native_sc_gather  # noqa: E402

pytestmark = pytest.mark.tpu_test

FLAG = "TPU_MLA_KV_WRITE_IN_GATHER"
OUT_SIZE = 4 * 2048  # 4 query tokens x topk 2048 (decode-sized gather)
NUM_OWNERS = 32
_WRITE_CELLS = tuple(n for n, (_, t) in tcw._CELLS.items() if t <= 32)


def _require_tpu():
    if jax.devices()[0].platform != "tpu":
        pytest.skip("the native SparseCore gather needs a TPU")


def _indices(dst_rows, num_rows, seed):
    """Random cache rows with every owned step row planted where each
    owner's first batch starts (gathered first) and again at random spots."""
    rng = np.random.default_rng(seed)
    flat = rng.integers(0, num_rows, OUT_SIZE, dtype=np.int32)
    owned = np.asarray(dst_rows)[np.asarray(dst_rows) >= 0]
    if owned.size:
        per_owner = OUT_SIZE // NUM_OWNERS
        flat[::per_owner] = np.resize(owned, NUM_OWNERS)
        spots = rng.choice(OUT_SIZE, size=4 * owned.size, replace=False)
        flat[spots] = np.resize(owned, spots.size)
    return jnp.asarray(flat.reshape(4, 2048))


def _packed(inp):
    """As the wiring hands them over: 128-lane rows (rope zero-padded)."""
    from vllm_torchtpu.layers.core import attention_interface

    nope_spec, rope_spec = tcw._specs()
    return (
        attention_interface._pack_lane_dense(inp["kv_c_normed"], nope_spec.token_bytes),
        attention_interface._pack_lane_dense(inp["k_pe"], rope_spec.token_bytes),
    )


def _gather(caches, indices, num_live, kv_write=None, fault=None):
    return jax.block_until_ready(
        native_sc_gather.dsa_gather_native_sc(
            caches[0],
            caches[1],
            indices,
            out_size=OUT_SIZE,
            atoms_per_batch=1,
            num_live_indices=jnp.int32(num_live),
            live_multiple=2048,
            kv_write=kv_write,
            write_fault=fault,
        )
    )


def _run_rank(inp, rank, num_live, fault=None):
    """(candidate outputs, oracle outputs): 2 gathered + 2 caches each."""
    dst = tcw._dst_rows(inp, rank)
    want_caches = tcw._golden(inp, rank)
    num_rows = want_caches[0].shape[0] * want_caches[0].shape[1]
    idx = _indices(dst, num_rows, seed=rank)
    want = (*_gather(want_caches, idx, num_live), *want_caches)
    got = _gather(inp["caches"], idx, num_live, (dst, *_packed(inp)), fault)
    return got, want, dst


def _compare(got, want, num_live):
    """Caches whole; gathered rows only below the live bound."""
    groups = num_live // native_sc_gather.ROPE_PACKING
    pairs = (
        (np.asarray(got[0])[:num_live], np.asarray(want[0])[:num_live]),
        (np.asarray(got[1])[:groups], np.asarray(want[1])[:groups]),
        (np.asarray(got[2]), np.asarray(want[2])),
        (np.asarray(got[3]), np.asarray(want[3])),
    )
    return [np.array_equal(g, w) for g, w in pairs]


@pytest.mark.parametrize("num_live", [OUT_SIZE, 0])
@pytest.mark.parametrize("cell", _WRITE_CELLS)
def test_gather_write_matches_writer_then_gather(cell, num_live):
    _require_tpu()
    seqs, num_tokens = tcw._CELLS[cell]
    inp = tcw._inputs(seqs, num_tokens)
    owned = 0
    for rank in range(tcw.DCP_SIZE):
        got, want, dst = _run_rank(inp, rank, num_live)
        owned += int(jnp.sum(dst >= 0))
        ok = _compare(got, want, num_live)
        assert all(ok), f"{cell} rank {rank} live {num_live}: nope/rope/caches {ok}"
    assert owned == inp["num_valid"]


@pytest.mark.parametrize(
    "fault", ["wrong_row", "skip_write", "ignore_owner", "no_fence", "late_write"]
)
@pytest.mark.parametrize("cell", tcw._CONTROL_CELLS)
def test_gather_write_positive_control_fails(cell, fault):
    """Every rank that owns a row must go red (ignore_owner: every rank
    that does not own all of them); no_fence must at least change a gathered
    row somewhere, or the fence is not shown to matter."""
    _require_tpu()
    seqs, num_tokens = tcw._CELLS[cell]
    inp = tcw._inputs(seqs, num_tokens)
    owning = caught = gathered_caught = 0
    # no_fence is a race (hit on 0-2 of ~10 owning ranks per pass, j-a5040f59
    # / j-04b743b8): repeat it so a pass that never reads stale is unlikely.
    reps = 8 if fault == "no_fence" else 1
    for rank in [r for r in range(tcw.DCP_SIZE) for _ in range(reps)]:
        got, want, dst = _run_rank(inp, rank, OUT_SIZE, fault)
        ok = _compare(got, want, OUT_SIZE)
        owns = bool(jnp.any(dst >= 0))
        if fault == "ignore_owner":
            owns = not bool(jnp.all(dst >= 0))
        if owns:
            owning += 1
            caught += not all(ok)
            gathered_caught += not (ok[0] and ok[1])
    assert owning > reps
    if fault in ("no_fence", "late_write"):
        # Caches end right; only the gathered rows can show the ordering.
        print(f"{fault}: gathered rows differ on {gathered_caught}/{owning} ranks")
        if fault == "no_fence" and cell == "decode_t16" and not gathered_caught:
            # One owned row per owning rank: the race is hit on 0-12 % of
            # passes (0/9, 1/9, 9/72, 3/72, 0/72 over five jobs);
            # straddle_chunk_t32 (2-4 rows per owner) hits it every job.
            pytest.xfail("no_fence race not hit with one row per owner")
        assert gathered_caught > 0, f"{fault} never read a stale row"
    else:
        assert caught == owning, f"{fault} missed on {owning - caught} ranks"


# --- DCP: sparse_mla_attention_dcp, caches + partials on every rank --------


def _dcp(dcp, flag, kv_write_fault=None):
    env = dict(prologue.PROD_ENV, TPU_MLA_DECODE_PROLOGUE="1")
    if flag:
        env[FLAG] = "1"
    from vllm_torchtpu.kernels.mla import tc_scatter

    jax.clear_caches()  # the spies below count traces
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
    ):
        if not flag:
            os.environ.pop(FLAG, None)
        extra = {} if kv_write_fault is None else {"kv_write_fault": kv_write_fault}
        res = prologue._dcp_run(dcp, candidate=True, **extra)
        writes_in_gather = sum(
            c.kwargs.get("kv_write") is not None for c in gather_spy.call_args_list
        )
    return res, tc_spy.call_count, writes_in_gather


def _dcp_size():
    return min(8, len(jax.devices()))


def test_dcp_caches_partials_and_lse_bit_identical():
    _require_tpu()
    dcp = _dcp_size()
    if dcp < 2:
        pytest.skip("DCP needs at least 2 devices")
    ref, ref_tc, ref_kvw = _dcp(dcp, flag=False)
    got, got_tc, got_kvw = _dcp(dcp, flag=True)
    # Wiring: the incumbent writes the 2 decode steps with tc_scatter; the
    # candidate writes them in the gather and never calls tc_scatter.
    assert (ref_tc, ref_kvw) == (2, 0), (ref_tc, ref_kvw)
    assert (got_tc, got_kvw) == (0, 2), (got_tc, got_kvw)
    for i, (a, b) in enumerate(zip(ref, got)):
        for k in ("nope", "rope", "out", "lse"):
            np.testing.assert_array_equal(b[k], a[k], err_msg=f"step {i} {k}")
        lse = a["lse"].view(np.float32).reshape(dcp, -1, prologue.HEADS)
        assert np.isfinite(lse).any(axis=0)[: a["real"]].all(), f"step {i}"


@pytest.mark.parametrize(
    "fault",
    [
        "wrong_owner",
        "wrong_row",
        "skip_write",
        "no_fence",
        "late_write",
        "ignore_owner",
    ],
)
def test_dcp_positive_control_fails(fault):
    _require_tpu()
    dcp = _dcp_size()
    if dcp < 2:
        pytest.skip("DCP needs at least 2 devices")
    ref, _, _ = _dcp(dcp, flag=False)
    got, _, kvw = _dcp(dcp, flag=True, kv_write_fault=fault)
    assert kvw == 2
    diff = {
        k: [not np.array_equal(a[k], b[k]) for a, b in zip(ref, got)]
        for k in ("nope", "rope", "out", "lse")
    }
    print(f"{fault}: differs per step {diff}")
    if fault == "no_fence" and not any(diff["out"] + diff["lse"]):
        # A race, not an order: through the full DCP call the single writer
        # has finished before any read here (measured j-a5040f59). The
        # kernel-level control above, which plants the step's rows first in
        # every owner's gather, is the one that catches it.
        pytest.xfail("no_fence race not hit at DCP timing")
    if fault in ("no_fence", "late_write"):
        # Caches end right either way; only the attention can see the order.
        assert any(diff["out"] + diff["lse"]), f"{fault} not caught"
    else:
        assert any(sum(diff.values(), [])), f"fault {fault} not caught"


def test_flag_defaults_off():
    from vllm_torchtpu import envs

    with mock.patch.dict(os.environ):
        os.environ.pop(FLAG, None)
        assert not envs.TPU_MLA_KV_WRITE_IN_GATHER


@pytest.mark.parametrize("cell", _WRITE_CELLS)
def test_lane_dense_pack_is_pack_tokens(cell):
    """The wiring's 128-lane pack == `pack_tokens` (rope: words [0, 32),
    zeros after), on raw random bytes."""
    from vllm_torchtpu.layers.core import attention_interface

    seqs, num_tokens = tcw._CELLS[cell]
    inp = tcw._inputs(seqs, num_tokens)
    for values, spec in zip(("kv_c_normed", "k_pe"), tcw._specs()):
        want = np.asarray(kv_cache_utils.pack_tokens(inp[values], spec.token_bytes))
        got = np.asarray(
            attention_interface._pack_lane_dense(inp[values], spec.token_bytes)
        )
        assert got.shape == (num_tokens, 128)
        np.testing.assert_array_equal(got[:, : want.shape[1]], want)
        assert not got[:, want.shape[1] :].any()


def test_live_rows_with_two_round_write_is_rejected():
    """row_live_atoms with a kv_write of more than 16 tokens (two write rounds)
    hangs the kernel on v7x; the wrapper refuses it before launching."""
    _require_tpu()
    tokens = 32
    out_size = tokens * 2048
    write = native_sc_gather.pack_write_rows(
        jnp.full((tokens,), -1, jnp.int32),
        jnp.zeros((tokens, 128), jnp.uint32),
        jnp.zeros((tokens, native_sc_gather.ROPE_WORDS), jnp.uint32),
    )
    with pytest.raises(ValueError, match="hangs the kernel"):
        native_sc_gather.dsa_gather_native_sc(
            jnp.zeros((64, 256, 128), jnp.uint32),
            jnp.zeros((64, 64, 128), jnp.uint32),
            jnp.zeros((out_size // 128 + 8, 128), jnp.int32),
            out_size=out_size,
            atoms_per_batch=1,
            live_multiple=2048,
            live_vec_offset=out_size,
            linear_indices=True,
            row_live_atoms=jnp.zeros((4 * tokens,), jnp.int32),
            kv_write=write,
        )
