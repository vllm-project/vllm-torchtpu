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
"""MSA index reuse (IndexCache, TPU_MINIMAX_M3_MSA_INDEX_REUSE_DECODE) under
DCP, interpret mode: the layer patterns, the default path unchanged, a
Shared layer equal to its own forward with the Full layer's selection forced
in, the attend+merge selection programs, and which ops the factory builds."""

import os
import subprocess
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.minimax_m3 import index_reuse
from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp, msa_fp4

PAGE = 128
HEAD_DIM = 128
HEADS = 4  # query heads per KV head
MAX_SEQS = 4
PAGES = 24  # physical pages per rank
SCALE = HEAD_DIM**-0.5
TOPK, CONFIG = 3, dict(init_blocks=0, local_blocks=1)
# A decode row, a chunk, spec-decode rows and a short request.
Q_LENS, CTX_LENS = [1, 130, 3, 1], [1100, 200, 0, 40]
PACKING = {jnp.bfloat16: 2, jnp.float8_e4m3fn: 4}


# ------------------------------------------------------------- patterns
def test_patterns():
    full = index_reuse.parse_pattern
    assert full("", 6) == full("1", 6) == frozenset(range(6))
    assert full("2", 6) == full("stride:2", 6) == {0, 2, 4}
    assert full("4", 10) == {0, 4, 8}
    assert full("full:0,1,5", 6) == {0, 1, 5}
    assert full("FFSFSS", 6) == {0, 1, 3}
    assert full("ffsfss", 6) == {0, 1, 3}
    for bad in ("0", "stride:0", "full:1,2", "FSF", "SFFFFF", "x", "full:0,9"):
        with pytest.raises(ValueError, match=index_reuse.ENV):
            full(bad, 6)


def test_plan_from_env(monkeypatch):
    monkeypatch.delenv(index_reuse.ENV, raising=False)
    assert index_reuse.plan_from_env(60) is None  # default off
    assert index_reuse.plan_from_env(60, "1") is None
    plan = index_reuse.plan_from_env(60, "2")
    assert plan.active and plan.full_decode == frozenset(range(0, 60, 2))
    assert plan.is_full(0) and not plan.is_full(1) and plan.is_full(58)
    monkeypatch.setenv(index_reuse.ENV, "full:0,1,2,4")
    plan = index_reuse.plan_from_env(8)
    assert [plan.is_full(i) for i in range(8)] == [1, 1, 1, 0, 1, 0, 0, 0]
    assert index_reuse.pattern_string(plan.full_decode, 8) == "FFFSFSSS"


def test_decode_step():
    # 16 request slots, 1 + 3 spec tokens: buckets up to 64 rows are decode.
    assert index_reuse.is_decode_step(64, 16, 4)
    assert index_reuse.is_decode_step(48, 16, 4)
    assert not index_reuse.is_decode_step(65, 16, 4)
    assert not index_reuse.is_decode_step(2048, 16, 4)


# ------------------------------------------------------- DCP layer helpers
def _setup(seed, dcp):
    """A step of `Q_LENS` new tokens over `CTX_LENS` context: ``(rng,
    num_tokens, (seq_lens, block_tables, cu_q_lens))`` with vLLM's logical
    DCP blocks, each rank backing them with its own pages."""
    rng = np.random.default_rng(seed)
    logical = -(-max(np.add(Q_LENS, CTX_LENS)) // (PAGE * dcp))
    num_seqs, num_real = len(Q_LENS), sum(Q_LENS)
    num_tokens = num_real + 3
    seq_lens = np.zeros(MAX_SEQS, np.int32)
    seq_lens[:num_seqs] = np.add(CTX_LENS, Q_LENS)
    cu_q = np.full(MAX_SEQS + 1, num_real, np.int32)
    cu_q[: num_seqs + 1] = np.concatenate([[0], np.cumsum(Q_LENS)])
    tables = np.zeros((MAX_SEQS, logical), np.int32)
    tables[:num_seqs] = rng.permutation(np.arange(1, PAGES))[
        : num_seqs * logical
    ].reshape(num_seqs, logical)
    meta = (jnp.asarray(seq_lens), jnp.asarray(tables.reshape(-1)), jnp.asarray(cu_q))
    return rng, num_tokens, meta


def _pack(rows, dtype):
    """``[pages, S, W]`` logical rows -> a packed ``[pages, S / p, p, W]``
    cache of `dtype`."""
    pages, page, width = rows.shape
    p = PACKING[dtype]
    return jnp.asarray(rows.reshape(pages, page // p, p, width)).astype(dtype)


def _layer(rng, num_tokens, dcp, g, dtype):
    """One layer's per-rank caches (random context) and step inputs, for
    `g` KV heads per rank (4-bit caches for ``dtype`` "fp4")."""

    def normal(*shape):
        return rng.standard_normal(shape)

    def bf16(*shape):
        return jnp.asarray(normal(*shape)).astype(jnp.bfloat16)

    layer = {"scale": [None] * dcp}
    if dtype == "fp4":
        nib = lambda w: rng.integers(0, 16, (PAGES, PAGE, w))  # noqa: E731
        layer["kv"] = [
            msa_fp4.pack_rows(jnp.asarray(nib(2 * g * HEAD_DIM))) for _ in range(dcp)
        ]
        layer["index"] = [
            msa_fp4.pack_rows(
                jnp.asarray(nib(HEAD_DIM)),
                scales=jnp.asarray(rng.uniform(0.05, 0.2, (PAGES, PAGE)), jnp.float32),
            )
            for _ in range(dcp)
        ]
        layer["scale"] = [
            jnp.asarray(
                rng.uniform(0.05, 0.2, (PAGES, msa_fp4.scale_rows(g), PAGE)),
                jnp.float32,
            )
            for _ in range(dcp)
        ]
    else:
        layer["kv"] = [
            _pack(normal(PAGES, PAGE, 2 * g * HEAD_DIM), dtype) for _ in range(dcp)
        ]
        layer["index"] = [
            _pack(normal(PAGES, PAGE, HEAD_DIM), dtype) for _ in range(dcp)
        ]
    # The group's query heads, KV head h owning [h * HEADS, (h + 1) * HEADS).
    layer["q"] = bf16(num_tokens, g * HEADS, HEAD_DIM)
    layer["k"] = bf16(num_tokens, g, HEAD_DIM)
    layer["v"] = bf16(num_tokens, g, HEAD_DIM)
    layer["iq"] = bf16(num_tokens, g, HEAD_DIM)
    layer["ik"] = bf16(num_tokens, HEAD_DIM)
    return layer


def _full(layer, meta, dcp):
    """Every rank's Full-layer write + candidates: (per-rank caches, the
    all-gathered candidates [dcp, T, G * 2 * topk])."""
    distribution = jnp.asarray([0, 0, len(Q_LENS)], jnp.int32)
    caches, cands = [], []
    for rank in range(dcp):
        kv, index, *rest = msa_dcp.msa_write_and_candidates_dcp(
            layer["k"],
            layer["v"],
            layer["iq"],
            layer["ik"],
            layer["kv"][rank],
            layer["index"][rank],
            *meta,
            distribution,
            topk=TOPK,
            rank=rank,
            dcp=dcp,
            scale_cache=layer["scale"][rank],
            interpret=True,
            **CONFIG,
        )
        caches.append((kv, index, rest[0] if len(rest) > 1 else None))
        cands.append(rest[-1])
    return caches, jnp.stack(cands)


def _shared_writes(layer, meta, dcp):
    out = []
    for rank in range(dcp):
        kv, index, *scale = msa_dcp.msa_write_dcp(
            layer["k"],
            layer["v"],
            layer["ik"],
            layer["kv"][rank],
            layer["index"][rank],
            *meta,
            rank=rank,
            dcp=dcp,
            scale_cache=layer["scale"][rank],
            interpret=True,
        )
        out.append((kv, index, scale[0] if scale else None))
    return out


def _q(layer, fp4):
    return msa_fp4.rotate_queries(layer["q"]) if fp4 else layer["q"]


def _attend(q, caches, meta, dcp, candidates=None, selected=None):
    """Every rank's partial ``(out, lse)`` over `candidates`
    (`msa_attend_dcp`) or a merged `selected` (`msa_attend_selected_dcp`)."""
    res = []
    for rank in range(dcp):
        kv, _, scale = caches[rank]
        common = dict(sm_scale=SCALE, rank=rank, dcp=dcp, scale_cache=scale)
        if selected is None:
            out = msa_dcp.msa_attend_dcp(
                q, kv, candidates, *meta, topk=TOPK, interpret=True, **common
            )
        else:
            out = msa_dcp.msa_attend_selected_dcp(
                q, kv, selected, *meta, interpret=True, **common
            )
        res.append(tuple(np.asarray(x.astype(jnp.float32)) for x in out))
    return res


# ------------------------------------------------- off: same results
@pytest.mark.parametrize("fp4", [False, True])
def test_attend_split_is_bitwise(fp4):
    """`msa_attend_dcp` (the default path) = merge + `msa_attend_selected_dcp`,
    and `return_selected` adds the merge without changing the outputs."""
    dcp = 2
    rng, num_tokens, meta = _setup(1, dcp)
    layer = _layer(rng, num_tokens, dcp, 1, "fp4" if fp4 else jnp.bfloat16)
    caches, cands = _full(layer, meta, dcp)
    selected = msa_dcp.merge_head_candidates(cands, TOPK, True)
    q = _q(layer, fp4)
    for rank in range(dcp):
        kv, _, scale = caches[rank]
        common = dict(sm_scale=SCALE, rank=rank, dcp=dcp, scale_cache=scale)
        common["interpret"] = True
        ref = msa_dcp.msa_attend_dcp(q, kv, cands, *meta, topk=TOPK, **common)
        out, lse, sel = msa_dcp.msa_attend_dcp(
            q, kv, cands, *meta, topk=TOPK, return_selected=True, **common
        )
        split = msa_dcp.msa_attend_selected_dcp(q, kv, selected, *meta, **common)
        np.testing.assert_array_equal(np.asarray(sel), np.asarray(selected))
        for want, a, b in zip(ref, (out, lse), split):
            np.testing.assert_array_equal(np.asarray(want), np.asarray(a))
            np.testing.assert_array_equal(np.asarray(want), np.asarray(b))


@pytest.mark.parametrize(
    "dtype,g", [(jnp.bfloat16, 1), (jnp.float8_e4m3fn, 2), ("fp4", 1), ("fp4", 2)]
)
def test_shared_write_matches_full_write(dtype, g):
    """A Shared layer's writes (`msa_write_dcp`) are the Full forward's, byte
    for byte (K/V, index keys and, 4-bit, the scale page)."""
    dcp = 2 * g
    rng, num_tokens, meta = _setup(2, dcp)
    layer = _layer(rng, num_tokens, dcp, g, dtype)
    caches, _ = _full(layer, meta, dcp)
    for want, got in zip(caches, _shared_writes(layer, meta, dcp)):
        for w, x in zip(want, got):
            if w is None:
                assert x is None
            else:
                np.testing.assert_array_equal(
                    np.asarray(w.view(jnp.uint8) if w.dtype.itemsize == 1 else w),
                    np.asarray(x.view(jnp.uint8) if x.dtype.itemsize == 1 else x),
                )


# --------------------------------------- on: the Full layer's selection
@pytest.mark.parametrize(
    "dcp,g,dtype",
    [(2, 1, jnp.bfloat16), (2, 1, "fp4"), (4, 2, jnp.bfloat16), (8, 4, "fp4")],
)
def test_shared_layer_equals_forced_full_selection(dcp, g, dtype):
    """Layer B Shared after Full layer A: every rank's partial of B equals
    B's own Full forward with A's gathered candidates forced into its
    attend, bit for bit (G = 4 heads at DCP 8 is the c64 layout); B's own
    selection would differ."""
    fp4 = dtype == "fp4"
    rng, num_tokens, meta = _setup(10 + dcp, dcp)
    a = _layer(rng, num_tokens, dcp, g, dtype)
    b = _layer(rng, num_tokens, dcp, g, dtype)
    _, cands_a = _full(a, meta, dcp)
    selected_a = msa_dcp.merge_head_candidates(cands_a, TOPK, True)
    q = _q(b, fp4)
    got = _attend(q, _shared_writes(b, meta, dcp), meta, dcp, selected=selected_a)
    caches_b, cands_b = _full(b, meta, dcp)
    want = _attend(q, caches_b, meta, dcp, candidates=cands_a)
    for w, x in zip(want, got):
        for wi, xi in zip(w, x):
            np.testing.assert_array_equal(wi, xi)
    own_b = np.asarray(msa_dcp.merge_head_candidates(cands_b, TOPK, True))
    num_real = sum(Q_LENS)
    assert np.any(own_b[:num_real] != np.asarray(selected_a)[:num_real])


# ------------------------------------- attend+merge selection programs
_SCRIPT = r"""
import os, sys
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import Mesh, PartitionSpec as P
sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), "src")]
from tests.models.minimax_m3 import test_msa_index_reuse as t
from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp, msa_fp4

small, fp4 = sys.argv[1] == "small", sys.argv[2] == "fp4"
if not small:
    msa_dcp.DCP_SMALL_BATCH_TOKENS = 0
dcp = 2
rng, num_tokens, meta = t._setup(20, dcp)
layer = t._layer(rng, num_tokens, dcp, 1, "fp4" if fp4 else jnp.bfloat16)
caches, cands = t._full(layer, meta, dcp)
q = t._q(layer, fp4)  # [T, dcp * Hl, D]
hl = q.shape[1] // dcp
q_hm = q.transpose(1, 0, 2)  # head-major: rank r holds heads [r * hl, (r + 1) * hl)
mesh = Mesh(np.array(jax.devices()[:dcp]), ("dcp",))
sh, rep = P("dcp"), P()
extra, extra_specs = (), ()
if fp4:
    had = jnp.asarray(msa_fp4.hadamard(t.HEAD_DIM), jnp.bfloat16)
    extra = (jnp.concatenate([c[2] for c in caches]), had)
    extra_specs = (sh, rep)
args = (jnp.arange(dcp, dtype=jnp.int32), q_hm, jnp.concatenate([c[0] for c in caches]))

def run(selection, out_specs, sel_or_cands):
    return jax.shard_map(
        lambda *a: msa_dcp.attend_and_merge(*a, axis="dcp", dcp=dcp,
            sm_scale=t.SCALE, topk=t.TOPK, interpret=True,
            **({} if selection == "candidates" else {"selection": selection})),
        mesh=mesh, in_specs=(sh,) * 4 + (rep,) * 3 + extra_specs,
        out_specs=out_specs, check_vma=False)(*args, sel_or_cands, *meta, *extra)

base = run("candidates", sh, jnp.concatenate(list(cands)))
base = np.asarray(base.astype(jnp.float32))
assert np.all(np.isfinite(base[:sum(t.Q_LENS)])), "non-finite"
got_r, sel = run("return", (sh, sh), jnp.concatenate(list(cands)))
assert np.array_equal(np.asarray(got_r.astype(jnp.float32)), base), "return"
want_sel = np.asarray(msa_dcp.merge_head_candidates(cands, t.TOPK, True))
sel = np.asarray(sel).reshape(dcp, num_tokens, 1, t.TOPK)
for r in range(dcp):
    assert np.array_equal(sel[r], want_sel), ("selection", r)
got_g = run("given", sh, jnp.concatenate([jnp.asarray(want_sel)] * dcp))
assert np.array_equal(np.asarray(got_g.astype(jnp.float32)), base), "given"
print("OK")
"""


@pytest.mark.parametrize("fp4", ["bf16", "fp4"])
@pytest.mark.parametrize("path", ["small", "large"])
def test_attend_and_merge_selection_programs(path, fp4):
    """The single-KV-head attend+merge SPMD program (c4/c16/c48 decode) on a
    2-device CPU platform (a subprocess: the device count is fixed at JAX
    start-up), permutes (small) and XLA collectives (large): "return" gives
    the default program's output bit for bit and every rank the merged
    selection; "given" over that selection gives the same output."""
    root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS="--xla_force_host_platform_device_count=2",
        PYTHONPATH=f"{root}:{root}/src",
    )
    res = subprocess.run(
        [sys.executable, "-c", _SCRIPT, path, fp4],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=1500,
    )
    assert res.returncode == 0 and "OK" in res.stdout, res.stderr[-4000:]


# ---------------------------------------------------------- op factory
# Scales unique to this file: torch_tpu registers each jax_op by name, and a
# name registered twice in one process corrupts the op registry.
_BUILD_SCALES = iter(SCALE * (1 + i / 64) for i in range(1, 64))


@pytest.mark.parametrize(
    "fp4,dcp,reps,want_reuse,want_built",
    [
        (False, 2, 2, True, ["write0", "return0", "given0"]),  # c4 / c16
        (True, 2, 2, True, ["write1", "return1", "given1"]),  # c48
        (True, 8, 2, True, ["write1"]),  # c64: the head-parallel op as is
        (False, 8, 2, False, []),  # bf16 over 4 KV heads: refused
    ],
)
def test_factory_builds_reuse_ops_only_when_on(
    fp4, dcp, reps, want_reuse, want_built, monkeypatch
):
    """Off: no reuse op is built (the op set of before). On: the served
    paths build theirs; others refuse at build time and the op ignores the
    step's dict."""
    from vllm_torchtpu.models.minimax_m3.ops import msa as msa_ops

    built = []
    real_am = msa_ops._get_msa_dcp_attend_merge_op

    def am(*a, **kw):
        sel = a[5] if len(a) > 5 else kw.get("selection", "candidates")
        if sel != "candidates":
            built.append(f"{sel}{int(a[3])}")
            return object()
        return real_am(*a, **kw)

    monkeypatch.setattr(msa_ops, "_get_msa_dcp_attend_merge_op", am)
    monkeypatch.setattr(
        msa_ops,
        "_get_msa_dcp_write_op",
        lambda dcp, suffix, fp4: built.append(f"write{int(fp4)}") or object(),
    )
    off = msa_ops.get_msa_dcp_op(
        next(_BUILD_SCALES), 16, 0, 1, dcp, reps, fp4=fp4, decode_rows=4
    )
    assert built == [] and off.index_reuse is False
    on = msa_ops.get_msa_dcp_op(
        next(_BUILD_SCALES),
        16,
        0,
        1,
        dcp,
        reps,
        fp4=fp4,
        decode_rows=4,
        index_reuse=True,
    )
    assert built == want_built and on.index_reuse is want_reuse
    _KEEP.extend((off, on))


_KEEP: list = []
