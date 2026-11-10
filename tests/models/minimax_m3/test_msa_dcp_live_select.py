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
"""`local_candidates` with `live_select` (interpret mode): scanning only each
row block's live columns picks bit for bit what the full-width scan picks,
at the served decode shapes, with an off-by-one live bound as the positive
control."""

import functools
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.minimax_m3 import attention
from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp

PAGE = 128
TOPK = 16
Q_PER_SEQ = 8  # draft tokens per decode request


def _decode_positions(rng, *, seqs, heads, cols, dcp, rank, kinds):
    """Head-major positions ``[heads * seqs * Q_PER_SEQ]`` of a decode batch
    (``Q_PER_SEQ`` consecutive tokens per request), one context kind per
    request, cycling through `kinds`:

    - ``mixed``: 30-120K tokens, clipped to the width;
    - ``short``: fewer than `TOPK` of this rank's blocks;
    - ``none``: none of this rank's blocks (only blocks below `rank`);
    - ``full``: every column of the width;
    - ``ownlast``: 30-120K tokens whose last block is on this rank (a
      forced local pick in the block's last live column).
    """
    max_len = cols * dcp * PAGE
    lens = []
    for s in range(seqs):
        kind = kinds[s % len(kinds)]
        if kind == "mixed":
            n = int(rng.integers(30_000, 120_000))
        elif kind == "short":
            n = int(rng.integers(Q_PER_SEQ, (TOPK // 2) * dcp * PAGE))
        elif kind == "none":
            n = int(rng.integers(Q_PER_SEQ, max(rank, 1) * PAGE + 1))
        elif kind == "ownlast":
            n = int(rng.integers(30_000, 120_000)) // PAGE
            n = (n - n % dcp + rank) * PAGE + int(rng.integers(1, PAGE + 1))
        else:
            n = max_len
        lens.append(min(n, max_len))
    pos = np.concatenate(
        [np.arange(n - Q_PER_SEQ, n) for n in lens]
    ).astype(np.int32)
    return np.tile(pos, heads)


def _scores(rng, rows, cols):
    """Small integer scores (many ties) with a few ``-inf`` entries."""
    s = rng.integers(-3, 4, (rows, cols)).astype(np.float32)
    s[rng.random((rows, cols)) < 0.01] = -np.inf
    return s


def _picks(scores, pos, *, live, live_rows=None, **kw):
    s, b = msa_dcp.local_candidates(
        jnp.asarray(scores),
        jnp.asarray(pos),
        block_size=PAGE,
        topk=TOPK,
        live_rows=live_rows,
        live_select=live,
        interpret=True,
        **kw,
    )
    return np.asarray(s), np.asarray(b)


def _assert_same(want, got, rows=None):
    """Bit-identical picks (scores compared as bit patterns)."""
    rows = slice(None) if rows is None else slice(0, rows)
    np.testing.assert_array_equal(
        want[0].view(np.int32)[rows], got[0].view(np.int32)[rows]
    )
    np.testing.assert_array_equal(want[1][rows], got[1][rows])


# (name, sequences, KV heads per rank, columns per rank, dcp, rank, kinds)
_SHAPES = {
    # c64 DCP4 int4: G = 2, T = 512 -> 1024 rows in 4 transposed blocks.
    "int4-dcp4-c64": (64, 2, 512, 4, 3, ("mixed", "short", "mixed", "none")),
    "int4-dcp4-c64-rank0": (64, 2, 512, 4, 0, ("mixed", "full", "short")),
    # c4 DCP2 fp8: G = 1, T = 32 rows, row-major block.
    "fp8-dcp2-c4": (4, 1, 1024, 2, 1, ("mixed", "short", "none", "full")),
    "fp8-dcp2-c4-allfull": (4, 1, 1024, 2, 0, ("full",)),
    # Every row of every block without a live column on this rank.
    "dcp4-all-none": (32, 1, 512, 4, 3, ("none",)),
    # Rows mostly short of topk, ties everywhere.
    "dcp2-all-short": (16, 1, 256, 2, 1, ("short",)),
    # Each row block's longest request ends on this rank: an off-by-one live
    # bound drops a forced pick (the positive control's batches).
    "fp8-dcp2-c4-ownlast": (4, 1, 1024, 2, 1, ("ownlast",)),
    "int4-dcp4-c64-ownlast": (64, 2, 512, 4, 3, ("ownlast",)),
}


def check_exact(name, seed=0):
    seqs, heads, cols, dcp, rank, kinds = _SHAPES[name]
    rng = np.random.default_rng(seed)
    pos = _decode_positions(
        rng, seqs=seqs, heads=heads, cols=cols, dcp=dcp, rank=rank, kinds=kinds
    )
    scores = _scores(rng, pos.shape[0], cols)[:, None, :]
    kw = dict(init_blocks=1, local_blocks=1, rank=rank, dcp=dcp)
    want = _picks(scores, pos, live=False, **kw)
    got = _picks(scores, pos, live=True, **kw)
    _assert_same(want, got)
    return want


@pytest.mark.parametrize("name", list(_SHAPES))
def test_live_select_matches_full_width(name):
    want = check_exact(name)
    b = want[1]
    if "none" in name:
        assert (b == -1).all()
    elif "allfull" not in name and "short" not in name and "ownlast" not in name:
        # Some rows have fewer live blocks than topk, some more.
        assert ((b == -1).any(axis=1)).any() and (b >= 0).all(axis=1).any()


def test_live_select_matches_full_width_pretransposed():
    """The large-tile layout ``[P, T]`` with `live_rows`: only the live row
    blocks are picked (the rest are unset on both paths)."""
    rng = np.random.default_rng(5)
    cols, dcp, rank = 512, 4, 2
    pos = _decode_positions(
        rng, seqs=64, heads=1, cols=cols, dcp=dcp, rank=rank,
        kinds=("mixed", "short", "none", "full"),
    )
    scores = _scores(rng, pos.shape[0], cols).T
    kw = dict(init_blocks=1, local_blocks=1, rank=rank, dcp=dcp)
    live_rows = jnp.int32(300)  # 2 of 4 blocks of 128
    want = _picks(scores, pos, live=False, live_rows=live_rows, block_rows=128, **kw)
    got = _picks(scores, pos, live=True, live_rows=live_rows, block_rows=128, **kw)
    _assert_same(want, got, rows=256)


def test_live_select_fault_is_caught(monkeypatch):
    """Positive control: a live bound one loop step short (floor instead of
    ceil) drops the block's last live columns, which hold its longest
    request's forced local block. (A bound one column short is not a fault:
    loops cover whole steps and every column is still masked per row.)"""
    monkeypatch.setattr(msa_dcp, "_live_chunks", lambda need, cols: need // cols)
    jax.clear_caches()
    try:
        with pytest.raises(AssertionError):
            check_exact("fp8-dcp2-c4-ownlast")
        with pytest.raises(AssertionError):
            check_exact("int4-dcp4-c64-ownlast")
    finally:
        monkeypatch.undo()
        jax.clear_caches()


def test_live_select_flag():
    assert not attention.live_select_enabled(SimpleNamespace(additional_config={}))
    assert not attention.live_select_enabled(SimpleNamespace(additional_config=None))
    on = SimpleNamespace(additional_config={attention.LIVE_SELECT_KEY: True})
    assert attention.live_select_enabled(on)
    with pytest.raises(ValueError):
        attention.live_select_enabled(
            SimpleNamespace(additional_config={attention.LIVE_SELECT_KEY: "yes"})
        )


def test_live_select_flag_reaches_decode_candidates(monkeypatch):
    """`msa_write_and_candidates_dcp` passes `LIVE_SELECT_ENABLED` to the
    decode pick, and nothing else changes with it off."""
    seen = []

    class _Stop(Exception):
        pass

    def spy(*args, **kwargs):
        seen.append(kwargs.get("live_select"))
        raise _Stop

    monkeypatch.setattr(msa_dcp, "local_candidates", spy)
    monkeypatch.setattr(msa_dcp.msa, "write_kv_and_index", lambda kv, idx, *a, **k: (kv, idx))
    monkeypatch.setattr(
        msa_dcp.msa, "_block_scores", lambda *a, **k: jnp.zeros((16, 1, 128), jnp.float32)
    )
    tokens, max_seqs = 16, 4
    args = (
        jnp.zeros((tokens, 1, 128), jnp.bfloat16),
        jnp.zeros((tokens, 1, 128), jnp.bfloat16),
        jnp.zeros((tokens, 1, 128), jnp.bfloat16),
        jnp.zeros((tokens, 128), jnp.bfloat16),
        jnp.zeros((8, PAGE // 2, 2, 256), jnp.float8_e4m3fn),
        jnp.zeros((8, PAGE // 2, 2, 128), jnp.float8_e4m3fn),
        jnp.full((max_seqs,), 1000, jnp.int32),
        jnp.zeros((max_seqs * 8,), jnp.int32),
        jnp.asarray([0, 4, 8, 12, 16], jnp.int32),
        jnp.asarray([0, 0, 4], jnp.int32),
    )
    call = functools.partial(
        msa_dcp.msa_write_and_candidates_dcp,
        topk=TOPK, init_blocks=0, local_blocks=1, rank=1, dcp=2, interpret=True,
    )
    for flag in (False, True):
        monkeypatch.setattr(msa_dcp, "LIVE_SELECT_ENABLED", flag)
        with pytest.raises(_Stop):
            call(*args)
    assert seen == [False, True]


@pytest.mark.parametrize(
    "rows,cols,dcp,pretransposed",
    [(1024, 512, 4, False), (32, 1024, 2, False), (512, 512, 4, True)],
    ids=["c64-dcp4", "c4-dcp2", "pretransposed"],
)
def test_live_select_lowers_for_tpu(rows, cols, dcp, pretransposed):
    """The live pick lowers to Mosaic without a TPU at the decode shapes."""
    from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

    shape = (cols, rows) if pretransposed else (rows, 1, cols)
    fn = functools.partial(
        msa_dcp.local_candidates,
        block_size=PAGE,
        topk=TOPK,
        init_blocks=0,
        local_blocks=1,
        dcp=dcp,
        live_select=True,
    )
    tpu7x = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(tpu7x):
        lowered = (
            jax.jit(lambda s, p, r: fn(s, p, rank=r))
            .trace(
                jax.ShapeDtypeStruct(shape, jnp.float32),
                jax.ShapeDtypeStruct((rows,), jnp.int32),
                jax.ShapeDtypeStruct((1,), jnp.int32),
            )
            .lower(lowering_platforms=("tpu",))
        )
    text = lowered.as_text()
    assert "minimax_m3_msa_dcp_live_candidates" in text
