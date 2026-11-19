# SPDX-License-Identifier: Apache-2.0
"""q38 L1-G1 GDN-COMMIT1: the commit kernel writes, for each request, the accepted checkpoint's bytes exactly as today's
scatter writes them, and nothing else (CPU interpret).

`commit_gdn_ckpts_fast(pool, conv, rec, wb, accept)` vs the flag-off `scatter_gdn_ckpts_fast`:
  (a) the whole pool equals today's scatter run with every non-accepted checkpoint id nulled (`accepted_write_blocks`),
      byte for byte (the reference is the unchanged scatter, so the bytes are the served bytes);
  (b) every accepted live block equals, byte for byte over all 512 rows, that block after today's full scatter;
  (c) every other manager block (non-accepted checkpoint blocks, unrelated blocks; not the null block 0, which the
      scatter never writes either) keeps its input bytes; the changed blocks are exactly the accepted live ids, and
      their count is > 0 wherever a live id exists (positive control of the zero-count cases);
across accept counts 0..3 (all four values per case, plus exhaustive per-request patterns), pad requests (null ids),
null non-accepted ids, kernel blocks of 128 and 512 rows, ring / prefetch settings incl. the auto deep ring, and through
the runner form `commit_from_offsets` (accept = read_offsets[state ids]).
Mutants that must be caught: committing accept + 1 / accept - 1, swapping two requests' accepts, and a source index
off by one checkpoint (each fails (a)).
Run: JAX_PLATFORMS=cpu PYTHONPATH=src pytest tests/q38_l1g1/test_g1_commit_kernel.py
"""

from __future__ import annotations

import itertools
import sys
import types

try:
    import vllm.envs  # noqa: F401
except ImportError:
    _vllm = types.ModuleType("vllm")
    _envs = types.ModuleType("vllm.envs")
    _envs.VLLM_KV_CACHE_LAYOUT = "NHD"
    _vllm.envs = _envs
    sys.modules.setdefault("vllm", _vllm)
    sys.modules.setdefault("vllm.envs", _envs)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from vllm_torchtpu.megakernel_runtime import qwen38_gdn_commit as gc  # noqa: E402
from vllm_torchtpu.megakernel_runtime import qwen38_gdn_state as gs  # noqa: E402
from vllm_torchtpu.megakernel_runtime.qwen38_scatter import scatter_gdn_ckpts_fast  # noqa: E402

T = 4


def _setup(NL: int, NS: int, kb: int, seed: int, pads: tuple = (), null_others: bool = False):
    """Random pool bytes, checkpoint outputs and distinct write ids [NL, NS, T] (pad requests -> 0; optionally a random
    non-accepted id nulled per request)."""
    rng = np.random.default_rng(seed)
    nmb = 1 + NL * NS * T + 3  # 3 blocks no id names
    shape = (nmb * (512 // kb), kb, 1, 4, 256)
    raw = rng.integers(0, 256, size=shape, dtype=np.uint8)
    pool = jax.lax.bitcast_convert_type(jnp.asarray(raw), jnp.float8_e4m3fn)
    conv = jnp.asarray(rng.standard_normal((NL, NS, 8, 1280)), jnp.float32)
    rec = jnp.asarray(rng.standard_normal((NL, NS * T * 6, 128, 128)), jnp.float32).astype(jnp.bfloat16)
    wb = (1 + rng.permutation(NL * NS * T)).astype(np.int32).reshape(NL, NS, T)
    for s in pads:
        wb[:, s] = 0
    layout = gs.Qwen38GdnPoolLayout(pool_shape=shape, pool_dtype="float8_e4m3fn")
    return raw, pool, conv, rec, wb, layout, rng, null_others


def _u8(x) -> np.ndarray:
    return np.asarray(jax.lax.bitcast_convert_type(x, jnp.uint8))


def _mgr(a: np.ndarray, kb: int) -> np.ndarray:
    """[blocks * split, kb, ...] bytes -> [manager blocks, 512 rows * row bytes]."""
    return a.reshape(a.shape[0] // (512 // kb), -1)


def _check(raw, pool, conv, rec, wb, acc, layout, kb, *, ring=None, prefetch=None, commit=None, tag=""):
    commit = commit or gc.commit_gdn_ckpts_fast
    got = _u8(commit(pool, conv, rec, jnp.asarray(wb), jnp.asarray(acc), layout=layout, interpret=True, ring=ring,
                     prefetch=prefetch))
    ref = _u8(scatter_gdn_ckpts_fast(pool, conv, rec, gc.accepted_write_blocks(jnp.asarray(wb), jnp.asarray(acc)),
                                     layout=layout, interpret=True))
    full = _u8(scatter_gdn_ckpts_fast(pool, conv, rec, jnp.asarray(wb), layout=layout, interpret=True))
    g, r, f, x = (_mgr(a, kb) for a in (got, ref, full, raw))
    NL, NS, _ = wb.shape
    live = {int(wb[ln, s, acc[ln, s]]) for ln in range(NL) for s in range(NS)} - {0}
    nbad = int((g != r).sum())
    changed = {int(m) for m in np.nonzero((g != x).any(-1))[0]}
    print(f"Q38G1K {tag}: accepted live ids {len(live)}; blocks changed {len(changed)}; bytes differing from the "
          f"masked-id scatter {nbad}", flush=True)
    # (a) == today's scatter with only the accepted ids
    assert nbad == 0, f"{tag}: commit differs from the accepted-id scatter in {nbad} bytes"
    # (b) accepted live blocks == the full scatter's bytes there
    for m in sorted(live):
        np.testing.assert_array_equal(g[m], f[m], err_msg=f"{tag}: accepted block {m} != full scatter")
    # (c) nothing else moved; changed set == accepted live ids (the null block 0 never written by either)
    assert changed == live, f"{tag}: changed {sorted(changed ^ live)[:8]} vs accepted live ids"
    assert not (g[0] != x[0]).any(), f"{tag}: the null block was written"
    return len(live)


ACCEPTS = {
    "all0": lambda NL, NS, rng: np.zeros((NL, NS), np.int32),
    "all3": lambda NL, NS, rng: np.full((NL, NS), 3, np.int32),
    "mixed": lambda NL, NS, rng: (np.arange(NL * NS).reshape(NL, NS) % T).astype(np.int32),
    "random": lambda NL, NS, rng: rng.integers(0, T, (NL, NS)).astype(np.int32),
}
# (NL, NS, kb, accept, pads, ring, prefetch)
CASES = {
    "k128_mixed": (2, 3, 128, "mixed", (), None, None),
    "k128_all0": (2, 3, 128, "all0", (), None, None),
    "k128_all3_pad": (2, 3, 128, "all3", (2,), None, None),
    "k512_random_pad": (3, 4, 512, "random", (0, 3), None, None),
    "k512_mixed": (2, 3, 512, "mixed", (), None, None),
    "k128_random_r2": (2, 5, 128, "random", (1,), 2, 1),
    "k128_mixed_r8p3": (3, 4, 128, "mixed", (), 8, 3),
    "k128_random_r8p7": (2, 6, 128, "random", (5,), 8, 7),
    "k128_mixed_r12p6": (4, 4, 128, "mixed", (0,), 12, 6),
    "k512_random_r4p3": (2, 5, 512, "random", (), 4, 3),
}


@pytest.mark.parametrize("case", list(CASES))
def test_commit_matches_the_scatter_at_the_accepted_blocks(case):
    NL, NS, kb, acc_kind, pads, ring, prefetch = CASES[case]
    raw, pool, conv, rec, wb, layout, rng, _ = _setup(NL, NS, kb, seed=sum(map(ord, case)), pads=pads)
    acc = ACCEPTS[acc_kind](NL, NS, rng)
    n = _check(raw, pool, conv, rec, wb, acc, layout, kb, ring=ring, prefetch=prefetch, tag=case)
    assert n == NL * (NS - len(pads)) > 0
    if acc_kind in ("mixed", "random") and NL * NS >= 8:
        assert len(set(acc.reshape(-1).tolist())) >= 3, "accept values not varied"


def test_every_accept_pattern_of_two_requests():
    """Exhaustive: both requests of 1 layer take every (a0, a1) in 0..3 x 0..3 (16 patterns), one kernel each."""
    raw, pool, conv, rec, wb, layout, rng, _ = _setup(1, 2, 128, seed=21)
    for a0, a1 in itertools.product(range(T), range(T)):
        acc = np.array([[a0, a1]], np.int32)
        assert _check(raw, pool, conv, rec, wb, acc, layout, 128, tag=f"exh {a0}{a1}") == 2


def test_null_non_accepted_ids_and_all_pads():
    """Null ids among the non-accepted checkpoints change nothing; an all-pad step writes nothing (zero count; its
    positive control is every other case)."""
    raw, pool, conv, rec, wb, layout, rng, _ = _setup(2, 4, 128, seed=33)
    acc = ACCEPTS["random"](2, 4, rng)
    for ln in range(2):
        for s in range(4):
            t = (acc[ln, s] + 1 + rng.integers(0, T - 1)) % T
            wb[ln, s, t] = 0
    assert _check(raw, pool, conv, rec, wb, acc, layout, 128, tag="null others") == 8
    wb0 = np.zeros_like(wb)
    assert _check(raw, pool, conv, rec, wb0, acc, layout, 128, tag="all pads") == 0
    got = _u8(gc.commit_gdn_ckpts_fast(pool, conv, rec, jnp.asarray(wb0), jnp.asarray(acc), layout=layout,
                                       interpret=True))
    assert not (got != raw).any(), "an all-null step moved bytes"


def test_auto_ring_follows_the_item_count(monkeypatch):
    """Unset ring / prefetch: the deep ring from AUTO_MIN_CKPTS items (forced low), bytes == the explicit settings."""
    raw, pool, conv, rec, wb, layout, rng, _ = _setup(3, 4, 128, seed=44, pads=(1,))
    acc = ACCEPTS["random"](3, 4, rng)
    monkeypatch.setattr(gc, "AUTO_MIN_CKPTS", 1)
    a = _u8(gc.commit_gdn_ckpts_fast(pool, conv, rec, jnp.asarray(wb), jnp.asarray(acc), layout=layout, interpret=True))
    b = _u8(gc.commit_gdn_ckpts_fast(pool, conv, rec, jnp.asarray(wb), jnp.asarray(acc), layout=layout, interpret=True,
                                     ring=4, prefetch=1))
    assert np.array_equal(a, b)
    _check(raw, pool, conv, rec, wb, acc, layout, 128, tag="auto ring")


def test_runner_form_reads_the_post_verify_offsets():
    """`commit_from_offsets`: accept = read_offsets[state ids] (the buffer indexed by state block id, as gdn_blocks reads
    it); same bytes as the direct form; a stale offset table (the pre-verify one) writes other blocks."""
    raw, pool, conv, rec, wb, layout, rng, _ = _setup(2, 4, 128, seed=55, pads=(3,))
    acc = ACCEPTS["random"](2, 4, rng)
    st = (100 + np.arange(8, dtype=np.int32)).reshape(2, 4)  # state block ids
    offs = rng.integers(0, T, 200).astype(np.int32)
    offs[st] = acc
    a = _u8(gc.commit_from_offsets(pool, conv, rec, jnp.asarray(wb), jnp.asarray(st), jnp.asarray(offs),
                                   layout=layout, interpret=True))
    b = _u8(gc.commit_gdn_ckpts_fast(pool, conv, rec, jnp.asarray(wb), jnp.asarray(acc), layout=layout, interpret=True))
    assert np.array_equal(a, b)
    stale = offs.copy()
    stale[st] = (acc + 1) % T
    c = _u8(gc.commit_from_offsets(pool, conv, rec, jnp.asarray(wb), jnp.asarray(st), jnp.asarray(stale),
                                   layout=layout, interpret=True))
    assert not np.array_equal(a, c), "a stale read-offset table must commit other blocks"


def test_rejects_bad_shapes():
    raw, pool, conv, rec, wb, layout, rng, _ = _setup(2, 3, 128, seed=66)
    acc = np.zeros((2, 3), np.int32)
    with pytest.raises(ValueError, match="rec_ckpt"):
        gc.commit_gdn_ckpts_fast(pool, conv, rec[:, :-6], jnp.asarray(wb), jnp.asarray(acc), layout=layout,
                                 interpret=True)
    with pytest.raises(ValueError, match="accept"):
        gc.commit_gdn_ckpts_fast(pool, conv, rec, jnp.asarray(wb), jnp.asarray(acc[:, :2]), layout=layout,
                                 interpret=True)


# ------------------------------------------------------------------------------------------------ mutants


def _mutant_compare(kind: str):
    raw, pool, conv, rec, wb, layout, rng, _ = _setup(2, 4, 128, seed=77)
    acc = ACCEPTS["mixed"](2, 4, rng)
    if kind == "plus1":
        commit = lambda p, c, r, w, a, **k: gc.commit_gdn_ckpts_fast(p, c, r, w, (a + 1) % T, **k)  # noqa: E731
    elif kind == "minus1":
        commit = lambda p, c, r, w, a, **k: gc.commit_gdn_ckpts_fast(p, c, r, w, (a + T - 1) % T, **k)  # noqa: E731
    elif kind == "swap":
        commit = lambda p, c, r, w, a, **k: gc.commit_gdn_ckpts_fast(p, c, r, w, a[:, ::-1], **k)  # noqa: E731
    else:  # source checkpoint off by one (rows of the neighbouring token), destination unchanged

        def commit(p, c, r, w, a, **k):
            orig = gc.commit_plan

            def plan(wbx, ax):
                src, dst, accx = orig(wbx, ax)
                return jnp.where(accx < T - 1, src + 1, src - 1), dst, accx

            gc.commit_plan = plan
            try:
                return gc.commit_gdn_ckpts_fast(p, c, r, w, a, **k)
            finally:
                gc.commit_plan = orig

    _check(raw, pool, conv, rec, wb, acc, layout, 128, commit=commit, tag=f"mutant {kind}")


@pytest.mark.parametrize("kind", ["plus1", "minus1", "swap", "src_off_by_one"])
def test_mutants_are_caught(kind):
    with pytest.raises(AssertionError):
        _mutant_compare(kind)
