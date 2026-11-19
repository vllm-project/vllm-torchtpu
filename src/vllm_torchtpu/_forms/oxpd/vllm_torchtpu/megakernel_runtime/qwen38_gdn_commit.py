# SPDX-License-Identifier: Apache-2.0
"""q38 L1-G1 GDN-COMMIT1 (VLLM_TPU_MK_GDN_COMMIT1=1): persist one accepted GDN checkpoint per request per layer.

Today (flag off) the whole-step call scatters all T = 4 verify checkpoints of every request into the pool
(`qwen38_scatter.scatter_gdn_ckpts_fast`): it loads every checkpoint (pad rows included), widens the bf16 state to f32
and writes 400 pool rows per live checkpoint. The next step reads exactly one of them, checkpoint
`read_offsets[state block] = num_accepted - 1` (`runner/kv_cache_manager.py` read-offset buffer; `gdn_blocks`:
`rb = ck[off[st]]`), and the seed copies at block crossings move only that one too (`tpu_runner._spec_seed_rows_impl`).

With the flag:
  1. the whole-step call (and the head call) skips the scatter and returns the kernel's `conv_ckpt` / `rec_ckpt` and the step's write ids
     `wb [NL, NS, T]` and state ids `st [NL, NS]` as four extra tp-sharded outputs (`call_outputs` / `out_specs`);
  2. the torch op (whole step / head kind) keeps them on its kind (`stash`; the op's declared outputs are unchanged, so the compiled torch program
     is the flag-off program);
  3. after the step's post-verify read-offset update, once the sampled-token D2H has been issued, the runner calls
     `commit_pending`: per rank, `commit_gdn_ckpts_fast` writes checkpoint `a = read_offsets[st[l, s]]` of request `s`
     of GDN layer `l` into block `wb[l, s, a]`, with the scatter's exact bytes, and loads nothing else;
  4. staging that was not committed (a forward with no sample_tokens after it: warmup, dummy and multi-chunk steps) is
     written with the full scatter (`flush_pending`, the flag-off bytes) before any later forward or seed copy.

The pool keeps today's meaning: the block at the read offset holds the accepted state; the other checkpoint blocks of the
step keep older bytes (no reader names them: the poison tests in tests/q38_l1g1 fill them with NaN encodings).
Scope: the whole-step call (model layers 3..46, 33 GDN layers) and the head group calls (layers 0..2, 3 GDN layers;
`qwen38_segment_call` with head=True, driver `_SegmentKind`), in the whole step and in the segment program. The segment
program's body segments keep today's scatter.
Refused (ValueError at build): VLLM_TPU_MK_L47_FOLD (its wrapper owns the call's output list) and
VLLM_TPU_MK_GDN_CKPT_INK (the in-kernel checkpoint write is the other answer to the same bytes).
Paths below are relative to src/vllm_torchtpu/.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.megakernel.models.qwen38_gdn_pool import CONV_ROWS, SSM_BLOCKS, SSM_ROWS, pool_rows

from .qwen38_scatter import AUTO_MIN_CKPTS, AUTO_PREFETCH, AUTO_RING, RING, _store_typed, scatter_gdn_ckpts_fast

ENV = "VLLM_TPU_MK_GDN_COMMIT1"
ENGAGED = "q38 L1-G1 GDN-COMMIT1"
N_OUTS = 4  # extra outputs of the whole-step call with the flag: conv_ckpt, rec_ckpt, wb, st
INTERPRET = False  # the torch ops' Pallas interpret mode (CPU tests only)
HEAD_CHUNK_ROWS = 64  # the driver's KERNEL_ROWS (its head kind runs larger buckets in chunks of these rows; tested equal)
_log = logging.getLogger("vllm." + __name__)
_LOGGED: set = set()


def _once(key: str, msg: str, *args: Any) -> None:
    if key not in _LOGGED:
        _LOGGED.add(key)
        _log.info(msg, *args)


def commit1_on() -> bool:
    """VLLM_TPU_MK_GDN_COMMIT1 (registered in envs and hashed: it changes the traced whole-step program)."""
    from vllm_torchtpu import envs

    return bool(envs.VLLM_TPU_MK_GDN_COMMIT1)


def refusal() -> str | None:
    """Why the flag cannot be on with the current environment (None = composable)."""
    from vllm_torchtpu import envs

    if bool(getattr(envs, "VLLM_TPU_MK_L47_FOLD", False)):
        return "VLLM_TPU_MK_L47_FOLD (its call wrapper asserts the flag-off output list)"
    if bool(getattr(envs, "VLLM_TPU_MK_GDN_CKPT_INK", False)):
        return "VLLM_TPU_MK_GDN_CKPT_INK (the in-kernel checkpoint write)"
    return None


# --------------------------------------------------------------------------------------------- the commit kernel


def accepted_write_blocks(write_blocks: Any, accept: Any) -> Any:
    """`write_blocks [NL, NS, T]` with every checkpoint but `accept [NL, NS]` nulled (the reference: the flag-off scatter
    with these ids writes exactly the bytes `commit_gdn_ckpts_fast` writes)."""
    T = write_blocks.shape[-1]
    t = jnp.arange(T, dtype=jnp.int32)[None, None, :]
    a = jnp.clip(jnp.asarray(accept, jnp.int32), 0, T - 1)[..., None]
    return jnp.where(t == a, write_blocks, 0).astype(jnp.int32)


def commit_plan(write_blocks: Any, accept: Any) -> tuple[Any, Any, Any]:
    """Per item `i = l * NS + s`: (source checkpoint `i * T + a`, its write block, `a`); `a` clipped to [0, T)."""
    NL, NS, T = write_blocks.shape
    n = NL * NS
    acc = jnp.clip(jnp.asarray(accept, jnp.int32).reshape(n), 0, T - 1)
    src = jnp.arange(n, dtype=jnp.int32) * T + acc
    dst = jnp.take(write_blocks.reshape(-1).astype(jnp.int32), src)
    return src, dst, acc


def commit_gdn_ckpts_fast(
    pool: Any,
    conv_ckpt: Any,
    rec_ckpt: Any,
    write_blocks: Any,
    accept: Any,
    *,
    layout: Any,
    interpret: bool = False,
    ring: int | None = None,
    prefetch: int | None = None,
) -> Any:
    """Write checkpoint `accept[l, s]` of every request into its write block; returns the pool (aliased in place).

    pool: the kernel-granularity unified pool (`layout.pool_shape`).
    conv_ckpt: `[NL, NS, HR, QKV]` f32 library history (rows 0..2 old, 3..3+T-1 new inputs).
    rec_ckpt: `[NL, NS*T*vh, KD, VD]` (`REC`), row block `s*T + t`.
    write_blocks: `[NL, NS, T]` int32, checkpoint `t` of request `s` -> manager block (0 = null, never written).
    accept: `[NL, NS]` int32, the checkpoint to keep (clipped to [0, T)).

    The written bytes are `scatter_gdn_ckpts_fast`'s for that checkpoint (the same widen, typed stores and DMAs: the 384
    f32 state rows and the 16 conv rows, the conv window = history rows a+1..a+3, zero-padded), and no other block
    changes. Loads: one rec checkpoint and the request's HR conv rows per non-null item; null items load nothing.
    `ring` / `prefetch` as the scatter's (auto on the item count).
    """
    NL, NS, T = write_blocks.shape
    n = NL * NS
    HR, QKV = int(conv_ckpt.shape[2]), int(conv_ckpt.shape[3])
    vh, KD, VD = layout.n_v, layout.d_k, layout.d_v
    if conv_ckpt.shape[:2] != (NL, NS) or HR < 3 + T or QKV != layout.conv_dim:
        raise ValueError(f"conv_ckpt {conv_ckpt.shape} vs write_blocks {(NL, NS, T)} / conv_dim {layout.conv_dim}")
    if rec_ckpt.shape != (NL, NS * T * vh, KD, VD):
        raise ValueError(f"rec_ckpt {rec_ckpt.shape} vs write_blocks {(NL, NS, T)}")
    if accept.shape != (NL, NS):
        raise ValueError(f"accept {accept.shape} vs write_blocks {(NL, NS, T)}")
    if ring is None and prefetch is None:
        ring, prefetch = (AUTO_RING, AUTO_PREFETCH) if n >= AUTO_MIN_CKPTS else (RING, 1)
    RING_ = RING if ring is None else ring
    P = 1 if prefetch is None else prefetch
    assert 1 <= P < RING_, (prefetch, ring)
    src, dst, acc = commit_plan(write_blocks, accept)
    rec = rec_ckpt.reshape(n * T, vh, KD, VD)
    conv = conv_ckpt.astype(jnp.float32).reshape(n, HR, QKV)
    per = QKV // 256
    f8 = pool.dtype

    def kernel(src_ref, dst_ref, acc_ref, rec_ref, conv_ref, pool_in, pool_out, vin, cin, vout, cout, sin, sout):
        del pool_in

        def out_copies(slot, m):
            cps = [
                pltpu.make_async_copy(
                    vout.at[slot, j], pool_rows(pool_out, m, j * SSM_ROWS, SSM_ROWS), sout.at[slot, j]
                )
                for j in range(SSM_BLOCKS)
            ]
            cps.append(
                pltpu.make_async_copy(
                    cout.at[slot], pool_rows(pool_out, m, SSM_BLOCKS * SSM_ROWS, CONV_ROWS), sout.at[slot, SSM_BLOCKS]
                )
            )
            return cps

        def in_copies(i, slot):
            return (
                pltpu.make_async_copy(rec_ref.at[src_ref[i]], vin.at[slot], sin.at[slot, 0]),
                pltpu.make_async_copy(conv_ref.at[i], cin.at[slot], sin.at[slot, 1]),
            )

        for k in range(min(P, n)):

            @pl.when(dst_ref[k] != 0)
            def _(k=k):
                for c in in_copies(k, k):
                    c.start()

        def body(i, carry):
            slot = lax.rem(i, RING_)
            ahead = i + P
            nslot = lax.rem(ahead, RING_)
            # Slot `nslot` is reused by item i + P: its write from i + P - RING (if any) must have landed before the
            # item's conversion overwrites the slot's staging (waited here, as the scatter does).
            prev = dst_ref[jnp.clip(ahead - RING_, 0, n - 1)]

            @pl.when((ahead >= RING_) & (ahead < n + RING_) & (prev != 0))
            def _():
                for c in out_copies(nslot, prev):
                    c.wait()

            nxt = jnp.minimum(ahead, n - 1)

            @pl.when((ahead < n) & (dst_ref[nxt] != 0))
            def _():
                for c in in_copies(nxt, nslot):
                    c.start()

            m = dst_ref[i]

            @pl.when(m != 0)
            def _():
                for c in in_copies(i, slot):
                    c.wait()
                vals = vin[slot].astype(jnp.float32).reshape(vh * KD, VD)
                rows = vals.shape[0] // SSM_BLOCKS
                for j in range(SSM_BLOCKS):
                    _store_typed(vout.at[slot, j], vals[j * rows : (j + 1) * rows], interpret)
                cv = cin[slot]  # [HR, QKV]: the request's whole history; window of checkpoint t = rows t+1 .. t+3
                a = acc_ref[i]
                for t in range(T):

                    @pl.when(a == t)
                    def _(t=t):
                        crow = jnp.concatenate(
                            [cv[t + 1 + r : t + 2 + r, c * 256 : (c + 1) * 256] for r in range(3) for c in range(per)]
                            + [jnp.zeros((CONV_ROWS - 3 * per, 256), jnp.float32)],
                            axis=0,
                        )
                        _store_typed(cout.at[slot], crow, interpret)

                for c in out_copies(slot, m):
                    c.start()

            return carry

        lax.fori_loop(0, n, body, 0)
        # Writes not yet waited on: the last RING - P items (the loop waits for i + P - RING when it reuses a slot).
        for k in range(max(0, n - RING_ + P), n):

            @pl.when(dst_ref[k] != 0)
            def _(k=k):
                for c in out_copies(k % RING_, dst_ref[k]):
                    c.wait()

    any_ = pltpu.HBM
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(),
            in_specs=[
                pl.BlockSpec(memory_space=any_),
                pl.BlockSpec(memory_space=any_),
                pl.BlockSpec(memory_space=any_),
            ],
            out_specs=pl.BlockSpec(memory_space=any_),
            scratch_shapes=[
                pltpu.VMEM((RING_, vh, KD, VD), rec.dtype),
                pltpu.VMEM((RING_, HR, QKV), jnp.float32),
                pltpu.VMEM((RING_, SSM_BLOCKS, SSM_ROWS) + tuple(pool.shape[2:]), f8),
                pltpu.VMEM((RING_, CONV_ROWS) + tuple(pool.shape[2:]), f8),
                pltpu.SemaphoreType.DMA((RING_, 2)),
                pltpu.SemaphoreType.DMA((RING_, SSM_BLOCKS + 1)),
            ],
        ),
        out_shape=jax.ShapeDtypeStruct(pool.shape, pool.dtype),
        input_output_aliases={5: 0},
        interpret=interpret,
    )(src, dst, acc, rec, conv, pool)


def commit_from_offsets(pool: Any, conv_ckpt: Any, rec_ckpt: Any, write_blocks: Any, state_ids: Any,
                        read_offsets: Any, *, layout: Any, interpret: bool = False) -> Any:
    """The runner's commit (per rank): `accept = read_offsets[state_ids]` (the post-verify read offsets, indexed by state
    block id as `gdn_blocks` reads them: `rb = ck[off[st]]`), then `commit_gdn_ckpts_fast`."""
    acc = jnp.take(jnp.asarray(read_offsets, jnp.int32), jnp.asarray(state_ids, jnp.int32), mode="clip")
    return commit_gdn_ckpts_fast(pool, conv_ckpt, rec_ckpt, write_blocks, acc, layout=layout, interpret=interpret)


def full_scatter(pool: Any, conv_ckpt: Any, rec_ckpt: Any, write_blocks: Any, *, layout: Any,
                 interpret: bool = False) -> Any:
    """The flag-off write of staged checkpoints (every checkpoint; `scatter_gdn_ckpts_fast` itself, same arguments)."""
    return scatter_gdn_ckpts_fast(pool, conv_ckpt, rec_ckpt, write_blocks, layout=layout, interpret=interpret)


# ------------------------------------------------------------------------------- the whole-step call (JAX side)


def scatter_fn(scatter: Any, write: bool) -> Any:
    """`per_rank_writes`' checkpoint scatter: `scatter` itself unless the flag is on for a writing call; on, a pass-through
    (the checkpoints leave the call as outputs and the runner commits the accepted one)."""
    if not (write and commit1_on()):
        return scatter

    def staged(pool: Any, conv_ck: Any, rec_ck: Any, wb: Any, **_: Any) -> Any:
        return pool

    return staged


def state_ids(sts: Any, NS: int) -> Any:
    """`[NL, NS]` int32 state ids of the step (each layer's ids fitted to NS rows like `gdn_blocks`)."""
    from .qwen38_segment_call import _fit

    return jnp.stack([_fit(jnp.asarray(s), NS) for s in sts]).astype(jnp.int32)


def call_outputs(ret: tuple, write: bool, conv_ck: Any, rec_ck: Any, wb: Any, sts: Any, NS: int, R: int) -> tuple:
    """The whole-step call's return: `ret` unchanged unless the flag is on for a writing call; on, `ret` + (conv_ckpt,
    rec_ckpt, wb, st). Every output is tp-sharded, as the call's others (torch sees each rank's shard): the ids are tiled
    R times, so each rank's shard is the step's whole `wb [NL, NS, T]` / `st [NL, NS]` (L1-L47F's sink export does the
    same)."""
    if not (write and commit1_on()):
        return ret
    st = state_ids(sts, NS)
    return tuple(ret) + (conv_ck, rec_ck, jnp.tile(wb.astype(jnp.int32), (R, 1, 1)), jnp.tile(st, (R, 1)))


def out_specs(specs: tuple, write: bool, tp: Any, rep: Any) -> tuple:
    """The whole-step call's output specs: `specs` unless the flag is on for a writing call; on, + (tp, tp, tp, tp) (`rep`
    unused: the ids leave tiled per rank). Raises ValueError for a refused composition."""
    if not (write and commit1_on()):
        return specs
    why = refusal()
    if why is not None:
        raise ValueError(f"{ENV}=1 is not composed with {why}; unset one of them ({ENGAGED})")
    _once("call", "%s whole step: the call skips scatter_gdn_ckpts_fast and returns conv_ckpt / rec_ckpt / write ids / "
          "state ids; the runner commits the accepted checkpoint after the read-offset update", ENGAGED)
    return tuple(specs) + (tp, tp, tp, tp)


# --------------------------------------------------------------------------------------------- torch side


@dataclasses.dataclass
class Staged:
    """One whole-step call's checkpoints (this rank's tensors), waiting for the commit."""

    conv: Any
    rec: Any
    wb: Any
    st: Any
    rows: int


def _writes(kind: Any) -> bool:
    """The kind's `write` (the whole step's `kw["write"]`, a segment kind's `write`): shadows never stage."""
    kw = getattr(kind, "kw", None)
    return bool(kw.get("write", True)) if isinstance(kw, dict) else bool(getattr(kind, "write", True))


def stash(kind: Any, outs: list, rows: int) -> list:
    """A whole-step or head torch op's result: with the flag on for a writing kind, pop the 4 staging outputs into
    `kind._g1_pending` and return the rest (the op's declared outputs); otherwise `outs` unchanged."""
    if not (_writes(kind) and commit1_on()):
        return outs
    conv, rec, wb, st = outs[-N_OUTS:]
    pend = kind.__dict__.setdefault("_g1_pending", [])
    pend.append(Staged(conv=conv, rec=rec, wb=wb, st=st, rows=rows))
    Counters.stashed += 1
    return outs[:-N_OUTS]


def _targets(runner: Any) -> list:
    """(kind, pool) of every kind that can hold staging: the whole step's and the head groups' (shared segments)."""
    drv = getattr(runner, "_megakernel", None)
    if drv is None:
        return []
    out = []
    whole = getattr(drv, "whole_module", None)
    if whole is not None and getattr(whole, "kind", None) is not None:
        out.append((whole.kind, whole.pool))
    segs = getattr(drv, "segments", None)
    for key, kind in (getattr(segs, "kinds", None) or {}).items():
        if key[1] and _writes(kind):  # (L, head)
            out.append((kind, segs.pool))
    return out


def pending(runner: Any) -> list:
    return [s for kind, _ in _targets(runner) for s in kind.__dict__.get("_g1_pending", [])]


@functools.lru_cache(maxsize=None)
def _ops(layout: Any) -> Any:
    """Torch callables for `layout`: `commit1` (one call's staging), `commit2` (two calls' staging, e.g. the head group and
    the whole step of one step, in ONE program: one dispatch per step) and `full` (one call in full). Each is a per-rank
    jax op donating the pool inside a compiled function that copies the result back into the pool (the seed-copy pattern,
    layers/adapter/custom_ops/mamba_state_copy_op.py)."""
    import types

    import torch
    from torch_tpu._internal import pallas

    interp = bool(INTERPRET)

    # torch_tpu's jax_op requires every argument and the return annotated (jax.Array; strings resolve in this module's
    # globals): unannotated it raises 'Missing argument type annotation for JAX function' at op construction.
    def commit1_fn(pool: jax.Array, offsets: jax.Array, conv: jax.Array, rec: jax.Array, wb: jax.Array,
                   st: jax.Array) -> tuple[jax.Array, jax.Array]:
        return commit_from_offsets(pool, conv, rec, wb, st, offsets, layout=layout, interpret=interp), wb.reshape(-1)[:1]

    def commit2_fn(pool: jax.Array, offsets: jax.Array, conv_a: jax.Array, rec_a: jax.Array, wb_a: jax.Array,
                   st_a: jax.Array, conv_b: jax.Array, rec_b: jax.Array, wb_b: jax.Array,
                   st_b: jax.Array) -> tuple[jax.Array, jax.Array]:
        pool = commit_from_offsets(pool, conv_a, rec_a, wb_a, st_a, offsets, layout=layout, interpret=interp)
        pool = commit_from_offsets(pool, conv_b, rec_b, wb_b, st_b, offsets, layout=layout, interpret=interp)
        return pool, wb_b.reshape(-1)[:1]

    def full_fn(pool: jax.Array, conv: jax.Array, rec: jax.Array, wb: jax.Array) -> tuple[jax.Array, jax.Array]:
        return full_scatter(pool, conv, rec, wb, layout=layout, interpret=interp), wb.reshape(-1)[:1]

    k = _ops.cache_info().currsize  # one layout per process in practice; a second one gets its own op names
    sfx = "" if k == 0 else f"_{k}"
    c1 = pallas.jax_op(f"pallas::q38_g1_gdn_commit1{sfx}", commit1_fn, donate_argnums=(0,))
    c2 = pallas.jax_op(f"pallas::q38_g1_gdn_commit2{sfx}", commit2_fn, donate_argnums=(0,))
    fo = pallas.jax_op(f"pallas::q38_g1_gdn_full{sfx}", full_fn, donate_argnums=(0,))
    c1.register_fake(lambda pool, offsets, conv, rec, wb, st: (torch.empty_like(pool), wb.new_empty((1,))))
    c2.register_fake(lambda pool, offsets, ca, ra, wa, sa, cb, rb, wbb, sb: (torch.empty_like(pool), wbb.new_empty((1,))))
    fo.register_fake(lambda pool, conv, rec, wb: (torch.empty_like(pool), wb.new_empty((1,))))

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def commit1(pool, offsets, conv, rec, wb, st):
        new_pool, marker = c1(pool, offsets, conv, rec, wb, st)
        pool.copy_(new_pool)
        return marker

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def commit2(pool, offsets, conv_a, rec_a, wb_a, st_a, conv_b, rec_b, wb_b, st_b):
        new_pool, marker = c2(pool, offsets, conv_a, rec_a, wb_a, st_a, conv_b, rec_b, wb_b, st_b)
        pool.copy_(new_pool)
        return marker

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def full(pool, conv, rec, wb):
        new_pool, marker = fo(pool, conv, rec, wb)
        pool.copy_(new_pool)
        return marker

    return types.SimpleNamespace(commit1=commit1, commit2=commit2, full=full, ops=(c1, c2, fo))


class Counters:
    """Process-wide counts (engagement lines; a STEP reader checks commits + flushes == stashed calls)."""

    stashed = 0
    committed = 0
    flushed = 0


def _records(runner: Any) -> list:
    """Every pending staging as (kind, pool, Staged), the whole step's first, then the head groups' (`_targets` order)."""
    return [(kind, pool, s) for kind, pool in _targets(runner) for s in kind.__dict__.get("_g1_pending", [])]


def _clear(runner: Any) -> None:
    for kind, _ in _targets(runner):
        kind.__dict__.get("_g1_pending", []).clear()


def commit_pending(runner: Any) -> int:
    """sample_tokens, after the read offsets of every chunk are applied and the sampled-token D2H is issued: commit every
    pending whole-step / head call's accepted checkpoints, two calls per program (the served whole-step step's head group
    and whole step: one dispatch). Returns the number of calls committed. With no read-offset buffer (no speculative
    decoding) the staging is written in full instead."""
    recs = _records(runner)
    if not recs:
        return 0
    offsets = getattr(runner, "mamba_slot_read_offsets", None)
    if offsets is None:
        return flush_pending(runner, why="no read offsets")
    i = 0
    while i < len(recs):
        ka, pa, a = recs[i]
        ops = _ops(ka.layout)
        if i + 1 < len(recs) and recs[i + 1][1] is pa and recs[i + 1][0].layout == ka.layout:
            b = recs[i + 1][2]
            ops.commit2(pa, offsets, a.conv, a.rec, a.wb, a.st, b.conv, b.rec, b.wb, b.st)
            i += 2
        else:
            ops.commit1(pa, offsets, a.conv, a.rec, a.wb, a.st)
            i += 1
    _clear(runner)
    n = len(recs)
    Counters.committed += n
    if Counters.committed <= 4 or Counters.committed % 4096 < n:
        _log.info("%s: committed %d calls (accepted checkpoint per request and GDN layer); flushed in full %d; staged %d",
                  ENGAGED, Counters.committed, Counters.flushed, Counters.stashed)
    return n


def flush_pending(runner: Any, why: str = "forward") -> int:
    """Write every pending call's checkpoints in full (the flag-off bytes): before a forward or a seed copy that could
    read the pool while staging is still uncommitted. Returns the number of calls flushed."""
    n = 0
    for kind, pool in _targets(runner):
        pend = kind.__dict__.get("_g1_pending")
        if not pend:
            continue
        full = _ops(kind.layout).full
        for s in pend:
            full(pool, s.conv, s.rec, s.wb)
        n += len(pend)
        pend.clear()
    if n:
        Counters.flushed += n
        if Counters.flushed <= 4 or Counters.flushed % 4096 < n:
            _log.info("%s: flushed %d calls in full (%s); committed %d; staged %d", ENGAGED, Counters.flushed, why,
                      Counters.committed, Counters.stashed)
    return n


def staging_shapes(layout: Any, NL: int, B: int, T: int = 4) -> dict:
    """Per-rank shapes / dtypes of one call's staging at `B` rows and `NL` GDN layers (the kernel's checkpoint outputs and
    the ids)."""
    from vllm_torchtpu.megakernel.ops import REC

    NS = B // T
    return dict(conv=((NL, NS, layout.hist_rows(T), layout.conv_dim), jnp.float32),
                rec=((NL, NS * T * layout.n_v, layout.d_k, layout.d_v), REC),
                wb=((NL, NS, T), jnp.int32), st=((NL, NS), jnp.int32))


def staging_bytes(layout: Any, NL: int, B: int, T: int = 4) -> int:
    """HBM bytes per rank one call's staging holds between the forward and the commit."""
    import math

    return sum(math.prod(s) * jnp.dtype(d).itemsize for s, d in staging_shapes(layout, NL, B, T).values())


def _plan(runner: Any) -> list:
    """(kind, pool, NL, rows) of every staging shape the served calls can produce: the whole step at its buckets, each
    head group at the driver's buckets (chunked above KERNEL_ROWS into KERNEL_ROWS pieces and a remainder)."""
    drv = getattr(runner, "_megakernel", None)
    whole = getattr(drv, "whole_module", None)
    out = []
    for kind, pool in _targets(runner):
        if whole is not None and kind is getattr(whole, "kind", None):
            out += [(kind, pool, len(whole.gdn_ids), b) for b in whole.buckets]
            continue
        K = HEAD_CHUNK_ROWS
        rows = set()
        for b in getattr(drv, "buckets", ()):
            rows |= {min(b, K)} | ({b % K} if b > K and b % K else set())
        out += [(kind, pool, int(kind.L), b) for b in sorted(rows)]
    return out


def precompile(runner: Any) -> None:
    """After bind_and_warmup: compile the commit and the full flush at every staging shape on all-null ids (writes
    nothing), so no program compiles mid-serving. Logs the engagement line with the staging bytes."""
    import torch

    plan = _plan(runner)
    if not plan:
        _log.warning("%s: no whole-step or head kind bound; nothing to precompile (the flag has no effect)", ENGAGED)
        return
    flush_pending(runner, why="warmup")
    tdt = {jnp.dtype(jnp.float32): torch.float32, jnp.dtype(jnp.bfloat16): torch.bfloat16,
           jnp.dtype(jnp.int32): torch.int32}
    offsets = getattr(runner, "mamba_slot_read_offsets", None)
    sizes, zs = [], {}
    for kind, pool, NL, B in plan:
        ops = _ops(kind.layout)
        z = {k: torch.zeros(s, dtype=tdt[jnp.dtype(d)], device=pool.device)
             for k, (s, d) in staging_shapes(kind.layout, NL, B).items()}
        zs[(id(kind), B)] = z
        if offsets is not None:
            ops.commit1(pool, offsets, z["conv"], z["rec"], z["wb"], z["st"])
        ops.full(pool, z["conv"], z["rec"], z["wb"])
        sizes.append(f"{NL}L b{B} {staging_bytes(kind.layout, NL, B) / 2**20:.1f} MiB")
    pairs = 0
    targets = _targets(runner)
    if offsets is not None and len(targets) > 1:  # the whole-step route: (whole step b, head group b) in one program
        (kw, pw), (kh, _) = targets[0], targets[1]
        for _, _, _, B in [p for p in plan if p[0] is kw]:
            a, b = zs[(id(kw), B)], zs.get((id(kh), B))
            if b is not None:
                _ops(kw.layout).commit2(pw, offsets, a["conv"], a["rec"], a["wb"], a["st"], b["conv"], b["rec"], b["wb"],
                                        b["st"])
                pairs += 1
    _log.info("%s engaged: commit + full flush precompiled for %d staging shapes and %d (whole, head) pairs (per rank: %s)",
              ENGAGED, len(plan), pairs, "; ".join(sizes))
