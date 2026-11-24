# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""In-kernel all-reduce of a bf16 [m, h] partial over the n ranks of a 1-D mesh.

A fused kernel calls `prologue` at entry and `epilogue` in place of its final
store, and adds `scratch_shapes` / `vmem_bytes` to its pallas_call;
`standalone` wraps the pair for the start-up self-test.

A rank is a device's index in the mesh. Protocol: reduce-scatter of n column
chunks [m, h / n] (rank r owns chunk r), an f32 sum of the n bf16 slots in
rank (mesh) order rounded to bf16, then an all-gather into out_ref. Each
receive slot has its own semaphore, indexed by sender; the own chunk moves by
a local copy; out_ref is both the reduce-scatter source and the all-gather
target, except on the fold path, which exchanges its folded matrix; rank r's
i-th copy goes to rank (r + i) % n, so no two ranks' i-th copies share a
target. Rows >= n_valid come out +0, and n_valid picks the path, which sets
how many rows travel (fold exists only when spec.fold, trim only when
tile_rows < m):

  zero  n_valid == 0: barrier wait, zeros.
  fold  1 <= n_valid <= fold_rows: the valid rows are folded into one
        [tile_rows, h / k] matrix, q[j * fold_rows + r, x] = p[r, j * h / k + x]
        (k = lane groups per chunk), and that is all-reduced instead.
  trim  above the fold range, n_valid <= tile_rows: rows [0, tile_rows) only.
  full  otherwise: all m rows.

n_valid must be the same on every rank (one replicated value). The barrier
semaphore gets one signal from every rank at entry and is waited for n
signals before each path's first remote copy, so every peer's buffers are
live. Every send is waited before exit, so every semaphore ends at 0.
"""

import dataclasses
import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BF16, F32 = jnp.bfloat16, jnp.float32
COLLECTIVE_ID = 0
VMEM_FRACTION = 0.75
_MESH = pl.DeviceIdType.MESH
# `standalone` asks for this multiple of its planned bytes.
_STANDALONE_HEADROOM = 2.0


def tiled_bytes(shape, dtype, info) -> int:
    """VMEM bytes of a buffer padded to Mosaic tiles: the last dim to the lane
    count, the second-last to the sublane tiling of the dtype."""
    itemsize = jnp.dtype(dtype).itemsize
    sublanes = info.num_sublanes * max(1, 4 // itemsize)
    *lead, rows, cols = (1, *shape) if len(shape) == 1 else shape
    rows = pl.cdiv(rows, sublanes) * sublanes
    cols = pl.cdiv(cols, info.num_lanes) * info.num_lanes
    return math.prod(lead) * rows * cols * itemsize


def supported(n: int, m: int, h: int, info) -> bool:
    """Whether an [m, h] partial over n ranks splits into lane-aligned column
    chunks and whole bf16 row tiles."""
    tile_rows = info.get_sublane_tiling(BF16)
    return (
        n >= 2
        and h % n == 0
        and (h // n) % info.num_lanes == 0
        and m >= tile_rows
        and m % tile_rows == 0
    )


@dataclasses.dataclass(frozen=True)
class AllReduceSpec:
    """Static geometry of one all-reduce (hashable; a jit key)."""

    n: int
    m: int
    h: int
    tile_rows: int  # bf16 sublane tiling
    lanes: int

    @property
    def chunk(self) -> int:
        return self.h // self.n

    @property
    def lane_groups(self) -> int:
        return self.chunk // self.lanes

    @property
    def fold_rows(self) -> int:
        return self.tile_rows // self.lane_groups

    @property
    def fold(self) -> bool:
        return self.lane_groups >= 2 and self.fold_rows >= 1


def make_spec(n: int, m: int, h: int) -> AllReduceSpec:
    info = pltpu.get_tpu_info()
    if not supported(n, m, h, info):
        raise ValueError(
            f"all-reduce of [{m}, {h}] over {n} ranks needs n >= 2, h / n a "
            f"multiple of {info.num_lanes} lanes and m a multiple of "
            f"{info.get_sublane_tiling(BF16)} rows"
        )
    return AllReduceSpec(n, m, h, info.get_sublane_tiling(BF16), info.num_lanes)


def kernel_name(spec: AllReduceSpec) -> str:
    return f"moe_tp_decode_all_reduce-m_{spec.m}-h_{spec.h}-n_{spec.n}"


def _scratch_names(spec: AllReduceSpec) -> tuple[str, ...]:
    fold = ("q",) if spec.fold else ()
    return ("recv", "red", *fold, "rs_send", "rs_recv", "ag_send", "ag_recv", "loc")


def scratch_shapes(spec: AllReduceSpec) -> list:
    """The epilogue's scratch, in the order `epilogue` takes it."""
    n, m, w = spec.n, spec.m, spec.chunk
    fold = (
        [pltpu.VMEM((spec.tile_rows, spec.h // spec.lane_groups), BF16)]
        if spec.fold
        else []
    )
    return [
        pltpu.VMEM((n, m, w), BF16),
        pltpu.VMEM((m, w), BF16),
        *fold,
        *[pltpu.SemaphoreType.DMA((n,)) for _ in range(4)],
        pltpu.SemaphoreType.DMA((2,)),
    ]


def vmem_bytes(spec: AllReduceSpec, info) -> int:
    """Tiled bytes of the scratch plus an upper bound of the epilogue's f32
    temporaries (two [m, h] matrices, k + 1 fold blocks): an estimate."""
    t = functools.partial(tiled_bytes, info=info)
    n, m, h, w = spec.n, spec.m, spec.h, spec.chunk
    total = t((n, m, w), BF16) + t((m, w), BF16) + 2 * t((m, h), F32)
    if spec.fold:
        block = (spec.tile_rows, h // spec.lane_groups)
        total += t(block, BF16) + (spec.lane_groups + 1) * t(block, F32)
    return total


def _cols(ref, t, size):
    """Column block t (traced) of width `size` of a 2-D ref view."""
    return ref.at[:, pl.ds(pl.multiple_of(t * size, size), size)]


def _send_all(spec, me, src_of, dst_of, send_sem, recv_sem):
    """Start my src_of(t) -> rank t's dst_of(me) for every other rank t, the
    k-th to t = (me + k) % n; rank t's recv_sem.at[me] counts the bytes."""
    sends = []
    for step in range(1, spec.n):
        t = lax.rem(me + step, spec.n)
        copy = pltpu.make_async_remote_copy(
            src_of(t),
            dst_of(me),
            send_sem.at[step],
            recv_sem.at[me],
            device_id=(t,),
            device_id_type=_MESH,
        )
        copy.start()
        sends.append(copy)
    return sends


def _recv_all(spec, me, dst_of, recv_sem):
    """Wait for every other rank t's copy into my dst_of(t) (byte exact: a
    local descriptor of the received view)."""
    peers = [lax.rem(me + (spec.n - step), spec.n) for step in range(1, spec.n)]
    for t in peers:
        pltpu.make_async_copy(dst_of(t), dst_of(t), recv_sem.at[t]).wait_recv()


def _sum_slots(load, count):
    """f32 sum of count bf16 slots in slot (rank) order."""
    acc = load(0).astype(F32)
    for s in range(1, count):
        acc = acc + load(s).astype(F32)
    return acc


def _masked(partial, nv, rows, dtype):
    """partial[:rows] rounded to bf16, then in `dtype`, with rows >= nv set
    to +0 (nv None: no mask). The select runs in f32 (the int32 row mask has
    the 32-bit layout), so the values stay exact."""
    p = partial if rows == partial.shape[0] else partial[:rows]
    p = p.astype(BF16)
    if nv is None:
        return p.astype(dtype)
    row = lax.broadcasted_iota(jnp.int32, p.shape, 0)
    return jnp.where(row < nv, p.astype(F32), 0.0).astype(dtype)


def _stage_rows(out_ref, partial, nv):
    """out_ref = partial (bf16) with rows >= nv zeroed; a full bucket skips
    the mask."""
    m = out_ref.shape[0]

    @pl.when(nv == m)
    def _all_valid():
        out_ref[...] = _masked(partial, None, m, out_ref.dtype)

    @pl.when(nv < m)
    def _some_padding():
        out_ref[...] = _masked(partial, nv, m, out_ref.dtype)


def _fold(spec, p32):
    """[tile_rows, h] f32 (rows >= n_valid zero) -> [tile_rows, h / k] f32."""
    tr, k, r = spec.tile_rows, spec.lane_groups, spec.fold_rows
    hk = spec.h // k
    row = lax.broadcasted_iota(jnp.int32, (tr, hk), 0)
    q = jnp.zeros((tr, hk), F32)
    for j in range(k):
        blk = p32[:, j * hk : (j + 1) * hk]
        if j:
            blk = pltpu.roll(blk, j * r, 0)
        q = jnp.where((row >= j * r) & (row < (j + 1) * r), blk, q)
    return q


def _unfold(spec, q32, nv, out_ref):
    """Inverse of _fold into out_ref; rows >= nv are +0."""
    tr, k, r, m = spec.tile_rows, spec.lane_groups, spec.fold_rows, spec.m
    hk = spec.h // k
    row = lax.broadcasted_iota(jnp.int32, (tr, hk), 0)
    for j in range(k):
        blk = q32 if j == 0 else pltpu.roll(q32, tr - j * r, 0)
        out_ref[pl.ds(0, tr), pl.ds(j * hk, hk)] = jnp.where(row < nv, blk, 0.0).astype(
            out_ref.dtype
        )
    if m > tr:
        out_ref[pl.ds(tr, m - tr), :] = jnp.zeros((m - tr, spec.h), out_ref.dtype)


def _exchange(spec, me, buf, rows, cols, s):
    """All-reduce of buf ([rows, cols], in place): reduce-scatter, f32 sum,
    all-gather."""
    n, cw = spec.n, cols // spec.n

    def slot(q):
        return s["recv"].at[q, pl.ds(0, rows), pl.ds(0, cw)]

    def chunk(t):
        return _cols(buf, t, cw)

    red = s["red"].at[pl.ds(0, rows), pl.ds(0, cw)]
    own = pltpu.make_async_copy(chunk(me), slot(me), s["loc"].at[0])
    own.start()
    rs = _send_all(spec, me, chunk, slot, s["rs_send"], s["rs_recv"])
    own.wait()
    _recv_all(spec, me, slot, s["rs_recv"])
    red[...] = _sum_slots(
        lambda q: s["recv"][q, pl.ds(0, rows), pl.ds(0, cw)], n
    ).astype(red.dtype)
    own_ag = pltpu.make_async_copy(red, chunk(me), s["loc"].at[1])
    own_ag.start()
    ag = _send_all(spec, me, lambda t: red, chunk, s["ag_send"], s["ag_recv"])
    own_ag.wait()
    _recv_all(spec, me, chunk, s["ag_recv"])
    for copy in (*rs, *ag):
        copy.wait_send()


def _run_path(spec, kind, me, nv, partial, out_ref, s, barrier):
    """One row path (fold / trim / full): stage, barrier wait, exchange."""
    m, h, tr = spec.m, spec.h, spec.tile_rows
    if kind == "fold":
        q = _fold(spec, _masked(partial, nv, tr, F32))
        s["q"][...] = q.astype(s["q"].dtype)
        buf, rows, cols = s["q"], tr, h // spec.lane_groups
    else:
        rows, cols = (m if kind == "full" else tr), h
        _stage_rows(out_ref, partial, nv)
        buf = out_ref if rows == m else out_ref.at[pl.ds(0, rows)]
    pl.semaphore_wait(barrier, spec.n)
    _exchange(spec, me, buf, rows, cols, s)
    if kind == "fold":
        _unfold(spec, s["q"][...].astype(F32), nv, out_ref)


def prologue(spec: AllReduceSpec) -> None:
    """Kernel entry: signal the barrier semaphore +1 on every rank, self
    included, so the work before the epilogue hides the barrier."""
    barrier = pltpu.get_barrier_semaphore()
    for d in range(spec.n):
        pl.semaphore_signal(
            barrier,
            inc=1,
            device_id=(d,),
            device_id_type=_MESH,
        )


def epilogue(spec: AllReduceSpec, rank, n_valid, partial, out_ref, scr) -> None:
    """All-reduce `partial` ([m, h], rounded to bf16) over the n ranks into
    out_ref, in place of the kernel's final store. `prologue` must have run
    at entry; scr: the refs of scratch_shapes(spec).

    rank (traced int32) must be this device's index in the 1-D mesh and
    n_valid (traced int32) the same on every rank; a violation hangs the
    kernel, which the start-up self-test exists to catch.
    """
    if partial.shape != (spec.m, spec.h) or tuple(out_ref.shape) != (spec.m, spec.h):
        raise ValueError(
            f"{kernel_name(spec)}: partial {partial.shape} / out {out_ref.shape} "
            f"!= ({spec.m}, {spec.h})"
        )
    s = dict(zip(_scratch_names(spec), scr, strict=True))
    barrier = pltpu.get_barrier_semaphore()
    m, tr = spec.m, spec.tile_rows
    nv = jnp.minimum(jnp.maximum(n_valid, 0), m)

    @pl.when(nv == 0)
    def _zero():
        out_ref[...] = jnp.zeros(out_ref.shape, out_ref.dtype)
        pl.semaphore_wait(barrier, spec.n)

    lo = 0
    if spec.fold:

        @pl.when((nv > 0) & (nv <= spec.fold_rows))
        def _fold_path():
            _run_path(spec, "fold", rank, nv, partial, out_ref, s, barrier)

        lo = spec.fold_rows
    if tr < m:

        @pl.when((nv > lo) & (nv <= tr))
        def _trim_path():
            _run_path(spec, "trim", rank, nv, partial, out_ref, s, barrier)

        lo = tr

    @pl.when(nv > lo)
    def _full_path():
        _run_path(spec, "full", rank, nv, partial, out_ref, s, barrier)


# ---------------------------------------------------------------------------
# Start-up self-test: the epilogue alone, on partials whose sums are exact.
# ---------------------------------------------------------------------------
def standalone(spec: AllReduceSpec):
    """pallas_call(meta, partial) -> out running prologue and epilogue only.

    meta: int32 [2] = (n_valid, rank) by scalar prefetch; partial and out:
    bf16 [m, h]. Call under shard_map over the 1-D mesh of n ranks.
    """

    def kernel(meta_ref, x_ref, out_ref, *scr):
        rank = meta_ref[1]
        prologue(spec)
        epilogue(spec, rank, meta_ref[0], x_ref[...], out_ref, scr)

    info = pltpu.get_tpu_info()
    need = 2 * tiled_bytes((spec.m, spec.h), BF16, info) + vmem_bytes(spec, info)
    limit = int(need * _STANDALONE_HEADROOM)
    if limit > int(info.vmem_capacity_bytes * VMEM_FRACTION):
        raise ValueError(f"{kernel_name(spec)} needs a VMEM limit of {limit} B")
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((spec.m, spec.h), BF16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[vmem],
            out_specs=vmem,
            scratch_shapes=scratch_shapes(spec),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=limit, collective_id=COLLECTIVE_ID
        ),
        name=kernel_name(spec),
    )


def selftest_inputs(spec: AllReduceSpec) -> np.ndarray:
    """[n, m, h] f32 partials whose partial and full sums are integers exact
    in bf16 and differ between column blocks: rank 0 holds its lane-block
    index modulo 256 - 6 (n - 1), rank d >= 1 holds
    ((d + 1) 3 + 5 row + 7 block) % 11 - 5. A chunk landing in the wrong
    column block (a wrong rank) changes the sum."""
    n, m, h = spec.n, spec.m, spec.h
    big = 2 ** (jnp.finfo(BF16).nmant + 1) - 6 * (n - 1)
    if big < 2:
        raise ValueError(f"the self-test sums are not exact in bf16 at n = {n}")
    d = np.arange(n)[:, None, None]
    r = np.arange(m)[None, :, None]
    c = np.arange(h)[None, None, :] // spec.lanes
    x = (((d + 1) * 3 + r * 5 + c * 7) % 11 - 5).astype(np.float32)
    x[0] = np.broadcast_to((c[0] % big).astype(np.float32), (m, h))
    return x


def selftest_row_counts(spec: AllReduceSpec) -> list[int]:
    """One n_valid per row path: full, fold, trim (where they exist), zero."""
    counts = [spec.m]
    if spec.fold:
        counts.append(spec.fold_rows)
    if spec.tile_rows < spec.m:
        counts.append(spec.tile_rows)
    return [*counts, 0]


def selftest_ok(spec: AllReduceSpec, x: np.ndarray, out, n_valid: int) -> bool:
    """One rank's check of an all-reduce of selftest_inputs: rows < n_valid
    equal the exact sum, the other rows are +0."""
    want = np.asarray(x, np.float32).sum(0)
    got = np.asarray(out, np.float32).reshape(spec.m, spec.h)
    valid = min(max(int(n_valid), 0), spec.m)
    pad = got[valid:]
    return bool(
        np.array_equal(got[:valid], want[:valid])
        and np.all((pad == 0) & ~np.signbit(pad))
    )
