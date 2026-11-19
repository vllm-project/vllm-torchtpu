# SPDX-License-Identifier: Apache-2.0
"""Hierarchical all-reduce of an 8-core tensor-parallel group on four chips (H8).

Q38 DEC-FB-2 lever 12: the TP8 translation of vllm-torchtpu PR 2821 commit
b41e1c6ea (``TPU_TP4_HIER_ALL_REDUCE``, ``tp4_all_reduce.py``), which is TP4
only. The group is four v7x chips x two TensorCores; ``pairs[c] = (i0, i1)``
gives the mesh indices (along ``axis``) of chip ``c``'s core 0 and core 1. The
chip-to-chip (ICI) hop is the slow one, so each chip sums its two partials over
the die-to-die link first and only half of the data crosses chips:

1. intra-chip reduce-scatter: core ``k`` keeps column half ``k`` and adds the
   other core's copy of it, one rounding to bf16 (the pair sum; the wire stays
   bf16, which is what halves the cross-chip bytes),
2. cross-chip reduce-scatter: inside half ``k``, chip ``c`` finishes shard
   ``(k, c)`` from the four chips' pair sums, accumulated in f32 in the fixed
   chip order 0, 1, 2, 3 and rounded to bf16 once,
3. all-gather of the finished shards. ``ag="fwd"`` (default, fewest ICI bytes):
   the shard goes to the three other chips' core ``k`` and to the chip's other
   core, and each core forwards the shards it receives to its other core.
   ``ag="direct"``: the owner sends its shard to all seven members (twice the
   all-gather ICI bytes, one die-to-die hop fewer).

Every element is ``bf16(((p0 + p1) + p2) + p3)`` in f32, with ``p_c =
bf16(x[c, core 0] + x[c, core 1])``, on every member: each shard is finished by
exactly one member and copied, so all eight outputs are bitwise identical, run
after run. ``reference`` is the numpy model of that order.

Precision. The bf16 pair sum (``wire="bf16"``, the PR's choice) is the only
rounding before the final one, and it is what halves the cross-chip bytes; it
also moves the result off the single-rounding f32 sum (on N(0, 1) partials
about 12% of elements by more than 1 ulp). ``wire="f32"`` sends the pair sums
in f32 instead: the result is then exactly ``bf16(((q0 + q1) + q2) + q3)``,
``q_c = x[c, core 0] + x[c, core 1]`` in f32 (``reference(..., wire="f32")``),
for twice the reduce-scatter bytes across chips (2.25x instead of 1.5x the
payload per chip, against 3x for a flat one-hop all-reduce).

Shards. Hidden 2560 is 20 lane tiles: halves of 10 tiles, which do not split
four ways. Half 0 gives 3, 3, 2, 2 tiles to chips 0..3 and half 1 gives 2, 2,
3, 3, so every chip finishes exactly 5 tiles (its two cores 3 + 2): each chip's
ICI load is the even split, only the per-core load is uneven. Padding to 3072
(as ``fused_moe_decode`` does) has the same largest shard and moves 20% more
bytes; a row split needs rows / 4 to be a multiple of 16, which 112 rows is not.

Shapes are static per member, so the kernel is eight static branches on the
member's index (one per (chip, core) role): every DMA has a static size and
offset and every wait matches its sender's size.

Barrier: one round on the global barrier semaphore with the members this
member writes to (its other core and the three other chips' core ``k``; all
seven with ``ag="direct"``). A remote DMA lands only after its target entered
the kernel. One round suffices across back-to-back calls: every barrier peer
P receives data that this member sends after its own barrier wait and P waits
for it before leaving, so P cannot signal the next call before this member has
consumed this call's signal.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_MESH = pl.DeviceIdType.MESH
_LANES = 128
N_CHIPS, N_CORES = 4, 2
N_MEMBERS = N_CHIPS * N_CORES
MAX_ROWS = 256
# Barrier semaphore of this kernel. In use elsewhere in the tree: 0
# (fused_moe_decode), and PR 2821 adds 0 (tp4) and 6 (dma_all_gather).
COLLECTIVE_ID = 7
AG_MODES = ("fwd", "direct")
WIRES = ("bf16", "f32")
DEFAULT_PAIRS = ((0, 1), (2, 3), (4, 5), (6, 7))
# Kernel refs in pallas_call order (input, output, then scratch).
_REF_NAMES = ("x_hbm", "o_hbm", "xv", "r1", "ps", "r2", "fin", "ld", "s1s", "s1r",
              "s2s", "s2r", "s3s", "agr", "fws", "fwr", "st")
# Trace-time schedule record for the CPU protocol test: when a list, tracing
# the kernel appends (member, event, ...) for every barrier signal and wait and
# every remote-copy start / wait. None (the default) records nothing.
_SCHEDULE_TRACE: list | None = None


def shard_table(hidden: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    """``table[k][c] = (start, width)``: the columns chip ``c`` finishes in half
    ``k``. Whole lane tiles; half 0 gives its extra tiles to the first chips and
    half 1 to the last ones, so each chip finishes the same number of tiles
    whenever the extras split evenly (hidden 2560: 5 tiles per chip)."""
    if hidden % (2 * _LANES):
        raise ValueError(f"hidden {hidden} is not two halves of whole lane tiles")
    half_tiles = hidden // (2 * _LANES)
    if half_tiles < N_CHIPS:
        raise ValueError(f"hidden {hidden}: a half has fewer than {N_CHIPS} tiles")
    base, extra = divmod(half_tiles, N_CHIPS)
    table = []
    for k in range(N_CORES):
        has = [
            (c < extra) if k == 0 else (c >= N_CHIPS - extra) for c in range(N_CHIPS)
        ]
        start = k * half_tiles
        row = []
        for c in range(N_CHIPS):
            tiles = base + int(has[c])
            row.append((start * _LANES, tiles * _LANES))
            start += tiles
        table.append(tuple(row))
    return tuple(table)


def supports(rows: int, hidden: int, dtype, max_rows: int = MAX_ROWS) -> bool:
    """Shapes the kernel takes: bf16, whole 16-row tiles up to ``max_rows`` rows
    (VMEM staging is about 3 x rows x hidden bf16), halves of whole lane tiles."""
    if jnp.dtype(dtype) != jnp.bfloat16 or not (0 < rows <= max_rows) or rows % 16:
        return False
    try:
        shard_table(hidden)
    except ValueError:
        return False
    return True


def check_pairs(pairs) -> tuple[tuple[int, int], ...]:
    pairs = tuple(tuple(int(i) for i in p) for p in pairs)
    flat = [i for p in pairs for i in p]
    if len(pairs) != N_CHIPS or any(len(p) != N_CORES for p in pairs):
        raise ValueError(f"pairs must be {N_CHIPS} (core0, core1) tuples: {pairs}")
    if sorted(flat) != list(range(N_MEMBERS)):
        raise ValueError(f"pairs must use each mesh index 0..7 once: {pairs}")
    return pairs


def placement_from_devices(devices) -> tuple[tuple[int, int], ...]:
    """``pairs`` for a mesh axis whose index ``i`` is ``devices[i]``: chips in
    ``coords`` order, cores by ``core_on_chip``. Without coordinates, falls back
    to device ids (two cores per chip numbered consecutively). A wrong guess
    costs speed only: the kernel addresses members by mesh index."""
    devices = list(devices)
    if len(devices) != N_MEMBERS:
        raise ValueError(f"need {N_MEMBERS} devices, got {len(devices)}")
    chips: dict[tuple, dict[int, int]] = {}
    if all(
        getattr(d, "coords", None) is not None and hasattr(d, "core_on_chip")
        for d in devices
    ):
        for i, d in enumerate(devices):
            chips.setdefault(tuple(d.coords), {})[int(d.core_on_chip)] = i
    else:
        for i, d in enumerate(devices):
            chips.setdefault((int(d.id) // 2,), {})[int(d.id) % 2] = i
    if len(chips) != N_CHIPS or any(sorted(v) != [0, 1] for v in chips.values()):
        raise ValueError(f"devices are not {N_CHIPS} chips x {N_CORES} cores: {chips}")
    return check_pairs(tuple((chips[key][0], chips[key][1]) for key in sorted(chips)))


def _bf16_round(a: np.ndarray) -> np.ndarray:
    import ml_dtypes

    return np.asarray(a, np.float32).astype(ml_dtypes.bfloat16).astype(np.float32)


_F32_TINY = np.float32(2.0**-126)


def _flush(v: np.ndarray) -> np.ndarray:
    """Subnormal f32 values to signed zero."""
    return np.where((np.abs(v) < _F32_TINY) & (v != 0), np.copysign(np.float32(0), v), v)


def _add(a, b, ftz: bool) -> np.ndarray:
    if ftz:
        return _flush((_flush(a) + _flush(b)).astype(np.float32))
    return (a + b).astype(np.float32)


def reference(x, pairs=DEFAULT_PAIRS, wire: str = "bf16", ftz: bool = True) -> np.ndarray:
    """The kernel's exact order in numpy. ``x`` [8, R, H] (bf16 values, any float
    dtype), member order = mesh index. Returns [R, H] float32 holding bf16
    values: ``bf16(((p0 + p1) + p2) + p3)`` in f32, ``p_c = x[c0] + x[c1]`` in
    f32, rounded to bf16 when ``wire="bf16"``. ``ftz``: f32 adds flush
    subnormal inputs and results to signed zero, as XLA:CPU does (measured: the
    Pallas interpreter's adds) and as the TPU vector unit is expected to (not
    verified on device; the bench plants no subnormals)."""
    pairs = check_pairs(pairs)
    if wire not in WIRES:
        raise ValueError(f"wire must be one of {WIRES}: {wire!r}")
    x = _bf16_round(np.asarray(x, np.float32))
    p = [_add(x[a], x[b], ftz) for a, b in pairs]
    if wire == "bf16":
        p = [_bf16_round(t) for t in p]
    acc = p[0]
    for t in p[1:]:
        acc = _add(acc, t, ftz)
    return _bf16_round(acc)


def reference_f32(x, pairs=DEFAULT_PAIRS, ftz: bool = True) -> np.ndarray:
    """The same chip order with no intermediate rounding: bf16 of the f32 sum
    ``((q0 + q1) + q2) + q3``, ``q_c = x[c0] + x[c1]`` (f32)."""
    return reference(x, pairs, wire="f32", ftz=ftz)


def tp8_hier_all_reduce_local(
    x: jax.Array,
    *,
    axis: str,
    pairs=DEFAULT_PAIRS,
    ag: str = "fwd",
    wire: str = "bf16",
    collective_id: int = COLLECTIVE_ID,
    interpret: bool = False,
) -> jax.Array:
    """All-reduce of this member's ``x`` [R, H] bf16 over mesh axis ``axis``
    (size 8, chips ``pairs``); call inside shard_map. ``interpret`` (True or a
    ``pltpu.InterpretParams``) runs the Pallas TPU interpreter (CPU tests)."""
    pairs = check_pairs(pairs)
    if ag not in AG_MODES:
        raise ValueError(f"ag must be one of {AG_MODES}: {ag!r}")
    if wire not in WIRES:
        raise ValueError(f"wire must be one of {WIRES}: {wire!r}")
    rows, hidden = x.shape
    if not supports(rows, hidden, x.dtype):
        raise ValueError(f"unsupported shape {x.shape} {x.dtype}")
    table = shard_table(hidden)
    half = hidden // 2
    wmax = max(w for row in table for _, w in row)
    fwd = ag == "fwd"
    f32, bf16 = jnp.float32, jnp.bfloat16
    wdt = bf16 if wire == "bf16" else f32  # pair sums: VMEM and cross-chip wire

    def loc(k, c):
        s, w = table[k][c]
        return s - k * half, w

    def role(c, k, refs):
        r = dict(zip(_REF_NAMES, refs))
        x_hbm, o_hbm, xv, r1, ps, r2, fin = (r[n] for n in _REF_NAMES[:7])
        me = pairs[c][k]
        o = pairs[c][1 - k]
        others = [(c + j) % N_CHIPS for j in range(1, N_CHIPS)]
        trace = _SCHEDULE_TRACE

        def dev(i):
            return {axis: i}

        class _Remote:
            """A remote copy whose semaphores are named (name, slot); every
            start and wait is recorded in ``_SCHEDULE_TRACE`` when it is set."""

            def __init__(self, src, dst, send, recv, peer):
                self.d = pltpu.make_async_remote_copy(
                    src, dst, r[send[0]].at[send[1]], r[recv[0]].at[recv[1]],
                    device_id=dev(peer), device_id_type=_MESH,
                )
                self.send, self.recv, self.peer = send, recv, peer
                self.nbytes = math.prod(dst.shape) * jnp.dtype(dst.dtype).itemsize

            def _log(self, *event):
                if trace is not None:
                    trace.append((me,) + event)

            def start(self):
                self._log("start", self.peer, self.send, self.recv, self.nbytes)
                self.d.start()
                return self

            def wait_recv(self):
                self._log("wait_recv", self.recv, self.nbytes)
                self.d.wait_recv()

            def wait_send(self):
                self._log("wait_send", self.send, self.nbytes)
                self.d.wait_send()

        # My partial of half k, block by block (one block per finishing chip).
        loads = {}
        for cc in others + [c]:
            s, w = table[k][cc]
            l, _ = loc(k, cc)
            cp = pltpu.make_async_copy(
                x_hbm.at[:, pl.ds(s, w)], xv.at[:, pl.ds(l, w)], r["ld"].at[cc]
            )
            cp.start()
            loads[cc] = cp

        if fwd:
            peers = [o] + [pairs[cc][k] for cc in others]
        else:
            peers = [pairs[cc][kk] for cc in range(N_CHIPS) for kk in range(N_CORES)
                     if (cc, kk) != (c, k)]
        barrier = pltpu.get_barrier_semaphore()

        def barrier_signal(p):
            if trace is not None:
                trace.append((me, "signal", p))
            pl.semaphore_signal(barrier, 1, device_id=dev(p), device_id_type=_MESH)

        def barrier_wait(n):
            if trace is not None:
                trace.append((me, "barrier_wait", n))
            pl.semaphore_wait(barrier, n)

        for p in peers:
            barrier_signal(p)
        barrier_wait(len(peers))

        # 1. The other core's half goes to the other core, one block per shard.
        sends = []
        for cc in others + [c]:
            s, w = table[1 - k][cc]
            l, _ = loc(1 - k, cc)
            sends.append(_Remote(x_hbm.at[:, pl.ds(s, w)], r1.at[:, pl.ds(l, w)],
                                 ("s1s", 0), ("s1r", cc), o).start())

        def pair_sum(cc):
            s, w = table[k][cc]
            l, _ = loc(k, cc)
            loads[cc].wait()
            _Remote(x_hbm.at[:, pl.ds(s, w)], r1.at[:, pl.ds(l, w)], ("s1s", 0),
                    ("s1r", cc), o).wait_recv()
            ps[:, pl.ds(l, w)] = (
                xv[:, pl.ds(l, w)].astype(f32) + r1[:, pl.ds(l, w)].astype(f32)
            ).astype(wdt)

        # 2. Pair sums of each other chip's shard go to that chip's core k.
        for cc in others:
            pair_sum(cc)
            l, w = loc(k, cc)
            sends.append(_Remote(ps.at[:, pl.ds(l, w)], r2.at[c, :, pl.ds(0, w)],
                                 ("s2s", 0), ("s2r", c), pairs[cc][k]).start())
        pair_sum(c)

        # My shard (k, c): the four chips' pair sums in chip order, f32, one rounding.
        lm, wm = loc(k, c)
        sm, _ = table[k][c]
        for cc in others:
            _Remote(ps.at[:, pl.ds(lm, wm)], r2.at[cc, :, pl.ds(0, wm)], ("s2s", 0),
                    ("s2r", cc), pairs[cc][k]).wait_recv()
        acc = None
        for cc in range(N_CHIPS):
            t = ps[:, pl.ds(lm, wm)] if cc == c else r2[cc, :, pl.ds(0, wm)]
            acc = t.astype(f32) if acc is None else acc + t.astype(f32)
        fin[:, pl.ds(0, wm)] = acc.astype(bf16)

        # 3. All-gather of the finished shards.
        store = pltpu.make_async_copy(
            fin.at[:, pl.ds(0, wm)], o_hbm.at[:, pl.ds(sm, wm)], r["st"].at[0]
        )
        store.start()
        if fwd:
            targets = [(pairs[cc][k], ("agr", c)) for cc in others] + [(o, ("fwr", 0))]
        else:
            targets = [(p, ("fwr", 0)) for p in peers]
        for p, rsem in targets:
            sends.append(_Remote(fin.at[:, pl.ds(0, wm)], o_hbm.at[:, pl.ds(sm, wm)],
                                 ("s3s", 0), rsem, p).start())
        if fwd:
            # Each other chip's shard of half k is forwarded to the other core as
            # it lands.
            for cc in others:
                s, w = table[k][cc]
                _Remote(fin.at[:, pl.ds(0, w)], o_hbm.at[:, pl.ds(s, w)], ("s3s", 0),
                        ("agr", cc), pairs[cc][k]).wait_recv()
                sends.append(_Remote(o_hbm.at[:, pl.ds(s, w)], o_hbm.at[:, pl.ds(s, w)],
                                     ("fws", 0), ("fwr", 0), o).start())
            incoming = [table[1 - k][cc] for cc in range(N_CHIPS)]
        else:
            incoming = [table[kk][cc] for cc in range(N_CHIPS) for kk in range(N_CORES)
                        if (cc, kk) != (c, k)]
        # fwd: half 1 - k from the other core; direct: every other member's shard.
        for s, w in incoming:
            _Remote(o_hbm.at[:, pl.ds(s, w)], o_hbm.at[:, pl.ds(s, w)], ("fws", 0),
                    ("fwr", 0), o).wait_recv()
        for d in sends:
            d.wait_send()
        store.wait()

    def kernel(*refs):
        me = lax.axis_index(axis)
        for c in range(N_CHIPS):
            for k in range(N_CORES):

                @pl.when(me == pairs[c][k])
                def _(c=c, k=k):
                    role(c, k, refs)

    space = pl.ANY if interpret else pltpu.MemorySpace.HBM
    if isinstance(interpret, pltpu.InterpretParams):
        extra = {"interpret": interpret}
    else:
        extra = {"interpret": pltpu.InterpretParams()} if interpret else {}
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[pl.BlockSpec(memory_space=space)],
        out_specs=pl.BlockSpec(memory_space=space),
        scratch_shapes=[
            pltpu.VMEM((rows, half), bf16),  # xv: my partial of half k
            pltpu.VMEM((rows, half), bf16),  # r1: the other core's partial of half k
            pltpu.VMEM((rows, half), wdt),  # ps: my chip's pair sums of half k
            pltpu.VMEM((N_CHIPS, rows, wmax), wdt),  # r2: pair sums of my shard, by chip
            pltpu.VMEM((rows, wmax), bf16),  # fin: my finished shard
            pltpu.SemaphoreType.DMA((N_CHIPS,)),  # ld: loads of xv, by block
            pltpu.SemaphoreType.DMA((1,)),  # s1s
            pltpu.SemaphoreType.DMA((N_CHIPS,)),  # s1r: by block
            pltpu.SemaphoreType.DMA((1,)),  # s2s
            pltpu.SemaphoreType.DMA((N_CHIPS,)),  # s2r: by sending chip
            pltpu.SemaphoreType.DMA((1,)),  # s3s
            pltpu.SemaphoreType.DMA((N_CHIPS,)),  # agr: same-core shards, by owner chip
            pltpu.SemaphoreType.DMA((1,)),  # fws
            pltpu.SemaphoreType.DMA((1,)),  # fwr: shards from the other core (all, direct)
            pltpu.SemaphoreType.DMA((1,)),  # st
        ],
        compiler_params=pltpu.CompilerParams(collective_id=collective_id),
        name=f"tp8_hier_all_reduce_{ag}_{wire}",
        **extra,
    )(x)


def tp8_hier_all_reduce(
    x: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    axis: str,
    pairs=DEFAULT_PAIRS,
    ag: str = "fwd",
    wire: str = "bf16",
    max_rows: int = MAX_ROWS,
    collective_id: int = COLLECTIVE_ID,
    interpret: bool = False,
) -> jax.Array:
    """Global view: ``x`` [8, R, H] stacks every member's partial along ``axis``;
    returns the same layout with every member's sum. Shapes the kernel does not
    take use XLA's all-reduce (``lax.psum``)."""
    spec = jax.sharding.PartitionSpec(axis)
    rows, hidden = x.shape[1], x.shape[2]
    if mesh.shape[axis] != N_MEMBERS or not supports(rows, hidden, x.dtype, max_rows):

        def body(a):
            return lax.psum(a, axis)
    else:

        def body(a):
            return tp8_hier_all_reduce_local(
                a[0], axis=axis, pairs=pairs, ag=ag, wire=wire, collective_id=collective_id,
                interpret=interpret,
            )[None]

    return jax.shard_map(body, mesh=mesh, in_specs=(spec,), out_specs=spec,
                         check_vma=False)(x)


# ---- no-new-kernel XLA arms (inside shard_map over a ("chip", "core") mesh) ----


def xla_two_level(x: jax.Array, *, chip_axis: str, core_axis: str) -> jax.Array:
    """psum over the chip's two cores, then over the four chips (full bytes)."""
    return lax.psum(lax.psum(x, core_axis), chip_axis)


def xla_hier(x: jax.Array, *, chip_axis: str, core_axis: str) -> jax.Array:
    """XLA's hierarchical all-reduce: reduce-scatter of the columns over the
    core pair, all-reduce of the half over the chips, all-gather over the pair."""
    half = lax.psum_scatter(x, core_axis, scatter_dimension=x.ndim - 1, tiled=True)
    half = lax.psum(half, chip_axis)
    return lax.all_gather(half, core_axis, axis=x.ndim - 1, tiled=True)


def xla_rs_ag(x: jax.Array, *, axis) -> jax.Array:
    """Flat reduce-scatter of the columns over all members, then all-gather."""
    part = lax.psum_scatter(x, axis, scatter_dimension=x.ndim - 1, tiled=True)
    return lax.all_gather(part, axis, axis=x.ndim - 1, tiled=True)


def chip_core_mesh(devices, pairs, names=("chip", "core")) -> jax.sharding.Mesh:
    """The (4, 2) mesh [chip][core] of a TP8 group from its 1-D device order."""
    pairs = check_pairs(pairs)
    devices = list(devices)
    grid = np.array([[devices[a], devices[b]] for a, b in pairs], dtype=object)
    return jax.sharding.Mesh(grid, names)


def ulp_distance(a, b) -> np.ndarray:
    """Elementwise distance in bf16 ulps between two arrays of bf16 values
    (+0 and -0 are 0 apart; NaN gives a huge distance)."""
    import ml_dtypes

    def ordered(v):
        u = np.asarray(v, np.float32).astype(ml_dtypes.bfloat16).view(np.uint16).astype(
            np.int32
        )
        mag = u & 0x7FFF
        return np.where(u & 0x8000, -mag, mag)

    d = np.abs(ordered(a) - ordered(b))
    nan = np.isnan(np.asarray(a, np.float32)) | np.isnan(np.asarray(b, np.float32))
    return np.where(nan, 1 << 20, d)
