"""Pallas TPU all-reduce (sum) of a small bf16 [t, h] activation over a 1-D mesh.

Every device of the mesh holds a partial [t, h]; every device ends with the
same sum. The kernel is meant for the decode-size tensor-parallel all-reduce
(t = 16 tokens, h = 6144, n = 8 or 16 devices), where XLA's TensorCore
all-reduce costs about 14 us and a Pallas call with one barrier and two or
three exchange rounds costs less.

Algorithms (`Plan.algo`):

  flat2  Two-shot over all n devices. Reduce-scatter: device r sends column
         chunk j (h / n columns) of its partial to device j. Each device sums
         its n slots in float32 in mesh order, rounds to bf16, and sends the
         reduced chunk to every device (all-gather). 1 barrier + 2 rounds of
         n - 1 remote copies. Correct for any mesh.
  hier1  Two levels. Needs mesh positions (2c, 2c + 1) to be the two
         TensorCores of one chip (true for the vllm-torchtpu rank order on
         tpu7x; see `pairs_share_chip`). Core k of chip c owns column half k.
           1. die-to-die (D2D) reduce-scatter: send the other half to the
              sibling core; add the sibling's half in float32 -> chip-half
              sum, rounded to bf16.
           2. inter-chip one-shot: send the chip-half sum to core k of every
              other chip; sum the n / 2 chip-half sums in float32 in chip
              order, round to bf16.
           3. D2D all-gather: send the reduced half to the sibling.
         1 barrier + 3 rounds; only round 2 leaves the chip.
  hier2  As hier1, but round 2 is a two-shot among the n / 2 chips (each core
         reduces h / n columns), and the reduced chunk goes directly to all
         n - 1 other devices. 1 barrier + 3 rounds, fewer inter-chip bytes.

Summation order and rounding. flat2 rounds once (float32 sum of n bf16
values). hier1/hier2 round twice (the chip-half sum, then the final sum).
Every device computes a given output element from the same bf16 inputs in
the same order, so the result is bit-identical on all devices.

Synchronization. Every device signals the barrier semaphore of every device
(itself included) at entry and waits for n signals before its first remote
copy, so every remote destination buffer belongs to a peer already inside
the same call. One barrier round is enough across back-to-back calls:
every device's exit depends, directly or through its sibling, on data that
every peer sent after its own barrier wait, so a signal from call i + 1 can
never be counted by a peer still waiting in call i. Every send is waited
before exit and every receive semaphore is waited for its exact byte count,
so every semaphore ends at zero.

`rank` is this device's mesh index as int32 data (the torch_tpu export
bridge rejects `lax.axis_index`). Removal arms (`Plan.skip`) and faults
(`Plan.fault`) exist for the write-tpu-kernels evidence rules and are never
used in production.
"""

import dataclasses

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

F32 = jnp.float32
COLLECTIVE_ID = 0  # repo convention for all-pairs-barrier kernels
_MESH = pl.DeviceIdType.MESH
ALGOS = ("flat2", "hier1", "hier2")
# Fault classes (test only). rotate_peer: a device sends the wrong column
# chunk (data meant for another owner). drop_rank: the last slot is left out
# of a sum. stale_self: one slot is counted twice and another not at all.
# drop_tail: the last row tile keeps a partial value instead of the sum.
FAULTS = ("rotate_peer", "drop_rank", "stale_self", "drop_tail")


@dataclasses.dataclass(frozen=True)
class Plan:
    n: int  # devices in the mesh
    t: int  # rows (tokens)
    h: int  # columns (hidden size)
    algo: str = "hier1"
    skip: str = ""  # removal arm (timing only): "ag", "ic", "d2d"
    fault: str | None = None
    interpret: bool = False

    @property
    def half(self) -> int:
        return self.h // 2

    @property
    def chips(self) -> int:
        return self.n // 2


def supported(n: int, t: int, h: int, algo: str, dtype=jnp.bfloat16) -> bool:
    """Whether the kernel handles [t, h] over n devices with this algorithm:
    whole sublane tiles of rows and lane-aligned column chunks."""
    lanes = 128
    tile_rows = 16 if jnp.dtype(dtype).itemsize == 2 else 8
    if n < 2 or t <= 0 or t % tile_rows or algo not in ALGOS:
        return False
    if algo == "flat2":
        return h % n == 0 and (h // n) % lanes == 0
    if n % 2 or (h // 2) % lanes:
        return False
    if algo == "hier2":
        c = n // 2
        return (h // 2) % c == 0 and ((h // 2) // c) % lanes == 0
    return True


def _barrier(n):
    sem = pltpu.get_barrier_semaphore()
    for d in range(n):
        pl.semaphore_signal(sem, 1, device_id=(d,), device_id_type=_MESH)
    pl.semaphore_wait(sem, n)


def _cols(ref, j, w):
    """Column block j (traced) of width w of a 2-D ref, as a DMA view."""
    return ref.at[:, pl.ds(pl.multiple_of(j * w, w), w)]


def _remote(src, dst, send_sem, recv_sem, dev):
    c = pltpu.make_async_remote_copy(
        src, dst, send_sem, recv_sem, device_id=(dev,), device_id_type=_MESH
    )
    c.start()
    return c


def _wait_recv(view, sem):
    """Wait until the bytes of `view` have arrived on `sem`."""
    pltpu.make_async_copy(view, view, sem).wait_recv()


def _row_blocks(t, cols):
    """Static row blocks of at most ~64 float32 vregs each."""
    rows = max(16, min(t, (64 * 8 * 128) // max(cols, 1) // 16 * 16))
    return [(r, min(rows, t - r)) for r in range(0, t, rows)]


def _sum_slots(slots, count, out, plan, tail_partial=None):
    """out = bf16(sum_{s < count} slots[s] in float32, slot order).
    Faults: drop_rank leaves the last slot out; stale_self counts slot 0 twice
    and slot 1 never; drop_tail writes `tail_partial` for the last row tile."""
    t, w = out.shape
    order = list(range(count))
    if plan.fault == "drop_rank" and count > 1:
        order = order[:-1]
    if plan.fault == "stale_self" and count > 1:
        order = [0, 0] + order[2:]
    for r0, rn in _row_blocks(t, w):
        acc = slots[order[0], pl.ds(r0, rn), :].astype(F32)
        for s in order[1:]:
            acc = acc + slots[s, pl.ds(r0, rn), :].astype(F32)
        out[pl.ds(r0, rn), :] = acc.astype(out.dtype)
    if plan.fault == "drop_tail" and tail_partial is not None:
        tr = 16
        out[pl.ds(t - tr, tr), :] = tail_partial[pl.ds(t - tr, tr), :]


# ---------------------------------------------------------------- flat2
def _flat2_scratch(p):
    w = p.h // p.n
    return [
        pltpu.VMEM((p.n, p.t, w), jnp.bfloat16),  # rs slots, slot s = from device s
        pltpu.VMEM((p.t, w), jnp.bfloat16),  # reduced chunk
        pltpu.SemaphoreType.DMA((p.n,)),  # rs send
        pltpu.SemaphoreType.DMA((p.n,)),  # rs recv, by sender
        pltpu.SemaphoreType.DMA((p.n,)),  # ag send
        pltpu.SemaphoreType.DMA((p.n,)),  # ag recv, by sender
        pltpu.SemaphoreType.DMA((2,)),  # local copies
    ]


def _flat2(p, me, x_ref, o_ref, slots, red, rs_s, rs_r, ag_s, ag_r, loc):
    n, w = p.n, p.h // p.n
    own = pltpu.make_async_copy(_cols(x_ref, me, w), slots.at[me], loc.at[0])
    own.start()  # local, needs no barrier
    _barrier(n)
    sends = []
    for step in range(1, n):
        tgt = lax.rem(me + step, n)
        chunk = lax.rem(tgt + 1, n) if p.fault == "rotate_peer" else tgt
        sends.append(
            _remote(
                _cols(x_ref, chunk, w), slots.at[me], rs_s.at[step], rs_r.at[me], tgt
            )
        )
    own.wait()
    for step in range(1, n):
        src = lax.rem(me + n - step, n)
        _wait_recv(slots.at[src], rs_r.at[src])
    _sum_slots(slots, n, red, p, tail_partial=slots.at[me])
    own_ag = pltpu.make_async_copy(red, _cols(o_ref, me, w), loc.at[1])
    own_ag.start()
    if p.skip != "ag":
        for step in range(1, n):
            tgt = lax.rem(me + step, n)
            sends.append(
                _remote(red, _cols(o_ref, me, w), ag_s.at[step], ag_r.at[me], tgt)
            )
        for step in range(1, n):
            src = lax.rem(me + n - step, n)
            _wait_recv(_cols(o_ref, src, w), ag_r.at[src])
    own_ag.wait()
    for c in sends:
        c.wait_send()


# ---------------------------------------------------------------- hier1 / hier2
def _hier_scratch(p):
    c, hf = p.chips, p.half
    s = [
        pltpu.VMEM((p.t, hf), jnp.bfloat16),  # own half of x
        pltpu.VMEM((1, p.t, hf), jnp.bfloat16),  # sibling's half (d2d rs)
        pltpu.VMEM((c, p.t, hf), jnp.bfloat16),  # chip-half sums, slot = chip
        pltpu.VMEM((p.t, hf), jnp.bfloat16),  # reduced half (hier1) / staging
    ]
    if p.algo == "hier2":
        q = hf // c
        s += [pltpu.VMEM((c, p.t, q), jnp.bfloat16), pltpu.VMEM((p.t, q), jnp.bfloat16)]
    s += [
        pltpu.SemaphoreType.DMA((4,)),  # d2d send / local copies
        pltpu.SemaphoreType.DMA((2,)),  # d2d recv: [0] rs, [1] ag
        pltpu.SemaphoreType.DMA(
            (2 * p.n,)
        ),  # inter-chip send; hier2 all-gather at n + step
        pltpu.SemaphoreType.DMA((p.n,)),  # inter-chip recv, by sender chip / device
        pltpu.SemaphoreType.DMA((p.n,)),  # hier2 all-gather recv, by sender device
    ]
    return s


def _hier(p, me, x_ref, o_ref, own_half, sib_half, chip_sums, res, *rest):
    if p.algo == "hier2":
        rs2, red2, dsem, drecv, ics, icr, agr = rest
    else:
        dsem, drecv, ics, icr, agr = rest
    n, nc, hf = p.n, p.chips, p.half
    k = lax.rem(me, 2)
    c = me // 2
    sib = lax.bitwise_xor(me, 1)
    own = pltpu.make_async_copy(_cols(x_ref, k, hf), own_half, dsem.at[1])
    own.start()
    _barrier(n)
    sends = []
    # 1. D2D reduce-scatter.
    if p.skip != "d2d":
        give = k if p.fault == "rotate_peer" else 1 - k
        sends.append(
            _remote(
                _cols(x_ref, give, hf), sib_half.at[0], dsem.at[0], drecv.at[0], sib
            )
        )
    own.wait()
    if p.skip != "d2d":
        _wait_recv(sib_half.at[0], drecv.at[0])
    t = p.t
    for r0, rn in _row_blocks(t, hf):
        a = own_half[pl.ds(r0, rn), :].astype(F32)
        if p.skip != "d2d":
            a = a + sib_half[0, pl.ds(r0, rn), :].astype(F32)
        chip_sums[c, pl.ds(r0, rn), :] = a.astype(chip_sums.dtype)
    if p.algo == "hier1":
        # 2. inter-chip one-shot among core k of every chip.
        if p.skip != "ic":
            for j in range(1, nc):
                tc = lax.rem(c + j, nc)
                sends.append(
                    _remote(
                        chip_sums.at[c],
                        chip_sums.at[c],
                        ics.at[j],
                        icr.at[c],
                        2 * tc + k,
                    )
                )
            for j in range(1, nc):
                sc = lax.rem(c + nc - j, nc)
                _wait_recv(chip_sums.at[sc], icr.at[sc])
            _sum_slots(chip_sums, nc, res, p, tail_partial=own_half)
        else:
            res[...] = chip_sums[0]
        # 3. D2D all-gather.
        mine = pltpu.make_async_copy(res, _cols(o_ref, k, hf), dsem.at[2])
        mine.start()
        if p.skip != "d2d":
            sends.append(
                _remote(res, _cols(o_ref, k, hf), dsem.at[3], drecv.at[1], sib)
            )
            _wait_recv(_cols(o_ref, 1 - k, hf), drecv.at[1])
        mine.wait()
    else:
        q = hf // nc
        # 2. two-shot among chips: reduce-scatter chunk j of my half to chip j.
        ownq = pltpu.make_async_copy(
            _cols(chip_sums.at[c], c, q), rs2.at[c], dsem.at[2]
        )
        ownq.start()
        for j in range(1, nc):
            tc = lax.rem(c + j, nc)
            sends.append(
                _remote(
                    _cols(chip_sums.at[c], tc, q),
                    rs2.at[c],
                    ics.at[j],
                    icr.at[c],
                    2 * tc + k,
                )
            )
        ownq.wait()
        for j in range(1, nc):
            sc = lax.rem(c + nc - j, nc)
            _wait_recv(rs2.at[sc], icr.at[sc])
        _sum_slots(rs2, nc, red2, p, tail_partial=rs2.at[c])
        # 3. the reduced chunk (columns k * hf + c * q) to every device.
        col = k * nc + c  # block index in units of q
        mine = pltpu.make_async_copy(red2, _cols(o_ref, col, q), dsem.at[3])
        mine.start()
        for step in range(1, n):
            tgt = lax.rem(me + step, n)
            sends.append(
                _remote(red2, _cols(o_ref, col, q), ics.at[n + step], agr.at[me], tgt)
            )
        for step in range(1, n):
            src = lax.rem(me + n - step, n)
            scol = lax.rem(src, 2) * nc + src // 2
            _wait_recv(_cols(o_ref, scol, q), agr.at[src])
        mine.wait()
    for s in sends:
        s.wait_send()


def vmem_bytes(p: Plan) -> int:
    """Upper bound of the kernel's VMEM: x, out and every scratch buffer
    (bf16), plus float32 temporaries of one row block."""
    b = 2 * p.t * p.h * 2
    if p.algo == "flat2":
        w = p.h // p.n
        b += (p.n + 1) * p.t * w * 2
    else:
        b += (3 + p.chips) * p.t * p.half * 2
        if p.algo == "hier2":
            b += (p.chips + 1) * p.t * (p.half // p.chips) * 2
    return b + 2 * 64 * 4096


def build(p: Plan):
    """pallas_call(rank int32[1], x bf16[t, h]) -> bf16[t, h] (the sum)."""
    if not supported(p.n, p.t, p.h, p.algo):
        raise ValueError(f"unsupported all-reduce {p}")
    body = _flat2 if p.algo == "flat2" else _hier
    scratch = _flat2_scratch(p) if p.algo == "flat2" else _hier_scratch(p)

    def kernel(rank_ref, x_ref, o_ref, *scr):
        body(p, rank_ref[0], x_ref, o_ref, *scr)

    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    skip = f"-skip_{p.skip}" if p.skip else ""
    limit = min(int(vmem_bytes(p) * 1.5) + (4 << 20), 60 << 20)
    extra = {"interpret": pltpu.InterpretParams()} if p.interpret else {}
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((p.t, p.h), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[vmem],
            out_specs=vmem,
            scratch_shapes=scratch,
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=COLLECTIVE_ID, vmem_limit_bytes=limit
        ),
        name=f"small_all_reduce-{p.algo}{skip}-t_{p.t}-h_{p.h}-n_{p.n}",
        **extra,
    )


def make_factory(algo, skip="", interpret=False, max_t=256):
    """test_accuracy / bench factory: make(n, t, h, fault=None) -> fn(x, rank),
    or None above max_t rows (the size dispatch keeps XLA's all-reduce there)."""

    def make(n, t, h, fault=None):
        if fault is not None and fault not in FAULTS:
            raise NotImplementedError(fault)
        if t > max_t or not supported(n, t, h, algo):
            return None
        call = build(Plan(n, t, h, algo, skip, fault, interpret))
        return lambda x, rank: call(rank.reshape(1).astype(jnp.int32), x)

    return make


def pairs_share_chip(devices) -> bool:
    """True when mesh positions (2c, 2c + 1) are the two TensorCores of one
    chip, judged from each device's `coords` and `core_on_chip`; False when
    the attributes are missing."""
    try:
        for i in range(0, len(devices) - 1, 2):
            a, b = devices[i], devices[i + 1]
            if tuple(a.coords) != tuple(b.coords) or {
                a.core_on_chip,
                b.core_on_chip,
            } != {0, 1}:
                return False
        return len(devices) % 2 == 0
    except AttributeError:
        return False
