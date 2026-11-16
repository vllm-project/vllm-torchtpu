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
"""Streamed decode MoE expert MLP: gate_up + act + down in one weight stream.

Decode-sized grouped matmuls read each hit expert's weights once and are
bound by that read. gmm_v2_mlp runs a pipeline whose grid step is one whole
expert block (9.4-19 MB), so every call exposes one block: the first block's
DMA before any compute and the last block's compute after the last DMA. At
decode that is ~11 us per MoE layer per core, as much as one expert.

This kernel streams every hit expert's gate_up rows and then its down rows as
one sequence of equal k-slices ("chunks", chunk_rows x N fp4 + its block
scales) through a ring of VMEM slots, slots - 1 chunks ahead. Only one chunk's
DMA is exposed at the start and one chunk's compute at the end, and the down
weights of an expert stream in under its gate_up compute.

Numerics equal gmm_v2's fp8 rhs feed (decode tiles, at most
FP8_RHS_MAX_BUCKET_M... per tile there; here any bucket, rows are independent):
per output column, the f32 sum over 128-deep quant blocks runs in the same
order with the same block products (bf16 lhs x fp8(fp4) rhs, f32 result times
the f32 block scale), carried in VMEM between chunks; the activation and the
bf16 roundings of h and out are the same. Rows outside the group are masked
the same way.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels import tight_vmem
from vllm_torchtpu.kernels.megablox import gmm_v2 as _gmm

_QB = 128  # quant block rows (NVFP4 requantized to block 128)
_NCOL = 256  # output columns per matmul (MXU width)
_SUB = 16  # bf16 sublane tiling of lhs / out rows
_ACT_COLS = 512  # columns per activation / output-merge step


def _buckets(max_rows: int) -> tuple[int, ...]:
    """Row buckets one expert's rows (plus <= _SUB - 1 alignment rows) use."""
    cap = _gmm.align_to(max_rows + _SUB - 1, _SUB)
    out = [b for b in (16, 32, 64, 128, 256, 512) if b < cap]
    return (*out, cap)


def parse_spec(raw: str) -> tuple[int, int, int] | None:
    """'<chunk_rows>:<slots>[@<max size_m>]' -> (chunk_rows, slots, max_m).

    'on' / '1' mean 768:4@512. Empty / '0' = off.
    """
    raw = (raw or "").strip()
    if raw in ("", "0", "off", "false", "False"):
        return None
    if raw in ("1", "on", "true", "True"):
        raw = "768:4@512"
    spec, _, max_m = raw.partition("@")
    parts = [int(x) for x in spec.split(":") if x.strip()]
    chunk = parts[0] if parts else 768
    slots = parts[1] if len(parts) > 1 else 4
    if chunk % _QB or slots < 2:
        raise ValueError(f"TPU_GMM_V2_DECODE_STREAM={raw!r}: chunk % 128, slots >= 2")
    return chunk, slots, int(max_m) if max_m else 512


def eligible(
    lhs, rhs1, rhs2, rhs1_scale, rhs2_scale, fuse_act, chunk_rows, max_m
) -> str | None:
    """None if the streamed kernel serves this call, else why not."""
    fp4 = jnp.float4_e2m1fn
    if fuse_act is None:
        return "no fused activation"
    if rhs1.dtype != fp4 or rhs2.dtype != fp4:
        return "rhs not fp4"
    if rhs1_scale is None or rhs2_scale is None:
        return "no scales"
    if lhs.dtype != jnp.bfloat16:
        return "lhs not bf16"
    size_m, k1 = lhs.shape
    g, k1w, n1 = rhs1.shape
    g2, f, n2 = rhs2.shape
    if size_m > max_m or size_m % _SUB:
        return "size_m"
    if k1w != k1 or g2 != g or n1 != 2 * f or n2 != n1:
        return "shapes"
    if k1 % chunk_rows or f % chunk_rows:
        return "chunk rows"
    if rhs1_scale.shape != (g, k1 // _QB, 1, n1) or rhs2_scale.shape != (
        g, f // _QB, 1, n2
    ):
        return "scale block"
    if n1 % _NCOL or f % _ACT_COLS:
        return "lanes"
    if not pltpu.get_tpu_info().is_matmul_supported(
        jnp.bfloat16, jnp.float8_e4m3fn
    ):
        return "no bf16 x fp8 matmul"
    return None


def _kernel(
    # Scalar prefetch
    gs_ref,  # int32[size_lhs_group]
    go_ref,  # int32[1]
    # In (HBM)
    lhs_hbm,  # [Mp, K1] bf16
    w1_hbm,  # [G, K1, N] fp4
    s1_hbm,  # [G, K1 / 128, 1, N] f32
    w2_hbm,  # [G, F, N] fp4
    s2_hbm,  # [G, F / 128, 1, N] f32
    # Out (HBM)
    out_hbm,  # [Mp, N] out dtype
    # Scratch
    z_ref,  # VMEM [Mp + MCAP, K1] bf16: lhs rows, then h at row HROW
    a_ref,  # VMEM [2 * MCAP, N] f32: gate_up acc rows [0, MCAP), down acc after
    o_ref,  # VMEM [Mp + MCAP, N] out dtype
    w_ref,  # VMEM [slots, chunk, N] fp4
    s_ref,  # VMEM [slots, chunk / 128, 1, N] f32
    hit_ref,  # SMEM [3, G] int32: group, row offset, rows of each hit group
    cnt_ref,  # SMEM [2] int32: hits, out rows written
    wsem,  # DMA [slots, 2]
    iosem,  # DMA [2]
    *,
    num_groups: int,
    chunk: int,
    slots: int,
    buckets: tuple[int, ...],
    fuse_act: str,
    stride0: bool = True,
):
    k1 = lhs_hbm.shape[1]
    n = w1_hbm.shape[2]
    f = w2_hbm.shape[1]
    mp = lhs_hbm.shape[0]
    mcap = buckets[-1]
    hrow = mp
    c1 = k1 // chunk
    ce = c1 + f // chunk
    kbc = chunk // _QB

    # Hit groups in row order, as fill_metadata lays them out.
    group_offset = go_ref[0]
    cnt_ref[0] = 0
    cnt_ref[1] = 0

    def scan(lg, m_off):
        g = lg - group_offset
        size = gs_ref[lg]

        @pl.when(jnp.logical_and(size > 0, g >= 0))
        def _():
            h = cnt_ref[0]
            hit_ref[0, h] = g
            hit_ref[1, h] = m_off
            hit_ref[2, h] = size
            cnt_ref[0] = h + 1

        return m_off + size

    end_rows = lax.fori_loop(0, group_offset + num_groups, scan, jnp.int32(0))
    num_hit = cnt_ref[0]
    num_chunks = num_hit * ce
    lhs_rows = pl.multiple_of(pl.cdiv(end_rows, _SUB) * _SUB, _SUB)

    def lhs_copy():
        return pltpu.make_async_copy(
            lhs_hbm.at[pl.ds(0, lhs_rows)], z_ref.at[pl.ds(0, lhs_rows)], iosem.at[0]
        )

    def chunk_copies(t, w1_side: bool):
        """The weight and scale copies of chunk t from rhs1 or rhs2."""
        slot = t % slots
        c = t % ce
        e = hit_ref[0, t // ce]
        if w1_side:
            w_hbm, s_hbm, r = w1_hbm, s1_hbm, jnp.where(c < c1, c, 0)
        else:
            w_hbm, s_hbm, r = w2_hbm, s2_hbm, jnp.where(c < c1, 0, c - c1)
        return (
            pltpu.make_async_copy(
                w_hbm.at[e, pl.ds(pl.multiple_of(r * chunk, chunk), chunk)],
                w_ref.at[slot],
                wsem.at[slot, 0],
            ),
            pltpu.make_async_copy(
                s_hbm.at[e, pl.ds(r * kbc, kbc)], s_ref.at[slot], wsem.at[slot, 1]
            ),
        )

    def start_chunk(t):
        is_w1 = t % ce < c1

        @pl.when(is_w1)
        def _():
            for cp in chunk_copies(t, True):
                cp.start()

        @pl.when(jnp.logical_not(is_w1))
        def _():
            for cp in chunk_copies(t, False):
                cp.start()

    def wait_chunk(t):
        # Same byte counts from either side.
        for cp in chunk_copies(t, True):
            cp.wait()

    def compute(t, mb):
        slot = t % slots
        j = t // ce
        c = t % ce
        m_off = hit_ref[1, j]
        rows = hit_ref[2, j]
        a = pl.multiple_of(m_off - m_off % _SUB, _SUB)
        is_w2 = c >= c1
        cc = jnp.where(is_w2, c - c1, c)
        row0 = pl.multiple_of(jnp.where(is_w2, hrow, a), _SUB)
        koff = pl.multiple_of(cc * chunk, _QB)
        arow0 = pl.multiple_of(jnp.where(is_w2, mcap, 0), _SUB)

        @pl.when(cc == 0)
        def _():
            a_ref[pl.ds(arow0, mb), :] = jnp.zeros((mb, n), jnp.float32)

        x = z_ref[pl.ds(row0, mb), pl.ds(koff, chunk)]
        w_slot = w_ref.at[slot]
        s_slot = s_ref.at[slot]
        fp8 = jnp.float8_e4m3fn
        for c0 in range(0, n, _NCOL):
            acc = a_ref[pl.ds(arow0, mb), pl.ds(c0, _NCOL)]
            for kb in range(kbc):
                w = w_slot[pl.ds(kb * _QB, _QB), pl.ds(c0, _NCOL)].astype(fp8)
                blk = jnp.matmul(
                    x[:, kb * _QB : (kb + 1) * _QB],
                    w,
                    preferred_element_type=jnp.float32,
                )
                if stride0:  # zero-stride sublane broadcast load
                    scale = s_slot[kb, pl.ds(0, mb, 0), pl.ds(c0, _NCOL)]
                else:  # interpret mode
                    scale = s_slot[kb, 0:1, pl.ds(c0, _NCOL)]
                blk = blk * scale
                acc = acc + blk
            a_ref[pl.ds(arow0, mb), pl.ds(c0, _NCOL)] = acc

        @pl.when(c == c1 - 1)
        def _():
            # h = act(gate, up), rounded to the lhs dtype (gmm_v2's h).
            for c0 in range(0, f, _ACT_COLS):
                gate = a_ref[0:mb, c0 : c0 + _ACT_COLS]
                up = a_ref[0:mb, f + c0 : f + c0 + _ACT_COLS]
                z_ref[hrow : hrow + mb, c0 : c0 + _ACT_COLS] = _gmm.gate_up_act(
                    gate, up, fuse_act
                ).astype(z_ref.dtype)

        @pl.when(c == ce - 1)
        def _():
            r = a + lax.broadcasted_iota(jnp.int32, (mb, _ACT_COLS), 0)
            keep = r < m_off
            mine = jnp.logical_and(m_off <= r, r < m_off + rows)
            for c0 in range(0, n, _ACT_COLS):
                y = a_ref[mcap : mcap + mb, c0 : c0 + _ACT_COLS].astype(o_ref.dtype)
                old = o_ref[pl.ds(a, mb), pl.ds(c0, _ACT_COLS)]
                o_ref[pl.ds(a, mb), pl.ds(c0, _ACT_COLS)] = jnp.where(
                    keep, old, jnp.where(mine, y, jnp.zeros_like(y))
                )

    branches = [functools.partial(lambda t, mb: compute(t, mb), mb=mb) for mb in buckets]

    def flush_out(j):
        # Rows below the next group's first slab are final: write them.
        m_end = hit_ref[1, j] + hit_ref[2, j]
        is_last = j == num_hit - 1
        final = jnp.where(is_last, pl.cdiv(m_end, _SUB), m_end // _SUB) * _SUB
        w0 = pl.multiple_of(cnt_ref[1], _SUB)

        @pl.when(final > w0)
        def _():
            size = pl.multiple_of(final - w0, _SUB)
            pltpu.make_async_copy(
                o_ref.at[pl.ds(w0, size)], out_hbm.at[pl.ds(w0, size)], iosem.at[1]
            ).start()
            cnt_ref[1] = final

    @pl.when(num_chunks > 0)
    def _():
        lhs_copy().start()
        for i in range(slots - 1):

            @pl.when(i < num_chunks)
            def _():
                start_chunk(i)

        lhs_copy().wait()

        def body(t, carry):
            nxt = t + slots - 1

            @pl.when(nxt < num_chunks)
            def _():
                start_chunk(nxt)

            wait_chunk(t)
            j = t // ce
            m_off = hit_ref[1, j]
            need = m_off % _SUB + hit_ref[2, j]
            bidx = sum((need > b).astype(jnp.int32) for b in buckets[:-1])
            if len(buckets) == 1:
                branches[0](t)
            else:
                lax.switch(bidx, branches, t)

            @pl.when(t % ce == ce - 1)
            def _():
                flush_out(j)

            return carry

        lax.fori_loop(0, num_chunks, body, 0)
        written = pl.multiple_of(cnt_ref[1], _SUB)
        pltpu.make_async_copy(
            o_ref.at[pl.ds(0, written)], out_hbm.at[pl.ds(0, written)], iosem.at[1]
        ).wait()


def vmem_bytes(size_m, k1, f, n, chunk, slots, mcap, out_itemsize=2) -> int:
    z = (size_m + mcap) * k1 * 2
    acc = 2 * mcap * n * 4
    o = (size_m + mcap) * n * out_itemsize
    w = slots * chunk * n // 2
    s = slots * (chunk // _QB) * 8 * n * 4  # [kb, 1, N] f32 pads to 8 sublanes
    return z + acc + o + w + s


def gmm_decode_stream_mlp(
    lhs: jax.Array,  # [Mp, K1] bf16, Mp % 16 == 0
    rhs1: jax.Array,  # [G, K1, 2F] fp4
    rhs2: jax.Array,  # [G, F, N] fp4
    group_sizes: jax.Array,  # int32[size_lhs_group]
    rhs1_scale: jax.Array,  # [G, K1/128, 1, 2F]
    rhs2_scale: jax.Array,  # [G, F/128, 1, N]
    group_offset: jax.Array,  # int32[1]
    *,
    fuse_act: str,
    out_dtype,
    chunk_rows: int,
    slots: int,
    max_group_size: int | None = None,
    interpret: bool = False,
) -> jax.Array:
    size_m, k1 = lhs.shape
    g, _, n = rhs1.shape
    f = rhs2.shape[1]
    max_rows = min(max_group_size or size_m, size_m)
    buckets = _buckets(max_rows)
    mcap = buckets[-1]
    out_dtype = jnp.dtype(out_dtype or lhs.dtype)
    need = vmem_bytes(size_m, k1, f, n, chunk_rows, slots, mcap, out_dtype.itemsize)
    capacity = 64 << 20 if interpret else pltpu.get_tpu_info().vmem_capacity_bytes
    vmem_limit = min(
        int(capacity * 0.9),
        _gmm.align_to(need + (6 << 20), 1 << 20),
    )
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    kernel = functools.partial(
        _kernel,
        num_groups=g,
        chunk=chunk_rows,
        slots=slots,
        buckets=buckets,
        fuse_act=fuse_act,
        stride0=not interpret,
    )
    w_bytes = g * (k1 * n + f * n) // 2 + g * (k1 + f) // _QB * n * 4
    call = pl.pallas_call if interpret else tight_vmem.pallas_call
    extra = {"interpret": pltpu.InterpretParams()} if interpret else {}
    return call(
        kernel,
        **extra,
        out_shape=jax.ShapeDtypeStruct((size_m, n), out_dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[hbm, hbm, hbm, hbm, hbm],
            out_specs=hbm,
            scratch_shapes=[
                pltpu.VMEM((size_m + mcap, k1), lhs.dtype),
                pltpu.VMEM((2 * mcap, n), jnp.float32),
                pltpu.VMEM((size_m + mcap, n), out_dtype),
                pltpu.VMEM((slots, chunk_rows, n), rhs1.dtype),
                pltpu.VMEM((slots, chunk_rows // _QB, 1, n), jnp.float32),
                pltpu.SMEM((3, g), jnp.int32),
                pltpu.SMEM((2,), jnp.int32),
                pltpu.SemaphoreType.DMA((slots, 2)),
                pltpu.SemaphoreType.DMA((2,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit,
            disable_bounds_checks=True,
        ),
        name=f"gmm_decode_stream-g_{g}-m_{size_m}-k_{k1}-f_{f}-c_{chunk_rows}x{slots}"
        f"-b_{'_'.join(map(str, buckets))}",
        cost_estimate=pl.CostEstimate(
            flops=2 * size_m * (k1 * n + f * n),
            bytes_accessed=w_bytes + size_m * (k1 + n) * 2,
            transcendentals=0,
        ),
    )(
        group_sizes,
        group_offset,
        lhs,
        rhs1,
        rhs1_scale.astype(jnp.float32),
        rhs2,
        rhs2_scale.astype(jnp.float32),
    )
