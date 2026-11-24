"""One Mosaic call for fused EP MoE v2's combine at small token counts.

Below kernel.COMBINE_MIN_TOKENS the combine is an XLA epilogue of about 19
ops: a gather that stages the whole arrival buffer into VMEM, a second HBM
round for the scale mirror with a one-hot matmul pick, relayouts of the index
tables and the weighted sum. This call does the same work with one HBM round:
it copies the whole arrival buffer and the whole scale mirror HBM->VMEM with
one DMA each, started together with local DMAs of the three [t, topk] tables
VMEM->SMEM, then picks rows and scales by index in VMEM. The result is the same
as layer._combine_arrivals' XLA path, bit for bit.
"""

import threading

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

LANES = 128
# Tokens per reduction loop body, as the large-t combine kernel uses.
CHUNK = 16
# Tokens per bf16 output relayout: the sublanes of one packed bf16 tile.
GROUP = 16
# f32 sublanes of one vreg; a token's topk scale rows are gathered into one.
SUBLANES = 8
# Measured at t_local 16 and 32 only (decode); larger buckets keep XLA's path.
MAX_TOKENS = 32


def fits(*, t_local, topk, tile_m, scale_dtype):
    """Whether the one-call combine applies to this shape."""
    return (
        t_local <= MAX_TOKENS
        and not t_local % CHUNK
        and not t_local % GROUP
        and topk <= SUBLANES
        and tile_m == LANES
        and scale_dtype == jnp.float32
    )


def _build(
    *,
    t_local,
    topk,
    recv_rows,
    lane_blocks,
    mirror_sublanes,
    tile_m,
    wire_dtype,
    out_dtype,
):
    hidden = lane_blocks * LANES
    shift = tile_m.bit_length() - 1
    # Same clamps as the XLA path: a gather clamps to the last index, a VMEM
    # read does not.
    last_row, last_scale = recv_rows - 1, mirror_sublanes * tile_m - 1

    def kernel(
        pos_vm,
        scl_vm,
        w_vm,
        arr_hbm,
        mir_hbm,
        out_hbm,
        pos_sm,
        scl_sm,
        w_sm,
        rows_vm,
        mir_vm,
        coef_vm,
        tile_vm,
        out_vm,
        sems,
    ):
        # Tables (needed first), the mirror and the arrival buffer all in flight
        # together: one HBM round instead of the XLA path's two.
        copies = [
            pltpu.make_async_copy(src, dst, sems.at[i])
            for i, (src, dst) in enumerate(
                (
                    (pos_vm, pos_sm),
                    (scl_vm, scl_sm),
                    (w_vm, w_sm),
                    (mir_hbm, mir_vm),
                    (arr_hbm, rows_vm),
                )
            )
        ]
        for c in copies:
            c.start()
        for c in copies[:4]:
            c.wait()

        # Coefficients row_scale * weight, one token per pass, while the
        # arrival buffer streams in. The lane is picked on raw int32 bits (a
        # masked integer sum of one nonzero word is that word), so NaN payloads
        # of unwritten mirror entries and -0.0 survive as in the served lookup.
        lane_iota = lax.broadcasted_iota(jnp.int32, (SUBLANES, tile_m), 1)
        sub_iota = lax.broadcasted_iota(jnp.int32, (SUBLANES, tile_m), 0)

        def coef_token(i, carry):
            want = jnp.zeros((SUBLANES, tile_m), jnp.int32)
            words = jnp.zeros((SUBLANES, tile_m), jnp.int32)
            for k in range(topk):
                s = jnp.minimum(scl_sm[i, k], last_scale)
                row = lax.bitcast_convert_type(
                    mir_vm[pl.ds(s >> shift, 1), :], jnp.int32
                )
                words = jnp.where(
                    sub_iota == k, jnp.broadcast_to(row, (SUBLANES, tile_m)), words
                )
                want = jnp.where(sub_iota == k, s & (tile_m - 1), want)
            bits = jnp.sum(
                jnp.where(lane_iota == want, words, 0), axis=1, keepdims=True
            )
            scale = jnp.broadcast_to(
                lax.bitcast_convert_type(bits, jnp.float32), (SUBLANES, LANES)
            )
            for k in range(topk):
                coef_vm[i * topk + k] = scale[k : k + 1] * w_sm[i, k]
            return carry

        lax.fori_loop(0, t_local, coef_token, 0)
        copies[4].wait()

        # Weighted sum in the arrival row's [lane blocks, 128] layout, slots
        # k = 0..topk-1 added left to right in f32 as the XLA path's sum.
        def chunk(c, carry):
            for u in range(CHUNK):
                i = c * CHUNK + u

                def term(k, i=i):
                    r = jnp.minimum(pos_sm[i, k], last_row)
                    return rows_vm[r].astype(jnp.float32) * coef_vm[i * topk + k]

                acc = term(0)
                for k in range(1, topk):
                    acc = acc + term(k)
                tile_vm[i] = acc.astype(out_dtype)
            return carry

        lax.fori_loop(0, t_local // CHUNK, chunk, 0)
        # [GROUP, lane blocks, 128] -> [GROUP, hidden] relayout, then one store.
        for g in range(t_local // GROUP):
            out_vm[pl.ds(g * GROUP, GROUP)] = tile_vm[pl.ds(g * GROUP, GROUP)].reshape(
                GROUP, hidden
            )
        store = pltpu.make_async_copy(out_vm, out_hbm, sems.at[5])
        store.start()
        store.wait()

    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    # The tables come from small XLA ops; as VMEM operands those ops write VMEM
    # (~0.01 us each) instead of HBM (~0.65 us each, measured in serving).
    vmem = pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM)
    call = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[vmem] * 3 + [hbm] * 2,
            out_specs=hbm,
            scratch_shapes=(
                pltpu.SMEM((t_local, topk), jnp.int32),
                pltpu.SMEM((t_local, topk), jnp.int32),
                pltpu.SMEM((t_local, topk), jnp.float32),
                pltpu.VMEM((recv_rows, lane_blocks, LANES), wire_dtype),
                pltpu.VMEM((mirror_sublanes, tile_m), jnp.float32),
                pltpu.VMEM((t_local * topk, 1, LANES), jnp.float32),
                pltpu.VMEM((t_local, lane_blocks, LANES), out_dtype),
                pltpu.VMEM((t_local, hidden), out_dtype),
                pltpu.SemaphoreType.DMA((6,)),
            ),
            grid=(),
        ),
        out_shape=jax.ShapeDtypeStruct((t_local, hidden), out_dtype),
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
        name=f"moe_v2_combine_one_call_t{t_local}_k{topk}",
    )

    def combine(arrivals, arrival_scales, pos, mirror_pos, topk_weights):
        # The two buffers are DMA'd from HBM inside the kernel; pinning them
        # there keeps XLA from staging them into VMEM.
        pin = lambda a: pltpu.with_memory_space_constraint(a, pltpu.MemorySpace.HBM)  # noqa: E731
        return call(
            pos.astype(jnp.int32),
            mirror_pos.astype(jnp.int32),
            topk_weights.astype(jnp.float32),
            pin(arrivals),
            pin(arrival_scales),
        )

    return combine


_CACHE = {}
_CACHE_LOCK = threading.Lock()


def build_one_call_combine(**kwargs):
    """The combine fn(arrivals, arrival_scales, pos, mirror_pos, topk_weights)
    -> out_dtype[t_local, hidden], memoized by shape."""
    key = tuple(sorted((k, str(v)) for k, v in kwargs.items()))
    with _CACHE_LOCK:
        if key not in _CACHE:
            _CACHE[key] = _build(**kwargs)
        return _CACHE[key]
