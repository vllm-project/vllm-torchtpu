# SPDX-License-Identifier: Apache-2.0
"""KV fetch arms for the QSA block-sparse prefill kernel (pure DMA, no compute).

Walks the same active KV tiles as the attention kernel at served TP8 shapes
(see bench_prefill_vpu.py) with the kernel's double-buffered DMA loop and no
compute, so the time is the fetch path alone. Arms:

  seg128_4d  the kernel today: 128-row copies into a (kv,1,4,256) FP8 buffer
  page_4d    one copy per page-sized run (512 rows) into the same buffer
  seg128_2d  128-row copies of a (N, 1024) FP8 view into a (kv, 1024) buffer
  page_2d    page-sized copies, (N, 1024) view
  half_2d    page-sized copies of only one layer's [K, V] bytes (cols 0..511)
  none       no DMA (loop and semaphore floor)

Also prints the compiled HLO of the attention op, so the cache operand's HBM
layout (tiling, padding) is visible, and whether the (N, 1024) view is a free
bitcast or a copy.
"""

import argparse
import functools
import re
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

import bench_prefill_vpu as bpv
import vllm_torchtpu.kernels.qwen4_exp.block_sparse_gqa_prefill as bsp

PAGE = bpv.PAGE


def active_tiles(inputs, shape):
    rows, context, qh, qb, kv_tile, qt, dcp_world = shape
    blocks, lengths = np.asarray(inputs[1]), np.asarray(inputs[2])
    nblocks = rows // qb
    entries = inputs[3].shape[1]
    ntiles = (entries * PAGE + kv_tile - 1) // kv_tile
    tiles = np.zeros((nblocks, ntiles), np.int32)
    counts = np.zeros((nblocks,), np.int32)
    for b in range(nblocks):
        seen = set()
        for row in range(b * qb, (b + 1) * qb):
            length = int(lengths[row])
            toks = [g * 4 for g in blocks[row] if 0 <= g < length // 4]
            toks.append((length // 4) * 4)
            seen.update(t // kv_tile for t in toks)
        ordered = sorted(seen)
        tiles[b, : len(ordered)] = ordered
        counts[b] = len(ordered)
    return tiles, counts, ntiles


def _kernel(table_ref, tiles_ref, counts_ref, cache_ref, out_ref, buf0, buf1, sem0, sem1,
            *, mode, kv_tile, ppq, ntiles):
    block = pl.program_id(0)
    count = counts_ref[block]
    copy_rows = 128 if mode.startswith("seg128") else min(kv_tile, PAGE)

    def tile_of(slot):
        return tiles_ref[block * ntiles + jnp.minimum(slot, ntiles - 1)]

    def copies(tile, buf, sem):
        out = []
        if mode == "none":
            return out
        for offset in range(0, kv_tile, copy_rows):
            logical = tile * kv_tile + offset
            page = logical // PAGE
            in_page = logical % PAGE
            physical = table_ref[block * ppq + jnp.minimum(page, ppq - 1)]
            start = physical * PAGE + in_page
            if mode == "half_2d":
                src = cache_ref.at[pl.ds(start, copy_rows), pl.ds(0, 512)]
            else:
                src = cache_ref.at[pl.ds(start, copy_rows)]
            out.append(pltpu.make_async_copy(src, buf.at[pl.ds(offset, copy_rows)], sem))
        return out

    def start(slot, buf, sem):
        for c in copies(tile_of(slot), buf, sem):
            c.start()

    def wait(slot, buf, sem):
        for c in copies(tile_of(slot), buf, sem):
            c.wait()

    @pl.when(count > 0)
    def _():
        start(0, buf0, sem0)

    @pl.when(count > 1)
    def _():
        start(1, buf1, sem1)

    def pair(p, carry):
        slot = p * 2
        wait(slot, buf0, sem0)

        @pl.when(slot + 2 < count)
        def _():
            start(slot + 2, buf0, sem0)

        wait(slot + 1, buf1, sem1)

        @pl.when(slot + 3 < count)
        def _():
            start(slot + 3, buf1, sem1)

        return carry

    lax.fori_loop(0, count // 2, pair, 0)

    @pl.when(count % 2 != 0)
    def _():
        wait(count - 1, buf0, sem0)

    out_ref[...] = jnp.full(out_ref.shape, count.astype(jnp.float32))


def fetch_fn(mode, kv_tile, ppq, ntiles, nblocks, cache_shape):
    two_d = mode.endswith("_2d")
    width = 512 if mode == "half_2d" else 1024
    buf_shape = (kv_tile, width) if two_d else (kv_tile, 1, 4, 256)

    def run(table, tiles, counts, cache):
        if two_d:
            cache = cache.reshape(cache.shape[0] * cache.shape[1], 1024)
        else:
            cache = cache.reshape(cache.shape[0] * cache.shape[1], 1, 4, 256)
        return pl.pallas_call(
            functools.partial(_kernel, mode=mode, kv_tile=kv_tile, ppq=ppq, ntiles=ntiles),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=3,
                grid=(nblocks,),
                in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
                out_specs=pl.BlockSpec((1, 8, 128), lambda b, *_: (b, 0, 0)),
                scratch_shapes=[
                    pltpu.VMEM(buf_shape, jnp.float8_e4m3fn),
                    pltpu.VMEM(buf_shape, jnp.float8_e4m3fn),
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA,
                ],
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("arbitrary",), vmem_limit_bytes=62 * 1024**2
            ),
            out_shape=jax.ShapeDtypeStruct((nblocks, 8, 128), jnp.float32),
            name=f"kv_fetch_{mode}",
        )(table, tiles.reshape(-1), counts, cache)

    return jax.jit(run)


def print_layouts(inputs, shape):
    rows, context, qh, qb, kv_tile, qt, dcp_world = shape
    fn = jax.jit(
        functools.partial(
            bsp.block_sparse_gqa_prefill, ratio=4, kv_heads=1, query_block=qb,
            kv_tile=kv_tile, bf16_io=True, query_tile=qt,
        )
    )
    text = fn.lower(*inputs).compile().as_text()
    for line in text.splitlines():
        if "f8e4m3fn[" in line and ("custom-call" in line or "parameter(4)" in line
                                    or "copy" in line or "bitcast" in line):
            print("HLO", re.sub(r"\s+", " ", line.strip())[:400])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", default="r3072_c233k,r512_c233k")
    parser.add_argument("--modes", default="seg128_4d,page_4d,seg128_2d,page_2d,half_2d,none")
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()
    for shape_name in args.shapes.split(","):
        shape = bpv.SHAPES[shape_name]
        rows, context, qh, qb, kv_tile, qt, dcp_world = shape
        inputs, host = bpv.make_inputs(rows, context, qh, qb, dcp_world)
        if shape_name == args.shapes.split(",")[0]:
            print_layouts(inputs, shape)
        tiles, counts, ntiles = active_tiles(inputs, shape)
        nblocks = rows // qb
        table = jnp.asarray(np.tile(host[2], nblocks).astype(np.int32))
        ppq = host[2].shape[0]
        cache = inputs[4]
        visits = int(counts.sum())
        for mode in args.modes.split(","):
            try:
                fn = fetch_fn(mode, kv_tile, ppq, ntiles, nblocks, cache.shape)
                if mode == "page_2d":
                    hlo = fn.lower(table, jnp.asarray(tiles), jnp.asarray(counts), cache).compile().as_text()
                    ops = [l.strip()[:200] for l in hlo.splitlines()
                           if ("copy" in l or "fusion" in l) and "f8e4m3fn" in l]
                    print(f"HLO2D shape={shape_name} non-bitcast ops on the cache: {len(ops)}")
                    for l in ops[:4]:
                        print("HLO2D", l)
                us, _ = bpv.device_time(
                    fn, (table, jnp.asarray(tiles), jnp.asarray(counts), cache), args.iters
                )
                print(f"FETCH shape={shape_name} mode={mode} us={us:.1f} visits={visits} "
                      f"us_per_tile={us / max(visits, 1):.3f}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"FETCH shape={shape_name} mode={mode} FAILED {type(exc).__name__}: "
                      f"{str(exc)[:500]}", flush=True)


if __name__ == "__main__":
    main()
