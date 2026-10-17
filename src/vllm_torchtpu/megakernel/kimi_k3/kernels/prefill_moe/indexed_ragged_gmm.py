# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Indexed activation loads into the ragged GMM pipeline."""

import dataclasses

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.megakernel.kimi_k3.kernels import prefill_megakernel as k


class _PackedRows:
    """Load only the static row bucket selected by the GMM."""

    def __init__(self, ref):
        self.ref = ref

    def reshape(self, rows, width):
        assert rows == -1 and width == 2 * self.ref.shape[-1]
        return self

    def __getitem__(self, rows):
        return self.ref[rows].reshape(-1, 2 * self.ref.shape[-1])


def indexed_gate_up(latent, indices, rhs, sizes, scales, layer):
    """Packed [T,2,K/2] source; bounded token indices and per-expert route counts.

    Runs the GMM weight/scale/output pipeline and inner arithmetic
    (``prefill_megakernel.inner_kernel``) on rows gathered by sorted token
    index. All index reads and row DMAs are bounded by aligned valid metadata
    rows, including partial expert tiles.
    """
    rows, width = (indices.shape[0], latent.shape[1] * latent.shape[2])
    limit = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)
    offset = jnp.zeros((1,), jnp.int32)
    cfg = k.make_gmm_configs(
        jax.ShapeDtypeStruct((rows, width), latent.dtype),
        jax.ShapeDtypeStruct(rhs.shape[1:], rhs.dtype),
        jax.ShapeDtypeStruct(scales.shape[1:], scales.dtype),
        None,
        sizes,
        offset,
        tile_info=k.calculate_tiling,
        vmem_limit_bytes=limit,
        out_dtype=None,
        acc_dtype=None,
        maybe_quantize_lhs=False,
        zero_initialize=False,
        fuse_act="situ:4:25",
        rhs_quant_dtype=jnp.float4_e2m1fn,
    )
    tm, tn, tk = (cfg.tiles.tile_m, cfg.tiles.tile_n, cfg.tiles.tile_k)
    sub = cfg.dims.size_lhs_sublane
    if tk != width:
        raise ValueError(f"Indexed gate/up requires full K: {cfg.tiles}")
    max_gm = cfg.dims.size_group + pl.cdiv(rows, tm) - 1
    if rows % sub or latent.dtype != jnp.bfloat16 or indices.dtype != jnp.int32:
        raise ValueError("Expected sublane-aligned BF16 rows and int32 token indices")
    smem_bytes = indices.size * 4 + (2 * max_gm + 1 + sizes.size + 2) * 4
    if smem_bytes > pltpu.get_tpu_info().smem_capacity_bytes:
        raise ValueError("The sorted index array does not fit in SMEM")

    def kernel(
        counts, offset, layer, idx, x, weights, out, partial, acc, metadata, acts, sems
    ):
        weights = dataclasses.replace(
            weights,
            weight=weights.weight.at[layer[0]].bitcast(jnp.float4_e2m1fn),
            scale=weights.scale.at[layer[0]],
        )
        ng = k.fill_metadata(counts, offset, metadata, cfgs=cfg)
        (_, ws), ys = k.generate_block_specs(metadata, cfg)
        weights = k.FusedWeightsRef(
            gate=weights,
            up=jax.tree.map(lambda r: r.at[..., cfg.out_size_n :], weights),
        )
        ws = k.FusedWeightsRef(gate=ws, up=ws)

        def bounds(g):
            start = metadata.gm_id_to_m_offset[g] // sub * sub
            end = pl.cdiv(metadata.gm_id_to_m_offset[g + 1], sub) * sub
            return (start, end - start)

        def start(g, bank):
            begin, nrows = bounds(g)

            @pl.loop(0, nrows)
            def row(i):
                source = idx[begin + i]
                pltpu.make_async_copy(
                    x.at[pl.ds(source, 1)], acts.at[bank, pl.ds(i, 1)], sems.at[bank]
                ).start()

        def inner(w, y, partial, acc, metadata):
            g = pl.program_id(1)
            bank = g % 2

            @pl.when(g == 0)
            def initial():
                start(g, bank)

            _, nrows = bounds(g)
            active = acts.at[bank, pl.ds(0, nrows)]
            pltpu.make_async_copy(active, active, sems.at[bank]).wait()

            @pl.when(g + 1 < ng)
            def following():
                start(g + 1, 1 - bank)

            k.inner_kernel(
                _PackedRows(acts.at[bank]), w, y, partial, acc, metadata, cfgs=cfg
            )

        pltpu.emit_pipeline(
            inner,
            grid=(pl.cdiv(cfg.out_size_n, tn), ng, 1),
            in_specs=(ws,),
            out_specs=ys,
        )(
            weights,
            out.reshape(-1, sub, out.shape[-1]),
            scratches=(partial, acc, metadata),
        )

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((rows, cfg.out_size_n), cfg.out_dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            in_specs=(hbm, k.WeightsRef(weight=hbm, scale=hbm, bias=None)),
            out_specs=hbm,
            scratch_shapes=(
                pltpu.VMEM((sub, tn), cfg.out_dtype),
                pltpu.VMEM((tm, 2 * tn), cfg.acc_dtype),
                k.MetadataRef(
                    gm_id_to_group_id=pltpu.SMEM((max_gm,), jnp.int32),
                    gm_id_to_m_offset=pltpu.SMEM((max_gm + 1,), jnp.int32),
                ),
                pltpu.VMEM((2, tm, 2, width // 2), latent.dtype),
                pltpu.SemaphoreType.DMA((2,)),
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=limit, disable_bounds_checks=True
        ),
        cost_estimate=None,
        name=f"indexed_ragged_gu_{'prefetch'}" + "_bucket",
    )(
        sizes,
        offset,
        layer,
        indices,
        latent,
        k.WeightsRef(weight=rhs, scale=scales, bias=None),
    )
