# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

# Source provenance: inferact/k3-megakernel@30c9ba966b
# src/vllm_torchtpu/megakernel/kimi_k3/kernels/prefill_moe/indexed_ragged_gmm.py
# Source provenance: vllm_torchtpu/kernels/megablox/gmm_v2.py@644426d851 (modified copy)
"""Indexed activation loads into the ragged GMM pipeline."""

from __future__ import annotations

import dataclasses
from typing import cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .. import prefill_megakernel as k

_VMEM_CAPACITY_FRACTION = 0.9
_INT32_BYTES = 4


class PackedRows:
    """Load only the static row bucket selected by the GMM."""

    def __init__(self, ref: object) -> None:
        """Initialize the packed row wrapper around a VMEM activation reference."""
        self.ref: jax.Array = cast("jax.Array", ref)

    def reshape(self, rows: int, width: int) -> PackedRows:
        """Validate that the requested reshape matches the unpacked row width.

        Invariants:
            assert rows == -1 and width == 2 * self.ref.shape[-1]
        """
        expected_width = 2 * self.ref.shape[-1]
        if rows != -1 or width != expected_width:
            msg = f"Expected reshape (-1, {expected_width}), got ({rows}, {width})"
            raise ValueError(msg)
        return self

    def __getitem__(self, rows: slice) -> jax.Array:
        """Load the selected row slice and unpack the trailing sublane dimension."""
        return self.ref[rows].reshape(-1, 2 * self.ref.shape[-1])


def _prepare_gate_up_cfg(
    latent: jax.Array,
    indices: jax.Array,
    rhs: jax.Array,
    sizes: jax.Array,
    scales: jax.Array,
) -> tuple[k.GmmConfigs, int, int, jax.Array]:
    """Validate shapes and build the GMM configuration for indexed gate/up.

    Invariants:
        assert tk == width
        assert rows % sub == 0 and latent.dtype == jnp.bfloat16
        assert indices.dtype == jnp.int32
        assert smem_bytes <= pltpu.get_tpu_info().smem_capacity_bytes
    """
    rows, width = (indices.shape[0], latent.shape[1] * latent.shape[2])
    limit = int(pltpu.get_tpu_info().vmem_capacity_bytes * _VMEM_CAPACITY_FRACTION)
    offset = jnp.zeros((1,), jnp.int32)
    cfg = k.make_gmm_configs(
        jax.ShapeDtypeStruct((rows, width), latent.dtype),
        jax.ShapeDtypeStruct(rhs.shape[1:], rhs.dtype),
        jax.ShapeDtypeStruct(scales.shape[1:], scales.dtype),
        sizes,
        offset,
        tile_info=k.calculate_tiling,
        vmem_limit_bytes=limit,
        out_dtype=None,
        acc_dtype=None,
        fuse_act="situ:4:25",
        rhs_quant_dtype=jnp.float4_e2m1fn,
    )
    tm, _, tk = (cfg.tiles.tile_m, cfg.tiles.tile_n, cfg.tiles.tile_k)
    sub = cfg.dims.size_lhs_sublane
    if tk != width:
        msg = f"Indexed gate/up requires full K: {cfg.tiles}"
        raise ValueError(msg)
    max_gm = cfg.dims.size_group + pl.cdiv(rows, tm) - 1
    if rows % sub or latent.dtype != jnp.bfloat16 or indices.dtype != jnp.int32:
        msg = "Expected sublane-aligned BF16 rows and int32 token indices"
        raise ValueError(msg)
    smem_bytes = (
        indices.size * _INT32_BYTES + (2 * max_gm + 1 + sizes.size + 2) * _INT32_BYTES
    )
    if smem_bytes > pltpu.get_tpu_info().smem_capacity_bytes:
        msg = "The sorted index array does not fit in SMEM"
        raise ValueError(msg)
    return cfg, limit, max_gm, offset


def indexed_gate_up(
    latent: jax.Array,
    indices: jax.Array,
    rhs: jax.Array,
    sizes: jax.Array,
    *scales_and_layer: jax.Array | None,
) -> jax.Array:
    """Packed [T,2,K/2] source; bounded token indices and per-expert route counts.

    Runs the GMM weight/scale/output pipeline and inner arithmetic
    (``prefill_megakernel.inner_kernel``) on rows gathered by sorted token
    index. All index reads and row DMAs are bounded by aligned valid metadata
    rows, including partial expert tiles.

    Invariants:
        assert scales is not None
    """
    scales_opt, layer = scales_and_layer
    if scales_opt is None:
        msg = "Indexed gate/up requires E8M0 scales"
        raise ValueError(msg)
    rows, width = (indices.shape[0], latent.shape[1] * latent.shape[2])
    cfg, limit, max_gm, offset = _prepare_gate_up_cfg(
        latent, indices, rhs, sizes, scales_opt
    )
    tm, tn = (cfg.tiles.tile_m, cfg.tiles.tile_n)
    sub = cfg.dims.size_lhs_sublane

    def _slice_up_ref(ref: jax.Array) -> object:
        """Slice the upper half of the fused gate/up output dimension."""
        return ref.at[..., cfg.out_size_n :]

    def kernel(*refs: jax.Array) -> None:
        """Execute the indexed ragged gate/up Pallas pipeline."""
        (
            counts,
            offset_ref,
            layer_ref,
            idx,
            x,
            weights_obj,
            out,
            partial,
            acc,
            metadata_obj,
            acts,
            sems,
        ) = refs
        weights = cast("k.WeightsRef", weights_obj)
        metadata = cast("k.MetadataRef", metadata_obj)
        scale_ref = cast("jax.Array", weights.scale)
        weights = dataclasses.replace(
            weights,
            weight=weights.weight.at[layer_ref[0]].bitcast(jnp.float4_e2m1fn),
            scale=scale_ref.at[layer_ref[0]],
        )
        ng = k.fill_metadata(counts, offset_ref, metadata, cfgs=cfg)
        (_, ws), ys = k.generate_block_specs(metadata, cfg)
        fused_weights = k.FusedWeightsRef(
            gate=weights,
            up=jax.tree.map(_slice_up_ref, weights),
        )
        fused_ws = k.FusedWeightsRef(gate=ws, up=ws)

        def bounds(group_idx: jax.Array) -> tuple[jax.Array, jax.Array]:
            """Compute sublane-aligned row start and count for a group tile."""
            start_row = metadata.gm_id_to_m_offset[group_idx] // sub * sub
            end_row = pl.cdiv(metadata.gm_id_to_m_offset[group_idx + 1], sub) * sub
            return (start_row, end_row - start_row)

        def start(group_idx: jax.Array, bank: jax.Array) -> None:
            """Issue async row gathers from HBM into the target VMEM bank."""
            begin, nrows = bounds(group_idx)

            @pl.loop(0, nrows)
            def row(row_idx: jax.Array) -> None:
                """Issue a single-row async DMA copy for one sorted token."""
                source = idx[begin + row_idx]
                pltpu.make_async_copy(
                    x.at[pl.ds(source, 1)],
                    acts.at[bank, pl.ds(row_idx, 1)],
                    sems.at[bank],
                ).start()

        def inner(
            w: k.RhsRef,
            y: jax.Array,
            partial_ref: jax.Array,
            acc_ref: jax.Array,
            metadata_ref: k.MetadataRef,
        ) -> None:
            """Run one group tile of the double-buffered gate/up pipeline."""
            group_idx = pl.program_id(1)
            bank = group_idx % 2

            @pl.when(group_idx == 0)
            def initial() -> None:
                """Prefetch the first group tile into bank 0."""
                start(group_idx, bank)

            _, nrows = bounds(group_idx)
            active = acts.at[bank, pl.ds(0, nrows)]
            pltpu.make_async_copy(active, active, sems.at[bank]).wait()

            @pl.when(group_idx + 1 < ng)
            def following() -> None:
                """Prefetch the next group tile into the alternate bank."""
                start(group_idx + 1, 1 - bank)

            k.inner_kernel(
                PackedRows(acts.at[bank]),
                w,
                y,
                partial_ref,
                acc_ref,
                metadata_ref,
                cfgs=cfg,
            )

        pltpu.emit_pipeline(
            inner,
            grid=(pl.cdiv(cfg.out_size_n, tn), ng, 1),
            in_specs=(fused_ws,),
            out_specs=ys,
        )(
            fused_weights,
            out.reshape(-1, sub, out.shape[-1]),
            scratches=(partial, acc, metadata),
        )

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((rows, cfg.out_size_n), cfg.out_dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            in_specs=(hbm, k.WeightsRef(weight=hbm, scale=hbm)),
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
        name="indexed_ragged_gu_prefetch_bucket",
    )(
        sizes,
        offset,
        layer,
        indices,
        latent,
        k.WeightsRef(weight=rhs, scale=scales_opt),
    )
