# SPDX-License-Identifier: Apache-2.0
"""Optional prefill HyperConnection mixer with resident weights and two tiles.

The kernel keeps the checkpoint's BF16 rounding boundaries. It stages the
normalization and both projection weights once, overlaps the next input tile
with the current computation, and drains output tiles only before reuse.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp.hc_prefill_fused import (
    NORM_IMPLS,
    branch_variance,
    norm_epilogue,
    norm_rows_per_step,
)


def _kernel(
    x_hbm,
    weight_hbm,
    down_hbm,
    up_hbm,
    mixed_hbm,
    norm_hbm,
    x_vm,
    weight_vm,
    down_vm,
    up_vm,
    mixed_vm,
    norm_vm,
    x_sems,
    weight_sem,
    down_sem,
    up_sem,
    mixed_sems,
    norm_sems,
    *,
    token_tile,
    tokens,
    hidden,
    branches,
    eps,
    norm_impl="vpu",
):
    weight_dma = pltpu.make_async_copy(weight_hbm, weight_vm, weight_sem)
    down_dma = pltpu.make_async_copy(down_hbm, down_vm, down_sem)
    up_dma = pltpu.make_async_copy(up_hbm, up_vm, up_sem)
    weight_dma.start()
    down_dma.start()
    up_dma.start()
    pltpu.make_async_copy(
        x_hbm.at[pl.ds(0, token_tile)], x_vm.at[0], x_sems.at[0]
    ).start()

    num_full = tokens // token_tile
    # TPU_QWEN4_HC_PREFILL_PARTIAL_TILE: a last tile of `remainder` rows (a
    # multiple of 32; e.g. 3,200 = 12 x 256 + 128). Its copies and waits use
    # the remainder's byte count; every other step is unchanged, and a whole
    # tile count emits the same program as before.
    remainder = tokens % token_tile

    def rows_of(ref, parity, rows):
        return ref.at[parity] if rows == token_tile else ref.at[parity, pl.ds(0, rows)]

    def one_step(step, parity, rows=token_tile):
        # The previous step started this input copy. Wait at its first read.
        pltpu.make_async_copy(
            rows_of(x_vm, parity, rows), rows_of(x_vm, parity, rows), x_sems.at[parity]
        ).wait()

        @pl.when(step + 1 < num_full)
        def _start_next_input():
            pltpu.make_async_copy(
                x_hbm.at[pl.ds((step + 1) * token_tile, token_tile)],
                x_vm.at[1 - parity],
                x_sems.at[1 - parity],
            ).start()

        if remainder and rows == token_tile:

            @pl.when(step + 1 == num_full)
            def _start_partial_input():
                pltpu.make_async_copy(
                    x_hbm.at[pl.ds((step + 1) * token_tile, remainder)],
                    x_vm.at[1 - parity, pl.ds(0, remainder)],
                    x_sems.at[1 - parity],
                ).start()

        if token_tile >= 32 and token_tile % 32 == 0:

            @pl.when(step == 0)
            def _wait_weight():
                weight_dma.wait()

            @pl.when(step >= 2)
            def _wait_prior_norm_output():
                pltpu.make_async_copy(
                    norm_vm.at[parity], norm_vm.at[parity], norm_sems.at[parity]
                ).wait()

            step_rows = norm_rows_per_step(norm_impl, rows)

            def normalize_rows(index, carry):
                offset = index * step_rows
                weight = weight_vm[...].astype(jnp.float32)
                for branch in range(branches):
                    start = branch * hidden
                    xb = x_vm[parity, pl.ds(offset, step_rows), pl.ds(start, hidden)]
                    values = xb.astype(jnp.float32)
                    variance = branch_variance(values, xb, hidden, norm_impl)
                    if norm_impl.startswith("bf16ep"):
                        norm_vm[
                            parity, pl.ds(offset, step_rows), pl.ds(start, hidden)
                        ] = norm_epilogue(
                            values,
                            variance,
                            None,
                            weight_vm[pl.ds(start, hidden)][None, :],
                            eps,
                            norm_impl,
                        )
                        continue
                    normalized = (
                        (values * jax.lax.rsqrt(variance + eps))
                        .astype(jnp.bfloat16)
                        .astype(jnp.float32)
                    )
                    scaled = (
                        (normalized * weight[None, start : start + hidden])
                        .astype(jnp.bfloat16)
                        .astype(jnp.float32)
                    )
                    norm_vm[parity, pl.ds(offset, step_rows), pl.ds(start, hidden)] = (
                        normalized + scaled
                    ).astype(jnp.bfloat16)
                return carry

            jax.lax.fori_loop(0, rows // step_rows, normalize_rows, jnp.int32(0))
            norm = norm_vm[parity] if rows == token_tile else norm_vm[parity, pl.ds(0, rows)]
        else:
            x = x_vm[parity].astype(jnp.float32)

            @pl.when(step == 0)
            def _wait_weight():
                weight_dma.wait()

            weight = weight_vm[...].astype(jnp.float32)
            pieces = []
            for branch in range(branches):
                start = branch * hidden
                values = x[:, start : start + hidden]
                variance = jnp.mean(values * values, axis=-1, keepdims=True)
                normalized = (
                    (values * jax.lax.rsqrt(variance + eps))
                    .astype(jnp.bfloat16)
                    .astype(jnp.float32)
                )
                scaled = (
                    (normalized * weight[None, start : start + hidden])
                    .astype(jnp.bfloat16)
                    .astype(jnp.float32)
                )
                pieces.append(normalized + scaled)
            norm = jnp.concatenate(pieces, axis=-1).astype(jnp.bfloat16)

            @pl.when(step >= 2)
            def _wait_prior_norm_output():
                pltpu.make_async_copy(
                    norm_vm.at[parity], norm_vm.at[parity], norm_sems.at[parity]
                ).wait()

            norm_vm[parity] = norm
        pltpu.make_async_copy(
            rows_of(norm_vm, parity, rows),
            norm_hbm.at[pl.ds(step * token_tile, rows)],
            norm_sems.at[parity],
        ).start()

        @pl.when(step == 0)
        def _wait_down():
            down_dma.wait()

        down = jax.lax.dot_general(
            norm,
            down_vm[...],
            dimension_numbers=(((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        ).astype(jnp.bfloat16)
        gate = (down / branches).astype(jnp.bfloat16)
        gate = jax.nn.silu(gate.astype(jnp.float32)).astype(jnp.bfloat16)

        @pl.when(step == 0)
        def _wait_up():
            up_dma.wait()

        logits = (
            jax.lax.dot_general(
                gate,
                up_vm[...],
                dimension_numbers=(((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            .astype(jnp.bfloat16)
            .astype(jnp.float32)
        )
        weighted = jax.nn.sigmoid(logits) * norm.astype(jnp.float32)
        mixed = weighted[:, :hidden]
        for branch in range(1, branches):
            start = branch * hidden
            mixed = mixed + weighted[:, start : start + hidden]

        @pl.when(step >= 2)
        def _wait_prior_mixed_output():
            pltpu.make_async_copy(
                mixed_vm.at[parity], mixed_vm.at[parity], mixed_sems.at[parity]
            ).wait()

        if rows == token_tile:
            mixed_vm[parity] = (mixed / branches).astype(jnp.bfloat16)
        else:
            mixed_vm[parity, pl.ds(0, rows)] = (mixed / branches).astype(jnp.bfloat16)
        pltpu.make_async_copy(
            rows_of(mixed_vm, parity, rows),
            mixed_hbm.at[pl.ds(step * token_tile, rows)],
            mixed_sems.at[parity],
        ).start()

    def step_pair(pair, carry):
        one_step(2 * pair, 0)
        one_step(2 * pair + 1, 1)
        return carry

    jax.lax.fori_loop(0, num_full // 2, step_pair, jnp.int32(0))
    if num_full % 2:
        # An odd tile count (TPU_QWEN4_HC_PREFILL_ODD_TILES, e.g. 6,400 rows
        # in 256-row tiles): the last tile runs on parity 0. Its input copy was
        # started by the previous (parity 1) step and its waits pair with the
        # step two back, as inside the loop. Even counts emit no extra code.
        one_step(jnp.int32(num_full - 1), 0)
    if remainder:
        one_step(jnp.int32(num_full), num_full % 2, remainder)
    # Drain: each parity's last output copy; the partial step's are smaller.
    last_rows = {parity: token_tile for parity in range(2)}
    if remainder:
        last_rows[num_full % 2] = remainder
    for parity in range(2):
        pltpu.make_async_copy(
            rows_of(norm_vm, parity, last_rows[parity]),
            rows_of(norm_vm, parity, last_rows[parity]),
            norm_sems.at[parity],
        ).wait()
        pltpu.make_async_copy(
            rows_of(mixed_vm, parity, last_rows[parity]),
            rows_of(mixed_vm, parity, last_rows[parity]),
            mixed_sems.at[parity],
        ).wait()


@functools.partial(
    jax.jit,
    static_argnames=(
        "branches", "hidden", "eps", "token_tile", "interpret", "partial", "norm_impl"
    ),
)
def _full_mix(
    x,
    weight,
    down,
    up,
    *,
    branches,
    hidden,
    eps,
    token_tile,
    interpret=False,
    partial=False,
    norm_impl="vpu",
):
    tokens, width = x.shape
    if norm_impl not in NORM_IMPLS:
        raise ValueError(f"unknown HC norm_impl {norm_impl!r}")
    lowrank = down.shape[0]
    # At least two whole tiles: the drain below waits on both parities. With
    # `partial`, a last tile of any multiple of 32 rows.
    if (
        (tokens % 32 or token_tile % 32 if partial else tokens % token_tile)
        or tokens < 2 * token_tile
        or width != branches * hidden
        or weight.shape != (width,)
        or down.shape != (lowrank, width)
        or up.shape != (width, lowrank)
        or any(a.dtype != jnp.bfloat16 for a in (x, weight, down, up))
    ):
        raise ValueError("HC full mix requires even BF16 token tiles")
    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    return pl.pallas_call(
        functools.partial(
            _kernel,
            token_tile=token_tile,
            tokens=tokens,
            hidden=hidden,
            branches=branches,
            eps=eps,
            norm_impl=norm_impl,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(),
            in_specs=(hbm,) * 4,
            out_specs=(hbm,) * 2,
            scratch_shapes=(
                pltpu.VMEM((2, token_tile, width), jnp.bfloat16),
                pltpu.VMEM((width,), jnp.bfloat16),
                pltpu.VMEM((lowrank, width), jnp.bfloat16),
                pltpu.VMEM((width, lowrank), jnp.bfloat16),
                pltpu.VMEM((2, token_tile, hidden), jnp.bfloat16),
                pltpu.VMEM((2, token_tile, width), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
            ),
        ),
        out_shape=(
            jax.ShapeDtypeStruct((tokens, hidden), jnp.bfloat16),
            jax.ShapeDtypeStruct((tokens, width), jnp.bfloat16),
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=60 << 20),
        interpret=interpret,
        name=f"qwen4exp_hc_branchstore32_m{token_tile}",
    )(x, weight, down, up)
