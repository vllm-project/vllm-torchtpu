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


def _kernel(*refs, token_tile, tokens, hidden, branches, eps, up_t, combine):
    if combine:
        (x_hbm, block_hbm, gates_hbm, weight_hbm, down_hbm, up_hbm) = refs[:6]
        (mixed_hbm, norm_hbm, combined_hbm) = refs[6:9]
        (x_vm, block_vm, gates_vm, weight_vm, down_vm, up_vm, mixed_vm, norm_vm) = refs[
            9:17
        ]
        (x_sems, block_sems, gates_sems, weight_sem, down_sem, up_sem) = refs[17:23]
        (mixed_sems, norm_sems, combined_sems) = refs[23:]
    else:
        (x_hbm, weight_hbm, down_hbm, up_hbm, mixed_hbm, norm_hbm) = refs[:6]
        (x_vm, weight_vm, down_vm, up_vm, mixed_vm, norm_vm) = refs[6:12]
        (x_sems, weight_sem, down_sem, up_sem, mixed_sems, norm_sems) = refs[12:]
    weight_dma = pltpu.make_async_copy(weight_hbm, weight_vm, weight_sem)
    down_dma = pltpu.make_async_copy(down_hbm, down_vm, down_sem)
    up_dma = pltpu.make_async_copy(up_hbm, up_vm, up_sem)
    weight_dma.start()
    down_dma.start()
    up_dma.start()

    def input_copies(step, slot):
        rows = pl.ds(step * token_tile, token_tile)
        copies = [pltpu.make_async_copy(x_hbm.at[rows], x_vm.at[slot], x_sems.at[slot])]
        if combine:
            copies.append(
                pltpu.make_async_copy(
                    block_hbm.at[rows], block_vm.at[slot], block_sems.at[slot]
                )
            )
            copies.append(
                pltpu.make_async_copy(
                    gates_hbm.at[rows], gates_vm.at[slot], gates_sems.at[slot]
                )
            )
        return copies

    for copy in input_copies(0, 0):
        copy.start()

    def combine_rows(parity):
        """x tile <- bf16(residual + block * gate) per branch (the gated residual)."""

        def rows(index, carry):
            offset = pl.ds(index * 32, 32)
            block = block_vm[parity, offset, :].astype(jnp.float32)
            gates = gates_vm[parity, offset, :]
            for branch in range(branches):
                cols = pl.ds(branch * hidden, hidden)
                residual = x_vm[parity, offset, cols].astype(jnp.float32)
                x_vm[parity, offset, cols] = (
                    residual + block * gates[:, branch : branch + 1]
                ).astype(jnp.bfloat16)
            return carry

        jax.lax.fori_loop(0, token_tile // 32, rows, jnp.int32(0))

    def one_step(step, parity):
        # The previous step started this input copy. Wait at its first read.
        for copy in input_copies(step, parity):
            copy.wait()

        if combine:

            @pl.when(step >= 1)
            def _wait_prior_combined_output():
                # The next input copy reuses the slot the previous step wrote out.
                pltpu.make_async_copy(
                    x_vm.at[1 - parity],
                    x_vm.at[1 - parity],
                    combined_sems.at[1 - parity],
                ).wait()

        @pl.when(step + 1 < tokens // token_tile)
        def _start_next_input():
            for copy in input_copies(step + 1, 1 - parity):
                copy.start()

        if combine:
            combine_rows(parity)
            pltpu.make_async_copy(
                x_vm.at[parity],
                combined_hbm.at[pl.ds(step * token_tile, token_tile)],
                combined_sems.at[parity],
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

            def normalize_rows(index, carry):
                offset = index * 32
                weight = weight_vm[...].astype(jnp.float32)
                for branch in range(branches):
                    start = branch * hidden
                    values = x_vm[
                        parity, pl.ds(offset, 32), pl.ds(start, hidden)
                    ].astype(jnp.float32)
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
                    norm_vm[parity, pl.ds(offset, 32), pl.ds(start, hidden)] = (
                        normalized + scaled
                    ).astype(jnp.bfloat16)
                return carry

            jax.lax.fori_loop(0, token_tile // 32, normalize_rows, jnp.int32(0))
            norm = norm_vm[parity]
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
            norm_vm.at[parity],
            norm_hbm.at[pl.ds(step * token_tile, token_tile)],
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
                dimension_numbers=(((1,), (0 if up_t else 1,)), ((), ())),
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

        mixed_vm[parity] = (mixed / branches).astype(jnp.bfloat16)
        pltpu.make_async_copy(
            mixed_vm.at[parity],
            mixed_hbm.at[pl.ds(step * token_tile, token_tile)],
            mixed_sems.at[parity],
        ).start()

    def step_pair(pair, carry):
        one_step(2 * pair, 0)
        one_step(2 * pair + 1, 1)
        return carry

    jax.lax.fori_loop(0, tokens // (2 * token_tile), step_pair, jnp.int32(0))
    for parity in range(2):
        pltpu.make_async_copy(
            norm_vm.at[parity], norm_vm.at[parity], norm_sems.at[parity]
        ).wait()
        pltpu.make_async_copy(
            mixed_vm.at[parity], mixed_vm.at[parity], mixed_sems.at[parity]
        ).wait()
    if combine:
        # The last step (odd parity) wrote the final combined tile.
        pltpu.make_async_copy(x_vm.at[1], x_vm.at[1], combined_sems.at[1]).wait()


def _pallas_mix(x, weight, down, up, extra, *, branches, hidden, eps, token_tile, up_t):
    """The full mix over ``x`` (``extra``: (block, gates) to combine into it first)."""
    tokens, width = x.shape
    lowrank = down.shape[0]
    if (
        tokens % (2 * token_tile)
        or width != branches * hidden
        or weight.shape != (width,)
        or down.shape != (lowrank, width)
        or up.shape != ((lowrank, width) if up_t else (width, lowrank))
        or any(a.dtype != jnp.bfloat16 for a in (x, weight, down, up))
    ):
        raise ValueError("HC full mix requires even BF16 token tiles")
    combine = bool(extra)
    if combine:
        block, gates = extra
        if (
            block.shape != (tokens, hidden)
            or gates.shape != (tokens, branches)
            or block.dtype != jnp.bfloat16
            or gates.dtype != jnp.float32
        ):
            raise ValueError("HC combine requires a BF16 block and FP32 gates")
    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    dma = pltpu.SemaphoreType.DMA
    tiles = [pltpu.VMEM((2, token_tile, width), jnp.bfloat16)]
    if combine:
        tiles += [
            pltpu.VMEM((2, token_tile, hidden), jnp.bfloat16),
            pltpu.VMEM((2, token_tile, branches), jnp.float32),
        ]
    scratch = (
        *tiles,
        pltpu.VMEM((width,), jnp.bfloat16),
        pltpu.VMEM((lowrank, width), jnp.bfloat16),
        pltpu.VMEM(up.shape, jnp.bfloat16),
        pltpu.VMEM((2, token_tile, hidden), jnp.bfloat16),
        pltpu.VMEM((2, token_tile, width), jnp.bfloat16),
        *(dma((2,)) for _ in tiles),
        dma,
        dma,
        dma,
        dma((2,)),
        dma((2,)),
        *((dma((2,)),) if combine else ()),
    )
    out_shape = [
        jax.ShapeDtypeStruct((tokens, hidden), jnp.bfloat16),
        jax.ShapeDtypeStruct((tokens, width), jnp.bfloat16),
    ]
    if combine:
        out_shape.append(jax.ShapeDtypeStruct((tokens, width), jnp.bfloat16))
    inputs = (x, *extra, weight, down, up)
    return pl.pallas_call(
        functools.partial(
            _kernel,
            token_tile=token_tile,
            tokens=tokens,
            hidden=hidden,
            branches=branches,
            eps=eps,
            up_t=up_t,
            combine=combine,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(),
            in_specs=(hbm,) * len(inputs),
            out_specs=(hbm,) * len(out_shape),
            scratch_shapes=scratch,
        ),
        out_shape=tuple(out_shape),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=60 << 20),
        name=f"qwen4exp_hc_{'combine_' if combine else ''}branchstore32_m{token_tile}",
    )(*inputs)


@functools.partial(
    jax.jit, static_argnames=("branches", "hidden", "eps", "token_tile", "up_t")
)
def _full_mix(x, weight, down, up, *, branches, hidden, eps, token_tile, up_t=False):
    """``up_t``: ``up`` is given as its [low rank, width] transpose."""
    return _pallas_mix(
        x,
        weight,
        down,
        up,
        (),
        branches=branches,
        hidden=hidden,
        eps=eps,
        token_tile=token_tile,
        up_t=up_t,
    )


@functools.partial(
    jax.jit, static_argnames=("branches", "hidden", "eps", "token_tile", "up_t")
)
def _combine_full_mix(
    residual, block, gates, weight, down, up, *, branches, hidden, eps, token_tile, up_t
):
    """``_full_mix(gated_residual(residual, block, gates), ...)`` in one kernel.

    Returns ``(mixed, norm, combined)``; ``combined`` is the gated residual
    (the next hyper input), written once while its tile is normalized.
    """
    return _pallas_mix(
        residual,
        weight,
        down,
        up,
        (block, gates),
        branches=branches,
        hidden=hidden,
        eps=eps,
        token_tile=token_tile,
        up_t=up_t,
    )
