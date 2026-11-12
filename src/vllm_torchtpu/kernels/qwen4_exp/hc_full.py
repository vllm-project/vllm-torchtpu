# SPDX-License-Identifier: Apache-2.0
"""Optional prefill HyperConnection mixer with resident weights and two tiles.

Normalization and SiLU use FP32 internally, matching upstream NVIDIA HC.
Projection and stored outputs remain BF16. The kernel stages the normalization
and both projection weights once, overlaps the next input tile with the current
computation, and drains output tiles only before reuse.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp.hc_resources import (
    resolve_kernel_vmem,
    supports_full_mix,
)

# RMS-norm variants of the norm loop (TPU_QWEN4_HC_PREFILL_NORM).
NORM_IMPLS = ("vpu", "rows64", "gram", "gram64", "bf16ep", "bf16ep64")


def norm_rows_per_step(norm_impl: str, rows: int) -> int:
    """Rows one norm loop step covers (64 for the *64 variants when it divides)."""
    return 64 if norm_impl.endswith("64") and rows % 64 == 0 else 32


def branch_variance(values, xb, hidden: int, norm_impl: str):
    """mean(x^2) over one branch's rows: ``values`` f32 [r, hidden], ``xb`` the
    same rows in bf16. vpu / rows64 / bf16ep* keep ``jnp.mean`` (bitwise);
    gram / gram64 take the sum of squares from the MXU (diag of xb xb^T)."""
    if not norm_impl.startswith("gram"):
        return jnp.mean(values * values, axis=-1, keepdims=True)
    gram = jax.lax.dot_general(
        xb, xb, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
    )
    r = gram.shape[0]
    eye = jax.lax.broadcasted_iota(jnp.int32, (r, r), 0) == jax.lax.broadcasted_iota(
        jnp.int32, (r, r), 1
    )
    return jnp.sum(jnp.where(eye, gram, 0.0), axis=1, keepdims=True) / hidden


def norm_epilogue(
    values, variance, weight_f32, weight_bf16, eps: float, norm_impl: str
):
    """RMS-norm epilogue for one branch's rows.

    vpu / rows64 / gram* preserve FP32 internal normalization
    ``bf16(norm_f32 + norm_f32 * weight_f32)``; bf16ep* evaluates ``n * w`` and
    ``n + .`` in BF16 arithmetic on ``n = bf16(values * rsqrt(variance + eps))``.
    """
    norm_f32 = values * jax.lax.rsqrt(variance + eps)
    if norm_impl.startswith("bf16ep"):
        normalized = norm_f32.astype(jnp.bfloat16)
        return normalized + normalized * weight_bf16
    return (norm_f32 + norm_f32 * weight_f32).astype(jnp.bfloat16)


def _kernel(
    *refs,
    token_tile,
    tokens,
    hidden,
    branches,
    eps,
    up_t,
    combine,
    inject,
    norm_impl="vpu",
):
    refs = iter(refs)
    take = lambda n: [next(refs) for _ in range(n)]  # noqa: E731
    x_hbm, *extra_hbm = take(3 if combine else 1)
    weight_hbm, down_hbm, up_hbm, *inject_hbm = take(4 if inject else 3)
    # With ``inject`` the kernel emits the combine's gate projection in place
    # of the normalized input (its only other consumer).
    mixed_hbm, norm_hbm, *combined_hbm = take(3 if combine else 2)
    x_vm, *extra_vm = take(3 if combine else 1)
    # With ``inject`` the gate weight rows follow the down projection's in
    # ``down_vm``: one matmul yields both (the MXU pads 320 columns anyway).
    weight_vm, down_vm, up_vm = take(3)
    mixed_vm, norm_vm, *proj_vm = take(3 if inject else 2)
    x_sems, *extra_sems = take(3 if combine else 1)
    weight_sem, down_sem, up_sem, *inject_sem = take(4 if inject else 3)
    mixed_sems, norm_sems, *combined_sems = take(3 if combine else 2)
    if combine:
        block_hbm, gates_hbm = extra_hbm
        block_vm, gates_vm = extra_vm
        block_sems, gates_sems = extra_sems
        (combined_hbm,) = combined_hbm
        (combined_sems,) = combined_sems
    lowrank = down_hbm.shape[0]
    if inject:
        (proj_vm,) = proj_vm
        inject_dma = pltpu.make_async_copy(
            inject_hbm[0], down_vm.at[pl.ds(lowrank, branches)], inject_sem[0]
        )
        inject_dma.start()
    weight_dma = pltpu.make_async_copy(weight_hbm, weight_vm, weight_sem)
    down_dma = pltpu.make_async_copy(
        down_hbm, down_vm.at[pl.ds(0, lowrank)] if inject else down_vm, down_sem
    )
    up_dma = pltpu.make_async_copy(up_hbm, up_vm, up_sem)
    weight_dma.start()
    down_dma.start()
    up_dma.start()

    def wait_norm_output(slot):
        """Wait for the slot's last ``norm`` (or gate projection) output copy."""
        out_vm = proj_vm if inject else norm_vm
        pltpu.make_async_copy(
            out_vm.at[slot], out_vm.at[slot], norm_sems.at[slot]
        ).wait()

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

    tiles = tokens // token_tile
    uses_bf16ep = norm_impl.startswith("bf16ep")

    def one_step(step, parity, weight, first=False):
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

        @pl.when(step + 1 < tiles)
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

        if first:
            weight_dma.wait()
            if not uses_bf16ep:
                weight = weight_vm[...].astype(jnp.float32)

        @pl.when(step >= 2)
        def _wait_prior_norm_output():
            wait_norm_output(parity)

        step_rows = norm_rows_per_step(norm_impl, token_tile)

        def normalize_rows(index, carry):
            offset = index * step_rows
            for branch in range(branches):
                start = branch * hidden
                xb = x_vm[parity, pl.ds(offset, step_rows), pl.ds(start, hidden)]
                values = xb.astype(jnp.float32)
                variance = branch_variance(values, xb, hidden, norm_impl)
                w_bf16 = (
                    weight_vm[pl.ds(start, hidden)][None, :] if uses_bf16ep else None
                )
                w_f32 = None if uses_bf16ep else weight[None, start : start + hidden]
                norm_vm[parity, pl.ds(offset, step_rows), pl.ds(start, hidden)] = (
                    norm_epilogue(
                        values,
                        variance,
                        w_f32,
                        w_bf16,
                        eps,
                        norm_impl,
                    )
                )
            return carry

        jax.lax.fori_loop(0, token_tile // step_rows, normalize_rows, jnp.int32(0))
        norm = norm_vm[parity]
        if not inject:
            pltpu.make_async_copy(
                norm_vm.at[parity],
                norm_hbm.at[pl.ds(step * token_tile, token_tile)],
                norm_sems.at[parity],
            ).start()

        if first:
            down_dma.wait()
            if inject:
                inject_dma.wait()

        down = jax.lax.dot_general(
            norm,
            down_vm[...],
            dimension_numbers=(((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        ).astype(jnp.bfloat16)
        if inject:
            proj_vm[parity] = down[:, lowrank:]
            pltpu.make_async_copy(
                proj_vm.at[parity],
                norm_hbm.at[pl.ds(step * token_tile, token_tile)],
                norm_sems.at[parity],
            ).start()
            down = down[:, :lowrank]
        gate = jax.nn.silu(down.astype(jnp.float32) / branches).astype(jnp.bfloat16)

        if first:
            up_dma.wait()

        # Match the BF16 up Linear output before NVIDIA's FP32 sigmoid/mix.
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
        return weight

    # Peel the first pair so weight waits/cast are static first-tile work.
    # Down/up waits stay at their first reads, overlapping normalization.
    weight = one_step(0, 0, None, first=True)
    if tiles > 1:
        one_step(1, 1, weight)

    def step_pair(pair, carry):
        one_step(2 * pair, 0, weight)
        one_step(2 * pair + 1, 1, weight)
        return carry

    jax.lax.fori_loop(1, tiles // 2, step_pair, jnp.int32(0))
    if tiles > 1 and tiles % 2:
        one_step(tiles - 1, 0, weight)
    # A one-tile call never started parity 1; do not wait its semaphore.
    for parity in range(min(tiles, 2)):
        wait_norm_output(parity)
        pltpu.make_async_copy(
            mixed_vm.at[parity], mixed_vm.at[parity], mixed_sems.at[parity]
        ).wait()
    if combine:
        last = (tiles - 1) % 2
        pltpu.make_async_copy(
            x_vm.at[last], x_vm.at[last], combined_sems.at[last]
        ).wait()


def _pallas_mix(
    x,
    weight,
    down,
    up,
    extra,
    inject,
    *,
    branches,
    hidden,
    eps,
    token_tile,
    up_t,
    hardware=None,
    norm_impl="vpu",
):
    """The full mix over ``x`` (``extra``: (block, gates) to combine into it first).

    ``inject`` (the combine's [branches, width] gate weight, or ``()``): return the
    BF16 gate projection ``norm @ inject.T`` [tokens, branches] in place of ``norm``.
    """
    if branches <= 0 or hidden <= 0 or hidden % 128:
        raise ValueError(
            "HC full mix requires positive branches and 128-aligned hidden"
        )
    if token_tile <= 0 or token_tile % 128:
        raise ValueError("HC full mix requires positive 128-aligned token tiles")
    tokens, width = x.shape
    lowrank = down.shape[0]
    if (
        tokens == 0
        or lowrank == 0
        or tokens % token_tile
        or width != branches * hidden
        or weight.shape != (width,)
        or down.shape != (lowrank, width)
        or up.shape != ((lowrank, width) if up_t else (width, lowrank))
        or any(a.dtype != jnp.bfloat16 for a in (x, weight, down, up))
    ):
        raise ValueError("HC full mix requires full BF16 token tiles")
    if norm_impl not in NORM_IMPLS:
        raise ValueError(f"unknown HC norm_impl {norm_impl!r}")
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
    if inject and (
        inject[0].shape != (branches, width) or inject[0].dtype != jnp.bfloat16
    ):
        raise ValueError("HC inject requires a BF16 [branches, width] weight")
    hardware, vmem_bytes = resolve_kernel_vmem(hardware)
    if hardware is not None and not supports_full_mix(
        branches,
        hidden,
        lowrank,
        token_tile,
        hardware,
        up_t=up_t,
        combine=combine,
        inject=bool(inject),
    ):
        raise ValueError("HC full mix requires a supported layout and VMEM budget")
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
        pltpu.VMEM((lowrank + (branches if inject else 0), width), jnp.bfloat16),
        pltpu.VMEM(up.shape, jnp.bfloat16),
        pltpu.VMEM((2, token_tile, hidden), jnp.bfloat16),
        pltpu.VMEM((2, token_tile, width), jnp.bfloat16),
        *((pltpu.VMEM((2, token_tile, branches), jnp.bfloat16),) if inject else ()),
        *(dma((2,)) for _ in tiles),
        dma,
        dma,
        dma,
        *((dma,) if inject else ()),
        dma((2,)),
        dma((2,)),
        *((dma((2,)),) if combine else ()),
    )
    out_shape = [
        jax.ShapeDtypeStruct((tokens, hidden), jnp.bfloat16),
        jax.ShapeDtypeStruct((tokens, branches if inject else width), jnp.bfloat16),
    ]
    if combine:
        out_shape.append(jax.ShapeDtypeStruct((tokens, width), jnp.bfloat16))
    inputs = (x, *extra, weight, down, up, *inject)
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
            inject=bool(inject),
            norm_impl=norm_impl,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(),
            in_specs=(hbm,) * len(inputs),
            out_specs=(hbm,) * len(out_shape),
            scratch_shapes=scratch,
        ),
        out_shape=tuple(out_shape),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_bytes),
        # Two GEMMs at two FLOPs/MAC; normalization, SiLU and branch mixing.
        # Count each logical rsqrt/sigmoid as one transcendental operation.
        # Weights are read once; input and outputs once per token. These
        # unpadded logical bytes do not claim exact DMA bus traffic.
        cost_estimate=pl.CostEstimate(
            flops=4 * tokens * width * lowrank
            + (2 * tokens * width * branches if inject else 0)
            + (2 * tokens * width if combine else 0)
            + tokens * (7 * width + branches + 2 * lowrank),
            transcendentals=tokens * (branches + lowrank + width),
            bytes_accessed=sum(a.size * a.dtype.itemsize for a in inputs)
            + sum(a.size * a.dtype.itemsize for a in out_shape),
        ),
        name=(
            f"qwen4exp_hc_{'combine_' if combine else ''}branchstore32_m{token_tile}"
            f"{'' if norm_impl == 'vpu' else '_n' + norm_impl}"
        ),
    )(*inputs)


@functools.partial(
    jax.jit,
    static_argnames=(
        "branches",
        "hidden",
        "eps",
        "token_tile",
        "up_t",
        "hardware",
        "norm_impl",
    ),
)
def _full_mix(
    x,
    weight,
    down,
    up,
    inject=None,
    *,
    branches,
    hidden,
    eps,
    token_tile,
    up_t=False,
    hardware=None,
    norm_impl="vpu",
):
    """``up_t``: ``up`` is given as its [low rank, width] transpose. ``inject``: the
    second output is the gate projection (see ``_pallas_mix``)."""
    return _pallas_mix(
        x,
        weight,
        down,
        up,
        (),
        () if inject is None else (inject,),
        branches=branches,
        hidden=hidden,
        eps=eps,
        token_tile=token_tile,
        up_t=up_t,
        hardware=hardware,
        norm_impl=norm_impl,
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "branches",
        "hidden",
        "eps",
        "token_tile",
        "up_t",
        "hardware",
        "norm_impl",
    ),
)
def _combine_full_mix(
    residual,
    block,
    gates,
    weight,
    down,
    up,
    inject=None,
    *,
    branches,
    hidden,
    eps,
    token_tile,
    up_t,
    hardware=None,
    norm_impl="vpu",
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
        () if inject is None else (inject,),
        branches=branches,
        hidden=hidden,
        eps=eps,
        token_tile=token_tile,
        up_t=up_t,
        hardware=hardware,
        norm_impl=norm_impl,
    )
