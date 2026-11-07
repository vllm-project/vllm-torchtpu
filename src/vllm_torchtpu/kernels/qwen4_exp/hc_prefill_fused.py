# SPDX-License-Identifier: Apache-2.0
"""Prefill HyperConnection mix that keeps ``norm`` on chip.

The served prefill path (``hc_full`` + an XLA injection matmul + the
``gated_residual`` combine) writes the normalized hyper input [T, hc*H] to HBM
and reads it back for the combine's injection gate: about 40 KB per token per
hyper-connection of traffic that only exists to carry ``norm`` between kernels.

This kernel is the prefill translation of the decode HC kernel
(``hc_decode.py``): token tiles stream through BlockSpec pipelining, the
layer's HC weights are fetched into VMEM on the first tile, and every
intermediate stays on chip. It returns ``mixed`` and the combine's f32
injection gate. With ``prologue`` it first applies the previous
hyper-connection's combine (attn combine fused with the mlp mix), writing the
combined hyper input once.

Rounding boundaries are the prefill path's, so ``mixed`` matches ``hc_full``:

* norm = bf16(n + bf16(n * w)), n = bf16(f32(x) * rsqrt(mean(x^2) + eps))
  (``hc_norm.py``/``hc_full.py``; the decode kernel's single rounding differs)
* gate = bf16(silu(f32(bf16(bf16(norm @ down^T) / hc))))
* mixed = bf16(sum_b(sigmoid(f32(bf16(gate @ up_b^T))) * f32(norm_b)) / hc)
* injection = 2 * sigmoid(f32(bf16(norm @ inject^T)) / hc), kept in f32
* prologue: h = bf16(f32(residual) + f32(block) * injection_prev[:, b])
  (``gated_residual.py``)

The injection logits come from the same MXU pass as the down projection: the
inject rows are appended to ``down`` (zero padded to a 128-row multiple), so
they cost no extra MXU passes on 256-wide MXUs. Only the summation order of
the injection logits differs from the XLA matmul (ulp level, before the bf16
rounding).
"""

import contextlib
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_ROW_CHUNK = 32
# TPU_QWEN4_HC_PREFILL_NORM: how the per-branch RMS norm forms its sum of
# squares and how many rows one loop step normalizes.
#   vpu     (default) VPU square + lane reduction, 32-row steps
#   rows64  the same math in 64-row steps (bitwise equal to vpu)
#   gram    sum of squares from the MXU: the diagonal of x x^T of a row
#           block (exact bf16 products, f32 accumulation), 32-row steps;
#           NOT bitwise: only the f32 summation order differs
#   gram64  gram in 64-row steps
NORM_IMPLS = ("vpu", "rows64", "gram", "gram64")


def norm_rows_per_step(norm_impl: str, rows: int) -> int:
    """Rows one norm loop step covers (64 for the *64 variants when it divides)."""
    return 64 if norm_impl.endswith("64") and rows % 64 == 0 else _ROW_CHUNK


def branch_variance(values, xb, hidden: int, norm_impl: str):
    """mean(x^2) over one branch's rows: values f32 [r, hidden], xb the same
    rows in bf16. vpu/rows64 keep today's jnp.mean (bitwise); gram/gram64 take
    the sum of squares from the MXU (diag of xb xb^T)."""
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


def prefill_token_tile_ok(tokens: int, token_tile: int, partial: bool = False) -> bool:
    """Whether ``tokens`` rows run in whole ``token_tile`` tiles, or with
    ``partial`` in tiles whose last one holds any multiple of 16 rows."""
    if partial:
        return tokens >= token_tile and tokens % 16 == 0
    return tokens >= token_tile and tokens % token_tile == 0


def _kernel(*refs, branches, hidden, eps, lowrank, prologue, scopes=0, norm_impl="vpu"):
    def scope(name):
        # Region-trace scopes (with --xla_enable_custom_call_region_trace=true
        # they appear on the "XLA TraceMe" line). Off by default; a scoped
        # build is for timing splits only, never a numerics reference.
        return jax.named_scope(name) if scopes else contextlib.nullcontext()

    refs = list(refs)
    if prologue:
        prev_ref, block_ref, prev_inj_ref = refs[:3]
        refs = refs[3:]
        x_ref = None
    else:
        x_ref = refs.pop(0)
    w_hbm, down_hbm, up_hbm = refs[:3]
    refs = refs[3:]
    mixed_ref, inj_ref = refs[:2]
    refs = refs[2:]
    h_ref = refs.pop(0) if prologue else None
    w_vm, down_vm, up_vm, norm_vm, sems = refs

    first = pl.program_id(0) == 0
    w_copy = pltpu.make_async_copy(w_hbm, w_vm, sems.at[0])
    down_copy = pltpu.make_async_copy(down_hbm, down_vm, sems.at[1])
    up_copies = [
        pltpu.make_async_copy(
            up_hbm.at[pl.ds(b * hidden, hidden)],
            up_vm.at[pl.ds(b * hidden, hidden)],
            sems.at[2 + b],
        )
        for b in range(branches)
    ]

    # Weights stay resident in scratch across token tiles: fetch them once.
    @pl.when(first)
    def _start_weights():
        w_copy.start()
        down_copy.start()
        for copy in up_copies:
            copy.start()

    def wait_first(copy):
        @pl.when(first)
        def _wait():
            copy.wait()

    wait_first(w_copy)

    def normalize_rows(offset, rows):
        weight = w_vm[...].astype(jnp.float32)
        if prologue:
            block = block_ref[pl.ds(offset, rows), :].astype(jnp.float32)
            gates = prev_inj_ref[pl.ds(offset, rows), :]
        for b in range(branches):
            start = b * hidden
            if prologue:
                residual = prev_ref[pl.ds(offset, rows), pl.ds(start, hidden)].astype(
                    jnp.float32
                )
                h = (residual + block * gates[:, b : b + 1]).astype(jnp.bfloat16)
                h_ref[pl.ds(offset, rows), pl.ds(start, hidden)] = h
                xb = h
                values = h.astype(jnp.float32)
            else:
                xb = x_ref[pl.ds(offset, rows), pl.ds(start, hidden)]
                values = xb.astype(jnp.float32)
            variance = branch_variance(values, xb, hidden, norm_impl)
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
            norm_vm[pl.ds(offset, rows), pl.ds(start, hidden)] = (
                normalized + scaled
            ).astype(jnp.bfloat16)

    tile = norm_vm.shape[0]
    step_rows = norm_rows_per_step(norm_impl, tile)
    with scope("hcp_norm"):
        if tile >= 2 * step_rows and tile % step_rows == 0:
            # Bound the live FP32 row working set like hc_full's 32-row loop.
            def chunk(index, carry):
                normalize_rows(pl.multiple_of(index * step_rows, step_rows), step_rows)
                return carry

            jax.lax.fori_loop(0, tile // step_rows, chunk, jnp.int32(0))
        else:
            normalize_rows(0, tile)

    nt = (((1,), (1,)), ((), ()))
    wait_first(down_copy)
    with scope("hcp_down"):
        down = jax.lax.dot_general(
            norm_vm[...], down_vm[...], dimension_numbers=nt, preferred_element_type=jnp.float32
        ).astype(jnp.bfloat16)

    # Injection gate from the appended inject rows: one exact masked lane
    # reduction per branch (a single non-zero term), as hc_decode extracts a
    # lane column without an unaligned lane slice.
    with scope("hcp_inj"):
        down32 = down.astype(jnp.float32)
        lanes = jax.lax.broadcasted_iota(jnp.int32, down32.shape, 1)
        inj_shape = (tile, branches)
        inj_lanes = jax.lax.broadcasted_iota(jnp.int32, inj_shape, 1)
        logits_inj = jnp.zeros(inj_shape, jnp.float32)
        for j in range(branches):
            column = jnp.sum(
                jnp.where(lanes == lowrank + j, down32, 0.0), axis=1, keepdims=True
            )
            logits_inj = jnp.where(inj_lanes == j, column, logits_inj)
        inj_ref[...] = 2.0 * jax.nn.sigmoid(logits_inj / branches)

    with scope("hcp_up"):
        gate = (down[:, :lowrank] / branches).astype(jnp.bfloat16)
        gate = jax.nn.silu(gate.astype(jnp.float32)).astype(jnp.bfloat16)
        mixed = None
        for b in range(branches):
            start = b * hidden
            wait_first(up_copies[b])
            logits = (
                jax.lax.dot_general(
                    gate,
                    up_vm[pl.ds(start, hidden), :],
                    dimension_numbers=nt,
                    preferred_element_type=jnp.float32,
                )
                .astype(jnp.bfloat16)
                .astype(jnp.float32)
            )
            weighted = jax.nn.sigmoid(logits) * norm_vm[:, pl.ds(start, hidden)].astype(
                jnp.float32
            )
            mixed = weighted if mixed is None else mixed + weighted
        mixed_ref[...] = (mixed / branches).astype(jnp.bfloat16)


def extend_down(down: jax.Array, inject: jax.Array) -> jax.Array:
    """``down`` [lowrank, W] with the inject rows [hc, W] appended, zero padded
    to a multiple of 128 rows."""
    lowrank, width = down.shape
    rows = lowrank + inject.shape[0]
    pad = -rows % 128
    return jnp.concatenate([down, inject, jnp.zeros((pad, width), down.dtype)])


def hc_prefill_mix(
    x,
    weight,
    down,
    up,
    inject,
    *,
    branches: int,
    hidden: int,
    eps: float,
    token_tile: int,
    block=None,
    prev_injection=None,
    interpret=False,
    partial=False,
    scopes=0,
    norm_impl="vpu",
):
    """Fused prefill HC mix with the injection gate, optionally after the
    previous HC's combine. ``partial`` admits a partial last tile (any multiple
    of 16 rows): the grid covers it and its out-of-range rows are dropped on
    write; rows are independent, so they never reach real rows.

    ``x`` is the [T, hc*H] hyper input, or with ``block``/``prev_injection``
    the previous HC's residual that is combined first. Returns
    ``(mixed, injection)``, plus the combined hyper input when ``block`` is
    given.
    """
    return _hc_prefill_mix(
        x,
        weight,
        down,
        up,
        inject,
        block,
        prev_injection,
        branches=branches,
        hidden=hidden,
        eps=eps,
        token_tile=token_tile,
        interpret=interpret,
        partial=partial,
        scopes=scopes,
        norm_impl=norm_impl,
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "branches", "hidden", "eps", "token_tile", "interpret", "partial", "scopes", "norm_impl"
    ),
)
def _hc_prefill_mix(
    x,
    weight,
    down,
    up,
    inject,
    block,
    prev_injection,
    *,
    branches,
    hidden,
    eps,
    token_tile,
    interpret,
    partial=False,
    scopes=0,
    norm_impl="vpu",
):
    tokens, width = x.shape
    if norm_impl not in NORM_IMPLS:
        raise ValueError(f"unknown HC norm_impl {norm_impl!r}")
    lowrank = down.shape[0]
    prologue = block is not None
    if (
        not prefill_token_tile_ok(tokens, token_tile, partial)
        or token_tile % 8
        or width != branches * hidden
        or hidden % 128
        or weight.shape != (width,)
        or down.shape != (lowrank, width)
        or up.shape != (width, lowrank)
        or inject.shape != (branches, width)
        or (prologue and block.shape != (tokens, hidden))
        or (prologue and prev_injection.shape != (tokens, branches))
        or (prologue and prev_injection.dtype != jnp.float32)
        or any(
            a.dtype != jnp.bfloat16
            for a in (x, weight, down, up, inject) + ((block,) if prologue else ())
        )
    ):
        raise ValueError("HC prefill fused mix requires BF16 whole token tiles")
    down_ext = extend_down(down, inject)

    def rows(cols):
        return pl.BlockSpec((token_tile, cols), lambda i: (i, 0))

    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    in_specs = (
        [rows(width), rows(hidden), rows(branches)] if prologue else [rows(width)]
    )
    in_specs += [hbm] * 3
    out_specs = [rows(hidden), rows(branches)]
    out_shape = [
        jax.ShapeDtypeStruct((tokens, hidden), jnp.bfloat16),
        jax.ShapeDtypeStruct((tokens, branches), jnp.float32),
    ]
    if prologue:
        out_specs.append(rows(width))
        out_shape.append(jax.ShapeDtypeStruct((tokens, width), jnp.bfloat16))
    scratch = [
        pltpu.VMEM((width,), jnp.bfloat16),
        pltpu.VMEM(down_ext.shape, jnp.bfloat16),
        pltpu.VMEM((width, lowrank), jnp.bfloat16),
        pltpu.VMEM((token_tile, width), jnp.bfloat16),
        pltpu.SemaphoreType.DMA((2 + branches,)),
    ]
    args = [x, block, prev_injection] if prologue else [x]
    args += [weight, down_ext, up]
    outs = pl.pallas_call(
        functools.partial(
            _kernel,
            branches=branches,
            hidden=hidden,
            eps=eps,
            lowrank=lowrank,
            prologue=prologue,
            scopes=scopes,
            norm_impl=norm_impl,
        ),
        grid=(pl.cdiv(tokens, token_tile),),
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shape,
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(
            # Weights are fetched on the first tile only: keep tiles in order.
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=60 << 20,
        ),
        interpret=interpret,
        name=f"qwen4exp_hc_prefill{'_combine' if prologue else ''}_inj_m{token_tile}",
    )(*args)
    return tuple(outs)
