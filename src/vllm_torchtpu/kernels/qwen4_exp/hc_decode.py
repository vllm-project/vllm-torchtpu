# SPDX-License-Identifier: Apache-2.0
"""Decode-sized HyperConnection mix as one Pallas call per call site.

At decode token counts the HC math (per-branch RMS norm, the low-rank
down/SiLU/up gate, the sigmoid-weighted branch mean and the combine's
injection gate) is a chain of ~15 tiny XLA fusions per call that are
dominated by launch and HBM round trips. This kernel streams the layer's
HC weights into VMEM once per call (one DMA per branch slice, overlapped
with the normalization) and keeps every intermediate on chip.

It reproduces the TorchTPU eager path's rounding boundaries (see
``GatedResidual.mix``/``combine`` in models/vllm/qwen4_exp/hyperconnection.py):

* norm = bf16(f32(x) * rsqrt(mean(x^2) + eps) * (1 + f32(w))) per branch
* gate = bf16(silu(f32(bf16(norm @ down^T) / hc)))
* mixed = bf16(sum_b(sigmoid(f32(bf16(gate @ up_b^T))) * f32(norm_b)) / hc)
* injection = 2 * sigmoid(f32(bf16(norm @ inject^T)) / hc), kept in f32

so it only differs from XLA by the matmul accumulation order and the
transcendental implementations (ulp level).

With ``prologue=True`` the kernel first applies the previous HC's combine,
h = bf16(f32(residual) + f32(block) * injection[:, b]), writes h (the next
residual) and mixes it; this fuses attn-combine with mlp-mix.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

MAX_TOKENS = 512
_SINGLE_TILE_MAX = 128
_ROW_CHUNK = 32


def decode_token_tile(tokens: int) -> int | None:
    """Token tile for ``tokens`` rows, or None when the kernel does not apply."""
    if tokens < 1 or tokens > MAX_TOKENS:
        return None
    if tokens <= _SINGLE_TILE_MAX:
        return tokens
    for tile in (128, 64, 32, 16):
        if tokens % tile == 0:
            return tile
    return None


def _kernel(
    *refs,
    branches: int,
    hidden: int,
    eps: float,
    prologue: bool,
    inject: bool,
    up_transposed: bool = False,
):
    refs_list: list = list(refs)
    prev_ref = block_ref = prev_inj_ref = x_ref = None
    if prologue:
        prev_ref, block_ref, prev_inj_ref = refs_list[:3]
        refs_list = refs_list[3:]
    else:
        x_ref = refs_list.pop(0)
    w_hbm, down_hbm, up_hbm = refs_list[:3]
    refs_list = refs_list[3:]
    injw_hbm = refs_list.pop(0) if inject else None
    mixed_ref = refs_list.pop(0)
    inj_ref = refs_list.pop(0) if inject else None
    h_ref = refs_list.pop(0) if prologue else None
    w_vm, down_vm, up_vm = refs_list[:3]
    refs_list = refs_list[3:]
    injw_vm = refs_list.pop(0) if inject else None
    norm_vm, sems = refs_list

    first = pl.program_id(0) == 0
    w_copy = pltpu.make_async_copy(w_hbm, w_vm, sems.at[0])
    down_copy = pltpu.make_async_copy(down_hbm, down_vm, sems.at[1])
    if up_transposed:
        up_copy = pltpu.make_async_copy(up_hbm, up_vm, sems.at[2])
        up_copies = None
        inj_sem_idx = 3
    else:
        up_copy = None
        up_copies = [
            pltpu.make_async_copy(
                up_hbm.at[pl.ds(b * hidden, hidden)],
                up_vm.at[pl.ds(b * hidden, hidden)],
                sems.at[2 + b],
            )
            for b in range(branches)
        ]
        inj_sem_idx = 2 + branches
    inj_copy = (
        pltpu.make_async_copy(injw_hbm, injw_vm, sems.at[inj_sem_idx])
        if inject
        else None
    )

    # Weights stay resident in scratch across token tiles: fetch them once.
    @pl.when(first)
    def _start_weights():
        w_copy.start()
        if inj_copy is not None:
            inj_copy.start()
        down_copy.start()
        if up_transposed:
            assert up_copy is not None
            up_copy.start()
        else:
            assert up_copies is not None
            for copy in up_copies:
                copy.start()

    def wait_first(copy):
        @pl.when(first)
        def _wait():
            copy.wait()

    def lane_column(values, column):
        # values[:, column:column + 1] as a masked lane reduction (exact:
        # one non-zero term), avoiding an unaligned lane slice.
        lanes = jax.lax.broadcasted_iota(jnp.int32, values.shape, 1)
        return jnp.sum(jnp.where(lanes == column, values, 0.0), axis=1, keepdims=True)

    nt = (((1,), (1,)), ((), ()))
    nn_dim = (((1,), (0,)), ((), ()))
    wait_first(w_copy)
    if inj_copy is not None:
        wait_first(inj_copy)

    def normalize_rows(offset, rows):
        """Norm (and combine prologue, injection gate) for ``rows`` rows."""
        weight = w_vm[...].astype(jnp.float32)
        block = prev_inj = None
        if prologue:
            assert block_ref is not None and prev_inj_ref is not None
            block = block_ref[pl.ds(offset, rows), :].astype(jnp.float32)
            prev_inj = prev_inj_ref[pl.ds(offset, rows), :]
        inj_terms: list[jax.Array | None] = [None] * branches
        for b in range(branches):
            start = b * hidden
            if prologue:
                assert (
                    prev_ref is not None
                    and h_ref is not None
                    and block is not None
                    and prev_inj is not None
                )
                h = (
                    prev_ref[pl.ds(offset, rows), pl.ds(start, hidden)].astype(
                        jnp.float32
                    )
                    + block * lane_column(prev_inj, b)
                ).astype(jnp.bfloat16)
                h_ref[pl.ds(offset, rows), pl.ds(start, hidden)] = h
                values = h.astype(jnp.float32)
            else:
                assert x_ref is not None
                values = x_ref[pl.ds(offset, rows), pl.ds(start, hidden)].astype(
                    jnp.float32
                )
            variance = jnp.sum(values * values, axis=-1, keepdims=True) / hidden
            normalized = values * jax.lax.rsqrt(variance + eps)
            norm = normalized * (1.0 + weight[None, start : start + hidden])
            norm = norm.astype(jnp.bfloat16)
            norm_vm[pl.ds(offset, rows), pl.ds(start, hidden)] = norm
            if inject:
                # The [T, hc] injection logits on the VPU: BF16 x BF16
                # products are exact in FP32; only the summation order
                # differs from the MXU.
                assert injw_vm is not None
                norm32 = norm.astype(jnp.float32)
                for j in range(branches):
                    row = injw_vm[pl.ds(j, 1), pl.ds(start, hidden)]
                    term = norm32 * row.astype(jnp.float32)
                    prev_term = inj_terms[j]
                    inj_terms[j] = term if prev_term is None else prev_term + term
        if inject:
            assert inj_ref is not None
            shape = (rows, branches)
            lanes = jax.lax.broadcasted_iota(jnp.int32, shape, 1)
            logits = jnp.zeros(shape, jnp.float32)
            for j in range(branches):
                term_j = inj_terms[j]
                assert term_j is not None
                column = jnp.sum(term_j, axis=1, keepdims=True)
                logits = jnp.where(lanes == j, column, logits)
            logits = logits.astype(jnp.bfloat16).astype(jnp.float32)
            inj_ref[pl.ds(offset, rows), :] = 2.0 * jax.nn.sigmoid(logits / branches)

    tile = norm_vm.shape[0]
    if tile >= 2 * _ROW_CHUNK and tile % _ROW_CHUNK == 0:
        # Bound the live FP32 row working set like hc_full's 32-row loop.
        def chunk(index, carry):
            normalize_rows(pl.multiple_of(index * _ROW_CHUNK, _ROW_CHUNK), _ROW_CHUNK)
            return carry

        jax.lax.fori_loop(0, tile // _ROW_CHUNK, chunk, jnp.int32(0))
    else:
        normalize_rows(0, tile)

    wait_first(down_copy)
    down_acc = jax.lax.dot_general(
        norm_vm[...],
        down_vm[...],
        dimension_numbers=nt,
        preferred_element_type=jnp.float32,
    )

    gate = (down_acc.astype(jnp.bfloat16) / branches).astype(jnp.bfloat16)
    gate = jax.nn.silu(gate.astype(jnp.float32)).astype(jnp.bfloat16)
    if up_transposed:
        assert up_copy is not None
        wait_first(up_copy)
        logits = (
            jax.lax.dot_general(
                gate,
                up_vm[...],
                dimension_numbers=nn_dim,
                preferred_element_type=jnp.float32,
            )
            .astype(jnp.bfloat16)
            .astype(jnp.float32)
        )
        weighted = jax.nn.sigmoid(logits) * norm_vm[...].astype(jnp.float32)
        mixed = weighted[:, :hidden]
        for b in range(1, branches):
            start = b * hidden
            mixed = mixed + weighted[:, start : start + hidden]
    else:
        assert up_copies is not None
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
    assert mixed is not None
    mixed_ref[...] = (mixed / branches).astype(jnp.bfloat16)


def hc_decode_mix(
    x,
    weight,
    down,
    up,
    inject_weight=None,
    *,
    branches: int,
    hidden: int,
    eps: float,
    block=None,
    prev_injection=None,
    interpret=False,
    up_transposed: bool = False,
):
    """Fused HC mix for at most ``MAX_TOKENS`` rows.

    ``x`` is the [T, hc*H] hyper input, or with ``block``/``prev_injection``
    the previous HC's residual that is combined first. Returns
    ``(mixed, injection)`` (``injection`` is None without ``inject_weight``),
    plus the combined hyper input when ``block`` is given.
    """
    return _hc_decode_mix(
        x,
        weight,
        down,
        up,
        inject_weight,
        block,
        prev_injection,
        branches=branches,
        hidden=hidden,
        eps=eps,
        interpret=interpret,
        up_transposed=up_transposed,
    )


@functools.partial(
    jax.jit,
    static_argnames=("branches", "hidden", "eps", "interpret", "up_transposed"),
)
def _hc_decode_mix(
    x,
    weight,
    down,
    up,
    inject_weight,
    block,
    prev_injection,
    *,
    branches,
    hidden,
    eps,
    interpret,
    up_transposed=False,
):
    tokens, width = x.shape
    lowrank = down.shape[0]
    prologue = block is not None
    inject = inject_weight is not None
    tile = decode_token_tile(tokens)
    expected_up_shape = (lowrank, width) if up_transposed else (width, lowrank)
    if (
        tile is None
        or width != branches * hidden
        or hidden % 128
        or weight.shape != (width,)
        or down.shape != (lowrank, width)
        or up.shape != expected_up_shape
        or (inject and inject_weight.shape != (branches, width))
        or (prologue and block.shape != (tokens, hidden))
        or (prologue and prev_injection.shape != (tokens, branches))
        or (prologue and prev_injection.dtype != jnp.float32)
        or any(
            a.dtype != jnp.bfloat16
            for a in (x, weight, down, up)
            + (() if inject_weight is None else (inject_weight,))
            + (() if block is None else (block,))
        )
    ):
        raise ValueError("HC decode mix requires BF16 decode-sized token tiles")

    def rows(cols):
        return pl.BlockSpec((tile, cols), lambda i: (i, 0))

    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    in_specs = (
        [rows(width), rows(hidden), rows(branches)] if prologue else [rows(width)]
    )
    in_specs += [hbm] * (4 if inject else 3)
    out_specs = [rows(hidden)]
    out_shape = [jax.ShapeDtypeStruct((tokens, hidden), jnp.bfloat16)]
    if inject:
        out_specs.append(rows(branches))
        out_shape.append(jax.ShapeDtypeStruct((tokens, branches), jnp.float32))
    if prologue:
        out_specs.append(rows(width))
        out_shape.append(jax.ShapeDtypeStruct((tokens, width), jnp.bfloat16))
    scratch = [
        pltpu.VMEM((width,), jnp.bfloat16),
        pltpu.VMEM((lowrank, width), jnp.bfloat16),
        pltpu.VMEM(expected_up_shape, jnp.bfloat16),
    ]
    if inject:
        scratch.append(pltpu.VMEM((branches, width), jnp.bfloat16))
    num_sems = 4 if up_transposed else (3 + branches)
    scratch += [
        pltpu.VMEM((tile, width), jnp.bfloat16),
        pltpu.SemaphoreType.DMA((num_sems,)),
    ]
    args = [x, block, prev_injection] if prologue else [x]
    args += [weight, down, up] + ([inject_weight] if inject else [])
    out_bytes = tokens * hidden * 2
    if inject:
        out_bytes += tokens * branches * 4
    if prologue:
        out_bytes += tokens * width * 2
    outs = pl.pallas_call(
        functools.partial(
            _kernel,
            branches=branches,
            hidden=hidden,
            eps=eps,
            prologue=prologue,
            inject=inject,
            up_transposed=up_transposed,
        ),
        grid=(tokens // tile,),
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shape,
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(
            # Weights are fetched on the first tile only: keep tiles in order.
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=56 << 20,
        ),
        cost_estimate=pl.CostEstimate(
            flops=4 * tokens * width * lowrank
            + tokens * (7 * width + branches + 2 * lowrank)
            + (2 * tokens * width * branches if inject else 0)
            + (2 * tokens * width if prologue else 0),
            transcendentals=tokens
            * (branches + lowrank + width + (branches if inject else 0)),
            bytes_accessed=sum(a.size * a.dtype.itemsize for a in args if a is not None)
            + out_bytes,
        ),
        interpret=interpret,
        name=(
            f"qwen4exp_hc_decode{'_combine' if prologue else ''}"
            f"{'_ut' if up_transposed else ''}"
        ),
    )(*args)
    mixed = outs[0]
    injection = outs[1] if inject else None
    if prologue:
        return mixed, injection, outs[-1]
    return mixed, injection
