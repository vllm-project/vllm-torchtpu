# SPDX-License-Identifier: Apache-2.0
"""Torch custom-op boundaries for Qwen TPU kernels."""

import functools
import hashlib

import jax
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.qwen4_exp.gated_residual import gated_residual_fn
from vllm_torchtpu.kernels.qwen4_exp.hc_decode import hc_decode_mix
from vllm_torchtpu.kernels.qwen4_exp.hc_full import _full_mix
from vllm_torchtpu.kernels.qwen4_exp.hc_norm import _normalize
from vllm_torchtpu.kernels.qwen4_exp.hc_prefill_fused import hc_prefill_mix
from vllm_torchtpu.kernels.qwen4_exp.hc_up import _up_epilogue


@functools.cache
def build_gated_residual_op(
    branches: int, hidden: int, token_tile: int, partial: bool = False
):
    op = pallas.jax_op(
        f"qwen4_exp::gated_residual_b{branches}_h{hidden}_m{token_tile}"
        f"{'_p' if partial else ''}_v1",
        gated_residual_fn(branches, hidden, token_tile, partial),
    )

    @op.register_fake
    def fake(residual, block, gates):
        return residual.new_empty(residual.shape)

    return op


@functools.cache
def build_full_mix_op(
    branches: int,
    hidden: int,
    eps: float,
    token_tile: int,
    partial: bool = False,
    norm_impl: str = "vpu",
):
    eps_tag = hashlib.sha1(repr(eps).encode()).hexdigest()[:8]

    def core(
        x: jax.Array, weight: jax.Array, down: jax.Array, up: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        return _full_mix(
            x,
            weight,
            down,
            up,
            branches=branches,
            hidden=hidden,
            eps=eps,
            token_tile=token_tile,
            partial=partial,
            norm_impl=norm_impl,
        )

    op = pallas.jax_op(
        f"qwen4_exp::hc_full_c{branches}_h{hidden}_m{token_tile}_e{eps_tag}"
        f"{'_p' if partial else ''}{'' if norm_impl == 'vpu' else '_n' + norm_impl}_v1",
        core,
    )

    @op.register_fake
    def fake(x, weight, down, up):
        return (
            x.new_empty((x.shape[0], hidden), dtype=torch.bfloat16),
            x.new_empty(x.shape, dtype=torch.bfloat16),
        )

    return op


@functools.cache
def build_norm_op(branches: int, hidden: int, eps: float, token_tile: int):
    """Register a local Pallas op for complete prefill token tiles."""
    eps_tag = hashlib.sha1(repr(eps).encode()).hexdigest()[:8]

    def core(input_tensor: jax.Array, weight: jax.Array) -> jax.Array:
        return _normalize(
            input_tensor,
            weight,
            branches=branches,
            hidden=hidden,
            eps=eps,
            token_tile=token_tile,
        )

    op = pallas.jax_op(
        f"qwen4_exp::hc_norm_c{branches}_h{hidden}_m{token_tile}_e{eps_tag}_v2",
        core,
    )

    @op.register_fake
    def fake(input_tensor, weight):
        return input_tensor.new_empty(input_tensor.shape)

    return op


@functools.cache
def build_up_epilogue_op(branches: int, hidden: int, token_tile: int):
    """Register a local JAX op; callers gate it to whole prefill tiles."""

    def core(mid: jax.Array, norm: jax.Array, weight: jax.Array) -> jax.Array:
        return _up_epilogue(
            mid, norm, weight, branches=branches, hidden=hidden, token_tile=token_tile
        )

    op = pallas.jax_op(
        f"qwen4_exp::hc_up_epilogue_c{branches}_h{hidden}_m{token_tile}_v1", core
    )

    @op.register_fake
    def fake(mid, norm, weight):
        return mid.new_empty((mid.shape[0], hidden), dtype=torch.bfloat16)

    return op


@functools.cache
def build_decode_mix_op(
    branches: int,
    hidden: int,
    eps: float,
    inject: bool,
    up_t: bool = False,
    max_tile: int = 0,
):
    """Register the fused decode HC mix; callers gate it on decode_token_tile.

    Returns ``(mixed, injection)`` with ``inject`` (the combine's f32 gate,
    from the same normalized input), else ``mixed`` only. With ``up_t``
    the kernel reads the up weight through ``up.T``, a bitcast of its served
    layout (TPU_QWEN4_HC_DECODE_UP_T); the op's inputs are unchanged.
    ``max_tile``: the kernel's row tile limit (TPU_QWEN4_HC_DECODE_TILE_ROWS,
    q38 PORT-BL HC; 0 = off, the op and its name exactly as before).
    """
    eps_tag = hashlib.sha1(repr(eps).encode()).hexdigest()[:8]
    static = dict(branches=branches, hidden=hidden, eps=eps, up_t=up_t)
    if max_tile:
        static["max_tile"] = max_tile
    tile_tag = f"_t{max_tile}" if max_tile else ""
    if inject:

        def core(
            x: jax.Array,
            weight: jax.Array,
            down: jax.Array,
            up: jax.Array,
            inject_weight: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            return hc_decode_mix(x, weight, down, up, inject_weight, **static)

    else:

        def core(
            x: jax.Array, weight: jax.Array, down: jax.Array, up: jax.Array
        ) -> jax.Array:
            return hc_decode_mix(x, weight, down, up, **static)[0]

    op = pallas.jax_op(
        f"qwen4_exp::hc_decode_c{branches}_h{hidden}_e{eps_tag}"
        f"{'_inj' if inject else ''}{'_upt' if up_t else ''}{tile_tag}_v1",
        core,
    )

    def mixed_like(x):
        return x.new_empty((x.shape[0], hidden), dtype=torch.bfloat16)

    if inject:

        @op.register_fake
        def fake(x, weight, down, up, inject_weight):
            gate = x.new_empty((x.shape[0], branches), dtype=torch.float32)
            return mixed_like(x), gate

    else:

        @op.register_fake
        def fake(x, weight, down, up):
            return mixed_like(x)

    return op


@functools.cache
def build_decode_combine_mix_op(
    branches: int,
    hidden: int,
    eps: float,
    up_t: bool = False,
    inject: bool = True,
    max_tile: int = 0,
):
    """The previous HC's combine fused ahead of this HC's decode mix.

    Returns ``(mixed, injection, combined)``; ``combined`` is the hyper input
    this HC mixed (the next combine's residual). ``up_t`` as in
    ``build_decode_mix_op``.

    q38 PORT-BL HC (ported from Benliang Wang <benliangw@google.com>, commit
    28419c4a5 on benliangw/q38fh-integ-pin-5b44702aea): ``inject=False`` is the
    final mixer's variant (TPU_QWEN4_HC_DECODE_FOLD hands the last layer's mlp
    combine to it): no injection gate, returns ``(mixed, combined)``; it honours
    ``up_t`` exactly like the injection variant. ``max_tile`` as in
    ``build_decode_mix_op``. With the defaults (inject, no tile limit) the op,
    its name and its body are exactly as before.
    """
    eps_tag = hashlib.sha1(repr(eps).encode()).hexdigest()[:8]
    static = dict(branches=branches, hidden=hidden, eps=eps, up_t=up_t)
    if max_tile:
        static["max_tile"] = max_tile
    tile_tag = f"_t{max_tile}" if max_tile else ""

    if inject:

        def core(
            residual: jax.Array,
            block: jax.Array,
            prev_injection: jax.Array,
            weight: jax.Array,
            down: jax.Array,
            up: jax.Array,
            inject_weight: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            return hc_decode_mix(
                residual,
                weight,
                down,
                up,
                inject_weight,
                block=block,
                prev_injection=prev_injection,
                **static,
            )

        name = (
            f"qwen4_exp::hc_decode_combine_c{branches}_h{hidden}_e{eps_tag}"
            f"{'_upt' if up_t else ''}{tile_tag}_v1"
        )
    else:

        def core(
            residual: jax.Array,
            block: jax.Array,
            prev_injection: jax.Array,
            weight: jax.Array,
            down: jax.Array,
            up: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            mixed, _, combined = hc_decode_mix(
                residual,
                weight,
                down,
                up,
                block=block,
                prev_injection=prev_injection,
                **static,
            )
            return mixed, combined

        name = (
            f"qwen4_exp::hc_decode_combine_c{branches}_h{hidden}_e{eps_tag}"
            f"_noinj{'_upt' if up_t else ''}{tile_tag}_v1"
        )

    op = pallas.jax_op(name, core)

    if inject:

        @op.register_fake
        def fake(residual, block, prev_injection, weight, down, up, inject_weight):
            tokens = residual.shape[0]
            return (
                residual.new_empty((tokens, hidden), dtype=torch.bfloat16),
                residual.new_empty((tokens, branches), dtype=torch.float32),
                residual.new_empty(residual.shape, dtype=torch.bfloat16),
            )

    else:

        @op.register_fake
        def fake_noinj(residual, block, prev_injection, weight, down, up):
            tokens = residual.shape[0]
            return (
                residual.new_empty((tokens, hidden), dtype=torch.bfloat16),
                residual.new_empty(residual.shape, dtype=torch.bfloat16),
            )

    return op


@functools.cache
def build_prefill_inj_mix_op(
    branches: int,
    hidden: int,
    eps: float,
    token_tile: int,
    partial: bool = False,
    norm_impl: str = "vpu",
    fold: bool = False,
):
    """TPU_QWEN4_HC_PREFILL_FUSED_MIX: prefill HC mix returning ``(mixed,
    injection)``; ``norm`` stays on chip. Callers gate it to whole tiles.
    ``fold``: the folded-weight kernel (TPU_QWEN4_HC_PREFILL_FOLD)."""
    eps_tag = hashlib.sha1(repr(eps).encode()).hexdigest()[:8]

    def core(
        x: jax.Array,
        weight: jax.Array,
        down: jax.Array,
        up: jax.Array,
        inject_weight: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return hc_prefill_mix(
            x,
            weight,
            down,
            up,
            inject_weight,
            branches=branches,
            hidden=hidden,
            eps=eps,
            token_tile=token_tile,
            partial=partial,
            norm_impl=norm_impl,
            fold=fold,
        )

    op = pallas.jax_op(
        f"qwen4_exp::hc_prefill_inj_c{branches}_h{hidden}_m{token_tile}_e{eps_tag}"
        f"{'_p' if partial else ''}{'' if norm_impl == 'vpu' else '_n' + norm_impl}"
        f"{'_fold' if fold else ''}_v1",
        core,
    )

    @op.register_fake
    def fake(x, weight, down, up, inject_weight):
        return (
            x.new_empty((x.shape[0], hidden), dtype=torch.bfloat16),
            x.new_empty((x.shape[0], branches), dtype=torch.float32),
        )

    return op


@functools.cache
def build_prefill_combine_mix_op(
    branches: int,
    hidden: int,
    eps: float,
    token_tile: int,
    partial: bool = False,
    norm_impl: str = "vpu",
    fold: bool = False,
    reread: bool = False,
):
    """The previous HC's combine fused ahead of this HC's prefill mix.
    ``reread``: the prologue's norm reads the combined rows back from VMEM
    (TPU_QWEN4_HC_PREFILL_DEFER_WHOLE's deferred boundary).

    Returns ``(mixed, injection, combined)``; ``combined`` is the hyper input
    this HC mixed (the next combine's residual).
    """
    eps_tag = hashlib.sha1(repr(eps).encode()).hexdigest()[:8]

    def core(
        residual: jax.Array,
        block: jax.Array,
        prev_injection: jax.Array,
        weight: jax.Array,
        down: jax.Array,
        up: jax.Array,
        inject_weight: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return hc_prefill_mix(
            residual,
            weight,
            down,
            up,
            inject_weight,
            branches=branches,
            hidden=hidden,
            eps=eps,
            token_tile=token_tile,
            block=block,
            prev_injection=prev_injection,
            partial=partial,
            norm_impl=norm_impl,
            fold=fold,
            **({"reread": True} if reread else {}),
        )

    op = pallas.jax_op(
        f"qwen4_exp::hc_prefill_combine_c{branches}_h{hidden}_m{token_tile}"
        f"_e{eps_tag}{'_p' if partial else ''}"
        f"{'' if norm_impl == 'vpu' else '_n' + norm_impl}{'_fold' if fold else ''}"
        f"{'_rr' if reread else ''}_v1",
        core,
    )

    @op.register_fake
    def fake(residual, block, prev_injection, weight, down, up, inject_weight):
        tokens = residual.shape[0]
        return (
            residual.new_empty((tokens, hidden), dtype=torch.bfloat16),
            residual.new_empty((tokens, branches), dtype=torch.float32),
            residual.new_empty(residual.shape, dtype=torch.bfloat16),
        )

    return op
