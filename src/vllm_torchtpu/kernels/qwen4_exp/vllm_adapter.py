# SPDX-License-Identifier: Apache-2.0
"""Torch custom-op boundaries for Qwen TPU kernels."""

import functools
import hashlib

import jax
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.qwen4_exp.gated_residual import gated_residual_fn
from vllm_torchtpu.kernels.qwen4_exp.hc_decode import hc_decode_mix
from vllm_torchtpu.kernels.qwen4_exp.hc_decode_cmp import hc_decode_mix_auto
from vllm_torchtpu.kernels.qwen4_exp.hc_decode_wide import (
    MODES as WIDE_MODES,
    hc_decode_mix_wide_auto,
)
from vllm_torchtpu.kernels.qwen4_exp.hc_full import _full_mix
from vllm_torchtpu.kernels.qwen4_exp.hc_norm import _normalize
from vllm_torchtpu.kernels.qwen4_exp.hc_prefill_cmp import hc_prefill_mix_cmp
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


def _decode_mix_choice():
    """The fused decode HC mix and its op-name tag.

    TPU_QWEN4_HC_DECODE_WIDE (q38 HC-HIGHROWS, 0 = off) 1 / 2 runs
    hc_decode_wide.py's served-order / cmp-math mode at 160-511 rows (a
    multiple of 16); TPU_QWEN4_HC_DECODE_COMPARE runs hc_decode_cmp.py at
    16-256 rows. They are exclusive. TPU_QWEN4_HC_DECODE_COMPARE_SIGMOID
    (q38 HC-HIGHROWS-2 CMP-S, only with COMPARE) takes the served sigmoid in
    the cmp mix term (_cmps). Unset, it is hc_decode_mix, untagged.
    """
    wide = envs.TPU_QWEN4_HC_DECODE_WIDE
    if wide:
        if wide not in WIDE_MODES:
            raise ValueError(
                f"TPU_QWEN4_HC_DECODE_WIDE={wide}: expected 0 (off), 1, 2 or 3"
            )
        cmp_set = envs.TPU_QWEN4_HC_DECODE_COMPARE
        if cmp_set or envs.TPU_QWEN4_HC_DECODE_COMPARE_SIGMOID:
            raise ValueError(
                "TPU_QWEN4_HC_DECODE_WIDE and TPU_QWEN4_HC_DECODE_COMPARE are "
                "exclusive"
            )
        return functools.partial(hc_decode_mix_wide_auto, mode=wide), f"_w{wide}"
    if envs.TPU_QWEN4_HC_DECODE_COMPARE_SIGMOID:
        if not envs.TPU_QWEN4_HC_DECODE_COMPARE:
            raise ValueError(
                "TPU_QWEN4_HC_DECODE_COMPARE_SIGMOID needs "
                "TPU_QWEN4_HC_DECODE_COMPARE=1"
            )
        return functools.partial(hc_decode_mix_auto, sigmoid=True), "_cmps"
    if envs.TPU_QWEN4_HC_DECODE_COMPARE:
        return hc_decode_mix_auto, "_cmp"
    return hc_decode_mix, ""


@functools.cache
def build_decode_mix_op(branches: int, hidden: int, eps: float, inject: bool):
    """Register the fused decode HC mix; callers gate it on decode_token_tile.

    Returns ``(mixed, injection)`` with ``inject`` (the combine's f32 gate,
    from the same normalized input), else ``mixed`` only.
    """
    eps_tag = hashlib.sha1(repr(eps).encode()).hexdigest()[:8]
    max_tile = envs.TPU_QWEN4_HC_DECODE_TILE_ROWS
    static = dict(branches=branches, hidden=hidden, eps=eps, max_tile=max_tile)
    mix, cmp_tag = _decode_mix_choice()
    if inject:

        def core(
            x: jax.Array,
            weight: jax.Array,
            down: jax.Array,
            up: jax.Array,
            inject_weight: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            return mix(x, weight, down, up, inject_weight, **static)

    else:

        def core(
            x: jax.Array, weight: jax.Array, down: jax.Array, up: jax.Array
        ) -> jax.Array:
            return mix(x, weight, down, up, **static)[0]

    op = pallas.jax_op(
        f"qwen4_exp::hc_decode_c{branches}_h{hidden}_e{eps_tag}"
        f"{'_inj' if inject else ''}{f'_t{max_tile}' if max_tile else ''}"
        f"{cmp_tag}_v1",
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
    branches: int, hidden: int, eps: float, inject: bool = True
):
    """The previous HC's combine fused ahead of this HC's decode mix.

    Returns ``(mixed, injection, combined)``, or ``(mixed, combined)`` without
    ``inject`` (the final mixer); ``combined`` is the hyper input this HC
    mixed (the next combine's residual).
    """
    eps_tag = hashlib.sha1(repr(eps).encode()).hexdigest()[:8]
    max_tile = envs.TPU_QWEN4_HC_DECODE_TILE_ROWS
    static = dict(branches=branches, hidden=hidden, eps=eps, max_tile=max_tile)
    tile_tag = f"_t{max_tile}" if max_tile else ""
    mix, impl_tag = _decode_mix_choice()
    tile_tag += impl_tag
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
            return mix(
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
            f"{tile_tag}_v1"
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
            mixed, _, combined = mix(
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
            f"_noinj{tile_tag}_v1"
        )

    op = pallas.jax_op(name, core)

    def mixed_like(residual):
        return residual.new_empty((residual.shape[0], hidden), dtype=torch.bfloat16)

    def combined_like(residual):
        return residual.new_empty(residual.shape, dtype=torch.bfloat16)

    if inject:

        @op.register_fake
        def fake(residual, block, prev_injection, weight, down, up, inject_weight):
            gate = residual.new_empty(
                (residual.shape[0], branches), dtype=torch.float32
            )
            return mixed_like(residual), gate, combined_like(residual)

    else:

        @op.register_fake
        def fake(residual, block, prev_injection, weight, down, up):
            return mixed_like(residual), combined_like(residual)

    return op


@functools.cache
def build_prefill_inj_mix_op(
    branches: int,
    hidden: int,
    eps: float,
    token_tile: int,
    partial: bool = False,
    norm_impl: str = "vpu",
    cmp: bool = False,
    down_asm: bool = False,
):
    """TPU_QWEN4_HC_PREFILL_FUSED_MIX: prefill HC mix returning ``(mixed,
    injection)``; ``norm`` stays on chip. Callers gate it to whole tiles.
    ``cmp`` (TPU_QWEN4_HC_PREFILL_COMPARE): hc_prefill_cmp's kernel, which
    ignores ``norm_impl``. ``down_asm``: TPU_QWEN4_HC_PREFILL_DOWN_ASM (not
    with ``cmp``)."""
    eps_tag = hashlib.sha1(repr(eps).encode()).hexdigest()[:8]
    if cmp and down_asm:
        raise ValueError(
            "TPU_QWEN4_HC_PREFILL_COMPARE and TPU_QWEN4_HC_PREFILL_DOWN_ASM cannot be combined"
        )
    mix = hc_prefill_mix_cmp if cmp else hc_prefill_mix
    # hc_prefill_cmp's kernel takes no PSTEP-HC kwarg (merge of both lines).
    extra = {} if cmp else {"down_asm": down_asm}

    def core(
        x: jax.Array,
        weight: jax.Array,
        down: jax.Array,
        up: jax.Array,
        inject_weight: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return mix(
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
            **extra,
        )

    op = pallas.jax_op(
        f"qwen4_exp::hc_prefill_inj_c{branches}_h{hidden}_m{token_tile}_e{eps_tag}"
        f"{'_p' if partial else ''}"
        f"{'_cmp' if cmp else '' if norm_impl == 'vpu' else '_n' + norm_impl}"
        f"{'_da' if down_asm else ''}_v1",
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
    cmp: bool = False,
    reread: bool = False,
    down_asm: bool = False,
):
    """The previous HC's combine fused ahead of this HC's prefill mix.

    Returns ``(mixed, injection, combined)``; ``combined`` is the hyper input
    this HC mixed (the next combine's residual). ``cmp`` as in
    ``build_prefill_inj_mix_op``. ``reread``: the norm reads the combined rows
    back from VMEM (TPU_QWEN4_HC_PREFILL_FOLD); ``down_asm``:
    TPU_QWEN4_HC_PREFILL_DOWN_ASM (neither with ``cmp``).
    """
    eps_tag = hashlib.sha1(repr(eps).encode()).hexdigest()[:8]
    if cmp and (reread or down_asm):
        raise ValueError(
            "TPU_QWEN4_HC_PREFILL_COMPARE cannot be combined with "
            "TPU_QWEN4_HC_PREFILL_FOLD or TPU_QWEN4_HC_PREFILL_DOWN_ASM"
        )
    mix = hc_prefill_mix_cmp if cmp else hc_prefill_mix
    # hc_prefill_cmp's kernel takes no PSTEP-HC kwarg (merge of both lines).
    extra = {} if cmp else {"reread": reread, "down_asm": down_asm}

    def core(
        residual: jax.Array,
        block: jax.Array,
        prev_injection: jax.Array,
        weight: jax.Array,
        down: jax.Array,
        up: jax.Array,
        inject_weight: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return mix(
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
            **extra,
        )

    op = pallas.jax_op(
        f"qwen4_exp::hc_prefill_combine_c{branches}_h{hidden}_m{token_tile}"
        f"_e{eps_tag}{'_p' if partial else ''}"
        f"{'_cmp' if cmp else '' if norm_impl == 'vpu' else '_n' + norm_impl}"
        f"{'_rr' if reread else ''}{'_da' if down_asm else ''}_v1",
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
