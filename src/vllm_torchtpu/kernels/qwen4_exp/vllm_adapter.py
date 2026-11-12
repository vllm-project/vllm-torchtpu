# SPDX-License-Identifier: Apache-2.0
"""Torch custom-op boundaries for Qwen TPU kernels."""

import functools
import hashlib

import jax
import torch
from jax.experimental.pallas import tpu as pltpu
from torch_tpu._internal import pallas

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.qwen4_exp.gated_residual import gated_residual_fn
from vllm_torchtpu.kernels.qwen4_exp.hc_decode import hc_decode_mix
from vllm_torchtpu.kernels.qwen4_exp.hc_full import _combine_full_mix, _full_mix
from vllm_torchtpu.kernels.qwen4_exp.hc_norm import _normalize
from vllm_torchtpu.kernels.qwen4_exp.hc_resources import HcHardware
from vllm_torchtpu.kernels.qwen4_exp.hc_up import _up_epilogue


def current_hc_hardware() -> HcHardware | None:
    """Read layout only after worker TPU initialization; never claim a device."""
    # torch_tpu registers this public module dynamically.
    # pyrefly: ignore[missing-attribute]
    if (
        not getattr(torch, "tpu", None)
        or not torch.tpu.is_initialized()
        or not pltpu.is_tpu_device()
    ):
        return None
    return HcHardware.from_tpu_info(pltpu.get_tpu_info())


def _hardware_tag(hardware: HcHardware | None) -> str:
    return hashlib.sha1(repr(hardware).encode()).hexdigest()[:12]


@functools.cache
def build_gated_residual_op(
    branches: int, hidden: int, token_tile: int, hardware: HcHardware | None = None
):
    op = pallas.jax_op(
        f"qwen4_exp::gated_residual_b{branches}_h{hidden}_m{token_tile}"
        f"_l{_hardware_tag(hardware)}_v2",
        gated_residual_fn(branches, hidden, token_tile, hardware),
    )

    @op.register_fake
    def fake(residual, block, gates):
        return residual.new_empty(residual.shape)

    return op


def _hc_prefill_norm() -> str:
    """TPU_QWEN4_HC_PREFILL_NORM, lower-cased (the full-mix norm variant)."""
    return (envs.TPU_QWEN4_HC_PREFILL_NORM or "vpu").lower()


def _up_operand(up_t: bool):
    """The kernel's up operand from the checkpoint's [width, low rank] weight.

    Transposing here rather than at the Torch call site keeps every graph's
    operands (and their first-use order) the checkpoint parameters.
    """
    return (lambda up: up.T) if up_t else (lambda up: up)


@functools.cache
def build_full_mix_op(
    branches: int,
    hidden: int,
    eps: float,
    token_tile: int,
    hardware: HcHardware | None = None,
    up_t: bool = False,
    inject: bool = False,
):
    """``up_t``: the kernel reads the up projection as its [low rank, width]
    transpose, taken inside the op (a bitcast of the checkpoint's column-major
    weight); the op's operands stay the checkpoint tensors either way.
    ``inject``: it also takes the combine's gate weight and returns the BF16 gate
    projection [tokens, branches] in place of the normalized input."""
    eps_tag = hashlib.sha1(repr(eps).encode()).hexdigest()[:8]

    norm_impl = _hc_prefill_norm()
    static = dict(
        branches=branches,
        hidden=hidden,
        eps=eps,
        token_tile=token_tile,
        up_t=up_t,
        hardware=hardware,
        norm_impl=norm_impl,
    )
    _up = _up_operand(up_t)
    # torch.library.infer_schema takes explicit tensor parameters only.
    if inject:

        def core(
            x: jax.Array,
            weight: jax.Array,
            down: jax.Array,
            up: jax.Array,
            inject_weight: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            return _full_mix(x, weight, down, _up(up), inject_weight, **static)

    else:

        def core(
            x: jax.Array, weight: jax.Array, down: jax.Array, up: jax.Array
        ) -> tuple[jax.Array, jax.Array]:
            return _full_mix(x, weight, down, _up(up), **static)

    op = pallas.jax_op(
        f"qwen4_exp::hc_full_c{branches}_h{hidden}_m{token_tile}_e{eps_tag}"
        f"_l{_hardware_tag(hardware)}"
        f"{'_upt' if up_t else ''}{'_inj' if inject else ''}"
        f"{'' if norm_impl == 'vpu' else '_n' + norm_impl}_v2",
        core,
    )

    @op.register_fake
    def fake(x, weight, down, up, inject_weight=None):
        return (
            x.new_empty((x.shape[0], hidden), dtype=torch.bfloat16),
            x.new_empty(
                (x.shape[0], branches) if inject else x.shape, dtype=torch.bfloat16
            ),
        )

    return op


@functools.cache
def build_combine_full_mix_op(
    branches: int,
    hidden: int,
    eps: float,
    token_tile: int,
    hardware: HcHardware | None = None,
    up_t: bool = False,
    inject: bool = False,
):
    """``(mixed, norm, combined)``: the gated residual merge, then the full mix
    (``inject``: the gate projection in place of ``norm``, as ``build_full_mix_op``)."""
    eps_tag = hashlib.sha1(repr(eps).encode()).hexdigest()[:8]

    norm_impl = _hc_prefill_norm()
    static = dict(
        branches=branches,
        hidden=hidden,
        eps=eps,
        token_tile=token_tile,
        up_t=up_t,
        hardware=hardware,
        norm_impl=norm_impl,
    )
    _up = _up_operand(up_t)
    if inject:

        def core(
            residual: jax.Array,
            block: jax.Array,
            gates: jax.Array,
            weight: jax.Array,
            down: jax.Array,
            up: jax.Array,
            inject_weight: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            return _combine_full_mix(
                residual, block, gates, weight, down, _up(up), inject_weight, **static
            )

    else:

        def core(
            residual: jax.Array,
            block: jax.Array,
            gates: jax.Array,
            weight: jax.Array,
            down: jax.Array,
            up: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            return _combine_full_mix(
                residual, block, gates, weight, down, _up(up), **static
            )

    op = pallas.jax_op(
        f"qwen4_exp::hc_combine_full_c{branches}_h{hidden}_m{token_tile}_e{eps_tag}"
        f"_l{_hardware_tag(hardware)}"
        f"{'_upt' if up_t else ''}{'_inj' if inject else ''}"
        f"{'' if norm_impl == 'vpu' else '_n' + norm_impl}_v2",
        core,
    )

    @op.register_fake
    def fake(residual, block, gates, weight, down, up, inject_weight=None):
        rows = residual.shape[0]
        return (
            residual.new_empty((rows, hidden), dtype=torch.bfloat16),
            residual.new_empty(
                (rows, branches) if inject else residual.shape, dtype=torch.bfloat16
            ),
            residual.new_empty(residual.shape, dtype=torch.bfloat16),
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
def build_decode_mix_op(branches: int, hidden: int, eps: float, inject: bool):
    """Register the fused decode HC mix; callers gate it on decode_token_tile.

    Returns ``(mixed, injection)`` with ``inject`` (the combine's f32 gate,
    from the same normalized input), else ``mixed`` only.
    """
    eps_tag = hashlib.sha1(repr(eps).encode()).hexdigest()[:8]
    static = dict(branches=branches, hidden=hidden, eps=eps)
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
        f"{'_inj' if inject else ''}_v1",
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
def build_decode_combine_mix_op(branches: int, hidden: int, eps: float):
    """The previous HC's combine fused ahead of this HC's decode mix.

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
        return hc_decode_mix(
            residual,
            weight,
            down,
            up,
            inject_weight,
            branches=branches,
            hidden=hidden,
            eps=eps,
            block=block,
            prev_injection=prev_injection,
        )

    op = pallas.jax_op(
        f"qwen4_exp::hc_decode_combine_c{branches}_h{hidden}_e{eps_tag}_v1", core
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
