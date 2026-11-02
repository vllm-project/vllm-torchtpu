# Copyright 2026 Google LLC
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
"""Torch custom-op boundaries for Qwen TPU kernels."""

import functools
import hashlib

import jax
import torch
from jax.experimental.pallas import tpu as pltpu
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.qwen4_exp.gated_residual import gated_residual_fn
from vllm_torchtpu.kernels.qwen4_exp.hc_combine_mix import (
    combine_mix_inject,
    combine_mix_inject_broadcast,
    combine_mix_terminal,
)
from vllm_torchtpu.kernels.qwen4_exp.hc_full import _full_mix
from vllm_torchtpu.kernels.qwen4_exp.hc_inject_mix import (
    full_mix_inject,
    full_mix_inject_broadcast,
)
from vllm_torchtpu.kernels.qwen4_exp.hc_resources import HcHardware


def current_hc_hardware() -> HcHardware | None:
    """Read layout only after worker TPU initialization; never claim a device."""
    # torch_tpu registers this public module dynamically.
    # pyrefly: ignore[missing-attribute]
    if not torch.tpu.is_initialized() or not pltpu.is_tpu_device():
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


@functools.cache
def build_full_mix_op(
    branches: int,
    hidden: int,
    eps: float,
    token_tile: int,
    hardware: HcHardware | None = None,
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
            hardware=hardware,
        )

    op = pallas.jax_op(
        f"qwen4_exp::hc_full_c{branches}_h{hidden}_m{token_tile}_e{eps_tag}"
        f"_l{_hardware_tag(hardware)}_v2",
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
def build_full_mix_inject_op(
    branches: int,
    hidden: int,
    eps: float,
    token_tile: int,
    hardware: HcHardware | None = None,
):
    eps_tag = hashlib.sha1(repr(eps).encode()).hexdigest()[:8]

    def core(
        x: jax.Array,
        weight: jax.Array,
        down: jax.Array,
        up: jax.Array,
        inject: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return full_mix_inject(
            x,
            weight,
            down,
            up,
            inject,
            branches=branches,
            hidden=hidden,
            eps=eps,
            token_tile=token_tile,
            hardware=hardware,
        )

    op = pallas.jax_op(
        f"qwen4_exp::hc_inject_c{branches}_h{hidden}_m{token_tile}_e{eps_tag}"
        f"_l{_hardware_tag(hardware)}_v1",
        core,
    )

    @op.register_fake
    def fake(x, weight, down, up, inject):
        return (
            x.new_empty((x.shape[0], hidden), dtype=torch.bfloat16),
            x.new_empty((x.shape[0], branches), dtype=torch.bfloat16),
        )

    return op


@functools.cache
def build_full_mix_inject_broadcast_op(
    branches: int,
    hidden: int,
    eps: float,
    token_tile: int,
    hardware: HcHardware | None = None,
):
    eps_tag = hashlib.sha1(repr(eps).encode()).hexdigest()[:8]

    def core(
        x: jax.Array,
        weight: jax.Array,
        down: jax.Array,
        up: jax.Array,
        inject: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return full_mix_inject_broadcast(
            x,
            weight,
            down,
            up,
            inject,
            branches=branches,
            hidden=hidden,
            eps=eps,
            token_tile=token_tile,
            hardware=hardware,
        )

    op = pallas.jax_op(
        f"qwen4_exp::hc_inject_bcast_c{branches}_h{hidden}_m{token_tile}_e{eps_tag}"
        f"_l{_hardware_tag(hardware)}_v1",
        core,
    )

    @op.register_fake
    def fake(x, weight, down, up, inject):
        return (
            x.new_empty((x.shape[0], hidden), dtype=torch.bfloat16),
            x.new_empty((x.shape[0], branches), dtype=torch.bfloat16),
        )

    return op


@functools.cache
def build_combine_mix_broadcast_op(
    branches: int,
    hidden: int,
    eps: float,
    token_tile: int,
    hardware: HcHardware | None = None,
):
    eps_tag = hashlib.sha1(repr(eps).encode()).hexdigest()[:8]

    def core(
        residual: jax.Array,
        block: jax.Array,
        in_logits: jax.Array,
        weight: jax.Array,
        down: jax.Array,
        up: jax.Array,
        inject: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return combine_mix_inject_broadcast(
            residual,
            block,
            in_logits,
            weight,
            down,
            up,
            inject,
            branches=branches,
            hidden=hidden,
            eps=eps,
            token_tile=token_tile,
            hardware=hardware,
        )

    op = pallas.jax_op(
        f"qwen4_exp::hc_combine_bcast_c{branches}_h{hidden}_m{token_tile}_e{eps_tag}"
        f"_l{_hardware_tag(hardware)}_v1",
        core,
    )

    @op.register_fake
    def fake(residual, block, in_logits, weight, down, up, inject):
        return (
            residual.new_empty(
                (residual.shape[0], branches * hidden), dtype=torch.bfloat16
            ),
            residual.new_empty((residual.shape[0], hidden), dtype=torch.bfloat16),
            residual.new_empty((residual.shape[0], branches), dtype=torch.bfloat16),
        )

    return op


@functools.cache
def build_combine_mix_op(
    branches: int,
    hidden: int,
    eps: float,
    token_tile: int,
    hardware: HcHardware | None = None,
):
    eps_tag = hashlib.sha1(repr(eps).encode()).hexdigest()[:8]

    def core(
        residual: jax.Array,
        block: jax.Array,
        in_logits: jax.Array,
        weight: jax.Array,
        down: jax.Array,
        up: jax.Array,
        inject: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return combine_mix_inject(
            residual,
            block,
            in_logits,
            weight,
            down,
            up,
            inject,
            branches=branches,
            hidden=hidden,
            eps=eps,
            token_tile=token_tile,
            hardware=hardware,
        )

    op = pallas.jax_op(
        f"qwen4_exp::hc_combine_mix_c{branches}_h{hidden}_m{token_tile}_e{eps_tag}"
        f"_l{_hardware_tag(hardware)}_v2",
        core,
    )

    @op.register_fake
    def fake(residual, block, in_logits, weight, down, up, inject):
        return (
            residual.new_empty(residual.shape, dtype=torch.bfloat16),
            residual.new_empty((residual.shape[0], hidden), dtype=torch.bfloat16),
            residual.new_empty((residual.shape[0], branches), dtype=torch.bfloat16),
        )

    return op


@functools.cache
def build_combine_mix_terminal_op(
    branches: int,
    hidden: int,
    eps: float,
    token_tile: int,
    hardware: HcHardware | None = None,
):
    eps_tag = hashlib.sha1(repr(eps).encode()).hexdigest()[:8]

    def core(
        residual: jax.Array,
        block: jax.Array,
        in_logits: jax.Array,
        weight: jax.Array,
        down: jax.Array,
        up: jax.Array,
    ) -> jax.Array:
        return combine_mix_terminal(
            residual,
            block,
            in_logits,
            weight,
            down,
            up,
            branches=branches,
            hidden=hidden,
            eps=eps,
            token_tile=token_tile,
            hardware=hardware,
        )

    op = pallas.jax_op(
        f"qwen4_exp::hc_combine_mix_terminal_c{branches}_h{hidden}_m{token_tile}"
        f"_e{eps_tag}_l{_hardware_tag(hardware)}_v1",
        core,
    )

    @op.register_fake
    def fake(residual, block, in_logits, weight, down, up):
        return residual.new_empty((residual.shape[0], hidden), dtype=torch.bfloat16)

    return op
