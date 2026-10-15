# SPDX-License-Identifier: Apache-2.0
"""Torch custom-op boundaries for Qwen TPU kernels."""

import functools

from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.qwen4_exp.gated_residual import gated_residual_fn


@functools.cache
def build_gated_residual_op(branches: int, hidden: int, token_tile: int):
    op = pallas.jax_op(
        f"qwen4_exp::gated_residual_b{branches}_h{hidden}_m{token_tile}_v1",
        gated_residual_fn(branches, hidden, token_tile),
    )

    @op.register_fake
    def fake(residual, block, gates):
        return residual.new_empty(residual.shape)

    return op
