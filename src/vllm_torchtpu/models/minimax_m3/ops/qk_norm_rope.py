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
"""Torch entry point for the fused qk norm + rope kernel."""

import functools
from collections.abc import Callable

import jax
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.models.minimax_m3.kernels import qk_norm_rope


@functools.cache
def get_qk_norm_rope_op(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float,
    q_head_major: bool,
    rotate: bool = False,
    pieces: int = 1,
    row_blocks: int = 1,
) -> Callable[..., tuple[torch.Tensor, ...]]:
    """`qk_norm_rope.qk_norm_rope` as a torch op, built once per config from
    module ``__init__`` (as `ops.msa.get_msa_op`). With ``pieces=2`` the op
    takes two qkv row pieces (`qk_norm_rope`'s `row_blocks`) before
    ``cos_sin`` and ``weight``."""
    if pieces not in (1, 2):
        raise ValueError(f"qk_norm_rope ops take 1 or 2 qkv pieces, got {pieces}")

    def run(
        qkv: jax.Array | tuple[jax.Array, jax.Array],
        cos_sin: jax.Array,
        weight: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        return qk_norm_rope.qk_norm_rope(
            qkv,
            cos_sin,
            weight,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            eps=eps,
            q_head_major=q_head_major,
            rotate=rotate,
            row_blocks=row_blocks,
        )

    # torch op schemas take no varargs: one signature per piece count.
    def _norm_rope_jax(
        qkv: jax.Array, cos_sin: jax.Array, weight: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        return run(qkv, cos_sin, weight)

    def _pieces_norm_rope_jax(
        qkv0: jax.Array, qkv1: jax.Array, cos_sin: jax.Array, weight: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        return run((qkv0, qkv1), cos_sin, weight)

    op = pallas.jax_op(
        f"pallas::minimax_m3_qk_norm_rope_q{num_heads}_kv{num_kv_heads}_d{head_dim}"
        f"_e{eps!r}_hm{int(q_head_major)}{'_rot' if rotate else ''}"
        f"{f'_p{pieces}x{row_blocks}' if pieces > 1 else ''}".replace(".", "p").replace(
            "-", "m"
        ),
        _norm_rope_jax if pieces == 1 else _pieces_norm_rope_jax,
    )

    def _shapes(qkv):
        num_tokens = qkv.shape[0] * pieces
        q_shape = (
            (num_heads, num_tokens, head_dim)
            if q_head_major
            else (num_tokens, num_heads * head_dim)
        )
        kv = qkv.new_empty((num_tokens, num_kv_heads * head_dim))
        return (
            qkv.new_empty(q_shape),
            kv,
            torch.empty_like(kv),
            torch.empty_like(kv),
            qkv.new_empty((num_tokens, head_dim)),
        )

    def _fake(qkv, cos_sin, weight):
        return _shapes(qkv)

    def _pieces_fake(qkv0, qkv1, cos_sin, weight):
        return _shapes(qkv0)

    op.register_fake(_fake if pieces == 1 else _pieces_fake)
    return op
