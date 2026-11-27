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
    piece_ranks: int = 0,
) -> Callable[..., tuple[torch.Tensor, ...]]:
    """`qk_norm_rope.qk_norm_rope` as a torch op, built once per config from
    module ``__init__`` (as `ops.msa.get_msa_op`). With `piece_ranks` the op
    takes ``qkv`` as the two row pieces of a sequence-parallel step over that
    many ranks (`qk_norm_rope.qk_norm_rope`'s pieces), then the rest."""
    norm_rope = functools.partial(
        qk_norm_rope.qk_norm_rope,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        eps=eps,
        q_head_major=q_head_major,
        rotate=rotate,
    )

    def _norm_rope_jax(
        qkv: jax.Array, cos_sin: jax.Array, weight: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        return norm_rope(qkv, cos_sin, weight)

    def _norm_rope_pieces_jax(
        qkv0: jax.Array, qkv1: jax.Array, cos_sin: jax.Array, weight: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        return norm_rope((qkv0, qkv1), cos_sin, weight, piece_ranks=piece_ranks)

    op = pallas.jax_op(
        f"pallas::minimax_m3_qk_norm_rope_q{num_heads}_kv{num_kv_heads}_d{head_dim}"
        f"_e{eps!r}_hm{int(q_head_major)}{'_rot' if rotate else ''}"
        f"{f'_pieces{piece_ranks}' if piece_ranks else ''}".replace(".", "p").replace(
            "-", "m"
        ),
        _norm_rope_pieces_jax if piece_ranks else _norm_rope_jax,
    )

    def _fake(*args):
        pieces = 2 if piece_ranks else 1
        qkv = args[0]
        num_tokens = pieces * qkv.shape[0]
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

    op.register_fake(_fake)
    return op
