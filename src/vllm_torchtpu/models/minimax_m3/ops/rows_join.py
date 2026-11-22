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
"""Torch entry point for `kernels.rows_join`."""

from collections.abc import Callable

import jax
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.models.minimax_m3.kernels import rows_join


def _join2(q0: jax.Array, s0: jax.Array, q1: jax.Array, s1: jax.Array) -> jax.Array:
    return rows_join.join_pieces(((q0, s0), (q1, s1)))


def _join3(
    q0: jax.Array,
    s0: jax.Array,
    q1: jax.Array,
    s1: jax.Array,
    q2: jax.Array,
    s2: jax.Array,
) -> jax.Array:
    return rows_join.join_pieces(((q0, s0), (q1, s1), (q2, s2)))


def _join4(
    q0: jax.Array,
    s0: jax.Array,
    q1: jax.Array,
    s1: jax.Array,
    q2: jax.Array,
    s2: jax.Array,
    q3: jax.Array,
    s3: jax.Array,
) -> jax.Array:
    return rows_join.join_pieces(((q0, s0), (q1, s1), (q2, s2), (q3, s3)))


def _fake(*operands: torch.Tensor) -> torch.Tensor:
    qs = operands[::2]
    return qs[0].new_empty(
        (sum(q.shape[0] for q in qs), qs[0].shape[1]), dtype=torch.bfloat16
    )


# Built at import: torch_tpu cannot create a jax_op inside a traced forward;
# jax_op takes annotated fixed signatures, so one per piece count.
_JOIN_OPS: dict[int, Callable[..., torch.Tensor]] = {}
for _pieces, _fn in ((2, _join2), (3, _join3), (4, _join4)):
    _op = pallas.jax_op(f"pallas::minimax_m3_rows_join_{_pieces}", _fn)
    _op.register_fake(_fake)
    _JOIN_OPS[_pieces] = _op


def join_rows_op(pieces: int) -> Callable[..., torch.Tensor] | None:
    """The op joining ``pieces`` e4m3 pieces given as ``q0, s0, q1, s1, ...``,
    or None for an unsupported count."""
    return _JOIN_OPS.get(pieces)
