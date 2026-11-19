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
"""In-place update of the leading rows of a persistent module buffer."""

import jax
import torch
from torch_tpu._internal import pallas


def rows_update_jax(buf: jax.Array, rows: jax.Array) -> jax.Array:
    """`buf[:rows.shape[0]] = rows` as one row update of a donated buffer."""
    return jax.lax.dynamic_update_slice(buf, rows.astype(buf.dtype), (0, 0))


# `buf[:n] = rows` on a module buffer reaches XLA as a graph input that is
# mutated but not donated -- aot_autograd returns the new value and writes it
# back after the graph -- plus a view write lowered as two row updates of the
# same base. Both force a full copy of the buffer in every graph that writes
# it. Routing the write through a donating op with a `copy_` writeback (the
# KV-cache pattern) marks the buffer a donor, so XLA updates it in place. Built
# at import, not lazily: dynamo cannot trace the op's construction from inside
# a compiled forward. (Named for its first user, the MLA indexer's top-k table.)
ROWS_UPDATE_OP = pallas.jax_op(
    "pallas::indexer_topk_rows_update", rows_update_jax, donate_argnums=(0,)
)


# torch.compile traces the forward with fake tensors (shape and dtype, no data),
# so it can't run the JAX function to learn what this op returns. This fake
# tells it: a tensor shaped like `buf`, i.e. the whole updated buffer, which the
# caller copies back into `buf`. The real `rows_update_jax` runs only when the
# compiled graph executes.
def _fake_rows_update(buf: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    del rows
    return torch.empty_like(buf)


ROWS_UPDATE_OP.register_fake(_fake_rows_update)


def update_rows_(buf: torch.Tensor, rows: torch.Tensor) -> None:
    """`buf[: rows.shape[0]] = rows`, in place and without a full-buffer copy."""
    buf.copy_(ROWS_UPDATE_OP(buf, rows))
