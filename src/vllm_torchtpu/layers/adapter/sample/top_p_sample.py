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
"""Torch bridge for the dense-row top-p recovery kernel
(``kernels/top_p_sample.py``)."""

import jax
import torch
from torch_tpu._internal import pallas

import vllm_torchtpu.envs as envs
from vllm_torchtpu.kernels import top_p_sample as kernel

HASH_MULTS = kernel.HASH_MULTS
ID_MULT = kernel.ID_MULT


def _fake(logits, recover_u, params, draft):
    return torch.empty((kernel.ROWS, 128), dtype=torch.int32, device=logits.device)


# Registered at import: Dynamo cannot trace a registration inside a forward.
_op = pallas.jax_op("pallas::top_p_rows", kernel.top_p_rows)
_op.register_fake(_fake)


# Ops for vocab-parallel shards side by side, by (shard width, vocab): built
# by `prepare_shards` before any forward traces `recover_shards`.
_shard_ops: dict[tuple[int, int], object] = {}


def prepare_shards(row_width: int, vocab: int) -> None:
    """Register the kernel for logits laid out as ``[tp, padded shard]`` rows
    of shards ``row_width`` wide over ``vocab`` ids. Call outside a forward."""
    if (row_width, vocab) in _shard_ops:
        return

    # torch_tpu reads the operand types from the annotations.
    def rows_of_shards(
        logits: jax.Array, recover_u: jax.Array, params: jax.Array, draft: jax.Array
    ) -> jax.Array:
        return kernel.top_p_rows(
            logits, recover_u, params, draft, row_width=row_width, vocab=vocab
        )

    op = pallas.jax_op(f"pallas::top_p_rows_w{row_width}_v{vocab}", rows_of_shards)
    op.register_fake(_fake)
    _shard_ops[(row_width, vocab)] = op


def applies_shards(logits: torch.Tensor, row_width: int, vocab: int) -> bool:
    """Whether recovering ``logits [rows, tp, padded shard]`` can run on the
    kernel (`prepare_shards` registered its op)."""
    return (
        envs.TPU_SAMPLER_TOP_P_KERNEL
        and logits.device.type == "tpu"
        and (row_width, vocab) in _shard_ops
        and logits.shape[-1] % 128 == 0
        and logits.shape[0] <= kernel.ROWS
    )


def recover_shards(
    logits: torch.Tensor,
    draft_token_ids: torch.Tensor,
    temperatures: torch.Tensor,
    top_p: torch.Tensor | None,
    recover_u: torch.Tensor,
    row_width: int,
    vocab: int,
) -> torch.Tensor:
    """`recover` of ``logits [rows, tp, padded shard]`` (vocab id ``s *
    row_width + w`` at ``[s, w]``) with uniforms of the same layout, or
    int32 ``[rows]`` stream keys the kernel makes them from."""
    rows = logits.shape[0]
    temperatures = temperatures.float().view(rows)
    top_p = torch.ones_like(temperatures) if top_p is None else top_p.float().view(rows)
    params = torch.stack([temperatures, top_p])
    draft = draft_token_ids.to(torch.int32).view(rows)
    out = _shard_ops[(row_width, vocab)](logits, recover_u, params, draft)
    return out[:rows, 0]


def applies(logits: torch.Tensor) -> bool:
    """Whether recovering ``logits [rows, vocab]`` can run on the kernel."""
    return (
        envs.TPU_SAMPLER_TOP_P_KERNEL
        and logits.device.type == "tpu"
        and logits.dim() == 2
        and logits.shape[-1] % 128 == 0
    )


def recover(
    logits: torch.Tensor,
    draft_token_ids: torch.Tensor,
    temperatures: torch.Tensor,
    top_p: torch.Tensor | None,
    recover_u: torch.Tensor,
) -> torch.Tensor:
    """int32 ``[rows]``: `rejection_sampler._recover_tokens` without top-k
    or draft probabilities (``top_p`` None: no top-p), 8 rows per call."""
    rows, vocab = logits.shape
    shape = (rows, vocab // 128, 128)
    logits, recover_u = logits.view(shape), recover_u.view(shape)
    temperatures = temperatures.float().view(rows)
    top_p = torch.ones_like(temperatures) if top_p is None else top_p.float().view(rows)
    draft = draft_token_ids.to(torch.int32).view(rows)
    tokens = []
    for lo in range(0, rows, kernel.ROWS):
        hi = min(lo + kernel.ROWS, rows)
        params = torch.stack([temperatures[lo:hi], top_p[lo:hi]])
        out = _op(logits[lo:hi], recover_u[lo:hi], params, draft[lo:hi])
        tokens.append(out[: hi - lo, 0])
    return tokens[0] if len(tokens) == 1 else torch.cat(tokens)
