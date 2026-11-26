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
(``kernels/top_p_sample.py``, TPU_SAMPLER_TOP_P_KERNEL)."""

import torch

import vllm_torchtpu.envs as envs
from vllm_torchtpu.kernels import top_p_sample as kernel


def _fake(logits, recover_u, params, draft):
    return torch.empty((kernel.ROWS, 128), dtype=torch.int32, device=logits.device)


_op = None


def _get_op():
    """The registered op; built at first use outside a forward (`arm`)."""
    global _op
    if _op is None:
        from torch_tpu._internal import pallas

        _op = pallas.jax_op("pallas::top_p_rows", kernel.top_p_rows)
        _op.register_fake(_fake)
    return _op


def arm() -> bool:
    """Register the op eagerly (Dynamo cannot trace a registration inside a
    forward). Returns whether the kernel path is on."""
    if not envs.TPU_SAMPLER_TOP_P_KERNEL:
        return False
    _get_op()
    return True


def applies(logits: torch.Tensor) -> bool:
    """Whether recovering ``logits [rows, vocab]`` can run on the kernel."""
    return (
        envs.TPU_SAMPLER_TOP_P_KERNEL
        and _op is not None
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
    op = _get_op()
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
        out = op(logits[lo:hi], recover_u[lo:hi], params, draft[lo:hi])
        tokens.append(out[: hi - lo, 0])
    return tokens[0] if len(tokens) == 1 else torch.cat(tokens)
