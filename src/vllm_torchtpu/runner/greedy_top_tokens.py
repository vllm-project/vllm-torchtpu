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
"""Greedy steps sampled from the vocabulary-parallel argmax.

With ``TPU_GREEDY_TOP_TOKENS``, a step whose requests are all greedy, ask for
no logprobs and carry no grammar takes ``LogitsProcessor.get_top_tokens``:
each rank argmaxes its slice of the LM head and only the (value, index) pairs
are exchanged, so the full logits are never gathered. Ties resolve to the
lowest token id, the same as ``torch.argmax`` over the full logits: the local
max keeps the first index of its slice and the cross-rank argmax keeps the
first rank, which owns the lower ids.

Importing this module replaces ``LogitsProcessor.get_top_tokens`` with
``tpu_get_top_tokens``, which returns the same tokens as vLLM's for a BF16
head and adds:

- a W8A8 FP8 quantized LM head (``quantize_lm_head``), whose local
  max+argmax runs in fused Pallas ops registered by
  ``_ensure_tpu_quant_ops_registered``;
- the shard's vocab start and padding as tensor buffers, so compiled graphs
  do not bake rank-specific Python integers;
- a row-major (``dim=0``) all-gather of the pairs.

It also adds ``LogitsProcessor.get_topk_candidates``: for sampled steps whose
requests all have ``1 <= top_k <= 32`` (``topk_candidates_step_eligible``),
each rank keeps its exact local top-32 and only those ``(logit, id)`` pairs
are exchanged; ``RejectionSampler.fused_verify_and_sample_candidates`` then
samples and verifies over the ``32 * tp_size`` candidates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import torch
from vllm.lora.layers import LogitsProcessorWithLoRA
from vllm.model_executor.layers import logits_processor as lp_module
from vllm.model_executor.layers.logits_processor import LogitsProcessor

from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )
    from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

logger = init_logger(__name__)

_TPU_QUANT_OPS: dict[str, Any] = {}
_FP8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
LOCAL_TOPK_CANDIDATES = 32
_LOCAL_TOPK_BLOCK_SIZE = 64
# Keep as many 64-wide blocks as candidates. The r-th largest local logit
# has at most r - 1 blocks with a larger block max, so its block is always
# kept and the candidates are the exact local top-32. Eight blocks were exact
# only up to k = 8, but topk_candidates_step_eligible admits top_k <= 32
# (Qwen's generation_config default is top_k = 20).
_LOCAL_TOPK_NUM_BLOCKS = LOCAL_TOPK_CANDIDATES


def _quantize_fp8_per_row(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D tensor along ``axis=-1`` to ``torch.float8_e4m3fn``."""
    t_f32 = tensor.to(torch.float32)
    abs_max = t_f32.abs().amax(dim=-1, keepdim=True)
    dtype_max_recip = torch.tensor(
        1.0 / _FP8_E4M3_MAX, dtype=torch.float32, device=tensor.device
    )
    scale = abs_max * dtype_max_recip
    scale_inv = torch.where(scale == 0, torch.zeros_like(scale), 1.0 / scale)
    q = torch.clamp(t_f32 * scale_inv, min=-_FP8_E4M3_MAX, max=_FP8_E4M3_MAX).to(
        torch.float8_e4m3fn
    )
    return q, scale.squeeze(-1).to(torch.float32)


def _parse_lm_head_quant(mode: str | None) -> str | None:
    """``"fp8"`` for a W8A8 FP8 head, ``None`` to keep it in BF16."""
    raw = (mode or "").strip().lower()
    if raw in ("", "0", "false", "no", "off", "none", "bf16", "bfloat16"):
        return None
    if raw in ("fp8", "w8a8", "float8", "float8_e4m3fn"):
        return "fp8"
    raise ValueError(
        f"Unsupported LM-head quantization {mode!r}; expected 'fp8' ('w8a8') "
        "or 'bf16' ('none' / 'off')."
    )


def _ensure_tpu_quant_ops_registered() -> dict[str, Any]:
    """Lazily register fused JAX/Pallas ``W8A8`` local max+argmax and top-32
    ops on TPU before ``@torch.compile`` traces ``tpu_get_top_tokens``."""
    if _TPU_QUANT_OPS:
        return _TPU_QUANT_OPS

    import jax
    import jax.numpy as jnp
    from torch_tpu._internal import pallas

    from vllm_torchtpu.kernels.quantized_matmul.util import (
        quantize_tensor as jax_quantize_tensor,
    )

    # pallas.jax_op uses inspect.signature(..., eval_str=True), which resolves
    # stringified PEP-563 annotations against module globals().
    globals()["jax"] = jax

    def _w8a8_2d_logits_jax(x: jax.Array, w_q: jax.Array, w_s: jax.Array) -> jax.Array:
        orig_b = x.shape[0]
        if orig_b == 1:
            x = jnp.pad(x, ((0, 1), (0, 0)))
        # hidden_states (x) is replicated identically across all TP shards, so
        # the positive per-row activation scale x_s[i] > 0 is identical across
        # ranks for each row i and strictly monotonic under intra-shard and
        # cross-shard max/argmax (matching the CPU path).
        x_q, _ = jax_quantize_tensor(x, jnp.float8_e4m3fn)
        logits = (
            jax.lax.dot_general(
                x_q,
                w_q,
                dimension_numbers=(((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            * w_s[None, :]
        )
        if orig_b == 1:
            logits = logits[:1]
        return logits

    def _w8a8_fused_max_jax(
        x: jax.Array, w_q: jax.Array, w_s: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        logits = _w8a8_2d_logits_jax(x, w_q, w_s)
        vals = jnp.max(logits, axis=-1)
        idx = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        return vals, idx

    def _w8a8_fused_max_padded_jax(
        x: jax.Array, w_q: jax.Array, w_s: jax.Array, num_pad: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        logits = _w8a8_2d_logits_jax(x, w_q, w_s)
        shard_vocab = logits.shape[-1]
        valid_cols = shard_vocab - num_pad.astype(jnp.int32)
        col_idx = jnp.arange(shard_vocab, dtype=jnp.int32)
        logits = jnp.where(col_idx[None, :] >= valid_cols, -jnp.inf, logits)
        vals = jnp.max(logits, axis=-1)
        idx = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        return vals, idx

    def _extract_local_topk32_jax(
        logits: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        t_rows, shard_vocab = logits.shape
        block_size = _LOCAL_TOPK_BLOCK_SIZE
        num_local_blocks = _LOCAL_TOPK_NUM_BLOCKS
        if shard_vocab < num_local_blocks * block_size:
            if shard_vocab < LOCAL_TOPK_CANDIDATES:
                pad_cols = LOCAL_TOPK_CANDIDATES - shard_vocab
                logits_pad = jnp.pad(
                    logits.astype(jnp.float32),
                    ((0, 0), (0, pad_cols)),
                    constant_values=-jnp.inf,
                )
            else:
                logits_pad = logits.astype(jnp.float32)
            vals32, ids32 = jax.lax.top_k(logits_pad, LOCAL_TOPK_CANDIDATES)
            return vals32, ids32.astype(jnp.int32)

        if shard_vocab % block_size != 0:
            pad_cols = block_size - (shard_vocab % block_size)
            lg_padded = jnp.pad(
                logits,
                ((0, 0), (0, pad_cols)),
                constant_values=jnp.array(-jnp.inf, dtype=logits.dtype),
            )
        else:
            lg_padded = logits
        n_blocks = lg_padded.shape[1] // block_size
        blocks = lg_padded.reshape(t_rows, n_blocks, block_size)
        block_max = jnp.max(blocks, axis=-1)
        _, blk_ids = jax.lax.top_k(block_max, num_local_blocks)
        local_cand512 = jnp.take_along_axis(
            blocks, blk_ids[:, :, None], axis=1
        ).reshape(t_rows, num_local_blocks * block_size)
        intra = jnp.arange(block_size, dtype=jnp.int32)[None, None, :]
        local_ids512 = (
            blk_ids.astype(jnp.int32)[:, :, None] * block_size + intra
        ).reshape(t_rows, num_local_blocks * block_size)
        vals32, idx32 = jax.lax.top_k(
            local_cand512.astype(jnp.float32), LOCAL_TOPK_CANDIDATES
        )
        ids32 = jnp.take_along_axis(local_ids512, idx32, axis=-1)
        return vals32, ids32.astype(jnp.int32)

    def _w8a8_fused_topk32_jax(
        x: jax.Array, w_q: jax.Array, w_s: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        orig_b = x.shape[0]
        if orig_b == 1:
            x = jnp.pad(x, ((0, 1), (0, 0)))
        x_q, x_s = jax_quantize_tensor(x, jnp.float8_e4m3fn)
        logits = (
            jax.lax.dot_general(
                x_q,
                w_q,
                dimension_numbers=(((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            * w_s[None, :]
        )
        if orig_b == 1:
            logits = logits[:1]
            x_s = x_s[:1]
        vals32, ids32 = _extract_local_topk32_jax(logits)
        return vals32 * x_s, ids32

    def _w8a8_fused_topk32_padded_jax(
        x: jax.Array, w_q: jax.Array, w_s: jax.Array, num_pad: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        orig_b = x.shape[0]
        if orig_b == 1:
            x = jnp.pad(x, ((0, 1), (0, 0)))
        x_q, x_s = jax_quantize_tensor(x, jnp.float8_e4m3fn)
        logits = (
            jax.lax.dot_general(
                x_q,
                w_q,
                dimension_numbers=(((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            * w_s[None, :]
        )
        if orig_b == 1:
            logits = logits[:1]
            x_s = x_s[:1]
        shard_vocab = logits.shape[-1]
        valid_cols = shard_vocab - num_pad.astype(jnp.int32)
        col_idx = jnp.arange(shard_vocab, dtype=jnp.int32)
        logits = jnp.where(
            col_idx[None, :] >= valid_cols,
            jnp.array(-jnp.inf, dtype=logits.dtype),
            logits,
        )
        vals32, ids32 = _extract_local_topk32_jax(logits)
        return vals32 * x_s, ids32

    def _fake_3arg(x: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor):
        return (
            torch.empty((x.shape[0],), dtype=torch.float32, device=x.device),
            torch.empty((x.shape[0],), dtype=torch.int32, device=x.device),
        )

    def _fake_4arg(
        x: torch.Tensor,
        w_q: torch.Tensor,
        w_s: torch.Tensor,
        num_pad: torch.Tensor,
    ):
        return (
            torch.empty((x.shape[0],), dtype=torch.float32, device=x.device),
            torch.empty((x.shape[0],), dtype=torch.int32, device=x.device),
        )

    def _fake_topk32_3arg(x: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor):
        return (
            torch.empty(
                (x.shape[0], LOCAL_TOPK_CANDIDATES),
                dtype=torch.float32,
                device=x.device,
            ),
            torch.empty(
                (x.shape[0], LOCAL_TOPK_CANDIDATES),
                dtype=torch.int32,
                device=x.device,
            ),
        )

    def _fake_topk32_4arg(
        x: torch.Tensor,
        w_q: torch.Tensor,
        w_s: torch.Tensor,
        num_pad: torch.Tensor,
    ):
        return (
            torch.empty(
                (x.shape[0], LOCAL_TOPK_CANDIDATES),
                dtype=torch.float32,
                device=x.device,
            ),
            torch.empty(
                (x.shape[0], LOCAL_TOPK_CANDIDATES),
                dtype=torch.int32,
                device=x.device,
            ),
        )

    for key, op_name, fn, fake_fn in (
        ("w8a8", "pallas::w8a8_fused_max_v2", _w8a8_fused_max_jax, _fake_3arg),
        (
            "w8a8_padded",
            "pallas::w8a8_fused_max_padded_v2",
            _w8a8_fused_max_padded_jax,
            _fake_4arg,
        ),
        (
            "w8a8_topk32",
            "pallas::w8a8_fused_topk32_v2",
            _w8a8_fused_topk32_jax,
            _fake_topk32_3arg,
        ),
        (
            "w8a8_topk32_padded",
            "pallas::w8a8_fused_topk32_padded_v2",
            _w8a8_fused_topk32_padded_jax,
            _fake_topk32_4arg,
        ),
    ):
        op = pallas.jax_op(op_name, fn)
        op.register_fake(fake_fn)
        _TPU_QUANT_OPS[key] = op

    return _TPU_QUANT_OPS


def _quantized_head_local_max(
    processor: LogitsProcessor,
    lm_head: torch.nn.Module,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``(local_max_vals, local_max_indices)`` for a ``W8A8``
    quantized ``lm_head``."""
    pad_buf: Any = getattr(lm_head, "num_org_vocab_padding", None)
    num_pad_int = int(
        getattr(
            getattr(lm_head, "shard_indices", None),
            "num_org_vocab_padding",
            0,
        )
    )
    has_vocab_padding = (
        isinstance(pad_buf, torch.Tensor)
        and getattr(processor, "vocab_size", 0)
        < getattr(processor, "padded_vocab_size", 0)
    ) or (not isinstance(pad_buf, torch.Tensor) and num_pad_int > 0)

    if hidden_states.device.type == "tpu":
        ops = _ensure_tpu_quant_ops_registered()
        if has_vocab_padding:
            pad_t = (
                pad_buf.to(device=hidden_states.device, dtype=torch.int32)
                if isinstance(pad_buf, torch.Tensor)
                else torch.tensor(
                    num_pad_int,
                    device=hidden_states.device,
                    dtype=torch.int32,
                )
            )
            return ops["w8a8_padded"](
                hidden_states,
                lm_head._w8a8_weight_kn,
                lm_head._w8a8_scale,
                pad_t,
            )
        return ops["w8a8"](
            hidden_states,
            lm_head._w8a8_weight_kn,
            lm_head._w8a8_scale,
        )

    # CPU execution path (unit tests and host validation).
    x_q, _ = _quantize_fp8_per_row(hidden_states.float())
    x_f32 = x_q.float()
    logits = (x_f32 @ cast(torch.Tensor, lm_head._w8a8_weight_kn).float()) * (
        cast(torch.Tensor, lm_head._w8a8_scale).float().unsqueeze(0)
    )

    if isinstance(pad_buf, torch.Tensor) and (
        getattr(processor, "vocab_size", 0) < getattr(processor, "padded_vocab_size", 0)
    ):
        shard_vocab = logits.shape[-1]
        col_idx = torch.arange(shard_vocab, device=logits.device, dtype=torch.int32)
        valid_cols = shard_vocab - pad_buf.to(device=logits.device, dtype=torch.int32)
        logits = logits.masked_fill((col_idx >= valid_cols).unsqueeze(0), -float("inf"))
    elif num_pad_int > 0:
        logits[..., -num_pad_int:] = -float("inf")

    local_max_vals, local_max_indices = logits.max(dim=-1)
    return local_max_vals.float(), local_max_indices.to(torch.int32)


def tpu_get_top_tokens(
    self: LogitsProcessor,
    lm_head: VocabParallelEmbedding,
    hidden_states: torch.Tensor,
    embedding_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """TPU-optimized vocabulary-parallel greedy ``lm_head + argmax``."""
    if self.scale <= 0.0 and self.scale != 1.0:
        raise ValueError(
            f"scale must be > 0.0 for greedy argmax reduction, got {self.scale}"
        )
    if getattr(lm_head, "_vp_draft", False):
        return _vocab_parallel_draft_top_tokens(
            self, lm_head, hidden_states, embedding_bias
        )
    tp_size = lm_head.tp_size
    quant_mode = getattr(lm_head, "_quant_mode", None)
    if quant_mode == "fp8" and embedding_bias is None and self.soft_cap is None:
        local_max_vals, local_max_indices = _quantized_head_local_max(
            self, lm_head, hidden_states
        )
        vocab_start = getattr(
            lm_head,
            "org_vocab_start_index",
            lm_head.shard_indices.org_vocab_start_index,
        )
        local_indices_i32 = local_max_indices.to(torch.int32)
        if tp_size == 1:
            if isinstance(vocab_start, torch.Tensor):
                local_indices_i32 = local_indices_i32 + vocab_start.to(
                    device=hidden_states.device, dtype=torch.int32
                )
            elif vocab_start != 0:
                local_indices_i32 = local_indices_i32 + int(vocab_start)
            return local_indices_i32.to(torch.int64)

        if isinstance(vocab_start, torch.Tensor):
            global_indices = local_indices_i32 + vocab_start.to(
                device=hidden_states.device, dtype=torch.int32
            )
        elif vocab_start != 0:
            global_indices = local_indices_i32 + int(vocab_start)
        else:
            global_indices = local_indices_i32
        local_pair = torch.stack(
            [local_max_vals.float(), global_indices.float()], dim=-1
        )
        rows = hidden_states.shape[0]
        gathered = lp_module.tensor_model_parallel_all_gather(local_pair, dim=0)
        if gathered.shape[0] == tp_size * rows:
            gathered = gathered.view(tp_size, rows, 2).transpose(0, 1)
        else:
            gathered = gathered.view(rows, tp_size, 2)
        max_rank_idx = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
        top_tokens = gathered[:, :, 1].gather(dim=-1, index=max_rank_idx)
        return top_tokens.squeeze(-1).to(torch.int64)

    fallback_head = getattr(lm_head, "_fallback_head", lm_head)
    logits = self._apply_head(fallback_head, hidden_states, embedding_bias)
    if self.soft_cap is not None:
        logits = torch.tanh(logits / self.soft_cap) * self.soft_cap
    if self.scale != 1.0:
        logits = logits * self.scale

    pad_buf: Any = getattr(lm_head, "num_org_vocab_padding", None)
    if isinstance(pad_buf, torch.Tensor) and (
        getattr(self, "vocab_size", 0) < getattr(self, "padded_vocab_size", 0)
    ):
        shard_vocab = logits.shape[-1]
        col_idx = torch.arange(shard_vocab, device=logits.device, dtype=torch.int32)
        valid_cols = shard_vocab - pad_buf.to(device=logits.device, dtype=torch.int32)
        logits = logits.masked_fill((col_idx >= valid_cols).unsqueeze(0), -float("inf"))
    else:
        num_pad = lm_head.shard_indices.num_org_vocab_padding
        if num_pad > 0:
            logits[..., -num_pad:] = -float("inf")

    vocab_start = getattr(
        lm_head,
        "org_vocab_start_index",
        lm_head.shard_indices.org_vocab_start_index,
    )
    if tp_size == 1:
        local_max_indices = logits.argmax(dim=-1)
        if isinstance(vocab_start, torch.Tensor):
            local_max_indices = local_max_indices.to(torch.int32) + vocab_start.to(
                device=logits.device, dtype=torch.int32
            )
        elif vocab_start != 0:
            local_max_indices = local_max_indices + vocab_start
        return local_max_indices.to(torch.int64)

    local_max_vals, local_max_indices = logits.max(dim=-1)
    local_indices_i32 = local_max_indices.to(torch.int32)
    if isinstance(vocab_start, torch.Tensor):
        global_indices = local_indices_i32 + vocab_start.to(
            device=logits.device, dtype=torch.int32
        )
    elif vocab_start != 0:
        global_indices = local_indices_i32 + int(vocab_start)
    else:
        global_indices = local_indices_i32
    local_pair = torch.stack([local_max_vals.float(), global_indices.float()], dim=-1)
    rows = hidden_states.shape[0]
    gathered = lp_module.tensor_model_parallel_all_gather(local_pair, dim=0)
    if gathered.shape[0] == tp_size * rows:
        gathered = gathered.view(tp_size, rows, 2).transpose(0, 1)
    else:
        gathered = gathered.view(rows, tp_size, 2)
    max_rank_idx = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
    top_tokens = gathered[:, :, 1].gather(dim=-1, index=max_rank_idx)
    return top_tokens.squeeze(-1).to(torch.int64)


LogitsProcessor.get_top_tokens = tpu_get_top_tokens


def _extract_local_topk32_torch(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two-stage local Top-32 extraction in PyTorch.

    Matches ``_extract_local_topk32_jax``.
    """
    t_rows, shard_vocab = logits.shape
    block_size = _LOCAL_TOPK_BLOCK_SIZE
    num_local_blocks = _LOCAL_TOPK_NUM_BLOCKS
    if shard_vocab < num_local_blocks * block_size:
        if shard_vocab < LOCAL_TOPK_CANDIDATES:
            logits_pad = torch.nn.functional.pad(
                logits.float(),
                (0, LOCAL_TOPK_CANDIDATES - shard_vocab),
                value=-float("inf"),
            )
        else:
            logits_pad = logits.float()
        vals32, ids32 = torch.topk(logits_pad, k=LOCAL_TOPK_CANDIDATES, dim=-1)
        return vals32.float(), ids32.to(torch.int32)

    if shard_vocab % block_size != 0:
        pad_cols = block_size - (shard_vocab % block_size)
        lg_padded = torch.nn.functional.pad(logits, (0, pad_cols), value=-float("inf"))
    else:
        lg_padded = logits
    n_blocks = lg_padded.shape[1] // block_size
    blocks = lg_padded.reshape(t_rows, n_blocks, block_size)
    block_max = blocks.amax(dim=-1)
    _, blk_ids = torch.topk(block_max, k=num_local_blocks, dim=-1)
    gather_idx = blk_ids.unsqueeze(-1).expand(-1, -1, block_size)
    local_cand512 = torch.gather(blocks, 1, gather_idx).reshape(
        t_rows, num_local_blocks * block_size
    )
    intra = torch.arange(block_size, device=logits.device, dtype=torch.int32).view(
        1, 1, block_size
    )
    local_ids512 = (blk_ids.to(torch.int32).unsqueeze(-1) * block_size + intra).reshape(
        t_rows, num_local_blocks * block_size
    )
    vals32, idx32 = torch.topk(local_cand512.float(), k=LOCAL_TOPK_CANDIDATES, dim=-1)
    ids32 = torch.gather(local_ids512, 1, idx32)
    return vals32.float(), ids32.to(torch.int32)


def _quantized_head_local_topk(
    processor: LogitsProcessor,
    lm_head: torch.nn.Module,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``(local_topk_vals, local_topk_indices)`` of shape ``[rows, 32]``
    for a ``W8A8`` quantized ``lm_head``."""
    pad_buf: Any = getattr(lm_head, "num_org_vocab_padding", None)
    num_pad_int = int(
        getattr(
            getattr(lm_head, "shard_indices", None),
            "num_org_vocab_padding",
            0,
        )
    )
    has_vocab_padding = (
        isinstance(pad_buf, torch.Tensor)
        and getattr(processor, "vocab_size", 0)
        < getattr(processor, "padded_vocab_size", 0)
    ) or (not isinstance(pad_buf, torch.Tensor) and num_pad_int > 0)

    if hidden_states.device.type == "tpu":
        ops = _ensure_tpu_quant_ops_registered()
        if has_vocab_padding:
            pad_t = (
                pad_buf.to(device=hidden_states.device, dtype=torch.int32)
                if isinstance(pad_buf, torch.Tensor)
                else torch.tensor(
                    num_pad_int,
                    device=hidden_states.device,
                    dtype=torch.int32,
                )
            )
            return ops["w8a8_topk32_padded"](
                hidden_states,
                lm_head._w8a8_weight_kn,
                lm_head._w8a8_scale,
                pad_t,
            )
        return ops["w8a8_topk32"](
            hidden_states,
            lm_head._w8a8_weight_kn,
            lm_head._w8a8_scale,
        )

    # CPU execution path (unit tests and host validation).
    x_q, x_s = _quantize_fp8_per_row(hidden_states.float())
    x_f32 = x_q.float()
    logits = (x_f32 @ cast(torch.Tensor, lm_head._w8a8_weight_kn).float()) * (
        x_s.unsqueeze(-1) * cast(torch.Tensor, lm_head._w8a8_scale).float().unsqueeze(0)
    )
    logits = logits.to(hidden_states.dtype)

    if isinstance(pad_buf, torch.Tensor) and (
        getattr(processor, "vocab_size", 0) < getattr(processor, "padded_vocab_size", 0)
    ):
        shard_vocab = logits.shape[-1]
        col_idx = torch.arange(shard_vocab, device=logits.device, dtype=torch.int32)
        valid_cols = shard_vocab - pad_buf.to(device=logits.device, dtype=torch.int32)
        logits = logits.masked_fill((col_idx >= valid_cols).unsqueeze(0), -float("inf"))
    elif num_pad_int > 0:
        logits[..., -num_pad_int:] = -float("inf")

    return _extract_local_topk32_torch(logits)


def tpu_get_topk_candidates(
    self: LogitsProcessor,
    lm_head: VocabParallelEmbedding,
    hidden_states: torch.Tensor,
    embedding_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """TPU-optimized vocabulary-parallel top-32 candidate extraction
    (``S3_MA_Top32_BS``).

    Each rank extracts its local top-32 ``(logit, token_id)`` pairs from its
    vocab shard and exchanges only ``[rows, 32, 2]`` ``float32`` across TP
    ranks instead of all-gathering the full ``[rows, vocab_size]`` logits.
    Returns a packed 2D ``float32`` tensor of shape ``[rows, 2 * C]`` where
    ``C = 32 * tp_size``, ``packed[:, :C]`` are the candidate logits and
    ``packed[:, C:]`` are the global candidate token IDs.
    """
    if self.scale <= 0.0 and self.scale != 1.0:
        raise ValueError(
            f"scale must be > 0.0 for top-k candidate reduction, got {self.scale}"
        )
    tp_size = lm_head.tp_size
    quant_mode = getattr(lm_head, "_quant_mode", None)
    if quant_mode == "fp8" and embedding_bias is None and self.soft_cap is None:
        local_vals, local_indices = _quantized_head_local_topk(
            self, lm_head, hidden_states
        )
        if self.scale != 1.0:
            local_vals = local_vals * self.scale
    else:
        fallback_head = getattr(lm_head, "_fallback_head", lm_head)
        logits = self._apply_head(fallback_head, hidden_states, embedding_bias)
        if self.soft_cap is not None:
            logits = torch.tanh(logits / self.soft_cap) * self.soft_cap
        if self.scale != 1.0:
            logits = logits * self.scale

        pad_buf: Any = getattr(lm_head, "num_org_vocab_padding", None)
        if isinstance(pad_buf, torch.Tensor) and (
            getattr(self, "vocab_size", 0) < getattr(self, "padded_vocab_size", 0)
        ):
            shard_vocab = logits.shape[-1]
            col_idx = torch.arange(shard_vocab, device=logits.device, dtype=torch.int32)
            valid_cols = shard_vocab - pad_buf.to(
                device=logits.device, dtype=torch.int32
            )
            logits = logits.masked_fill(
                (col_idx >= valid_cols).unsqueeze(0), -float("inf")
            )
        else:
            num_pad = lm_head.shard_indices.num_org_vocab_padding
            if num_pad > 0:
                logits[..., -num_pad:] = -float("inf")

        local_vals, local_indices = _extract_local_topk32_torch(logits)

    vocab_start = getattr(
        lm_head,
        "org_vocab_start_index",
        lm_head.shard_indices.org_vocab_start_index,
    )
    local_indices_i32 = local_indices.to(torch.int32)
    if isinstance(vocab_start, torch.Tensor):
        global_indices = local_indices_i32 + vocab_start.to(
            device=hidden_states.device, dtype=torch.int32
        )
    elif vocab_start != 0:
        global_indices = local_indices_i32 + int(vocab_start)
    else:
        global_indices = local_indices_i32

    if tp_size == 1:
        return torch.cat([local_vals.float(), global_indices.float()], dim=-1)

    rows = hidden_states.shape[0]
    k = local_vals.shape[1]
    local_pair = torch.stack([local_vals.float(), global_indices.float()], dim=-1)
    gathered = lp_module.tensor_model_parallel_all_gather(local_pair, dim=0)
    if gathered.shape[0] == tp_size * rows:
        gathered = gathered.view(tp_size, rows, k, 2).transpose(0, 1)
    else:
        gathered = gathered.view(rows, tp_size, k, 2)
    cand_logits = gathered[:, :, :, 0].reshape(rows, tp_size * k)
    cand_token_ids = gathered[:, :, :, 1].reshape(rows, tp_size * k)
    return torch.cat([cand_logits, cand_token_ids], dim=-1)


LogitsProcessor.get_topk_candidates = tpu_get_topk_candidates


def _ensure_head_shard_buffers(
    head: torch.nn.Module, device: torch.device | None = None
) -> None:
    """Ensure ``org_vocab_start_index`` and ``num_org_vocab_padding`` are
    registered as tensor buffers on ``head`` so compiled graphs do not bake
    rank-specific Python integers."""
    if not isinstance(head, torch.nn.Module):
        return
    if device is None:
        weight: Any = getattr(head, "weight", None)
        if not isinstance(weight, torch.Tensor):
            weight = getattr(head, "_w8a8_weight_kn", None)
        device = weight.device if isinstance(weight, torch.Tensor) else None
    shard_indices = getattr(head, "shard_indices", None)
    if shard_indices is None:
        return
    if hasattr(shard_indices, "org_vocab_start_index"):
        existing_start: Any = getattr(head, "org_vocab_start_index", None)
        if not isinstance(existing_start, torch.Tensor):
            head.register_buffer(
                "org_vocab_start_index",
                torch.tensor(
                    shard_indices.org_vocab_start_index,
                    dtype=torch.int32,
                    device=device,
                ),
                persistent=False,
            )
        elif existing_start.dtype != torch.int32:
            head.org_vocab_start_index = existing_start.to(torch.int32)
    if hasattr(shard_indices, "num_org_vocab_padding"):
        existing_pad: Any = getattr(head, "num_org_vocab_padding", None)
        if not isinstance(existing_pad, torch.Tensor):
            head.register_buffer(
                "num_org_vocab_padding",
                torch.tensor(
                    shard_indices.num_org_vocab_padding,
                    dtype=torch.int32,
                    device=device,
                ),
                persistent=False,
            )
        elif existing_pad.dtype != torch.int32:
            head.num_org_vocab_padding = existing_pad.to(torch.int32)


def _apply_w8a8_lm_head(
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply an in-place ``W8A8`` quantized ``ParallelLMHead`` when full logits
    are requested (e.g. ``_precompile_compute_logits_from_hidden_states`` or
    non-greedy sampling)."""
    w_kn = cast(torch.Tensor, layer._w8a8_weight_kn)
    w_s = cast(torch.Tensor, layer._w8a8_scale)
    if x.device.type == "tpu":
        from vllm_torchtpu.layers.adapter import linear_common

        out = linear_common.quantized_matmul(x, w_kn, w_s)
    else:
        orig_shape = (*x.shape[:-1], w_kn.shape[-1])
        x_2d = x.reshape(-1, x.shape[-1])
        x_q, x_s = _quantize_fp8_per_row(x_2d.float())
        out_2d = (x_q.float() @ w_kn.float()) * (
            x_s.unsqueeze(-1) * w_s.float().unsqueeze(0)
        )
        out = out_2d.to(x.dtype).reshape(orig_shape)
    if bias is not None:
        out = out + bias
    return out


class _W8A8EmbeddingQuantMethod:
    """Replacement ``quant_method`` for an in-place ``W8A8`` ``ParallelLMHead``."""

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return _apply_w8a8_lm_head(layer, x, bias=bias)


def quantize_lm_head(
    head: torch.nn.Module,
    mode: str = "fp8",
    *,
    weight: torch.Tensor | None = None,
    in_place: bool = False,
) -> torch.nn.Module:
    """Quantize ``head`` (or ``weight``) to W8A8 FP8 at load time for
    ``tpu_get_top_tokens`` and (when ``in_place=True``)
    ``LogitsProcessor._apply_head``."""
    if getattr(head, "_quant_mode", None) is not None and (
        weight is None and getattr(head, "weight", None) is None
    ):
        return head

    parsed_mode = _parse_lm_head_quant(mode)
    w: Any = weight if weight is not None else getattr(head, "weight", None)
    if not isinstance(w, torch.Tensor) or w.ndim != 2:
        if in_place:
            return head
        raise ValueError(
            "quantize_lm_head requires a 2D [vocab_shard, hidden_size] weight "
            f"tensor, got {type(w)}"
        )
    _ensure_head_shard_buffers(head, device=w.device)
    if parsed_mode is None:
        # torch's Module.__setattr__ stub only accepts modules and tensors.
        # pyrefly: ignore[bad-argument-type]
        head._quant_mode = None
        # pyrefly: ignore[bad-argument-type]
        head._quant_desc = "bf16"
        return head

    vocab_shard, hidden_size = w.shape
    if w.device.type == "tpu" and (hidden_size % 128 != 0 or vocab_shard % 8 != 0):
        logger.warning_once(
            "quantize_lm_head(%s) skipped on TPU for shape (%d, %d).",
            mode,
            vocab_shard,
            hidden_size,
        )
        return head

    w_fp8_nk_cpu, w_scale_f32_cpu = _quantize_fp8_per_row(w.detach().float().cpu())
    w_scale_f32_cpu = w_scale_f32_cpu.reshape(vocab_shard).contiguous()
    w_fp8_kn_cpu = w_fp8_nk_cpu.t().contiguous()
    if w.device.type == "tpu":
        _ensure_tpu_quant_ops_registered()
        from vllm_torchtpu.layers.adapter import linear_common
        from vllm_torchtpu.utils import synchronize_tensors

        linear_common._get_quantized_matmul_op()
        w_fp8_kn = torch.empty_strided(
            w_fp8_kn_cpu.shape,
            (vocab_shard, 1),
            dtype=torch.float8_e4m3fn,
            device=w.device,
        ).copy_(w_fp8_kn_cpu)
        w_scale_f32 = w_scale_f32_cpu.to(w.device)
        synchronize_tensors([w_fp8_kn, w_scale_f32])
    else:
        w_fp8_kn = w_fp8_kn_cpu.to(w.device)
        w_scale_f32 = w_scale_f32_cpu.to(w.device)
    head.register_buffer("_w8a8_weight_kn", w_fp8_kn, persistent=False)
    head.register_buffer("_w8a8_scale", w_scale_f32, persistent=False)
    # torch's Module.__setattr__ stub only accepts modules and tensors.
    # pyrefly: ignore[bad-argument-type]
    head._quant_mode = "fp8"
    # pyrefly: ignore[bad-argument-type]
    head._quant_desc = "w8a8"
    if in_place:
        # torch's Module.__setattr__ stub only accepts modules and tensors.
        # pyrefly: ignore[bad-argument-type]
        head.quant_method = _W8A8EmbeddingQuantMethod()
        if hasattr(head, "weight"):
            delattr(head, "weight")
        logger.info_once(
            "Quantized shared ParallelLMHead in-place to W8A8 FP8: "
            "_w8a8_weight_kn=%s, %s.",
            tuple(w_fp8_kn.shape),
            "BF16 weight kept for the draft's own head"
            if draft_bf16_head_of(head) is not None
            else "freed BF16 weight",
        )
    return head


class _DraftHeadOwnerProxy(torch.nn.Module):
    """Owner proxy pairing ``logits_processor`` with a draft ``lm_head`` (the
    vocab-parallel draft head)."""

    def __init__(
        self, logits_processor: LogitsProcessor, lm_head: torch.nn.Module
    ) -> None:
        super().__init__()
        self.logits_processor = logits_processor
        self.lm_head = lm_head


class _VocabParallelDraftHead(torch.nn.Module):
    """The draft's BF16 greedy head split over all DP x TP ranks.

    Under EP-DP lockstep every DP engine's TP group holds the same TP shard
    of the head, so each draft pass streams ``vocab / tp`` rows per rank (at
    DP4 x TP2: 124,160 x 4,096 BF16, ~1 GB) for that engine's own few rows.
    Here DP rank ``d`` of TP rank ``t`` keeps only the ``d``-th of ``dp``
    equal slices of its TP shard, i.e. global vocabulary slice
    ``t * dp + d``. A pass all-gathers every engine's hidden rows over DP,
    computes its slice's logits for all of them with the head's own quant
    method (the same BF16 matmul, mask and local max as the per-TP-group
    path), all-gathers the ``(value, index)`` pairs over the EP group
    (= DP x TP), puts them in vocabulary order and keeps this engine's rows.
    The argmax over slices in vocabulary order takes the first maximum, so
    ties resolve to the lowest token id, as ``torch.argmax`` does over the
    full logits. Each rank streams ``1 / dp`` of the bytes.

    Every rank must run each pass, with the same row count: true under
    EP-DP lockstep for a TP-sharded draft, where busy and idle engines run
    the same ``_draft_propose_token`` / fused-loop programs at the loop
    bucket of ``max_num_reqs`` (``Eagle3Proposer._loop_bucket``).
    Rank-specific values (vocab start, padding, DP rank) are tensor buffers
    so every rank compiles the same program.
    """

    def __init__(
        self,
        base_head: torch.nn.Module,
        dp_rank: int,
        dp_size: int,
        tp_rank: int,
        tp_size: int,
    ) -> None:
        super().__init__()
        weight = base_head.weight
        shard_vocab = int(weight.shape[0])
        if shard_vocab % dp_size:
            raise ValueError(
                f"TP shard of {shard_vocab} rows does not split into {dp_size} "
                "equal DP slices"
            )
        n = shard_vocab // dp_size
        si = base_head.shard_indices
        shard_pad = int(si.num_org_vocab_padding)
        valid = max(0, min(n, shard_vocab - shard_pad - dp_rank * n))
        start = int(si.org_vocab_start_index) + dp_rank * n
        self.weight = torch.nn.Parameter(
            weight.detach()[dp_rank * n : (dp_rank + 1) * n].clone(),
            requires_grad=False,
        )
        self.quant_method = base_head.quant_method
        self.tp_size = tp_size
        self.dp_size = dp_size
        self.num_slices = dp_size * tp_size
        from types import SimpleNamespace

        self.shard_indices = SimpleNamespace(
            org_vocab_start_index=start, num_org_vocab_padding=n - valid
        )
        device = weight.device
        self.register_buffer(
            "org_vocab_start_index",
            torch.tensor(start, dtype=torch.int32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "num_org_vocab_padding",
            torch.tensor(n - valid, dtype=torch.int32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "vp_dp_rank",
            torch.tensor([dp_rank], dtype=torch.int64, device=device),
            persistent=False,
        )
        # Pairs arrive in EP-group order (rank dp * tp_size + tp, checked in
        # vocab_parallel_draft_owner); slice k belongs to tp = k // dp_size,
        # dp = k % dp_size. perm[k] is that rank's position in the gather.
        perm = [(k % dp_size) * tp_size + k // dp_size for k in range(self.num_slices)]
        self.register_buffer(
            "vp_slice_order",
            torch.tensor(perm, dtype=torch.int64, device=device),
            persistent=False,
        )
        self._vp_draft = True
        self._quant_mode = None
        self._quant_group_size = None
        self._quant_desc = "bf16"


def _vocab_parallel_draft_top_tokens(
    processor: LogitsProcessor,
    lm_head: torch.nn.Module,
    hidden_states: torch.Tensor,
    embedding_bias: torch.Tensor | None,
) -> torch.Tensor:
    """``tpu_get_top_tokens`` for a ``_VocabParallelDraftHead``: the same
    logits, mask and local max as its BF16 fallback path, over all DP
    engines' rows and this rank's vocabulary slice."""
    from vllm.distributed.parallel_state import get_dp_group, get_ep_group

    dp_size = int(lm_head.dp_size)
    num_slices = int(lm_head.num_slices)
    rows = hidden_states.shape[0]
    all_rows = get_dp_group().all_gather(hidden_states, dim=0)
    logits = processor._apply_head(lm_head, all_rows, embedding_bias)
    if processor.soft_cap is not None:
        logits = torch.tanh(logits / processor.soft_cap) * processor.soft_cap
    if processor.scale != 1.0:
        logits = logits * processor.scale
    slice_vocab = logits.shape[-1]
    col_idx = torch.arange(slice_vocab, device=logits.device, dtype=torch.int32)
    valid_cols = slice_vocab - lm_head.num_org_vocab_padding.to(
        device=logits.device, dtype=torch.int32
    )
    logits = logits.masked_fill((col_idx >= valid_cols).unsqueeze(0), -float("inf"))
    local_max_vals, local_max_indices = logits.max(dim=-1)
    global_indices = local_max_indices.to(torch.int32) + (
        lm_head.org_vocab_start_index.to(device=logits.device, dtype=torch.int32)
    )
    local_pair = torch.stack(
        [local_max_vals.float(), global_indices.float()], dim=-1
    )  # [dp * rows, 2]
    gathered = get_ep_group().all_gather(local_pair, dim=0)
    gathered = gathered.view(num_slices, dp_size, rows, 2)
    gathered = gathered.index_select(0, lm_head.vp_slice_order)
    mine = gathered.index_select(1, lm_head.vp_dp_rank).squeeze(1)  # [slices, rows, 2]
    max_slice = mine[:, :, 0].argmax(dim=0, keepdim=True)
    top_tokens = mine[:, :, 1].gather(dim=0, index=max_slice)
    return top_tokens.squeeze(0).to(torch.int64)


def vocab_parallel_draft_owner(owner: torch.nn.Module) -> torch.nn.Module:
    """``owner`` with its BF16 head replaced by a ``_VocabParallelDraftHead``
    when the DP / EP / TP groups allow it (EP group = DP x TP with rank
    ``dp * tp_size + tp``, TP shard divisible by DP); ``owner`` unchanged
    otherwise. The caller decides that the draft runs under EP-DP lockstep
    with a TP-sharded head."""
    from vllm.distributed.parallel_state import (
        get_dp_group,
        get_ep_group,
        get_tp_group,
    )

    head = owner.lm_head
    weight: Any = getattr(head, "weight", None)
    if (
        getattr(head, "_quant_mode", None) is not None
        or not isinstance(weight, torch.Tensor)
        or weight.dtype != torch.bfloat16
    ):
        return owner
    dp, ep, tp = get_dp_group(), get_ep_group(), get_tp_group()
    dp_size, tp_size = int(dp.world_size), int(tp.world_size)
    reason = None
    if dp_size <= 1:
        reason = "a single DP engine"
    elif int(head.tp_size) != tp_size:
        reason = f"head tp_size {head.tp_size} != TP group {tp_size}"
    elif int(ep.world_size) != dp_size * tp_size:
        reason = f"EP group {ep.world_size} != DP {dp_size} x TP {tp_size}"
    elif int(ep.rank_in_group) != int(dp.rank_in_group) * tp_size + int(
        tp.rank_in_group
    ):
        reason = (
            f"EP rank {ep.rank_in_group} != DP rank {dp.rank_in_group} x "
            f"{tp_size} + TP rank {tp.rank_in_group}"
        )
    elif weight.shape[0] % dp_size:
        reason = f"TP shard of {weight.shape[0]} rows not divisible by DP {dp_size}"
    if reason is not None:
        logger.warning(
            "TPU_DRAFT_VOCAB_PARALLEL: keeping the per-TP-group draft head (%s).",
            reason,
        )
        return owner
    vp_head = _VocabParallelDraftHead(
        head, int(dp.rank_in_group), dp_size, int(tp.rank_in_group), tp_size
    )
    logger.info(
        "TPU_DRAFT_VOCAB_PARALLEL: draft lm_head vocab-parallel over DP%d x TP%d "
        "= %d slices; this rank: vocab [%d, %d), weight %s %s (was %s per TP "
        "group). Same BF16 logits; ties to the lowest token id.",
        dp_size,
        tp_size,
        dp_size * tp_size,
        vp_head.shard_indices.org_vocab_start_index,
        vp_head.shard_indices.org_vocab_start_index + vp_head.weight.shape[0],
        tuple(vp_head.weight.shape),
        str(vp_head.weight.dtype).replace("torch.", ""),
        tuple(weight.shape),
    )
    return _DraftHeadOwnerProxy(owner.logits_processor, vp_head)


# The draft's BF16 head, kept on the target head when the target's is
# quantized in place (see keep_bf16_head_for_draft). Stored outside
# ``_modules`` so the target never lists it as a submodule.
_DRAFT_BF16_HEAD_ATTR = "_tpu_draft_bf16_head"


def bf16_head_twin(head: torch.nn.Module) -> torch.nn.Module:
    """A second ``ParallelLMHead`` holding ``head``'s current BF16 weight.

    The twin is a shallow copy with its own parameter, buffer and submodule
    tables, so freeing ``head``'s weight (``quantize_lm_head(...,
    in_place=True)``) leaves the twin's weight, quant method and shard
    layout as loaded. The weight tensor is the same one, not a copy: no
    extra memory is used until the target's weight is dropped.
    """
    import copy

    twin = copy.copy(head)
    state = twin.__dict__
    for table in ("_parameters", "_buffers", "_modules"):
        if table in state:
            state[table] = type(state[table])(state[table])
    if "_non_persistent_buffers_set" in state:
        state["_non_persistent_buffers_set"] = set(state["_non_persistent_buffers_set"])
    state.pop(_DRAFT_BF16_HEAD_ATTR, None)
    for name in ("_w8a8_weight_kn", "_w8a8_scale"):
        state.get("_buffers", {}).pop(name, None)
    state["_quant_mode"] = None
    state["_quant_group_size"] = None
    state["_quant_desc"] = "bf16"
    return twin


def keep_bf16_head_for_draft(head: torch.nn.Module) -> torch.nn.Module | None:
    """Keeps a BF16 twin of ``head`` for the draft before ``head`` is
    quantized in place. Returns the twin, or None when ``head`` has no BF16
    weight left to keep (already quantized)."""
    weight: Any = getattr(head, "weight", None)
    if not isinstance(weight, torch.Tensor) or getattr(head, "_quant_mode", None):
        return None
    twin = bf16_head_twin(head)
    object.__setattr__(head, _DRAFT_BF16_HEAD_ATTR, twin)
    logger.info(
        "Kept a BF16 copy of the target lm_head for the draft "
        "(TPU_DRAFT_LM_HEAD_BF16=1): weight %s %s.",
        tuple(weight.shape),
        str(weight.dtype).replace("torch.", ""),
    )
    return twin


def draft_bf16_head_of(target_head: object) -> torch.nn.Module | None:
    """The BF16 head kept for the draft on ``target_head``, if any."""
    return getattr(target_head, "__dict__", {}).get(_DRAFT_BF16_HEAD_ATTR)


def release_draft_bf16_head(target_head: object) -> None:
    """Drops the BF16 head kept for a draft that turned out to have its own."""
    state = getattr(target_head, "__dict__", None)
    if state is not None and state.pop(_DRAFT_BF16_HEAD_ATTR, None) is not None:
        logger.info(
            "The draft has its own lm_head; released the BF16 copy of the "
            "target's head."
        )


def head_precision(head: object) -> str:
    """``bf16`` / ``w8a8`` / ``w4a8_g<N>``: the precision a head computes in."""
    desc = getattr(head, "_quant_desc", None)
    if desc:
        return str(desc)
    weight: Any = getattr(head, "weight", None)
    if isinstance(weight, torch.Tensor):
        return str(weight.dtype).replace("torch.", "").replace("bfloat16", "bf16")
    return "unknown"


def target_head_dp_mode(dp_lockstep: bool, tp_size: int) -> str:
    """How the target's LM head is served under TPU_GREEDY_TOP_TOKENS.

    ``"fast"``: the vocab-parallel argmax / top-32 candidates. ``"stock"``:
    full logits (``compute_selected_logits``), the path an idle engine's
    lockstep dummy replays. Under expert-parallel data parallelism with a
    tensor-parallel head, busy engines would pick the fast or stock program
    per step from their own requests' sampling params, while idle engines
    always replay the stock one; collectives pair across ranks in emission
    order, so the target stays on the stock path there. The draft is not
    affected: its tokens are always a greedy argmax, and an idle engine's
    dummy draft runs the same ``_draft_propose_token`` program
    (see ``draft_head_dp_lockstep``).
    """
    if dp_lockstep and tp_size > 1:
        return "stock"
    return "fast"


def draft_head_dp_lockstep(dp_lockstep: bool) -> bool:
    """The ``dp_lockstep`` a draft head owner is resolved with.

    Always False: the MTP/EAGLE draft head's argmax runs in
    ``_draft_propose_token`` for busy engines and for an idle engine's
    ``run_dp_dummy_draft`` alike, so every rank emits the same pair exchange
    in the same order, whatever the requests' sampling params.
    """
    del dp_lockstep
    return False


def resolve_head_owner(model: torch.nn.Module, dp_lockstep: bool) -> torch.nn.Module:
    """The module holding ``lm_head`` and its ``logits_processor``.

    Qwen3.5's wrapper keeps both on ``language_model``. Raises when the model
    cannot be served by the vocabulary-parallel argmax.
    """
    compute_logits = getattr(type(model), "compute_logits", None)
    if getattr(compute_logits, "__module__", "").startswith("vllm_torchtpu."):
        raise NotImplementedError(
            f"TPU_GREEDY_TOP_TOKENS: {type(model).__name__} computes its logits "
            "in a TPU-specific compute_logits that get_top_tokens does not "
            "reproduce; unset TPU_GREEDY_TOP_TOKENS for this model."
        )
    # A draft model or owner proxy without ``modules()`` is its own owner.
    modules_fn = getattr(model, "modules", None)
    for module in modules_fn() if callable(modules_fn) else (model,):
        processor = getattr(module, "logits_processor", None)
        head = getattr(module, "lm_head", None)
        # vLLM's LoRA wrappers forward attribute lookups to the layer they
        # wrap, so checking the instance would find the base layer's
        # get_top_tokens and drop the wrapper's change to the logits. Checking
        # the class makes such a wrapper fail here; top_tokens handles the
        # LoRA one.
        has_top_tokens = isinstance(processor, LogitsProcessorWithLoRA) or hasattr(
            type(processor), "get_top_tokens"
        )
        if has_top_tokens and hasattr(head, "shard_indices"):
            break
    else:
        raise NotImplementedError(
            f"TPU_GREEDY_TOP_TOKENS: {type(model).__name__} has no module with "
            "an lm_head and a logits_processor.get_top_tokens; unset "
            "TPU_GREEDY_TOP_TOKENS for this model."
        )
    assert processor is not None and head is not None
    if processor.scale <= 0.0:
        raise NotImplementedError(
            "TPU_GREEDY_TOP_TOKENS: the argmax reduction needs a positive "
            f"logit scale, got {processor.scale}; unset TPU_GREEDY_TOP_TOKENS."
        )
    if dp_lockstep and head.tp_size > 1:
        # An idle engine issues the full-logits collective in lockstep with
        # its busy peers, which would be waiting in the pair exchange.
        raise NotImplementedError(
            "TPU_GREEDY_TOP_TOKENS with expert-parallel data parallelism and a "
            "tensor-parallel LM head is not supported; unset "
            "TPU_GREEDY_TOP_TOKENS or use tensor parallelism alone."
        )
    _ensure_head_shard_buffers(cast("torch.nn.Module", head))
    return module


def top_tokens(owner: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """The greedy token of each row, ``[rows, 1]`` int64 like
    ``torch.argmax(logits, dim=-1, keepdim=True)``."""
    processor = cast(
        "LogitsProcessor | LogitsProcessorWithLoRA", owner.logits_processor
    )
    head = cast("VocabParallelEmbedding", owner.lm_head)
    if isinstance(processor, LogitsProcessorWithLoRA):
        # The LoRA wrapper has no get_top_tokens, so the call would reach the
        # base layer and drop the lm_head LoRA delta. Take the argmax of the
        # wrapper's logits, which include the delta.
        return torch.argmax(processor(head, hidden_states), dim=-1, keepdim=True)
    tokens = processor.get_top_tokens(head, hidden_states)
    return tokens.to(torch.int64).view(-1, 1)


def log_step_path(is_spec_step: bool, req_ids: Sequence[str | None]) -> None:
    """Logs, once per path, that a step's tokens came from the top tokens.

    Steps holding only the runner's synthetic warmup requests (ids
    ``__..__``) are skipped, so the line marks served traffic.
    """
    if all(req_id is None or req_id.startswith("__") for req_id in req_ids):
        return
    if is_spec_step:
        logger.info_once(
            "TPU_GREEDY_TOP_TOKENS: a speculative verify step took the top tokens"
        )
    else:
        logger.info_once(
            "TPU_GREEDY_TOP_TOKENS: a non-speculative step took the top tokens"
        )


def step_eligible(
    input_batch: InputBatch, requests: Mapping[str, CachedRequestState]
) -> bool:
    """Whether the step's tokens are fully determined by the top tokens:
    every request greedy, no logprobs and no grammar, which needs the full
    logits to mask."""
    if not input_batch.all_greedy or input_batch.max_num_logprobs is not None:
        return False
    for req_id in input_batch.req_ids:
        if req_id is None:
            continue
        params = requests[req_id].sampling_params
        if params is not None and params.structured_outputs is not None:
            logger.warning_once(
                "TPU_GREEDY_TOP_TOKENS: steps with a structured-output request "
                "gather the full logits"
            )
            return False
    return True


def topk_candidates_supported(owner: torch.nn.Module) -> bool:
    """Whether ``owner``'s head can give packed top-32 candidates.

    vLLM's LoRA wrapper forwards attribute lookups to the processor it wraps,
    so ``get_topk_candidates`` would reach the base layer and drop the lm_head
    LoRA delta; steps of such a model keep the full logits.
    """
    return not isinstance(owner.logits_processor, LogitsProcessorWithLoRA)


def num_packed_candidates(owner: torch.nn.Module) -> int:
    """Total number of vocabulary-parallel candidates across TP ranks."""
    head = cast("VocabParallelEmbedding", owner.lm_head)
    return LOCAL_TOPK_CANDIDATES * int(head.tp_size)


def topk_candidates(
    owner: torch.nn.Module, hidden_states: torch.Tensor
) -> torch.Tensor:
    """Packed ``[rows, 2 * C]`` ``float32`` vocabulary-parallel top-32
    candidates per TP rank (``C = 32 * tp_size``)."""
    # Module.__getattr__ is typed Tensor | Module; LogitsProcessor gets this at import.
    # pyrefly: ignore[missing-attribute]
    return owner.logits_processor.get_topk_candidates(owner.lm_head, hidden_states)


def log_topk_candidates_step_path(
    is_spec_step: bool, req_ids: Sequence[str | None]
) -> None:
    """Logs, once per path, that a step used vocabulary-parallel top-k candidates."""
    if all(req_id is None or req_id.startswith("__") for req_id in req_ids):
        return
    if is_spec_step:
        logger.info_once(
            "TPU_GREEDY_TOP_TOKENS: a speculative verify step took vocab-parallel "
            "top-32 candidates"
        )
    else:
        logger.info_once(
            "TPU_GREEDY_TOP_TOKENS: a non-speculative step took vocab-parallel "
            "top-32 candidates"
        )


def topk_candidates_step_eligible(
    input_batch: InputBatch,
    requests: Mapping[str, CachedRequestState],
    max_local_k: int = LOCAL_TOPK_CANDIDATES,
) -> bool:
    """Whether a non-greedy step can be sampled from shard-local top-32
    candidates per TP rank without gathering the full vocabulary logits.

    When every active request has ``1 <= top_k <= max_local_k`` (e.g.
    ``top_k = 20 <= 32``) and no logprobs or structured-output grammar, the
    global top-``k`` tokens are guaranteed to be a subset of the union of each
    TP shard's top-32 tokens, so both ``top_k`` and ``top_p`` masking over the
    ``32 * tp_size`` candidates are exact.
    """
    if input_batch.all_greedy or input_batch.max_num_logprobs is not None:
        return False
    has_req = False
    for req_id in input_batch.req_ids:
        if req_id is None:
            continue
        has_req = True
        state = requests.get(req_id)
        params = getattr(state, "sampling_params", None) if state is not None else None
        if params is None:
            return False
        if params.structured_outputs is not None:
            return False
        if (
            getattr(params, "presence_penalty", 0.0) != 0.0
            or getattr(params, "frequency_penalty", 0.0) != 0.0
            or getattr(params, "repetition_penalty", 1.0) != 1.0
            or getattr(params, "min_p", 0.0) != 0.0
            or getattr(params, "bad_words_token_ids", None)
        ):
            return False
        if getattr(params, "temperature", 1.0) == 0:
            continue
        top_k = int(getattr(params, "top_k", 0))
        if req_id.startswith("__"):
            if top_k > max_local_k:
                return False
        elif not (1 <= top_k <= max_local_k):
            return False
    return has_req
