# SPDX-License-Identifier: Apache-2.0
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

- W8A8 / W4A8 quantized LM heads (``quantize_lm_head``, the draft proposer's
  ``_QuantizedLMHeadProxy``), whose local max+argmax runs in fused Pallas ops
  registered by ``_ensure_tpu_quant_ops_registered``;
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


# Exact positive non-zero values of IEEE-like FP4 E2M1 (sign=1, exp=2, mantissa=1):
# {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}, max = 6.0.
_FP4_E2M1_MAX = 6.0
_FP4_E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_FP4_E2M1_THRESHOLDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


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
    q = torch.clamp(
        t_f32 * scale_inv, min=-_FP8_E4M3_MAX, max=_FP8_E4M3_MAX
    ).to(torch.float8_e4m3fn)
    return q, scale.squeeze(-1).to(torch.float32)


def parse_draft_lm_head_quant(
    quant_mode: str | None, fp4_group_size: int = 128
) -> tuple[str | None, int | None]:
    """Parse ``TPU_DRAFT_LM_HEAD_QUANT`` and ``TPU_DRAFT_LM_HEAD_FP4_GROUP_SIZE``.

    Returns ``(mode, group_size)`` where ``mode`` is ``"fp8"``, ``"fp4"``, or
    ``None`` (keep BF16). For ``mode == "fp4"``, ``group_size == -1`` indicates
    per-channel (``group_size == hidden_size``).
    """
    raw = (quant_mode or "").strip().lower()
    if raw in ("", "0", "false", "no", "off", "none", "bf16", "bfloat16"):
        return None, None
    if raw in ("fp8", "w8a8", "float8", "float8_e4m3fn"):
        return "fp8", None
    if raw in ("fp4", "w4a8", "float4", "float4_e2m1fn"):
        if int(fp4_group_size) <= 0:
            raise ValueError(
                f"fp4_group_size must be > 0, got {fp4_group_size}"
            )
        return "fp4", int(fp4_group_size)
    if raw in ("fp4_ch", "w4a8_ch"):
        return "fp4", -1
    if raw.startswith(("fp4_g", "w4a8_g")):
        suffix = raw.split("_g", 1)[1]
        try:
            gs = int(suffix)
        except ValueError as e:
            raise ValueError(
                f"Invalid FP4 group size in TPU_DRAFT_LM_HEAD_QUANT={quant_mode!r}"
            ) from e
        if gs <= 0:
            raise ValueError(
                f"FP4 group size must be > 0 in TPU_DRAFT_LM_HEAD_QUANT={quant_mode!r}"
            )
        return "fp4", gs
    raise ValueError(
        f"Unsupported TPU_DRAFT_LM_HEAD_QUANT={quant_mode!r}; expected 'fp8' "
        "('w8a8'), 'fp4' ('w4a8' / 'fp4_g128' / 'fp4_g256' / ... / 'fp4_ch'), "
        "or 'bf16' ('none' / 'off')."
    )


def _ensure_tpu_quant_ops_registered() -> dict[str, Any]:
    """Lazily register fused JAX/Pallas ``W8A8`` and ``W4A8`` local max+argmax
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

    def _w8a8_2d_logits_jax(
        x: jax.Array, w_q: jax.Array, w_s: jax.Array
    ) -> jax.Array:
        orig_b = x.shape[0]
        if orig_b == 1:
            x = jnp.pad(x, ((0, 1), (0, 0)))
        # hidden_states (x) is replicated identically across all TP shards, so
        # the positive per-row activation scale x_s[i] > 0 is identical across
        # ranks for each row i and strictly monotonic under intra-shard and
        # cross-shard max/argmax (matching _w4a8_3d_logits_jax and CPU path).
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

    def _w4a8_3d_logits_jax(
        x: jax.Array, w_q_3d: jax.Array, w_s_gn: jax.Array
    ) -> jax.Array:
        orig_b = x.shape[0]
        if orig_b == 1:
            x = jnp.pad(x, ((0, 1), (0, 0)))
        b = x.shape[0]
        num_groups = w_q_3d.shape[0]
        group_size = w_q_3d.shape[1]
        x_q, _ = jax_quantize_tensor(x, jnp.float8_e4m3fn)
        p = jax.lax.dot_general(
            x_q.reshape(b, num_groups, group_size),
            w_q_3d,
            dimension_numbers=(((2,), (1,)), ((1,), (0,))),
            preferred_element_type=jnp.float32,
        )
        logits = jnp.sum(p * w_s_gn[:, None, :], axis=0)
        if orig_b == 1:
            logits = logits[:1]
        return logits

    def _w4a8_3d_fused_max_jax(
        x: jax.Array, w_q_3d: jax.Array, w_s_gn: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        logits = _w4a8_3d_logits_jax(x, w_q_3d, w_s_gn)
        vals = jnp.max(logits, axis=-1)
        idx = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        return vals, idx

    def _w4a8_3d_fused_max_padded_jax(
        x: jax.Array,
        w_q_3d: jax.Array,
        w_s_gn: jax.Array,
        num_pad: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        logits = _w4a8_3d_logits_jax(x, w_q_3d, w_s_gn)
        shard_vocab = logits.shape[-1]
        valid_cols = shard_vocab - num_pad.astype(jnp.int32)
        col_idx = jnp.arange(shard_vocab, dtype=jnp.int32)
        logits = jnp.where(col_idx[None, :] >= valid_cols, -jnp.inf, logits)
        vals = jnp.max(logits, axis=-1)
        idx = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        return vals, idx

    def _w8a8_2d_scaled_logits_jax(
        x: jax.Array, w_q: jax.Array, w_s: jax.Array
    ) -> jax.Array:
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
            * x_s
            * w_s[None, :]
        )
        if orig_b == 1:
            logits = logits[:1]
        return logits.astype(x.dtype)

    def _w4a8_3d_scaled_logits_jax(
        x: jax.Array, w_q_3d: jax.Array, w_s_gn: jax.Array
    ) -> jax.Array:
        orig_b = x.shape[0]
        if orig_b == 1:
            x = jnp.pad(x, ((0, 1), (0, 0)))
        b = x.shape[0]
        num_groups = w_q_3d.shape[0]
        group_size = w_q_3d.shape[1]
        x_q, x_s = jax_quantize_tensor(x, jnp.float8_e4m3fn)
        p = jax.lax.dot_general(
            x_q.reshape(b, num_groups, group_size),
            w_q_3d,
            dimension_numbers=(((2,), (1,)), ((1,), (0,))),
            preferred_element_type=jnp.float32,
        )
        logits = jnp.sum(p * w_s_gn[:, None, :], axis=0) * x_s
        if orig_b == 1:
            logits = logits[:1]
        return logits.astype(x.dtype)

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

    def _w4a8_3d_fused_topk32_jax(
        x: jax.Array, w_q_3d: jax.Array, w_s_gn: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        logits = _w4a8_3d_scaled_logits_jax(x, w_q_3d, w_s_gn)
        return _extract_local_topk32_jax(logits)

    def _w4a8_3d_fused_topk32_padded_jax(
        x: jax.Array,
        w_q_3d: jax.Array,
        w_s_gn: jax.Array,
        num_pad: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        logits = _w4a8_3d_scaled_logits_jax(x, w_q_3d, w_s_gn)
        shard_vocab = logits.shape[-1]
        valid_cols = shard_vocab - num_pad.astype(jnp.int32)
        col_idx = jnp.arange(shard_vocab, dtype=jnp.int32)
        logits = jnp.where(
            col_idx[None, :] >= valid_cols,
            jnp.array(-jnp.inf, dtype=logits.dtype),
            logits,
        )
        return _extract_local_topk32_jax(logits)

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

    def _fake_topk32_3arg(
        x: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor
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
            "w4a8_2d",
            "pallas::fp4_ch_w4a8_fused_max_v2",
            _w8a8_fused_max_jax,
            _fake_3arg,
        ),
        (
            "w4a8_2d_padded",
            "pallas::fp4_ch_w4a8_fused_max_padded_v2",
            _w8a8_fused_max_padded_jax,
            _fake_4arg,
        ),
        (
            "w4a8_3d",
            "pallas::fp4_3d_w4a8_fused_max_v2",
            _w4a8_3d_fused_max_jax,
            _fake_3arg,
        ),
        (
            "w4a8_3d_padded",
            "pallas::fp4_3d_w4a8_fused_max_padded_v2",
            _w4a8_3d_fused_max_padded_jax,
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
        (
            "w4a8_2d_topk32",
            "pallas::fp4_ch_w4a8_fused_topk32_v2",
            _w8a8_fused_topk32_jax,
            _fake_topk32_3arg,
        ),
        (
            "w4a8_2d_topk32_padded",
            "pallas::fp4_ch_w4a8_fused_topk32_padded_v2",
            _w8a8_fused_topk32_padded_jax,
            _fake_topk32_4arg,
        ),
        (
            "w4a8_3d_topk32",
            "pallas::fp4_3d_w4a8_fused_topk32_v2",
            _w4a8_3d_fused_topk32_jax,
            _fake_topk32_3arg,
        ),
        (
            "w4a8_3d_topk32_padded",
            "pallas::fp4_3d_w4a8_fused_topk32_padded_v2",
            _w4a8_3d_fused_topk32_padded_jax,
            _fake_topk32_4arg,
        ),
    ):
        op = pallas.jax_op(op_name, fn)
        op.register_fake(fake_fn)
        _TPU_QUANT_OPS[key] = op

    def _quantize_to_native_fp4_kmajor_jax(
        w: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        w_f32 = w.astype(jnp.float32)
        fp4_max = float(jnp.finfo(jnp.float4_e2m1fn).max)
        abs_max = jnp.max(jnp.abs(w_f32), axis=-2, keepdims=True)
        scale = abs_max / fp4_max
        scale_inv = jnp.where(scale == 0, 0.0, 1.0 / scale)
        quantized = jnp.clip(w_f32 * scale_inv, -fp4_max, fp4_max).astype(
            jnp.float4_e2m1fn
        )
        return quantized, scale.astype(jnp.float32)

    _TPU_QUANT_OPS["quantize_fp4"] = pallas.jax_op(
        "pallas::lm_head_quantize_native_fp4_kmajor_v2",
        _quantize_to_native_fp4_kmajor_jax,
    )

    return _TPU_QUANT_OPS


def _quantize_fp4_e2m1_cpu(
    w_kn: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Quantize a K-major ``[H, V]`` tensor on CPU to the exact FP4 E2M1 grid."""
    hidden_size, vocab_shard = w_kn.shape
    num_groups = hidden_size // group_size
    is_2d = num_groups == 1
    w_3d = w_kn.float().reshape(num_groups, group_size, vocab_shard)
    amax = w_3d.abs().amax(dim=1, keepdim=True)
    scale = (amax / _FP4_E2M1_MAX).clamp(min=1e-12)
    scaled = (w_3d / scale).clamp(-_FP4_E2M1_MAX, _FP4_E2M1_MAX)
    sign = torch.sign(scaled)
    abs_val = scaled.abs()
    snapped = torch.zeros_like(abs_val)
    for thresh, grid_val in zip(_FP4_E2M1_THRESHOLDS, _FP4_E2M1_GRID[1:]):
        snapped = torch.where(
            abs_val >= thresh,
            torch.full_like(snapped, grid_val),
            snapped,
        )
    for mid, even_val in (
        (0.25, 0.0),
        (1.25, 1.0),
        (2.5, 2.0),
        (5.0, 4.0),
    ):
        snapped = torch.where(
            abs_val == mid,
            torch.full_like(snapped, even_val),
            snapped,
        )
    q_3d = (sign * snapped).to(torch.float8_e4m3fn)
    if is_2d:
        return (
            q_3d.reshape(hidden_size, vocab_shard).contiguous(),
            scale.reshape(vocab_shard).contiguous(),
            True,
        )
    return (
        q_3d.contiguous(),
        scale.reshape(num_groups, vocab_shard).contiguous(),
        False,
    )


def _quantized_head_local_max(
    processor: LogitsProcessor,
    lm_head: torch.nn.Module,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``(local_max_vals, local_max_indices)`` for a ``W8A8`` or
    ``W4A8`` quantized ``lm_head``."""
    quant_mode = getattr(lm_head, "_quant_mode", None)
    pad_buf = getattr(lm_head, "num_org_vocab_padding", None)
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
            if quant_mode == "fp8":
                return ops["w8a8_padded"](
                    hidden_states,
                    lm_head._w8a8_weight_kn,
                    lm_head._w8a8_scale,
                    pad_t,
                )
            if getattr(lm_head, "_w4a8_is_2d", False):
                return ops["w4a8_2d_padded"](
                    hidden_states,
                    lm_head._w4a8_weight_kn,
                    lm_head._w4a8_scale,
                    pad_t,
                )
            return ops["w4a8_3d_padded"](
                hidden_states,
                lm_head._w4a8_weight_3d,
                lm_head._w4a8_scale_gn,
                pad_t,
            )

        if quant_mode == "fp8":
            return ops["w8a8"](
                hidden_states,
                lm_head._w8a8_weight_kn,
                lm_head._w8a8_scale,
            )
        if getattr(lm_head, "_w4a8_is_2d", False):
            return ops["w4a8_2d"](
                hidden_states,
                lm_head._w4a8_weight_kn,
                lm_head._w4a8_scale,
            )
        return ops["w4a8_3d"](
            hidden_states,
            lm_head._w4a8_weight_3d,
            lm_head._w4a8_scale_gn,
        )

    # CPU execution path (unit tests and host validation).
    x_q, _ = _quantize_fp8_per_row(hidden_states.float())
    x_f32 = x_q.float()
    if quant_mode == "fp8":
        logits = (x_f32 @ lm_head._w8a8_weight_kn.float()) * (
            lm_head._w8a8_scale.float().unsqueeze(0)
        )
    elif getattr(lm_head, "_w4a8_is_2d", False):
        logits = (x_f32 @ lm_head._w4a8_weight_kn.float()) * (
            lm_head._w4a8_scale.float().unsqueeze(0)
        )
    else:
        w_3d = lm_head._w4a8_weight_3d.float()
        s_gn = lm_head._w4a8_scale_gn.float()
        num_groups, group_size, _ = w_3d.shape
        x_3d = x_f32.reshape(x_f32.shape[0], num_groups, group_size)
        p = torch.einsum("bgk,gkn->gbn", x_3d, w_3d)
        logits = (p * s_gn.unsqueeze(1)).sum(dim=0)

    if isinstance(pad_buf, torch.Tensor) and (
        getattr(processor, "vocab_size", 0)
        < getattr(processor, "padded_vocab_size", 0)
    ):
        shard_vocab = logits.shape[-1]
        col_idx = torch.arange(
            shard_vocab, device=logits.device, dtype=torch.int32
        )
        valid_cols = shard_vocab - pad_buf.to(
            device=logits.device, dtype=torch.int32
        )
        logits = logits.masked_fill(
            (col_idx >= valid_cols).unsqueeze(0), -float("inf")
        )
    elif num_pad_int > 0:
        logits[..., -num_pad_int:] = -float("inf")

    local_max_vals, local_max_indices = logits.max(dim=-1)
    return local_max_vals.float(), local_max_indices.to(torch.int32)


def tpu_get_top_tokens(
    self: LogitsProcessor,
    lm_head: torch.nn.Module,
    hidden_states: torch.Tensor,
    embedding_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """TPU-optimized vocabulary-parallel greedy ``lm_head + argmax``."""
    if self.scale <= 0.0 and self.scale != 1.0:
        raise ValueError(
            f"scale must be > 0.0 for greedy argmax reduction, got {self.scale}"
        )
    tp_size = lm_head.tp_size
    quant_mode = getattr(lm_head, "_quant_mode", None)
    if (
        quant_mode in ("fp8", "fp4")
        and embedding_bias is None
        and self.soft_cap is None
    ):
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

    pad_buf = getattr(lm_head, "num_org_vocab_padding", None)
    if isinstance(pad_buf, torch.Tensor) and (
        getattr(self, "vocab_size", 0) < getattr(self, "padded_vocab_size", 0)
    ):
        shard_vocab = logits.shape[-1]
        col_idx = torch.arange(
            shard_vocab, device=logits.device, dtype=torch.int32
        )
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

    vocab_start = getattr(
        lm_head,
        "org_vocab_start_index",
        lm_head.shard_indices.org_vocab_start_index,
    )
    if tp_size == 1:
        local_max_indices = logits.argmax(dim=-1)
        if isinstance(vocab_start, torch.Tensor):
            local_max_indices = local_max_indices.to(
                torch.int32
            ) + vocab_start.to(device=logits.device, dtype=torch.int32)
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


LogitsProcessor.get_top_tokens = tpu_get_top_tokens


def _extract_local_topk32_torch(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two-stage local Top-32 extraction in PyTorch (matches ``_extract_local_topk32_jax``)."""
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
        vals32, ids32 = torch.topk(
            logits_pad, k=LOCAL_TOPK_CANDIDATES, dim=-1
        )
        return vals32.float(), ids32.to(torch.int32)

    if shard_vocab % block_size != 0:
        pad_cols = block_size - (shard_vocab % block_size)
        lg_padded = torch.nn.functional.pad(
            logits, (0, pad_cols), value=-float("inf")
        )
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
    intra = torch.arange(
        block_size, device=logits.device, dtype=torch.int32
    ).view(1, 1, block_size)
    local_ids512 = (
        blk_ids.to(torch.int32).unsqueeze(-1) * block_size + intra
    ).reshape(t_rows, num_local_blocks * block_size)
    vals32, idx32 = torch.topk(
        local_cand512.float(), k=LOCAL_TOPK_CANDIDATES, dim=-1
    )
    ids32 = torch.gather(local_ids512, 1, idx32)
    return vals32.float(), ids32.to(torch.int32)


def _quantized_head_local_topk(
    processor: LogitsProcessor,
    lm_head: torch.nn.Module,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``(local_topk_vals, local_topk_indices)`` of shape ``[rows, 32]``
    for a ``W8A8`` or ``W4A8`` quantized ``lm_head``."""
    quant_mode = getattr(lm_head, "_quant_mode", None)
    pad_buf = getattr(lm_head, "num_org_vocab_padding", None)
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
            if quant_mode == "fp8":
                return ops["w8a8_topk32_padded"](
                    hidden_states,
                    lm_head._w8a8_weight_kn,
                    lm_head._w8a8_scale,
                    pad_t,
                )
            if getattr(lm_head, "_w4a8_is_2d", False):
                return ops["w4a8_2d_topk32_padded"](
                    hidden_states,
                    lm_head._w4a8_weight_kn,
                    lm_head._w4a8_scale,
                    pad_t,
                )
            return ops["w4a8_3d_topk32_padded"](
                hidden_states,
                lm_head._w4a8_weight_3d,
                lm_head._w4a8_scale_gn,
                pad_t,
            )

        if quant_mode == "fp8":
            return ops["w8a8_topk32"](
                hidden_states,
                lm_head._w8a8_weight_kn,
                lm_head._w8a8_scale,
            )
        if getattr(lm_head, "_w4a8_is_2d", False):
            return ops["w4a8_2d_topk32"](
                hidden_states,
                lm_head._w4a8_weight_kn,
                lm_head._w4a8_scale,
            )
        return ops["w4a8_3d_topk32"](
            hidden_states,
            lm_head._w4a8_weight_3d,
            lm_head._w4a8_scale_gn,
        )

    # CPU execution path (unit tests and host validation).
    x_q, x_s = _quantize_fp8_per_row(hidden_states.float())
    x_f32 = x_q.float()
    if quant_mode == "fp8":
        logits = (x_f32 @ lm_head._w8a8_weight_kn.float()) * (
            x_s.unsqueeze(-1) * lm_head._w8a8_scale.float().unsqueeze(0)
        )
    elif getattr(lm_head, "_w4a8_is_2d", False):
        logits = (x_f32 @ lm_head._w4a8_weight_kn.float()) * (
            x_s.unsqueeze(-1) * lm_head._w4a8_scale.float().unsqueeze(0)
        )
    else:
        w_3d = lm_head._w4a8_weight_3d.float()
        s_gn = lm_head._w4a8_scale_gn.float()
        num_groups, group_size, _ = w_3d.shape
        x_3d = x_f32.reshape(x_f32.shape[0], num_groups, group_size)
        p = torch.einsum("bgk,gkn->gbn", x_3d, w_3d)
        logits = (p * s_gn.unsqueeze(1)).sum(dim=0) * x_s.unsqueeze(-1)
    logits = logits.to(hidden_states.dtype)

    if isinstance(pad_buf, torch.Tensor) and (
        getattr(processor, "vocab_size", 0)
        < getattr(processor, "padded_vocab_size", 0)
    ):
        shard_vocab = logits.shape[-1]
        col_idx = torch.arange(
            shard_vocab, device=logits.device, dtype=torch.int32
        )
        valid_cols = shard_vocab - pad_buf.to(
            device=logits.device, dtype=torch.int32
        )
        logits = logits.masked_fill(
            (col_idx >= valid_cols).unsqueeze(0), -float("inf")
        )
    elif num_pad_int > 0:
        logits[..., -num_pad_int:] = -float("inf")

    return _extract_local_topk32_torch(logits)


def tpu_get_topk_candidates(
    self: LogitsProcessor,
    lm_head: torch.nn.Module,
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
    if (
        quant_mode in ("fp8", "fp4")
        and embedding_bias is None
        and self.soft_cap is None
    ):
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

        pad_buf = getattr(lm_head, "num_org_vocab_padding", None)
        if isinstance(pad_buf, torch.Tensor) and (
            getattr(self, "vocab_size", 0)
            < getattr(self, "padded_vocab_size", 0)
        ):
            shard_vocab = logits.shape[-1]
            col_idx = torch.arange(
                shard_vocab, device=logits.device, dtype=torch.int32
            )
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
        return torch.cat(
            [local_vals.float(), global_indices.float()], dim=-1
        )

    rows = hidden_states.shape[0]
    k = local_vals.shape[1]
    local_pair = torch.stack(
        [local_vals.float(), global_indices.float()], dim=-1
    )
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
        weight = getattr(head, "weight", None)
        if not isinstance(weight, torch.Tensor):
            weight = getattr(head, "_w8a8_weight_kn", None)
        device = weight.device if isinstance(weight, torch.Tensor) else None
    shard_indices = getattr(head, "shard_indices", None)
    if shard_indices is None:
        return
    if hasattr(shard_indices, "org_vocab_start_index"):
        existing_start = getattr(head, "org_vocab_start_index", None)
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
        existing_pad = getattr(head, "num_org_vocab_padding", None)
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
    w_kn = layer._w8a8_weight_kn
    w_s = layer._w8a8_scale
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
    group_size: int = 128,
    *,
    weight: torch.Tensor | None = None,
    in_place: bool = False,
) -> torch.nn.Module:
    """Quantize ``head`` (or ``weight``) at load time for ``tpu_get_top_tokens``
    and (when ``in_place=True``) ``LogitsProcessor._apply_head``."""
    if getattr(head, "_quant_mode", None) is not None and (
        weight is None and getattr(head, "weight", None) is None
    ):
        return head

    parsed_mode, parsed_gs = parse_draft_lm_head_quant(mode, group_size)
    w = weight if weight is not None else getattr(head, "weight", None)
    if not isinstance(w, torch.Tensor) or w.ndim != 2:
        if in_place:
            return head
        raise ValueError(
            f"quantize_lm_head requires a 2D [vocab_shard, hidden_size] weight tensor, got {type(w)}"
        )
    _ensure_head_shard_buffers(head, device=w.device)
    if parsed_mode is None:
        head._quant_mode = None
        head._quant_group_size = None
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

    if parsed_mode == "fp8":
        w_fp8_nk_cpu, w_scale_f32_cpu = _quantize_fp8_per_row(
            w.detach().float().cpu()
        )
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
        head._quant_mode = "fp8"
        head._quant_group_size = None
        head._quant_desc = "w8a8"
        if in_place:
            head.quant_method = _W8A8EmbeddingQuantMethod()
            if hasattr(head, "weight"):
                delattr(head, "weight")
            logger.info_once(
                "Quantized shared ParallelLMHead in-place to W8A8 FP8: "
                "_w8a8_weight_kn=%s, freed BF16 weight.",
                tuple(w_fp8_kn.shape),
            )
        return head

    eff_gs = (
        hidden_size
        if (parsed_gs is None or parsed_gs <= 0 or parsed_gs == hidden_size)
        else int(parsed_gs)
    )
    if hidden_size % eff_gs != 0:
        raise ValueError(
            f"hidden_size ({hidden_size}) must be divisible by FP4 group_size ({eff_gs})"
        )
    num_groups = hidden_size // eff_gs
    is_2d = num_groups == 1

    if w.device.type == "tpu":
        ops = _ensure_tpu_quant_ops_registered()
        quantize_fp4_op = ops["quantize_fp4"]

        def _fresh(t: torch.Tensor) -> torch.Tensor:
            return torch.empty(t.shape, dtype=t.dtype, device=t.device).copy_(t)

        w_bf16_kn = _fresh(
            w.detach().to(dtype=torch.bfloat16).t().contiguous()
        )
        if is_2d:
            w_fp4_kn, s_fp4_2d = quantize_fp4_op(w_bf16_kn)
            w_fp4_kn = _fresh(w_fp4_kn)
            s_fp4_f32 = _fresh(s_fp4_2d.reshape(vocab_shard))
            head.register_buffer("_w4a8_weight_kn", w_fp4_kn, persistent=False)
            head.register_buffer("_w4a8_scale", s_fp4_f32, persistent=False)
        else:
            w_bf16_3d = _fresh(
                w_bf16_kn.reshape(num_groups, eff_gs, vocab_shard)
            )
            w_fp4_3d, s_fp4_3d = quantize_fp4_op(w_bf16_3d)
            w_fp4_3d = _fresh(w_fp4_3d)
            s_fp4_gn = _fresh(s_fp4_3d.reshape(num_groups, vocab_shard))
            head.register_buffer("_w4a8_weight_3d", w_fp4_3d, persistent=False)
            head.register_buffer("_w4a8_scale_gn", s_fp4_gn, persistent=False)
    else:
        w_kn_cpu = w.detach().float().cpu().t().contiguous()
        q_w, q_s, is_2d = _quantize_fp4_e2m1_cpu(w_kn_cpu, eff_gs)
        if is_2d:
            head.register_buffer(
                "_w4a8_weight_kn", q_w.to(w.device), persistent=False
            )
            head.register_buffer(
                "_w4a8_scale", q_s.to(w.device), persistent=False
            )
        else:
            head.register_buffer(
                "_w4a8_weight_3d", q_w.to(w.device), persistent=False
            )
            head.register_buffer(
                "_w4a8_scale_gn", q_s.to(w.device), persistent=False
            )

    head._quant_mode = "fp4"
    head._w4a8_is_2d = is_2d
    head._quant_group_size = eff_gs
    head._quant_desc = f"w4a8_g{eff_gs}"
    return head


class _QuantizedLMHeadProxy(torch.nn.Module):
    """Lightweight draft ``lm_head`` proxy holding quantized ``W8A8`` / ``W4A8``
    buffers without mutating the shared target ``lm_head``."""

    def __init__(
        self,
        base_head: torch.nn.Module,
        mode: str = "fp8",
        group_size: int = 128,
    ) -> None:
        super().__init__()
        self.tp_size = base_head.tp_size
        self.shard_indices = base_head.shard_indices
        object.__setattr__(self, "_fallback_head", base_head)
        w = getattr(base_head, "weight", None)
        if w is None and getattr(base_head, "_quant_mode", None) == "fp8":
            w = (
                (
                    base_head._w8a8_weight_kn.detach().float().cpu()
                    * base_head._w8a8_scale.detach().float().cpu().unsqueeze(0)
                )
                .t()
                .to(torch.bfloat16)
            )
        quantize_lm_head(self, mode=mode, group_size=group_size, weight=w)
        # A head already quantized to fp8 has no weight; w was dequantized on the host,
        # so quantize_lm_head registered this proxy's buffers on the CPU. The compiled
        # draft programs need them where the head lives (ix-unified-port: band H,
        # TPU_DRAFT_LM_HEAD_QUANT=fp4, crashed in capture_model with a CPU
        # _w4a8_weight_3d argument). Same values, only the device changes.
        _move_buffers_to(self, _head_device(base_head))


def _head_device(head: torch.nn.Module) -> torch.device | None:
    """Device of a head's weight, or of its fp8 buffer once the weight was freed."""
    for name in ("weight", "_w8a8_weight_kn"):
        t = getattr(head, name, None)
        if isinstance(t, torch.Tensor):
            return t.device
    return None


def _move_buffers_to(module: torch.nn.Module, device: torch.device | None) -> None:
    """Re-register ``module``'s own buffers on ``device`` (no-op when None or already there)."""
    if device is None:
        return
    for name, buf in list(module.named_buffers(recurse=False)):
        if buf is not None and buf.device != device:
            module.register_buffer(name, buf.to(device), persistent=False)


class _DraftHeadOwnerProxy(torch.nn.Module):
    """Owner proxy pairing ``logits_processor`` with a quantized draft
    ``lm_head`` proxy."""

    def __init__(
        self, logits_processor: LogitsProcessor, lm_head: torch.nn.Module
    ) -> None:
        super().__init__()
        self.logits_processor = logits_processor
        self.lm_head = lm_head


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
        if hasattr(processor, "get_top_tokens") and hasattr(head, "shard_indices"):
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
    processor = cast("LogitsProcessor", owner.logits_processor)
    head = cast("VocabParallelEmbedding", owner.lm_head)
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


def num_packed_candidates(owner: torch.nn.Module) -> int:
    """Total number of vocabulary-parallel candidates across TP ranks."""
    return LOCAL_TOPK_CANDIDATES * int(owner.lm_head.tp_size)


def topk_candidates(
    owner: torch.nn.Module, hidden_states: torch.Tensor
) -> torch.Tensor:
    """Packed ``[rows, 2 * C]`` ``float32`` vocabulary-parallel top-32
    candidates per TP rank (``C = 32 * tp_size``)."""
    return owner.logits_processor.get_topk_candidates(
        owner.lm_head, hidden_states
    )


def log_topk_candidates_step_path(
    is_spec_step: bool, req_ids: Sequence[str | None]
) -> None:
    """Logs, once per path, that a step used vocabulary-parallel top-k candidates."""
    if all(req_id is None or req_id.startswith("__") for req_id in req_ids):
        return
    if is_spec_step:
        logger.info_once(
            "TPU_GREEDY_TOP_TOKENS: a speculative verify step took vocab-parallel top-32 candidates"
        )
    else:
        logger.info_once(
            "TPU_GREEDY_TOP_TOKENS: a non-speculative step took vocab-parallel top-32 candidates"
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
