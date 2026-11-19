# Copyright 2025 Google LLC
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
"""TPU pluggable attention backends for multimodal encoders (ViT).

vLLM's ``MMEncoderAttention`` is a ``CustomOp``. ``vllm-torchtpu`` registers
``TpuPlatform`` as an out-of-tree plugin, so ``CustomOp.dispatch_forward``
routes to ``forward_oot`` (``vllm/model_executor/custom_op.py``).
``TpuMMEncoderAttention`` dispatches by ``self.attn_backend`` (selected via
``--mm-encoder-attn-backend`` and ``TpuPlatform.get_vit_attn_backend``):

- ``FLASH_ATTN`` (default): routes packed variable-length vision attention to
  the in-tree Pallas ``flash_attention`` kernel with ``SegmentIds`` derived from
  ``cu_seqlens`` instead of SDPA (``forward_native``), which ``torch_tpu``
  lowers to the slower SHLO/MATH path. GQA / head-count-mismatch encoders (e.g.
  Whisper/ASR) that the flash kernel cannot serve fall back to
  ``forward_native``.
- ``FLYWHEEL``: ``flywheel_tpu.flash_attn_varlen_func`` wrapped in a shared
  ``jax.jit`` + ``pallas.jax_op``, supporting variable-length ``cu_seqlens``,
  MHA and GQA, unpadded paired ``token_major=True`` at ``head_dim=64`` (even
  MHA heads), and 128-lane padded ``head_dim`` up to 256.
- ``TORCH_SDPA``: upstream ``forward_native`` fallback.

Both Pallas paths are safe under ``torch.compile`` (e.g.
``@support_torch_compile(is_encoder=True)`` in ``vit_native_compile.py``): jax
ops are built eagerly in ``__init__`` (never inside a compiled region),
``sm_scale`` is folded into ``q`` so one static op serves every layer, and
``segment_ids`` on ``FLASH_ATTN`` are built with tensor ops only
(``seg[t] = #{cu_seqlens[i] <= t}`` via a broadcast compare, giving each image
and the flash padding a distinct equality-masked segment).
"""

from collections.abc import Callable
from typing import Any

import jax
import torch
import torch.nn.functional as F
from torch_tpu._internal import pallas
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from vllm_torchtpu.kernels.flash_attention.kernel import SegmentIds, flash_attention
from vllm_torchtpu.layers.adapter.attention_backend import MAX_FLYWHEEL_VIT_HEAD_DIM
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Flash kernel q-block size and lane alignment.
_BLOCK_SIZE = 128
_HEAD_DIM_ALIGNMENT = 128
_PAIRED_HEAD_DIM = 64
_VMEM_LIMIT_BYTES = 64 * 1024 * 1024


def _vision_flash_core(
    q: jax.Array, k: jax.Array, v: jax.Array, q_seg: jax.Array, kv_seg: jax.Array
) -> jax.Array:
    # q,k,v: [batch, num_heads, seq, head_dim]; seg: [batch, seq]. sm_scale is
    # folded into q by the caller, so use 1.0 here.
    return flash_attention(
        q,
        k,
        v,
        segment_ids=SegmentIds(q=q_seg, kv=kv_seg),
        causal=False,
        sm_scale=1.0,
        vmem_limit_bytes=_VMEM_LIMIT_BYTES,
    )


def _build_flywheel_ops() -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Build shared jax.jit + pallas.jax_op wrappers for FlyWheel ViT attention.

    Lazily imports flywheel_tpu only when a FLYWHEEL encoder layer initializes,
    and jits flash_attn_varlen_func once at module scope so all encoder layers
    share a single lowered Pallas executable per static shape.
    """
    import flywheel_tpu

    jitted_varlen = jax.jit(
        flywheel_tpu.flash_attn_varlen_func,
        static_argnames=(
            "max_seqlen_q",
            "max_seqlen_k",
            "softmax_scale",
            "causal",
            "interpret",
            "token_major",
            "head_dim",
        ),
    )

    def _vision_flywheel_token_major_d64(
        q: jax.Array, k: jax.Array, v: jax.Array, cu_seqlens: jax.Array
    ) -> jax.Array:
        out = jitted_varlen(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            max_seqlen_q=q.shape[0],
            max_seqlen_k=k.shape[0],
            softmax_scale=1.0,
            causal=False,
            token_major=True,
            head_dim=_PAIRED_HEAD_DIM,
        )
        assert isinstance(out, jax.Array)
        return out

    def _vision_flywheel_head_major(
        q: jax.Array, k: jax.Array, v: jax.Array, cu_seqlens: jax.Array
    ) -> jax.Array:
        out = jitted_varlen(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            max_seqlen_q=q.shape[1],
            max_seqlen_k=k.shape[1],
            softmax_scale=1.0,
            causal=False,
            token_major=False,
        )
        assert isinstance(out, jax.Array)
        return out

    token_major_op = pallas.jax_op(
        "pallas::vision_flywheel_token_major_d64",
        _vision_flywheel_token_major_d64,
    )
    token_major_op.register_fake(lambda q, k, v, cu_seqlens: torch.empty_like(q))

    head_major_op = pallas.jax_op(
        "pallas::vision_flywheel_head_major",
        _vision_flywheel_head_major,
    )
    head_major_op.register_fake(lambda q, k, v, cu_seqlens: torch.empty_like(q))
    return token_major_op, head_major_op


@CustomOp.register_oot(name="MMEncoderAttention")
class TpuMMEncoderAttention(MMEncoderAttention):
    """MMEncoderAttention with pluggable TPU backends (FLASH_ATTN, FLYWHEEL, SDPA)."""

    # Default for unit tests constructing instances via __new__.
    attn_backend: AttentionBackendEnum = AttentionBackendEnum.FLASH_ATTN

    # Shared jax ops (sm_scale folded into q -> scale-agnostic), built eagerly
    # in __init__ -- never inside a torch.compile region.
    _vision_op: Callable[..., Any] | None = None
    _flywheel_token_major_op: Callable[..., Any] | None = None
    _flywheel_head_major_op: Callable[..., Any] | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if (
            self.attn_backend == AttentionBackendEnum.FLASH_ATTN
            and TpuMMEncoderAttention._vision_op is None
        ):
            op = pallas.jax_op("pallas::vision_flash_attention", _vision_flash_core)
            op.register_fake(lambda q, k, v, q_seg, kv_seg: torch.empty_like(q))
            TpuMMEncoderAttention._vision_op = op
        elif (
            self.attn_backend.name == "FLYWHEEL"
            and TpuMMEncoderAttention._flywheel_head_major_op is None
        ):
            tm_op, hm_op = _build_flywheel_ops()
            TpuMMEncoderAttention._flywheel_token_major_op = tm_op
            TpuMMEncoderAttention._flywheel_head_major_op = hm_op

    def forward_oot(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
        sequence_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)
        if q_len != kv_len:
            return self.forward_native(
                query, key, value, cu_seqlens, max_seqlen, sequence_lengths
            )

        backend = self.attn_backend
        if backend == AttentionBackendEnum.TORCH_SDPA:
            return self.forward_native(
                query, key, value, cu_seqlens, max_seqlen, sequence_lengths
            )

        if backend.name == "FLYWHEEL":
            assert 0 < self.head_size <= MAX_FLYWHEEL_VIT_HEAD_DIM, (
                f"FlyWheel ViT head_size must be in (0, {MAX_FLYWHEEL_VIT_HEAD_DIM}], "
                f"got {self.head_size}"
            )
            is_reshaped = query.dim() != 4
            query, key, value = self.view_qkv_to_4d(
                query, key, value, bsz, q_len, kv_len
            )
            output = self._flywheel_attention(query, key, value, cu_seqlens)
            if is_reshaped:
                output = output.reshape(bsz, q_len, -1)
            return output

        # Default TPU ViT backend: in-tree Pallas flash kernel (MHA only).
        if self.num_heads != self.num_kv_heads:
            return self.forward_native(
                query, key, value, cu_seqlens, max_seqlen, sequence_lengths
            )
        is_reshaped = query.dim() != 4
        query, key, value = self.view_qkv_to_4d(query, key, value, bsz, q_len, kv_len)
        output = self._flash_attention(query, key, value, cu_seqlens)
        if is_reshaped:
            output = output.reshape(bsz, q_len, -1)
        return output

    def _flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
    ) -> torch.Tensor:
        """q,k,v: ``[batch, seq, num_heads, head_dim]``; cu_seqlens packs images."""
        bsz, seq_len, num_heads, head_dim = q.shape
        pad = (-seq_len) % _BLOCK_SIZE  # flash kernel needs seq % 128 == 0
        padded = seq_len + pad

        # segment ids (compile-safe): seg[t] = number of cu_seqlens boundaries
        # <= t. Real image i -> id i+1; flash-padding positions (>=
        # cu_seqlens[-1]) -> the highest id. The kernel mask is pure equality, so
        # real tokens attend only within their image and never to padding.
        positions = torch.arange(padded, device=q.device)
        if cu_seqlens is not None:
            cu = cu_seqlens.to(device=q.device, dtype=positions.dtype)
            seg = (positions[None, :] >= cu[:, None]).sum(dim=0).to(torch.int32)
            seg = seg.unsqueeze(0).expand(bsz, padded).contiguous()
        else:
            seg = torch.zeros((bsz, padded), dtype=torch.int32, device=q.device)
            seg[:, :seq_len] = 1

        def _prep(x: torch.Tensor) -> torch.Tensor:
            # [b, s, h, d] -> pad seq -> [b, h, s_pad, d]
            if pad:
                x = F.pad(x, (0, 0, 0, 0, 0, pad))
            return x.permute(0, 2, 1, 3).contiguous()

        op = TpuMMEncoderAttention._vision_op
        # __init__ builds the shared op before any forward runs.
        assert op is not None
        # Fold sm_scale into q so the op stays scale-agnostic.
        out = op(_prep(q * self.scale), _prep(k), _prep(v), seg, seg)
        out = out.permute(0, 2, 1, 3)  # [b, s_pad, h, d]
        return out[:, :seq_len].contiguous()

    def _flywheel_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run FlyWheel varlen flash attention on 4D ``[B, S, H, D]`` inputs."""
        bsz, seq_len, num_heads, head_dim = q.shape
        num_kv_heads = k.shape[2]
        orig_dtype = q.dtype
        total_tokens = bsz * seq_len

        if cu_seqlens is not None:
            cu = cu_seqlens.to(device=q.device, dtype=torch.int32).contiguous()
        else:
            cu = torch.arange(
                0,
                (bsz + 1) * seq_len,
                seq_len,
                dtype=torch.int32,
                device=q.device,
            )

        q_bf16 = (q * self.scale).to(torch.bfloat16)
        k_bf16 = k.to(torch.bfloat16)
        v_bf16 = v.to(torch.bfloat16)

        if (
            head_dim == _PAIRED_HEAD_DIM
            and num_heads == num_kv_heads
            and num_heads % 2 == 0
        ):
            tm_op = TpuMMEncoderAttention._flywheel_token_major_op
            assert tm_op is not None
            fused_dim = num_heads * _PAIRED_HEAD_DIM
            out_2d = tm_op(
                q_bf16.reshape(total_tokens, fused_dim).contiguous(),
                k_bf16.reshape(total_tokens, fused_dim).contiguous(),
                v_bf16.reshape(total_tokens, fused_dim).contiguous(),
                cu,
            )
            return out_2d.reshape(bsz, seq_len, num_heads, head_dim).to(orig_dtype)

        hm_op = TpuMMEncoderAttention._flywheel_head_major_op
        assert hm_op is not None
        pad_hd = (-head_dim) % _HEAD_DIM_ALIGNMENT

        def _prep_head_major(x: torch.Tensor, heads: int) -> torch.Tensor:
            flat = x.reshape(total_tokens, heads, head_dim)
            if pad_hd:
                flat = F.pad(flat, (0, pad_hd))
            return flat.permute(1, 0, 2).contiguous()

        out_3d = hm_op(
            _prep_head_major(q_bf16, num_heads),
            _prep_head_major(k_bf16, num_kv_heads),
            _prep_head_major(v_bf16, num_kv_heads),
            cu,
        )
        out = out_3d.permute(1, 0, 2)
        if pad_hd:
            out = out[:, :, :head_dim]
        return (
            out.reshape(bsz, seq_len, num_heads, head_dim).contiguous().to(orig_dtype)
        )

    # vllm-torchtpu registers TpuPlatform as an out-of-tree plugin, so vLLM's
    # CustomOp.dispatch_forward routes to forward_oot. Alias forward_tpu to the
    # same impl so an in-tree TPU platform would also pick it up.
    forward_tpu = forward_oot
