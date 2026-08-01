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
"""TPU flash-attention for vision encoders (ViT) as an out-of-tree CustomOp.

vLLM's ``MMEncoderAttention`` is a ``CustomOp``. torchtpu-vllm registers its TPU
platform as an out-of-tree plugin, so ``CustomOp.dispatch_forward`` routes to
``forward_oot`` (vllm/model_executor/custom_op.py). Following the vLLM
device-extension design (docs/design/custom_op.md), we register an out-of-tree
subclass via ``CustomOp.register_oot`` whose ``forward_oot`` routes packed
variable-length vision attention to the Pallas ``flash_attention`` kernel
with ``segment_ids`` derived from ``cu_seqlens`` -- instead of the default SDPA
(``forward_native``), which torch_tpu lowers to the slow SHLO/MATH path. This
replaces MMEncoderAttention everywhere vLLM instantiates it (e.g. inside
Qwen2_5_VisionAttention), with no monkeypatching of vLLM internals.

GQA / head-count-mismatch encoders (e.g. Whisper/ASR) that the flash kernel
cannot serve fall back to ``forward_native``.

The path is safe under ``torch.compile`` (so the ViT can be compiled via the
native ``@support_torch_compile(is_encoder=True)`` path, see
``vit_native_compile.py``): the jax op is built eagerly in ``__init__`` (never
inside a compiled region), ``sm_scale`` is folded into ``q`` so one static op
serves every layer, and ``segment_ids`` are built with tensor ops only --
``seg[t] = #{cu_seqlens[i] <= t}`` via a broadcast compare, so each image (and
the flash padding) gets a distinct, equality-masked segment.
"""
import jax
import torch
import torch.nn.functional as F
from torch_tpu._internal import pallas
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.attention.mm_encoder_attention import \
    MMEncoderAttention

from vllm_torchtpu.kernels.flash_attention.kernel import (SegmentIds,
                                                          flash_attention)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Flash kernel q-block size; the packed sequence is padded up to a multiple.
_BLOCK_SIZE = 128
_VMEM_LIMIT_BYTES = 64 * 1024 * 1024


def _vision_flash_core(q: jax.Array, k: jax.Array, v: jax.Array,
                       q_seg: jax.Array, kv_seg: jax.Array) -> jax.Array:
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


@CustomOp.register_oot(name="MMEncoderAttention")
class TpuMMEncoderAttention(MMEncoderAttention):
    """MMEncoderAttention whose TPU path uses the Pallas flash kernel."""

    # One shared jax op (sm_scale folded into q -> scale-agnostic), built once,
    # eagerly -- never inside a compiled region.
    _vision_op = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if TpuMMEncoderAttention._vision_op is None:
            op = pallas.jax_op("pallas::vision_flash_attention",
                               _vision_flash_core)
            op.register_fake(
                lambda q, k, v, q_seg, kv_seg: torch.empty_like(q))
            TpuMMEncoderAttention._vision_op = op

    def forward_oot(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            cu_seqlens: torch.Tensor | None = None,
            max_seqlen: torch.Tensor | None = None,
            sequence_lengths: torch.Tensor | None = None) -> torch.Tensor:
        # The flash kernel requires equal q/kv head counts; GQA encoders fall
        # back to the default SDPA path.
        if self.num_heads != self.num_kv_heads:
            return self.forward_native(query, key, value, cu_seqlens,
                                       max_seqlen, sequence_lengths)
        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)
        if q_len != kv_len:
            return self.forward_native(query, key, value, cu_seqlens,
                                       max_seqlen, sequence_lengths)
        is_reshaped = query.dim() != 4
        query, key, value = self.view_qkv_to_4d(query, key, value, bsz, q_len,
                                                kv_len)
        output = self._flash_attention(query, key, value, cu_seqlens)
        if is_reshaped:
            output = output.reshape(bsz, q_len, -1)
        return output

    def _flash_attention(self, q: torch.Tensor, k: torch.Tensor,
                         v: torch.Tensor,
                         cu_seqlens: torch.Tensor | None) -> torch.Tensor:
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
            cu = cu_seqlens.to(positions.dtype)
            seg = (positions[None, :] >= cu[:,
                                            None]).sum(dim=0).to(torch.int32)
            seg = seg.unsqueeze(0).expand(bsz, padded).contiguous()
        else:
            seg = torch.zeros((bsz, padded),
                              dtype=torch.int32,
                              device=q.device)
            seg[:, :seq_len] = 1

        def _prep(x: torch.Tensor) -> torch.Tensor:
            # [b, s, h, d] -> pad seq -> [b, h, s_pad, d]
            if pad:
                x = F.pad(x, (0, 0, 0, 0, 0, pad))
            return x.permute(0, 2, 1, 3).contiguous()

        op = TpuMMEncoderAttention._vision_op
        # Fold sm_scale into q so the op stays scale-agnostic.
        out = op(_prep(q * self.scale), _prep(k), _prep(v), seg, seg)
        out = out.permute(0, 2, 1, 3)  # [b, s_pad, h, d]
        return out[:, :seq_len].contiguous()

    # torchtpu-vllm registers TpuPlatform as an out-of-tree plugin, so vLLM's
    # CustomOp.dispatch_forward routes to forward_oot. Alias forward_tpu to the
    # same impl so an in-tree TPU platform would also pick it up.
    forward_tpu = forward_oot
