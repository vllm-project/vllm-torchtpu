# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_DRAFT_LMHEAD_FP8: the MTP draft's own FP8 copy of the shared lm_head.

The Qwen4Exp draft borrows the target's vocab-parallel lm_head (`has_own_lm_head =
False`). Each draft step reads that rank's whole bf16 shard (~159 MB) for a few rows,
so the draft lm-head is weight-bandwidth bound. With the flag on, the draft gets its
own FP8 E4M3 copy of the shard, one float32 scale per vocab row (the per-channel
scheme of the online-FP8 dense linears, `quantize_online_fp8`), and projects through
the repo's FP8 dense matmul (`linear_common.quantized_matmul`: per-row dynamic FP8
activations, f32 accumulate, bf16 logits). The target's lm_head module, weight and
logits are untouched: the copy is a new module bound only on the draft.

NOT bitwise: draft logits carry FP8 rounding error, so the draft argmax can flip where
the top logits nearly tie. Target verify and rejection sampling are unchanged, so the
output distribution is unchanged; only the real acceptance length can move.

Every stock draft path reaches the head through `logits_processor(lm_head, h)` or
`LogitsProcessor._get_logits(h, lm_head, ...)` (compute_logits, the vocab-shard
argmax, the fused loop), all of which call `lm_head.quant_method.apply`. The
megakernel draft (VLLM_TPU_MK_DRAFT) packs raw bf16 head tiles and keeps the bf16
head (`bf16_lm_head`).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from vllm_torchtpu import envs
from vllm_torchtpu.layers.core.quantization import quantize_tensor

# Attributes of the borrowed head that the logits processor, the vocab-shard argmax
# (`shard_argmax._usable`) and the greedy-top-tokens owner checks read.
_HEAD_ATTRS = (
    "tp_size",
    "tp_rank",
    "org_vocab_size",
    "org_vocab_size_padded",
    "num_embeddings",
    "num_embeddings_padded",
    "num_embeddings_per_partition",
    "embedding_dim",
    "shard_indices",
)


def quantized_matmul(x: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor):
    """`linear_common.quantized_matmul` (imported lazily: it pulls in torch_tpu)."""
    from vllm_torchtpu.layers.adapter.linear_common import quantized_matmul as qmm

    return qmm(x, w_q, w_s)


def quantize_head_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`(w_q [H, V] float8_e4m3fn, w_s [V] float32)` of a `[V, H]` head shard: one
    scale per vocab row (absmax / 448), canonical (k, n) layout for the matmul.

    `w_s` is an owned buffer: quantize_tensor's scale is a view of a `[V, 1, 1]`
    tensor (the reshape keeps the view, so `.contiguous()` is a no-op), and torch_tpu
    runs an eager execute_as_strided before every compiled call that takes a buffer
    with a view in its history. `clone()` keeps that history, a host round trip does
    not (megakernel_runtime/operand_hold.py). Same values, ~V*4 bytes once at load."""
    q, s = quantize_tensor(weight.to(torch.float32).t(), torch.float8_e4m3fn, axis=0)
    s = s.reshape(weight.shape[0])
    return q.contiguous(), s.to("cpu", copy=True).to(s.device)


class _Fp8HeadMethod:
    """The `quant_method` the logits processor calls on the draft head."""

    def apply(self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None):
        out = quantized_matmul(x, layer.w_q, layer.w_s)
        return out if bias is None else out + bias

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        del layer  # quantized once, from the loaded target shard


class Fp8DraftLMHead(nn.Module):
    """FP8 copy of one rank's vocab-parallel head shard, same TP layout."""

    def __init__(self, head: nn.Module) -> None:
        super().__init__()
        for name in _HEAD_ATTRS:
            if hasattr(head, name):
                setattr(self, name, getattr(head, name))
        if getattr(head, "bias", None) is not None:
            raise NotImplementedError("FP8 draft lm_head: head bias not supported")
        with torch.no_grad():
            w_q, w_s = quantize_head_weight(head.weight.detach())
        # Parameters, as the borrowed weight and the FP8 dense linears' weights are,
        # so the compiled draft programs take them the same way.
        self.w_q = nn.Parameter(w_q, requires_grad=False)
        self.w_s = nn.Parameter(w_s, requires_grad=False)
        self.quant_method = _Fp8HeadMethod()
        # The runner pre-warms the op (`_initialize_quantization_kernels`): its lazy
        # build takes a Lock that Dynamo cannot trace. Build it here too, eagerly.
        from vllm_torchtpu.layers.adapter.linear_common import _get_quantized_matmul_op

        _get_quantized_matmul_op()

    def extra_repr(self) -> str:
        return f"w_q={tuple(self.w_q.shape)} {self.w_q.dtype}, per-row f32 scale"


def maybe_fp8_draft_lm_head(draft: Any) -> bool:
    """With VLLM_TPU_DRAFT_LMHEAD_FP8, bind an FP8 copy of the draft's (borrowed) head
    as `draft.lm_head`; the bf16 head stays reachable as `draft.bf16_lm_head` (not a
    submodule, so the target's parameters are not re-registered under the draft).
    Returns whether it bound. The bf16 head object and its weight are not modified."""
    if not envs.VLLM_TPU_DRAFT_LMHEAD_FP8:
        return False
    head = draft.lm_head
    if isinstance(head, Fp8DraftLMHead):
        return True
    fp8 = Fp8DraftLMHead(head)
    object.__setattr__(draft, "bf16_lm_head", head)
    draft._modules["lm_head"] = fp8
    return True


def bf16_lm_head(draft: Any) -> nn.Module:
    """The draft's bf16 head (the shared target head) whether or not the FP8 copy is
    bound; raw-weight consumers (the megakernel draft) read this one."""
    head = getattr(draft, "bf16_lm_head", None)
    return draft.lm_head if head is None else head
