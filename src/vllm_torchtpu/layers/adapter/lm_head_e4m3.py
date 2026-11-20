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
"""e4m3 LM head with one scale per vocab row (``TPU_LM_HEAD_E4M3``).

The LM head is a weight-bound matmul: every logits call (target verify and
the draft) reads this rank's whole vocab shard, 25008 x 6144 bf16 = 307 MB
for a 200064-token vocab at TP 8. Stored as e4m3 with a float32 scale per
row it reads half the bytes. Each row is scaled by its own absolute maximum
over 448, so every row uses the full e4m3 range; the matmul runs in the
activation dtype on the converted weight and the row scale multiplies the
output column. Not exact: logits differ from the bf16 head by the e4m3
rounding of the weight (about 2^-4 relative per element).
"""

import torch
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_E4M3_MAX = 448.0


def quantize_rows(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``weight`` [rows, hidden] -> (e4m3 weight, float32 scale [rows])."""
    w = weight.float()
    amax = w.abs().amax(dim=1)
    scale = torch.where(amax > 0, amax / _E4M3_MAX, torch.ones_like(amax))
    q = (w / scale[:, None]).clamp(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn)
    return q, scale


class E4m3RowLMHeadMethod(UnquantizedEmbeddingMethod):
    """Logits from an e4m3 weight and a per-row scale."""

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = torch.matmul(x, layer.weight.to(x.dtype).t())
        out = (out.float() * layer.weight_row_scale).to(x.dtype)
        if bias is not None:
            out = out + bias
        return out

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("the e4m3 LM head is not an input embedding")


def _blocker(head: torch.nn.Module, processors: list) -> str | None:
    if type(head.quant_method) is not UnquantizedEmbeddingMethod:
        return f"the LM head uses {type(head.quant_method).__name__}"
    if head.weight.dtype != torch.bfloat16:
        return f"the LM head weight is {head.weight.dtype}"
    for p in processors:
        if p is not None and getattr(p, "head_dtype", None) is not None:
            return "a logits processor sets head_dtype"
    return None


def arm(model: torch.nn.Module, draft: torch.nn.Module | None) -> bool:
    """Convert the target's LM head (shared with the draft when the draft
    holds the same module) to e4m3 rows. Returns whether it was converted."""
    head = getattr(model, "lm_head", None)
    if head is None:
        logger.warning("TPU_LM_HEAD_E4M3=1 ignored: the model has no lm_head")
        return False
    draft_head = getattr(draft, "lm_head", None) if draft is not None else None
    if draft_head is not None and draft_head is not head:
        logger.warning(
            "TPU_LM_HEAD_E4M3=1 ignored: the draft holds its own LM head"
        )
        return False
    procs = [
        getattr(model, "logits_processor", None),
        getattr(draft, "logits_processor", None) if draft is not None else None,
    ]
    why = _blocker(head, procs)
    if why is not None:
        logger.warning("TPU_LM_HEAD_E4M3=1 ignored: %s", why)
        return False
    q, scale = quantize_rows(head.weight.data)
    head.weight = torch.nn.Parameter(q, requires_grad=False)
    head.register_buffer("weight_row_scale", scale, persistent=False)
    head.quant_method = E4m3RowLMHeadMethod()
    logger.info(
        "TPU_LM_HEAD_E4M3=1: LM head shard %s stored as e4m3 with a per-row "
        "scale (shared with the draft: %s)",
        tuple(q.shape),
        draft_head is head,
    )
    return True
