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

import torch
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)


@LogitsProcessor.register_oot
class TpuLogitsProcessor(LogitsProcessor):
    """LogitsProcessor that applies soft_cap and scale before TP all-gather.

    Upstream applies ``soft_cap`` and ``scale`` after ``_gather_logits()``,
    which runs the elementwise tanh/division/multiply on every TP rank over the
    full vocabulary ``[B, V]``. Applying them on each rank's local shard
    ``[B, V_shard]`` before the all-gather divides VPU work by ``tp_size`` and
    lets XLA fuse the elementwise epilogue into the local ``lm_head``
    projection before the collective.

    Only the multi-rank gathering path with active ``soft_cap`` or ``scale`` is
    overridden; everything else delegates to ``LogitsProcessor.forward``.
    """

    def _apply_soft_cap_and_scale(self, logits: torch.Tensor) -> torch.Tensor:
        if self.soft_cap is not None:
            logits = logits / self.soft_cap
            logits = torch.tanh(logits)
            logits = logits * self.soft_cap
        if self.scale != 1.0:
            logits = logits * self.scale
        return logits

    def forward(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
        skip_gather: bool = False,
    ) -> torch.Tensor | None:
        # LogitsProcessorWithLoRA calls type(self.base_layer).forward(self, ...)
        # unbound and adds the LoRA delta post-gather inside _get_logits, so
        # wrapped, non-gathering, TP=1, and unscaled calls keep upstream order.
        if (
            not isinstance(self, TpuLogitsProcessor)
            or self.logits_as_input
            or skip_gather
            or lm_head.tp_size == 1
            or (self.soft_cap is None and self.scale == 1.0)
        ):
            return LogitsProcessor.forward(
                self, lm_head, hidden_states, embedding_bias, skip_gather
            )
        logits = self._apply_head(lm_head, hidden_states, embedding_bias)
        logits = self._apply_soft_cap_and_scale(logits)
        logits = self._gather_logits(logits)
        return None if logits is None else logits[..., : self.org_vocab_size]
