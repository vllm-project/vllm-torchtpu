# SPDX-License-Identifier: Apache-2.0
"""The DSA MTP draft (DeepSeek-V3.2, GLM-5.x) that one pipeline stage hosts whole."""

from collections.abc import Iterable
from typing import ClassVar, Literal

import torch
from vllm.model_executor.models import deepseek_mtp as upstream
from vllm.sequence import IntermediateTensors


class DeepSeekMTP(upstream.DeepSeekMTP):
    """Upstream's draft, admitted under pipeline parallelism.

    vLLM's config check requires every model under a pipeline, the draft
    included, to satisfy `SupportsPP`, which `supports_pp` tests by
    attribute; upstream's forward already takes `intermediate_tensors`. Off
    CUDA, vLLM serves the DSA draft (`DeepseekV32MTPModel`) with this class.
    The pipeline never splits the draft: the last stage builds and runs it
    whole, and its forward reads no pipeline rank, so it never receives
    intermediate tensors.
    """

    supports_pp: ClassVar[Literal[True]] = True

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        raise NotImplementedError(
            "the MTP draft runs whole on the last pipeline stage and never "
            "receives intermediate tensors"
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Upstream's loader, also taking the checkpoint's top-level embedding.

        Upstream loads only names inside an MTP layer and leaves the embedding
        to the proposer, which shares the target's. GLM-5.2 keeps no copy in
        its MTP layer, and on a pipeline's last stage the target's embedding
        is on the first stage, so the draft keeps its own
        (`maybe_share_embeddings`), which nothing loaded: every draft KV row
        the stage shipped came from an uninitialized table. The top-level name
        comes in as the first MTP layer's, which upstream renames back to
        ``model.embed_tokens``. A draft that borrows the target's embedding
        never sees the name (the sharing loader drops it).
        """
        first = f"model.layers.{self.model.mtp_start_layer_idx}.embed_tokens.weight"
        return super().load_weights(
            (first if name == "model.embed_tokens.weight" else name, weight)
            for name, weight in weights
        )
