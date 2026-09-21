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
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.vocab_parallel_embedding import \
    VocabParallelEmbedding

# Shard bounds the masking reads, all fields of
# ``VocabParallelEmbeddingShardIndices``. Each becomes a buffer of that name.
_SHARD_BOUND_NAMES = (
    "org_vocab_start_index",
    "org_vocab_end_index",
    "num_org_vocab_padding",
    "added_vocab_start_index",
    "added_vocab_end_index",
)


def _masked_input_and_mask(
    input_: torch.Tensor,
    org_vocab_start_index: torch.Tensor,
    org_vocab_end_index: torch.Tensor,
    num_org_vocab_padding: torch.Tensor,
    added_vocab_start_index: torch.Tensor,
    added_vocab_end_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Upstream's ``get_masked_input_and_mask`` with tensor bounds.

    The arithmetic is upstream's, term for term. Only the type of the bounds
    differs: 0-dim tensors instead of Python ints, so that tracing records
    tensor-tensor ops whose operands are graph inputs rather than literals.
    """
    device = input_.device
    start = org_vocab_start_index.to(device=device)
    end = org_vocab_end_index.to(device=device)
    num_padding = num_org_vocab_padding.to(device=device)
    added_start = added_vocab_start_index.to(device=device)
    added_end = added_vocab_end_index.to(device=device)

    org_vocab_mask = (input_ >= start) & (input_ < end)
    added_vocab_mask = (input_ >= added_start) & (input_ < added_end)
    added_offset = added_start - (end - start) - num_padding
    valid_offset = (start * org_vocab_mask) + (added_offset * added_vocab_mask)
    vocab_mask = org_vocab_mask | added_vocab_mask
    input_ = vocab_mask * (input_ - valid_offset)
    return input_, ~vocab_mask


@VocabParallelEmbedding.register_oot
class TpuVocabParallelEmbedding(VocabParallelEmbedding):
    """VocabParallelEmbedding whose traced graph is the same on every TP rank.

    Upstream's forward passes this rank's shard bounds to
    ``get_masked_input_and_mask`` as Python ints. Tracing bakes them into the
    graph as scalar constants, so every TP rank compiles a different program
    and TorchTPU derives a different CompilationCacheKey per rank. Here the
    five bounds are non-persistent int64 buffers (kept out of ``state_dict()``
    so checkpoints are unaffected) and the masking is tensor arithmetic: the
    rank-specific values reach the graph as inputs and the graph itself is
    rank-invariant.

    Only ``tp_size > 1`` is overridden; ``tp_size == 1`` has no bounds to bake
    in and delegates to ``super().forward()``. Upstream's other ``tp_size > 1``
    branches are intentionally not mirrored, matching the class patch this
    replaces: the fused kernel is CUDA-only, and the FP8 branch, which views
    the embedding output as int8 for the all-reduce, has never been validated
    on TPU.

    Registration is keyed by class name, so it replaces
    ``VocabParallelEmbedding`` itself and not its subclasses.
    ``ParallelLMHead`` gets no buffers, and needs none: its forward raises and
    its weights are used by the sampler instead. vLLM's model-specific
    subclasses (``BartScaledWordEmbedding``, ``PLEVocabParallelEmbedding`` and
    the Transformers backend's ``_VocabParallelEmbeddingBase``) run upstream's
    forward with int bounds; no model served by this plugin uses them.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Unguarded on purpose: register_buffer raises if upstream ever grows
        # an attribute with one of these names, instead of shadowing it.
        for name in _SHARD_BOUND_NAMES:
            self.register_buffer(
                name,
                torch.tensor(getattr(self.shard_indices, name),
                             dtype=torch.int64),
                persistent=False,
            )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if self.tp_size == 1:
            return super().forward(input_)

        masked_input, input_mask = _masked_input_and_mask(
            input_,
            self.org_vocab_start_index,
            self.org_vocab_end_index,
            self.num_org_vocab_padding,
            self.added_vocab_start_index,
            self.added_vocab_end_index,
        )
        output_parallel = self.quant_method.embedding(self,
                                                      masked_input.long())
        output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
        return tensor_model_parallel_all_reduce(output_parallel)
