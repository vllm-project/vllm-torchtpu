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
"""Contracts of the unchanged shared RPA wait and operand protocols."""

import jax
import jax.numpy as jnp

# bref_override.py BufferRef._wait_* bitcasts uint32 and reshapes (-1, 128).
# This is the shared wait ABI, not a substitute for arbitrary chip geometry.
RPA_WAIT_ROW_WORDS = 128
WORD_BYTES = jnp.dtype(jnp.uint32).itemsize
RPA_WAIT_ROW_BYTES = RPA_WAIT_ROW_WORDS * WORD_BYTES
KV_DMA_FIELDS = ("source_index", "destination_token", "token_count")
QUERY_DMA_FIELDS = ("source_row", "row_count")
SCALAR_OPERANDS = ("cu_q_lens", "compressed_query_offsets", "q_lens", "pages")
TENSOR_OPERANDS = ("query", "unused_new_kv", "cache", "gate")
OUTPUT_OPERANDS = ("query", "cache")


def output_aliases(scalars: tuple, metadata: object) -> dict[int, int]:
    """Locate named operands after Pallas scalar/tree flattening."""
    if len(scalars) != len(SCALAR_OPERANDS):
        raise ValueError("Decode scalar prefetch ABI changed")
    first_tensor = len(scalars) + len(jax.tree.leaves(metadata))
    return {
        first_tensor + TENSOR_OPERANDS.index(name): OUTPUT_OPERANDS.index(name)
        for name in OUTPUT_OPERANDS
    }


def scalar_prefetch_words(batch: int, page_indices: int) -> int:
    shapes = dict(
        cu_q_lens=batch + 1,
        compressed_query_offsets=batch,
        q_lens=batch,
        pages=page_indices,
    )
    assert tuple(shapes) == SCALAR_OPERANDS
    return sum(shapes.values())


def schedule_copy_size(prefix_words, capacity_words: int, alignment_words: int):
    """Zero-start complete-row prefix, capped before indexing either ref.

    Every schedule leaf has max_steps (a vector-row multiple) times words per
    step. These alignments preserve complete HBM layout rows. Pinned Pallas
    forwards the same slices to enqueue_dma/wait_dma2; 1024 is not its minimum.
    The smallest supported row-preserving policy is 128 words on TPU7x; this
    does not claim a silicon minimum for arbitrary DMA.
    """
    if alignment_words <= 0 or alignment_words % RPA_WAIT_ROW_WORDS:
        raise ValueError("Schedule DMA policy must preserve whole uint32 rows")
    if capacity_words % RPA_WAIT_ROW_WORDS:
        raise ValueError("Schedule leaf capacity must preserve whole rows")
    return jnp.minimum(
        ((prefix_words + alignment_words - 1) // alignment_words) * alignment_words,
        capacity_words,
    )
