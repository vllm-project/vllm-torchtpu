# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded query tiles for V4.1's native FP8 index cache.

The query-tile lifetime follows the SGLang-JAX DSA indexer/paged-score pattern:
retain a tile's scores until selection completes, then reuse the score buffer.
Unlike a BF16 cache scorer, this kernel reads four FP8 keys per native int32 row and
reads E8M0 scale words from the separate companion array. Selection reuses
vllm-torchtpu's existing SparseCore top-k, not a full-context XLA sort.
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import config

QUERY_TILE = 32
KEY_TILE = 128
HEAD_DIM = 128
# The bounded pipeline unrolls up to this many tiles; more run in a loop.
MAX_UNROLLED_TILES = 16


def pipeline_score_topk_unrolled(
    num_tiles: int,
    prepare: Callable[[jax.Array], Any],
    score: Callable[[Any, int | None], jax.Array],
    select: Callable[[jax.Array, jax.Array, int | None], Any],
    write: Callable[[int, Any, Any, Any], tuple[Any, Any]],
    output: Any,
) -> Any:
    """Leave independent score/selection tiles available to the scheduler.

    Use for tuned small-query batches; score lifetimes are not bounded here.
    ``write`` follows `pipeline_score_topk_bounded`'s contract.
    """
    if num_tiles < 1:
        raise ValueError("The score/top-k pipeline requires at least one tile")
    for tile in range(num_tiles):
        index = jnp.int32(tile)
        scores = score(prepare(index), None)
        selected = select(index, scores, None)
        output, _ = write(tile, selected, output, None)
    return output


def pipeline_score_topk_bounded(
    num_tiles: int,
    prepare: Callable[[jax.Array], Any],
    score: Callable[[Any, int | None], jax.Array],
    select: Callable[[jax.Array, jax.Array, int | None], Any],
    write: Callable[[int, Any, Any, Any], tuple[Any, Any]],
    output: Any,
    *,
    scheduling_groups: bool = True,
    gate_output: bool = True,
) -> Any:
    """Bound score lifetime and pair next-tile scoring with current selection.

    Callbacks accept an optional scheduling group after their data arguments.
    With ``scheduling_groups``, each pair gets a program-unique group so the
    compiler keeps TensorCore scoring between SparseCore selection start and
    completion. A group admits no other work into that window, so a writer
    that starts SparseCore work of its own runs without groups. With
    ``gate_output``, each write's output also waits for the next selection;
    a writer whose output carries work independent of it (attention one tile
    behind its gather) runs without, so that work need not wait.
    ``write(tile, selected, output, following)`` returns ``(output,
    following)``: ``following`` is the input of the pipeline's next step,
    the next tile's prepared scoring or the last tile's scores (None at the
    end), and work the writer gates it on is issued before that step. Up to
    ``MAX_UNROLLED_TILES``, ``tile`` is a Python int.
    """
    if num_tiles < 1:
        raise ValueError("The score/top-k pipeline requires at least one tile")
    if num_tiles > MAX_UNROLLED_TILES:
        return pipeline_score_topk(
            num_tiles,
            prepare,
            lambda prepared: score(prepared, None),
            lambda tile, scores: select(tile, scores, None),
            write,
            output,
        )

    group_base = (
        config.reserve_scheduling_group_ids(num_tiles - 1)
        if num_tiles > 1 and scheduling_groups
        else None
    )
    # Gate current scores and next preparation on the previous selection
    # so unrolling does not extend every score buffer's lifetime.
    with jax.named_scope("DSV41ScoreTopKBoundedPipeline"):
        current = score(prepare(jnp.int32(0)), None)
        previous = None
        for tile in range(num_tiles - 1):
            prepared = prepare(jnp.int32(tile + 1))
            if previous is not None:
                # Gating on the output too completes each write within the
                # next pair, so work the write starts on a tile's selection
                # overlaps the following scoring instead of trailing it.
                if gate_output:
                    previous, current, prepared, output = lax.optimization_barrier(
                        (previous, current, prepared, output)
                    )
                else:
                    previous, current, prepared = lax.optimization_barrier(
                        (previous, current, prepared)
                    )
                output, prepared = write(tile - 1, previous, output, prepared)
            group = None if group_base is None else group_base + tile
            following = score(prepared, group)
            previous = select(jnp.int32(tile), current, group)
            current = following
        if previous is not None:
            if gate_output:
                previous, current, output = lax.optimization_barrier(
                    (previous, current, output)
                )
            else:
                previous, current = lax.optimization_barrier((previous, current))
            output, current = write(num_tiles - 2, previous, output, current)
        last = num_tiles - 1
        selected = select(jnp.int32(last), current, None)
        return write(last, selected, output, None)[0]


def pipeline_score_topk(num_tiles, prepare, score, select, write, output):
    """Order TensorCore scoring and SparseCore selection across two query tiles.

    The barrier gates score-buffer reuse on selection completion, before
    output scatter. Physical execution overlap depends on the compiler.
    """
    first = score(prepare(jnp.int32(0)))

    def advance(tile, state):
        current, result = state
        following = score(prepare(tile + 1))
        return following, write(tile, select(tile, current), result, None)[0]

    def body(pair, state):
        tile = 2 * pair
        current, result = state
        following = score(prepare(tile + 1))
        selected = select(tile, current)
        prepared, selected = lax.optimization_barrier((prepare(tile + 2), selected))
        replacement = score(prepared)
        following_selected = select(tile + 1, following)
        result, _ = write(tile, selected, result, None)
        result, _ = write(tile + 1, following_selected, result, None)
        return replacement, result

    pairs = (num_tiles - 1) // 2
    with jax.named_scope("DSV41ScoreTopKPipeline"):
        state = lax.fori_loop(0, pairs, body, (first, output))
        if 2 * pairs < num_tiles - 1:
            state = advance(jnp.int32(2 * pairs), state)
        last, result = state
        last_selected = select(jnp.int32(num_tiles - 1), last)
        return write(jnp.int32(num_tiles - 1), last_selected, result, None)[0]
