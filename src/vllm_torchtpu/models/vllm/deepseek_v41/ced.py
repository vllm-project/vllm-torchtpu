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
"""Host planning and DP lockstep helpers for Context-Encoder-Decoder (CED) replay.

Note: Upstream ``decoder_replay_layers.py`` branches on
CUDA ``in_piecewise_cudagraph`` and uses ``out.index_copy_``, which lowers to an
``XlaScatter`` with duplicate padded sink indices and halts SparseCore (``E0200``).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

if TYPE_CHECKING:  # pragma: no cover
    from vllm.config import VllmConfig


def prefix_replay_enabled(vllm_config: VllmConfig, *, ced_enabled: bool = True) -> bool:
    """Return whether CED runs with prefix caching through SWA bounded replay.

    Args:
        vllm_config: Active vLLM runtime configuration (Used).
        ced_enabled: Whether Context-Encoder-Decoder splitting is enabled.

    Returns:
        ``True`` when prefix caching, SWA bounded replay, and ``PCP == 1`` are active.
    """
    cache_config = vllm_config.cache_config
    parallel_config = vllm_config.parallel_config
    replay_ok = (
        ced_enabled
        and cache_config.enable_prefix_caching
        and cache_config.swa_bounded_replay
    )
    return bool(replay_ok and parallel_config.prefill_context_parallel_size == 1)


@dataclass(frozen=True)
class DecoderReplayPlan:
    """Host-side token narrowing and bucket plan for the CED decoder stage.

    Attributes:
        indices: Kept encoder token row indices forwarded into the decoder stage.
        query_start_loc: Cumulative decoder query token offsets ``[num_reqs + 1]``.
        replay_starts: Per-request lower-bound KV position for decoder SWA replay.
        padded_num_tokens: Target compiled decoder bucket size (``0`` skips decode).
        is_identity: True when decoder keeps all scheduled rows at ``num_tokens``.
    """

    indices: tuple[int, ...]
    query_start_loc: tuple[int, ...]
    replay_starts: tuple[int, ...]
    padded_num_tokens: int
    is_identity: bool = False


def _as_int_sequence(values: Sequence[int] | torch.Tensor) -> Sequence[int]:
    """Convert a 1-D CPU tensor or Python sequence into a Python sequence of ints."""
    if isinstance(values, torch.Tensor):
        return values.tolist()
    return values


def plan_decoder_replay(
    metadata: Any,
    window: int,
    num_tokens: int,
    compile_sizes: Iterable[int],
) -> DecoderReplayPlan | None:
    """Compute the prompt-tail SWA replay plan across scheduled chunks and decode.

    Args:
        metadata: Per-layer ``AttentionMetadata`` carrying ``*_cpu`` scheduling
            tuples (Used; returns ``None`` during startup warmup when unset).
        window: Sliding-window size ``>= 1`` retained for decoder SWA replay.
        num_tokens: Padded encoder batch token count for the active step.
        compile_sizes: Candidate compiled token bucket sizes.

    Returns:
        Frozen ``DecoderReplayPlan`` (Created), or ``None`` when host scheduling
        tuples are absent.

    Raises:
        ValueError: If ``window < 1`` or ``metadata`` scheduling bounds are invalid.
    """
    query_start_loc = metadata.query_start_loc_cpu
    seq_lens = metadata.seq_lens_cpu
    prompt_lens = metadata.prompt_lens_cpu
    if query_start_loc is None or seq_lens is None or prompt_lens is None:
        return None

    query_bounds = _as_int_sequence(query_start_loc)
    seq_lengths = _as_int_sequence(seq_lens)
    prompt_lengths = _as_int_sequence(prompt_lens)
    replay_starts_cpu = metadata.replay_starts_cpu
    num_reqs = len(seq_lengths)
    replay_hits = (
        (0,) * num_reqs
        if replay_starts_cpu is None
        else _as_int_sequence(replay_starts_cpu)
    )
    invalid_lengths = (
        len(query_bounds),
        len(prompt_lengths),
        len(replay_hits),
    ) != (num_reqs + 1, num_reqs, num_reqs)
    if (
        window < 1
        or invalid_lengths
        or query_bounds[0] != 0
        or query_bounds[-1] > num_tokens
    ):
        raise ValueError("Invalid decoder replay scheduling metadata")

    replay_starts: list[int] = []
    kept_indices: list[int] = []
    out_query_bounds: list[int] = [0]
    for start, end, seq_len, prompt_len, hit_start in zip(
        query_bounds,
        query_bounds[1:],
        seq_lengths,
        prompt_lengths,
        replay_hits,
    ):
        if end < start or seq_len < end - start or prompt_len < 0:
            raise ValueError("Invalid scheduled query or prompt length")
        prompt_tail_start = prompt_len - window if prompt_len > window else 0
        replay_start = max(prompt_tail_start, hit_start)
        replay_starts.append(replay_start)
        chunk_start_pos = seq_len - (end - start)
        first_kept_offset = max(0, replay_start - chunk_start_pos)
        kept_indices.extend(range(start + first_kept_offset, end))
        out_query_bounds.append(len(kept_indices))

    kept_count = len(kept_indices)
    candidate_buckets = (
        int(bucket)
        for bucket in compile_sizes
        if kept_count <= int(bucket) < num_tokens
    )
    padded_num_tokens = (
        0 if kept_count == 0 else min(candidate_buckets, default=num_tokens)
    )
    is_identity = (
        kept_count > 0
        and kept_count == query_bounds[-1]
        and padded_num_tokens == num_tokens
    )
    return DecoderReplayPlan(
        indices=tuple(kept_indices),
        query_start_loc=tuple(out_query_bounds),
        replay_starts=tuple(replay_starts),
        padded_num_tokens=padded_num_tokens,
        is_identity=is_identity,
    )


def sync_decoder_replay_plan(
    plan: DecoderReplayPlan | None,
    metadata: Any,
    num_tokens: int,
    dp_cpu_group: Any = None,
    *,
    window: int | None = None,
    prefix_replay: bool = False,
    compile_sizes: tuple[int, ...] = (),
) -> DecoderReplayPlan | None:
    """Synchronize decoder stage execution and padded token bucket across DP ranks.

    Args:
        plan: Local ``DecoderReplayPlan`` or ``None`` if idle on this rank (Used).
        metadata: Per-layer ``AttentionMetadata`` for the current step (Used).
        num_tokens: Padded encoder batch token count for the active step.
        dp_cpu_group: Optional CPU process group for DP all-reduce synchronization.
        window: Sliding-window size for eliding host all-reduce on short batches.
        prefix_replay: Whether prefix caching with SWA bounded replay is enabled.
        compile_sizes: Configured compiled token bucket sizes.

    Returns:
        Synchronized ``DecoderReplayPlan`` (Created/Used), or ``None`` when all DP
        ranks skip decoder execution.
    """
    if dp_cpu_group is None:
        return plan
    min_compile_bucket = min((size for size in compile_sizes if size > 0), default=0)
    short_batch_limit = (
        0 if not window or window <= 0 else max(window, min_compile_bucket)
    )
    if not prefix_replay and 0 < num_tokens <= short_batch_limit:
        target_tokens = num_tokens
    else:
        local_vals = [
            1 if plan is not None else 0,
            plan.padded_num_tokens if plan is not None else 0,
        ]
        state = torch.tensor(local_vals, dtype=torch.int32, device="cpu")
        dist.all_reduce(state, op=dist.ReduceOp.MAX, group=dp_cpu_group)
        if state[0].item() == 0:
            return None
        target_tokens = int(state[1].item())

    if plan is None:
        num_reqs = int(metadata.seq_lens.shape[0])
        return DecoderReplayPlan(
            indices=(),
            query_start_loc=(0,) * (num_reqs + 1),
            replay_starts=(0,) * num_reqs,
            padded_num_tokens=target_tokens,
            is_identity=False,
        )
    if plan.padded_num_tokens == target_tokens:
        return plan
    query_bounds = metadata.query_start_loc_cpu
    kept_count = len(plan.indices)
    is_identity = (
        kept_count > 0
        and query_bounds is not None
        and kept_count == query_bounds[-1]
        and target_tokens == num_tokens
    )
    return replace(plan, padded_num_tokens=target_tokens, is_identity=is_identity)


def build_decoder_inverse_rows(
    indices: tuple[int, ...],
    padded: int,
    total: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Build the conflict-free inverse row gather map for ``scatter_decoder_outputs``.

    Unkept encoder rows map to index ``padded`` (which ``scatter_decoder_outputs``
    pads with a trailing zero row), avoiding duplicate scatter sink indices.

    Args:
        indices: Kept encoder row indices forwarded to the decoder stage.
        padded: Padded decoder token count (index of the trailing zero sink row).
        total: Full encoder batch token count to restore.
        device: Target PyTorch device for the returned index tensor.

    Returns:
        ``int32[total]`` gather index tensor mapping each encoder row to its kept
        decoder row or ``padded`` (Created).
    """
    kept_row_by_encoder_row = {
        encoder_row: kept_pos for kept_pos, encoder_row in enumerate(indices)
    }
    gather_indices = [kept_row_by_encoder_row.get(row, padded) for row in range(total)]
    return torch.tensor(gather_indices, dtype=torch.int32, device=device)


def scatter_decoder_outputs(
    tensor: torch.Tensor, inverse_rows: torch.Tensor
) -> torch.Tensor:
    """Restore narrowed decoder rows to full-stage positions via gather.

    Args:
        tensor: Narrowed decoder tensor ``[padded, ...]`` (Used).
        inverse_rows: ``int32[total]`` inverse row index tensor from
            :func:`build_decoder_inverse_rows` (Used).

    Returns:
        Full-batch tensor ``[total, ...]`` with unkept rows zero-filled (Created).
    """
    return F.pad(tensor, (0, 0, 0, 1)).index_select(0, inverse_rows)
