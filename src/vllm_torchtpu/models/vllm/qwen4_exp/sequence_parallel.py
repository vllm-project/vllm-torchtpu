# SPDX-License-Identifier: Apache-2.0
"""Sequence-parallel prefill for the qwen4_exp backbone.

At prefill sizes every TP rank otherwise repeats the row-wise work between
two attention blocks on all tokens: both hyper-connection mixes and combines
over the [tokens, hc * hidden] streams, the PLE gate and the padding masks.
With QWEN4_EXP_PREFILL_SP=1 a step of at least QWEN4_EXP_PREFILL_SP_MIN_TOKENS
tokens keeps the streams sharded by token across the TP ranks instead:

- the attention block input is all-gathered right before the TP projections,
- the attention output is reduce-scattered instead of all-reduced, and
- the MoE runs on the gathered block (its EP kernel takes replicated rows)
  and each rank keeps its own rows of the reduced output.

Rank r owns rows [r * T / tp, (r + 1) * T / tp), the order the reduce-scatter
produces and the all-gather consumes. Slicing goes through vLLM's
``sequence_parallel_chunk`` custom op so every rank traces the same graph.

The sequence-parallel backbone is a compiled graph of its own
(``Qwen4ExpBackboneSP``): ``Qwen4ExpForCausalLM.forward`` routes each step on
its padded bucket size, outside both graphs, and each graph compiles only its
own buckets (compilation/bucket_filter). Neither branches on the token count.
Only the backbone takes this path; the MTP draft keeps the replicated one.

With VLLM_TPU_SP_MICROBATCH_OVERLAP the step is split into two static token
halves A = [0, T/2) and B = [T/2, T), and rank r owns rows
[r * T / 2tp, (r + 1) * T / 2tp) of each half (A's rows, then B's). The
all-gather of each half of the shard then yields that token half in order,
and the reduce-scatter of a half's partial output yields this rank's rows of
it, so a layer can run its attention side half by half
(``Qwen4ExpDecoderLayer``) while every row-wise op between attention blocks
stays as it is. ``shard``, ``gather`` and ``reduce`` follow this layout.
"""

import math
import os
from collections.abc import Iterable

import torch
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    model_parallel_is_initialized,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)
from vllm.model_executor.models.utils import sequence_parallel_chunk

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Each rank's shard stays a multiple of a sublane tile.
ROW_ALIGNMENT = 8
# Decode buckets (the megakernel's, <= 64 rows) must never take this path.
MIN_TOKENS_FLOOR = 1024


def enabled(tp_size: int) -> bool:
    """Whether sequence-parallel prefill is on for this TP size."""
    return bool(envs.QWEN4_EXP_PREFILL_SP) and tp_size > 1


def shard_rows(hc_tiles: Iterable[int]) -> int:
    """Row multiple of a shard at which every HC prefill kernel takes its
    Pallas path.

    ``GatedResidual`` picks a kernel or its XLA fallback by whether the row
    count divides the kernel's tile (twice the tile for the full mix). Only
    shards that divide all of them keep the sequence-parallel graph one
    structure across its buckets, so it traces once.
    """
    return math.lcm(ROW_ALIGNMENT, *(tile for tile in hc_tiles if tile))


def pad_hc_enabled() -> bool:
    """QWEN4_EXP_PREFILL_SP_PAD_HC, honoured only with QWEN4_EXP_PREFILL_SP_MOE.

    Padded shards feeding the replicated MoE lowered real-AL to 3.09-3.11
    against 3.24 with GSM8K held, so the pair stays apart until understood.
    Localized so far: it takes a padded bucket (3072 at TP8) together with
    the replicated MoE on the gathered rows; tracing the graph at 2048
    instead of 1024 keeps it (s04, AL 3.109). Suspect a token-count choice
    the replicated MoE makes at trace time without a guard.
    """
    if not envs.QWEN4_EXP_PREFILL_SP_PAD_HC:
        return False
    if not envs.QWEN4_EXP_PREFILL_SP_MOE:
        logger.warning(
            "QWEN4_EXP_PREFILL_SP_PAD_HC needs QWEN4_EXP_PREFILL_SP_MOE=1; ignoring it."
        )
        return False
    return True


def padded_rows(rows, multiple: int):
    """``rows`` rounded up to ``multiple``. Shape arithmetic rather than a
    branch, so one graph serves every bucket."""
    return (rows + multiple - 1) // multiple * multiple


def pad_shard(x: torch.Tensor, multiple: int) -> torch.Tensor:
    """A shard with zero rows appended up to a multiple of ``multiple``.

    With QWEN4_EXP_PREFILL_SP_PAD_HC the sequence-parallel graph keeps the HC
    streams at this padded length, so every HC kernel takes its Pallas path
    for any shard. The padding rows stay zero (or False) and are cut off with
    :func:`unpad` before any collective or token-indexed op.
    """
    if multiple <= 1:
        return x
    rows = x.shape[0]
    pad = padded_rows(rows, multiple) - rows
    return torch.cat([x, x.new_zeros((pad, *x.shape[1:]))])


def unpad(x: torch.Tensor, rows) -> torch.Tensor:
    """The shard's real rows of a padded tensor."""
    return x.narrow(0, 0, rows)


def sequence_parallel_eligible(
    num_tokens: int, tp_size: int, rows: int = ROW_ALIGNMENT
) -> bool:
    """Whether a bucket of ``num_tokens`` tokens takes the sequence-parallel
    graph, each rank's shard a multiple of ``rows``. The router, both graphs'
    bucket filters and the warmup retraces use this one rule, so every bucket
    is compiled by exactly one graph."""
    if not enabled(tp_size):
        return False
    return (
        num_tokens >= min_tokens(tp_size, rows) and num_tokens % (tp_size * rows) == 0
    )


def replicated_token_shard_enabled() -> bool:
    """VLLM_TPU_QSA_PREFILL_REPLICATED_TOKEN_SHARD, unless it cannot run here.

    Off under VLLM_TPU_QSA_COMPRESSED_DCP (VLLM_TPU_QSA_DECODE_TOKEN_SHARD
    shards that indexer over the DCP pairs) and while LIBTPU_INIT_ARGS turns on
    the SparseCore all-gather offload (the c32 recipe): there the graph's
    selection all-gather halts a SparseCore ("Node 3 halted unexpectedly ...
    all_gather_into_tensor") in the first served replicated prefill bucket of
    the spec-decode warmup, and the run dies.
    """
    if not envs.VLLM_TPU_QSA_PREFILL_REPLICATED_TOKEN_SHARD:
        return False
    if envs.VLLM_TPU_QSA_COMPRESSED_DCP:
        return False
    from vllm_torchtpu.models.vllm.qwen4_exp.hyperconnection import (
        sparse_core_all_gather_offload,
    )

    if sparse_core_all_gather_offload():
        logger.warning_once(
            "VLLM_TPU_QSA_PREFILL_REPLICATED_TOKEN_SHARD is ignored while the "
            "SparseCore all-gather offload is on (its selection all-gather halts "
            "a SparseCore)."
        )
        return False
    return True


def replicated_token_shard_eligible(
    num_tokens: int, tp_size: int, max_num_seqs: int
) -> bool:
    """Whether a bucket outside the sequence-parallel graph takes the
    replicated backbone with the token-sharded QSA indexer
    (VLLM_TPU_QSA_PREFILL_REPLICATED_TOKEN_SHARD, ``Qwen4ExpBackboneRP``).

    Each rank's shard must keep the ragged prefill indexer (more rows than
    request slots), as the token-sharded op asserts. The router applies it
    after ``sequence_parallel_eligible``.
    """
    if not (tp_size > 1 and replicated_token_shard_enabled()):
        return False
    return (
        num_tokens >= int(envs.VLLM_TPU_QSA_PREFILL_REPLICATED_TOKEN_SHARD_MIN_TOKENS)
        and num_tokens % tp_size == 0
        and num_tokens // tp_size > max_num_seqs
    )


def min_tokens(tp_size: int, rows: int = ROW_ALIGNMENT) -> int:
    """Smallest bucket of the sequence-parallel graph."""
    return max(
        int(envs.QWEN4_EXP_PREFILL_SP_MIN_TOKENS), MIN_TOKENS_FLOOR, tp_size * rows
    )


def tp_world_size() -> int:
    if not model_parallel_is_initialized():
        return 1
    return get_tensor_model_parallel_world_size()


def check_divisible(num_tokens, groups: int = 1) -> None:
    """State that the token count divides by TP (times the micro-batch token
    groups) so shard shapes stay exact."""
    tp_size = tp_world_size()
    if tp_size > 1:
        torch._check(num_tokens % (tp_size * groups) == 0)


# Token groups of the micro-batched sequence-parallel graph.
MICROBATCH_GROUPS = 2

_SC_AG_FLAG = "--xla_tpu_enable_sparse_core_collective_offload_all_gather"
_SC_RS_FLAG = "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter"

# Set by Qwen4ExpForCausalLM when the deployment cannot take the split.
_microbatch_blocked: str | None = None
# The decision, made once (at model construction) so traced code only reads it.
_microbatch_decision: bool | None = None


def sparse_core_collective_offload() -> tuple[bool, bool]:
    """``(all_gather, reduce_scatter)``: whether LIBTPU_INIT_ARGS turns on the
    SparseCore offload of each (the last setting of a flag wins, as in libtpu)."""
    settings = {_SC_AG_FLAG: False, _SC_RS_FLAG: False}
    for arg in os.environ.get("LIBTPU_INIT_ARGS", "").split():
        name, _, value = arg.partition("=")
        if name in settings:
            settings[name] = value.lower() in ("true", "1")
    return settings[_SC_AG_FLAG], settings[_SC_RS_FLAG]


def block_microbatch(reason: str) -> None:
    """Turn VLLM_TPU_SP_MICROBATCH_OVERLAP off for this process (static: called
    at model construction, before any graph is traced)."""
    global _microbatch_blocked, _microbatch_decision
    _microbatch_blocked = reason
    _microbatch_decision = None


def reset_microbatch() -> None:
    """Forget the decision and any block (tests)."""
    global _microbatch_blocked, _microbatch_decision
    _microbatch_blocked = None
    _microbatch_decision = None


def microbatch_enabled() -> bool:
    """VLLM_TPU_SP_MICROBATCH_OVERLAP, unless it cannot help or cannot run here.

    Off under VLLM_TPU_SP_MICROBATCH_OVERLAP_FORCE_OFF, while LIBTPU_INIT_ARGS
    offloads neither the all-gather nor the reduce-scatter to the SparseCore
    (the TensorCore would run the collectives itself, so the halves only add
    launches), under VLLM_TPU_MAMBA_TAIL_FOLD (a prefill checkpoint offset is
    a whole number of GDN tiles from the request's first query token, which
    the second half's start is not), and when the model said so
    (``block_microbatch``). A static decision, made on the first call (the
    model makes it at construction): every trace sees the same one and logs
    nothing.
    """
    global _microbatch_decision
    if _microbatch_decision is None:
        _microbatch_decision = _decide_microbatch()
    return _microbatch_decision


def _decide_microbatch() -> bool:
    if not envs.VLLM_TPU_SP_MICROBATCH_OVERLAP:
        return False
    if envs.VLLM_TPU_SP_MICROBATCH_OVERLAP_FORCE_OFF:
        logger.warning_once(
            "VLLM_TPU_SP_MICROBATCH_OVERLAP is off: "
            "VLLM_TPU_SP_MICROBATCH_OVERLAP_FORCE_OFF is set."
        )
        return False
    if not any(sparse_core_collective_offload()):
        logger.warning_once(
            "VLLM_TPU_SP_MICROBATCH_OVERLAP is ignored while LIBTPU_INIT_ARGS "
            "offloads no collective to the SparseCore (nothing runs beside the "
            "TensorCore to overlap)."
        )
        return False
    if envs.VLLM_TPU_MAMBA_TAIL_FOLD:
        logger.warning_once(
            "VLLM_TPU_SP_MICROBATCH_OVERLAP is ignored under "
            "VLLM_TPU_MAMBA_TAIL_FOLD (prefill checkpoints would not land on "
            "GDN tile boundaries of the second half)."
        )
        return False
    if _microbatch_blocked is not None:
        logger.warning_once(
            "VLLM_TPU_SP_MICROBATCH_OVERLAP is ignored: %s", _microbatch_blocked
        )
        return False
    logger.info_once(
        "VLLM_TPU_SP_MICROBATCH_OVERLAP: sequence-parallel attention runs in %d "
        "token halves (SparseCore offload all-gather=%s reduce-scatter=%s).",
        MICROBATCH_GROUPS,
        *sparse_core_collective_offload(),
    )
    return True


def microbatch_groups() -> int:
    """Token groups of the sequence-parallel layout: 1, or
    ``MICROBATCH_GROUPS`` with VLLM_TPU_SP_MICROBATCH_OVERLAP."""
    return MICROBATCH_GROUPS if microbatch_enabled() else 1


def group_rows(x: torch.Tensor, groups: int) -> list[torch.Tensor]:
    """``x`` cut into ``groups`` equal row ranges (shape arithmetic only)."""
    rows = x.shape[0] // groups
    return [x.narrow(0, g * rows, rows) for g in range(groups)]


def all_gather_rows(x: torch.Tensor) -> torch.Tensor:
    """Every rank's ``x`` stacked by rank along dim 0."""
    return tensor_model_parallel_all_gather(x, 0)


def reduce_scatter_rows(x: torch.Tensor) -> torch.Tensor:
    """This rank's contiguous rows of the sum of every rank's ``x``."""
    return tensor_model_parallel_reduce_scatter(x, 0)


def chunk_rows(x: torch.Tensor) -> torch.Tensor:
    """This rank's contiguous rows of a replicated tensor."""
    return sequence_parallel_chunk(x)


def shard(x: torch.Tensor) -> torch.Tensor:
    """This rank's rows of a replicated [tokens, ...] tensor (contiguous, or
    its rows of each token half with VLLM_TPU_SP_MICROBATCH_OVERLAP)."""
    groups = microbatch_groups()
    if groups == 1:
        return chunk_rows(x)
    return torch.cat([chunk_rows(part) for part in group_rows(x, groups)])


def gather(x: torch.Tensor, num_tokens) -> torch.Tensor:
    """All ranks' rows of a sharded tensor, in token order."""
    groups = microbatch_groups()
    if groups == 1:
        return all_gather_rows(x).narrow(0, 0, num_tokens)
    parts = [all_gather_rows(part) for part in group_rows(x, groups)]
    return torch.cat(parts).narrow(0, 0, num_tokens)


def reduce(x: torch.Tensor, sequence_parallel: bool) -> torch.Tensor:
    """Sum TP partial outputs: this rank's rows, or all rows on the TP path."""
    if tp_world_size() <= 1:
        return x
    if sequence_parallel:
        groups = microbatch_groups()
        if groups == 1:
            return reduce_scatter_rows(x)
        return torch.cat([reduce_scatter_rows(p) for p in group_rows(x, groups)])
    return tensor_model_parallel_all_reduce(x)


class MicroBatchPlan:
    """One sequence-parallel step's token halves, built once per forward and
    shared by every layer (VLLM_TPU_SP_MICROBATCH_OVERLAP).

    Half ``g`` holds token rows ``[g * rows, (g + 1) * rows)`` of the step.
    Each attention core runs a half as if it were a step of its own, the way
    chunked prefill would have cut it: request starts re-based to the half
    and clipped to it, and every request's history advanced by its rows in
    earlier halves (``half_starts`` / ``rows_before``, the transformation of
    ``pooled_qsa.token_shard_step``). Half B runs after half A in program
    order and reads A's cache writes: QSA's keys and GDN's recurrent and conv
    state, which the A call wrote as a chunk boundary.
    """

    def __init__(self, num_tokens, groups: int = MICROBATCH_GROUPS):
        self.groups = groups
        self.rows = num_tokens // groups
        self._qsa: list = []
        self._gdn: list = []

    def positions(self, positions: torch.Tensor, g: int) -> torch.Tensor:
        return positions.narrow(0, g * self.rows, self.rows)

    def qsa(self, query_start_loc, context_lens, g: int):
        """``(starts, context)`` of half ``g`` for the QSA op."""
        for key, value in self._qsa:
            if key[0] is query_start_loc and key[1] is context_lens:
                return value[g]
        value = [
            (
                half_starts(query_start_loc, h * self.rows, self.rows),
                (
                    context_lens
                    + rows_before(query_start_loc, h * self.rows).to(context_lens.dtype)
                ),
            )
            for h in range(self.groups)
        ]
        self._qsa.append(((query_start_loc, context_lens), value))
        return value[g]

    def gdn(self, attn_metadata, query_start_loc, seq_lens, distribution, g: int):
        """Half ``g`` of a GDN group's metadata, a ``GdnHalf``. Memoized per
        metadata object, so the layers of one KV-cache group share the
        tensors (and VLLM_TPU_GDN_HOIST_METADATA's memo, keyed on the half)."""
        for key, value in self._gdn:
            if key is attn_metadata:
                return value[g]
        value = [
            gdn_half(query_start_loc, seq_lens, distribution, h, self.rows)
            for h in range(self.groups)
        ]
        self._gdn.append((attn_metadata, value))
        return value[g]


class GdnHalf:
    """A GDN core call's metadata for one token half (see ``gdn_half``)."""

    def __init__(self, query_start_loc, seq_lens, distribution, resume_from_carry):
        self.query_start_loc = query_start_loc
        self.seq_lens = seq_lens
        self.distribution = distribution
        # bool [num_seqs] or None: sequences whose state an earlier half of
        # this step wrote, so they read checkpoint 0.
        self.resume_from_carry = resume_from_carry


def half_starts(starts: torch.Tensor, base, rows) -> torch.Tensor:
    """Request starts of token rows ``[base, base + rows)``: re-based and
    clipped, so requests outside the half are empty and rows past
    ``starts[-1]`` stay outside every request."""
    return (starts - base).clamp(0, rows).to(starts.dtype)


def rows_before(starts: torch.Tensor, base) -> torch.Tensor:
    """Each request's rows before token row ``base`` of this step."""
    lengths = starts[1:] - starts[:-1]
    return torch.minimum((base - starts[:-1]).clamp_min(0), lengths)


def gdn_half(starts, seq_lens, distribution, g: int, rows):
    """GDN metadata of token half ``g`` (``GdnHalf``).

    - ``query_start_loc``: ``half_starts``.
    - ``seq_lens``: each request's length up to the end of the half (its rows
      in later halves taken off), so ``seq_lens - query_lens`` is the history
      before the half and a request continued from an earlier half reads its
      initial state.
    - ``distribution`` ``[decode_end, prefill_end, mixed_end]``: the leading
      decode/verify segment belongs to the first half (the model requires its
      rows to fit there), so later halves start PER_SEQ at sequence 0, where
      requests without rows in the half take no tile and move no state.
    - ``resume_from_carry``: requests with rows both before and in the half.
      The earlier half wrote their final state to checkpoint 0, so they read
      checkpoint 0 rather than the step's read offset (a prefix-cache resume
      or a verify step's accepted checkpoint applies to the first half only).
    """
    base = g * rows
    sub = half_starts(starts, base, rows)
    lengths = starts[1:] - starts[:-1]
    sub_lengths = sub[1:] - sub[:-1]
    before = rows_before(starts, base)
    after = lengths - before - sub_lengths
    half_seq_lens = seq_lens - after.to(seq_lens.dtype)
    if g == 0:
        return GdnHalf(sub, half_seq_lens, distribution, None)
    later = torch.cat([torch.zeros_like(distribution[:1]), distribution[1:]])
    resume = (before > 0) & (sub_lengths > 0)
    return GdnHalf(sub, half_seq_lens, later, resume)
