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
"""N-gram drafting on device (`ngram_gpu`), for async scheduling.

The host proposer (`ngram`) drafts from the committed token ids, which async
scheduling holds back a step. This one keeps each request's token history on
device and appends the step's sampled tokens there before matching, so it
drafts from the sampler's output with no host round trip. The matching rule is
vLLM's (`vllm/v1/spec_decode/ngram_proposer.py`): the longest suffix n-gram in
`[prompt_lookup_min, prompt_lookup_max]` with an earlier occurrence, its
earliest occurrence, and the up to K tokens that follow it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm_torchtpu.kernels.partial_kv_update import pallas_index_copy_
from vllm_torchtpu.layers.core.attention_metadata import (
    stage_h2d_upload,
    zeros_h2d_source,
)
from vllm_torchtpu.runner.tpu_runner_async_output import INVALID_TOKEN_ID
from vllm_torchtpu.utils import synchronize_tensors

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput

    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
    from vllm_torchtpu.spec_decode.utils import DraftChunkInputs

# Drafts past the matched continuation. Any in-vocab id is safe: greedy
# verification accepts a draft only where it equals the target's own token.
FILLER_TOKEN_ID = 0
# The match scans only the leading history columns of a width bucket (powers
# of 4 from this) that holds every row's sequence: the full 262152-token width
# cost 0.41 ms per step on a few hundred tokens.
MIN_MATCH_WIDTH = 1024


# The history is `[rows * blocks, 128]` slots, a row's tokens in `blocks`
# consecutive slots (a multiple of 8, the in-place write's sublane group). A
# step reads the slots of its match width and writes back only the two it
# appends into, in place: a functional scatter copies the whole 270 MB history
# (0.2 ms per step), and whole-row writes move 1 MB per row.
HIST_LANES = 128


def _write_row_impl(hist: torch.Tensor, packed: torch.Tensor) -> None:
    """History row `packed[0]` = `packed[1:]`, a request's tokens from the host
    (a new or moved request; a functional write)."""
    blocks = (packed.shape[0] - 1) // HIST_LANES
    slots = packed[0].long() * blocks + torch.arange(blocks, device=hist.device)
    hist.index_copy_(0, slots, packed[1:].view(blocks, HIST_LANES))


def propose_step(
    hist: torch.Tensor,
    meta: torch.Tensor,
    new_tokens: torch.Tensor,
    min_n: int,
    max_n: int,
    k: int,
    width: int,
    blocks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Appends one step's tokens to the history rows and drafts from them.

    Args:
        hist: [num_hist_rows * blocks, HIST_LANES] persistent token history,
            updated in place.
        meta: [2 * rows] int32: each chunk row's history row (padded rows point
            at the dump row), then its committed length.
        new_tokens: [rows, S] tokens sampled this step, INVALID_TOKEN_ID padded
            at the end of each row.
        min_n, max_n: n-gram sizes to match.
        k: drafts per row.
        width: history columns to match over (a multiple of HIST_LANES);
            base_len + S <= width.
        blocks: slots per history row.

    Returns:
        [rows, k] int64 drafts, FILLER_TOKEN_ID where nothing matched; each
        row's count of matched drafts, the rest filler ([rows] int32); and the
        next step's `meta` if every valid token commits: the same history
        rows, each length past this step's tokens.
    """
    rows, steps = new_tokens.shape
    device = hist.device
    hist_rows = meta[:rows].long()
    base_len = meta[rows:].long()
    new = new_tokens.to(hist.dtype)
    # [rows, S] columns the step's tokens land in; base_len + S <= width. The
    # INVALID padding lands at or past seq_len, where nothing reads it.
    cols = base_len[:, None] + torch.arange(steps, device=device)
    n_slots = width // HIST_LANES
    first = hist_rows * blocks
    window = first[:, None] + torch.arange(n_slots, device=device)
    tokens = hist[window].reshape(rows, width).scatter(1, cols, new)
    # The step's tokens span the slot of base_len and maybe the next one.
    touched = (base_len // HIST_LANES)[:, None] + torch.arange(2, device=device)
    touched = touched.clamp(max=n_slots - 1)
    vals = tokens.view(rows, n_slots, HIST_LANES).gather(
        1, touched[:, :, None].expand(-1, -1, HIST_LANES)
    )
    slots = (first[:, None] + touched).reshape(-1)
    vals = vals.reshape(-1, HIST_LANES)
    if hist.device.type == "tpu":
        pallas_index_copy_(hist, slots.to(torch.int32), vals)
    else:
        hist.index_copy_(0, slots, vals)
    seq_len = base_len + (new != INVALID_TOKEN_ID).sum(dim=1)

    positions = torch.arange(width, device=device)
    no_match = torch.full_like(seq_len, width)
    draft_start = torch.zeros_like(seq_len)
    found = torch.zeros_like(seq_len, dtype=torch.bool)
    # Ascending n, so a longer n-gram with a match overrides a shorter one.
    for n in range(min_n, max_n + 1):
        num_windows = width - n + 1
        suffix_cols = (seq_len[:, None] - n + torch.arange(n, device=device)).clamp(
            min=0
        )
        suffix = tokens.gather(1, suffix_cols)  # [rows, n]
        # Window p matches when tokens[p + j] == suffix[j] for every j; each j
        # compares one shifted [rows, num_windows] slice.
        match = tokens[:, :num_windows] == suffix[:, :1]
        for j in range(1, n):
            match &= tokens[:, j : j + num_windows] == suffix[:, j : j + 1]
        # A match must end before the suffix's last token, leaving at least
        # one token to draft; this also excludes the suffix itself.
        window_pos = positions[:num_windows]
        match &= window_pos[None, :] <= (seq_len - n - 1)[:, None]
        first = torch.where(match, window_pos[None, :], no_match[:, None]).amin(dim=1)
        has_match = first < width
        draft_start = torch.where(has_match, first + n, draft_start)
        found |= has_match

    draft_cols = draft_start[:, None] + torch.arange(k, device=device)  # [rows, k]
    drafts = tokens.gather(1, draft_cols.clamp(max=width - 1))
    keep = found[:, None] & (draft_cols < seq_len[:, None])
    next_meta = torch.cat([meta[:rows], seq_len.to(meta.dtype)])
    num_drafts = keep.sum(dim=1).to(torch.int32)
    return torch.where(keep, drafts, FILLER_TOKEN_ID).long(), num_drafts, next_meta


_tpu_compile = torch.compile(backend="tpu", fullgraph=True, dynamic=False)
_write_row_compiled = _tpu_compile(_write_row_impl)
_propose_compiled = _tpu_compile(propose_step)


class DeviceNgramProposer:
    """Exact n-gram drafter whose history and matching stay on device."""

    # One [rows, K] draft tensor per chunk, at the chunk's row bucket.
    bucketed_drafts = True
    # Token matching only: no draft network, no draft KV, no collectives.
    draft_model = None
    # Integer matching over the sampled tokens, which every TP rank shares, so
    # no rank-0 pin (an all-reduce per step) is needed.
    rank_identical_drafts = True

    def __init__(self, runner: TPUModelRunner, vllm_config: VllmConfig):
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        assert speculative_config.prompt_lookup_min is not None
        assert speculative_config.prompt_lookup_max is not None
        self.runner = runner
        self.min_n = speculative_config.prompt_lookup_min
        self.max_n = speculative_config.prompt_lookup_max
        self.k = speculative_config.num_speculative_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        # Set by the runner each step; only each chunk's batch span is read.
        self.draft_chunks: list[DraftChunkInputs] | None = None
        # [(max_num_reqs + 1) * blocks, HIST_LANES]; the last row absorbs the
        # padded rows' writes. A step appends at most 1 + K tokens past a
        # committed length of up to max_model_len.
        self._hist: torch.Tensor | None = None
        # The match width buckets, ascending, the last the full width.
        self._match_widths: list[int] = []
        self._blocks = 0
        # Per chunk `(start, num_reqs, rows)`, the `meta` the last step left
        # on device, while its rows stay current and all its tokens commit:
        # the next step reuses it instead of uploading the committed lengths.
        self._next_meta: dict[tuple[int, int, int], torch.Tensor] = {}
        # The request whose history each batch row of `_hist` holds.
        self._row_req_ids: list[str | None] = []

    @property
    def _dump_row(self) -> int:
        return self.runner.max_num_reqs

    def _is_compiled(self) -> bool:
        return self.runner.device.type == "tpu"

    def load_model(self, target_model: torch.nn.Module) -> None:
        """Allocates the device history before the KV cache is sized."""
        del target_model  # Drafts from tokens only.
        width = self.max_model_len + 1 + self.k
        self._blocks = -(-width // (8 * HIST_LANES)) * 8
        width = self._blocks * HIST_LANES
        self._hist = torch.zeros(
            ((self._dump_row + 1) * self._blocks, HIST_LANES),
            dtype=torch.int32,
            device=self.runner.device,
        )
        self._match_widths = []
        w = MIN_MATCH_WIDTH
        while w < width:
            self._match_widths.append(w)
            w *= 4
        self._match_widths.append(width)
        self._row_req_ids = [None] * self._dump_row

    def run_dp_dummy_draft(self, num_chunks: int) -> None:
        """No collectives to pair with busy DP ranks."""

    def _write_row(self, packed: torch.Tensor) -> None:
        assert self._hist is not None
        fn = _write_row_compiled if self._is_compiled() else _write_row_impl
        fn(self._hist, packed)

    def _step(
        self, meta: torch.Tensor, new_tokens: torch.Tensor, width: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self._hist is not None
        fn = _propose_compiled if self._is_compiled() else propose_step
        return fn(
            self._hist,
            meta,
            new_tokens,
            self.min_n,
            self.max_n,
            self.k,
            width,
            self._blocks,
        )

    def _stale_rows(self, scheduler_output: SchedulerOutput) -> list[int]:
        """The batch rows whose device history is stale.

        A row is current only when the previous step's propose saw the same
        request there and appended its tokens, and the request was not
        rescheduled from scratch (new under a reused id, or resumed). Every
        other row (new, moved or re-added request) is stale.
        """
        input_batch = self.runner.input_batch
        rescheduled = {r.req_id for r in scheduler_output.scheduled_new_reqs}
        rescheduled |= scheduler_output.scheduled_cached_reqs.resumed_req_ids
        return [
            row
            for row in range(input_batch.num_reqs)
            if self._row_req_ids[row] != input_batch.req_ids[row]
            or input_batch.req_ids[row] in rescheduled
        ]

    def _refresh_rows(self, scheduler_output: SchedulerOutput) -> bool:
        """Uploads the history of each stale batch row; whether any was."""
        assert self._hist is not None
        input_batch = self.runner.input_batch
        num_reqs = input_batch.num_reqs
        width = self._match_widths[-1]
        stale = self._stale_rows(scheduler_output)
        for row in stale:
            req_id = input_batch.req_ids[row]
            packed = zeros_h2d_source(self.runner, 1 + width)
            packed[0] = row
            packed[1 : 1 + self.max_model_len] = torch.from_numpy(
                input_batch.token_ids_cpu[row]
            )
            self._write_row(stage_h2d_upload(self.runner, packed))
            self._row_req_ids[row] = req_id
        # Rows past the batch miss this step's appends.
        self._row_req_ids[num_reqs:] = [None] * (len(self._row_req_ids) - num_reqs)
        return bool(stale)

    def fused_inputs(
        self,
        scheduler_output: SchedulerOutput,
        start: int,
        num_reqs: int,
        rows: int,
        steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int, tuple[int, int, int]] | None:
        """`(history, meta, match width, blocks, key)` for a caller that folds this
        chunk's propose (`propose_step`) into its own program ahead of the
        host commit, or None unless the last step left the chunk's lengths on
        device and its rows are current. The host lengths then lag one step,
        so the width bucket leaves room for two steps of tokens."""
        key = (start, num_reqs, rows)
        meta = self._next_meta.get(key)
        if meta is None or self._stale_rows(scheduler_output):
            return None
        assert self._hist is not None
        _, width = self._chunk_meta(start, num_reqs, rows, 2 * steps, meta)
        return self._hist, meta, width, self._blocks, key

    def commit_fused(self, key: tuple[int, int, int], next_meta: torch.Tensor) -> None:
        """Keep the lengths a folded `propose_step` left (see `fused_inputs`)."""
        self._next_meta = {key: next_meta}

    def _chunk_meta(
        self,
        start: int,
        num_reqs: int,
        rows: int,
        steps: int,
        carried: torch.Tensor | None,
    ) -> tuple[torch.Tensor, int]:
        """The chunk's `meta` on device (`carried`, else uploaded), and the
        match width bucket that holds its sequences after the step's `steps`
        appended tokens."""
        committed = self.runner.input_batch.num_tokens_no_spec[start : start + num_reqs]
        # The step's 1 + K appended tokens must fit the history row.
        assert committed.max() <= self.max_model_len, (
            f"committed lengths {committed} exceed max_model_len {self.max_model_len}"
        )
        need = int(committed.max()) + steps
        width = next(w for w in self._match_widths if w >= need)
        if carried is not None:
            return carried, width
        meta = zeros_h2d_source(self.runner, 2 * rows)
        meta[:rows] = self._dump_row
        meta[:num_reqs] = torch.arange(start, start + num_reqs, dtype=torch.int32)
        meta[rows : rows + num_reqs] = torch.from_numpy(committed)
        return stage_h2d_upload(self.runner, meta), width

    @torch.no_grad()
    def propose(
        self,
        sampled_token_ids: list[list[int]] | None,
        discard_sampled_tokens_req_indices: list[int],
        num_rejected_tokens_np: np.ndarray | None,
        scheduler_output: SchedulerOutput,
        return_device: bool = False,
        next_tokens_per_chunk: list[torch.Tensor] | None = None,
        device_seed: list[torch.Tensor] | None = None,
    ) -> list[torch.Tensor]:
        """Drafts K tokens per request from this step's device tokens.

        Runs after the host committed the previous step's tokens, so a batch
        row's `num_tokens_no_spec` is where this step's tokens start. Those are
        `next_tokens_per_chunk` (a verify step's `[rows, 1 + K]` output) or
        `device_seed` (one `[rows]` sample per chunk). A discarded partial
        prefill sample lands past the committed prompt, where the next sample
        overwrites it. While a chunk's rows stay current and its tokens all
        commit, the next step starts from the lengths this one leaves on
        device, which equal `num_tokens_no_spec` then. The other host-seed
        arguments are unused; they keep the signature
        `SpeculativeDecodingManager` calls every drafter with.

        Returns:
            One `[rows, K]` int64 tensor per chunk, at the chunk's row bucket.
        """
        del sampled_token_ids, num_rejected_tokens_np
        assert return_device, "ngram_gpu drafts only under async scheduling"
        chunks = self.draft_chunks
        assert chunks, "ngram_gpu propose needs this step's chunks"
        if next_tokens_per_chunk is not None:
            new_tokens = next_tokens_per_chunk
        else:
            assert device_seed is not None
            new_tokens = [seed.view(-1, 1) for seed in device_seed]
        carried = {} if self._refresh_rows(scheduler_output) else self._next_meta
        self._next_meta = {}
        discarded = set(discard_sampled_tokens_req_indices)
        drafts = []
        for chunk, chunk_tokens in zip(chunks, new_tokens, strict=True):
            start, num_reqs = chunk.start_index, chunk.num_reqs
            key = (start, num_reqs, chunk_tokens.shape[0])
            meta, width = self._chunk_meta(
                start, num_reqs, *chunk_tokens.shape, carried.get(key)
            )
            chunk_drafts, _, next_meta = self._step(meta, chunk_tokens, width)
            drafts.append(chunk_drafts)
            # A discarded sample does not commit: the next step uploads.
            if not discarded.intersection(range(start, start + num_reqs)):
                self._next_meta[key] = next_meta
        return drafts

    @torch.no_grad()
    def precompile(self) -> None:
        """Compiles the row upload and every (row bucket, S, dtype, match
        width) step.

        Plain sampling yields int64 tokens ([rows] seeds, and the [rows,
        1 + K] padding of a spec step's chunk without drafts); the rejection
        sampler yields int32 [rows, 1 + K]. Only the dump row is written.
        """
        assert self._hist is not None
        runner = self.runner
        verify_len = 1 + self.k
        shapes = [
            (1, torch.int64),
            (verify_len, torch.int32),
            (verify_len, torch.int64),
        ]
        with runner._precompile_timed("n-gram drafter"):
            packed = torch.zeros(1 + self._match_widths[-1], dtype=torch.int32)
            packed[0] = self._dump_row
            self._write_row(packed.to(runner.device))
            for rows in runner.num_reqs_paddings:
                meta = torch.zeros(2 * rows, dtype=torch.int32)
                meta[:rows] = self._dump_row
                meta_dev = meta.to(runner.device)
                for steps, dtype in shapes:
                    new_tokens = torch.full(
                        (rows, steps), INVALID_TOKEN_ID, dtype=dtype
                    ).to(runner.device)
                    for width in self._match_widths:
                        synchronize_tensors(
                            list(self._step(meta_dev, new_tokens, width))
                        )
