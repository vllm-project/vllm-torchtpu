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


def _write_row_impl(hist: torch.Tensor, packed: torch.Tensor) -> None:
    """`hist[packed[0]] = packed[1:]`: one request's history from the host."""
    hist.index_copy_(0, packed[:1].long(), packed[1:].unsqueeze(0))


def _propose_impl(
    hist: torch.Tensor,
    meta: torch.Tensor,
    new_tokens: torch.Tensor,
    min_n: int,
    max_n: int,
    k: int,
) -> torch.Tensor:
    """Appends one step's tokens to the history rows and drafts from them.

    Args:
        hist: [num_hist_rows, width] persistent token history, updated in place.
        meta: [2 * rows] int32: each chunk row's history row (padded rows point
            at the dump row), then its committed length.
        new_tokens: [rows, S] tokens sampled this step, INVALID_TOKEN_ID padded
            at the end of each row.
        min_n, max_n: n-gram sizes to match.
        k: drafts per row.

    Returns:
        [rows, k] int64 drafts, FILLER_TOKEN_ID where nothing matched.
    """
    rows, steps = new_tokens.shape
    width = hist.shape[1]
    device = hist.device
    hist_rows = meta[:rows].long()
    base_len = meta[rows:].long()
    new = new_tokens.to(hist.dtype)
    # [rows, S] columns the step's tokens land in; base_len + S <= width. The
    # INVALID padding lands at or past seq_len, where nothing reads it.
    cols = base_len[:, None] + torch.arange(steps, device=device)
    tokens = hist[hist_rows].scatter(1, cols, new)  # [rows, width]
    hist.index_put_((hist_rows[:, None].expand(-1, steps), cols), new)
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
    return torch.where(keep, drafts, FILLER_TOKEN_ID).long()


_tpu_compile = torch.compile(backend="tpu", fullgraph=True, dynamic=False)
_write_row_compiled = _tpu_compile(_write_row_impl)
_propose_compiled = _tpu_compile(_propose_impl)


class DeviceNgramProposer:
    """Exact n-gram drafter whose history and matching stay on device."""

    # One [rows, K] draft tensor per chunk, at the chunk's row bucket.
    bucketed_drafts = True
    # Token matching only: no draft network, no draft KV, no collectives.
    draft_model = None

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
        # [max_num_reqs + 1, width]; the last row absorbs the padded rows'
        # writes. A step appends at most 1 + K tokens past a committed length
        # of up to max_model_len.
        self._hist: torch.Tensor | None = None
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
        self._hist = torch.zeros(
            (self._dump_row + 1, width), dtype=torch.int32, device=self.runner.device
        )
        self._row_req_ids = [None] * self._dump_row

    def run_dp_dummy_draft(self, num_chunks: int) -> None:
        """No collectives to pair with busy DP ranks."""

    def _write_row(self, packed: torch.Tensor) -> None:
        assert self._hist is not None
        fn = _write_row_compiled if self._is_compiled() else _write_row_impl
        fn(self._hist, packed)

    def _step(self, meta: torch.Tensor, new_tokens: torch.Tensor) -> torch.Tensor:
        assert self._hist is not None
        fn = _propose_compiled if self._is_compiled() else _propose_impl
        return fn(self._hist, meta, new_tokens, self.min_n, self.max_n, self.k)

    def _refresh_rows(self, scheduler_output: SchedulerOutput) -> None:
        """Uploads the history of each batch row whose device copy is stale.

        A row is current only when the previous step's propose saw the same
        request there and appended its tokens, and the request was not
        rescheduled from scratch (new under a reused id, or resumed). Every
        other row (new, moved or re-added request) is stale.
        """
        assert self._hist is not None
        input_batch = self.runner.input_batch
        num_reqs = input_batch.num_reqs
        rescheduled = {r.req_id for r in scheduler_output.scheduled_new_reqs}
        rescheduled |= scheduler_output.scheduled_cached_reqs.resumed_req_ids
        width = self._hist.shape[1]
        for row in range(num_reqs):
            req_id = input_batch.req_ids[row]
            if self._row_req_ids[row] == req_id and req_id not in rescheduled:
                continue
            packed = zeros_h2d_source(self.runner, 1 + width)
            packed[0] = row
            packed[1 : 1 + self.max_model_len] = torch.from_numpy(
                input_batch.token_ids_cpu[row]
            )
            self._write_row(stage_h2d_upload(self.runner, packed))
            self._row_req_ids[row] = req_id
        # Rows past the batch miss this step's appends.
        self._row_req_ids[num_reqs:] = [None] * (len(self._row_req_ids) - num_reqs)

    def _upload_meta(self, start: int, num_reqs: int, rows: int) -> torch.Tensor:
        meta = zeros_h2d_source(self.runner, 2 * rows)
        meta[:rows] = self._dump_row
        meta[:num_reqs] = torch.arange(start, start + num_reqs, dtype=torch.int32)
        committed = self.runner.input_batch.num_tokens_no_spec[start : start + num_reqs]
        # The step's 1 + K appended tokens must fit the history row.
        assert committed.max() <= self.max_model_len, (
            f"committed lengths {committed} exceed max_model_len {self.max_model_len}"
        )
        meta[rows : rows + num_reqs] = torch.from_numpy(committed)
        return stage_h2d_upload(self.runner, meta)

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
        overwrites it. The host-seed arguments are unused; they keep the
        signature `SpeculativeDecodingManager` calls every drafter with.

        Returns:
            One `[rows, K]` int64 tensor per chunk, at the chunk's row bucket.
        """
        del sampled_token_ids, discard_sampled_tokens_req_indices
        del num_rejected_tokens_np
        assert return_device, "ngram_gpu drafts only under async scheduling"
        chunks = self.draft_chunks
        assert chunks, "ngram_gpu propose needs this step's chunks"
        if next_tokens_per_chunk is not None:
            new_tokens = next_tokens_per_chunk
        else:
            assert device_seed is not None
            new_tokens = [seed.view(-1, 1) for seed in device_seed]
        self._refresh_rows(scheduler_output)
        drafts = []
        for chunk, chunk_tokens in zip(chunks, new_tokens, strict=True):
            meta = self._upload_meta(
                chunk.start_index, chunk.num_reqs, chunk_tokens.shape[0]
            )
            drafts.append(self._step(meta, chunk_tokens))
        return drafts

    @torch.no_grad()
    def precompile(self) -> None:
        """Compiles the row upload and every (row bucket, S, dtype) step.

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
            packed = torch.zeros(1 + self._hist.shape[1], dtype=torch.int32)
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
                    synchronize_tensors(self._step(meta_dev, new_tokens))
