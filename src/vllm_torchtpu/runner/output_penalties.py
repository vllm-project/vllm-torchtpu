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
"""``presence_penalty`` and ``frequency_penalty`` for the TPU sampler.

Both depend on how often each token is in a request's output so far.
``OutputPenaltyTracker`` keeps those counts on the device, one row of an
``int32 [max_num_reqs + 1, vocab]`` table per penalized request
(``layers/adapter/sample/output_penalties.py``), and adds each sampled token
right after sampling, on the device. The host only uploads, per chunk, which
row each request reads and its two penalties. Rows are leased from a
``RequestSlotAllocator`` (``runner/request_slot_table.py``), as the
thinking-token budget does; row 0 is the null slot, never written and all
zero, so padding and unpenalized rows read it and are left unchanged.

A request gets a row when it first runs with one of the penalties, and its
row is seeded from ``InputBatch.req_output_token_ids`` then. It is seeded
again whenever the host's output list no longer matches what was counted
(a different list object, or another length): a preempted request that
resumes, or a streaming-input continuation. The row is released when the
request leaves ``InputBatch.presence_penalties_reqs`` /
``frequency_penalties_reqs``. Rows follow request ids, not batch rows, so a
condensed or reordered batch needs nothing.

The counts are applied after ``repetition_penalty``
(``runner/repetition_penalty.py``), in the order of vLLM's
``apply_penalties``, as the first ``chunk_processors.PlainChunkProcessor`` of
the chunk: vLLM's ``Sampler.apply_logits_processors`` applies penalties
before the thinking-token budget.

Rows whose sampled token ``sample_tokens`` discards (partial prefill) are not
penalized and not counted.

The HBM cost is ``(max_num_reqs + 1) * vocab * 4`` bytes (about 127 MiB at
128 requests and a 248k vocabulary), allocated with the tracker when it is
supported and the runner samples, so ``profile_run`` sees it. The graphs
update the table in place, so it stays one buffer.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm_torchtpu.layers.adapter.sample import output_penalties as penalty_op
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.runner.request_slot_table import RequestSlotAllocator

if TYPE_CHECKING:
    from vllm_torchtpu.runner.vllm_v1.worker.gpu_input_batch import InputBatch

logger = init_logger(__name__)

# Output tokens counted per seeding pass; a longer output takes several.
SEED_WIDTH = 512


class _Counted:
    """What a request's row holds: the counts of ``output_ids[:length]``."""

    __slots__ = ("length", "output_ids")

    def __init__(self, output_ids: Sequence[int], length: int) -> None:
        self.output_ids = output_ids
        self.length = length


class OutputPenaltyTracker:
    """Device-side output-token counts, by request id, of the requests that
    set ``presence_penalty`` or ``frequency_penalty``, and the host staging
    buffers for one logits chunk of at most ``max_num_reqs`` rows."""

    def __init__(
        self,
        vocab_size: int,
        max_num_reqs: int,
        device: torch.device,
        *,
        async_scheduling: bool,
        speculative_decoding: bool,
        samples: bool = True,
    ) -> None:
        """``samples`` is False on a runner that never samples (a pipeline
        stage before the last, a pooling model): it then holds no table."""
        # TODO: apply under async_scheduling. There the host's output list ends
        # with a placeholder for the token still on the device, so seeding has
        # to skip it; on_sampled already counts the real token on the device.
        # TODO: apply in speculative decoding. There each verify row needs its
        # request's counts plus its draft prefix.
        self._samples = samples
        self.supported = samples and not async_scheduling and not speculative_decoding
        self._device = device
        self._counts: torch.Tensor | None = None
        if self.supported:
            self._counts = torch.zeros(
                (max_num_reqs + 1, vocab_size), dtype=torch.int32
            ).to(device)
        self._slots = RequestSlotAllocator(max_num_reqs)
        self._counted: dict[str, _Counted] = {}
        self._params_buf = np.zeros((max_num_reqs, 2), dtype=np.float32)

    @staticmethod
    def _penalized(input_batch: InputBatch) -> set[str]:
        return (
            input_batch.presence_penalties_reqs | input_batch.frequency_penalties_reqs
        )

    def active(self, input_batch: InputBatch) -> bool:
        """Whether a row of ``input_batch`` gets the penalties in this mode."""
        return self.supported and bool(self._penalized(input_batch))

    def begin_step(self, input_batch: InputBatch) -> bool:
        """``active(input_batch)``, warning once when a penalized request runs
        in an unsupported mode, and releasing the rows of requests that are
        no longer penalized."""
        penalized = self._penalized(input_batch)
        if not penalized:
            self._prune(())
            return False
        if not self._samples:
            return False
        if not self.supported:
            logger.warning_once(
                "presence_penalty and frequency_penalty are not yet applied on "
                "TPU with async_scheduling or speculative decoding; requests "
                "run without them. Run with async_scheduling=False and no "
                "speculative_config to apply them."
            )
            return False
        self._prune(penalized)
        return True

    def plain_chunk(
        self,
        input_batch: InputBatch,
        start: int,
        end: int,
        num_rows: int,
        discarded_req_ids: Collection[str] = (),
    ) -> OutputPenaltiesChunk | None:
        """The ``chunk_processors.PlainChunkProcessor`` for the plain
        1-row/req logits chunk of batch rows ``[start, end)`` padded to
        ``num_rows``, or ``None`` when no row in it is penalized. Seeds the
        rows of requests whose counts are missing or stale."""
        req_id_to_index = input_batch.req_id_to_index
        indices = sorted(
            index
            for req_id in self._penalized(input_batch)
            if start <= (index := req_id_to_index[req_id]) < end
            and req_id not in discarded_req_ids
        )
        if not indices:
            return None
        assert self._counts is not None
        self._params_buf[:num_rows] = 0.0
        counted: list[_Counted] = []
        for batch_idx in indices:
            req_id = input_batch.req_ids[batch_idx]
            output_ids = input_batch.req_output_token_ids[batch_idx]
            assert req_id is not None and output_ids is not None
            self._lease(req_id, output_ids)
            row = batch_idx - start
            self._params_buf[row, 0] = input_batch.presence_penalties_cpu[batch_idx]
            self._params_buf[row, 1] = input_batch.frequency_penalties_cpu[batch_idx]
            counted.append(self._counted[req_id])
        slots = self._slots.build_chunk_slot_ids(
            input_batch.req_ids, start, end, frozenset(discarded_req_ids)
        )
        assert slots is not None
        slots = slots[:num_rows]
        device = self._device
        return OutputPenaltiesChunk(
            self,
            torch.from_numpy(slots).to(device),
            torch.from_numpy(self._params_buf[:num_rows].copy()).to(device),
            torch.from_numpy((slots > 0).astype(np.int32)).to(device),
            counted,
        )

    def warmup(self, logits: torch.Tensor) -> list[torch.Tensor]:
        """Run every graph serving runs for a chunk of ``logits.shape[0]``
        rows, with inputs that change nothing. The runner's precompile calls
        this."""
        assert self._counts is not None
        num_rows = logits.shape[0]
        device = self._device
        slots = torch.zeros((num_rows,), dtype=torch.int32).to(device)
        params = torch.zeros((num_rows, 2), dtype=torch.float32).to(device)
        weights = torch.zeros((num_rows,), dtype=torch.int32).to(device)
        token_ids = torch.zeros((num_rows, 1), dtype=torch.int64).to(device)
        out = penalty_op.apply_output_penalties(logits, self._counts, slots, params)
        penalty_op.add_sampled_tokens(self._counts, slots, token_ids, weights)
        # Seeding the null row with padding only leaves it all zero.
        slot = torch.zeros((1,), dtype=torch.int32).to(device)
        empty = torch.full((SEED_WIDTH,), -1, dtype=torch.int32).to(device)
        penalty_op.seed_slot(self._counts, slot, empty)
        penalty_op.add_to_slot_compiled(self._counts, slot, empty)
        return [out, self._counts]

    def _lease(self, req_id: str, output_ids: Sequence[int]) -> None:
        """Lease ``req_id`` a row, seeding it when its counts are missing or
        do not match ``output_ids``."""
        slot = self._slots.lease(req_id)
        counted = self._counted.get(req_id)
        if (
            counted is None
            or counted.output_ids is not output_ids
            or counted.length != len(output_ids)
        ):
            self._seed(slot, output_ids)
            self._counted[req_id] = _Counted(output_ids, len(output_ids))

    def _seed(self, slot: int, output_ids: Sequence[int]) -> None:
        assert self._counts is not None
        device = self._device
        slot_t = torch.tensor([slot], dtype=torch.int32).to(device)
        ids = np.asarray(output_ids, dtype=np.int32)
        for begin in range(0, max(len(ids), 1), SEED_WIDTH):
            chunk = np.full((SEED_WIDTH,), -1, dtype=np.int32)
            part = ids[begin : begin + SEED_WIDTH]
            chunk[: len(part)] = part
            chunk_t = torch.from_numpy(chunk).to(device)
            if begin == 0:
                penalty_op.seed_slot(self._counts, slot_t, chunk_t)
            else:
                penalty_op.add_to_slot_compiled(self._counts, slot_t, chunk_t)

    def _prune(self, req_ids: Collection[str]) -> None:
        """Release the rows of requests that are not in ``req_ids``."""
        gone = [r for r in self._slots.allocated_req_ids() if r not in req_ids]
        if not gone:
            return
        self._slots.release(gone)
        for req_id in gone:
            self._counted.pop(req_id, None)


class OutputPenaltiesChunk:
    """``chunk_processors.PlainChunkProcessor`` for one chunk with a row that
    sets ``presence_penalty`` or ``frequency_penalty``."""

    def __init__(
        self,
        tracker: OutputPenaltyTracker,
        slots: torch.Tensor,
        params: torch.Tensor,
        weights: torch.Tensor,
        counted: list[_Counted],
    ) -> None:
        self._tracker = tracker
        self._slots = slots
        self._params = params
        self._weights = weights
        self._counted = counted

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        counts = self._tracker._counts
        assert counts is not None
        return penalty_op.apply_output_penalties(
            logits, counts, self._slots, self._params
        )

    def override_top_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            "a step with presence_penalty or frequency_penalty takes the full "
            "logits (TPUModelRunner._greedy_top_tokens_this_step)"
        )

    def on_sampled(self, token_ids: torch.Tensor) -> None:
        tracker = self._tracker
        assert tracker._counts is not None
        penalty_op.add_sampled_tokens(
            tracker._counts, self._slots, token_ids, self._weights
        )
        for counted in self._counted:
            counted.length += 1
