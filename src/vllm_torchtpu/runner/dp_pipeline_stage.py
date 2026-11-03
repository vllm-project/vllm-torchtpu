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
"""This engine's stage of a pipelined prefill (TPU_DP_PIPELINE_PREFILL).

See ``distributed/dp_pipeline``. A step that schedules a chunk of a pipelined
prompt runs only this engine's layers of the plan: the first stage from the
token ids, later stages from the hidden states the previous stage handed off
(``StageCall``, read by the model's forward). A stage before the last hands its
output on, then lets the runner compute logits and sample as usual. On a
helper the request is a copy with ``max_tokens=1``: that token ends it after
its last chunk, and nothing reads it. The last stage owns the request.

Pipelined chunks are full chunks, padded to the one bucket the layer
segments are compiled at (``plan.pipelined_tokens``); the owner computes any
rest of the prompt as an ordinary step. A stage after the first copies the
next chunk's input to its device while it computes the current one. With
``plan.kv`` every TP rank of a helper also sends the owner the K/V pages its
layers wrote (``kv_plan``), and the owner scatters them into its own pages
ahead of its forwards; it scatters the last of them before its first
whole-model step for the request. Pages move between the caches and the
hand-off buffers by DMA (kv_scatter's kernel), never by an XLA gather.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.dp_pipeline.kv_plan import (
    KvGroup,
    KvTransfer,
    chunk_pages,
    chunk_transfer,
    max_ring_rows,
    window_transfer,
)
from vllm_torchtpu.distributed.dp_pipeline.plan import (
    PipelinePlan,
    plan_of,
    stage_bounds,
)
from vllm_torchtpu.distributed.dp_pipeline.transport import (
    HIDDEN,
    EdgeRing,
    kv_channel,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Received K/V hand-offs the owner holds on its device before scattering them.
_MAX_HELD_KV = 4


@dataclass(frozen=True)
class StageCall:
    """What the model forward runs for a pipelined chunk: layers
    ``[start, end)``, from ``hidden_states`` (None on the first stage, which
    embeds the step's token ids)."""

    start: int
    end: int
    hidden_states: torch.Tensor | None


@dataclass(frozen=True)
class StageStep:
    plan: PipelinePlan
    stage: int
    req_id: str
    # Position of the chunk's first token, and its token count.
    first_position: int
    num_tokens: int

    @property
    def is_first(self) -> bool:
        return self.stage == 0

    @property
    def is_last(self) -> bool:
        return self.stage == self.plan.num_stages - 1

    @property
    def is_final_chunk(self) -> bool:
        return self.first_position + self.num_tokens == self.plan.tokens

    @property
    def token(self) -> int:
        return self.plan.chunk_token(self.first_position)


@dataclass(frozen=True)
class KvAccess:
    """The runner's KV caches as the K/V hand-off reads and writes them."""

    groups: tuple[KvGroup, ...]
    block_size: int
    page_shape: tuple[int, ...]
    dtype: torch.dtype
    # The cache a layer reads now (scatters replace donated caches).
    cache: Callable[[str], torch.Tensor]
    # Rebinds every reference to a donated cache to the tensor replacing it.
    replace: Callable[[torch.Tensor, torch.Tensor], None]
    # (group, request id, logical blocks) -> this engine's block ids.
    block_ids: Callable[[int, str, tuple[int, ...]], list[int]]
    # (caches, block ids, pages) -> per cache, its pages at the ids in the
    # first rows of a fresh [pages, *page_shape] tensor.
    gather: Callable[[list[torch.Tensor], list[int], int], list[torch.Tensor]]
    # Scatters per-layer pages into caches in place; returns the new caches.
    scatter: Callable[
        [list[torch.Tensor], list[torch.Tensor], list[int], int], list[torch.Tensor]
    ]


class DpPipelineStage:
    def __init__(
        self,
        dp_rank: int,
        tp_rank: int,
        tp_size: int,
        max_rows: int,
        row_shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        materialize: Callable[..., None],
    ) -> None:
        """``materialize(tensors, wait=True)`` runs the device work producing
        ``tensors``; with ``wait=False`` it only enqueues it."""
        self.dp_rank, self.tp_rank, self.tp_size = dp_rank, tp_rank, tp_size
        self.max_rows, self.row_shape, self.dtype = max_rows, row_shape, dtype
        self.device = device
        self._materialize = materialize
        self.timeout_s = envs.TPU_DP_PIPELINE_TIMEOUT_S
        self._rings: dict[tuple[int, int, int, str], EdgeRing] = {}
        # Hand-offs leave from these threads, so the runner can dispatch the
        # next chunk while the device finishes this one and the data copies
        # out; hidden states never queue behind K/V.
        self._hidden_sender = ThreadPoolExecutor(1, thread_name_prefix="dp-pp-send")
        self._kv_sender = ThreadPoolExecutor(1, thread_name_prefix="dp-pp-send-kv")
        self._sent: list[Future] = []
        # The next chunk's input, copied in while this chunk computes.
        self._prefetcher = ThreadPoolExecutor(1, thread_name_prefix="dp-pp-prefetch")
        self._prefetched: dict[int, tuple[str, Future]] = {}
        # Owner: K/V hand-offs per request, in arrival order.
        self._receiver = ThreadPoolExecutor(1, thread_name_prefix="dp-pp-recv-kv")
        self._held = threading.Semaphore(_MAX_HELD_KV)
        self._receipts: dict[str, tuple[int, list[Future]]] = {}
        # Plans whose prompt this engine has run, with their request, so a
        # recompute after a preemption runs the whole model.
        self._done: dict[int, str] = {}
        self._kv: KvAccess | None = None
        self._kv_rows = 0
        self._chunk_pages = 0

    def bind_kv(self, access: KvAccess, num_layers: int, max_chunk_tokens: int) -> None:
        """Called once the KV caches exist."""
        if max_chunk_tokens != self.max_rows:
            raise ValueError(
                f"TPU_DP_PIPELINE_PREFILL needs the largest token bucket "
                f"({self.max_rows}) to be the full chunk, --max-num-batched-tokens "
                f"({max_chunk_tokens}): the layer segments are compiled for it"
            )
        self._kv = access
        self._chunk_pages = chunk_pages(max_chunk_tokens, access.block_size)
        helper_layers = [
            (bounds[s], bounds[s + 1])
            for num_stages in range(2, envs.TPU_DP_PIPELINE_STAGES + 1)
            for bounds in [stage_bounds(num_stages, num_layers)]
            for s in range(num_stages - 1)
        ]
        self._kv_rows = max_ring_rows(
            access.groups, helper_layers, self._chunk_pages, access.block_size
        )
        self._warm_kv(helper_layers, max_chunk_tokens)

    def _warm_kv(self, helper_layers: list[tuple[int, int]], chunk_tokens: int) -> None:
        """Compiles the gathers and scatters every helper range uses, at their
        fixed shapes, so no pipelined request waits for them: scatters of no
        rows, which still hand back the donated caches."""
        kv = self._kv
        assert kv is not None
        shapes: dict[tuple[int, int], tuple[str, ...]] = {}
        for layers in helper_layers:
            transfers = (
                chunk_transfer(
                    kv.groups,
                    layers,
                    1,
                    0,
                    chunk_tokens,
                    kv.block_size,
                    self._chunk_pages,
                ),
                window_transfer(kv.groups, layers, 1, chunk_tokens, kv.block_size),
            )
            for transfer in filter(None, transfers):
                for entry in transfer.entries:
                    key = (len(entry.layer_names), transfer.pages)
                    shapes.setdefault(key, entry.layer_names)
        for (_, pages), names in shapes.items():
            caches = [kv.cache(name) for name in names]
            self._materialize(kv.gather(caches, [], pages))
            sources = [
                torch.zeros((pages, *kv.page_shape), dtype=kv.dtype).to(self.device)
                for _ in caches
            ]
            new = kv.scatter(sources, caches, [], pages)
            for old, cache in zip(caches, new):
                kv.replace(old, cache)
            self._materialize(new)
        logger.info(
            "dp-pipeline: K/V hand-off programs compiled for %d layer counts",
            len(shapes),
        )

    def _ring(self, plan: PipelinePlan, src: int, dst: int, channel: str) -> EdgeRing:
        key = (plan.server_id, src, dst, channel)
        ring = self._rings.get(key)
        if ring is None:
            if channel == HIDDEN:
                shape = (self.max_rows, self.row_shape, self.dtype, self.tp_size)
            else:
                assert self._kv is not None
                shape = (self._kv_rows, self._kv.page_shape, self._kv.dtype, 1)
            ring = EdgeRing(*key, *shape, self._materialize)
            self._rings[key] = ring
        return ring

    # -- every step ----------------------------------------------------------------
    def step(
        self,
        req_ids: list[str],
        input_batch: Any,
        requests: dict,
        num_scheduled: dict[str, int],
    ) -> StageStep | None:
        """The pipelined chunk this step schedules, or None."""
        for future in self._sent:
            if future.done():
                # Surface a failed hand-off on the step after it.
                future.result()
        self._sent = [f for f in self._sent if not f.done()]
        for plan_id, req_id in list(self._done.items()):
            if req_id not in requests:
                del self._done[plan_id]
        for token, (req_id, _) in list(self._prefetched.items()):
            if req_id not in requests:
                del self._prefetched[token]
        found = []
        for req_id in req_ids:
            plan = plan_of(requests[req_id].sampling_params)
            if plan is None or plan.pipeline_id in self._done:
                continue
            index = input_batch.req_id_to_index[req_id]
            computed = int(input_batch.num_computed_tokens_cpu[index])
            if computed >= plan.tokens:
                continue  # the owner past the pipelined tokens: an ordinary step
            if computed + num_scheduled[req_id] > plan.tokens:
                raise RuntimeError(
                    f"chunk [{computed}, {computed + num_scheduled[req_id]}) of "
                    f"{req_id} crosses the pipeline's end at {plan.tokens}"
                )
            found.append(
                StageStep(
                    plan,
                    plan.stage_of(self.dp_rank),
                    req_id,
                    computed,
                    num_scheduled[req_id],
                )
            )
        if not found:
            return None
        if len(req_ids) != 1:
            raise RuntimeError(
                "a pipelined prefill chunk must be alone in its step, got requests "
                f"{list(req_ids)}"
            )
        step = found[0]
        if step.is_final_chunk:
            self._done[step.plan.pipeline_id] = step.req_id
        if step.is_last and step.plan.kv:
            self._expect_kv(step)
        return step

    def call(self, step: StageStep, num_rows: int) -> StageCall:
        """Receives this chunk's input (stages after the first)."""
        if num_rows != self.max_rows:
            # The layer segments exist at the full chunk's bucket only.
            raise RuntimeError(
                f"pipelined chunk @{step.first_position} pads to {num_rows} rows, "
                f"not the full chunk's {self.max_rows}"
            )
        start, end = step.plan.layers(step.stage)
        if step.is_first:
            return StageCall(start, end, None)
        ring = self._ring(
            step.plan, step.plan.ranks[step.stage - 1], self.dp_rank, HIDDEN
        )
        began = time.monotonic()
        prefetched = self._prefetched.pop(step.token, None)
        if prefetched is None:
            hidden = ring.take(
                step.token, num_rows, self.device, self.tp_rank, self.timeout_s
            )
        else:
            hidden = prefetched[1].result()
            if hidden.shape[0] != num_rows:
                raise RuntimeError(
                    f"chunk {step.token} arrived with {hidden.shape[0]} rows, this "
                    f"stage expects {num_rows}"
                )
        if not step.is_final_chunk:
            following = step.plan.chunk_token(step.first_position + step.num_tokens)
            self._prefetched[following] = (
                step.req_id,
                self._prefetcher.submit(
                    ring.take,
                    following,
                    None,
                    self.device,
                    self.tp_rank,
                    self.timeout_s,
                ),
            )
        if self.tp_rank == 0 and logger.isEnabledFor(logging.INFO):
            logger.info(
                "dp-pipeline stage %d/%d chunk @%d rows %d: input after %.1f ms",
                step.stage,
                step.plan.num_stages,
                step.first_position,
                num_rows,
                1e3 * (time.monotonic() - began),
            )
        assert isinstance(hidden, torch.Tensor)
        return StageCall(start, end, hidden)

    def send(self, step: StageStep, hidden_states: torch.Tensor) -> None:
        """Hands this chunk's output to the next stage (TP rank 0 sends; every
        rank of a TP group holds the same hidden states) and, with
        ``plan.kv``, this rank's K/V pages to the owner. The owner
        instead scatters every page after its last pipelined forward, so the
        step's sampled token waits for them rather than the next step: the
        wait counts in the time to first token, not between tokens."""
        if step.is_last:
            if step.is_final_chunk and step.req_id in self._receipts:
                self._apply(step.req_id, wait=True)
            return
        if self.tp_rank == 0:
            ring = self._ring(
                step.plan, self.dp_rank, step.plan.ranks[step.stage + 1], HIDDEN
            )
            what = (
                f"stage {step.stage}/{step.plan.num_stages} chunk "
                f"@{step.first_position} rows {hidden_states.shape[0]}"
            )
            self._sent.append(
                self._hidden_sender.submit(
                    self._put, ring, step.token, hidden_states, what
                )
            )
        if step.plan.kv:
            ring = self._ring(
                step.plan, self.dp_rank, step.plan.ranks[-1], kv_channel(self.tp_rank)
            )
            for transfer in self._kv_transfers(step, step.stage):
                pages = self._gather(transfer, step.req_id)
                # Enqueued now, behind this forward: the next forward donates
                # the caches.
                self._materialize(pages, wait=False)
                what = (
                    f"stage {step.stage}/{step.plan.num_stages} K/V "
                    f"{transfer.token & 0xFFFFF} pages {transfer.rows}"
                )
                self._sent.append(
                    self._kv_sender.submit(self._put, ring, transfer.token, pages, what)
                )

    def _put(
        self,
        ring: EdgeRing,
        token: int,
        data: torch.Tensor | list[torch.Tensor],
        what: str,
    ) -> None:
        began = time.monotonic()
        self._materialize(data)
        computed = time.monotonic()
        ring.put(token, data, self.timeout_s)
        if self.tp_rank == 0:
            logger.info(
                "dp-pipeline %s: output ready after %.1f ms, handed off in %.1f ms",
                what,
                1e3 * (computed - began),
                1e3 * (time.monotonic() - computed),
            )

    # -- the K/V hand-off ------------------------------------------------------------
    def _kv_transfers(self, step: StageStep, stage: int) -> list[KvTransfer]:
        """What stage ``stage`` sends the owner for this chunk."""
        kv = self._kv
        if kv is None:
            raise RuntimeError("pipelined K/V hand-off before the KV caches exist")
        layers = step.plan.layers(stage)
        transfers = [
            chunk_transfer(
                kv.groups,
                layers,
                step.token,
                step.first_position,
                step.num_tokens,
                kv.block_size,
                self._chunk_pages,
            )
        ]
        if step.is_final_chunk:
            transfers.append(
                window_transfer(
                    kv.groups,
                    layers,
                    step.plan.chunk_token(step.plan.tokens),
                    step.plan.tokens,
                    kv.block_size,
                )
            )
        return [t for t in transfers if t is not None]

    def _gather(self, transfer: KvTransfer, req_id: str) -> list[torch.Tensor]:
        """This engine's pages of ``transfer``, ``transfer.pages`` rows per
        layer; rows past an entry's blocks are never scattered."""
        assert self._kv is not None
        pages = []
        for entry in transfer.entries:
            ids = self._kv.block_ids(entry.group_id, req_id, entry.blocks)
            caches = [self._kv.cache(name) for name in entry.layer_names]
            pages += self._kv.gather(caches, ids, transfer.pages)
        return pages

    def _expect_kv(self, step: StageStep) -> None:
        """Owner: queue the receipt of what every helper sends for this
        chunk."""
        _, receipts = self._receipts.setdefault(step.req_id, (step.plan.tokens, []))
        for stage in range(step.plan.num_stages - 1):
            ring = self._ring(
                step.plan,
                step.plan.ranks[stage],
                self.dp_rank,
                kv_channel(self.tp_rank),
            )
            for transfer in self._kv_transfers(step, stage):
                receipts.append(self._receiver.submit(self._receive, ring, transfer))

    def _receive(
        self, ring: EdgeRing, transfer: KvTransfer
    ) -> tuple[KvTransfer, list[torch.Tensor]]:
        # Released once the pages are scattered or dropped.
        self._held.acquire()
        try:
            pages = ring.take(
                transfer.token,
                transfer.rows,
                self.device,
                0,
                self.timeout_s,
                split=transfer.pages,
            )
        except BaseException:
            self._held.release()
            raise
        assert isinstance(pages, list)
        return transfer, pages

    def before_forward(
        self, req_ids: list[str], input_batch: Any, requests: dict
    ) -> None:
        """Owner: scatter the K/V that arrived for requests in the batch; a
        request past its prompt waits for all of it."""
        running = set(self._done.values())
        for req_id in list(self._receipts):
            tokens, receipts = self._receipts[req_id]
            index = input_batch.req_id_to_index.get(req_id)
            computed = 0
            if index is not None:
                computed = int(input_batch.num_computed_tokens_cpu[index])
            if index is None or (req_id in running and computed < tokens):
                # Finished, or preempted (its blocks freed; a recompute runs
                # the whole model here): its pages have no home.
                self._drop(req_id)
                continue
            # Past the pipelined tokens the owner runs every layer.
            self._apply(req_id, wait=req_id in req_ids and computed >= tokens)

    def _apply(self, req_id: str, wait: bool) -> None:
        """Scatters the hand-offs of ``req_id`` that arrived, or with ``wait``
        all of them."""
        _, receipts = self._receipts[req_id]
        began, waited = time.monotonic(), len(receipts)
        while receipts and (wait or receipts[0].done()):
            transfer, pages = receipts.pop(0).result()
            try:
                self._scatter(transfer, pages, req_id)
            finally:
                self._held.release()
        if wait and self.tp_rank == 0:
            logger.info(
                "dp-pipeline owner: the last %d K/V hand-offs of %s scattered "
                "after %.1f ms",
                waited,
                req_id,
                1e3 * (time.monotonic() - began),
            )
        if not receipts:
            del self._receipts[req_id]

    def _drop(self, req_id: str) -> None:
        """A finished request's hand-offs are still taken off their rings,
        and kept by nobody."""

        def release(future: Future) -> None:
            if future.exception() is None:
                self._held.release()

        for future in self._receipts.pop(req_id)[1]:
            future.add_done_callback(release)

    def _scatter(
        self, transfer: KvTransfer, pages: list[torch.Tensor], req_id: str
    ) -> None:
        assert self._kv is not None
        layer = 0
        for entry in transfer.entries:
            ids = self._kv.block_ids(entry.group_id, req_id, entry.blocks)
            caches = [self._kv.cache(name) for name in entry.layer_names]
            if len({id(c) for c in caches}) != len(caches):
                raise RuntimeError(
                    f"layers {entry.layer_names} of KV cache group "
                    f"{entry.group_id} share a cache tensor"
                )
            sources = pages[layer : layer + len(caches)]
            layer += len(caches)
            new = self._kv.scatter(sources, caches, ids, transfer.pages)
            for old, cache in zip(caches, new):
                self._kv.replace(old, cache)
            self._materialize(new, wait=False)

    def close(self) -> None:
        for executor in (
            self._hidden_sender,
            self._kv_sender,
            self._prefetcher,
            self._receiver,
        ):
            executor.shutdown(wait=True)
        for ring in self._rings.values():
            ring.close()
        self._rings.clear()


def runner_kv_access(runner: Any) -> KvAccess:
    """``runner``'s KV caches (a TPUModelRunner after initialize_kv_cache)."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec
    from vllm.v1.worker.utils import extract_layer_index

    from vllm_torchtpu.distributed.kv_transfer import kv_scatter

    if not kv_scatter.scatter_available():
        raise RuntimeError("TPU_DP_PIPELINE_PREFILL: the KV scatter kernel is missing")
    context = runner.vllm_config.compilation_config.static_forward_context
    groups, block_sizes = [], set()
    for group_id, group in enumerate(runner.kv_cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        if isinstance(spec, SlidingWindowSpec):
            window = spec.sliding_window
        elif isinstance(spec, FullAttentionSpec):
            window = None
        else:
            raise NotImplementedError(
                f"TPU_DP_PIPELINE_PREFILL hands off attention K/V only, not {spec}"
            )
        if (
            getattr(runner.input_batch.block_table[group_id], "blocks_per_kv_block", 1)
            > 1
        ):
            raise NotImplementedError(
                "TPU_DP_PIPELINE_PREFILL needs a kernel page per KV cache block"
            )
        block_sizes.add(spec.block_size)
        layers = sorted((extract_layer_index(n), n) for n in group.layer_names)
        groups.append(KvGroup(group_id, tuple(layers), window))
    pages = {
        (tuple(context[name].kv_cache.shape[1:]), context[name].kv_cache.dtype)
        for g in groups
        for _, name in g.layers
    }
    if len(pages) != 1 or len(block_sizes) != 1:
        raise NotImplementedError(
            "TPU_DP_PIPELINE_PREFILL needs one page shape and block size across "
            f"the KV caches, got {pages} and {block_sizes}"
        )
    ((page_shape, dtype),) = pages
    device = runner.device

    def block_ids(group_id: int, req_id: str, blocks: tuple[int, ...]) -> list[int]:
        row = runner.input_batch.req_id_to_index[req_id]
        table = runner.input_batch.block_table[group_id].get_cpu_tensor()
        return [int(table[row, block]) for block in blocks]

    def replace(old: torch.Tensor, new: torch.Tensor) -> None:
        runner.kv_caches[:] = [new if c is old else c for c in runner.kv_caches]
        for layer in context.values():
            if getattr(layer, "kv_cache", None) is old:
                layer.kv_cache = new

    def copy_pages(
        sources: list[torch.Tensor],
        destinations: list[torch.Tensor],
        source_rows: list[int],
        destination_rows: list[int],
        pages: int,
    ) -> list[torch.Tensor]:
        """kv_scatter's DMA kernel: copies ``sources[l][source_rows[i]]`` to
        ``destinations[l][destination_rows[i]]`` in place (the destinations
        are donated and returned), at fixed shapes: rows past the given ones
        are never copied. No XLA gather reads the caches; on the shared KV
        pool one reserves memory for the whole pool."""
        count = len(source_rows)

        def rows(values: list[int]) -> torch.Tensor:
            padded = values + [0] * (pages - len(values))
            return torch.tensor(padded, dtype=torch.int32).to(device)

        offsets = (
            torch.tensor([count], dtype=torch.int32).to(device),
            rows(source_rows),
            rows(destination_rows),
        )
        return kv_scatter.multi_layer_scatter_into(
            sources, destinations, offsets[2], prebuilt_args=offsets
        )

    def gather(
        caches: list[torch.Tensor], ids: list[int], pages: int
    ) -> list[torch.Tensor]:
        pages_out = [
            torch.empty((pages, *page_shape), dtype=dtype, device=device)
            for _ in caches
        ]
        return copy_pages(caches, pages_out, ids, list(range(len(ids))), pages)

    def scatter(
        sources: list[torch.Tensor],
        caches: list[torch.Tensor],
        ids: list[int],
        pages: int,
    ) -> list[torch.Tensor]:
        return copy_pages(sources, caches, list(range(len(ids))), ids, pages)

    return KvAccess(
        groups=tuple(groups),
        block_size=block_sizes.pop(),
        page_shape=page_shape,
        dtype=dtype,
        cache=lambda name: context[name].kv_cache,
        replace=replace,
        block_ids=block_ids,
        gather=gather,
        scatter=scatter,
    )
