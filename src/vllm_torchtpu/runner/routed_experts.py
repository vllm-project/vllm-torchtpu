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
"""Routed-experts capture (`--enable-return-routed-experts`) in TPU steps."""

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.distributed.aux_output_connector.connector import (
    AuxOutputConnectorMetadata,
    AuxRequestOutput,
)
from vllm.distributed.aux_output_connector.worker import (
    PendingAuxOutput,
    get_aux_output_connector,
)
from vllm.distributed.parallel_state import get_tp_group
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_torchtpu.distributed.aux_output_connector import (
    num_rejected_rows,
    use_capturer,
)
from vllm_torchtpu.layers.adapter.custom_ops.moe_runner import VllmMoERunner
from vllm_torchtpu.layers.adapter.routed_experts_capturer import (
    HostSnapshot,
    TPURoutedExpertsCapturer,
)

# The widest dtype vLLM's capturer stores expert ids in.
_RESERVED_ID_DTYPE = torch.uint16


def _row_size(vllm_config: VllmConfig) -> int:
    """Expert ids per token: one for each expert of each layer."""
    model_config = vllm_config.model_config
    return (
        model_config.get_total_num_hidden_layers()
        * model_config.get_num_experts_per_tok()
    )


def hbm_reserve_bytes(vllm_config: VllmConfig, token_paddings: list[int]) -> int:
    """HBM the capture takes after the memory profile; 0 without capture.

    The larger of two: vLLM's int32 table of `max_num_batched_tokens` rows,
    which lives until `use_capturer` swaps in our capturer, and our forward
    tables. Serving keeps a step's tables until its output is processed, for
    up to `max_concurrent_batches` steps. On top of those, `device_buffer`
    keeps the latest forward's table until the next forward returns, which
    in warmup holds two tables of the largest bucket at once. The tables are
    counted at the widest output dtype, and on every rank: only the first TP
    rank keeps them, but vLLM sizes every rank's KV cache by the smallest
    budget.
    """
    if not vllm_config.aux_output_config.enabled:
        return 0
    row_size = _row_size(vllm_config)
    scheduler_config = vllm_config.scheduler_config
    max_num_batched_tokens = scheduler_config.max_num_batched_tokens
    max_num_seqs = scheduler_config.max_num_seqs
    vllm_table_bytes = torch.int32.itemsize * max_num_batched_tokens * row_size
    # The scheduler stops a step at the smaller of the two token budgets, and
    # startup checks only that one against `compile_sizes`.
    step_tokens = max_num_batched_tokens
    max_num_scheduled_tokens = scheduler_config.max_num_scheduled_tokens
    if max_num_scheduled_tokens is not None:
        step_tokens = min(step_tokens, max_num_scheduled_tokens)
    # A step's tables, bounded two ways, since each forward runs at least one
    # request. First, each table holds its bucket's rows: the smallest bucket,
    # or less than R times its tokens, where R is the largest ratio between
    # neighboring buckets. Rounded up; with one bucket, every forward pads to
    # it, which `max_num_seqs * token_paddings[0]` already counts.
    padded_tokens = max(
        (
            -(-larger * step_tokens // smaller)
            for smaller, larger in zip(token_paddings, token_paddings[1:])
        ),
        default=0,
    )
    rows_by_ratio = padded_tokens + max_num_seqs * token_paddings[0]
    # Second, no forward has more tokens than the step, so none pads past the
    # bucket that holds `step_tokens`. This bound is the tighter one when the
    # buckets are far apart.
    step_bucket = min(size for size in token_paddings if size >= step_tokens)
    rows_by_forwards = max_num_seqs * step_bucket
    step_rows = min(rows_by_ratio, rows_by_forwards)
    largest_bucket = token_paddings[-1]
    rows = (
        max(vllm_config.max_concurrent_batches * step_rows, largest_bucket)
        + largest_bucket
    )
    tables_bytes = _RESERVED_ID_DTYPE.itemsize * row_size * rows
    return max(vllm_table_bytes, tables_bytes)


def _reject_uncapturable_layers(model: torch.nn.Module) -> None:
    """Rejects the MoE layers whose experts a fused kernel picks without
    `route()`, as capture would return zeros for them."""
    layer_ids = [
        module.layer_id
        for module in model.modules()
        if isinstance(module, VllmMoERunner) and not module.picks_experts_in_route
    ]
    if layer_ids:
        raise NotImplementedError(
            f"--enable-return-routed-experts cannot capture MoE layers "
            f"{layer_ids}: a fused kernel picks their experts. Unset "
            "USE_MOE_FUSED_EP_KERNEL, MOE_FUSED_EP_KERNEL_IMPL and "
            "USE_MOE_TP_DECODE_KERNEL."
        )


def _moe_layer_ids(model: torch.nn.Module) -> list[int]:
    """Returns the ids of the MoE layers, sorted, as the forward runs its
    layers in the order of their ids."""
    return sorted(
        module.layer_id for module in model.modules() if isinstance(module, MoERunner)
    )


@dataclass
class _StepInputBatch:
    """Stands in for vLLM's `InputBatch` in `prepare_output`: the fields it
    reads, covering every forward of the step."""

    req_ids: list[str]
    query_start_loc_np: np.ndarray
    num_computed_tokens_np: np.ndarray


@dataclass
class PendingRoutedExperts:
    """A step's routed experts on their way to the host, and vLLM's record of
    the step to hand them to."""

    pending_aux_output: PendingAuxOutput
    snapshot: HostSnapshot

    def process_output(
        self, sampled_token_ids: list[list[int]]
    ) -> dict[str, AuxRequestOutput]:
        """Hands the rows to vLLM's connector, given each request's sampled
        tokens in step order (none for a partial prefill)."""
        self.snapshot.wait()
        num_sampled = np.array([len(tokens) for tokens in sampled_token_ids])
        # TODO(haotianx): Count each request's draft tokens once speculative
        # decoding is supported. Until then no request has any.
        num_drafts = np.zeros_like(num_sampled)
        num_rejected = num_rejected_rows(num_drafts, num_sampled)
        # Called here rather than right after sampling, because the sampled
        # counts are only known now. The rows are already on the host, so its
        # copy to the host does nothing.
        self.pending_aux_output.enqueue_cpu_copy(num_sampled, num_rejected)
        return self.pending_aux_output.process_output()


class RoutedExpertsCapture:
    """Feeds vLLM's worker connector the rows of every forward in a step."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: torch.nn.Module,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        _reject_uncapturable_layers(model)
        moe_layer_ids = _moe_layer_ids(model)
        self._connector = get_aux_output_connector(model, vllm_config, kv_cache_config)
        # Every rank runs the same compiled forward, but vLLM's connector takes
        # the step's rows only from the first TP rank, so only that rank copies
        # them to the host.
        self._capturer = TPURoutedExpertsCapturer(
            vllm_config, moe_layer_ids, copies_to_host=get_tp_group().is_first_rank
        )
        # `hbm_reserve_bytes` sizes the tables from the config before the
        # capturer exists, so a row format vLLM changes must fail here.
        row_size = math.prod(self._capturer.shape_per_token)
        row_dtype = self._capturer.output_dtype
        assert (
            row_size == _row_size(vllm_config)
            and row_dtype.itemsize <= _RESERVED_ID_DTYPE.itemsize
        ), (
            f"vLLM's capturer stores {row_size} {row_dtype} ids per token, but "
            f"the routed-experts HBM reserve counts {_row_size(vllm_config)} "
            f"{_RESERVED_ID_DTYPE} ids"
        )
        use_capturer(self._connector, model, self._capturer)

    def begin_step(self, metadata: AuxOutputConnectorMetadata | None) -> None:
        self._connector.begin_step(metadata)

    def take_forward(self, num_padded_tokens: int, num_tokens: int) -> None:
        """Takes the expert ids of the forward that just ran, which come after
        the rows of the step's earlier forwards.

        Args:
            num_padded_tokens: the forward's token bucket.
            num_tokens: its real tokens.
        """
        self._capturer.take_forward(num_padded_tokens, num_tokens)

    def prepare_output(
        self,
        req_ids: list[str],
        num_scheduled_tokens: Mapping[str, int],
        token_starts: np.ndarray,
    ) -> PendingRoutedExperts | None:
        """Takes the step's snapshot; None on a rank that does not publish.

        Args:
            req_ids: the step's requests, in the order its forwards ran them.
            num_scheduled_tokens: each request's tokens this step.
            token_starts: [num_reqs] position of each request's first row.
        """
        num_tokens = [num_scheduled_tokens[req_id] for req_id in req_ids]
        # Start row of each request in the step's snapshot, which holds the
        # rows of every forward in the step.
        query_start_loc = np.cumsum([0, *num_tokens], dtype=np.int32)
        # vLLM reads `token_starts` only when the output is processed, and by
        # then the next step may have updated the runner's array.
        input_batch = _StepInputBatch(req_ids, query_start_loc, token_starts.copy())
        # pyrefly: ignore[bad-argument-type]
        pending_aux_output = self._connector.prepare_output(input_batch)
        if pending_aux_output is None:
            return None
        return PendingRoutedExperts(pending_aux_output, self._capturer.take_snapshot())

    def close(self) -> None:
        """Closes vLLM's connector, which stops its background writer thread."""
        self._connector.close()
