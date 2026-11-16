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
"""Routed-experts capture for TPU steps that run several forwards."""

import bisect

import jax
import torch
from torch_tpu._internal import pallas
from vllm.config import VllmConfig
from vllm.forward_context import ForwardContext
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
)


def _stage_rows_jax(
    staging: jax.Array, rows: jax.Array, row_offset: jax.Array
) -> jax.Array:
    """`staging[row_offset : row_offset + len(rows)] = rows`, cast to the
    staging dtype; `row_offset` is a [1] int32 array."""
    return jax.lax.dynamic_update_slice(
        staging, rows.astype(staging.dtype), (row_offset[0], 0, 0)
    )


# Donating the staging table lets XLA write the rows in place.
_STAGE_ROWS_OP = pallas.jax_op(
    "pallas::routed_experts_stage_rows", _stage_rows_jax, donate_argnums=(0,)
)
_STAGE_ROWS_OP.register_fake(
    lambda staging, rows, row_offset: torch.empty_like(staging)
)


class TPURoutedExpertsCapturer(RoutedExpertsCapturer):
    """Collects a step's routed experts across all of its forwards.

    Every forward is padded to a token bucket and writes `device_buffer` from
    row 0. The compiled forward writes that table but never reads it, so XLA
    copies it once per forward rather than once per layer. With `with_staging`,
    each forward's rows then move into a step-wide staging table at the step's
    running row count, and the step's snapshot is read from that table. Shapes
    follow the buckets and the offset is device data, so each program compiles
    once per bucket.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        token_paddings: list[int],
        *,
        with_staging: bool = False,
    ) -> None:
        # One forward's rows: each forward writes a whole token bucket from row 0.
        super().__init__(
            max_num_batched_tokens=token_paddings[-1], vllm_config=vllm_config
        )
        self._token_paddings = token_paddings
        self._staging: torch.Tensor | None = None
        # Only the rank that publishes the step's rows needs them staged.
        if with_staging:
            # A step's rows plus the padding of its last forward.
            num_rows = (
                vllm_config.scheduler_config.max_num_batched_tokens + token_paddings[-1]
            )
            self._staging = torch.zeros(
                (num_rows, *self.shape_per_token),
                dtype=self.output_dtype,
                device=self.device_buffer.device,
            )

    def _restore_local_topk_ids(
        self, topk_ids: torch.Tensor, ctx: ForwardContext
    ) -> torch.Tensor:
        # vLLM picks the row layout by comparing the row count with the forward
        # context's token count. The backbone is traced once for every bucket,
        # so that comparison would be fixed at trace time. Layers bound to this
        # capturer hand over this rank's rows in token order.
        # TODO(haotianx): Move this to vLLM's `RoutedExpertsRowRestorer` hook on
        # the MoE runner once the vLLM pin includes
        # https://github.com/vllm-project/vllm/pull/64689.
        return topk_ids

    def stage_rows(self, num_padded_tokens: int, row_offset: int) -> None:
        """Moves the last forward's rows to `row_offset` of the step; a no-op
        without a staging table.

        Args:
            num_padded_tokens: the forward's token bucket.
            row_offset: the real token count of the step's earlier forwards.
        """
        if self._staging is None:
            return
        # dynamic_update_slice moves an out-of-range start back in range
        # instead of failing.
        assert row_offset + num_padded_tokens <= self._staging.shape[0], (
            f"staging {num_padded_tokens} rows at row {row_offset} overruns the "
            f"{self._staging.shape[0]}-row routed-experts staging table"
        )
        # Built on the host so the offset is program data, not a constant.
        offset = torch.tensor([row_offset], dtype=torch.int32).to(self._staging.device)
        rows = self.device_buffer[:num_padded_tokens]
        self._staging.copy_(_STAGE_ROWS_OP(self._staging, rows, offset))

    def snapshot_routing_data(self, num_tokens: int) -> torch.Tensor:
        """The step's rows padded to a token bucket; only the first
        `num_tokens` are valid.

        vLLM's capturer returns exactly `num_tokens` rows; here that slice
        would compile a program per row count. The extra rows are never read:
        the only caller is `AuxOutputWorkerConnector.prepare_output`, and the
        only reader of its result, `_commit_output`, slices each request's rows
        by its real token range. Revisit this if vLLM adds a caller or a reader
        that uses the row count or the rows past `num_tokens`.
        """
        assert self._staging is not None, (
            "routed-experts snapshots need with_staging=True"
        )
        bucket_index = bisect.bisect_left(self._token_paddings, num_tokens)
        num_rows = self._token_paddings[bucket_index]
        # Cloned because the next step's staging writes can run before the copy
        # of this one to the host is done.
        return self._staging[:num_rows].clone()
