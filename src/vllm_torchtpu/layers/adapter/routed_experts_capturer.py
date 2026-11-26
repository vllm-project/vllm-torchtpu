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
"""Routed-experts capture that returns each TPU forward's expert ids as an
output of the compiled forward."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch_tpu._internal.stream.streams import TpuEvent
from vllm.config import VllmConfig
from vllm.forward_context import ForwardContext
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
)


@dataclass
class _ForwardCopy:
    """One forward's table (its `device_buffer`) on its way to the host."""

    # This forward's `device_buffer`. Held until the copy is done, because the
    # asynchronous copy does not keep its source alive.
    device_table: torch.Tensor
    # The host copy of this forward's `device_buffer` (`device_table`), valid
    # once the step's event has synchronized.
    host_table: torch.Tensor
    num_tokens: int


class HostSnapshot:
    """A step's routed experts on the host, made of its forwards' copies."""

    def __init__(
        self,
        forward_copies: list[_ForwardCopy],
        num_tokens: int,
        shape_per_token: tuple[int, int],
        dtype: torch.dtype,
    ) -> None:
        num_copied = sum(copy.num_tokens for copy in forward_copies)
        # vLLM's connector slices each request's rows by its scheduled tokens
        # and checks no total, so at least the total must match.
        assert num_copied == num_tokens, (
            f"routed-experts capture took {num_copied} rows for a step of "
            f"{num_tokens} tokens"
        )
        self._forward_copies = forward_copies
        # Recorded now, after every copy of this step has started, so it
        # covers them all and none of the next step's work.
        self._copies_done_event = TpuEvent()
        self._copies_done_event.record()
        if len(forward_copies) == 1:
            self.rows = forward_copies[0].host_table[:num_tokens]
        else:
            self.rows = torch.empty((num_tokens, *shape_per_token), dtype=dtype)

    def wait(self) -> None:
        """Waits until the copies are done, then fills `rows` with the
        real rows of each forward, in the order the forwards ran."""
        self._copies_done_event.synchronize()
        if len(self._forward_copies) > 1:
            parts = [
                copy.host_table[: copy.num_tokens] for copy in self._forward_copies
            ]
            torch.cat(parts, out=self.rows)
        # Releases the forwards' tables on the device.
        self._forward_copies = []


class TPURoutedExpertsCapturer(RoutedExpertsCapturer):
    """Returns each forward's expert ids as an output of the compiled forward.

    vLLM's capturer keeps one device buffer across forwards and snapshots it
    after each step. Here the first MoE layer of every forward points
    `device_buffer` at a table created inside the compiled forward, and
    vLLM's `capture()` writes it as usual. torch.compile makes that table an
    output of the forward and sets `device_buffer` to it after every call.
    A TPU step can run several forwards, each padded to its token bucket. Each
    forward's whole table is copied to the host, which needs no device program,
    and the host keeps each forward's real rows and joins them for the step. So
    the forward is the only program the capture adds to.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        moe_layer_ids: Sequence[int],
        *,
        copies_to_host: bool,
    ) -> None:
        """
        Args:
            vllm_config: the engine's config.
            moe_layer_ids: the ids of the MoE layers whose hooks call
                `capture()`, in the order the forward runs them.
            copies_to_host: whether this rank copies each forward's table to
                the host for vLLM's connector.
        """
        # vLLM accepts --enable-return-routed-experts for a model without MoE
        # layers, so the empty case is reachable from the command line.
        if not moe_layer_ids:
            raise ValueError(
                "--enable-return-routed-experts needs a model with MoE layers, "
                "but this model has none."
            )
        # Each forward creates its own table in `capture()`, so `device_buffer`
        # gets 0 rows here; the constructor is still called because it sets the
        # row shape and dtype.
        super().__init__(max_num_batched_tokens=0, vllm_config=vllm_config)
        self._first_moe_layer_id = moe_layer_ids[0]
        self._moe_layer_ids = tuple(moe_layer_ids)
        # Callers pass True only on the first TP rank, the one whose rows
        # vLLM's connector publishes.
        self._copies_to_host = copies_to_host
        # The MoE layers that wrote expert ids in the latest forward;
        # `check_forward` reads and clears it. A compiled forward sets it after
        # every call, to the ids collected while torch.compile traced the
        # forward.
        self._written_layer_ids: tuple[int, ...] = ()
        # Copies of this step's forwards, in the order they ran.
        self._step_copies: list[_ForwardCopy] = []
        self._latest_snapshot: HostSnapshot | None = None

    def capture(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        if layer_id == self._first_moe_layer_id:
            # A table for this forward only, created inside the compiled
            # forward and in the snapshot dtype; vLLM's write below casts the
            # ids to it.
            self.device_buffer = torch.zeros(
                (topk_ids.shape[0], *self.shape_per_token),
                dtype=self.output_dtype,
                device=topk_ids.device,
            )
            # Starts this forward's record, so that it holds no ids of an
            # earlier forward.
            self._written_layer_ids = ()
        super().capture(layer_id, topk_ids)
        self._written_layer_ids = (*self._written_layer_ids, layer_id)

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

    def check_forward(self, num_padded_tokens: int) -> torch.Tensor:
        """Returns the table of the forward that just ran, after checking that
        every MoE layer wrote its expert ids.

        It fails if the compiled forward was traced before the hooks were
        bound, was loaded from a compile cache built without them, or picks a
        layer's experts in a kernel that skips `route()`.

        Args:
            num_padded_tokens: the forward's token bucket.
        """
        written, table = self._written_layer_ids, self.device_buffer
        # Cleared, so a next forward whose program calls no hook fails here
        # instead of passing on this record.
        self._written_layer_ids = ()
        missing = sorted(set(self._moe_layer_ids) - set(written))
        if missing:
            raise RuntimeError(
                f"--enable-return-routed-experts: MoE layers {missing} did not "
                f"write expert ids in the {num_padded_tokens}-token forward. The "
                "compiled forward does not call their capture hook."
            )
        # No layer is missing by now, so this only catches a layer id that was
        # written twice or that belongs to no bound MoE layer.
        assert sorted(written) == sorted(self._moe_layer_ids), (
            f"the {num_padded_tokens}-token forward wrote the expert ids of MoE "
            f"layers {sorted(written)}, not {sorted(self._moe_layer_ids)}"
        )
        expected_shape = (num_padded_tokens, *self.shape_per_token)
        assert table.shape == expected_shape and table.dtype == self.output_dtype, (
            f"the {num_padded_tokens}-token forward returned a routed-experts "
            f"table of {tuple(table.shape)} {table.dtype}, not {expected_shape} "
            f"{self.output_dtype}"
        )
        return table

    def take_forward(self, num_padded_tokens: int, num_tokens: int) -> None:
        """Checks the forward that just ran and, on a rank that copies to the
        host, starts the copy of its table.

        Args:
            num_padded_tokens: the forward's token bucket.
            num_tokens: its real tokens, the rows the step keeps.
        """
        table = self.check_forward(num_padded_tokens)
        # A count above the bucket would pass the step's total check, but the
        # table has fewer rows, so the snapshot would come out short.
        assert 0 <= num_tokens <= num_padded_tokens, (
            f"the {num_padded_tokens}-token forward was taken with {num_tokens} "
            "real tokens"
        )
        if self._copies_to_host:
            # Copying a whole output buffer needs no device program, where a
            # slice of it would compile one for each row count.
            host_table = table.to("cpu", non_blocking=True)
            self._step_copies.append(_ForwardCopy(table, host_table, num_tokens))

    def snapshot_routing_data(self, num_tokens: int) -> torch.Tensor:
        """Returns the `num_tokens` rows of the forwards taken since the last
        snapshot, on the host, where vLLM's connector copies them anyway.

        They are valid once the `HostSnapshot` that `take_snapshot` returns
        has waited. vLLM's connector reads them only from
        `PendingAuxOutput.enqueue_cpu_copy` on, so it must be called after
        that wait.
        """
        self._latest_snapshot = HostSnapshot(
            self._step_copies, num_tokens, self.shape_per_token, self.output_dtype
        )
        self._step_copies = []
        return self._latest_snapshot.rows

    def take_snapshot(self) -> HostSnapshot:
        """Returns the snapshot whose rows `snapshot_routing_data` returned
        last."""
        snapshot = self._latest_snapshot
        assert snapshot is not None, "no routed-experts snapshot to take"
        self._latest_snapshot = None
        return snapshot
