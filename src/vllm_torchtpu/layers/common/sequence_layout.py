# SPDX-License-Identifier: Apache-2.0

import bisect
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

import torch


class SequenceLayoutKind(str, Enum):
    ALL = "all"
    PARTIAL = "partial"


DEFAULT_SEQUENCE_LAYOUT_PROTOCOL = "default"
PCP_STREAMING_SEQUENCE_LAYOUT_PROTOCOL = "pcp_streaming"


@dataclass(frozen=True)
class SequenceLayoutDescriptor:
    kind: SequenceLayoutKind = SequenceLayoutKind.ALL
    protocol: str = DEFAULT_SEQUENCE_LAYOUT_PROTOCOL
    version: int = 1

    @property
    def cache_key(self) -> tuple[str, str, int]:
        return (self.kind.value, self.protocol, int(self.version))


DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR = SequenceLayoutDescriptor()


@dataclass(frozen=True)
class SequenceLayoutPlan:
    descriptor: SequenceLayoutDescriptor
    token_slice: slice
    global_num_tokens: int
    global_padded_num_tokens: int
    local_num_tokens: int
    local_padded_num_tokens: int
    logits_indices_cpu: torch.Tensor | None = None
    logits_local_indices_cpu: torch.Tensor | None = None
    logits_owner_mask_cpu: torch.Tensor | None = None
    requires_hidden_state_gather: bool = False
    # Optional layout-owned remap from the runner's per-chunk request-major
    # token order to the layout's packed global token order. Default layouts
    # keep the same order and leave this as None.
    _request_major_to_packed_token_indices: Any | None = None

    @property
    def kind(self) -> SequenceLayoutKind:
        return self.descriptor.kind

    def local_index_for_request_major_token(
            self, request_major_index: int) -> int | None:
        """Map a runner chunk token index to this rank's local input index.

        The runner builds host-side token metadata in request-major order. A
        sequence layout may keep that order unchanged or repack tokens before
        the rank-local slice is sent to the device. This method is the layout
        boundary for code that needs to address the rank-local input tensor.

        Returns None when the logical token belongs to a different local slice.
        """
        request_major_index = int(request_major_index)
        packed_index = request_major_index
        token_index_remap = self._request_major_to_packed_token_indices
        if token_index_remap is not None:
            packed_index = int(token_index_remap[request_major_index])

        if self.token_slice.step not in (None, 1):
            raise RuntimeError("SequenceLayoutPlan only supports contiguous "
                               "token slices for local index mapping.")
        if self.token_slice.stop is None:
            raise RuntimeError("SequenceLayoutPlan requires a finite "
                               "token_slice stop for local index mapping.")
        local_start = int(self.token_slice.start or 0)
        local_end = int(self.token_slice.stop)
        if not local_start <= packed_index < local_end:
            return None
        return packed_index - local_start


class SequenceLayoutPlanner(Protocol):

    @property
    def requires_backend_preinit(self) -> bool:
        ...

    @property
    def backend_preinit_world_size(self) -> int:
        ...

    @property
    def uses_selected_logits_hidden_states(self) -> bool:
        ...

    def reserve_host_token_capacity(self, runner: Any,
                                    required_num_tokens: int) -> None:
        ...

    def prepare_real(
        self,
        *,
        runner: Any,
        scheduler_output: Any,
        start_index: int,
        num_reqs: int,
        num_scheduled_tokens_per_req: Any,
        total_num_scheduled_tokens: int,
        use_max_model_len: bool,
        target_num_reqs: int,
        padded_num_reqs: int,
    ) -> SequenceLayoutPlan:
        ...

    def prepare_dummy(
        self,
        *,
        num_tokens: int,
        num_reqs: int,
        kv_cache_initialized: bool,
    ) -> SequenceLayoutPlan:
        ...

    def finalize_hidden_states(
        self,
        hidden_states: torch.Tensor,
        plan: SequenceLayoutPlan | None,
    ) -> torch.Tensor:
        ...

    def maybe_select_logits_hidden_states(
        self,
        hidden_states: torch.Tensor,
        plan: SequenceLayoutPlan | None,
        logits_indices: torch.Tensor,
    ) -> torch.Tensor | None:
        ...


def _first_ge(paddings: list[int] | tuple[int, ...], value: int) -> int:
    index = bisect.bisect_left(paddings, value)
    if index >= len(paddings):
        raise ValueError(
            "Sequence local token length exceeds runner max compile bucket: "
            f"required_tokens={value} "
            f"max_compile_bucket={paddings[-1] if paddings else None}.")
    return int(paddings[index])


class AllSequenceLayoutPlanner:

    @property
    def requires_backend_preinit(self) -> bool:
        return False

    @property
    def backend_preinit_world_size(self) -> int:
        return 1

    @property
    def uses_selected_logits_hidden_states(self) -> bool:
        return False

    def reserve_host_token_capacity(self, runner: Any,
                                    required_num_tokens: int) -> None:
        del runner, required_num_tokens

    def prepare_real(
        self,
        *,
        runner: Any,
        scheduler_output: Any,
        start_index: int,
        num_reqs: int,
        num_scheduled_tokens_per_req: Any,
        total_num_scheduled_tokens: int,
        use_max_model_len: bool,
        target_num_reqs: int,
        padded_num_reqs: int,
    ) -> SequenceLayoutPlan:
        del (scheduler_output, start_index, num_reqs,
             num_scheduled_tokens_per_req, use_max_model_len, target_num_reqs,
             padded_num_reqs)
        padded_num_tokens = _first_ge(runner.num_tokens_paddings,
                                      int(total_num_scheduled_tokens))
        dp_target_bucket = runner._dp_target_bucket
        if dp_target_bucket is not None and dp_target_bucket > padded_num_tokens:
            padded_num_tokens = int(dp_target_bucket)

        return SequenceLayoutPlan(
            descriptor=DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR,
            token_slice=slice(0, padded_num_tokens),
            global_num_tokens=int(total_num_scheduled_tokens),
            global_padded_num_tokens=padded_num_tokens,
            local_num_tokens=int(total_num_scheduled_tokens),
            local_padded_num_tokens=padded_num_tokens,
        )

    def prepare_dummy(
        self,
        *,
        num_tokens: int,
        num_reqs: int,
        kv_cache_initialized: bool,
    ) -> SequenceLayoutPlan:
        del num_reqs, kv_cache_initialized
        num_tokens = int(num_tokens)
        return SequenceLayoutPlan(
            descriptor=DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR,
            token_slice=slice(0, num_tokens),
            global_num_tokens=num_tokens,
            global_padded_num_tokens=num_tokens,
            local_num_tokens=num_tokens,
            local_padded_num_tokens=num_tokens,
        )

    def finalize_hidden_states(
        self,
        hidden_states: torch.Tensor,
        plan: SequenceLayoutPlan | None,
    ) -> torch.Tensor:
        del plan
        return hidden_states

    def maybe_select_logits_hidden_states(
        self,
        hidden_states: torch.Tensor,
        plan: SequenceLayoutPlan | None,
        logits_indices: torch.Tensor,
    ) -> torch.Tensor | None:
        del hidden_states, plan, logits_indices
        return None


def create_sequence_layout_planner(vllm_config: Any) -> SequenceLayoutPlanner:
    from vllm_torchtpu.layers.common.pcp_sequence_layout import \
        PcpSequenceLayoutPlanner

    pcp_planner = PcpSequenceLayoutPlanner.from_vllm_config(vllm_config)
    if pcp_planner.enabled:
        return pcp_planner
    return AllSequenceLayoutPlanner()


def is_pcp_streaming_sequence_layout(
    *,
    kind: str | SequenceLayoutKind,
    protocol: str,
) -> bool:
    if isinstance(kind, SequenceLayoutKind):
        kind_value = kind.value
    else:
        kind_value = str(kind)
    return (kind_value == SequenceLayoutKind.PARTIAL.value
            and protocol == PCP_STREAMING_SEQUENCE_LAYOUT_PROTOCOL)


def is_pcp_streaming_attention_metadata(attn_metadata: Any) -> bool:
    return is_pcp_streaming_sequence_layout(
        kind=getattr(attn_metadata, "sequence_layout_kind",
                     SequenceLayoutKind.ALL.value),
        protocol=getattr(attn_metadata, "sequence_layout_protocol",
                         DEFAULT_SEQUENCE_LAYOUT_PROTOCOL),
    )
