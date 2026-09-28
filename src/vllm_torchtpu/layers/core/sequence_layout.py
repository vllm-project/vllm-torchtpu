# SPDX-License-Identifier: Apache-2.0

import bisect
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

import numpy as np
import torch

from vllm_torchtpu.distributed.pcp import all_reduce_sum as _pcp_all_reduce_sum


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


def is_pcp_streaming_sequence_layout_descriptor(
    descriptor: SequenceLayoutDescriptor,
) -> bool:
    return descriptor.cache_key == (
        SequenceLayoutKind.PARTIAL.value,
        PCP_STREAMING_SEQUENCE_LAYOUT_PROTOCOL,
        1,
    )


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
    # Optional packed-global to request-major token order. Padding rows use
    # -1. Real PCP plans carry both mapping directions.
    _packed_to_request_major_token_indices: np.ndarray | None = None
    # Optional layout-owned remap from the runner's per-chunk request-major
    # token order to the layout's packed global token order. Default layouts
    # keep the same order and leave this as None.
    _request_major_to_packed_token_indices: np.ndarray | None = None

    @property
    def kind(self) -> SequenceLayoutKind:
        return self.descriptor.kind

    def local_index_for_request_major_token(
        self, request_major_index: int
    ) -> int | None:
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
            raise RuntimeError(
                "SequenceLayoutPlan only supports contiguous "
                "token slices for local index mapping."
            )
        if self.token_slice.stop is None:
            raise RuntimeError(
                "SequenceLayoutPlan requires a finite "
                "token_slice stop for local index mapping."
            )
        local_start = int(self.token_slice.start or 0)
        local_end = int(self.token_slice.stop)
        if not local_start <= packed_index < local_end:
            return None
        return packed_index - local_start

    def localize_token_tensor_and_gather_indices(
        self,
        request_major_token_tensor: torch.Tensor,
        request_major_gather_indices: torch.Tensor,
        *,
        num_valid_gathers: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Localize request-major tokens and request-aligned gather values."""
        if self.kind is SequenceLayoutKind.ALL:
            return request_major_token_tensor, request_major_gather_indices

        num_gathers = request_major_gather_indices.shape[0]
        request_major_gather_indices = torch.nn.functional.pad(
            request_major_gather_indices[:num_valid_gathers],
            (0, num_gathers - num_valid_gathers),
            value=-1,
        )

        packed_to_request = self._packed_to_request_major_token_indices
        request_to_packed = self._request_major_to_packed_token_indices
        local_start = int(self.token_slice.start or 0)
        local_end = int(self.token_slice.stop)

        return _localize_token_tensor_and_gather_indices(
            request_major_token_tensor,
            request_major_gather_indices,
            packed_to_request[local_start:local_end],
            request_to_packed,
            local_start,
            local_end,
        )

    def aggregate_request_aligned_tensor(
        self,
        local_tensor: torch.Tensor,
        owner_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate request-aligned owner rows for this sequence layout."""
        descriptor = self.descriptor
        if descriptor == DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR:
            return local_tensor
        if is_pcp_streaming_sequence_layout_descriptor(descriptor):
            local_tensor = torch.where(
                owner_mask.unsqueeze(-1), local_tensor, torch.zeros_like(local_tensor)
            )
            return _pcp_all_reduce_sum(local_tensor)
        raise RuntimeError(
            f"Unsupported draft sequence layout descriptor: {descriptor.cache_key}"
        )


def _localize_token_tensor_and_gather_indices(
    request_major_token_tensor: torch.Tensor,
    request_major_gather_indices: torch.Tensor,
    local_packed_to_request: np.ndarray,
    request_to_packed: np.ndarray,
    local_start: int,
    local_end: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure tensor transform shared by all partial sequence layout plans."""
    source_indices = torch.tensor(
        local_packed_to_request,
        dtype=torch.long,
        device=request_major_token_tensor.device,
    )
    source_valid = source_indices.ge(0)
    safe_source_indices = source_indices.clamp_min(0)
    local_tokens = torch.index_select(
        request_major_token_tensor, 0, safe_source_indices
    )
    source_mask_shape = (source_valid.shape[0],) + (1,) * (local_tokens.ndim - 1)
    local_tokens = torch.where(
        source_valid.reshape(source_mask_shape),
        local_tokens,
        torch.zeros_like(local_tokens),
    )

    request_to_packed_tensor = torch.tensor(
        request_to_packed,
        dtype=torch.long,
        device=request_major_gather_indices.device,
    )
    request_gathers = request_major_gather_indices.to(torch.long)
    gather_valid = request_gathers.ge(0)
    safe_request_gathers = request_gathers.clamp_min(0)
    packed_gathers = torch.index_select(
        request_to_packed_tensor, 0, safe_request_gathers.reshape(-1)
    ).reshape(request_gathers.shape)
    owner_mask = (
        gather_valid & packed_gathers.ge(local_start) & packed_gathers.lt(local_end)
    )
    local_gathers = torch.where(
        owner_mask, packed_gathers - local_start, torch.full_like(packed_gathers, -1)
    )
    return local_tokens, local_gathers.to(request_major_gather_indices.dtype)


class SequenceLayoutPlanner(Protocol):
    @property
    def requires_backend_preinit(self) -> bool: ...

    @property
    def backend_preinit_world_size(self) -> int: ...

    @property
    def uses_selected_logits_hidden_states(self) -> bool: ...

    def reserve_host_token_capacity(
        self, runner: Any, required_num_tokens: int
    ) -> None: ...

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
    ) -> SequenceLayoutPlan: ...

    def prepare_dummy(
        self,
        *,
        num_tokens: int,
        num_reqs: int,
        kv_cache_initialized: bool,
    ) -> SequenceLayoutPlan: ...

    def finalize_hidden_states(
        self,
        hidden_states: torch.Tensor,
        plan: SequenceLayoutPlan | None,
    ) -> torch.Tensor: ...

    def maybe_select_logits_hidden_states(
        self,
        hidden_states: torch.Tensor,
        plan: SequenceLayoutPlan | None,
        logits_indices: torch.Tensor,
    ) -> torch.Tensor | None: ...


def _first_ge(paddings: list[int] | tuple[int, ...], value: int) -> int:
    index = bisect.bisect_left(paddings, value)
    if index >= len(paddings):
        raise ValueError(
            "Sequence local token length exceeds runner max compile bucket: "
            f"required_tokens={value} "
            f"max_compile_bucket={paddings[-1] if paddings else None}."
        )
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

    def reserve_host_token_capacity(
        self, runner: Any, required_num_tokens: int
    ) -> None:
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
        del (
            scheduler_output,
            start_index,
            num_reqs,
            num_scheduled_tokens_per_req,
            use_max_model_len,
            target_num_reqs,
            padded_num_reqs,
        )
        padded_num_tokens = _first_ge(
            runner.num_tokens_paddings, int(total_num_scheduled_tokens)
        )
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
    from vllm_torchtpu.layers.core.pcp_sequence_layout import PcpSequenceLayoutPlanner

    pcp_planner = PcpSequenceLayoutPlanner.from_vllm_config(vllm_config)
    if pcp_planner.enabled:
        return pcp_planner
    return AllSequenceLayoutPlanner()


def is_pcp_streaming_sequence_layout(
    *,
    kind: str | SequenceLayoutKind,
    protocol: str,
) -> bool:
    kind_value = kind.value if isinstance(kind, SequenceLayoutKind) else str(kind)
    return (
        kind_value == SequenceLayoutKind.PARTIAL.value
        and protocol == PCP_STREAMING_SEQUENCE_LAYOUT_PROTOCOL
    )


def is_pcp_streaming_attention_metadata(attn_metadata: Any) -> bool:
    return is_pcp_streaming_sequence_layout(
        kind=getattr(
            attn_metadata, "sequence_layout_kind", SequenceLayoutKind.ALL.value
        ),
        protocol=getattr(
            attn_metadata, "sequence_layout_protocol", DEFAULT_SEQUENCE_LAYOUT_PROTOCOL
        ),
    )
