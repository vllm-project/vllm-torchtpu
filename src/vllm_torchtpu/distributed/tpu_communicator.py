# Copyright 2025 Google LLC
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
"""TPU device communicator for EP across data-parallel engines.

Dispatch gathers tokens across DP ranks that share the same TP rank. Combine
sums those matching-TP partials across DP and returns this DP rank's token
slice. vLLM's existing ``reduce_results`` TP all-reduce remains responsible for
the final TP-side reduction. Callers must pass equal token counts on each DP
rank.

Dispatch gathers all of its per-token tensors (hidden states plus router logits,
or plus top-k weights and ids) in one all-gather. At decode batch sizes these
collectives are latency-bound, so the number of collectives per MoE layer, not
their size, sets the cost.
"""

import functools
import math

import torch
from vllm.distributed.device_communicators.base_device_communicator import (
    All2AllManagerBase,
    DeviceCommunicatorBase,
)
from vllm.distributed.parallel_state import get_dp_group, get_ep_group

from vllm_torchtpu.compilation import shape_variants
from vllm_torchtpu.distributed.hier_reduce_scatter import moe_combine_reduce_scatter
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Pack only decode-sized steps. There the gathers are latency-bound and one
# collective instead of two or three is a net win; at prefill sizes the
# concatenate / split copies cost more than the saved collective (measured on
# v7x: 32K-context prefill took 1.4x as long with packing always on). The
# count compared is this rank's padded token bucket as the communicator sees
# it: the per-rank `compile_sizes` bucket, divided by TP under
# sequence-parallel MoE.
_PACK_MAX_TOKENS = 512

# Packing is enabled when the hidden states are bf16 (the measured case);
# other hidden dtypes (fp32, fp8) keep per-tensor gathers. The rows travel as
# int16, which has bf16's width, so viewing the hidden states costs nothing,
# and an integer copy cannot flush subnormal lanes: in a float carrier, the
# halves of fp32 / int32 values (e.g. a small expert id) can be bf16
# subnormals, which TPU flushes to zero.
_PACK_HIDDEN_DTYPE = torch.bfloat16
_PACK_CARRIER = torch.int16


@functools.cache
def _chip_is_tpu7x() -> bool:
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.tpu_info import get_chip_version

    return get_chip_version() == pltpu.ChipVersion.TPU_7X


@torch.compiler.assume_constant_result
def _chip_allows_packing() -> bool:
    """The cutoff and the A/B data are from tpu7x; other chips keep
    per-tensor gathers until they are measured."""
    return _chip_is_tpu7x()


@torch.compiler.assume_constant_result
def _log_unpackable(dtypes: str) -> None:
    """Trace-time log; takes no symbolic values."""
    logger.warning_once(
        "EP dispatch gathers tensors one by one: their dtypes (%s) do not fit a "
        "16-bit carrier.",
        dtypes,
    )


def _should_pack(tensors: list[torch.Tensor], num_tokens) -> bool:
    if tensors[0].dtype != _PACK_HIDDEN_DTYPE:
        return False
    if any(t.dtype.itemsize % _PACK_CARRIER.itemsize for t in tensors):
        _log_unpackable(", ".join(str(t.dtype) for t in tensors))
        return False
    if not _chip_allows_packing():
        return False
    # Branch on the token count only where each side of the branch gets its
    # own trace: eager runs, or traces inside `shape_variants.warmup()`. A
    # module compiled once for every bucket (profile run, MTP / Eagle drafters)
    # keeps per-tensor gathers, so its trace does not specialize on the count.
    if torch.compiler.is_compiling() and shape_variants.aot_cache_tag() is None:
        return False
    return num_tokens <= _PACK_MAX_TOKENS


def _packed_all_gather(group, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    """All-gather per-token tensors along dim 0 with a single collective.

    The rows of every tensor are bit-cast to int16 and concatenated along the
    last dim, gathered once, then split and cast back. int16 has the bf16
    hidden states' width, so they keep their native layout (a byte carrier
    forces TPU to repack every element), and integer copies keep every bit,
    so the result is bit-identical to gathering each tensor separately.
    Everything `_should_pack` rejects is gathered tensor by tensor.
    """
    num_tokens = tensors[0].shape[0]
    if any(t.shape[0] != num_tokens for t in tensors):
        raise ValueError("packed all-gather needs the same token count per tensor")
    if not _should_pack(tensors, num_tokens):
        return [group.all_gather(t, dim=0) for t in tensors]
    rows = [
        t.contiguous().reshape(num_tokens, math.prod(t.shape[1:])).view(_PACK_CARRIER)
        for t in tensors
    ]
    gathered = group.all_gather(torch.cat(rows, dim=1), dim=0)
    parts = gathered.split([r.shape[1] for r in rows], dim=1)
    return [
        part.contiguous().view(t.dtype).reshape(gathered.shape[0], *t.shape[1:])
        for t, part in zip(tensors, parts)
    ]


class TpuNullAll2AllManager(All2AllManagerBase):
    """Probe-only stub; TPU EP dispatch/combine never goes through it."""


class TpuDeviceCommunicator(DeviceCommunicatorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TPU performs EP dispatch/combine directly (see dispatch/combine
        # below), not through an all2all manager. The stub only satisfies
        # vLLM's DP>1 + MoE fault-tolerance probe
        # (support_fault_tolerance=False); it is never used for compute.
        if self.use_all2all:
            self.all2all_manager = TpuNullAll2AllManager(self.cpu_group)

    def _moe_group(self, is_sequence_parallel: bool):
        group = get_ep_group() if is_sequence_parallel else get_dp_group()
        return group if group.world_size > 1 else None

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors=None,
    ):
        if extra_tensors is not None:
            raise NotImplementedError(
                "extra_tensors is not supported by TpuDeviceCommunicator"
            )
        group = self._moe_group(is_sequence_parallel)
        if group is None:
            return hidden_states, router_logits
        hidden_states, router_logits = _packed_all_gather(
            group, [hidden_states, router_logits]
        )
        return hidden_states, router_logits

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors=None,
    ):
        if extra_tensors is not None:
            raise NotImplementedError(
                "extra_tensors is not supported by TpuDeviceCommunicator"
            )
        group = self._moe_group(is_sequence_parallel)
        if group is None:
            return hidden_states, topk_weights, topk_ids
        hidden_states, topk_weights, topk_ids = _packed_all_gather(
            group, [hidden_states, topk_weights, topk_ids]
        )
        return hidden_states, topk_weights, topk_ids

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        group = self._moe_group(is_sequence_parallel)
        if group is None:
            return hidden_states
        return moe_combine_reduce_scatter(group, hidden_states)
