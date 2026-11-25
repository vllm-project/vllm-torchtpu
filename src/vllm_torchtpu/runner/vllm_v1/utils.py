# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

# Helpers the in-tree V1 copies use that vLLM v0.34.0 deleted, copied from vLLM
# v0.33.0 with their bodies unchanged; each names its source file.

from typing import TYPE_CHECKING, TypeVar

import numpy as np
import torch

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

    from vllm_torchtpu.runner.vllm_v1.worker.gpu_input_batch import InputBatch

_K = TypeVar("_K")
_V = TypeVar("_V")


# vllm/utils/collection_utils.py
def swap_dict_values(obj: dict[_K, _V], key1: _K, key2: _K) -> None:
    """Swap values between two keys."""
    v1 = obj.get(key1)
    v2 = obj.get(key2)
    if v1 is not None:
        obj[key2] = v1
    else:
        obj.pop(key2, None)
    if v2 is not None:
        obj[key1] = v2
    else:
        obj.pop(key1, None)


# vllm/v1/utils.py
def copy_slice(
    from_tensor: torch.Tensor, to_tensor: torch.Tensor, length: int
) -> torch.Tensor:
    """Copy the first length elements of a tensor into another tensor in a
    non-blocking manner.

    Used to copy pinned CPU tensor data to pre-allocated GPU tensors.

    Returns the sliced target tensor.
    """
    return to_tensor[:length].copy_(from_tensor[:length], non_blocking=True)


# vllm/v1/attention/backends/utils.py
def reorder_batch_to_split_decodes_and_prefills(
    input_batch: "InputBatch",
    scheduler_output: "SchedulerOutput",
    decode_threshold: int = 1,
) -> bool:
    """Reorders the batch to split into prefill and decode requests; places all
    requests with <= decode_threshold tokens at the front of the batch.

    The batch is reordered into 4 regions:
        decode:        (num_scheduled <= threshold AND is not prefilling)
        short_extend:  (num_scheduled <= threshold AND is chunked prefilling)
        long_extend:   (num_scheduled > threshold AND is chunked prefilling)
        prefill:       (num_computed == 0)   # First chunks

    Returns:
        True if the batch was modified, False otherwise.

    """
    num_reqs = len(input_batch.req_ids)
    num_scheduled_tokens = [
        scheduler_output.num_scheduled_tokens[id] for id in input_batch.req_ids
    ]
    num_scheduled_tokens_np = np.array(num_scheduled_tokens)
    num_computed_tokens_np = input_batch.num_computed_tokens_cpu[:num_reqs]
    num_prompt_tokens_np = input_batch.num_prompt_tokens[:num_reqs]

    has_context = num_computed_tokens_np > 0
    is_below_threshold = num_scheduled_tokens_np <= decode_threshold
    done_prefilling = num_computed_tokens_np >= num_prompt_tokens_np

    # Mutually exclusive categories (exactly one True per request):
    # 1. No context yet -> prefill
    # 2. Has context, above threshold -> long_extend
    # 3. Has context, below threshold, still prefilling -> short_extend
    # 4. Has context, below threshold, done prefilling -> decode
    is_pure_prefill = ~has_context
    is_long_extend = has_context & ~is_below_threshold
    is_short_extend = has_context & is_below_threshold & ~done_prefilling
    is_decode = has_context & is_below_threshold & done_prefilling

    # Desired order: decode → short_extend → long_extend → prefill
    req_regions = np.zeros(num_reqs, dtype=np.int32)  # 0 = decode by default
    req_regions[is_short_extend] = 1
    req_regions[is_long_extend] = 2
    req_regions[is_pure_prefill] = 3

    num_decodes = int(is_decode.sum())
    num_short_extends = int(is_short_extend.sum())
    num_long_extends = int(is_long_extend.sum())
    num_prefills = int(is_pure_prefill.sum())

    target_regions = np.repeat(
        [0, 1, 2, 3],
        [num_decodes, num_short_extends, num_long_extends, num_prefills],
    ).astype(np.int32)

    needs_swap = req_regions != target_regions

    if not needs_swap.any():
        return False

    # Extract indices that need swapping and sort by target region
    orig_indices = np.where(needs_swap)[0]
    sorted_order = np.argsort(req_regions[needs_swap], kind="stable")
    src_indices = orig_indices[sorted_order]

    src_dest_map = {int(src): int(dst) for src, dst in zip(src_indices, orig_indices)}

    for src in src_dest_map:
        dst = src_dest_map[src]
        while src != dst:
            input_batch.swap_states(src, dst)
            # Mark dst as done by updating its destination to itself
            next_dst = src_dest_map.get(dst, dst)
            src_dest_map[dst] = dst
            dst = next_dst

    return True
