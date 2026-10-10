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
"""Packed int32 prompt token ids for the engine-core -> worker broadcast.

``NewRequestData.prompt_token_ids`` is a Python list. Pickling it in the
engine core and unpickling it in each of the 8 TP workers, then copying it
element by element into ``InputBatch.token_ids_cpu``, costs ~7-8 ms per 94k
prompt on the worker critical path. ``PackedPromptIds`` is an int32 ndarray
whose pickle is a single inline ``bytes`` object; workers rebuild a plain
read-only int32 ndarray with ``np.frombuffer`` (no per-element work), which
``token_ids_cpu[row, :n] = ids`` copies with one memcpy.

Worker-side consumers on our path only need ``len()``, slicing/indexing and
array assignment (``CachedRequestState.num_prompt_tokens`` / ``get_token_id``,
``InputBatch.add_request``, M-RoPE ``len(input_tokens)``), which ndarrays
support.
"""

from array import array

import numpy as np


def _unpack_prompt_ids(buf: bytes) -> np.ndarray:
    # Read-only view of the (owned) bytes object: never aliases shared memory.
    return np.frombuffer(buf, dtype=np.int32)


class PackedPromptIds(np.ndarray):
    """int32 ndarray that pickles as inline bytes (see module docstring)."""

    def __reduce_ex__(self, protocol):
        return (_unpack_prompt_ids, (self.tobytes(),))

    def __reduce__(self):
        return (_unpack_prompt_ids, (self.tobytes(),))


def pack_prompt_ids(ids) -> PackedPromptIds | None:
    """Pack a list of token ids, or return None if it cannot be packed
    losslessly into int32."""
    try:
        buf = array("i", ids)
    except (OverflowError, TypeError):
        return None
    if buf.itemsize != 4:
        return None
    return np.frombuffer(buf, dtype=np.int32).view(PackedPromptIds)


def _safe_to_pack(new_req) -> bool:
    """Only plain text requests whose worker-side readers are known to accept
    an ndarray. Excluded (kept as lists): multimodal (M-RoPE grid scan over
    the ids), prompt embeds / mixed inputs, thinking_token_budget (upstream
    ThinkingBudgetState compares list slices), prompt logprobs and custom
    logits-processor args."""
    if getattr(new_req, "mm_features", None):
        return False
    if getattr(new_req, "prompt_embeds", None) is not None:
        return False
    if getattr(new_req, "prompt_is_token_ids", None) is not None:
        return False
    sp = getattr(new_req, "sampling_params", None)
    if sp is None:
        return False
    if getattr(sp, "thinking_token_budget", None) is not None:
        return False
    if getattr(sp, "prompt_logprobs", None) is not None:
        return False
    if getattr(sp, "extra_args", None):
        return False
    return True


def pack_new_request_prompt_ids(scheduler_output, min_len: int = 1024) -> int:
    """Replace list prompt ids of ``scheduled_new_reqs`` (len >= min_len) by
    ``PackedPromptIds``. Only the NewRequestData attribute is replaced; the
    engine's Request keeps its own list. Returns the number of tokens packed."""
    packed = 0
    for new_req in getattr(scheduler_output, "scheduled_new_reqs", None) or ():
        ids = getattr(new_req, "prompt_token_ids", None)
        if type(ids) is not list or len(ids) < min_len or not _safe_to_pack(new_req):
            continue
        arr = pack_prompt_ids(ids)
        if arr is None:
            continue
        new_req.prompt_token_ids = arr
        packed += len(arr)
    return packed
