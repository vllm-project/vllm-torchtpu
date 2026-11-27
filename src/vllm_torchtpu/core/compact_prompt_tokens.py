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
"""TPU_COMPACT_NEW_REQ_TOKENS: ship new requests' prompt ids as int32 arrays.

A request's first `NewRequestData` carries its whole prompt as a Python list.
The scheduler output is pickled once in the engine core and unpickled on every
worker, and the worker then copies the list into its `InputBatch` row. Both
steps cost ~20-50 ns per token, on the worker main thread, right before the
step that contains the request's first prefill chunk. With agentic prompts of
100K-680K tokens that is 30-40 ms of device idle per prefill chunk.

With the flag on:
- the engine core's input thread (`EngineCore.preprocess_add_request`, which
  runs in parallel with the busy loop) converts each prompt of at least
  `TPU_COMPACT_NEW_REQ_TOKENS_MIN` tokens to an int32 ndarray, once;
- `NewRequestData.from_request` sends that array instead of the list. Pickle
  protocol 5 moves it as one out-of-band buffer, and the worker's
  `token_ids_cpu[row, :n] = ids` becomes a memcpy;
- `CachedRequestState.get_token_id` returns a Python int for array-backed
  prompts, as it does for lists.

The engine core's own `Request.prompt_token_ids` stays a list (block hashing,
the KV connectors and the scheduler read it). Off: nothing is patched.
"""

from __future__ import annotations

import numpy as np

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_ATTR = "_tpu_prompt_token_ids_np"


def to_compact(token_ids, min_tokens: int) -> np.ndarray | None:
    """Returns an int32 copy of `token_ids`, or None if too short or too wide."""
    if token_ids is None or len(token_ids) < max(1, min_tokens):
        return None
    try:
        arr = np.asarray(token_ids, dtype=np.int32)
    except (OverflowError, TypeError, ValueError):
        return None
    if arr.ndim != 1:
        return None
    arr.flags.writeable = False
    return arr


def compact_for(request, prompt_token_ids):
    """The array to send for `request`, if it still matches its prompt."""
    arr = getattr(request, _ATTR, None)
    if (
        arr is None
        or prompt_token_ids is None
        or prompt_token_ids is not request.prompt_token_ids
        or len(arr) != len(prompt_token_ids)
    ):
        return None
    return arr


def install_compact_prompt_tokens() -> None:
    if not envs.TPU_COMPACT_NEW_REQ_TOKENS:
        return
    min_tokens = envs.TPU_COMPACT_NEW_REQ_TOKENS_MIN

    from vllm.v1.core.sched.output import NewRequestData
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.worker.gpu_input_batch import CachedRequestState

    if not getattr(EngineCore, "_tpu_compact_prompt_tokens", False):
        orig_preprocess = EngineCore.preprocess_add_request

        def preprocess_add_request(self, request):
            req, wave = orig_preprocess(self, request)
            arr = to_compact(req.prompt_token_ids, min_tokens)
            if arr is not None:
                setattr(req, _ATTR, arr)
            return req, wave

        EngineCore.preprocess_add_request = preprocess_add_request
        EngineCore._tpu_compact_prompt_tokens = True

    if not getattr(NewRequestData, "_tpu_compact_prompt_tokens", False):
        orig_from_request = NewRequestData.from_request.__func__

        def from_request(cls, request, *args, **kwargs):
            data = orig_from_request(cls, request, *args, **kwargs)
            arr = compact_for(request, data.prompt_token_ids)
            if arr is not None:
                data.prompt_token_ids = arr
            return data

        NewRequestData.from_request = classmethod(from_request)
        NewRequestData._tpu_compact_prompt_tokens = True

    if not getattr(CachedRequestState, "_tpu_compact_prompt_tokens", False):
        orig_get_token_id = CachedRequestState.get_token_id

        def get_token_id(self, idx: int) -> int:
            tok = orig_get_token_id(self, idx)
            return int(tok) if isinstance(tok, np.integer) else tok

        CachedRequestState.get_token_id = get_token_id
        CachedRequestState._tpu_compact_prompt_tokens = True

    logger.info(
        "TPU_COMPACT_NEW_REQ_TOKENS: new requests' prompt ids (>= %d tokens) "
        "go to the workers as int32 arrays.",
        min_tokens,
    )
