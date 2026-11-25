# SPDX-License-Identifier: Apache-2.0
"""Ships a new request's prompt token ids to the workers as an int32 array.

A ``NewRequestData`` carries the whole prompt in the ``SchedulerOutput`` that
admits the request. As a Python list, a 96k-token prompt costs the engine core
about 2 ms to pickle, every worker about 3 ms to unpickle and 3 ms to copy into
``InputBatch.token_ids_cpu``, and each garbage collection that sees the young
list a pass over all its items.

The engine core's input thread, which already builds the ``Request`` off the
main thread, keeps an int32 copy of the prompt on it. ``NewRequestData``
carries that copy in place of the list: it pickles as one buffer, lands in
``token_ids_cpu`` with one memcpy and is not tracked by the garbage collector.
The engine core's own ``Request`` keeps its list. When the list no longer
matches the copy (a connector trimmed the prompt), the list goes out as before.

On the worker the prompt is a read-only int32 array instead of a list. Every
worker-side reader of ``CachedRequestState.prompt_token_ids`` takes its length,
indexes it, or copies it into ``token_ids_cpu``; ``get_token_id`` returns a
numpy integer for a prompt position.

``VLLM_TPU_PROMPT_TOKEN_ARRAY=1`` enables it (engine core only).
"""

import array

import numpy as np

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Attribute on vllm.v1.request.Request holding the int32 copy.
ATTR = "_tpu_prompt_token_array"
_PATCH_FLAG = "_tpu_prompt_token_array_patch"


def prompt_array(token_ids: list[int]) -> np.ndarray | None:
    """Returns ``token_ids`` as a C-contiguous int32 array, or None when an id
    does not fit in int32."""
    try:
        return np.frombuffer(array.array("i", token_ids), dtype=np.int32)
    except OverflowError:
        return None


def matches(arr: np.ndarray, token_ids: list[int]) -> bool:
    """Whether ``arr`` still mirrors ``token_ids``. The engine core only ever
    shortens a prompt list (a producer's handoff trim pops its tail), so the
    length and the end tokens decide it."""
    n = len(token_ids)
    if len(arr) != n:
        return False
    return n == 0 or (int(arr[0]) == token_ids[0] and int(arr[-1]) == token_ids[-1])


def attach(request) -> None:
    """Stores the int32 copy of ``request``'s prompt on it."""
    token_ids = request.prompt_token_ids
    if type(token_ids) is list:
        arr = prompt_array(token_ids)
        if arr is not None:
            setattr(request, ATTR, arr)


def new_request_data(data, request):
    """Swaps ``data``'s prompt list for the request's int32 copy."""
    arr = request.__dict__.get(ATTR)
    if (
        arr is not None
        and data.prompt_token_ids is request.prompt_token_ids
        and data.prompt_token_ids is not None
        and matches(arr, data.prompt_token_ids)
    ):
        data.prompt_token_ids = arr
    return data


def patch_prompt_token_array() -> None:
    if not envs.VLLM_TPU_PROMPT_TOKEN_ARRAY:
        return
    from vllm.v1.core.sched.output import NewRequestData
    from vllm.v1.engine.core import EngineCore

    if getattr(EngineCore.preprocess_add_request, _PATCH_FLAG, False):
        return

    orig_preprocess = EngineCore.preprocess_add_request

    def preprocess_add_request(self, request):
        req, wave = orig_preprocess(self, request)
        attach(req)
        return req, wave

    setattr(preprocess_add_request, _PATCH_FLAG, True)
    preprocess_add_request.__wrapped__ = orig_preprocess
    EngineCore.preprocess_add_request = preprocess_add_request

    orig_from_request = NewRequestData.__dict__["from_request"].__func__

    def from_request(cls, request, *args, **kwargs):
        return new_request_data(
            orig_from_request(cls, request, *args, **kwargs), request
        )

    from_request.__wrapped__ = orig_from_request
    NewRequestData.from_request = classmethod(from_request)
    logger.info("VLLM_TPU_PROMPT_TOKEN_ARRAY: new requests carry int32 prompt arrays")
