# SPDX-License-Identifier: Apache-2.0
"""Ships a new request's prompt token ids to the workers as an int32 array.

A ``NewRequestData`` carries the whole prompt in the ``SchedulerOutput`` that
admits the request. As a Python list it is pickled on the engine core's main
thread, unpickled item by item by every worker, and copied into
``InputBatch.token_ids_cpu`` one int at a time; each young list also lengthens
every garbage collection that sees it.

The engine core's input thread, which builds the ``Request`` off the main
thread, keeps an int32 copy of the prompt on it. ``NewRequestData`` carries
that copy in place of the list: it pickles as one buffer, lands in
``token_ids_cpu`` with one memcpy and is not tracked by the garbage collector.
The engine core's own ``Request`` keeps its list. When the list no longer
matches the copy (a producer's handoff trim pops its last token, a streaming
update extends it), the list goes out as before.

On the worker the prompt is an int32 array instead of a list. The readers
that take it as is: ``len``, indexing (``get_token_id`` returns a numpy
integer for a prompt position) and the copy into ``token_ids_cpu``. Requests
whose prompt reaches a reader that needs a list keep the list: requests with
prompt logprobs or a thinking-token budget, pooling requests, models that
build M-RoPE or XD-RoPE positions from the prompt, and every request when a
custom logits processor is configured or installed (it receives the prompt in
its batch update).

``VLLM_TPU_PROMPT_TOKEN_ARRAY=1`` enables it (engine core).
"""

import array
import functools
from importlib.metadata import entry_points

import numpy as np

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Attribute on vllm.v1.request.Request holding the int32 copy.
ATTR = "_tpu_prompt_token_array"
_PATCH_FLAG = "_tpu_prompt_token_array_patch"
# vLLM's entry-point group for installed logits processor plugins.
_LOGITSPROCS_GROUP = "vllm.logits_processors"


def prompt_array(token_ids: list[int]) -> np.ndarray | None:
    """Returns ``token_ids`` as a C-contiguous int32 array, or None when an id
    does not fit in int32."""
    try:
        return np.frombuffer(array.array("i", token_ids), dtype=np.int32)
    except OverflowError:
        return None


def matches(arr: np.ndarray, token_ids: list[int]) -> bool:
    """Whether ``arr`` still mirrors ``token_ids``. The engine core changes a
    prompt list only by popping its tail (the producer's handoff trim) or
    extending it (a streaming update), so the length and the end tokens
    decide it."""
    n = len(token_ids)
    if len(arr) != n:
        return False
    return n == 0 or (int(arr[0]) == token_ids[0] and int(arr[-1]) == token_ids[-1])


def eligible(request) -> bool:
    """Whether every worker-side reader of ``request``'s prompt takes an
    array: not a pooling request (no sampling params), no prompt logprobs,
    no thinking-token budget."""
    params = request.sampling_params
    return (
        params is not None
        and params.prompt_logprobs is None
        and params.thinking_token_budget is None
    )


def attach(request) -> None:
    """Stores the int32 copy of ``request``'s prompt on it."""
    token_ids = request.prompt_token_ids
    if type(token_ids) is list and eligible(request):
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


def own_prompt_arrays(new_reqs) -> None:
    """Gives every received prompt array memory of its own.

    The executor's message queue pickles a buffer of 1 MiB or more (a prompt
    of 262,144 tokens or more) out of band, and a shared-memory reader
    unpickles it as a view of the ring slot, which the writer reuses once
    every reader has read it. The worker keeps the prompt for the life of the
    request (``get_token_id``, re-adding the request to the batch after it
    was left out of a step), so it copies the array here; one memcpy.
    """
    for data in new_reqs:
        ids = data.prompt_token_ids
        if isinstance(ids, np.ndarray) and not ids.flags.owndata:
            data.prompt_token_ids = ids.copy()


def _custom_logitsprocs(vllm_config) -> bool:
    if vllm_config.model_config.logits_processors:
        return True
    return bool(entry_points(group=_LOGITSPROCS_GROUP))


def patch_prompt_token_array() -> None:
    if not envs.VLLM_TPU_PROMPT_TOKEN_ARRAY:
        return
    from vllm.v1.core.sched.output import NewRequestData
    from vllm.v1.engine.core import EngineCore

    if getattr(EngineCore.preprocess_add_request, _PATCH_FLAG, False):
        return

    orig_preprocess = EngineCore.preprocess_add_request
    # Whether prompts may travel as arrays, decided at the first request
    # (only the engine core's input thread calls this).
    allowed: bool | None = None

    def preprocess_add_request(self, request):
        nonlocal allowed
        req, wave = orig_preprocess(self, request)
        if allowed is None:
            allowed = not _custom_logitsprocs(self.vllm_config)
            if not allowed:
                logger.info(
                    "VLLM_TPU_PROMPT_TOKEN_ARRAY: a custom logits processor "
                    "reads prompts; prompts stay lists"
                )
        if allowed:
            attach(req)
        return req, wave

    functools.update_wrapper(preprocess_add_request, orig_preprocess)
    setattr(preprocess_add_request, _PATCH_FLAG, True)
    EngineCore.preprocess_add_request = preprocess_add_request

    orig_from_request = NewRequestData.__dict__["from_request"].__func__

    def from_request(
        cls,
        request,
        block_ids,
        prefill_token_ids=None,
        uses_mrope=False,
        uses_xdrope=False,
    ):
        data = orig_from_request(
            cls, request, block_ids, prefill_token_ids, uses_mrope, uses_xdrope
        )
        if uses_mrope or uses_xdrope:
            # The worker builds rotary positions from the prompt list.
            return data
        return new_request_data(data, request)

    functools.update_wrapper(from_request, orig_from_request)
    NewRequestData.from_request = classmethod(from_request)
    logger.info("VLLM_TPU_PROMPT_TOKEN_ARRAY: new requests carry int32 prompt arrays")
