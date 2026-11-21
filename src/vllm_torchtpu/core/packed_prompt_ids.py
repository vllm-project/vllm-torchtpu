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
"""Prompt ids sent from the API server to the engine core as one packed array
(VLLM_TPU_PACKED_PROMPT_IDS).

vLLM's EngineCoreRequest carries ``prompt_token_ids`` as a list of Python ints.
For a 143k-token agent turn the hand-off then walks that list 3 times at
22 to 27 ns per token: msgspec encodes it int by int on the API server's
event loop, the engine core's input thread decodes it int by int, and the
byte block hasher copies it into a 32-bit buffer again.

With this patch the ids travel as one 32-bit array:

* The API server packs the ids once (the segment tokenizer hands over its own
  packed copy, see ``segment_tokenize.py``), and the prefix-affinity router
  hashes byte slices of that array.
* ``MsgpackEncoder.encode`` sends the request with ``prompt_token_ids=None``
  and the array as 2 trailing frames, a tag and the raw bytes, which zmq sends
  without a copy.
* ``MsgpackDecoder.decode`` in the engine core sees the tag, rebuilds the list
  with ``array.tolist()`` (1 C loop) and keeps the array, which the byte block
  hasher (``block_hash_bytes.py``) hashes instead of packing the list again.

With VLLM_TPU_ENGINE_ARRAY_PROMPT the decoder skips ``tolist()``: the request
carries the received array as a ``PromptIds``, and the engine's Request
(``ArrayPromptRequest``) reads its tokens as that array followed by the output
list (``TokenIds``). The scheduler output hands the same array to the model
runner, whose input batch copies it into its numpy buffer as bytes.
Resumable requests and every request of an engine with a KV connector keep the
list (``_keeps_array``).

The engine receives the same ids in the same order: only their encoding on
the wire changes. A request with prompt embeddings or a token mask keeps
vLLM's path. Both processes install the patch from the same environment, so
a tagged message only reaches a decoder that reads it.
"""

import itertools
import threading
from array import array
from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, overload

from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.engine import EngineCoreRequest

logger = init_logger(__name__)

TYPECODE = next(code for code in ("i", "l") if array(code).itemsize == 4)
FRAME_TAG = b"vllm-tpu-packed-prompt-ids-v1"


class PackedIds:
    """Packed copies of prompt id lists, found by the list's identity.

    Each entry holds its list, so the list's id is never reused by another list
    while the entry exists; an entry also checks the list's length, so a list
    grown or cut after packing is packed again. The oldest entries are dropped.
    """

    def __init__(self, capacity: int):
        self._capacity = capacity
        self._entries: OrderedDict[int, tuple[list[int], array]] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, ids: list[int], packed: array) -> None:
        with self._lock:
            self._entries[id(ids)] = (ids, packed)
            self._entries.move_to_end(id(ids))
            while len(self._entries) > self._capacity:
                self._entries.popitem(last=False)

    def get(self, ids: Any) -> array | None:
        with self._lock:
            entry = self._entries.get(id(ids))
        return _matching(entry, ids)

    def take(self, ids: Any) -> array | None:
        with self._lock:
            entry = self._entries.pop(id(ids), None)
        return _matching(entry, ids)


def _matching(entry: tuple[list[int], array] | None, ids: Any) -> array | None:
    if entry is None or entry[0] is not ids or len(entry[1]) != len(ids):
        return None
    return entry[1]


# API server: copies the renderer packed, read by the router and the encoder.
# Sized like segment_tokenize's TOKENIZER_MADE_IDS, which holds the same lists.
API_PACKED = PackedIds(capacity=64)
# Engine core: arrays the decoder received, taken by the byte block hasher in
# the same thread right after, so few are ever held.
ENGINE_PACKED = PackedIds(capacity=8)


def pack(ids: Sequence[int]) -> array:
    return array(TYPECODE, ids)


class PromptIds(array):
    """A prompt's ids as the engine core received them, read like vLLM's list.

    ``copy()`` is what vLLM's Request calls to start its token sequence: it
    gives the prompt followed by a list for later tokens, not a list of ints.
    """

    def copy(self) -> "TokenIds":
        return TokenIds(self, [])


class TokenIds(Sequence):
    """A prompt array followed by a list of tokens appended after it.

    The prompt is never changed (``_keeps_array``), so a copy shares it and
    copies only the tail: the scheduler copies a request's tokens for the model
    runner each time the request returns after a step without it.
    """

    __slots__ = ("prompt", "tail")

    def __init__(self, prompt: array, tail: list[int]):
        self.prompt = prompt
        self.tail = tail

    def __len__(self) -> int:
        return len(self.prompt) + len(self.tail)

    @overload
    def __getitem__(self, index: int) -> int: ...

    @overload
    def __getitem__(self, index: slice) -> list[int]: ...

    def __getitem__(self, index: int | slice) -> int | list[int]:
        prompt, tail = self.prompt, self.tail
        split = len(prompt)
        if isinstance(index, slice):
            start, stop, step = index.indices(split + len(tail))
            if step != 1:
                return [self[i] for i in range(start, stop, step)]
            if stop <= split:
                return prompt[start:stop].tolist()
            if start >= split:
                return tail[start - split : stop - split]
            return prompt[start:].tolist() + tail[: stop - split]
        if index < 0:
            index += split + len(tail)
            if index < 0:
                raise IndexError("token index out of range")
        return prompt[index] if index < split else tail[index - split]

    def __iter__(self):
        return itertools.chain(self.prompt, self.tail)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (list, TokenIds)):
            return NotImplemented
        return len(self) == len(other) and all(a == b for a, b in zip(self, other))

    def __repr__(self) -> str:
        return f"TokenIds(prompt={len(self.prompt)} ids, tail={self.tail!r})"

    def append(self, token: int) -> None:
        self.tail.append(token)

    def extend(self, tokens: Iterable[int]) -> None:
        self.tail.extend(tokens)

    def pop(self) -> int:
        # vLLM's structured output pops the spec tokens it put on a copy. The
        # shared prompt is never changed: a pop past the tail re-slices it.
        if self.tail:
            return self.tail.pop()
        token = self.prompt[-1]
        self.prompt = self.prompt[:-1]
        return token

    def copy(self) -> "TokenIds":
        return TokenIds(self.prompt, self.tail.copy())


# Engine core: whether its EngineCore has no KV connector, set before the
# input thread decodes any request (_install_array_prompt_request).
_engine_keeps_arrays = False


def _keeps_array(request: Any) -> bool:
    """vLLM edits the prompt list of a resumable (streaming) request in place,
    and KV connectors slice, copy or pop the token list of any request (offload
    hashing, P/D producer truncation), so those keep the list."""
    return _engine_keeps_arrays and not request.resumable


def _array_prompt_request(request_class: type) -> type:
    class ArrayPromptRequest(request_class):
        """vLLM's Request over a PromptIds prompt: its token list is a TokenIds
        that vLLM appends output tokens to, so ``_all_token_ids`` stays the
        prompt followed by ``_output_token_ids``, and its length is their sum."""

        @property
        def num_tokens(self) -> int:
            return self.num_prompt_tokens + len(self._output_token_ids)

        @property
        def num_tokens_with_spec(self) -> int:
            return (
                self.num_prompt_tokens
                + len(self._output_token_ids)
                + len(self.spec_token_ids)
            )

    return ArrayPromptRequest


def _install_array_prompt_request() -> None:
    """Build the engine core's Request as ArrayPromptRequest for a PromptIds
    prompt; every other request keeps vLLM's Request. The EngineCore records
    whether it has a KV connector (``_keeps_array``)."""
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.request import Request

    array_prompt_request = _array_prompt_request(Request)
    original = Request.from_engine_core_request.__func__
    original_init = EngineCore.__init__

    def from_engine_core_request(
        cls: type[Request],
        request: "EngineCoreRequest",
        block_hasher: Callable | None,
    ) -> Request:
        if cls is Request and type(request.prompt_token_ids) is PromptIds:
            cls = array_prompt_request
        return original(cls, request, block_hasher)

    def init_with_array_prompts(self, vllm_config, *args, **kwargs):
        global _engine_keeps_arrays
        # Keyed on the config, as the byte block hasher is: a connector reads
        # requests that carry no kv_transfer_params too.
        _engine_keeps_arrays = vllm_config.kv_transfer_config is None
        original_init(self, vllm_config, *args, **kwargs)

    Request.from_engine_core_request = classmethod(from_engine_core_request)
    EngineCore.__init__ = init_with_array_prompts


def packed_for(ids: list[int]) -> array:
    """The packed copy of ``ids``, packed now and remembered when none exists."""
    packed = API_PACKED.get(ids)
    if packed is None:
        packed = pack(ids)
        API_PACKED.put(ids, packed)
    return packed


def _sendable(request: Any) -> bool:
    return (
        isinstance(request.prompt_token_ids, list)
        and len(request.prompt_token_ids) > 0
        and request.prompt_embeds is None
        and request.prompt_is_token_ids is None
    )


def install() -> None:
    """Patch vLLM's msgpack encoder and decoder in this process."""
    from vllm_torchtpu import envs

    if not envs.VLLM_TPU_PACKED_PROMPT_IDS:
        if envs.VLLM_TPU_ENGINE_ARRAY_PROMPT:
            logger.warning(
                "VLLM_TPU_ENGINE_ARRAY_PROMPT needs VLLM_TPU_PACKED_PROMPT_IDS; "
                "the engine keeps the prompt as a list."
            )
        return
    import msgspec
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

    if getattr(MsgpackEncoder, "_tpu_packed_prompt_ids", False):
        return
    original_encode = MsgpackEncoder.encode
    original_decode = MsgpackDecoder.decode
    array_prompt = envs.VLLM_TPU_ENGINE_ARRAY_PROMPT

    def encode(self, obj):
        if type(obj) is not EngineCoreRequest or not _sendable(obj):
            return original_encode(self, obj)
        ids = obj.prompt_token_ids
        packed = API_PACKED.take(ids)
        if packed is None:
            packed = pack(ids)
        frames = original_encode(
            self, msgspec.structs.replace(obj, prompt_token_ids=None)
        )
        # The array stays referenced by its frame until zmq has sent it.
        return [*frames, FRAME_TAG, memoryview(packed).cast("B")]

    def decode(self, bufs):
        request = original_decode(self, bufs)
        if (
            type(request) is EngineCoreRequest
            and request.prompt_token_ids is None
            and request.prompt_embeds is None
            and not isinstance(bufs, (bytes, bytearray, memoryview))
            and len(bufs) >= 3
            and memoryview(bufs[-2]) == FRAME_TAG
        ):
            keep = array_prompt and _keeps_array(request)
            packed = PromptIds(TYPECODE) if keep else array(TYPECODE)
            packed.frombytes(memoryview(bufs[-1]))
            if keep:
                request.prompt_token_ids = packed
            else:
                ids = packed.tolist()
                request.prompt_token_ids = ids
                ENGINE_PACKED.put(ids, packed)
        return request

    MsgpackEncoder.encode = encode
    MsgpackDecoder.decode = decode
    MsgpackEncoder._tpu_packed_prompt_ids = True
    logger.info(
        "Applied TPU patch: prompt ids cross the engine hand-off as 1 packed array."
    )
    if array_prompt:
        _install_array_prompt_request()
        logger.info("Applied TPU patch: the engine's Request keeps the prompt array.")
