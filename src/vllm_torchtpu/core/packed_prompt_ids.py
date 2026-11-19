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

The engine receives the same ids in the same order: only their encoding on
the wire changes. A request with prompt embeddings or a token mask keeps
vLLM's path. Both processes install the patch from the same environment, so
a tagged message only reaches a decoder that reads it.
"""

import threading
from array import array
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

from vllm_torchtpu.logger import init_logger

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
        return
    import msgspec
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

    if getattr(MsgpackEncoder, "_tpu_packed_prompt_ids", False):
        return
    original_encode = MsgpackEncoder.encode
    original_decode = MsgpackDecoder.decode

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
            packed = array(TYPECODE)
            packed.frombytes(memoryview(bufs[-1]))
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
