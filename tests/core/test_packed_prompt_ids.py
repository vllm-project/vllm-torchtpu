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
"""Prompt ids cross vLLM's msgpack hand-off as 1 packed array and arrive as
the same list; the engine's block hashes and the router's prefix hashes from
the packed array match those from the list."""

import random
from array import array

import pytest
import zmq
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core import kv_cache_utils
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.request import Request
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from vllm_torchtpu.core import block_hash_bytes
from vllm_torchtpu.core import dp_prefix_affinity as affinity
from vllm_torchtpu.core import packed_prompt_ids as packed_ids

BLOCK = 16
VOCAB = 129_280


@pytest.fixture(autouse=True)
def none_hash(monkeypatch):
    monkeypatch.setattr(kv_cache_utils, "NONE_HASH", None, raising=False)
    kv_cache_utils.init_none_hash(sha256)


@pytest.fixture(scope="module")
def patched():
    saved = MsgpackEncoder.encode, MsgpackDecoder.decode
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("VLLM_TPU_PACKED_PROMPT_IDS", "1")
        packed_ids.install()
    yield
    MsgpackEncoder.encode, MsgpackDecoder.decode = saved
    MsgpackEncoder._tpu_packed_prompt_ids = False


def random_ids(n, seed=0):
    rng = random.Random(seed)
    return [rng.randrange(VOCAB) for _ in range(n)]


def core_request(ids, request_id="r", **kwargs):
    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=ids,
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        **kwargs,
    )


def over_zmq(frames):
    """The frames as the engine core's input thread receives them."""
    with zmq.Context() as ctx:
        push, pull = ctx.socket(zmq.PAIR), ctx.socket(zmq.PAIR)
        pull.bind("inproc://packed")
        push.connect("inproc://packed")
        push.send_multipart(frames, copy=False)
        received = pull.recv_multipart(copy=False)
        push.close()
        pull.close()
        return received


@pytest.mark.parametrize("n", [1, BLOCK, 5 * BLOCK + 3, 20_000])
def test_round_trip_gives_the_same_ids(patched, n):
    ids = random_ids(n, seed=n)
    frames = MsgpackEncoder().encode(core_request(ids))
    assert len(frames) == 3 and bytes(frames[1]) == packed_ids.FRAME_TAG
    decoded = MsgpackDecoder(EngineCoreRequest).decode(over_zmq(frames))
    assert decoded.prompt_token_ids == ids
    assert all(type(t) is int for t in decoded.prompt_token_ids)
    assert packed_ids.ENGINE_PACKED.take(decoded.prompt_token_ids).tolist() == ids


def test_renderer_copy_is_sent_and_dropped(patched):
    ids = random_ids(300)
    copy = packed_ids.pack(ids)
    packed_ids.API_PACKED.put(ids, copy)
    frames = MsgpackEncoder().encode(core_request(ids))
    assert frames[-1].obj is copy
    assert packed_ids.API_PACKED.get(ids) is None


def test_requests_with_embeddings_keep_the_list(patched):
    import torch

    request = core_request([1, 2, 3], prompt_embeds=torch.zeros(3, 4))
    frames = MsgpackEncoder().encode(request)
    assert packed_ids.FRAME_TAG not in [bytes(f) for f in frames]
    decoded = MsgpackDecoder(EngineCoreRequest).decode(over_zmq(frames))
    assert decoded.prompt_token_ids == [1, 2, 3]


def test_other_messages_are_unchanged(patched):
    frames = MsgpackEncoder().encode(("abort", ["r1", "r2"]))
    assert MsgpackDecoder().decode(over_zmq(frames)) == ["abort", ["r1", "r2"]]


def test_a_changed_list_is_packed_again():
    ids = [1, 2, 3]
    packed_ids.API_PACKED.put(ids, packed_ids.pack(ids))
    ids.append(4)
    assert packed_ids.API_PACKED.get(ids) is None
    assert packed_ids.packed_for(ids).tolist() == [1, 2, 3, 4]


def make_request(token_ids, hasher):
    return Request(
        "r",
        token_ids,
        SamplingParams(max_tokens=8),
        None,
        arrival_time=0.0,
        block_hasher=hasher,
    )


@pytest.mark.parametrize("n", [BLOCK - 1, BLOCK, 7 * BLOCK + 5])
def test_block_hashes_from_the_received_array(n):
    hasher = block_hash_bytes.get_request_block_hasher(BLOCK, sha256)
    ids = random_ids(n, seed=n + 1)
    expected = make_request(list(ids), hasher).block_hashes

    received = list(ids)
    packed_ids.ENGINE_PACKED.put(received, packed_ids.pack(received))
    request = make_request(received, hasher)
    assert request.block_hashes == expected
    assert packed_ids.ENGINE_PACKED.get(received) is None

    request.append_output_token_ids(random_ids(3 * BLOCK, seed=7))
    fresh = make_request(list(ids), hasher)
    fresh.append_output_token_ids(random_ids(3 * BLOCK, seed=7))
    assert request.block_hashes == fresh.block_hashes


def test_router_hashes_mark_the_same_prefixes():
    chunk = 64
    base = random_ids(10 * chunk + 7)
    other = list(base)
    other[5 * chunk + 3] ^= 1
    for ids in (base, other, base[: 3 * chunk]):
        assert len(affinity.packed_boundary_hashes(array("i", ids), chunk)) == len(
            affinity.boundary_hashes(ids, chunk)
        )
    a = affinity.packed_boundary_hashes(array("i", base), chunk)
    b = affinity.packed_boundary_hashes(array("i", other), chunk)
    assert a[:5] == b[:5] and all(x != y for x, y in zip(a[5:], b[5:]))
    assert (
        affinity.packed_boundary_hashes(array("i", base[: 3 * chunk]), chunk) == a[:3]
    )
