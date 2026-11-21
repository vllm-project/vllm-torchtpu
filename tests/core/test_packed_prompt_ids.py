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
import types
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
VLLM_CODECS = MsgpackEncoder.encode, MsgpackDecoder.decode


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


def engine_core_built(kv_transfer_config=None):
    """Runs the patched EngineCore.__init__ over the test's stub of vLLM's."""
    from vllm.v1.engine.core import EngineCore

    config = types.SimpleNamespace(kv_transfer_config=kv_transfer_config)
    EngineCore.__init__(object.__new__(EngineCore), config)


@pytest.fixture
def array_prompt(monkeypatch):
    """The packed hand-off with VLLM_TPU_ENGINE_ARRAY_PROMPT in an engine core
    without a KV connector, undone after."""
    from vllm.v1.engine.core import EngineCore

    saved = (
        MsgpackEncoder.encode,
        MsgpackDecoder.decode,
        Request.__dict__["from_engine_core_request"],
    )
    MsgpackEncoder.encode, MsgpackDecoder.decode = VLLM_CODECS
    MsgpackEncoder._tpu_packed_prompt_ids = False
    monkeypatch.setenv("VLLM_TPU_PACKED_PROMPT_IDS", "1")
    monkeypatch.setenv("VLLM_TPU_ENGINE_ARRAY_PROMPT", "1")
    monkeypatch.setattr(EngineCore, "__init__", lambda self, vllm_config: None)
    monkeypatch.setattr(packed_ids, "_engine_keeps_arrays", False, raising=False)
    packed_ids.install()
    engine_core_built()
    yield
    MsgpackEncoder.encode, MsgpackDecoder.decode = saved[:2]
    Request.from_engine_core_request = saved[2]
    MsgpackEncoder._tpu_packed_prompt_ids = saved[0] is not VLLM_CODECS[0]


def received_request(ids, **kwargs):
    frames = MsgpackEncoder().encode(core_request(ids, **kwargs))
    return MsgpackDecoder(EngineCoreRequest).decode(over_zmq(frames))


def test_engine_keeps_the_received_array(array_prompt):
    ids = random_ids(5 * BLOCK + 3)
    decoded = received_request(ids)
    assert type(decoded.prompt_token_ids) is packed_ids.PromptIds
    assert decoded.prompt_token_ids.tolist() == ids


def test_resumable_requests_keep_the_list(array_prompt):
    ids = random_ids(40)
    request = core_request(ids)
    request.resumable = True
    frames = MsgpackEncoder().encode(request)
    decoded = MsgpackDecoder(EngineCoreRequest).decode(over_zmq(frames))
    assert type(decoded.prompt_token_ids) is list and decoded.prompt_token_ids == ids


def test_engines_with_a_kv_connector_keep_the_list(array_prompt):
    """KV connectors slice, copy and pop every request's token list, with or
    without kv_transfer_params (offload hashing, P/D producer truncation)."""
    engine_core_built(kv_transfer_config=object())
    ids = random_ids(40)
    for extra_args in (None, {"kv_transfer_params": {"x": 1}}):
        request = core_request(ids)
        request.sampling_params = SamplingParams(max_tokens=8, extra_args=extra_args)
        frames = MsgpackEncoder().encode(request)
        decoded = MsgpackDecoder(EngineCoreRequest).decode(over_zmq(frames))
        assert type(decoded.prompt_token_ids) is list
        assert decoded.prompt_token_ids == ids
    engine_core_built()
    assert type(received_request(ids).prompt_token_ids) is packed_ids.PromptIds


def same_tokens(array_request, list_request):
    n = list_request.num_tokens
    assert array_request.num_tokens == n
    assert array_request.num_tokens_with_spec == list_request.num_tokens_with_spec
    assert list(array_request.all_token_ids) == list(list_request.all_token_ids)
    assert list(array_request.all_token_ids.copy()) == list_request.all_token_ids.copy()
    assert list(array_request.output_token_ids) == list(list_request.output_token_ids)
    assert array_request.block_hashes == list_request.block_hashes
    split = list_request.num_prompt_tokens
    for index in (0, split - 1, split, n - 1, -1, -n):
        if -n <= index < n:
            assert (
                array_request.all_token_ids[index] == list_request.all_token_ids[index]
            )
    for window in (
        slice(0, n),
        slice(split - 3, split + 3),
        slice(split, None),
        slice(-5, None),
        slice(BLOCK, 3 * BLOCK),
        slice(None, None, 7),
        slice(n, n + 4),
        slice(9, 2),
    ):
        assert array_request.all_token_ids[window] == list_request.all_token_ids[window]


@pytest.mark.parametrize("hashing", ["bytes", "vllm"])
@pytest.mark.parametrize("n", [3, BLOCK, 7 * BLOCK + 5])
@pytest.mark.parametrize("cache_salt", [None, "salt"])
def test_array_prompt_request_reads_like_the_list_request(
    array_prompt, hashing, n, cache_salt
):
    if hashing == "bytes":
        hasher = block_hash_bytes.get_request_block_hasher(BLOCK, sha256)
    else:
        hasher = kv_cache_utils.get_request_block_hasher(BLOCK, sha256)
    ids = random_ids(n, seed=n + 3)
    decoded = received_request(ids)
    decoded.cache_salt = cache_salt
    array_request = Request.from_engine_core_request(decoded, hasher)
    sent = core_request(list(ids))
    sent.cache_salt = cache_salt
    list_request = Request.from_engine_core_request(sent, hasher)
    assert type(array_request) is not Request and isinstance(array_request, Request)
    assert type(list_request) is Request
    assert array_request.prompt_token_ids is decoded.prompt_token_ids
    same_tokens(array_request, list_request)

    outputs = random_ids(4 * BLOCK + 1, seed=11)
    steps = [outputs[:1], outputs[1], outputs[2 : BLOCK + 5], outputs[BLOCK + 5 :]]
    for step in steps:
        array_request.append_output_token_ids(step)
        list_request.append_output_token_ids(step)
        array_request.spec_token_ids = list_request.spec_token_ids = [1, 2]
        same_tokens(array_request, list_request)
    assert array_request.prompt_token_ids.tolist() == ids


@pytest.mark.parametrize("committed", [False, True])
def test_structured_output_pops_a_copy_like_the_list(array_prompt, committed):
    """vLLM's structured output copies all_token_ids, then extends and pops the
    copy while it looks for the end of reasoning in the spec tokens."""
    from vllm.v1.structured_output import StructuredOutputManager

    end = 3
    spec_tokens = [7, 8, end]
    seen = {}

    class Reasoner:
        def is_reasoning_end_streaming(self, input_ids, delta_ids):
            seen[kind].append(list(input_ids))
            return end in delta_ids

    manager = types.SimpleNamespace(
        enable_in_reasoning=False, _get_reasoner=lambda request: Reasoner()
    )
    ids = random_ids(2 * BLOCK + 1)
    tokens = {}
    for kind, sent in (("array", received_request(ids)), ("list", core_request(ids))):
        request = Request.from_engine_core_request(sent, None)
        request.append_output_token_ids([5, 6] + (spec_tokens if committed else []))
        seen[kind] = []
        structured = types.SimpleNamespace(
            structured_output_request=types.SimpleNamespace(reasoning_ended=False),
            all_token_ids=request.all_token_ids,
        )
        assert (
            StructuredOutputManager._get_constraint_start(
                manager, structured, spec_tokens, committed
            )
            == 3
        )
        tokens[kind] = list(request.all_token_ids)
    assert seen["array"] == seen["list"] and len(seen["list"]) == 2
    assert tokens["array"] == tokens["list"]


def test_scheduler_and_runner_copies_share_the_prompt(array_prompt):
    import torch
    from vllm.v1.core.sched.output import NewRequestData
    from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

    ids = random_ids(3 * BLOCK + 1)
    request = Request.from_engine_core_request(received_request(ids), None)
    request.append_output_token_ids([5, 6])

    new_data = NewRequestData.from_request(request, ([0],))
    assert new_data.prompt_token_ids is request.prompt_token_ids
    batch = InputBatch(
        max_num_reqs=1,
        max_model_len=len(ids) + 8,
        max_num_batched_tokens=len(ids) + 8,
        device=torch.device("cpu"),
        vocab_size=VOCAB,
        block_sizes=[BLOCK],
        kernel_block_sizes=[BLOCK],
        max_num_blocks_per_req=[4],
    )
    batch.add_request(
        CachedRequestState(
            req_id=new_data.req_id,
            prompt_token_ids=new_data.prompt_token_ids,
            mm_features=[],
            sampling_params=new_data.sampling_params,
            generator=None,
            block_ids=new_data.block_ids,
            num_computed_tokens=0,
            output_token_ids=list(request.output_token_ids),
        )
    )
    assert batch.token_ids_cpu[0, : len(ids) + 2].tolist() == ids + [5, 6]

    snapshot = request.all_token_ids.copy()
    request.append_output_token_ids(7)
    assert snapshot.prompt is request.prompt_token_ids
    assert snapshot[-2:] == [5, 6] and request.all_token_ids[-3:] == [5, 6, 7]
    popped = [snapshot.pop() for _ in range(4)]
    assert popped == [6, 5, ids[-1], ids[-2]] and list(snapshot) == ids[:-2]
    assert request.prompt_token_ids.tolist() == ids

    import pickle

    for value in (request.prompt_token_ids, snapshot):
        copy = pickle.loads(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        assert type(copy) is type(value) and list(copy) == list(value)
