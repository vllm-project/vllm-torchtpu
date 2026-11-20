# SPDX-License-Identifier: Apache-2.0
"""The engine core rebuilds every prompt the API server sends with only its new ids."""

import asyncio
import random

import pytest
from vllm import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.core_client import (
    AsyncMPClient,
    DPAsyncMPClient,
    DPLBAsyncMPClient,
)
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from vllm_torchtpu.core import send_new_prompt_ids as snp
from vllm_torchtpu.core.send_new_prompt_ids import (
    PROBE_EVERY,
    Header,
    ReceivedPrompts,
    SentPrompts,
    shared_length,
)


class Client:
    """The parts of ``AsyncMPClient`` that adding a request uses."""

    def __init__(self, client_index=0):
        self.client_index = client_index
        self.encoder = MsgpackEncoder()
        self.core_engine = b"\x00\x00"
        self.messages = []

    _send_input = AsyncMPClient._send_input

    def _send_input_message(self, message, engine):
        self.messages.append(message)
        future = asyncio.get_running_loop().create_future()
        future.set_result(None)
        return future

    def _ensure_output_queue_task(self):
        pass


class Engine:
    """The parts of ``EngineCore`` that preprocessing a request uses."""

    mm_receiver_cache = None
    request_block_hasher = None

    def __init__(self):
        self.decoder = MsgpackDecoder(EngineCoreRequest)

    def receive(self, message):
        _, *frames = message
        request, _ = EngineCore.preprocess_add_request(
            self, self.decoder.decode(frames)
        )
        return request.prompt_token_ids


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(
        AsyncMPClient, "add_request_async", AsyncMPClient.add_request_async
    )
    monkeypatch.setattr(
        EngineCore, "preprocess_add_request", EngineCore.preprocess_add_request
    )
    snp.send_new_prompt_ids(capacity=8)
    snp.receive_new_prompt_ids(capacity=8)


def _request(request_id, ids, **fields):
    return EngineCoreRequest(
        request_id,
        ids,
        None,
        SamplingParams(max_tokens=8),
        None,
        0.0,
        None,
        None,
        None,
        **fields,
    )


def _send(client, request):
    asyncio.run(AsyncMPClient.add_request_async(client, request))
    return client.messages[-1]


def _fresh(ids):
    return [int(str(i)) for i in ids]


def _conversations(rng, conversations=12, turns=8):
    """Interleaved multi-turn prompts with a shared system prompt, edited turns and new
    conversations."""
    system = [rng.randrange(1000, 150000) for _ in range(3 * PROBE_EVERY + 7)]
    prompts = {c: list(system) for c in range(conversations)}
    order = [c for c in range(conversations) for _ in range(turns)]
    rng.shuffle(order)
    for c in order:
        roll = rng.random()
        if roll < 0.1:
            prompts[c] = list(system)
        elif roll < 0.25:
            del prompts[c][-rng.randrange(1, 300) :]
        prompts[c] += [
            rng.randrange(1000, 150000)
            for _ in range(rng.randrange(1, 2 * PROBE_EVERY))
        ]
        yield prompts[c]


@pytest.mark.parametrize("fresh", [False, True])
def test_every_prompt_is_rebuilt_exactly(patched, fresh):
    rng = random.Random(7)
    client, engine = Client(), Engine()
    sent = received = 0
    for n, prompt in enumerate(_conversations(rng)):
        ids = _fresh(prompt) if fresh else list(prompt)
        message = _send(client, _request(f"r{n}", ids))
        assert engine.receive(message) == prompt
        sent += len(
            MsgpackDecoder(EngineCoreRequest).decode(message[1:]).prompt_token_ids
        )
        received += len(prompt)
    assert sent < received / 2


def test_a_prompt_that_differs_between_probes_is_rebuilt_exactly(patched):
    client, engine = Client(), Engine()
    kept = list(range(10 * PROBE_EVERY))
    edited = list(kept)
    edited[5 * PROBE_EVERY + 3] = -7
    for n, prompt in enumerate([kept, edited + [1, 2, 3]]):
        assert engine.receive(_send(client, _request(f"r{n}", list(prompt)))) == prompt
    header = Header.decode(
        MsgpackDecoder(EngineCoreRequest)
        .decode(client.messages[-1][1:])
        .prompt_token_ids
    )
    assert header.shared == 5 * PROBE_EVERY + 3


def test_unshared_and_short_prompts_are_sent_whole(patched):
    client, engine = Client(), Engine()
    for n, prompt in enumerate(
        [[5] * 10, [6] * 3 * PROBE_EVERY, [5] * 11, list(range(PROBE_EVERY))]
    ):
        message = _send(client, _request(f"r{n}", list(prompt)))
        header = Header.decode(
            MsgpackDecoder(EngineCoreRequest).decode(message[1:]).prompt_token_ids
        )
        assert header.shared == 0
        assert engine.receive(message) == prompt


@pytest.mark.parametrize(
    "fields", [{"resumable": True}, {"prompt_is_token_ids": [True] * (2 * PROBE_EVERY)}]
)
def test_ineligible_requests_are_sent_unchanged(patched, fields):
    client = Client()
    prompt = list(range(2 * PROBE_EVERY))
    _send(client, _request("a", list(prompt)))
    message = _send(client, _request("b", list(prompt), **fields))
    assert (
        MsgpackDecoder(EngineCoreRequest).decode(message[1:]).prompt_token_ids == prompt
    )


def test_each_client_has_its_own_slots(patched):
    clients, engine = [Client(0), Client(1)], Engine()
    base = list(range(4 * PROBE_EVERY))
    for turn in range(4):
        for client in clients:
            prompt = base + [client.client_index] * (turn + 1)
            message = _send(
                client, _request(f"{client.client_index}-{turn}", list(prompt))
            )
            assert engine.receive(message) == prompt


def test_data_parallel_clients_keep_their_own_add_request(patched):
    assert "add_request_async" in DPAsyncMPClient.__dict__
    assert DPLBAsyncMPClient.add_request_async is DPAsyncMPClient.add_request_async
    assert DPAsyncMPClient.add_request_async is not AsyncMPClient.add_request_async


def test_patch_installs_only_when_enabled(monkeypatch):
    upstream_add = AsyncMPClient.add_request_async
    upstream_preprocess = EngineCore.preprocess_add_request
    monkeypatch.setattr(AsyncMPClient, "add_request_async", upstream_add)
    monkeypatch.setattr(EngineCore, "preprocess_add_request", upstream_preprocess)
    monkeypatch.setattr(EngineCore, "_tpu_send_new_prompt_ids", False, raising=False)
    monkeypatch.setenv("VLLM_TPU_SEND_NEW_PROMPT_IDS", "0")
    snp.patch_send_new_prompt_ids()
    assert AsyncMPClient.add_request_async is upstream_add
    assert EngineCore.preprocess_add_request is upstream_preprocess

    monkeypatch.setenv("VLLM_TPU_SEND_NEW_PROMPT_IDS", "8")
    snp.patch_send_new_prompt_ids()
    add = AsyncMPClient.add_request_async
    preprocess = EngineCore.preprocess_add_request
    assert add.__wrapped__ is upstream_add
    assert preprocess.__wrapped__ is upstream_preprocess
    snp.patch_send_new_prompt_ids()
    assert AsyncMPClient.add_request_async is add
    assert EngineCore.preprocess_add_request is preprocess


def test_the_kept_prompt_sharing_the_most_ids_is_chosen():
    sent = SentPrompts(capacity=4)
    base = list(range(8 * PROBE_EVERY))
    kept = [
        base[: 6 * PROBE_EVERY + 5] + [-1] * PROBE_EVERY,
        base[: 3 * PROBE_EVERY] + [-2] * 5 * PROBE_EVERY,
    ]
    for ids in kept + [[-3] * 8 * PROBE_EVERY]:
        sent.keep(sent.header(ids), ids)
    header = sent.header(base)
    assert (header.shared_slot, header.shared) == (0, 6 * PROBE_EVERY + 5)


def test_least_recently_used_slot_is_replaced():
    sent = SentPrompts(capacity=2)
    prompts = [[c] * (2 * PROBE_EVERY) for c in range(3)]
    for ids in prompts[:2]:
        sent.keep(sent.header(ids), ids)
    extended = prompts[0] + [9]
    sent.keep(sent.header(extended), extended)
    header = sent.header(prompts[2])
    assert header.keep_slot == 1


def test_a_wrong_slot_fails_the_request():
    received = ReceivedPrompts(capacity=2)
    received.join(Header(-1, 0, 0, 0, 3), [-1, -1, 0, 0, 0, 3, 1, 2, 3])
    with pytest.raises(ValueError, match="does not hold"):
        received.join(Header(0, 3, 4, 1, 4), [-1, 0, 3, 4, 1, 4, 9])
    with pytest.raises(ValueError, match="joined"):
        received.join(Header(0, 3, 3, 1, 5), [-1, 0, 3, 3, 1, 5, 9])


def test_a_change_to_the_request_prompt_leaves_the_kept_one(patched):
    client, engine = Client(), Engine()
    first = list(range(3 * PROBE_EVERY))
    received = engine.receive(_send(client, _request("a", list(first))))
    received.pop()
    second = first + [7, 8, 9]
    assert engine.receive(_send(client, _request("b", list(second)))) == second


def test_a_failed_join_empties_the_slot_it_keeps():
    received = ReceivedPrompts(capacity=2)
    received.join(Header(-1, 0, 0, 0, 3), [-1, -1, 0, 0, 0, 3, 1, 2, 3])
    received.join(Header(-1, 0, 0, 1, 3), [-1, -1, 0, 0, 1, 3, 4, 2, 3])
    with pytest.raises(ValueError, match="does not hold"):
        received.join(Header(0, 3, 9, 1, 4), [-1, 0, 3, 9, 1, 4, 5])
    assert received.kept[1] == []
    with pytest.raises(ValueError, match="does not hold"):
        received.join(Header(1, 3, 3, 1, 4), [-1, 1, 3, 3, 1, 4, 5])


class FailingClient(Client):
    """A client whose send fails for the named requests, before or after upstream queues
    the message."""

    def __init__(self, failing, after_queue):
        super().__init__()
        self.failing = failing
        self.after_queue = after_queue

    def _send_input(self, request_type, request, engine=None):
        self.sending = request.request_id
        if self.sending in self.failing and not self.after_queue:
            raise TypeError("cannot encode the request")
        return Client._send_input(self, request_type, request, engine)

    def _ensure_output_queue_task(self):
        if self.sending in self.failing and self.after_queue:
            raise RuntimeError("the output task failed")


@pytest.mark.parametrize("after_queue", [False, True])
def test_a_failed_send_keeps_no_prompt_in_its_slot(patched, after_queue):
    client, engine = FailingClient({"b"}, after_queue), Engine()
    first = list(range(3 * PROBE_EVERY))
    assert engine.receive(_send(client, _request("a", list(first)))) == first
    second = first + [7, 8, 9]
    queued = len(client.messages)
    with pytest.raises((TypeError, RuntimeError)):
        asyncio.run(AsyncMPClient.add_request_async(client, _request("b", second)))
    if len(client.messages) > queued:
        engine.receive(client.messages[-1])
    client.failing = set()
    third = second + [10, 11]
    assert engine.receive(_send(client, _request("c", list(third)))) == third


def test_a_request_added_while_an_earlier_send_fails_is_rebuilt_exactly(patched):
    client, engine = FailingClient({"b"}, after_queue=False), Engine()
    first = list(range(3 * PROBE_EVERY))
    assert engine.receive(_send(client, _request("a", list(first)))) == first
    second = first + [7, 8, 9]
    third = second + [10, 11]

    async def add_both():
        return await asyncio.gather(
            AsyncMPClient.add_request_async(client, _request("b", list(second))),
            AsyncMPClient.add_request_async(client, _request("c", list(third))),
            return_exceptions=True,
        )

    failed, added = asyncio.run(add_both())
    assert isinstance(failed, TypeError) and added is None
    assert engine.receive(client.messages[-1]) == third


@pytest.mark.parametrize("seed", range(20))
def test_shared_length_matches_a_scan(seed):
    rng = random.Random(seed)
    a = [rng.randrange(3) for _ in range(rng.randrange(0, 20000))]
    b = a[: rng.randrange(0, len(a) + 1)] + [
        rng.randrange(3) for _ in range(rng.randrange(0, 9000))
    ]
    expected = next(
        (i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))
    )
    assert shared_length(a, b) == expected
