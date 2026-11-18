# SPDX-License-Identifier: Apache-2.0
"""The engine core rebuilds every prompt the API server sends with only its new ids."""

import asyncio
import random

import pytest
from vllm import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.core_client import AsyncMPClient
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
        request, _ = EngineCore.preprocess_add_request(self, self.decoder.decode(frames))
        return request.prompt_token_ids


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(AsyncMPClient, "add_request_async", AsyncMPClient.add_request_async)
    monkeypatch.setattr(EngineCore, "preprocess_add_request", EngineCore.preprocess_add_request)
    snp.send_new_prompt_ids(capacity=8)
    snp.receive_new_prompt_ids(capacity=8)


def _request(request_id, ids, **fields):
    return EngineCoreRequest(
        request_id, ids, None, SamplingParams(max_tokens=8), None, 0.0, None, None, None, **fields
    )


def _send(client, request):
    asyncio.run(AsyncMPClient.add_request_async(client, request))
    return client.messages[-1]


def _fresh(ids):
    return [int(str(i)) for i in ids]


def _conversations(rng, conversations=12, turns=8):
    """Interleaved multi-turn prompts with a shared system prompt, edited turns and new conversations."""
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
        prompts[c] += [rng.randrange(1000, 150000) for _ in range(rng.randrange(1, 2 * PROBE_EVERY))]
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
        sent += len(MsgpackDecoder(EngineCoreRequest).decode(message[1:]).prompt_token_ids)
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
        MsgpackDecoder(EngineCoreRequest).decode(client.messages[-1][1:]).prompt_token_ids
    )
    assert header.shared == 5 * PROBE_EVERY + 3


def test_unshared_and_short_prompts_are_sent_whole(patched):
    client, engine = Client(), Engine()
    for n, prompt in enumerate([[5] * 10, [6] * 3 * PROBE_EVERY, [5] * 11, list(range(PROBE_EVERY))]):
        message = _send(client, _request(f"r{n}", list(prompt)))
        header = Header.decode(MsgpackDecoder(EngineCoreRequest).decode(message[1:]).prompt_token_ids)
        assert header.shared == 0
        assert engine.receive(message) == prompt


@pytest.mark.parametrize(
    "fields", [{"resumable": True}, {"prompt_is_token_ids": [True] * (2 * PROBE_EVERY)}]
)
def test_ineligible_requests_are_sent_unchanged(patched, fields):
    client, engine = Client(), Engine()
    prompt = list(range(2 * PROBE_EVERY))
    _send(client, _request("a", list(prompt)))
    message = _send(client, _request("b", list(prompt), **fields))
    assert MsgpackDecoder(EngineCoreRequest).decode(message[1:]).prompt_token_ids == prompt


def test_each_client_has_its_own_slots(patched):
    clients, engine = [Client(0), Client(1)], Engine()
    base = list(range(4 * PROBE_EVERY))
    for turn in range(4):
        for client in clients:
            prompt = base + [client.client_index] * (turn + 1)
            message = _send(client, _request(f"{client.client_index}-{turn}", list(prompt)))
            assert engine.receive(message) == prompt


def test_the_kept_prompt_sharing_the_most_ids_is_chosen():
    sent = SentPrompts(capacity=4)
    base = list(range(8 * PROBE_EVERY))
    kept = [base[: 6 * PROBE_EVERY + 5] + [-1] * PROBE_EVERY, base[: 3 * PROBE_EVERY] + [-2] * 5 * PROBE_EVERY]
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


@pytest.mark.parametrize("seed", range(20))
def test_shared_length_matches_a_scan(seed):
    rng = random.Random(seed)
    a = [rng.randrange(3) for _ in range(rng.randrange(0, 20000))]
    b = a[: rng.randrange(0, len(a) + 1)] + [rng.randrange(3) for _ in range(rng.randrange(0, 9000))]
    expected = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    assert shared_length(a, b) == expected
