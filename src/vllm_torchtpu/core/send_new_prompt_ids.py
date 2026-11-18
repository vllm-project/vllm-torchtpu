# SPDX-License-Identifier: Apache-2.0
"""Send the engine core only the prompt ids it has not seen yet.

Opt-in with ``VLLM_TPU_SEND_NEW_PROMPT_IDS=<n>``, the number of recent
prompts both sides keep (0, the default, keeps none).

Why: the API server sends every request's whole prompt to the engine core as
a list of ids. It encodes all of them into the message, and the engine core's
input thread decodes each back into a Python integer while holding the GIL.
A 157k-id prompt takes about 1.5 ms to encode and 7.6 ms to decode. In a
multi-turn conversation each turn's prompt starts with the previous turn's
prompt, which the engine core received one turn earlier.

How: the API server keeps the ids of its ``n`` most recent prompts in
numbered slots, and the engine core keeps a copy of each slot. A request
names the slot whose prompt shares the most leading ids with its own, how
many ids it shares, and the slot its whole prompt is kept in; it carries only
the ids after the shared ones. The engine core joins the kept ids and the new
ones into the full prompt before it builds the request.

Why it is exact: the API server decides every slot and compares the shared
ids with its own kept copy, so no id is guessed. Both sides change a slot
only for an add request, and add requests reach the engine core in the order
the API server sends them, on one socket. The engine core checks the prompt's
length and the last shared id against the API server's, and fails the request
if either differs.

A kept prompt is replaced only by a prompt that extends it, so a prompt that
shares only the system prompt with another conversation's prompt does not
drop it; other prompts leave the table oldest first. Requests with
multimodal inputs, prompt embeddings or streamed input, and every client with
data parallelism, send their whole prompt as before.
"""

from __future__ import annotations

import asyncio
import functools
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import msgspec

from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.engine import EngineCoreRequest

logger = init_logger(__name__)

MARK = -1
NO_SLOT = -1
HEADER_IDS = 6
PROBE_EVERY = 256
SUMMARY_EVERY_REQUESTS = 256


@dataclass(frozen=True)
class Header:
    """What a request tells the engine core in place of its shared ids."""

    shared_slot: int
    shared: int
    last_shared_id: int
    keep_slot: int
    total: int

    def encode(self, new_ids: list[int]) -> list[int]:
        ids = [MARK, self.shared_slot, self.shared, self.last_shared_id, self.keep_slot, self.total]
        ids.extend(new_ids)
        return ids

    @classmethod
    def decode(cls, ids: list[int]) -> Header | None:
        if not ids or ids[0] != MARK:
            return None
        return cls(*ids[1:HEADER_IDS])


@dataclass
class KeptPrompt:
    ids: list[int]
    probes: list[int]

    @classmethod
    def of(cls, ids: list[int]) -> KeptPrompt:
        return cls(ids, ids[PROBE_EVERY - 1 :: PROBE_EVERY])


def eligible(request: EngineCoreRequest) -> bool:
    return (
        isinstance(request.prompt_token_ids, list)
        and not request.mm_features
        and request.prompt_embeds is None
        and request.prompt_is_token_ids is None
        and not request.resumable
    )


def first_difference(a: list[Any], b: list[Any], start: int, end: int) -> int:
    """The first index in ``[start, end)`` where ``a`` and ``b`` differ, given that they differ there."""
    while end - start > 1:
        middle = (start + end) // 2
        if a[start:middle] == b[start:middle]:
            start = middle
        else:
            end = middle
    return start


def shared_length(a: list[Any], b: list[Any]) -> int:
    """The number of leading items ``a`` and ``b`` have in common."""
    n = min(len(a), len(b))
    if a[:n] == b[:n]:
        return n
    return first_difference(a, b, 0, n)


def shared_ids(ids: list[int], kept: KeptPrompt, probes: int) -> int:
    """The number of leading ids ``ids`` shares with ``kept``, whose first ``probes`` probes it matches.

    The ids up to the first probe that differs are compared at once, and only
    the probe interval that holds the first difference is searched.
    """
    n = min(len(ids), len(kept.ids))
    agreed = min(probes * PROBE_EVERY, n)
    if ids[:agreed] != kept.ids[:agreed]:
        return first_difference(ids, kept.ids, 0, agreed)
    end = min(agreed + PROBE_EVERY, n)
    if ids[agreed:end] == kept.ids[agreed:end]:
        return end
    return first_difference(ids, kept.ids, agreed, end)


class SentPrompts:
    """The API server's numbered slots of recent prompts, and the header of each new one."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.kept: OrderedDict[int, KeptPrompt] = OrderedDict()
        self.counts = Counts()

    def header(self, ids: list[int]) -> Header:
        """The slot ``ids`` shares the most leading ids with, and the slot it will be kept in."""
        slot, shared = self.most_shared(ids)
        extends = slot != NO_SLOT and shared == len(self.kept[slot].ids)
        keep_slot = slot if extends else self.free_slot(slot)
        if shared < PROBE_EVERY:
            slot, shared = NO_SLOT, 0
        return Header(slot, shared, ids[shared - 1] if shared else 0, keep_slot, len(ids))

    def keep(self, header: Header, ids: list[int]) -> None:
        if header.shared_slot != NO_SLOT:
            self.kept.move_to_end(header.shared_slot)
        self.kept[header.keep_slot] = KeptPrompt.of(ids)
        self.kept.move_to_end(header.keep_slot)
        self.counts.add(header)
        self.counts.log("the API server sent", len(self.kept))

    def most_shared(self, ids: list[int]) -> tuple[int, int]:
        extended = self.longest_extended(ids)
        if extended != NO_SLOT:
            return extended, len(self.kept[extended].ids)
        probes = ids[PROBE_EVERY - 1 :: PROBE_EVERY]
        best, best_probes = NO_SLOT, 0
        for slot, kept in reversed(self.kept.items()):
            if best_probes == len(probes):
                break
            if len(kept.probes) <= best_probes or kept.probes[best_probes] != probes[best_probes]:
                continue
            count = shared_length(probes, kept.probes)
            if count > best_probes:
                best, best_probes = slot, count
        if best == NO_SLOT:
            return NO_SLOT, 0
        return best, shared_ids(ids, self.kept[best], best_probes)

    def longest_extended(self, ids: list[int]) -> int:
        candidates = sorted(
            (
                (len(kept.ids), slot)
                for slot, kept in self.kept.items()
                if 0 < len(kept.ids) <= len(ids) and ids[len(kept.ids) - 1] == kept.ids[-1]
            ),
            reverse=True,
        )
        for length, slot in candidates:
            kept = self.kept[slot]
            if ids[PROBE_EVERY - 1 : length : PROBE_EVERY] == kept.probes and ids[:length] == kept.ids:
                return slot
        return NO_SLOT

    def free_slot(self, shared_slot: int) -> int:
        """A new slot while there is room, else the least recently used slot other than ``shared_slot``."""
        if len(self.kept) < self.capacity:
            return len(self.kept)
        return next((slot for slot in self.kept if slot != shared_slot), shared_slot)


class Counts:
    def __init__(self) -> None:
        self.requests = 0
        self.ids = 0
        self.sent_ids = 0

    def add(self, header: Header) -> None:
        self.requests += 1
        self.ids += header.total
        self.sent_ids += header.total - header.shared

    def log(self, side: str, kept: int) -> None:
        if self.requests == 1 or self.requests % SUMMARY_EVERY_REQUESTS == 0:
            logger.info(
                "VLLM_TPU_SEND_NEW_PROMPT_IDS: %s %d requests, %d of %d prompt ids, with %d kept prompts",
                side,
                self.requests,
                self.sent_ids,
                self.ids,
                kept,
            )


class ReceivedPrompts:
    """The engine core's copy of one client's slots."""

    def __init__(self, capacity: int) -> None:
        self.kept: list[list[int]] = [[] for _ in range(capacity)]
        self.counts = Counts()

    def join(self, header: Header, ids: list[int]) -> list[int]:
        if header.shared_slot == NO_SLOT:
            full = ids[HEADER_IDS:]
        else:
            kept = self.kept[header.shared_slot]
            if len(kept) < header.shared or kept[header.shared - 1] != header.last_shared_id:
                raise ValueError(
                    f"VLLM_TPU_SEND_NEW_PROMPT_IDS: slot {header.shared_slot} does not hold the "
                    f"API server's {header.shared} shared ids"
                )
            full = kept[: header.shared]
            full.extend(ids[HEADER_IDS:])
        if len(full) != header.total:
            raise ValueError(
                f"VLLM_TPU_SEND_NEW_PROMPT_IDS: joined {len(full)} prompt ids, the API server "
                f"sent a prompt of {header.total}"
            )
        self.kept[header.keep_slot] = full
        self.counts.add(header)
        self.counts.log("the engine core joined", sum(1 for kept in self.kept if kept))
        return full


def send_new_prompt_ids(capacity: int) -> None:
    """The API server's client sends each prompt's new ids with a header."""
    from vllm.v1.engine import EngineCoreRequestType
    from vllm.v1.engine.core_client import AsyncMPClient

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        request.client_index = self.client_index
        if eligible(request):
            sent = self.__dict__.setdefault("_tpu_sent_prompts", SentPrompts(capacity))
            header = sent.header(request.prompt_token_ids)
            ids = header.encode(request.prompt_token_ids[header.shared :])
            frames = self.encoder.encode(msgspec.structs.replace(request, prompt_token_ids=ids))
            sent.keep(header, request.prompt_token_ids)
        else:
            frames = self.encoder.encode(request)
        message = (EngineCoreRequestType.ADD.value, *frames)
        await asyncio.shield(self._send_input_message(message, self.core_engine))
        self._ensure_output_queue_task()

    AsyncMPClient.add_request_async = add_request_async


def receive_new_prompt_ids(capacity: int) -> None:
    """The engine core joins each request's kept and new prompt ids before it builds the request."""
    from vllm.v1.engine.core import EngineCore

    upstream = EngineCore.preprocess_add_request

    @functools.wraps(upstream)
    def preprocess_add_request(self, request: EngineCoreRequest):
        header = Header.decode(request.prompt_token_ids)
        if header is not None:
            received = self.__dict__.setdefault("_tpu_received_prompts", {})
            clients = received.setdefault(request.client_index, ReceivedPrompts(capacity))
            request.prompt_token_ids = clients.join(header, request.prompt_token_ids)
        return upstream(self, request)

    EngineCore.preprocess_add_request = preprocess_add_request


def patch_send_new_prompt_ids() -> None:
    """Send the engine core only the prompt ids it has not seen (VLLM_TPU_SEND_NEW_PROMPT_IDS)."""
    from vllm_torchtpu import envs

    capacity = envs.VLLM_TPU_SEND_NEW_PROMPT_IDS
    if capacity <= 0:
        return
    from vllm.v1.engine.core import EngineCore

    if EngineCore.__dict__.get("_tpu_send_new_prompt_ids", False):
        return
    send_new_prompt_ids(capacity)
    receive_new_prompt_ids(capacity)
    EngineCore._tpu_send_new_prompt_ids = True
    logger.info(
        "VLLM_TPU_SEND_NEW_PROMPT_IDS: prompts carry only the ids after those they share "
        "with one of the %d most recent prompts",
        capacity,
    )
