# SPDX-License-Identifier: Apache-2.0
"""Reuse a recent prompt's block hashes for the blocks a new prompt shares with it.

Opt-in with ``VLLM_TPU_BLOCK_HASH_REUSE_PROMPTS=<n>``, the number of recent
prompts the engine core keeps (0, the default, keeps none).

Why: the engine core hashes every full block of a new request's prompt when it
builds the request, in its input thread and holding the GIL: a pickle and a
sha256 of each block's token ids with its parent block's hash. A 130k-token
prompt in 256-token blocks takes about 7 ms. In a multi-turn conversation each
turn's prompt starts with the previous turn's prompt, so all but its last few
blocks were hashed one turn earlier.

Why it is exact: block i's hash is computed from block i - 1's hash, block i's
tokens and block i's extra keys, so it depends only on the tokens up to the end
of block i and the extra keys of blocks 0 to i. A request without multimodal
inputs, LoRA or prompt embeddings has extra keys on block 0 only (its cache
salt). Block 0 is always hashed, and a kept prompt is a candidate only when its
block 0 hash is the same. A kept hash is reused only for blocks whose tokens,
and every token before them, equal the kept prompt's; every later block is
hashed as upstream does. Kept prompts with the same hash at a block are taken
to hold the same tokens up to its end, as the prefix cache does.

A kept prompt is replaced only by a prompt that extends all of its full blocks,
so a prompt that shares only the system prompt with another conversation's
prompt does not drop it; other prompts leave the table oldest first.
"""

from __future__ import annotations

import functools
import threading
from array import array
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import BlockHash
    from vllm.v1.request import Request

logger = init_logger(__name__)

SUMMARY_EVERY_REQUESTS = 256


@dataclass
class KeptPrompt:
    tokens: bytes
    hashes: list[BlockHash]


def packed(token_ids: list[int], blocks: int, block_size: int) -> bytes:
    """The token ids of the first ``blocks`` full blocks as 4-byte integers.

    Kept prompts are compared as bytes: comparing two lists of separately
    decoded ints costs about 12 ns per token, and dropping a kept list frees
    one object per token; a bytes compare is a memcmp and a drop frees one
    buffer.
    """
    ids = array("I")
    ids.fromlist(token_ids)
    del ids[blocks * block_size :]
    return ids.tobytes()


def longest_extended(
    tokens: bytes, candidates: list[tuple[int, KeptPrompt]], blocks: int, block_bytes: int
) -> tuple[int | None, int]:
    """The longest kept prompt whose full blocks all lead ``tokens``, and its block count.

    A multi-turn prompt usually extends its previous turn's prompt, so the
    kept prompts whose last full block reappears at the same place are
    compared whole, longest first.
    """
    extended = [
        (len(kept.hashes), key, kept)
        for key, kept in candidates
        if len(kept.hashes) <= blocks
        and tokens[(len(kept.hashes) - 1) * block_bytes : len(kept.hashes) * block_bytes]
        == kept.tokens[-block_bytes:]
    ]
    for count, key, kept in sorted(extended, key=lambda item: item[0], reverse=True):
        if tokens.startswith(kept.tokens):
            return key, count
    return None, 0


def shared_blocks(tokens: bytes, kept: bytes, start: int, limit: int, block_bytes: int) -> int:
    """How many leading blocks ``tokens`` and ``kept`` share, given the first ``start`` are shared.

    Grows the compared span by doubling and restarts at one block after a
    mismatch, so a long shared stretch takes a few memcmp calls.
    """
    shared, step = start, 1
    while shared < limit:
        end = min(shared + step, limit)
        if tokens[shared * block_bytes : end * block_bytes] == kept[shared * block_bytes : end * block_bytes]:
            shared, step = end, step * 2
        elif step == 1:
            break
        else:
            step = 1
    return shared


def longest_shared_blockwise(
    tokens: bytes, candidates: list[tuple[int, KeptPrompt]], blocks: int, block_bytes: int
) -> tuple[int | None, int]:
    """The most recent kept prompt sharing the most leading blocks with ``tokens``.

    Every candidate shares block 0. Kept prompts with the same hash at a block
    have the same tokens up to its end, as the prefix cache itself assumes, so
    an older candidate can share more than the best so far only when its hash
    at the best's last shared block is the best's; only those are compared
    further. Many conversations share a system prompt; this compares it once
    rather than once per conversation.
    """
    best_key, best, shared = None, None, 0
    for key, kept in reversed(candidates):
        limit = min(len(kept.hashes), blocks)
        if best is not None and (limit <= shared or kept.hashes[shared - 1] != best.hashes[shared - 1]):
            continue
        reach = shared_blocks(tokens, kept.tokens, shared if best is not None else 1, limit, block_bytes)
        if best is None or reach > shared:
            best_key, best, shared = key, kept, reach
    return best_key, shared


class RecentPrompts:
    """The full-block tokens and block hashes of the most recent prompts."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.lock = threading.Lock()
        self._prompts: OrderedDict[int, KeptPrompt] = OrderedDict()
        self._next_key = 0

    def __len__(self) -> int:
        return len(self._prompts)

    def longest_shared(
        self, tokens: bytes, first_hash: BlockHash, blocks: int, block_bytes: int
    ) -> tuple[int | None, int, bool]:
        """The longest kept prompt that ``tokens`` extends, else the one sharing the most leading blocks.

        Returns its key, the number of shared blocks and whether ``tokens``
        extends all of its full blocks.
        """
        candidates = [
            (key, kept) for key, kept in self._prompts.items() if kept.hashes[0] == first_hash
        ]
        key, shared = longest_extended(tokens, candidates, blocks, block_bytes)
        if key is not None:
            return key, shared, True
        key, shared = longest_shared_blockwise(tokens, candidates, blocks, block_bytes)
        return key, shared, False

    def hashes(self, key: int) -> list[BlockHash]:
        return self._prompts[key].hashes

    def keep(self, prompt: KeptPrompt, replaces: int | None) -> None:
        if replaces is not None:
            self._prompts.pop(replaces, None)
        self._prompts[self._next_key] = prompt
        self._next_key += 1
        while len(self._prompts) > self.capacity:
            self._prompts.popitem(last=False)


@dataclass
class ReuseCounts:
    requests: int = 0
    blocks: int = 0
    reused_blocks: int = 0


def extra_keys_only_on_first_block(request: Request) -> bool:
    return (
        request.prompt_token_ids is not None
        and not request.mm_features
        and request.lora_request is None
        and request.prompt_embeds is None
    )


def reusing_block_hasher(
    block_size: int,
    hash_fn: Callable[[Any], bytes],
    upstream: Callable[[Request], list[BlockHash]],
    capacity: int,
) -> Callable[[Request], list[BlockHash]]:
    """Upstream's request block hasher, with a new prompt's shared blocks taken from a recent prompt."""
    from vllm.v1.core import kv_cache_utils

    recent = RecentPrompts(capacity)
    counts = ReuseCounts()
    block_bytes = block_size * array("I").itemsize

    def request_block_hasher(request: Request) -> list[BlockHash]:
        if (
            request.block_hashes
            or not extra_keys_only_on_first_block(request)
            or request.num_tokens != len(request.prompt_token_ids)
        ):
            return upstream(request)
        blocks = request.num_tokens // block_size
        if blocks == 0:
            return []
        token_ids = request.all_token_ids
        tokens = packed(request.prompt_token_ids, blocks, block_size)
        first_keys, _ = kv_cache_utils.generate_block_hash_extra_keys(request, 0, block_size, 0)
        first = kv_cache_utils.hash_block_tokens(hash_fn, None, token_ids[:block_size], first_keys)
        with recent.lock:
            key, shared, extends_kept = recent.longest_shared(tokens, first, blocks, block_bytes)
            hashes = recent.hashes(key)[:shared] if shared else [first]
        reused = len(hashes) if shared else 0
        parent = hashes[-1]
        for block in range(len(hashes), blocks):
            parent = kv_cache_utils.hash_block_tokens(
                hash_fn, parent, token_ids[block * block_size : (block + 1) * block_size], None
            )
            hashes.append(parent)
        with recent.lock:
            recent.keep(KeptPrompt(tokens, list(hashes)), key if extends_kept else None)
            counts.requests += 1
            counts.blocks += blocks
            counts.reused_blocks += reused
            if counts.requests == 1 or counts.requests % SUMMARY_EVERY_REQUESTS == 0:
                logger.info(
                    "VLLM_TPU_BLOCK_HASH_REUSE_PROMPTS: %d requests, %d of %d prompt blocks "
                    "reused from %d kept prompts",
                    counts.requests,
                    counts.reused_blocks,
                    counts.blocks,
                    len(recent),
                )
        return hashes

    request_block_hasher.recent = recent
    request_block_hasher.counts = counts
    return request_block_hasher


def patch_block_hash_reuse() -> None:
    """Make the engine core's request block hasher reuse recent prompts' hashes."""
    from vllm_torchtpu import envs

    capacity = envs.VLLM_TPU_BLOCK_HASH_REUSE_PROMPTS
    if capacity <= 0:
        return
    from vllm.v1.core import kv_cache_utils
    from vllm.v1.engine import core

    upstream_factory = kv_cache_utils.get_request_block_hasher
    if getattr(upstream_factory, "_tpu_block_hash_reuse", False):
        return

    @functools.wraps(upstream_factory)
    def get_request_block_hasher(hash_block_size, caching_hash_fn):
        upstream = upstream_factory(hash_block_size, caching_hash_fn)
        return reusing_block_hasher(hash_block_size, caching_hash_fn, upstream, capacity)

    get_request_block_hasher._tpu_block_hash_reuse = True
    kv_cache_utils.get_request_block_hasher = get_request_block_hasher
    core.get_request_block_hasher = get_request_block_hasher
    logger.info(
        "VLLM_TPU_BLOCK_HASH_REUSE_PROMPTS: the engine core reuses the block hashes "
        "of its %d most recent prompts",
        capacity,
    )
