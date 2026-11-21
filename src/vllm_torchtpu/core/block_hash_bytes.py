# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""Prefix-cache block hashes over the token ids' bytes.

vLLM names each full block of a request by
``sha256(pickle.dumps((parent_hash, tuple(block_token_ids), extra_keys)))``.
With 256-token blocks and 100k-token prompts, the tuple copy and the pickle
cost about 10 µs per block in the engine core, before the request is queued.

This hasher packs the request's new full blocks once into a 32-bit buffer and
names a block without extra keys by ``sha256(parent_hash + block bytes)``.
Blocks with extra keys (multimodal, LoRA, cache salt, prompt embeddings) keep
vLLM's pickled form. Up to sha256 collisions, two blocks get equal hashes
exactly when their parent hash, token ids and extra keys are equal, as before,
so the prefix cache finds the same blocks; only the hash values differ. The
hashes stay inside one engine process, so the change applies only when no KV
connector or KV event publisher reads them, and only for vLLM's default
``sha256`` algorithm. It is installed in the engine-core process
(patch_registry's ``engine_core`` stage).
"""

import hashlib
from array import array
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from vllm_torchtpu.core.packed_prompt_ids import ENGINE_PACKED, PromptIds
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import BlockHash
    from vllm.v1.request import Request

logger = init_logger(__name__)

_TOKEN_ID_TYPECODE = next(code for code in ("I", "L") if array(code).itemsize == 4)
_TOKEN_ID_BYTES = 4


def _is_plain(request: "Request") -> bool:
    return (
        not request.mm_features
        and not request.lora_request
        and request.prompt_embeds is None
    )


def get_request_block_hasher(
    hash_block_size: int,
    caching_hash_fn: Callable[[Any], bytes],
) -> Callable[["Request"], list["BlockHash"]]:
    """A request block hasher that names plain blocks by their token bytes.

    The returned function gives the hashes of the request's full blocks that
    have none yet, in order: from block ``len(request.block_hashes)`` up to
    the last block ``request.num_tokens`` fills. Each hash chains on the hash
    before it; the first on ``request.block_hashes[-1]``, or on ``NONE_HASH``
    for a request's first block. A block without extra keys is named by
    ``sha256(parent + its token ids as 32-bit bytes)``; a block with extra
    keys (multimodal, LoRA, cache salt, prompt embeddings), or whose ids do
    not fit 32 bits, is hashed by vLLM's ``hash_block_tokens``. For a hash
    function other than ``sha256`` this returns vLLM's own hasher.

    The first call on a request reads its own token list,
    ``request._all_token_ids``, rather than a slice of ``all_token_ids``,
    which would copy the whole prompt.
    """
    from vllm.utils.hashing import sha256
    from vllm.v1.core import kv_cache_utils
    from vllm.v1.core.kv_cache_utils import (
        BlockHash,
        generate_block_hash_extra_keys,
        get_mm_features_in_window,
        hash_block_tokens,
    )

    if caching_hash_fn is not sha256:
        return kv_cache_utils.get_request_block_hasher(hash_block_size, caching_hash_fn)

    block_bytes = hash_block_size * _TOKEN_ID_BYTES
    new_sha256 = hashlib.sha256

    def request_block_hasher(request: "Request") -> list[BlockHash]:
        start_token_idx = len(request.block_hashes) * hash_block_size
        # The prompt's array as the engine received it (packed_prompt_ids.py):
        # the same 4-byte ids, so the same block hashes, without a second pack.
        received = None
        if start_token_idx == 0:
            prompt = request.prompt_token_ids
            received = (
                prompt if type(prompt) is PromptIds else ENGINE_PACKED.take(prompt)
            )
        num_tokens = request.num_tokens
        if start_token_idx + hash_block_size > num_tokens:
            return []
        num_new_blocks = (num_tokens - start_token_idx) // hash_block_size
        end_token_idx = start_token_idx + num_new_blocks * hash_block_size

        plain = _is_plain(request)
        curr_mm_idx = 0
        mm_features = request.mm_features
        if start_token_idx > 0 and mm_features:
            last_mm_pos = mm_features[-1].mm_position
            if last_mm_pos.offset + last_mm_pos.length > start_token_idx:
                curr_mm_idx, _ = get_mm_features_in_window(
                    mm_features, start_token_idx, start_token_idx + hash_block_size
                )

        all_token_ids = request._all_token_ids
        if start_token_idx == 0 and isinstance(all_token_ids, list):
            token_ids = all_token_ids
        elif received is not None and len(received) >= end_token_idx:
            # Read without making the whole prompt a list (PromptIds).
            token_ids = received
        else:
            token_ids = all_token_ids[start_token_idx:end_token_idx]
        if received is not None and len(received) >= end_token_idx:
            packed = memoryview(received).cast("B")
        else:
            try:
                packed = memoryview(array(_TOKEN_ID_TYPECODE, token_ids)).cast("B")
            except OverflowError:
                packed = None

        parent = (
            request.block_hashes[-1]
            if request.block_hashes
            else kv_cache_utils.NONE_HASH
        )
        new_block_hashes: list[BlockHash] = []
        for block in range(num_new_blocks):
            block_start = start_token_idx + block * hash_block_size
            if plain and block_start > 0:
                extra_keys = None
            else:
                extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
                    request, block_start, block_start + hash_block_size, curr_mm_idx
                )
            if extra_keys is None and packed is not None:
                offset = block * block_bytes
                digest = new_sha256(parent)
                digest.update(packed[offset : offset + block_bytes])
                parent = BlockHash(digest.digest())
            else:
                offset = block * hash_block_size
                parent = hash_block_tokens(
                    caching_hash_fn,
                    parent,
                    token_ids[offset : offset + hash_block_size],
                    extra_keys,
                )
            new_block_hashes.append(parent)
        return new_block_hashes

    return request_block_hasher


def _hashes_leave_the_process(vllm_config: Any) -> bool:
    kv_events = vllm_config.kv_events_config
    return vllm_config.kv_transfer_config is not None or (
        kv_events is not None and kv_events.enable_kv_cache_events
    )


def install_byte_block_hasher() -> None:
    """Give each EngineCore the byte hasher when its hashes stay in the process."""
    from vllm.utils.hashing import sha256
    from vllm.v1.engine.core import EngineCore

    if getattr(EngineCore, "_tpu_block_hash_bytes_patch", False):
        return
    original_init = EngineCore.__init__

    def init_with_byte_hasher(self, vllm_config, *args, **kwargs):
        original_init(self, vllm_config, *args, **kwargs)
        hasher = self.request_block_hasher
        if hasher is None:
            return
        if _hashes_leave_the_process(vllm_config):
            logger.info_once(
                "Kept vLLM's prefix-cache block hasher, because a KV connector "
                "or KV events read the block hashes."
            )
            return
        arguments = _hasher_arguments(hasher)
        if arguments is None:
            logger.warning(
                "Kept vLLM's prefix-cache block hasher, because "
                "its closure no longer holds hash_block_size and caching_hash_fn."
            )
            return
        hash_block_size, caching_hash_fn = arguments
        if caching_hash_fn is not sha256:
            logger.info_once(
                "Kept vLLM's prefix-cache block hasher, because the prefix-cache "
                "hash algorithm is not sha256."
            )
            return
        self.request_block_hasher = get_request_block_hasher(
            hash_block_size, caching_hash_fn
        )
        logger.info(
            "Prefix-cache block hashes over token bytes, %d tokens per block.",
            hash_block_size,
        )

    EngineCore.__init__ = init_with_byte_hasher
    EngineCore._tpu_block_hash_bytes_patch = True
    logger.info("Applied TPU patch: prefix-cache block hashes over token bytes.")


def _hasher_arguments(
    hasher: Callable,
) -> tuple[int, Callable[[Any], bytes]] | None:
    """Read (hash_block_size, caching_hash_fn) from vLLM's hasher closure."""
    code = hasher.__code__
    cells = dict(zip(code.co_freevars, hasher.__closure__ or ()))
    if "hash_block_size" not in cells or "caching_hash_fn" not in cells:
        return None
    return cells["hash_block_size"].cell_contents, cells[
        "caching_hash_fn"
    ].cell_contents
