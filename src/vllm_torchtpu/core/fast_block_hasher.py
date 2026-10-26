# SPDX-License-Identifier: Apache-2.0
"""Fast prefix-cache block hasher (TPU_FAST_BLOCK_HASH=1).

vLLM's request block hasher builds one Python tuple per hash block, pickles
``(parent_hash, tokens_tuple, extra_keys)`` and hashes the pickle. DeepSeek-V4's
hybrid KV groups hash every 16 tokens, so a 150k-token AgentX prompt costs ~9.4k
tuple+pickle+hash rounds (~0.2-0.25 us per prompt token, ~35 ms) on the engine
core's input thread before the request can be queued -- and the hash function
itself is a small part of it (xxhash saves only ~10%).

This hasher packs the new tokens into bytes once (``array('I')``) and chains
``H(parent_digest || 4-byte little-endian token ids of the block)`` with the
same digest family as the configured ``prefix_caching_hash_algo`` (sha256 ->
hashlib.sha256, xxhash -> xxh3_128). The encoding is fixed-width, so it is as
injective as the pickle one; block hashes keep their meaning (a chained
fingerprint of the full prefix ending at the block boundary) but their values
differ from upstream's, so they must not be mixed with hashes produced by the
stock hasher in another process (an external KV store keyed by block hash).
TPU_FAST_BLOCK_HASH=2 also keeps recent prompts' hashes
(TPU_FAST_BLOCK_HASH_CACHE_MB, default 1024) and reuses them for the longest
byte-identical block prefix of a new prompt (agentic multi-turn: each turn is the
last prompt + its output + a new message), hashing only the new tail.
Requests that need extra keys (multimodal, LoRA, cache salt, prompt embeds) or
token ids outside uint32 fall back to the stock hasher for the whole request.
"""

import array
import collections
import hashlib
import os
import sys
from collections.abc import Callable
from typing import Any

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_ENV = "TPU_FAST_BLOCK_HASH"


def _digest_fn(caching_hash_fn: Callable[[Any], bytes]) -> Callable[[bytes], bytes]:
    name = getattr(caching_hash_fn, "__name__", "")
    if name.startswith("xxhash"):
        import xxhash

        return xxhash.xxh3_128_digest
    if name.startswith("sha256"):
        return lambda b: hashlib.sha256(b).digest()
    raise ValueError(f"fast block hasher: unsupported hash fn {name!r}")


def _needs_extra_keys(request) -> bool:
    return bool(
        getattr(request, "mm_features", None)
        or getattr(request, "lora_request", None)
        or getattr(request, "cache_salt", None)
        or getattr(request, "prompt_embeds", None) is not None
    )


class PrefixHashCache:
    """Recent prompts' packed tokens and block hashes, so a multi-turn
    request (the previous turn's prompt + output + a new message) re-hashes
    only the blocks past its longest common prefix with a recent prompt.

    Candidates are bucketed by the bytes of the first ``fp_blocks`` blocks; the
    common prefix is found by a block-granular binary search over byte-slice
    equality (memcmp), so a reuse is exact: hashes are taken only for blocks
    whose tokens and whole preceding prefix are byte-identical."""

    def __init__(self, step: int, max_bytes: int, fp_blocks: int = 16,
                 per_key: int = 16):
        self.step = step
        self.max_bytes = max_bytes
        self.fp = fp_blocks * step
        self.per_key = per_key
        self.buckets: dict[bytes, list] = {}
        self.order: "collections.OrderedDict[int, tuple]" = collections.OrderedDict()
        self.nbytes = 0
        self._seq = 0
        self.hits = 0
        self.lookups = 0
        self.reused_blocks = 0
        self.total_blocks = 0

    def _lcp_blocks(self, a: bytes, b: bytes) -> int:
        step = self.step
        hi = min(len(a), len(b)) // step
        if a[:hi * step] == b[:hi * step]:
            return hi
        lo = 0  # blocks [0, lo) equal; [0, hi) not
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if a[lo * step:mid * step] == b[lo * step:mid * step]:
                lo = mid
            else:
                hi = mid
        return lo

    def lookup(self, raw: bytes) -> tuple[int, list]:
        self.lookups += 1
        best_n, best = 0, None
        for entry in self.buckets.get(raw[:self.fp], ()):
            n = self._lcp_blocks(raw, entry[1])
            if n > best_n:
                best_n, best = n, entry
                if n * self.step == len(raw):
                    break
        if best is None:
            return 0, []
        self.hits += 1
        return best_n, best[2][:best_n]

    def put(self, raw: bytes, hashes: list) -> None:
        if len(raw) < self.fp or len(raw) * 2 > self.max_bytes:
            return
        self._seq += 1
        key = raw[:self.fp]
        entry = (self._seq, raw, hashes)
        bucket = self.buckets.setdefault(key, [])
        bucket.insert(0, entry)
        size = len(raw) + 64 * len(hashes)
        self.order[self._seq] = (key, size)
        self.nbytes += size
        while len(bucket) > self.per_key:
            self._drop(bucket[-1][0])
        while self.nbytes > self.max_bytes and self.order:
            self._drop(next(iter(self.order)))

    def _drop(self, seq: int) -> None:
        key, size = self.order.pop(seq)
        self.nbytes -= size
        bucket = self.buckets.get(key)
        if bucket is not None:
            bucket[:] = [e for e in bucket if e[0] != seq]
            if not bucket:
                del self.buckets[key]


def make_fast_request_block_hasher(
    hash_block_size: int,
    caching_hash_fn: Callable[[Any], bytes],
    fallback: Callable,
    reuse_cache_mb: int = 0,
) -> Callable:
    """Return a drop-in replacement for vLLM's ``request_block_hasher``.

    ``reuse_cache_mb > 0`` adds a :class:`PrefixHashCache` for new requests."""
    from vllm.v1.core import kv_cache_utils as kcu

    digest = _digest_fn(caching_hash_fn)
    step = hash_block_size * 4
    little = sys.byteorder == "little"
    cache = (PrefixHashCache(step, reuse_cache_mb << 20)
             if reuse_cache_mb > 0 else None)

    def request_block_hasher(request) -> list:
        if _needs_extra_keys(request):
            return fallback(request)
        start = len(request.block_hashes) * hash_block_size
        n_blocks = (request.num_tokens - start) // hash_block_size
        if n_blocks <= 0:
            return []
        end = start + n_blocks * hash_block_size
        try:
            packed = array.array("I", request.all_token_ids[start:end])
        except (OverflowError, TypeError):
            return fallback(request)
        if not little:
            packed.byteswap()
        raw = packed.tobytes()
        prev = request.block_hashes[-1] if request.block_hashes else kcu.NONE_HASH
        out = []
        first = 0
        if cache is not None and start == 0:
            first, out = cache.lookup(raw)
            if first:
                prev = out[-1]
            cache.reused_blocks += first
            cache.total_blocks += n_blocks
        append = out.append
        for off in range(first * step, len(raw), step):
            prev = digest(prev + raw[off:off + step])
            append(prev)
        if cache is not None and start == 0:
            cache.put(raw, list(out))  # the request extends its own list
            if cache.lookups % 200 == 0:
                logger.info(
                    "%s: prefix-hash reuse %d/%d lookups hit, %.1f%% of blocks "
                    "reused, %d MiB cached", _ENV, cache.hits, cache.lookups,
                    100.0 * cache.reused_blocks / max(cache.total_blocks, 1),
                    cache.nbytes >> 20)
        return out

    request_block_hasher.cache = cache  # type: ignore[attr-defined]
    return request_block_hasher


def install() -> None:
    """Swap the engine core's block hasher factory when the env asks for it."""
    mode = os.environ.get(_ENV, "0")
    if mode not in ("1", "2"):
        return
    cache_mb = (int(os.environ.get("TPU_FAST_BLOCK_HASH_CACHE_MB", "1024"))
                if mode == "2" else 0)
    try:
        from vllm.v1.engine import core as engine_core

        if getattr(engine_core, "_tpu_fast_block_hash", False):
            return
        orig = engine_core.get_request_block_hasher

        def get_request_block_hasher(hash_block_size, caching_hash_fn):
            fallback = orig(hash_block_size, caching_hash_fn)
            try:
                fast = make_fast_request_block_hasher(
                    hash_block_size, caching_hash_fn, fallback, cache_mb)
            except ValueError as e:
                logger.warning("%s: %s; keeping the stock hasher", _ENV, e)
                return fallback
            logger.info("%s=%s: fast block hasher (hash_block_size=%d, %s, "
                        "prefix reuse cache %d MiB)", _ENV, mode, hash_block_size,
                        getattr(caching_hash_fn, "__name__", "?"), cache_mb)
            return fast

        engine_core.get_request_block_hasher = get_request_block_hasher
        engine_core._tpu_fast_block_hash = True
    except Exception as e:  # pragma: no cover
        logger.warning("%s: install failed: %s", _ENV, e)
