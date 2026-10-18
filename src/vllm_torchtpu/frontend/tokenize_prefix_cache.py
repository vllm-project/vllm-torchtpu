# SPDX-License-Identifier: Apache-2.0
"""Incremental prompt tokenization for multi-turn chat in the API server.

Agentic clients re-send the whole conversation every turn, so the API server
re-tokenizes 100k+ tokens of unchanged history per request, serially, in the
renderer's tokenizer thread. This module caches the token ids of recent
prompts at ChatML turn boundaries (``<|im_start|>``) and tokenizes only the
text after the longest cached boundary prefix.

Why splitting at the boundary is exact: an HF fast tokenizer first splits the
input on added tokens, then normalizes, pre-tokenizes and runs the model on
each piece between them independently. If ``<|im_start|>`` at position ``p``
is matched as its added token in both inputs, then

    encode(text) == encode(text[:p]) + encode(text[p:])

as long as the post-processor adds no tokens. Added tokens are matched
leftmost-longest, so that match is the same for every text that shares the
bytes up to the end of the boundary, given that:

- ``<|im_start|>`` is an added token that does not strip whitespace to its
  left, is not ``single_word`` and is matched before normalization;
- no other added token contains ``<|im_start|>``.

The cache is built only for tokenizers that meet these conditions (true for
the Qwen tokenizers). On every call, the k-th ``<|im_start|>`` in the text
must map to the k-th boundary token id, or the prompt is tokenized in full and
not cached. The first hits are also compared with a full tokenization; a
mismatch disables the cache for the process.
"""

import functools
import hashlib
import inspect
import multiprocessing
import threading
from array import array
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

BOUNDARY = "<|im_start|>"
# One entry per live conversation, sized for ~256 concurrent agentic
# sessions of up to ~256k tokens. Token ids are stored as uint32, so the
# token cap bounds the cache at 256 MiB.
_MAX_CACHED_TOKENS = 1 << 26
_MAX_ENTRIES = 1024
# Shorter prompts take a few milliseconds to tokenize in full; caching them
# would only evict long conversations.
_MIN_CACHED_PROMPT_TOKENS = 2048
# Hits compared with a full tokenization before the cache is trusted.
_VERIFIED_HITS = 8
_CACHE_ATTR = "_tpu_prefix_token_cache"

_installed = False
_build_lock = threading.Lock()


@dataclass
class _Entry:
    ids: array
    # Token index of each boundary after position 0, in text order.
    boundary_tokens: list[int]
    keys: list[bytes]
    # Keys in the index that still point here. A later prompt that extends
    # this one takes over all of them, and the entry is dropped at zero.
    live_keys: int = 0


class TurnPrefixTokenCache:
    """Thread-safe LRU of tokenized prompts, addressable by boundary prefix.

    Args:
        encode: Tokenizes a string without special tokens or truncation.
        boundary_id: Token id of ``BOUNDARY``.
        max_entries: Most prompts kept.
        max_cached_tokens: Most token ids kept across all prompts.
        min_cached_tokens: Shortest prompt, in tokens, that is cached.
    """

    def __init__(
        self,
        encode: Callable[[str], list[int]],
        boundary_id: int,
        *,
        max_entries: int = _MAX_ENTRIES,
        max_cached_tokens: int = _MAX_CACHED_TOKENS,
        min_cached_tokens: int = _MIN_CACHED_PROMPT_TOKENS,
    ) -> None:
        self._encode = encode
        self._boundary_id = boundary_id
        self._max_entries = max_entries
        self._max_cached_tokens = max_cached_tokens
        self._min_cached_tokens = min_cached_tokens
        self._lock = threading.Lock()
        self._entries: OrderedDict[int, _Entry] = OrderedDict()
        # Prefix key -> (entry id, boundary index in that entry).
        self._index: dict[bytes, tuple[int, int]] = {}
        self._next_entry_id = 0
        self.cached_tokens = 0
        self.hits = 0
        self.misses = 0
        self._verified = 0
        self.disabled = False

    @property
    def num_entries(self) -> int:
        return len(self._entries)

    @property
    def num_keys(self) -> int:
        return len(self._index)

    def encode(self, text: str) -> list[int]:
        """Returns ``encode(text)``, reusing the longest cached prefix."""
        positions = _boundary_positions(text)
        if self.disabled or not positions:
            return self._encode(text)
        keys = _prefix_keys(text, positions)

        hit = None
        with self._lock:
            for i in range(len(keys) - 1, -1, -1):
                found = self._index.get(keys[i])
                if found is not None:
                    self._entries.move_to_end(found[0])
                    hit = (i, self._entries[found[0]])
                    break

        if hit is None:
            ids = self._encode(text)
            first = (
                1 if text.startswith(BOUNDARY) and ids[:1] == [self._boundary_id] else 0
            )
            boundary_tokens = self._boundary_token_indices(ids, first, len(positions))
            with self._lock:
                self.misses += 1
            if boundary_tokens is None:
                return ids
        else:
            i, entry = hit
            prefix_len = entry.boundary_tokens[i]
            suffix_ids = self._encode(text[positions[i] :])
            suffix_tokens = self._boundary_token_indices(
                suffix_ids, 0, len(positions) - i
            )
            if suffix_tokens is None or suffix_tokens[0] != 0:
                logger.warning_once(
                    "Tokenize prefix cache: <|im_start|> text and token ids "
                    "disagree in a prompt suffix; tokenizing it in full."
                )
                return self._encode(text)
            ids = entry.ids[:prefix_len].tolist() + suffix_ids
            boundary_tokens = entry.boundary_tokens[:i] + [
                prefix_len + t for t in suffix_tokens
            ]
            with self._lock:
                self.hits += 1
                verify = self._verified < _VERIFIED_HITS
                if verify:
                    self._verified += 1
            if verify:
                full = self._encode(text)
                if full != ids:
                    self.disabled = True
                    logger.error(
                        "Tokenize prefix cache disabled: cached tokenization "
                        "differs from a full tokenization (%d vs %d tokens).",
                        len(ids),
                        len(full),
                    )
                    return full

        if len(ids) >= self._min_cached_tokens:
            self._insert(ids, boundary_tokens, keys)
        return ids

    def _boundary_token_indices(
        self, ids: list[int], first: int, count: int
    ) -> list[int] | None:
        """Indices of the boundary token in ``ids[first:]``, or None unless
        there are exactly ``count`` of them."""
        indices = []
        i = first - 1
        try:
            for _ in range(count):
                i = ids.index(self._boundary_id, i + 1)
                indices.append(i)
        except ValueError:
            return None
        try:
            ids.index(self._boundary_id, i + 1)
        except ValueError:
            return indices
        return None

    def _insert(
        self, ids: list[int], boundary_tokens: list[int], keys: list[bytes]
    ) -> None:
        entry = _Entry(array("I", ids), boundary_tokens, keys, len(keys))
        with self._lock:
            entry_id = self._next_entry_id
            self._next_entry_id += 1
            self._entries[entry_id] = entry
            self.cached_tokens += len(ids)
            for j, key in enumerate(keys):
                prev = self._index.get(key)
                if prev is not None:
                    self._release_key(prev[0])
                self._index[key] = (entry_id, j)
            while self._entries and (
                len(self._entries) > self._max_entries
                or self.cached_tokens > self._max_cached_tokens
            ):
                old_id, old = self._entries.popitem(last=False)
                self.cached_tokens -= len(old.ids)
                for key in old.keys:
                    if self._index.get(key, (None,))[0] == old_id:
                        del self._index[key]

    def _release_key(self, entry_id: int) -> None:
        """Drops one index key from ``entry_id``, and the entry with its last
        key. Caller holds the lock."""
        entry = self._entries.get(entry_id)
        if entry is None:
            return
        entry.live_keys -= 1
        if entry.live_keys == 0:
            del self._entries[entry_id]
            self.cached_tokens -= len(entry.ids)


def _boundary_positions(text: str) -> list[int]:
    """Start of every ``BOUNDARY`` after position 0."""
    positions = []
    p = text.find(BOUNDARY, 1)
    while p != -1:
        positions.append(p)
        p = text.find(BOUNDARY, p + len(BOUNDARY))
    return positions


def _prefix_keys(text: str, positions: list[int]) -> list[bytes]:
    """Key of ``text[:p]`` for each ``p``: its length and a 128-bit hash."""
    h = hashlib.blake2b(digest_size=16)
    keys = []
    prev = 0
    for p in positions:
        h.update(text[prev:p].encode())
        prev = p
        keys.append(p.to_bytes(8, "little") + h.digest())
    return keys


def build_cache(tokenizer: Any) -> TurnPrefixTokenCache | None:
    """Returns a cache for ``tokenizer``, or None if splitting it at
    ``BOUNDARY`` is not known to be exact."""
    reason = _unsafe_tokenizer_reason(tokenizer)
    if reason is not None:
        logger.info("Tokenize prefix cache not enabled: %s.", reason)
        return None

    def encode(text: str) -> list[int]:
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    logger.info(
        "Tokenize prefix cache enabled: prompts are tokenized incrementally "
        "at %s boundaries.",
        BOUNDARY,
    )
    return TurnPrefixTokenCache(encode, tokenizer.convert_tokens_to_ids(BOUNDARY))


def _unsafe_tokenizer_reason(tokenizer: Any) -> str | None:
    if tokenizer is None or not tokenizer.is_fast:
        return "needs an HF fast tokenizer"
    if _adds_special_tokens(tokenizer):
        return "the tokenizer adds special tokens around every input"
    boundary_id = tokenizer.convert_tokens_to_ids(BOUNDARY)
    added = tokenizer.added_tokens_decoder
    token = added.get(boundary_id)
    if token is None or token.content != BOUNDARY:
        return f"{BOUNDARY} is not an added token"
    if token.lstrip or token.single_word or token.normalized:
        return f"{BOUNDARY} may absorb neighboring text"
    if any(BOUNDARY in t.content and t.content != BOUNDARY for t in added.values()):
        return f"another added token contains {BOUNDARY}"
    probe = "x" + BOUNDARY + "user\nhi"
    ids = tokenizer(probe, add_special_tokens=False)["input_ids"]
    split = (
        tokenizer("x", add_special_tokens=False)["input_ids"]
        + tokenizer(probe[1:], add_special_tokens=False)["input_ids"]
    )
    if ids != split or ids.count(boundary_id) != 1:
        return f"tokenization does not split at {BOUNDARY}"
    return None


def _adds_special_tokens(tokenizer: Any) -> bool:
    with_specials = tokenizer("a", add_special_tokens=True)["input_ids"]
    return with_specials != tokenizer("a", add_special_tokens=False)["input_ids"]


def _is_api_server_process() -> bool:
    name = multiprocessing.current_process().name
    # ApiServer_<i>: multi-API-server manager. APIServer_DPRank_<r>: the
    # multi-port data-parallel supervisor.
    return name == "MainProcess" or name.startswith(("ApiServer", "APIServer"))


def find_unsupported_vllm_api() -> str | None:
    """Returns why the installed vLLM cannot take the patch, or None."""
    try:
        from vllm.renderers.base import BaseRenderer
        from vllm.renderers.hf import HfRenderer
        from vllm.renderers.params import TokenizeParams
    except ImportError as e:
        return str(e)
    if not issubclass(HfRenderer, BaseRenderer):
        return "HfRenderer is not a BaseRenderer"
    if "_tokenize_prompt" not in vars(BaseRenderer):
        return "BaseRenderer._tokenize_prompt is not defined"
    params = list(inspect.signature(BaseRenderer._tokenize_prompt).parameters)
    if params != ["self", "prompt", "params"]:
        return f"BaseRenderer._tokenize_prompt takes {params}"
    for name in ("_wants_offsets", "_build_tokens_prompt"):
        if not hasattr(BaseRenderer, name):
            return f"BaseRenderer.{name} is not defined"
    if not hasattr(TokenizeParams, "get_encode_kwargs"):
        return "TokenizeParams.get_encode_kwargs is not defined"
    return None


def _cache_for(renderer: Any) -> TurnPrefixTokenCache | None:
    if _CACHE_ATTR not in renderer.__dict__:
        with _build_lock:
            if _CACHE_ATTR not in renderer.__dict__:
                setattr(renderer, _CACHE_ATTR, build_cache(renderer.tokenizer))
    return renderer.__dict__[_CACHE_ATTR]


_HANDLED_ENCODE_KWARGS = frozenset({"truncation", "max_length", "add_special_tokens"})


def _cached_token_ids(
    cache: TurnPrefixTokenCache, text: str, encode_kwargs: dict[str, Any]
) -> list[int] | None:
    """Token ids equal to ``tokenizer(text, **encode_kwargs)``, or None to
    leave the call to vLLM."""
    if cache.disabled or not encode_kwargs.keys() <= _HANDLED_ENCODE_KWARGS:
        return None
    ids = cache.encode(text)
    # Truncation only changes prompts longer than max_length.
    max_length = encode_kwargs.get("max_length")
    if (
        encode_kwargs.get("truncation")
        and max_length is not None
        and len(ids) > max_length
    ):
        return None
    return ids


def maybe_install_tokenize_prefix_cache() -> None:
    """Patches ``HfRenderer._tokenize_prompt`` in the API-server process.

    Idempotent. Must run before any renderer is built: the renderer binds
    ``_tokenize_prompt`` into its async wrapper at construction.
    """
    global _installed
    if (
        _installed
        or not envs.VLLM_TPU_TOKENIZE_PREFIX_CACHE
        or not _is_api_server_process()
    ):
        return
    unsupported = find_unsupported_vllm_api()
    if unsupported is not None:
        logger.warning(
            "Tokenize prefix cache not installed (%s); prompts are tokenized in full.",
            unsupported,
        )
        return

    from vllm.renderers.hf import HfRenderer

    orig_tokenize_prompt = HfRenderer._tokenize_prompt

    @functools.wraps(orig_tokenize_prompt)
    def _tokenize_prompt(self, prompt, params):
        text = prompt.get("prompt")
        if (
            isinstance(text, str)
            and not prompt.get("multi_modal_data")
            and not self._wants_offsets(prompt, params)
        ):
            # Any failure falls back to vLLM's own tokenization.
            try:
                cache = _cache_for(self)
                ids = (
                    None
                    if cache is None
                    else _cached_token_ids(cache, text, params.get_encode_kwargs())
                )
            except Exception:
                logger.warning_once(
                    "Tokenize prefix cache failed; tokenizing in full.",
                    exc_info=True,
                )
                ids = None
            if ids is not None:
                return self._build_tokens_prompt(ids, prompt)
        return orig_tokenize_prompt(self, prompt, params)

    HfRenderer._tokenize_prompt = _tokenize_prompt
    _installed = True
    logger.info("Tokenize prefix cache installed on HfRenderer._tokenize_prompt.")
