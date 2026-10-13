# SPDX-License-Identifier: Apache-2.0
"""Exact segment-cached prompt tokenization for long multi-turn chat prompts.

Opt-in with ``VLLM_TPU_SEGMENT_TOKENIZE=1``.

Why: the API server tokenizes the whole chat-templated prompt on every turn.
HF fast tokenizers run ~1.5-2.5 us/token on one core, so a 500k-token agent
turn spends ~0.75-1.25 s in ``tokenizer(prompt)`` before the request reaches
the engine -- directly on TTFT, with the device idle. Consecutive turns of a
conversation share almost all of that text.

How (and why it is exact): an HF ``tokenizers`` pipeline first splits the
input on added-token occurrences (leftmost-longest), then normalizes,
pre-tokenizes and BPE-encodes each remaining piece independently. When every
added token is ``normalized=False``, ``lstrip=False``, ``rstrip=False`` and
``single_word=False``, and the tokenizer does not encode special tokens as
text, that first split is a pure function of the raw string, so

    encode(text) == concat(encode(piece) for piece in split(text))

holds piece by piece. Chat templates put added tokens (``<|user|>``,
``<|assistant|>``, ``<tool_response>``, ...) around every message, so each
message body is one piece. This module performs the same split, serves
pieces already seen from an LRU cache keyed by the piece text, and encodes
only new pieces -- in one ``encode_batch`` call, so a cold long prompt is
also encoded across cores. Tokenizers that do not satisfy the conditions
above, or requests that need offsets, special-token insertion or tokenizer
truncation, take vLLM's original path unchanged.
"""

from __future__ import annotations

import copy
import re
import threading
from collections import OrderedDict
from typing import Any

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Pieces shorter than this are re-encoded every time (cheap) rather than
# cached, so the cache holds message bodies, not template glue.
_MIN_CACHED_CHARS = 64


class SegmentTokenizer:
    """Encodes text exactly as ``backend.encode(text, add_special_tokens=False)``."""

    def __init__(self, backend: Any, max_cached_tokens: int):
        # Private copy: transformers mutates truncation/padding on the shared
        # backend per call, which must not leak into these encodes.
        self._backend = copy.deepcopy(backend)
        self._backend.no_truncation()
        self._backend.no_padding()
        added = self._backend.get_added_tokens_decoder()
        self._added_id: dict[str, int] = {
            tok.content: tid for tid, tok in added.items()
        }
        alts = sorted(self._added_id, key=len, reverse=True)
        # Python's alternation takes the first alternative matching at the
        # leftmost position; longest-first order makes that leftmost-longest,
        # the same rule as tokenizers' Aho-Corasick added-token matcher.
        self._split = re.compile("|".join(re.escape(a) for a in alts))
        # Cached pieces are lists over one shared table of int objects: 8 B
        # per cached token, and ``list.extend`` copies pointers instead of
        # boxing a fresh int per token.
        vocab = max(
            self._backend.get_vocab_size(with_added_tokens=True),
            max(added, default=0) + 1,
        )
        self._ints = list(range(vocab))
        self._cache: OrderedDict[str, list[int]] = OrderedDict()
        self._cached_tokens = 0
        self._max_cached_tokens = max_cached_tokens
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def eligible(backend: Any) -> str | None:
        """Return why ``backend`` cannot use the piecewise path, or None."""
        if getattr(backend, "encode_special_tokens", False):
            return "encode_special_tokens is set"
        for tok in backend.get_added_tokens_decoder().values():
            if tok.normalized or tok.lstrip or tok.rstrip or tok.single_word:
                return f"added token {tok.content!r} is not a hard split point"
        return None

    def encode(self, text: str) -> list[int]:
        pieces: list[tuple[bool, str]] = []  # (is_added_token, text)
        pos = 0
        for m in self._split.finditer(text):
            if m.start() > pos:
                pieces.append((False, text[pos : m.start()]))
            pieces.append((True, m.group()))
            pos = m.end()
        if pos < len(text):
            pieces.append((False, text[pos:]))

        with self._lock:
            found: dict[str, list[int]] = {}
            todo: list[str] = []
            for is_added, piece in pieces:
                if is_added or piece in found:
                    continue
                ids = self._cache.get(piece)
                if ids is not None:
                    self._cache.move_to_end(piece)
                    found[piece] = ids
                    self.hits += 1
                else:
                    found[piece] = None  # type: ignore[assignment]
                    todo.append(piece)
            self.misses += len(todo)
            if todo:
                encs = self._backend.encode_batch(todo, add_special_tokens=False)
                for piece, enc in zip(todo, encs):
                    ids = list(map(self._ints.__getitem__, enc.ids))
                    found[piece] = ids
                    if len(piece) >= _MIN_CACHED_CHARS:
                        self._insert(piece, ids)

        out: list[int] = []
        for is_added, piece in pieces:
            if is_added:
                out.append(self._added_id[piece])
            else:
                out.extend(found[piece])
        return out

    def _insert(self, piece: str, ids: list[int]) -> None:
        if len(ids) > self._max_cached_tokens:
            return
        if piece in self._cache:
            return
        self._cache[piece] = ids
        self._cached_tokens += len(ids)
        while self._cached_tokens > self._max_cached_tokens:
            _, old = self._cache.popitem(last=False)
            self._cached_tokens -= len(old)


def _backend_of(tokenizer: Any) -> Any:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        backend = getattr(tokenizer, "_tokenizer", None)
    if backend is None or not hasattr(backend, "encode_batch"):
        return None
    return backend


_INSTANCES: dict[int, SegmentTokenizer | None] = {}
_INSTANCES_LOCK = threading.Lock()


def get_segment_tokenizer(tokenizer: Any, max_cached_tokens: int):
    """One ``SegmentTokenizer`` per vLLM tokenizer object, or None if ineligible."""
    key = id(tokenizer)
    with _INSTANCES_LOCK:
        if key in _INSTANCES:
            return _INSTANCES[key]
        seg = None
        backend = _backend_of(tokenizer)
        if backend is None:
            reason = "no HF tokenizers backend"
        elif getattr(tokenizer, "split_special_tokens", False):
            reason = "split_special_tokens is set"
        else:
            reason = SegmentTokenizer.eligible(backend)
        if reason is None:
            seg = SegmentTokenizer(backend, max_cached_tokens)
            logger.info(
                "VLLM_TPU_SEGMENT_TOKENIZE: piecewise prompt tokenization on "
                "(%d added tokens, cache %d tokens, ~12 B/token).",
                len(seg._added_id),
                max_cached_tokens,
            )
        else:
            logger.warning(
                "VLLM_TPU_SEGMENT_TOKENIZE: falling back to full tokenization: %s",
                reason,
            )
        _INSTANCES[key] = seg
        return seg


def patch_renderer_segment_tokenize() -> None:
    """Route eligible ``BaseRenderer._tokenize_prompt`` calls through the cache.

    Must run before the renderer is constructed: ``BaseRenderer.__init__``
    binds ``self._tokenize_prompt`` into its async wrapper.
    """
    from vllm_torchtpu import envs

    if not envs.VLLM_TPU_SEGMENT_TOKENIZE:
        return
    from vllm.renderers import base as renderer_base

    cls = renderer_base.BaseRenderer
    if getattr(cls, "_tpu_segment_tokenize", False):
        return
    orig = cls._tokenize_prompt
    max_cached = envs.VLLM_TPU_SEGMENT_TOKENIZE_CACHE_TOKENS

    def _tokenize_prompt(self, prompt, params):
        kwargs = params.get_encode_kwargs()
        if kwargs.get("add_special_tokens") or self._wants_offsets(prompt, params):
            return orig(self, prompt, params)
        seg = get_segment_tokenizer(self.get_tokenizer(), max_cached)
        if seg is None:
            return orig(self, prompt, params)
        ids = seg.encode(prompt["prompt"])
        max_length = kwargs.get("max_length")
        if (
            kwargs.get("truncation")
            and max_length is not None
            and len(ids) > max_length
        ):
            # Over-long prompt: keep the tokenizer's own truncation semantics.
            return orig(self, prompt, params)
        return self._build_tokens_prompt(ids, prompt, offset_mapping=None)

    cls._tokenize_prompt = _tokenize_prompt
    cls._tpu_segment_tokenize = True
    logger.info("VLLM_TPU_SEGMENT_TOKENIZE: patched BaseRenderer._tokenize_prompt")
