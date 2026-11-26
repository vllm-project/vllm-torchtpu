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
"""Exact segment-cached prompt tokenization for long multi-turn chat prompts."""

from __future__ import annotations

import contextlib
import copy
import json
import re
import threading
import weakref
from collections import OrderedDict
from typing import Any

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_MIN_CACHED_CHARS = 64
_ATTR_NAME = "_tpu_segment_tokenizer"
_ATTR_MISSING = object()


def _node_prepends(node: Any) -> bool:
    if isinstance(node, dict):
        if node.get("type") == "Prepend" or node.get("prepend_scheme") == "first":
            return True
        return any(_node_prepends(v) for v in node.values())
    if isinstance(node, (list, tuple)):
        return any(_node_prepends(v) for v in node)
    return False


def _obj_prepends(obj: Any) -> bool:
    if obj is None:
        return False
    if (
        type(obj).__name__ == "Prepend"
        or getattr(obj, "type", None) == "Prepend"
        or getattr(obj, "prepend_scheme", None) == "first"
    ):
        return True
    rep = repr(obj)
    if "Prepend(" in rep or "prepend_scheme=first" in rep:
        return True
    for attr in ("normalizers", "pretokenizers"):
        sub = getattr(obj, attr, None)
        if isinstance(sub, (list, tuple)) and any(_obj_prepends(x) for x in sub):
            return True
    return False


def _prepend_ineligible_reason(backend: Any) -> str | None:
    to_str = getattr(backend, "to_str", None)
    if callable(to_str):
        try:
            cfg = json.loads(to_str())
        except Exception:  # noqa: BLE001
            cfg = None
        if isinstance(cfg, dict):
            if _node_prepends(cfg.get("normalizer")):
                return "normalizer prepends sequence-start marker"
            if _node_prepends(cfg.get("pre_tokenizer")):
                return "pre_tokenizer prepends sequence-start marker"
    if _obj_prepends(getattr(backend, "normalizer", None)):
        return "normalizer prepends sequence-start marker"
    if _obj_prepends(getattr(backend, "pre_tokenizer", None)):
        return "pre_tokenizer prepends sequence-start marker"
    return None


class SegmentTokenizer:
    """Encodes text identically to ``backend.encode(..., add_special_tokens=False)``."""

    def __init__(self, backend: Any, max_cached_tokens: int):
        self._backend = copy.deepcopy(backend)
        self._backend.no_truncation()
        self._backend.no_padding()
        added = self._backend.get_added_tokens_decoder()
        self._added_id = {tok.content: tid for tid, tok in added.items()}
        alts: list[str] = sorted(self._added_id, key=len, reverse=True)
        self._split = re.compile("|".join(re.escape(a) for a in alts))
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
        """Return why ``backend`` cannot use piecewise caching, or ``None``."""
        if getattr(backend, "encode_special_tokens", False):
            return "encode_special_tokens is set"
        prepend_reason = _prepend_ineligible_reason(backend)
        if prepend_reason is not None:
            return prepend_reason
        added = backend.get_added_tokens_decoder()
        if not added:
            return "no added tokens to segment on"
        if max(added, default=0) >= 2_000_000:
            return "sparse added token ids"
        for tok in added.values():
            if tok.normalized or tok.lstrip or tok.rstrip or tok.single_word:
                return f"added token {tok.content!r} is not a hard split point"
        return None

    def encode(self, text: str) -> list[int]:
        pieces: list[tuple[bool, str]] = []
        pos = 0
        for m in self._split.finditer(text):
            if m.start() > pos:
                pieces.append((False, text[pos : m.start()]))
            pieces.append((True, m.group()))
            pos = m.end()
        if pos < len(text):
            pieces.append((False, text[pos:]))

        found: dict[str, list[int]] = {}
        todo: list[str] = []
        seen_todo: set[str] = set()
        with self._lock:
            for is_added, piece in pieces:
                if is_added or piece in found or piece in seen_todo:
                    continue
                ids = self._cache.get(piece)
                if ids is not None:
                    self._cache.move_to_end(piece)
                    found[piece] = ids
                    self.hits += 1
                else:
                    seen_todo.add(piece)
                    todo.append(piece)
            self.misses += len(todo)

        if todo:
            encs = self._backend.encode_batch(todo, add_special_tokens=False)
            to_cache: list[tuple[str, list[int]]] = []
            for piece, enc in zip(todo, encs):
                ids = list(map(self._ints.__getitem__, enc.ids))
                found[piece] = ids
                if len(piece) >= _MIN_CACHED_CHARS:
                    to_cache.append((piece, ids))
            if to_cache:
                with self._lock:
                    for piece, ids in to_cache:
                        self._insert(piece, ids)

        out: list[int] = []
        for is_added, piece in pieces:
            if is_added:
                out.append(self._added_id[piece])
            else:
                out.extend(found[piece])
        return out

    def _insert(self, piece: str, ids: list[int]) -> None:
        if (
            not ids
            or len(ids) > self._max_cached_tokens
            or len(piece) > self._max_cached_tokens * 16
            or piece in self._cache
        ):
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
    if (
        backend is None
        or not hasattr(backend, "encode_batch")
        or not hasattr(backend, "get_added_tokens_decoder")
    ):
        return None
    return backend


_INSTANCES: weakref.WeakKeyDictionary[Any, SegmentTokenizer | None] = (
    weakref.WeakKeyDictionary()
)
_INSTANCES_LOCK = threading.RLock()


def get_segment_tokenizer(
    tokenizer: Any, max_cached_tokens: int | None = None
) -> SegmentTokenizer | None:
    """Return a cached ``SegmentTokenizer`` for ``tokenizer``, or ``None``."""
    attached = getattr(tokenizer, _ATTR_NAME, _ATTR_MISSING)
    if attached is not _ATTR_MISSING:
        return attached  # type: ignore[return-value]

    if max_cached_tokens is None:
        from vllm_torchtpu.models.vllm.glm_dsa_dispatch import (
            SEGMENT_TOKENIZE_CACHE_TOKENS,
        )

        max_cached_tokens = SEGMENT_TOKENIZE_CACHE_TOKENS

    with _INSTANCES_LOCK:
        attached = getattr(tokenizer, _ATTR_NAME, _ATTR_MISSING)
        if attached is not _ATTR_MISSING:
            return attached  # type: ignore[return-value]
        try:
            if tokenizer in _INSTANCES:
                return _INSTANCES[tokenizer]
        except TypeError:
            pass

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
                "SegmentTokenizer enabled (%d added tokens, cache %d tokens).",
                len(seg._added_id),
                max_cached_tokens,
            )
        else:
            logger.debug(
                "SegmentTokenizer ineligible, keeping full tokenization: %s", reason
            )
        try:
            _INSTANCES[tokenizer] = seg
        except TypeError:
            with contextlib.suppress(AttributeError, TypeError):
                setattr(tokenizer, _ATTR_NAME, seg)
        return seg


def patch_renderer_segment_tokenize() -> None:
    """Route eligible long GLM-5 DSA prompts through ``SegmentTokenizer``."""
    from vllm.renderers import base as renderer_base

    cls = renderer_base.BaseRenderer
    if getattr(cls, "_tpu_segment_tokenize", False):
        return
    orig = cls._tokenize_prompt

    def _patched_tokenize_prompt(self: Any, prompt: Any, params: Any) -> Any:
        from vllm_torchtpu.models.vllm.glm_dsa_dispatch import (
            SEGMENT_TOKENIZE_MIN_MODEL_LEN,
            SEGMENT_TOKENIZE_MIN_PROMPT_CHARS,
            is_glm_dsa_model,
        )

        mc = getattr(self, "model_config", None)
        text = prompt.get("prompt", "") if isinstance(prompt, dict) else ""
        if (
            not is_glm_dsa_model(mc)
            or int(getattr(mc, "max_model_len", 0) or 0)
            < SEGMENT_TOKENIZE_MIN_MODEL_LEN
            or len(text) < SEGMENT_TOKENIZE_MIN_PROMPT_CHARS
        ):
            return orig(self, prompt, params)
        kwargs = params.get_encode_kwargs()
        if (
            kwargs.get("add_special_tokens")
            or kwargs.get("split_special_tokens")
            or self._wants_offsets(prompt, params)
        ):
            return orig(self, prompt, params)
        max_length = kwargs.get("max_length")
        if kwargs.get("truncation") and (
            max_length is None
            or int(max_length) <= 0
            or len(text) > int(max_length) * 16
        ):
            return orig(self, prompt, params)
        seg = get_segment_tokenizer(self.get_tokenizer())
        if seg is None:
            return orig(self, prompt, params)
        ids = seg.encode(text)
        if (
            kwargs.get("truncation")
            and max_length is not None
            and len(ids) > max_length
        ):
            return orig(self, prompt, params)
        return self._build_tokens_prompt(ids, prompt, offset_mapping=None)

    cls._tokenize_prompt = _patched_tokenize_prompt
    cls._tpu_segment_tokenize = True
