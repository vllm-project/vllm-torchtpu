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
"""Exact segment-cached prompt tokenization for multi-turn chat prompts.

Opt-in with ``VLLM_TPU_SEGMENT_TOKENIZE=1``, which registers
:class:`SegmentTokenizeHfRenderer` as vLLM's ``hf`` renderer.

The API server tokenizes the whole chat-templated prompt on every turn, at
about 2 us per token on one core, while consecutive turns of a conversation
share almost all of their text.

An HF ``tokenizers`` pipeline first splits its input on added-token
occurrences (leftmost-longest), then normalizes, pre-tokenizes and BPE-encodes
each remaining piece on its own. When every added token is ``normalized=False``
(or the normalizer is the identity), ``lstrip=False``, ``rstrip=False`` and
``single_word=False``, and special tokens are not encoded as text, that split
is a function of the raw string alone, so

    encode(text) == concat(encode(piece) for piece in split(text))

Chat templates put added tokens around every message, so each message body is
one piece. :class:`SegmentTokenizer` performs the same split, serves pieces
seen before from an LRU cache keyed by their text, and encodes only the new
ones, in one ``encode_batch`` call.
"""

from __future__ import annotations

import copy
import json
import re
import threading
from collections import OrderedDict
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from vllm.renderers.hf import HfRenderer

from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.inputs import TextPrompt, TokensPrompt
    from vllm.renderers.params import TokenizeParams

logger = init_logger(__name__)

# Shorter pieces (template glue between added tokens) are re-encoded each time.
_MIN_CACHED_CHARS = 64
# Token ids the renderer's cache keeps: 8 bytes each plus the piece text, so
# about 200 MB for ASCII text; a few times the live conversations of a server.
_MAX_CACHED_TOKENS = 1 << 24


def _alternation(contents: Iterable[str]) -> re.Pattern | None:
    # Python's alternation takes the first alternative that matches at the
    # leftmost position, so longest-first order gives leftmost-longest: the
    # rule of tokenizers' added-token matcher.
    alts = sorted(set(contents), key=len, reverse=True)
    if not alts:
        return None
    return re.compile("|".join(re.escape(a) for a in alts))


def _split_on(pattern: re.Pattern | None, text: str) -> list[tuple[bool, str]]:
    if not text:
        return []
    if pattern is None:
        return [(False, text)]
    pieces: list[tuple[bool, str]] = []
    pos = 0
    for m in pattern.finditer(text):
        if m.start() > pos:
            pieces.append((False, text[pos : m.start()]))
        pieces.append((True, m.group()))
        pos = m.end()
    if pos < len(text):
        pieces.append((False, text[pos:]))
    return pieces


def _normalizer_is_identity(config: dict) -> bool:
    normalizer = config.get("normalizer")
    return normalizer is None or normalizer == {"type": "Sequence", "normalizers": []}


def _pre_tokenizers(node: dict | None) -> Iterable[dict]:
    if node is None:
        return
    if node.get("type") == "Sequence":
        for child in node["pretokenizers"]:
            yield from _pre_tokenizers(child)
    else:
        yield node


class SegmentTokenizer:
    """Encodes text exactly as ``backend.encode(text, add_special_tokens=False)``.

    ``backend`` is a ``tokenizers.Tokenizer``; it is copied, so later
    truncation or padding settings on the original do not reach these encodes.
    """

    def __init__(self, backend: Any, max_cached_tokens: int):
        reason = self.ineligible_reason(backend)
        if reason is not None:
            raise ValueError(f"segment tokenization would not be exact: {reason}")
        self._backend = copy.deepcopy(backend)
        self._backend.no_truncation()
        self._backend.no_padding()
        added = self._backend.get_added_tokens_decoder()
        self._added_id = {tok.content: tid for tid, tok in added.items()}
        # tokenizers splits on normalized=False tokens first, then on
        # normalized=True tokens inside the remaining pieces.
        self._split_raw = _alternation(
            t.content for t in added.values() if not t.normalized
        )
        self._split_norm = _alternation(
            t.content for t in added.values() if t.normalized
        )
        # Cached pieces point into one table of int objects: 8 bytes per
        # cached token instead of a boxed int each.
        max_id = max(self._backend.get_vocab(with_added_tokens=True).values())
        self._ints = list(range(max_id + 1))
        self._cache: OrderedDict[str, list[int]] = OrderedDict()
        self._max_cached_tokens = max_cached_tokens
        self._lock = threading.Lock()
        self.cached_tokens = 0
        self.hits = 0
        self.misses = 0
        self._check_probe()

    @property
    def num_added_tokens(self) -> int:
        return len(self._added_id)

    @staticmethod
    def ineligible_reason(backend: Any) -> str | None:
        """Why piecewise encoding of ``backend`` would not be exact, or None."""
        if getattr(backend, "encode_special_tokens", False):
            return "encode_special_tokens is set"
        config = json.loads(backend.to_str())
        for pre in _pre_tokenizers(config.get("pre_tokenizer")):
            # Prefixes only the piece at offset 0 of the whole input.
            if pre["type"] == "Metaspace" and pre.get("prepend_scheme") == "first":
                return "the Metaspace pre-tokenizer prepends to the first piece only"
        identity_normalizer = _normalizer_is_identity(config)
        for tok in backend.get_added_tokens_decoder().values():
            if tok.lstrip or tok.rstrip or tok.single_word:
                return f"added token {tok.content!r} is not a hard split point"
            if tok.normalized and not identity_normalizer:
                return (
                    f"added token {tok.content!r} is matched after a "
                    "non-identity normalizer"
                )
        return None

    @property
    def probe(self) -> str:
        """Text with every added token, each followed by a piece that starts
        without a space: where piecewise and full encodes would first differ."""
        return "hello" + "".join(
            f"{content}hello world 12\n x" for content in self._added_id
        )

    def _check_probe(self) -> None:
        want = self._backend.encode(self.probe, add_special_tokens=False).ids
        if self.encode(self.probe) != want:
            raise ValueError(
                "segment tokenization would not be exact: piecewise ids differ "
                "from the full encode on a probe string"
            )
        self.hits = self.misses = 0

    def split(self, text: str) -> list[tuple[bool, str]]:
        """``(is_added_token, text)`` pieces, as tokenizers' added-token pass."""
        pieces: list[tuple[bool, str]] = []
        for is_added, piece in _split_on(self._split_raw, text):
            if is_added:
                pieces.append((True, piece))
            else:
                pieces.extend(_split_on(self._split_norm, piece))
        return pieces

    def encode(self, text: str) -> list[int]:
        pieces = self.split(text)
        found: dict[str, list[int]] = {}
        todo: dict[str, None] = {}  # new pieces, in order, once each
        with self._lock:
            for is_added, piece in pieces:
                if is_added or piece in found or piece in todo:
                    continue
                ids = self._cache.get(piece)
                if ids is None:
                    todo[piece] = None
                else:
                    self._cache.move_to_end(piece)
                    self.hits += 1
                    found[piece] = ids
            self.misses += len(todo)
        if todo:
            # Outside the lock: the private backend is never reconfigured, and
            # encode_batch releases the GIL and spreads pieces across cores.
            encodings = self._backend.encode_batch(list(todo), add_special_tokens=False)
            for piece, enc in zip(todo, encodings):
                found[piece] = list(map(self._ints.__getitem__, enc.ids))
            with self._lock:
                for piece in todo:
                    if len(piece) >= _MIN_CACHED_CHARS:
                        self._insert(piece, found[piece])

        out: list[int] = []
        for is_added, piece in pieces:
            if is_added:
                out.append(self._added_id[piece])
            else:
                out.extend(found[piece])
        return out

    def _insert(self, piece: str, ids: list[int]) -> None:
        if piece in self._cache or len(ids) > self._max_cached_tokens:
            return
        self._cache[piece] = ids
        self.cached_tokens += len(ids)
        while self.cached_tokens > self._max_cached_tokens:
            _, old = self._cache.popitem(last=False)
            self.cached_tokens -= len(old)


class SegmentTokenizeHfRenderer(HfRenderer):
    """``HfRenderer`` whose prompt tokenization goes through a
    :class:`SegmentTokenizer`.

    Requests that ask for token offsets, special-token insertion or a
    truncated prompt keep ``HfRenderer``'s own tokenization: those outputs are
    not a concatenation of piece encodings.
    """

    def __init__(self, config: VllmConfig, tokenizer: Any) -> None:
        if tokenizer is None:
            raise ValueError("VLLM_TPU_SEGMENT_TOKENIZE=1 needs a tokenizer")
        if not hasattr(tokenizer, "backend_tokenizer"):
            raise ValueError(
                "VLLM_TPU_SEGMENT_TOKENIZE=1 needs an HF fast tokenizer, got "
                f"{type(tokenizer).__name__}"
            )
        if getattr(tokenizer, "split_special_tokens", False):
            raise ValueError(
                "VLLM_TPU_SEGMENT_TOKENIZE=1 cannot serve a tokenizer that "
                "splits special tokens"
            )
        super().__init__(config, tokenizer)
        hf_tokenizer = self.get_tokenizer()
        self.segment_tokenizer = seg = SegmentTokenizer(
            hf_tokenizer.backend_tokenizer, _MAX_CACHED_TOKENS
        )
        # HfRenderer tokenizes through the transformers class, which may
        # rewrite the text before the backend sees it.
        want = hf_tokenizer(seg.probe, add_special_tokens=False)["input_ids"]
        if seg.encode(seg.probe) != list(want):
            raise ValueError(
                "VLLM_TPU_SEGMENT_TOKENIZE=1: the tokenizer's ids differ from "
                f"its backend's on a probe string ({type(hf_tokenizer).__name__})"
            )
        logger.info(
            "VLLM_TPU_SEGMENT_TOKENIZE: piecewise prompt tokenization on "
            "(%d added tokens, cache %d tokens).",
            seg.num_added_tokens,
            _MAX_CACHED_TOKENS,
        )

    def _tokenize_prompt(
        self, prompt: TextPrompt, params: TokenizeParams
    ) -> TokensPrompt:
        kwargs = params.get_encode_kwargs()
        if kwargs.get("add_special_tokens") or self._wants_offsets(prompt, params):
            return super()._tokenize_prompt(prompt, params)
        ids = self.segment_tokenizer.encode(prompt["prompt"])
        max_length = kwargs.get("max_length")
        if (
            kwargs.get("truncation")
            and max_length is not None
            and len(ids) > max_length
        ):
            return super()._tokenize_prompt(prompt, params)
        return self._build_tokens_prompt(ids, prompt)
