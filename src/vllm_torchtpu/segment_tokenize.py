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
added token is ``lstrip=False``, ``rstrip=False`` and
``single_word=False``, and the tokenizer does not encode special tokens as
text, that first split is a pure function of the raw string, so

    encode(text) == concat(encode(piece) for piece in split(text))

holds piece by piece. Added tokens with ``normalized=True`` are matched in a
second pass on the pieces the first leaves; they are accepted when the
normalizer keeps the text and none of them can overlap a ``normalized=False``
token, since the two passes then split as one. Chat templates put added tokens
(``<|user|>``, ``<|assistant|>``, ``<tool_response>``, ...) around every
message, so each message body is one piece. This module performs the same split, serves
pieces already seen from an LRU cache keyed by the piece text, and encodes
only new pieces -- in one ``encode_batch`` call, so a cold long prompt is
also encoded across cores. Tokenizers that do not satisfy the conditions
above, or requests that need offsets, special-token insertion or tokenizer
truncation, take vLLM's original path unchanged.

Prefix reuse (``VLLM_TPU_SEGMENT_TOKENIZE_PREFIX_PROMPTS=N``): the split, the
piece slicing, hashing and lookup and the id copy above still run over the
whole prompt in Python, ~0.05 ms per 1,000 tokens the previous turn already
had. With N > 0 the tokenizer keeps the ids of the N most recent prompts and
where each added token ends in them. A new prompt takes the ids of the longest
recent prompt it shares text with, up to the last added token inside the
shared text, and splits and encodes only the rest. The split up to the end
``e`` of an added token that starts at ``s`` is the same in both texts unless
an added token could match across ``e``: that needs ``text[p:e]`` to be a
proper prefix of an added token for some ``p <= s``, which is checked, and
such an end is not reused. The ids before ``e`` are then those of the same
pieces, and the rest is split from ``e`` as the full scan would.

Long pieces (``VLLM_TPU_SEGMENT_TOKENIZE_PART_CHARS=N``): ``encode_batch``
runs one input per core, so one long new piece (a tool result) still runs on
one core. With N > 0 a new piece of at least 2N chars is cut, every N chars or
more, right after a newline that is followed by a printable ASCII char, and
its parts are encoded in the same ``encode_batch`` call. BPE encodes each
pre-token alone, so the ids stay the same when each part, pre-tokenized alone,
gives the pre-tokens it has inside the whole piece. A pre-token starting at the
cut is not enough: GPT-2's ByteLevel regex splits ``"w \\nx"`` into ``" "``,
``"\\n"``, ``"x"`` but the part ``"w \\n"`` alone into ``" \\n"``. DeepSeek's
split regex keeps ``\\s*[\\r\\n]+`` as 1 pre-token in both. The cut is used only
when the backend's own pre-tokenizer passes every probe of
``_cut_probe_failure``.

Trusted ids: before a request leaves the API server, vLLM checks that every
prompt id is in the vocabulary with a ``max()`` and ``min()`` over the whole
id list, on the event loop. Every id this module returns comes from the
tokenizer's own output or its added-token table, so the check cannot fail.
The id list each patched tokenization returns is remembered by identity, and
``validate_token_ids`` skips that one list once, while it is unchanged in
length. Truncation, padding and multimodal processing build new lists and so
keep the check, as do prompts sent as ids and every other tokenization path.
"""

from __future__ import annotations

import bisect
import copy
import json
import re
import threading
from array import array
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from vllm_torchtpu.core.packed_prompt_ids import API_PACKED, pack
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Pieces shorter than this are re-encoded every time (cheap) rather than
# cached, so the cache holds message bodies, not template glue.
_MIN_CACHED_CHARS = 64

_FIRST_COMPARED_CHARS = 4096
# Texts that differ in the last chars before a position cannot share a prefix
# up to it; comparing this many first skips most prompts of other conversations.
_AGREEING_CHARS = 64

# Where a long piece may be cut: right after a newline followed by a printable
# ASCII char, where the parts must pre-tokenize as the whole (_cut_probe_failure).
_CUT = re.compile(r"\n(?=[!-~])")


@dataclass
class TokenizedPrompt:
    """A prompt's ids and, per added token, where it starts and ends and the ids up
    to its end."""

    text: str
    ids: list[int]
    token_starts: list[int]
    token_ends: list[int]
    ids_through_token: list[int]
    # ``ids`` packed as 32-bit ints (VLLM_TPU_PACKED_PROMPT_IDS), or None.
    packed: array | None = None


@dataclass
class SharedPrefix:
    prompt: TokenizedPrompt
    tokens: int

    @property
    def end(self) -> int:
        return self.prompt.token_ends[self.tokens - 1]

    @property
    def ids(self) -> int:
        return self.prompt.ids_through_token[self.tokens - 1]

    @property
    def whole(self) -> bool:
        """Whether the new prompt extends this prompt through its last added token,
        so it replaces it."""
        return self.tokens == len(self.prompt.token_ends)

    def join(self, rest: TokenizedPrompt, text: str) -> TokenizedPrompt:
        end, ids, tokens = self.end, self.ids, self.tokens
        prompt = self.prompt
        joined = prompt.ids[:ids]
        joined.extend(rest.ids)
        # The shared part's packed ids are copied as bytes; only the new ids
        # are packed one by one.
        packed = None
        if prompt.packed is not None:
            packed = prompt.packed[:ids]
            packed.extend(pack(rest.ids))
        return TokenizedPrompt(
            text=text,
            ids=joined,
            token_starts=prompt.token_starts[:tokens]
            + [s + end for s in rest.token_starts],
            token_ends=prompt.token_ends[:tokens] + [e + end for e in rest.token_ends],
            ids_through_token=prompt.ids_through_token[:tokens]
            + [n + ids for n in rest.ids_through_token],
            packed=packed,
        )


def _normalizer_keeps_text(backend: Any) -> bool:
    """Whether the backend's normalizer is absent or an empty sequence, so
    normalizing changes no text."""

    def keeps(normalizer: dict | None) -> bool:
        if normalizer is None:
            return True
        if normalizer.get("type") != "Sequence":
            return False
        return all(keeps(n) for n in normalizer.get("normalizers", []))

    return keeps(json.loads(backend.to_str()).get("normalizer"))


def _prepends_on_first_split_only(backend: Any) -> bool:
    """Whether the pre-tokenizer marks only the text's first split, as
    Metaspace with ``prepend_scheme="first"`` does.

    The full encode gives every piece between added tokens its offset in the
    whole text and prepends the marker only to the piece at offset 0; a piece
    encoded alone is at offset 0 and gets the marker too.
    """

    def first_only(pre_tokenizer: dict | None) -> bool:
        if pre_tokenizer is None:
            return False
        if pre_tokenizer.get("type") == "Sequence":
            return any(first_only(p) for p in pre_tokenizer.get("pretokenizers", []))
        return (
            pre_tokenizer.get("type") == "Metaspace"
            and pre_tokenizer.get("prepend_scheme") == "first"
        )

    return first_only(json.loads(backend.to_str()).get("pre_tokenizer"))


def _cut_probe_failure(backend: Any) -> str | None:
    """None when a cut at ``_CUT`` keeps the ids in every probe, else the failed
    probe. A probe passes when the text before the cut and the text after it,
    pre-tokenized alone, give the pre-tokens of the whole; the probes cover each
    printable ASCII char after the newline and each kind of char before it."""
    pre = backend.pre_tokenizer
    if pre is None:
        return "no pre-tokenizer"
    if not _normalizer_keeps_text(backend):
        return "a normalizer that changes the text"
    befores = [
        "a",
        "Z",
        "7",
        "123",
        ".",
        ":",
        "}",
        " ",
        "  ",
        "\t",
        "\r",
        "\n",
        "\n\n",
        ".\n",
        " \n",
        "\r\n",
        "\t\n",
        "\u00e9",
        "\u6771",
        "\U0001f600",
        "\u00a0",
    ]
    afters = [chr(c) for c in range(0x21, 0x7F)]
    tails = ["", "x", "1", " ", "\n", "abc def", "\u6771"]

    def words(text: str) -> list[str]:
        return [word for word, _ in pre.pre_tokenize_str(text)]

    # Each part is pre-tokenized once, so the probes cost 1 call per whole text.
    rights = {after + tail: words(after + tail) for after in afters for tail in tails}
    for before in befores:
        left = "w" + before + "\n"
        left_words = words(left)
        for right, right_words in rights.items():
            if words(left + right) != left_words + right_words:
                return f"{left!r} + {right!r} pre-tokenize unlike {left + right!r}"
    return None


def _overlapping_pair(first: list[str], second: list[str]) -> tuple[str, str] | None:
    """A token of ``first`` and one of ``second`` that can share characters in some
    text, or None.

    The tokenizer matches added tokens in two passes: those with ``normalized=False``
    on the raw text, then those with ``normalized=True`` on the pieces left between
    them. Under a normalizer that keeps the text, two passes split every text as one
    leftmost-longest pass over all added tokens does when no token of one pass can
    overlap a token of the other.
    """

    def proper_prefixes(tokens: list[str]) -> dict[str, str]:
        return {t[:i]: t for t in tokens for i in range(1, len(t))}

    def suffixes(token: str) -> list[str]:
        return [token[i:] for i in range(1, len(token))]

    for a, b in ((first, second), (second, first)):
        starts = proper_prefixes(b)
        for token in a:
            shared = next((s for s in suffixes(token) if s in starts), None)
            if shared is not None:
                return (
                    (token, starts[shared]) if a is first else (starts[shared], token)
                )
    for a in first:
        for b in second:
            if a in b or b in a:
                return a, b
    return None


def _agree_before(a: str, b: str, end: int) -> bool:
    start = max(0, end - _AGREEING_CHARS)
    return a[start:end] == b[start:end]


def shared_prefix_length(a: str, b: str) -> int:
    if len(b) <= len(a) and a.startswith(b):
        return len(b)
    if len(a) < len(b) and b.startswith(a):
        return len(a)
    n = min(len(a), len(b))
    lo, step = 0, _FIRST_COMPARED_CHARS
    while lo < n:
        hi = min(n, lo + step)
        if a[lo:hi] != b[lo:hi]:
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if a[lo:mid] == b[lo:mid]:
                    lo = mid
                else:
                    hi = mid
            return lo
        lo, step = hi, step * 2
    return n


class RecentPrompts:
    """The most recent tokenized prompts, searched for the longest prefix a new
    prompt can reuse."""

    def __init__(self, added_tokens: list[str], max_prompts: int):
        self._max_prompts = max_prompts
        self._prompts: OrderedDict[int, TokenizedPrompt] = OrderedDict()
        self._longest_token = max((len(t) for t in added_tokens), default=0)
        self._token_prefixes = {t[:i] for t in added_tokens for i in range(1, len(t))}

    def longest_prefix(self, text: str) -> SharedPrefix | None:
        best: SharedPrefix | None = None
        for prompt in reversed(self._prompts.values()):
            if self._may_start(text, prompt):
                best = self._longer(best, prompt, len(prompt.text))
        for prompt in reversed(self._prompts.values()):
            if best is None or (
                prompt.token_ends
                and prompt.token_ends[-1] > best.end
                and _agree_before(text, prompt.text, best.end)
            ):
                best = self._longer(
                    best, prompt, shared_prefix_length(text, prompt.text)
                )
        return best

    @staticmethod
    def _may_start(text: str, prompt: TokenizedPrompt) -> bool:
        n = len(prompt.text)
        return (
            n <= len(text)
            and _agree_before(text, prompt.text, n)
            and text.startswith(prompt.text)
        )

    def _longer(
        self, best: SharedPrefix | None, prompt: TokenizedPrompt, shared: int
    ) -> SharedPrefix | None:
        if best is not None and shared <= best.end:
            return best
        tokens = bisect.bisect_right(prompt.token_ends, shared)
        while tokens > 0 and not self._split_ends_at(prompt, tokens - 1):
            tokens -= 1
        if tokens > 0 and (best is None or prompt.token_ends[tokens - 1] > best.end):
            return SharedPrefix(prompt, tokens)
        return best

    def _split_ends_at(self, prompt: TokenizedPrompt, token: int) -> bool:
        start, end = prompt.token_starts[token], prompt.token_ends[token]
        first = max(0, end - self._longest_token + 1)
        text = prompt.text
        return not any(
            text[p:end] in self._token_prefixes for p in range(first, start + 1)
        )

    def add(self, prompt: TokenizedPrompt, replaces: TokenizedPrompt | None) -> None:
        if replaces is not None:
            self._prompts.pop(id(replaces), None)
        self._prompts[id(prompt)] = prompt
        while len(self._prompts) > self._max_prompts:
            self._prompts.popitem(last=False)


class SegmentTokenizer:
    """Encodes text exactly as ``backend.encode(text, add_special_tokens=False)``."""

    def __init__(
        self,
        backend: Any,
        max_cached_tokens: int,
        max_prompts: int = 0,
        packed: bool = False,
        part_chars: int = 0,
    ):
        # Private copy: transformers mutates truncation/padding on the shared
        # backend per call, which must not leak into these encodes.
        self._backend = copy.deepcopy(backend)
        self._backend.no_truncation()
        self._backend.no_padding()
        added = self._backend.get_added_tokens_decoder()
        self._added_id = {tok.content: tid for tid, tok in added.items()}
        alts: list[str] = sorted(self._added_id, key=len, reverse=True)
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
        self._recent = RecentPrompts(alts, max_prompts) if max_prompts > 0 else None
        self._packed = packed
        failure = _cut_probe_failure(self._backend) if part_chars > 0 else None
        if failure is not None:
            logger.warning(
                "VLLM_TPU_SEGMENT_TOKENIZE_PART_CHARS: cutting after a newline "
                "would change the ids under the pre-tokenizer %r (%s); long "
                "pieces stay whole.",
                str(self._backend.pre_tokenizer),
                failure,
            )
            part_chars = 0
        self._part_chars = part_chars
        self.hits = 0
        self.misses = 0

    @staticmethod
    def eligible(backend: Any) -> str | None:
        """Return why ``backend`` cannot use the piecewise path, or None."""
        if getattr(backend, "encode_special_tokens", False):
            return "encode_special_tokens is set"
        added = list(backend.get_added_tokens_decoder().values())
        if not added:
            return "no added tokens to split at"
        if _prepends_on_first_split_only(backend):
            return "the pre-tokenizer prepends its marker to the first split only"
        for tok in added:
            if tok.lstrip or tok.rstrip or tok.single_word:
                return f"added token {tok.content!r} is not a hard split point"
        normalized = [tok.content for tok in added if tok.normalized]
        if not normalized:
            return None
        if not _normalizer_keeps_text(backend):
            return f"added token {normalized[0]!r} is matched on normalized text"
        raw = [tok.content for tok in added if not tok.normalized]
        clash = _overlapping_pair(normalized, raw)
        if clash is not None:
            return f"normalized added token {clash[0]!r} can overlap {clash[1]!r}"
        return None

    def encode(self, text: str) -> list[int]:
        return self.encode_reusing(text)[0]

    def encode_reusing(self, text: str) -> tuple[list[int], int]:
        """The ids of ``text`` and how many of them came from a recent prompt."""
        prompt, reused = self.encode_prompt(text)
        return prompt.ids, reused

    def encode_prompt(self, text: str) -> tuple[TokenizedPrompt, int]:
        """``text`` tokenized, with its packed ids when packing is on and recent
        prompts are kept, and how many ids came from a recent prompt."""
        if self._recent is None:
            return self.encode_pieces(text), 0
        with self._lock:
            prefix = self._recent.longest_prefix(text)
        if prefix is None:
            prompt = self.encode_pieces(text)
            if self._packed:
                prompt.packed = pack(prompt.ids)
            reused = 0
        else:
            prompt = prefix.join(self.encode_pieces(text[prefix.end :]), text)
            reused = prefix.ids
        with self._lock:
            self._recent.add(
                prompt, replaces=prefix.prompt if prefix and prefix.whole else None
            )
        return prompt, reused

    def encode_pieces(self, text: str) -> TokenizedPrompt:
        pieces: list[tuple[bool, str]] = []  # (is_added_token, text)
        token_starts: list[int] = []
        token_ends: list[int] = []
        pos = 0
        for m in self._split.finditer(text):
            start, end = m.span()
            if start > pos:
                pieces.append((False, text[pos:start]))
            pieces.append((True, m.group()))
            token_starts.append(start)
            token_ends.append(end)
            pos = end
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
            parts, owners = self._cut(todo)
            encs = self._backend.encode_batch(parts, add_special_tokens=False)
            for piece in todo:
                found[piece] = []
            for owner, enc in zip(owners, encs):
                found[todo[owner]].extend(map(self._ints.__getitem__, enc.ids))
            with self._lock:
                for piece in todo:
                    if len(piece) >= _MIN_CACHED_CHARS:
                        self._insert(piece, found[piece])

        out: list[int] = []
        ids_through_token: list[int] = []
        for is_added, piece in pieces:
            if is_added:
                out.append(self._added_id[piece])
                ids_through_token.append(len(out))
            else:
                out.extend(found[piece])
        return TokenizedPrompt(text, out, token_starts, token_ends, ids_through_token)

    def _cut(self, pieces: list[str]) -> tuple[list[str], list[int]]:
        """The parts to encode and the index of the piece each part belongs to.
        Parts of one piece are in order; a piece shorter than 2 parts stays whole."""
        size = self._part_chars
        parts: list[str] = []
        owners: list[int] = []
        for i, piece in enumerate(pieces):
            start = 0
            while size and len(piece) - start >= 2 * size:
                m = _CUT.search(piece, start + size)
                if m is None:
                    break
                parts.append(piece[start : m.start() + 1])
                owners.append(i)
                start = m.start() + 1
            parts.append(piece[start:])
            owners.append(i)
        return parts, owners

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


class TokenizerMadeIds:
    """Prompt id lists built from the tokenizer's own ids, remembered by
    identity until vLLM's vocabulary check takes them.

    Each entry holds its list, so its id is never reused by another list while
    the entry exists. Lists whose check never comes (a ``/tokenize`` request,
    a prompt refused before the check) are dropped oldest first once the
    remembered lists hold more than ``max_tokens`` ids; a dropped list is
    simply checked.
    """

    def __init__(self, max_tokens: int = 1 << 22):
        self._max_tokens = max_tokens
        self._tokens = 0
        self._lists: OrderedDict[int, tuple[list[int], int]] = OrderedDict()
        self._lock = threading.Lock()

    def add(self, ids: list[int]) -> None:
        with self._lock:
            old = self._lists.pop(id(ids), None)
            if old is not None:
                self._tokens -= old[1]
            self._lists[id(ids)] = (ids, len(ids))
            self._tokens += len(ids)
            while self._tokens > self._max_tokens and len(self._lists) > 1:
                _, (_, length) = self._lists.popitem(last=False)
                self._tokens -= length

    def take(self, ids: Any) -> bool:
        """Whether ``ids`` is a remembered list of its remembered length; forgets it."""
        with self._lock:
            entry = self._lists.pop(id(ids), None)
            if entry is not None:
                self._tokens -= entry[1]
        return entry is not None and entry[0] is ids and entry[1] == len(ids)

    def __len__(self) -> int:
        return len(self._lists)


TOKENIZER_MADE_IDS = TokenizerMadeIds()


# Keyed by id(tokenizer); each entry holds the tokenizer, so its id is never
# reused by another object while the entry exists.
_INSTANCES: dict[int, tuple[Any, SegmentTokenizer | None]] = {}
_INSTANCES_LOCK = threading.Lock()


def get_segment_tokenizer(
    tokenizer: Any,
    max_cached_tokens: int,
    max_prompts: int = 0,
    packed: bool = False,
    part_chars: int = 0,
):
    """One ``SegmentTokenizer`` per vLLM tokenizer object, or None if ineligible."""
    with _INSTANCES_LOCK:
        if id(tokenizer) in _INSTANCES:
            return _INSTANCES[id(tokenizer)][1]
        seg = None
        backend = _backend_of(tokenizer)
        if backend is None:
            reason = "no HF tokenizers backend"
        elif getattr(tokenizer, "split_special_tokens", False):
            reason = "split_special_tokens is set"
        else:
            reason = SegmentTokenizer.eligible(backend)
        if reason is None:
            seg = SegmentTokenizer(
                backend, max_cached_tokens, max_prompts, packed, part_chars
            )
            logger.info(
                "VLLM_TPU_SEGMENT_TOKENIZE: piecewise prompt tokenization on "
                "(%d added tokens, cache %d tokens, ~12 B/token, prefix reuse "
                "over %d recent prompts).",
                len(seg._added_id),
                max_cached_tokens,
                max_prompts,
            )
        else:
            logger.warning(
                "VLLM_TPU_SEGMENT_TOKENIZE: falling back to full tokenization: %s",
                reason,
            )
        _INSTANCES[id(tokenizer)] = (tokenizer, seg)
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
    max_prompts = envs.VLLM_TPU_SEGMENT_TOKENIZE_PREFIX_PROMPTS
    packed_ids = envs.VLLM_TPU_PACKED_PROMPT_IDS
    part_chars = envs.VLLM_TPU_SEGMENT_TOKENIZE_PART_CHARS

    def _tokenize_prompt(self, prompt, params):
        kwargs = params.get_encode_kwargs()
        if kwargs.get("add_special_tokens") or self._wants_offsets(prompt, params):
            return orig(self, prompt, params)
        seg = get_segment_tokenizer(
            self.get_tokenizer(), max_cached, max_prompts, packed_ids, part_chars
        )
        if seg is None:
            return orig(self, prompt, params)
        tokenized, _ = seg.encode_prompt(prompt["prompt"])
        ids = tokenized.ids
        max_length = kwargs.get("max_length")
        if (
            kwargs.get("truncation")
            and max_length is not None
            and len(ids) > max_length
        ):
            # Over-long prompt: keep the tokenizer's own truncation semantics.
            return orig(self, prompt, params)
        tokens_prompt = self._build_tokens_prompt(ids, prompt, offset_mapping=None)
        TOKENIZER_MADE_IDS.add(tokens_prompt["prompt_token_ids"])
        if tokenized.packed is not None:
            API_PACKED.put(tokens_prompt["prompt_token_ids"], tokenized.packed)
        return tokens_prompt

    cls._tokenize_prompt = _tokenize_prompt
    _patch_validate_token_ids(cls)
    cls._tpu_segment_tokenize = True
    logger.info(
        "VLLM_TPU_SEGMENT_TOKENIZE: patched BaseRenderer._tokenize_prompt "
        "(prefix reuse over %d recent prompts; vocabulary check skipped for "
        "its ids)",
        max_prompts,
    )


def _patch_validate_token_ids(cls: Any) -> None:
    orig = cls.validate_token_ids

    def validate_token_ids(self, token_ids, *, parameter=None):
        if TOKENIZER_MADE_IDS.take(token_ids):
            return
        return orig(self, token_ids, parameter=parameter)

    cls.validate_token_ids = validate_token_ids
