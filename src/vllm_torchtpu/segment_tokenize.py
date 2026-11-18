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
also encoded across cores. With ``VLLM_TPU_SEGMENT_TOKENIZE_FASTOKENS=1``
that call goes to fastokens, which loads the same tokenizer file and returns
the same ids in a fraction of the time. Tokenizers that do not satisfy the
conditions above, or requests that need offsets, special-token insertion or
tokenizer truncation, take vLLM's original path unchanged.

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

Shared pieces (``VLLM_TPU_SEGMENT_TOKENIZE_SHARED_DIR``): the API server runs
several processes and a session's turns land on any of them, so a
per-process cache misses the history of most turns. With a directory set,
every encoded piece is also written there as one file named by a hash of its
text, holding the text and its ids; a process that misses its own cache reads
the file and uses its ids only when the stored text equals the piece. The
directory name carries a digest of the tokenizer, so only an identical
tokenizer reads the ids. Files are written to a temporary name and renamed,
so a reader sees a whole file or none.
"""

from __future__ import annotations

import bisect
import contextlib
import copy
import hashlib
import importlib.util
import os
import re
import struct
import sys
import threading
from array import array
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Pieces shorter than this are re-encoded every time (cheap) rather than
# cached, so the cache holds message bodies, not template glue.
_MIN_CACHED_CHARS = 64

_FIRST_COMPARED_CHARS = 4096
# Texts that differ in the last chars before a position cannot share a prefix
# up to it; comparing this many first skips most prompts of other conversations.
_AGREEING_CHARS = 64


class SharedPieceStore:
    """Encoded pieces in a directory that every process of the server reads and writes.

    File layout: magic, text byte count, the piece's UTF-8 text, then its ids
    as native uint32. A file is used only when its text equals the piece.
    """

    _MAGIC = b"SPv1"
    _HEADER = struct.Struct("=4sQ")

    def __init__(self, root: str, tokenizer_digest: str, max_written_tokens: int):
        self.dir = os.path.join(root, tokenizer_digest)
        os.makedirs(self.dir, exist_ok=True)
        self._max_written_tokens = max_written_tokens
        self._written: deque[tuple[str, int]] = deque()
        self._written_tokens = 0
        self._write_failed = False
        self._tmp_suffix = f".{os.getpid()}.tmp"

    def _path(self, text: bytes) -> str:
        return os.path.join(self.dir, hashlib.blake2b(text, digest_size=16).hexdigest())

    def get(self, text: bytes) -> array | None:
        try:
            with open(self._path(text), "rb") as f:
                data = f.read()
        except OSError:
            return None
        header = self._HEADER.size
        if len(data) < header:
            return None
        magic, text_len = self._HEADER.unpack_from(data)
        end = header + text_len
        if magic != self._MAGIC or (len(data) - end) % 4 or data[header:end] != text:
            return None
        ids = array("I")
        ids.frombytes(data[end:])
        return ids

    def put(self, text: bytes, ids: list[int]) -> None:
        if self._write_failed or len(ids) > self._max_written_tokens:
            return
        path = self._path(text)
        if os.path.exists(path):
            return
        tmp = path + self._tmp_suffix
        try:
            with open(tmp, "wb") as f:
                f.write(self._HEADER.pack(self._MAGIC, len(text)))
                f.write(text)
                f.write(array("I", ids).tobytes())
            os.replace(tmp, path)
        except OSError as e:
            self._write_failed = True
            logger.warning(
                "VLLM_TPU_SEGMENT_TOKENIZE: shared pieces read-only in this "
                "process after a failed write to %s: %s",
                self.dir,
                e,
            )
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            return
        self._written.append((path, len(ids)))
        self._written_tokens += len(ids)
        while self._written_tokens > self._max_written_tokens:
            old_path, old_tokens = self._written.popleft()
            self._written_tokens -= old_tokens
            with contextlib.suppress(OSError):
                os.unlink(old_path)


def tokenizer_digest(backend: Any) -> str:
    return hashlib.sha256(backend.to_str().encode()).hexdigest()[:16]


@dataclass
class TokenizedPrompt:
    """A prompt's ids and, per added token, where it starts and ends and the ids up to its end."""

    text: str
    ids: list[int]
    token_starts: list[int]
    token_ends: list[int]
    ids_through_token: list[int]


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
        """Whether the new prompt extends this prompt through its last added token, so it replaces it."""
        return self.tokens == len(self.prompt.token_ends)

    def join(self, rest: TokenizedPrompt, text: str) -> TokenizedPrompt:
        end, ids, tokens = self.end, self.ids, self.tokens
        prompt = self.prompt
        joined = prompt.ids[:ids]
        joined.extend(rest.ids)
        return TokenizedPrompt(
            text=text,
            ids=joined,
            token_starts=prompt.token_starts[:tokens] + [s + end for s in rest.token_starts],
            token_ends=prompt.token_ends[:tokens] + [e + end for e in rest.token_ends],
            ids_through_token=prompt.ids_through_token[:tokens]
            + [n + ids for n in rest.ids_through_token],
        )


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
    """The most recent tokenized prompts, searched for the longest prefix a new prompt can reuse."""

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
                best = self._longer(best, prompt, shared_prefix_length(text, prompt.text))
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
        return not any(text[p:end] in self._token_prefixes for p in range(first, start + 1))

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
        use_fastokens: bool = False,
        shared: SharedPieceStore | None = None,
    ):
        # Private copy: transformers mutates truncation/padding on the shared
        # backend per call, which must not leak into these encodes.
        self._backend = copy.deepcopy(backend)
        self._backend.no_truncation()
        self._backend.no_padding()
        self._miss_encoder = _fastokens_of(self._backend) if use_fastokens else self._backend
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
        self._shared = shared
        self.hits = 0
        self.shared_hits = 0
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
        return self.encode_reusing(text)[0]

    def encode_reusing(self, text: str) -> tuple[list[int], int]:
        """The ids of ``text`` and how many of them came from a recent prompt."""
        if self._recent is None:
            return self.encode_pieces(text).ids, 0
        with self._lock:
            prefix = self._recent.longest_prefix(text)
        if prefix is None:
            prompt = self.encode_pieces(text)
            reused = 0
        else:
            prompt = prefix.join(self.encode_pieces(text[prefix.end :]), text)
            reused = prefix.ids
        with self._lock:
            self._recent.add(prompt, replaces=prefix.prompt if prefix and prefix.whole else None)
        return prompt.ids, reused

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
                    continue
                if self._shared is not None and len(piece) >= _MIN_CACHED_CHARS:
                    stored = self._shared.get(piece.encode())
                    if stored is not None:
                        ids = list(map(self._ints.__getitem__, stored))
                        found[piece] = ids
                        self._insert(piece, ids)
                        self.shared_hits += 1
                        continue
                found[piece] = None  # type: ignore[assignment]
                todo.append(piece)
            self.misses += len(todo)
            if todo:
                encs = self._miss_encoder.encode_batch(todo, add_special_tokens=False)
                for piece, enc in zip(todo, encs):
                    ids = list(map(self._ints.__getitem__, enc.ids))
                    found[piece] = ids
                    if len(piece) >= _MIN_CACHED_CHARS:
                        self._insert(piece, ids)
                        if self._shared is not None:
                            self._shared.put(piece.encode(), enc.ids)

        out: list[int] = []
        ids_through_token: list[int] = []
        for is_added, piece in pieces:
            if is_added:
                out.append(self._added_id[piece])
                ids_through_token.append(len(out))
            else:
                out.extend(found[piece])
        return TokenizedPrompt(text, out, token_starts, token_ends, ids_through_token)

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


def _fastokens_of(backend: Any) -> Any:
    """A fastokens encoder of ``backend``'s tokenizer file, else ``backend`` itself."""
    try:
        return _import_fastokens().Tokenizer.from_json_str(backend.to_str())
    except Exception as exc:
        logger.warning("VLLM_TPU_SEGMENT_TOKENIZE_FASTOKENS: keeping the HF encoder: %s", exc)
        return backend


def _import_fastokens() -> Any:
    """The installed fastokens, else the copy a patch serves beside this package."""
    try:
        import fastokens

        return fastokens
    except ImportError:
        package = Path(__file__).resolve().parents[1] / "fastokens"
        if not (package / "__init__.py").is_file():
            raise
    spec = importlib.util.spec_from_file_location(
        "fastokens", package / "__init__.py", submodule_search_locations=[str(package)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["fastokens"] = module
    spec.loader.exec_module(module)
    return module


def _backend_of(tokenizer: Any) -> Any:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        backend = getattr(tokenizer, "_tokenizer", None)
    if backend is None or not hasattr(backend, "encode_batch"):
        return None
    return backend


# Tokens each process keeps written in the shared directory, beyond which its
# oldest writes are removed: ~8 B/token (piece text plus uint32 ids), so
# 128 MiB of /dev/shm per API server process.
_SHARED_WRITTEN_TOKENS = 1 << 24

_INSTANCES: dict[int, SegmentTokenizer | None] = {}
_INSTANCES_LOCK = threading.Lock()


def get_segment_tokenizer(
    tokenizer: Any,
    max_cached_tokens: int,
    max_prompts: int = 0,
    use_fastokens: bool = False,
    shared_dir: str = "",
):
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
            shared = None
            if shared_dir:
                shared = SharedPieceStore(
                    shared_dir, tokenizer_digest(backend), _SHARED_WRITTEN_TOKENS
                )
                logger.info(
                    "VLLM_TPU_SEGMENT_TOKENIZE: pid %d shares pieces in %s "
                    "(keeps up to %d tokens of its own writes).",
                    os.getpid(),
                    shared.dir,
                    _SHARED_WRITTEN_TOKENS,
                )
            seg = SegmentTokenizer(
                backend, max_cached_tokens, max_prompts, use_fastokens, shared
            )
            logger.info(
                "VLLM_TPU_SEGMENT_TOKENIZE: piecewise prompt tokenization on "
                "(%d added tokens, cache %d tokens, ~12 B/token, prefix reuse "
                "over %d recent prompts, missed pieces encoded by %s).",
                len(seg._added_id),
                max_cached_tokens,
                max_prompts,
                "HF tokenizers" if seg._miss_encoder is seg._backend else "fastokens",
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
    max_prompts = envs.VLLM_TPU_SEGMENT_TOKENIZE_PREFIX_PROMPTS
    use_fastokens = envs.VLLM_TPU_SEGMENT_TOKENIZE_FASTOKENS
    shared_dir = envs.VLLM_TPU_SEGMENT_TOKENIZE_SHARED_DIR

    def _tokenize_prompt(self, prompt, params):
        kwargs = params.get_encode_kwargs()
        if kwargs.get("add_special_tokens") or self._wants_offsets(prompt, params):
            return orig(self, prompt, params)
        seg = get_segment_tokenizer(
            self.get_tokenizer(),
            max_cached,
            max_prompts,
            use_fastokens,
            shared_dir,
        )
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
    logger.info(
        "VLLM_TPU_SEGMENT_TOKENIZE: patched BaseRenderer._tokenize_prompt "
        "(prefix reuse over %d recent prompts)",
        max_prompts,
    )
