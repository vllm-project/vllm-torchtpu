# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
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
"""Seven-category input token attribution and bounded two-tier session tracking."""

import itertools
import logging
from array import array
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("vllm." + __name__)

CATEGORIES: tuple[str, ...] = (
    "cache_hit",
    "cross_session_opportunity",
    "affinity_honored_eviction",
    "routing_loss",
    "prefix_mutation_loss",
    "unreused_reply_loss",
    "new_input",
)

_LOGGED_UNSUPPORTED_TOKENIZERS: set[str] = set()


class BoundedLRU[K, V]:
    """Fixed-capacity LRU mapping backed by OrderedDict."""

    def __init__(self, maxsize: int) -> None:
        if maxsize <= 0:
            raise ValueError(f"maxsize must be > 0, got {maxsize}")
        self.maxsize = maxsize
        self._data: OrderedDict[K, V] = OrderedDict()

    def get(self, key: K) -> V | None:
        val = self._data.get(key)
        if val is not None:
            self._data.move_to_end(key)
        return val

    def peek(self, key: K) -> V | None:
        return self._data.get(key)

    def put(self, key: K, value: V) -> tuple[K, V] | None:
        """Insert or update `key`, returning any evicted `(key, value)` pair."""
        if key in self._data:
            self._data.move_to_end(key)
            self._data[key] = value
            return None
        evicted: tuple[K, V] | None = None
        if len(self._data) >= self.maxsize:
            evicted = self._data.popitem(last=False)
        self._data[key] = value
        return evicted

    def pop(self, key: K) -> V | None:
        return self._data.pop(key, None)

    def pop_oldest(self) -> tuple[K, V] | None:
        """Pop and return the least-recently-used `(key, value)` pair, if any."""
        if not self._data:
            return None
        return self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()

    def items(self) -> list[tuple[K, V]]:
        return list(self._data.items())

    def recent_items(self, limit: int = 0) -> list[tuple[K, V]]:
        """Return up to `limit` most recent `(key, value)` pairs (`0` returns all)."""
        if limit <= 0 or limit >= len(self._data):
            return list(self._data.items())
        newest_first = list(itertools.islice(reversed(self._data.items()), limit))
        newest_first.reverse()
        return newest_first

    def __len__(self) -> int:
        return len(self._data)


@dataclass(frozen=True, slots=True)
class ChatTurnMarkers:
    """Tokenizer-resolved chat turn delimiter IDs for reply-boundary detection."""

    im_start_token_id: int | None = None
    im_end_token_id: int | None = None
    assistant_role_token_id: int | None = None
    think_start_token_id: int | None = None
    think_end_token_id: int | None = None


def _lookup_special_token_id(tokenizer: Any, token_str: str) -> int | None:
    """Resolve `token_str` to a single token ID on `tokenizer` without UNK fallback."""
    unk_id = getattr(tokenizer, "unk_token_id", None)
    vocab = None
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        try:
            vocab = get_vocab()
        except Exception:
            vocab = None
    if isinstance(vocab, dict) and token_str in vocab:
        tok_id = vocab[token_str]
        if isinstance(tok_id, int) and tok_id >= 0 and tok_id != unk_id:
            return tok_id

    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert):
        try:
            tok_id = convert(token_str)
            if isinstance(tok_id, int) and tok_id >= 0 and tok_id != unk_id:
                return tok_id
        except Exception:
            logger.debug("Skipping convert_tokens_to_ids for %s", token_str)
    return None


def _encode_single_token_id(tokenizer: Any, text: str) -> int | None:
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            ids = encode(text, add_special_tokens=False)
            if isinstance(ids, Sequence) and len(ids) == 1 and isinstance(ids[0], int):
                return int(ids[0])
        except Exception:
            logger.debug("Skipping single-token encode for %s", text)
    return _lookup_special_token_id(tokenizer, text)


def resolve_turn_markers_from_tokenizer(tokenizer: Any) -> ChatTurnMarkers:
    """Resolve turn boundary marker token IDs from a HuggingFace/vLLM tokenizer."""
    if tokenizer is None:
        return ChatTurnMarkers()

    think_start = _lookup_special_token_id(tokenizer, "<think>")
    think_end = _lookup_special_token_id(tokenizer, "</think>")

    # 1. ChatML (Qwen, Hermes, Yi, etc.): <|im_start|>assistant ... <|im_end|>
    im_start = _lookup_special_token_id(tokenizer, "<|im_start|>")
    im_end = _lookup_special_token_id(tokenizer, "<|im_end|>")
    if im_start is not None and im_end is not None:
        asst = _encode_single_token_id(tokenizer, "assistant")
        return ChatTurnMarkers(
            im_start_token_id=im_start,
            im_end_token_id=im_end,
            assistant_role_token_id=asst,
            think_start_token_id=think_start,
            think_end_token_id=think_end,
        )

    # 2. Llama 3 / 4: <|start_header_id|>assistant<|end_header_id|> ... <|eot_id|>
    hdr_start = _lookup_special_token_id(tokenizer, "<|start_header_id|>")
    eot_id = _lookup_special_token_id(tokenizer, "<|eot_id|>")
    if hdr_start is not None and eot_id is not None:
        asst = _encode_single_token_id(tokenizer, "assistant")
        return ChatTurnMarkers(
            im_start_token_id=hdr_start,
            im_end_token_id=eot_id,
            assistant_role_token_id=asst,
            think_start_token_id=think_start,
            think_end_token_id=think_end,
        )

    # 3. Gemma 2 / 3 / 4: <start_of_turn>model ... <end_of_turn>
    gemma_start = _lookup_special_token_id(tokenizer, "<start_of_turn>")
    gemma_end = _lookup_special_token_id(tokenizer, "<end_of_turn>")
    if gemma_start is not None and gemma_end is not None:
        model_role = _encode_single_token_id(tokenizer, "model")
        return ChatTurnMarkers(
            im_start_token_id=gemma_start,
            im_end_token_id=gemma_end,
            assistant_role_token_id=model_role,
            think_start_token_id=think_start,
            think_end_token_id=think_end,
        )

    # 4. DeepSeek: <｜Assistant｜> ... <｜end▁of▁sentence｜>
    ds_asst = _lookup_special_token_id(tokenizer, "<｜Assistant｜>")
    ds_eos = _lookup_special_token_id(tokenizer, "<｜end▁of▁sentence｜>")
    if ds_eos is not None:
        return ChatTurnMarkers(
            im_start_token_id=ds_asst,
            im_end_token_id=ds_eos,
            assistant_role_token_id=None,
            think_start_token_id=think_start,
            think_end_token_id=think_end,
        )

    tok_name = getattr(tokenizer, "name_or_path", None) or type(tokenizer).__name__
    if tok_name not in _LOGGED_UNSUPPORTED_TOKENIZERS:
        _LOGGED_UNSUPPORTED_TOKENIZERS.add(str(tok_name))
        logger.info(
            "Chat turn delimiter tokens not found on tokenizer %s; "
            "reply-boundary attribution heuristics are disabled.",
            tok_name,
        )
    return ChatTurnMarkers(
        think_start_token_id=think_start,
        think_end_token_id=think_end,
    )


@dataclass(frozen=True, slots=True)
class TokenBreakdown:
    """Mutually exclusive seven-category breakdown of prefill input tokens."""

    cache_hit: int = 0
    cross_session_opportunity: int = 0
    affinity_honored_eviction: int = 0
    routing_loss: int = 0
    prefix_mutation_loss: int = 0
    unreused_reply_loss: int = 0
    new_input: int = 0
    mutation_offset_token_idx: int | None = None
    parent_session_id: str | None = None

    def __post_init__(self) -> None:
        for name in CATEGORIES:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")

    @property
    def total_tokens(self) -> int:
        return (
            self.cache_hit
            + self.cross_session_opportunity
            + self.affinity_honored_eviction
            + self.routing_loss
            + self.prefix_mutation_loss
            + self.unreused_reply_loss
            + self.new_input
        )

    def as_category_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in CATEGORIES}

    def to_span_attributes(self) -> dict[str, Any]:
        attrs: dict[str, Any] = {
            f"gen_ai.usage.input_tokens.{name}": getattr(self, name)
            for name in CATEGORIES
        }
        if self.mutation_offset_token_idx is not None:
            attrs["gen_ai.usage.mutation_offset_token_idx"] = (
                self.mutation_offset_token_idx
            )
        if self.parent_session_id:
            attrs["gen_ai.session.parent_id"] = self.parent_session_id
        return attrs


@dataclass(frozen=True, slots=True)
class _Tier1State:
    turn_id: int
    tokens: array
    reply_start_idx: int = 0
    block_hashes: array = field(default_factory=lambda: array("q"))
    prior_im_ends: int = 0


@dataclass(frozen=True, slots=True)
class _Tier2State:
    turn_id: int
    seq_len: int
    block_hashes: array
    reply_start_idx: int = 0
    tail_hash: int = 0
    prior_im_ends: int = 0


def _compute_tail_hash(tokens: Sequence[int], block_size: int) -> int:
    """Hash any trailing partial block `[num_blocks * block_size : len(tokens)]`."""
    rem_start = (len(tokens) // block_size) * block_size
    if rem_start >= len(tokens):
        return 0
    return hash(tuple(tokens[rem_start:]))


def _find_last_assistant_reply_end(
    prompt_token_ids: Sequence[int],
    markers: ChatTurnMarkers,
) -> int | None:
    """Return the token index after the last completed assistant message."""
    im_start = markers.im_start_token_id
    im_end = markers.im_end_token_id
    asst_role = markers.assistant_role_token_id
    if im_start is None or im_end is None:
        return None

    total = len(prompt_token_ids)
    last_asst_end: int | None = None
    idx = 0
    if asst_role is None:
        # Single-token assistant turn marker (e.g. DeepSeek <｜Assistant｜>).
        while idx < total:
            if prompt_token_ids[idx] == im_start:
                end_idx = idx + 1
                while end_idx < total and prompt_token_ids[end_idx] != im_end:
                    end_idx += 1
                if end_idx < total:
                    last_asst_end = end_idx + 1
                    idx = end_idx + 1
                    continue
            idx += 1
        return last_asst_end

    while idx + 1 < total:
        if prompt_token_ids[idx] == im_start and prompt_token_ids[idx + 1] == asst_role:
            end_idx = idx + 2
            while end_idx < total and prompt_token_ids[end_idx] != im_end:
                end_idx += 1
            if end_idx < total:
                last_asst_end = end_idx + 1
                idx = end_idx + 1
                continue
        idx += 1
    return last_asst_end


def _find_server_hop_reply_end(
    prompt_token_ids: Sequence[int],
    markers: ChatTurnMarkers,
) -> int | None:
    """Return the token index after the last completed assistant message on a hop.

    Searches for the last assistant turn start marker closed by `im_end_token_id`
    so turns with multiple trailing tool/user messages after the assistant reply
    still split `routing_loss` vs `new_input` at the assistant reply boundary.
    When role start markers are absent, falls back to the second-to-last
    `im_end_token_id` only when at least three `im_end_token_id` markers exist.
    """
    if markers.im_end_token_id is None:
        return None
    last_asst_end = _find_last_assistant_reply_end(prompt_token_ids, markers)
    if last_asst_end is not None:
        return last_asst_end

    im_end = markers.im_end_token_id
    im_end_positions = [i for i, tok in enumerate(prompt_token_ids) if tok == im_end]
    if len(im_end_positions) >= 3:
        return im_end_positions[-2] + 1
    return None


def _detect_reply_start_idx(
    prompt_token_ids: Sequence[int],
    markers: ChatTurnMarkers,
) -> int:
    """Return the token index where the assistant reply or empty-think suffix starts.

    When `think_start_token_id` and `think_end_token_id` are configured on
    `markers`, detects a trailing 4-token `[<think>, ws, </think>, ws]` suffix
    appended by non-thinking chat templates (such as Qwen3 with
    `enable_thinking=False`) that is stripped when the turn is re-serialized on
    the next turn.
    """
    p_len = len(prompt_token_ids)
    t_start = markers.think_start_token_id
    t_end = markers.think_end_token_id
    if (
        p_len >= 4
        and t_start is not None
        and t_end is not None
        and prompt_token_ids[p_len - 4] == t_start
        and prompt_token_ids[p_len - 2] == t_end
        and prompt_token_ids[p_len - 3] == prompt_token_ids[p_len - 1]
    ):
        return p_len - 4
    return p_len


def _find_reply_end_idx(
    prompt_token_ids: Sequence[int],
    search_start: int,
    fallback_end: int,
    markers: ChatTurnMarkers,
    prior_im_ends: int = 0,
    matched_prefix_end: int = 0,
    *,
    edited_same_session: bool = False,
) -> int:
    """Return the token index just after the prior assistant reply's end marker."""
    im_end = markers.im_end_token_id
    if im_end is None:
        return fallback_end
    total = len(prompt_token_ids)
    start = min(max(0, search_start), total)
    if edited_same_session:
        asst_end = _find_last_assistant_reply_end(prompt_token_ids, markers)
        if asst_end is not None and asst_end > start:
            return asst_end
    if matched_prefix_end > 0 and start >= matched_prefix_end:
        seen_im_ends = prior_im_ends + prompt_token_ids[matched_prefix_end:start].count(
            im_end
        )
    else:
        seen_im_ends = prompt_token_ids[:start].count(im_end)
    for idx in range(start, total):
        if prompt_token_ids[idx] == im_end:
            seen_im_ends += 1
            if seen_im_ends > prior_im_ends:
                return idx + 1
    return fallback_end


def _compute_rolling_hashes_array(
    tokens: Sequence[int],
    block_size: int,
    *,
    prefix_hashes: Sequence[int] = (),
    start_block: int = 0,
) -> array:
    num_blocks = len(tokens) // block_size
    if num_blocks <= 0:
        return array("q")
    reuse = min(max(0, start_block), len(prefix_hashes), num_blocks)
    if reuse == num_blocks:
        if isinstance(prefix_hashes, array):
            return prefix_hashes[:num_blocks]
        return array("q", prefix_hashes[:num_blocks])
    hashes = array("q", prefix_hashes[:reuse])
    prev_hash = hashes[-1] if hashes else 0
    is_tuple = isinstance(tokens, tuple)
    for blk_idx in range(reuse, num_blocks):
        start = blk_idx * block_size
        chunk = (
            tokens[start : start + block_size]
            if is_tuple
            else tuple(tokens[start : start + block_size])
        )
        prev_hash = hash((prev_hash, chunk))
        hashes.append(prev_hash)
    return hashes


def compute_rolling_block_hashes(
    tokens: Sequence[int],
    block_size: int,
    *,
    prefix_hashes: Sequence[int] = (),
    start_block: int = 0,
) -> tuple[int, ...]:
    """Compute chained block hashes over full blocks, reusing `prefix_hashes`."""
    return tuple(
        _compute_rolling_hashes_array(
            tokens,
            block_size,
            prefix_hashes=prefix_hashes,
            start_block=start_block,
        )
    )


class SessionTurnTracker:
    """Two-tier session state tracker and seven-category prefill classifier.

    Tier 1 retains exact token IDs in compact 64-bit `array('q')` buffers (8 B
    per token) capped by both `tier1_maxsize` and `tier1_max_tokens`; sessions
    evicted from Tier 1 demote into Tier 2 as chained block hashes capped by
    both `tier2_maxsize` and `tier2_max_blocks`.
    """

    def __init__(
        self,
        *,
        block_size: int = 16,
        tier1_maxsize: int = 512,
        tier2_maxsize: int = 16384,
        cross_session_maxsize: int = 65536,
        tier1_max_tokens: int = 2_000_000,
        tier2_max_blocks: int = 2_000_000,
        turn_markers: ChatTurnMarkers | None = None,
    ) -> None:
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}")
        if tier1_max_tokens <= 0:
            raise ValueError(f"tier1_max_tokens must be > 0, got {tier1_max_tokens}")
        if tier2_max_blocks <= 0:
            raise ValueError(f"tier2_max_blocks must be > 0, got {tier2_max_blocks}")
        self.block_size = block_size
        self.tier1_max_tokens = tier1_max_tokens
        self.tier2_max_blocks = tier2_max_blocks
        self.turn_markers = turn_markers or ChatTurnMarkers()
        self._tokenizer_configured = turn_markers is not None
        self.tier1: BoundedLRU[str, _Tier1State] = BoundedLRU(tier1_maxsize)
        self.tier2: BoundedLRU[str, _Tier2State] = BoundedLRU(tier2_maxsize)
        self.cross_session: BoundedLRU[int, str] = BoundedLRU(cross_session_maxsize)
        self._verified_match_len: BoundedLRU[str, int] = BoundedLRU(tier1_maxsize)
        self._tier1_total_tokens: int = 0
        self._tier2_total_blocks: int = 0

    def configure_from_tokenizer(self, tokenizer: Any) -> None:
        """Resolve chat turn delimiter IDs once from `tokenizer` if not yet set."""
        if self._tokenizer_configured or tokenizer is None:
            return
        self.turn_markers = resolve_turn_markers_from_tokenizer(tokenizer)
        self._tokenizer_configured = True

    def discard_pending(self, session_id: str | None) -> None:
        """Drop stashed prefill match state for `session_id` on request abort."""
        if session_id:
            self._verified_match_len.pop(session_id)

    def _lookup_prior(
        self, session_id: str | None, parent_session_id: str | None
    ) -> tuple[_Tier1State | None, _Tier2State | None, str | None]:
        """Return `(tier1_state, tier2_state, resolved_parent_id)`."""
        if session_id:
            t1 = self.tier1.get(session_id)
            if t1 is not None:
                return t1, None, None
            t2 = self.tier2.get(session_id)
            if t2 is not None:
                return None, t2, None
        if parent_session_id:
            t1 = self.tier1.get(parent_session_id)
            if t1 is not None:
                return t1, None, parent_session_id
            t2 = self.tier2.get(parent_session_id)
            if t2 is not None:
                return None, t2, parent_session_id
        return None, None, None

    def classify_prefill(
        self,
        prompt_token_ids: Sequence[int],
        num_cached_tokens: int,
        *,
        session_id: str | None = None,
        turn_id: int | None = None,
        parent_session_id: str | None = None,
    ) -> TokenBreakdown:
        """Classify `prompt_token_ids` into the 7 mutually exclusive categories."""
        total = len(prompt_token_ids)
        hit = min(max(int(num_cached_tokens), 0), total)
        if total == 0:
            return TokenBreakdown()

        t1, t2, used_parent = self._lookup_prior(session_id, parent_session_id)
        prior_turn_id = (
            t1.turn_id if t1 is not None else (t2.turn_id if t2 is not None else None)
        )
        is_revisit_hop = (
            not used_parent
            and session_id is not None
            and turn_id is not None
            and prior_turn_id is not None
            and turn_id > prior_turn_id + 1
        )
        if is_revisit_hop:
            self._verified_match_len.pop(session_id)
        elif t1 is not None:
            return self._classify_from_tier1(
                prompt_token_ids,
                hit,
                t1,
                used_parent or parent_session_id,
                session_id=session_id if not used_parent else None,
            )
        elif t2 is not None:
            return self._classify_from_tier2(
                prompt_token_ids, hit, t2, used_parent or parent_session_id
            )

        # No prior state on this replica (or revisit hop after intermediate turns).
        if session_id and turn_id is not None and turn_id > 1:
            last_reply_end = _find_server_hop_reply_end(
                prompt_token_ids, self.turn_markers
            )
            if last_reply_end is not None and 0 < last_reply_end <= total:
                rloss = max(0, last_reply_end - hit)
                new_in = max(0, total - max(hit, last_reply_end))
                return TokenBreakdown(
                    cache_hit=hit,
                    routing_loss=rloss,
                    new_input=new_in,
                    parent_session_id=parent_session_id,
                )
            return TokenBreakdown(
                cache_hit=hit,
                routing_loss=total - hit,
                parent_session_id=parent_session_id,
            )

        # Turn 1 (or stateless request): check cross-session shared prefix blocks.
        cross_opp = self._count_cross_session_tokens(prompt_token_ids, hit, session_id)
        return TokenBreakdown(
            cache_hit=hit,
            cross_session_opportunity=cross_opp,
            new_input=total - hit - cross_opp,
            parent_session_id=parent_session_id,
        )

    def _classify_from_tier1(
        self,
        prompt_token_ids: Sequence[int],
        hit: int,
        prior: _Tier1State,
        parent_session_id: str | None,
        *,
        session_id: str | None = None,
    ) -> TokenBreakdown:
        total = len(prompt_token_ids)
        expected_prefix = max(hit, min(len(prior.tokens), total))
        new_input = total - expected_prefix

        diverge_idx: int | None = None
        limit = min(len(prior.tokens), total)
        prior_tokens = prior.tokens
        prompt_arr = (
            prompt_token_ids[:limit]
            if isinstance(prompt_token_ids, array)
            else array("q", prompt_token_ids[:limit])
        )
        if prompt_arr != prior_tokens[:limit]:
            bs = self.block_size
            for cur in range(0, limit, bs):
                nxt = min(cur + bs, limit)
                if prompt_arr[cur:nxt] != prior_tokens[cur:nxt]:
                    for idx in range(cur, nxt):
                        if prompt_arr[idx] != prior_tokens[idx]:
                            diverge_idx = idx
                            break
                    break

        if session_id:
            self._verified_match_len.put(
                session_id, diverge_idx if diverge_idx is not None else limit
            )

        if diverge_idx is None:
            return TokenBreakdown(
                cache_hit=hit,
                affinity_honored_eviction=expected_prefix - hit,
                new_input=new_input,
                parent_session_id=parent_session_id,
            )

        reply_boundary = (
            prior.reply_start_idx if prior.reply_start_idx > 0 else len(prior.tokens)
        )
        matched_prefix_end = (
            prior.reply_start_idx if diverge_idx >= prior.reply_start_idx > 0 else 0
        )
        return self._build_diverged_breakdown(
            prompt_token_ids,
            hit=hit,
            diverge_idx=diverge_idx,
            expected_prefix=expected_prefix,
            reply_boundary=reply_boundary,
            prior_im_ends=prior.prior_im_ends,
            parent_session_id=parent_session_id,
            matched_prefix_end=matched_prefix_end,
        )

    def _classify_from_tier2(
        self,
        prompt_token_ids: Sequence[int],
        hit: int,
        prior: _Tier2State,
        parent_session_id: str | None,
    ) -> TokenBreakdown:
        total = len(prompt_token_ids)
        expected_prefix = max(hit, min(prior.seq_len, total))
        new_input = total - expected_prefix

        cur_hashes = _compute_rolling_hashes_array(prompt_token_ids, self.block_size)
        max_blk = min(len(cur_hashes), len(prior.block_hashes))

        diverge_idx: int | None = None
        if cur_hashes[:max_blk] != prior.block_hashes[:max_blk]:
            for blk_idx in range(max_blk):
                if cur_hashes[blk_idx] != prior.block_hashes[blk_idx]:
                    diverge_idx = blk_idx * self.block_size
                    break

        im_end = self.turn_markers.im_end_token_id
        if (
            diverge_idx is None
            and prior.tail_hash != 0
            and total >= prior.seq_len
            and max_blk * self.block_size < prior.seq_len
        ):
            cur_tail = tuple(
                prompt_token_ids[max_blk * self.block_size : prior.seq_len]
            )
            if hash(cur_tail) != prior.tail_hash:
                diverge_idx = max_blk * self.block_size
        elif (
            diverge_idx is None
            and total < prior.seq_len
            and max_blk * self.block_size < total
            and (
                0 < prior.reply_start_idx < total
                or (
                    im_end is not None
                    and im_end in prompt_token_ids[max_blk * self.block_size : total]
                )
            )
        ):
            diverge_idx = max_blk * self.block_size

        if diverge_idx is None:
            return TokenBreakdown(
                cache_hit=hit,
                affinity_honored_eviction=expected_prefix - hit,
                new_input=new_input,
                parent_session_id=parent_session_id,
            )

        reply_boundary = (
            (prior.reply_start_idx // self.block_size) * self.block_size
            if 0 < prior.reply_start_idx < prior.seq_len
            else prior.seq_len
        )
        return self._build_diverged_breakdown(
            prompt_token_ids,
            hit=hit,
            diverge_idx=diverge_idx,
            expected_prefix=expected_prefix,
            reply_boundary=reply_boundary,
            prior_im_ends=prior.prior_im_ends,
            parent_session_id=parent_session_id,
        )

    def _build_diverged_breakdown(
        self,
        prompt_token_ids: Sequence[int],
        *,
        hit: int,
        diverge_idx: int,
        expected_prefix: int,
        reply_boundary: int,
        prior_im_ends: int,
        parent_session_id: str | None,
        matched_prefix_end: int = 0,
    ) -> TokenBreakdown:
        total = len(prompt_token_ids)
        block_base = max(hit, (diverge_idx // self.block_size) * self.block_size)
        reply_end = max(
            block_base,
            _find_reply_end_idx(
                prompt_token_ids,
                diverge_idx,
                expected_prefix,
                self.turn_markers,
                prior_im_ends,
                matched_prefix_end=matched_prefix_end,
                edited_same_session=(
                    diverge_idx < reply_boundary and not parent_session_id
                ),
            ),
        )
        if diverge_idx >= reply_boundary:
            return TokenBreakdown(
                cache_hit=hit,
                affinity_honored_eviction=block_base - hit,
                unreused_reply_loss=reply_end - block_base,
                new_input=total - reply_end,
                mutation_offset_token_idx=None,
                parent_session_id=parent_session_id,
            )
        return TokenBreakdown(
            cache_hit=hit,
            affinity_honored_eviction=block_base - hit,
            prefix_mutation_loss=reply_end - block_base,
            new_input=total - reply_end,
            mutation_offset_token_idx=diverge_idx,
            parent_session_id=parent_session_id,
        )

    def _count_cross_session_tokens(
        self,
        prompt_token_ids: Sequence[int],
        hit: int,
        session_id: str | None,
    ) -> int:
        cur_hashes = _compute_rolling_hashes_array(prompt_token_ids, self.block_size)
        start_blk = hit // self.block_size
        matched_tokens = 0
        for blk_idx in range(start_blk, len(cur_hashes)):
            owner = self.cross_session.get(cur_hashes[blk_idx])
            if owner is None or (session_id is not None and owner == session_id):
                break
            blk_start = max(hit, blk_idx * self.block_size)
            blk_end = (blk_idx + 1) * self.block_size
            if blk_end > blk_start:
                matched_tokens += blk_end - blk_start
        return matched_tokens

    def _put_tier2(self, session_id: str, state: _Tier2State) -> None:
        existing = self.tier2.peek(session_id)
        if existing is not None:
            self._tier2_total_blocks -= len(existing.block_hashes)
        evicted = self.tier2.put(session_id, state)
        if evicted is not None:
            self._tier2_total_blocks -= len(evicted[1].block_hashes)
        self._tier2_total_blocks += len(state.block_hashes)
        while len(self.tier2) > 1 and self._tier2_total_blocks > self.tier2_max_blocks:
            popped = self.tier2.pop_oldest()
            if popped is None:
                break
            self._tier2_total_blocks -= len(popped[1].block_hashes)

    def _demote_to_tier2(self, evicted_sid: str, evicted_state: _Tier1State) -> None:
        self._verified_match_len.pop(evicted_sid)
        evicted_hashes = (
            evicted_state.block_hashes
            if evicted_state.block_hashes
            else _compute_rolling_hashes_array(evicted_state.tokens, self.block_size)
        )
        self._put_tier2(
            evicted_sid,
            _Tier2State(
                turn_id=evicted_state.turn_id,
                seq_len=len(evicted_state.tokens),
                block_hashes=evicted_hashes,
                reply_start_idx=evicted_state.reply_start_idx,
                tail_hash=_compute_tail_hash(evicted_state.tokens, self.block_size),
                prior_im_ends=evicted_state.prior_im_ends,
            ),
        )

    def record_turn_completion(
        self,
        session_id: str | None,
        prompt_token_ids: Sequence[int],
        output_token_ids: Sequence[int] = (),
        *,
        turn_id: int = 1,
    ) -> tuple[tuple[int, ...], int]:
        """Record full sequence `(prompt + output)` upon turn completion.

        Returns `(full_block_hashes, reused_prefix_blocks)` so callers can avoid
        re-hashing or re-registering unchanged leading blocks.
        """
        full_tokens = array("q", prompt_token_ids)
        if output_token_ids:
            full_tokens.extend(output_token_ids)
        if not full_tokens:
            return (), 0

        bs = self.block_size
        reused_blocks = 0
        matched_len = self._verified_match_len.pop(session_id) if session_id else None
        prior_t1 = self.tier1.peek(session_id) if session_id else None
        if prior_t1 is not None and prior_t1.block_hashes:
            max_reuse = min(len(prior_t1.block_hashes), len(full_tokens) // bs)
            if matched_len is not None:
                reused_blocks = min(max_reuse, matched_len // bs)
            else:
                while reused_blocks < max_reuse:
                    s = reused_blocks * bs
                    if full_tokens[s : s + bs] != prior_t1.tokens[s : s + bs]:
                        break
                    reused_blocks += 1
            full_hashes_arr = _compute_rolling_hashes_array(
                full_tokens,
                bs,
                prefix_hashes=prior_t1.block_hashes,
                start_block=reused_blocks,
            )
        else:
            full_hashes_arr = _compute_rolling_hashes_array(full_tokens, bs)

        owner_tag = session_id or "_stateless"
        for blk_hash in full_hashes_arr[reused_blocks:]:
            self.cross_session.put(blk_hash, owner_tag)

        full_hashes = tuple(full_hashes_arr)
        if not session_id:
            return full_hashes, reused_blocks

        reply_start_idx = _detect_reply_start_idx(prompt_token_ids, self.turn_markers)
        verified_prefix = matched_len if matched_len is not None else reused_blocks * bs
        im_end = self.turn_markers.im_end_token_id
        if im_end is None:
            prior_im_ends = 0
        elif (
            prior_t1 is not None
            and reply_start_idx >= prior_t1.reply_start_idx
            and verified_prefix >= prior_t1.reply_start_idx
        ):
            prior_im_ends = prior_t1.prior_im_ends + full_tokens[
                prior_t1.reply_start_idx : reply_start_idx
            ].count(im_end)
        else:
            prior_im_ends = full_tokens[:reply_start_idx].count(im_end)

        popped_t2 = self.tier2.pop(session_id)
        if popped_t2 is not None:
            self._tier2_total_blocks -= len(popped_t2.block_hashes)

        if prior_t1 is not None:
            self._tier1_total_tokens -= len(prior_t1.tokens)

        new_state = _Tier1State(
            turn_id=turn_id,
            tokens=full_tokens,
            reply_start_idx=reply_start_idx,
            block_hashes=full_hashes_arr,
            prior_im_ends=prior_im_ends,
        )
        evicted = self.tier1.put(session_id, new_state)
        self._tier1_total_tokens += len(full_tokens)
        if evicted is not None:
            evicted_sid, evicted_state = evicted
            self._tier1_total_tokens -= len(evicted_state.tokens)
            self._demote_to_tier2(evicted_sid, evicted_state)

        while len(self.tier1) > 1 and self._tier1_total_tokens > self.tier1_max_tokens:
            oldest = self.tier1.pop_oldest()
            if oldest is None:
                break
            oldest_sid, oldest_state = oldest
            self._tier1_total_tokens -= len(oldest_state.tokens)
            self._demote_to_tier2(oldest_sid, oldest_state)

        return full_hashes, reused_blocks
