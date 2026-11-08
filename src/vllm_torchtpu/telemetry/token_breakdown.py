# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""Seven-category input token attribution and bounded two-tier session tracking."""

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")

CATEGORIES: tuple[str, ...] = (
    "cache_hit",
    "cross_session_opportunity",
    "affinity_honored_eviction",
    "routing_loss",
    "prefix_mutation_loss",
    "unreused_reply_loss",
    "new_input",
)


class BoundedLRU(Generic[K, V]):
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

    def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)


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
    tokens: tuple[int, ...]
    reply_start_idx: int = 0


@dataclass(frozen=True, slots=True)
class _Tier2State:
    turn_id: int
    seq_len: int
    block_hashes: tuple[int, ...]
    reply_start_idx: int = 0
    tail_hash: int = 0
    prior_im_ends: int = 0


def _compute_tail_hash(tokens: Sequence[int], block_size: int) -> int:
    """Hash any trailing partial block `[num_blocks * block_size : len(tokens)]`."""
    rem_start = (len(tokens) // block_size) * block_size
    if rem_start >= len(tokens):
        return 0
    return hash(tuple(tokens[rem_start:]))


_IM_END_TOKEN_ID = 151645


def _detect_reply_start_idx(prompt_token_ids: Sequence[int]) -> int:
    """Return the token index where the assistant reply / thinking suffix starts.

    Chat templates such as Qwen3 (`enable_thinking=False`) append a 4-token
    empty thinking block (`<think>\\n\\n</think>\\n\\n`) at the tail of
    `prompt_token_ids` that is stripped when the turn is re-serialized into
    history on the next turn. Anchoring the reply boundary at that suffix
    attributes template-stripped thinking blocks and uncached previous replies
    to `unreused_reply_loss` rather than `prefix_mutation_loss`.
    """
    p_len = len(prompt_token_ids)
    if (
        p_len >= 4
        and prompt_token_ids[p_len - 3] == prompt_token_ids[p_len - 1]
        and prompt_token_ids[p_len - 4] != prompt_token_ids[p_len - 2]
    ):
        return p_len - 4
    return p_len


def _find_reply_end_idx(
    prompt_token_ids: Sequence[int],
    search_start: int,
    fallback_end: int,
    prior_im_ends: int = 0,
) -> int:
    """Return the token index just after the prior assistant reply's `<|im_end|>`.

    When prior reasoning tokens are dropped before the assistant reply is
    carried back into the next turn's prompt, the carried-back reply in
    `prompt_token_ids` ends earlier than `prior.tokens` at `<|im_end|>`
    (`151645`). Stopping `unreused_reply_loss` (or `prefix_mutation_loss` on an
    edited turn) at that `<|im_end|>` prevents subsequent new user/tool tokens
    from being counted as unreused reply or mutation.
    """
    total = len(prompt_token_ids)
    start = min(max(0, search_start), total)
    seen_im_ends = prompt_token_ids[:start].count(_IM_END_TOKEN_ID)
    for idx in range(start, total):
        if prompt_token_ids[idx] == _IM_END_TOKEN_ID:
            seen_im_ends += 1
            if seen_im_ends > prior_im_ends:
                return idx + 1
    return fallback_end


def compute_rolling_block_hashes(
    tokens: Sequence[int], block_size: int
) -> tuple[int, ...]:
    """Compute chained block hashes over full blocks of `tokens`."""
    num_blocks = len(tokens) // block_size
    if num_blocks <= 0:
        return ()
    hashes: list[int] = []
    prev_hash = 0
    for blk_idx in range(num_blocks):
        start = blk_idx * block_size
        chunk = tuple(tokens[start : start + block_size])
        prev_hash = hash((prev_hash, chunk))
        hashes.append(prev_hash)
    return tuple(hashes)


class SessionTurnTracker:
    """Two-tier session state tracker and six-category prefill classifier.

    Tier 1 retains exact token IDs for hot sessions; sessions evicted from
    Tier 1 demote into Tier 2 as compact chained block hashes.
    """

    def __init__(
        self,
        *,
        block_size: int = 16,
        tier1_maxsize: int = 4096,
        tier2_maxsize: int = 131072,
        cross_session_maxsize: int = 65536,
    ) -> None:
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}")
        self.block_size = block_size
        self.tier1: BoundedLRU[str, _Tier1State] = BoundedLRU(tier1_maxsize)
        self.tier2: BoundedLRU[str, _Tier2State] = BoundedLRU(tier2_maxsize)
        self.cross_session: BoundedLRU[int, str] = BoundedLRU(cross_session_maxsize)

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
        """Classify `prompt_token_ids` into the 6 mutually exclusive categories."""
        total = len(prompt_token_ids)
        hit = min(max(int(num_cached_tokens), 0), total)
        if total == 0:
            return TokenBreakdown()

        t1, t2, used_parent = self._lookup_prior(session_id, parent_session_id)
        if t1 is not None:
            return self._classify_from_tier1(
                prompt_token_ids, hit, t1, used_parent or parent_session_id
            )
        if t2 is not None:
            return self._classify_from_tier2(
                prompt_token_ids, hit, t2, used_parent or parent_session_id
            )

        # No prior state on this replica for session_id or parent_session_id.
        if session_id and turn_id is not None and turn_id > 1:
            im_end_positions = [
                idx
                for idx, tok in enumerate(prompt_token_ids)
                if tok == _IM_END_TOKEN_ID
            ]
            if len(im_end_positions) >= 3:
                last_reply_end = im_end_positions[-2] + 1
                if hit < last_reply_end <= total:
                    return TokenBreakdown(
                        cache_hit=hit,
                        routing_loss=last_reply_end - hit,
                        new_input=total - last_reply_end,
                        parent_session_id=parent_session_id,
                    )
            return TokenBreakdown(
                cache_hit=hit,
                routing_loss=total - hit,
                parent_session_id=parent_session_id,
            )

        # Turn 1 (or stateless request): check cross-session shared prefix blocks.
        cross_opp = self._count_cross_session_tokens(
            prompt_token_ids, hit, session_id
        )
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
    ) -> TokenBreakdown:
        total = len(prompt_token_ids)
        expected_prefix = max(hit, min(len(prior.tokens), total))
        new_input = total - expected_prefix

        diverge_idx: int | None = None
        limit = min(len(prior.tokens), total)
        for idx in range(hit, limit):
            if prompt_token_ids[idx] != prior.tokens[idx]:
                diverge_idx = idx
                break

        if diverge_idx is None:
            return TokenBreakdown(
                cache_hit=hit,
                affinity_honored_eviction=expected_prefix - hit,
                new_input=new_input,
                parent_session_id=parent_session_id,
            )

        block_base = max(hit, (diverge_idx // self.block_size) * self.block_size)
        reply_boundary = (
            prior.reply_start_idx if prior.reply_start_idx > 0 else len(prior.tokens)
        )
        prior_im_ends = prior.tokens[: prior.reply_start_idx].count(_IM_END_TOKEN_ID)
        if diverge_idx >= reply_boundary or 0 < prior.reply_start_idx < total:
            search_start = max(
                diverge_idx,
                prior.reply_start_idx
                if 0 < prior.reply_start_idx < total
                else diverge_idx,
            )
            reply_end = max(
                block_base,
                _find_reply_end_idx(
                    prompt_token_ids, search_start, expected_prefix, prior_im_ends
                ),
            )
        else:
            reply_end = expected_prefix

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

        cur_hashes = compute_rolling_block_hashes(prompt_token_ids, self.block_size)
        start_blk = hit // self.block_size
        max_blk = min(len(cur_hashes), len(prior.block_hashes))

        diverge_idx: int | None = None
        for blk_idx in range(start_blk, max_blk):
            if cur_hashes[blk_idx] != prior.block_hashes[blk_idx]:
                diverge_idx = max(hit, blk_idx * self.block_size)
                break

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
                diverge_idx = max(hit, max_blk * self.block_size)

        if diverge_idx is None:
            return TokenBreakdown(
                cache_hit=hit,
                affinity_honored_eviction=expected_prefix - hit,
                new_input=new_input,
                parent_session_id=parent_session_id,
            )

        block_base = max(hit, (diverge_idx // self.block_size) * self.block_size)
        reply_boundary = (
            (prior.reply_start_idx // self.block_size) * self.block_size
            if 0 < prior.reply_start_idx < prior.seq_len
            else prior.seq_len
        )
        if diverge_idx >= reply_boundary or 0 < prior.reply_start_idx < total:
            search_start = max(
                diverge_idx,
                prior.reply_start_idx
                if 0 < prior.reply_start_idx < total
                else diverge_idx,
            )
            reply_end = max(
                block_base,
                _find_reply_end_idx(
                    prompt_token_ids, search_start, expected_prefix, prior.prior_im_ends
                ),
            )
        else:
            reply_end = expected_prefix

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
        cur_hashes = compute_rolling_block_hashes(prompt_token_ids, self.block_size)
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

    def record_turn_completion(
        self,
        session_id: str | None,
        prompt_token_ids: Sequence[int],
        output_token_ids: Sequence[int] = (),
        *,
        turn_id: int = 1,
    ) -> None:
        """Record full sequence `(prompt + output)` upon turn completion."""
        full_tokens = (*prompt_token_ids, *output_token_ids)
        if not full_tokens:
            return

        owner_tag = session_id or "_stateless"
        for blk_hash in compute_rolling_block_hashes(full_tokens, self.block_size):
            self.cross_session.put(blk_hash, owner_tag)

        if not session_id:
            return

        reply_start_idx = _detect_reply_start_idx(prompt_token_ids)
        self.tier2.pop(session_id)
        evicted = self.tier1.put(
            session_id,
            _Tier1State(
                turn_id=turn_id,
                tokens=full_tokens,
                reply_start_idx=reply_start_idx,
            ),
        )
        if evicted is not None:
            evicted_sid, evicted_state = evicted
            self.tier2.put(
                evicted_sid,
                _Tier2State(
                    turn_id=evicted_state.turn_id,
                    seq_len=len(evicted_state.tokens),
                    block_hashes=compute_rolling_block_hashes(
                        evicted_state.tokens, self.block_size
                    ),
                    reply_start_idx=evicted_state.reply_start_idx,
                    tail_hash=_compute_tail_hash(
                        evicted_state.tokens, self.block_size
                    ),
                    prior_im_ends=evicted_state.tokens[
                        : evicted_state.reply_start_idx
                    ].count(_IM_END_TOKEN_ID),
                ),
            )
