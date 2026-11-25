# SPDX-License-Identifier: Apache-2.0
"""Counts a finished stream's reasoning tokens without rebuilding its deltas.

Opt-in with ``VLLM_TPU_REASONING_COUNT_FAST=1`` (API server only).

With a reasoning parser configured, vLLM's chat streaming calls
``parser.count_reasoning_tokens(<all generated ids>)`` after a request's last
output, on the API server's event loop, to fill the usage chunk. A
parser-engine reasoning parser (Qwen3's ``qwen3`` among them) answers from its
streaming state only when the stream went through a reasoning phase. When
reasoning is over before the first generated token (the prompt closes it, or
``enable_thinking=False``, where Qwen3's ``is_reasoning_end`` is always true),
the stream never enters one, and
``ParserEngineReasoningAdapter.count_reasoning_tokens`` re-parses the whole
output with ``ParserEngine._single_pass_parse``.

That single pass also builds content and tool-call deltas. Qwen3's argument
converter re-reads every argument character accumulated so far at each lexer
chunk (chunks split at ``>`` and at tag terminals), so an output holding a long
tool call costs time quadratic in its length. Measured in served Qwen3.8-Flash
runs: 4 to 43 s for 3k to 60k-token outputs, during which the event loop
serves no other request.

The number returned is the counting engine's ``reasoning_token_count``: the
token counts of the REASONING_CHUNK events its streaming engine emits in
``feed`` and ``finish``. The deltas and tool-call results of the single pass
are discarded. ``count_reasoning_tokens_fast`` returns the same number two
ways:

- When no state that emits REASONING_CHUNK is reachable from the engine's
  initial state, the count is 0 for any output, and no parsing runs. The
  streaming engine changes state only through its config's transitions, or to
  CONTENT, so reachability over those decides it (Qwen3 with thinking off
  starts in CONTENT, and no transition leads back into REASONING).
- Otherwise it runs the same reset, feed and finish on the same counting
  engine and reads the same counter, without the delta and tool-call
  bookkeeping.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_PATCH_FLAG = "_tpu_reasoning_count_fast"
# Set on a counting engine: whether its config can emit REASONING_CHUNK.
_REACHABLE_ATTR = "_tpu_reasoning_reachable"


def reasoning_reachable(config: Any) -> bool:
    """Whether a streaming engine on ``config`` can emit REASONING_CHUNK."""
    from vllm.parser.engine.events import EventType
    from vllm.parser.engine.parser_engine_config import ParserState

    emitting = {
        state
        for state, event in config.content_events.items()
        if event == EventType.REASONING_CHUNK
    }
    seen = {config.initial_state, ParserState.CONTENT}
    todo = list(seen)
    while todo:
        state = todo.pop()
        for (src, _terminal), transition in config.transitions.items():
            if src == state and transition.next_state not in seen:
                seen.add(transition.next_state)
                todo.append(transition.next_state)
    return not emitting.isdisjoint(seen)


def count_reasoning_tokens_fast(adapter: Any, token_ids: Sequence[int], orig) -> int:
    """Returns ``orig(adapter, token_ids)``, skipping the deltas it would build."""
    from vllm.parser.engine.parser_engine import ParserEngine

    if adapter._streaming_count_valid or not token_ids:
        return orig(adapter, token_ids)
    engine = adapter._counting_parser_engine
    if engine is None:
        engine = adapter._parser_engine_cls(
            adapter.model_tokenizer, **adapter._parser_engine_kwargs
        )
        adapter._counting_parser_engine = engine
    if type(engine)._single_pass_parse is not ParserEngine._single_pass_parse:
        # A parser engine with its own single pass may count differently.
        return orig(adapter, token_ids)
    reachable = engine.__dict__.get(_REACHABLE_ATTR)
    if reachable is None:
        reachable = reasoning_reachable(engine.parser_engine_config)
        setattr(engine, _REACHABLE_ATTR, reachable)
    if not reachable:
        return 0
    # The first three steps of ParserEngine._single_pass_parse.
    engine._reset()
    engine._feed(adapter.model_tokenizer.decode(token_ids), token_ids)
    engine._engine.finish()
    return engine.count_reasoning_tokens(token_ids)


def patch_reasoning_count_fast() -> None:
    """Routes ``ParserEngineReasoningAdapter.count_reasoning_tokens`` through
    ``count_reasoning_tokens_fast``."""
    from vllm_torchtpu import envs

    if not envs.VLLM_TPU_REASONING_COUNT_FAST:
        return
    from vllm.parser.engine.adapters import ParserEngineReasoningAdapter

    cls = ParserEngineReasoningAdapter
    if getattr(cls.count_reasoning_tokens, _PATCH_FLAG, False):
        return
    orig = cls.count_reasoning_tokens

    def count_reasoning_tokens(self, token_ids):
        return count_reasoning_tokens_fast(self, token_ids, orig)

    setattr(count_reasoning_tokens, _PATCH_FLAG, True)
    count_reasoning_tokens.__wrapped__ = orig
    cls.count_reasoning_tokens = count_reasoning_tokens
    logger.info(
        "VLLM_TPU_REASONING_COUNT_FAST: reasoning-token counts skip the delta rebuild"
    )
