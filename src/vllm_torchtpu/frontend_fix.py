# SPDX-License-Identifier: Apache-2.0
"""API-server patches for reasoning-token counting and incremental detokenizer priming.

* ``QX_LAZY_REASONING_COUNT=1``: skips per-delta ``DelegatingParser.count_reasoning_tokens``
  recomputation on intermediate streaming chunks when ``continuous_usage_stats`` is
  disabled, and short-circuits to 0 when the ``StreamingParserEngine`` state machine
  has no reachable ``REASONING_CHUNK`` transition or state.
* ``QX_DETOK_PROMPT_TAIL=8``: primes ``FastIncrementalDetokenizer``'s Rust
  ``DecodeStream`` with a UTF-8 boundary-verified prompt suffix instead of decoding
  the full prompt twice on the first emitted token.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import time

logger = logging.getLogger("vllm.qx_fe_fix")
_P = "[QX_LAZY_REASONING_COUNT]"
_INSTALLED = False


class _Stats:
    skipped = 0
    zero = 0
    computed = 0
    compute_s = 0.0
    last_log = 0.0


_ZERO_CACHE: dict = {}


def _config_never_reasons(cfg) -> bool:
    """True iff the engine state machine can never emit REASONING_CHUNK.

    StreamingParserEngine only sets ``state`` to the config's initial state
    (reset), a transition's ``next_state``, or CONTENT; REASONING_CHUNK events
    come only from ``content_events[state]`` or ``transition.events``.  So if
    no state reachable from {initial, CONTENT} maps to REASONING_CHUNK and no
    reachable transition emits it, ``reasoning_token_count`` is always 0.
    """
    from vllm.parser.engine.events import EventType
    from vllm.parser.engine.parser_engine_config import ParserState

    rc = EventType.REASONING_CHUNK
    reach = {cfg.initial_state, ParserState.CONTENT}
    changed = True
    while changed:
        changed = False
        for (src, _term), tr in cfg.transitions.items():
            if src in reach:
                if rc in tr.events:
                    return False
                if tr.next_state not in reach:
                    reach.add(tr.next_state)
                    changed = True
    return all(cfg.content_events.get(s) != rc for s in reach)


def _provably_zero(rp) -> bool:
    try:
        pe = rp._parser_engine
        from vllm.parser.engine.parser_engine import ParserEngine
        if (type(pe).count_reasoning_tokens is not ParserEngine.count_reasoning_tokens
                or type(pe._engine).__name__ != "StreamingParserEngine"):
            return False
        cfg = pe.parser_engine_config
    except Exception:
        return False
    key = id(cfg)
    v = _ZERO_CACHE.get(key)
    if v is None or v[0] is not cfg:
        try:
            v = (cfg, _config_never_reasons(cfg))
        except Exception:
            v = (cfg, False)
        if len(_ZERO_CACHE) > 256:
            _ZERO_CACHE.clear()
        _ZERO_CACHE[key] = v
    return v[1]


def maybe_install_lazy_reasoning_count() -> bool:
    global _INSTALLED
    if _INSTALLED or os.environ.get("QX_LAZY_REASONING_COUNT", "1") != "1":
        return False
    pname = multiprocessing.current_process().name
    if not (pname == "MainProcess" or pname.startswith("ApiServer")):
        return False
    try:
        from vllm.parser.abstract_parser import DelegatingParser
        from vllm.parser.engine.adapters import ParserEngineReasoningAdapter
    except Exception as e:
        logger.warning("%s cannot import vLLM parser classes (%r); not enabled", _P, e)
        return False

    orig_parse_delta = DelegatingParser.parse_delta
    orig_count = DelegatingParser.count_reasoning_tokens
    log_s = float(os.environ.get("QX_LAZY_REASONING_COUNT_LOG_S", "60"))
    _Stats.last_log = time.monotonic()

    def parse_delta(self, *args, **kwargs):
        res = orig_parse_delta(self, *args, **kwargs)
        try:
            finished = kwargs.get("finished", True)
            request = kwargs.get("request", args[2] if len(args) > 2 else None)
            so = getattr(request, "stream_options", None)
            cont = bool(so is not None and getattr(so, "continuous_usage_stats", False))
            self._qx_skip_count = (finished is False) and not cont
        except Exception:
            self._qx_skip_count = False
        return res

    def _maybe_log():
        now = time.monotonic()
        if now - _Stats.last_log >= log_s:
            _Stats.last_log = now
            logger.warning("%s stats zero=%d skipped=%d computed=%d compute_s=%.2f", _P,
                           _Stats.zero, _Stats.skipped, _Stats.computed, _Stats.compute_s)

    def count_reasoning_tokens(self, token_ids):
        skip = False
        try:
            skip = self.__dict__.pop("_qx_skip_count", False)
            rp = getattr(self, "_reasoning_parser", None)
            if (isinstance(rp, ParserEngineReasoningAdapter)
                    and not getattr(rp, "_streaming_count_valid", True)):
                if _provably_zero(rp):
                    _Stats.zero += 1
                    _maybe_log()
                    return 0
            else:
                skip = False
        except Exception:
            skip = False
        if skip:
            _Stats.skipped += 1
            return self.__dict__.get("_qx_last_count", 0)
        t0 = time.perf_counter()
        n = orig_count(self, token_ids)
        _Stats.computed += 1
        _Stats.compute_s += time.perf_counter() - t0
        try:
            self._qx_last_count = n
            _maybe_log()
        except Exception:
            pass
        return n

    parse_delta.__qx_lazy__ = True
    count_reasoning_tokens.__qx_lazy__ = True
    DelegatingParser.parse_delta = parse_delta
    DelegatingParser.count_reasoning_tokens = count_reasoning_tokens
    _INSTALLED = True
    logger.warning("%s installed DelegatingParser.parse_delta/count_reasoning_tokens "
                   "patch (pid %d, %s)", _P, os.getpid(), pname)
    return True


# ---------------------------------------------------------------------------
# Detokenizer prompt tail (QX_DETOK_PROMPT_TAIL).
#
# vLLM's FastIncrementalDetokenizer primes ``tokenizers.decoders.DecodeStream``
# with the *whole* prompt.  The first ``step()`` then decodes the prompt twice
# (``prefix`` and ``prefix + new``) on the API-server event loop: ~27 ms at
# 94k prompt ids, directly in TTFT.  After that first emitted chunk the stream
# drops every prompt id (``ids.drain(prefix_index..)``), so the prompt only
# matters as left context for the first chunk.
#
# For byte-level decoders the decoded text is ``from_utf8_lossy(concat(bytes))``
# so the left context only matters through UTF-8 synchronization.  If
# ``decode(tail)`` is non-empty and does not end with U+FFFD, the tail ends on
# a character boundary; then ``decode(P + new)[len(decode(P)):]`` equals
# ``decode(T + new)[len(decode(T)):]`` for every continuation, the stream's
# emit/hold decisions are identical, and after the first emitted chunk both
# streams hold exactly ``[new ids]`` (same state).  When the check fails (prompt
# ends inside a multi-byte character, or the tail decodes to "" because it is
# all skipped special tokens) we keep the full prompt, i.e. upstream behaviour.
# ---------------------------------------------------------------------------

_DT = "[QX_DETOK_PROMPT_TAIL]"
_DETOK_INSTALLED = False


class _DetokStats:
    tail = 0
    full_fallback = 0
    short = 0
    last_log = 0.0


def safe_prompt_tail(tokenizer, prompt_ids, skip_special_tokens: bool, tail_n: int):
    """Shortest prompt suffix that primes a DecodeStream identically to the
    full prompt, or ``prompt_ids`` itself when no safe tail exists.

    ``tokenizer`` is a ``tokenizers.Tokenizer`` (rust) instance.
    """
    if prompt_ids is None or tail_n <= 0 or len(prompt_ids) <= tail_n:
        return prompt_ids
    tail = list(prompt_ids[-tail_n:])
    try:
        text = tokenizer.decode(tail, skip_special_tokens=skip_special_tokens)
    except Exception:
        return prompt_ids
    if not text or text.endswith("\ufffd"):
        return prompt_ids
    return tail


class _PromptTailView:
    """Read-only view of an EngineCoreRequest with a shortened prompt.

    FastIncrementalDetokenizer.__init__ only reads attributes of the request;
    everything except ``prompt_token_ids`` is delegated to the real request.
    """

    __slots__ = ("_qx_req", "prompt_token_ids")

    def __init__(self, req, prompt_token_ids):
        object.__setattr__(self, "_qx_req", req)
        object.__setattr__(self, "prompt_token_ids", prompt_token_ids)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_qx_req"), name)


def maybe_install_detok_prompt_tail() -> bool:
    global _DETOK_INSTALLED
    if _DETOK_INSTALLED:
        return False
    try:
        tail_n = int(os.environ.get("QX_DETOK_PROMPT_TAIL", "8"))
    except ValueError:
        tail_n = 8
    if tail_n <= 0:
        return False
    pname = multiprocessing.current_process().name
    if not (pname == "MainProcess" or pname.startswith("ApiServer")):
        return False
    try:
        import inspect

        from vllm.v1.engine import detokenizer as detok_mod
        cls = detok_mod.FastIncrementalDetokenizer
        src = inspect.getsource(cls.__init__)
    except Exception as e:
        logger.warning("%s cannot import FastIncrementalDetokenizer (%r); not enabled",
                       _DT, e)
        return False
    # Only patch the upstream shape we have proven equivalent: a DecodeStream
    # primed with ``ids=request.prompt_token_ids``.
    if "DecodeStream(" not in src or "ids=request.prompt_token_ids" not in src:
        logger.warning("%s FastIncrementalDetokenizer.__init__ changed upstream; "
                       "not enabled", _DT)
        return False

    orig_init = cls.__init__
    log_s = float(os.environ.get("QX_DETOK_PROMPT_TAIL_LOG_S", "300"))
    _DetokStats.last_log = time.monotonic()

    def __init__(self, tokenizer, request):
        try:
            ids = request.prompt_token_ids
            if ids is not None and len(ids) > tail_n:
                sp = request.sampling_params
                tail = safe_prompt_tail(
                    tokenizer._tokenizer, ids, bool(sp.skip_special_tokens), tail_n
                )
                if tail is not ids:
                    request = _PromptTailView(request, tail)
                    _DetokStats.tail += 1
                else:
                    _DetokStats.full_fallback += 1
            else:
                _DetokStats.short += 1
            now = time.monotonic()
            if now - _DetokStats.last_log >= log_s:
                _DetokStats.last_log = now
                logger.warning("%s stats tail=%d full_fallback=%d short=%d", _DT,
                               _DetokStats.tail, _DetokStats.full_fallback,
                               _DetokStats.short)
        except Exception:
            pass
        orig_init(self, tokenizer, request)

    __init__.__qx_detok_tail__ = True
    __init__.__wrapped__ = orig_init
    cls.__init__ = __init__
    _DETOK_INSTALLED = True
    logger.warning("%s installed FastIncrementalDetokenizer prompt-tail priming "
                   "(tail=%d, pid %d, %s)", _DT, tail_n, os.getpid(), pname)
    return True
