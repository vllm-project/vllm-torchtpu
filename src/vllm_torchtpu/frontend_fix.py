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
"""API-server patch that primes the detokenizer with a safe prompt tail.

Off by default and installed by ``additional_config["tpu_frontend"]``
(see ``frontend_options``):

* ``detok_prompt_tail``: primes ``FastIncrementalDetokenizer``'s Rust
  ``DecodeStream`` with a UTF-8 boundary-verified suffix of the last
  ``DETOK_PROMPT_TAIL_TOKENS`` prompt tokens instead of decoding the full prompt
  twice on the first emitted token.
"""

from __future__ import annotations

import multiprocessing
import os
import time

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Detokenizer prompt tail (tpu_frontend.detok_prompt_tail).
#
# vLLM's FastIncrementalDetokenizer primes ``tokenizers.decoders.DecodeStream``
# with the *whole* prompt.  The first ``step()`` then decodes the prompt twice
# (``prefix`` and ``prefix + new``) on the API server before the first token
# is emitted: 26 ms for a 105,596-token Qwen3.5 prompt, measured on one CPU.
# After that first emitted chunk the stream drops every prompt id
# (``ids.drain(prefix_index..)``), so the prompt only matters as left context
# for the first chunk.
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
#
# The other decoders of Hugging Face fast tokenizers (Metaspace, ByteFallback
# with Fuse and Strip, WordPiece) change only the start of the decoded text (a
# leading space or "##") or join neighbouring tokens, so the same holds for
# them; tests/test_frontend_detok_tail.py checks all four families.
# ---------------------------------------------------------------------------

_LOG_PREFIX = "tpu_frontend.detok_prompt_tail:"
_DETOK_INSTALLED = False
# Prompt tokens the detokenizer is primed with when the tail is safe.
DETOK_PROMPT_TAIL_TOKENS = 8
_DETOK_PROMPT_TAIL_LOG_S = 300.0


class _DetokStats:
    tail = 0
    full_fallback = 0
    short = 0
    error = 0
    last_log = 0.0


def safe_prompt_tail(tokenizer, prompt_ids, skip_special_tokens: bool, tail_n: int):
    """The last ``tail_n`` prompt ids when they prime a DecodeStream
    identically to the full prompt (they decode to non-empty text that does
    not end in U+FFFD), else ``prompt_ids`` itself.

    ``tokenizer`` is a ``tokenizers.Tokenizer`` (rust) instance.
    """
    if prompt_ids is None or tail_n <= 0 or len(prompt_ids) <= tail_n:
        return prompt_ids
    tail = list(prompt_ids[-tail_n:])
    text = tokenizer.decode(tail, skip_special_tokens=skip_special_tokens)
    if not text or text.endswith("\ufffd"):
        return prompt_ids
    return tail


class _PromptTailView:
    """Read-only view of an EngineCoreRequest with a shortened prompt.

    FastIncrementalDetokenizer.__init__ only reads attributes of the request;
    everything except ``prompt_token_ids`` is delegated to the real request.
    """

    __slots__ = ("_request", "prompt_token_ids")

    def __init__(self, req, prompt_token_ids):
        object.__setattr__(self, "_request", req)
        object.__setattr__(self, "prompt_token_ids", prompt_token_ids)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_request"), name)


def _tail_request(tokenizer, request, tail_n: int):
    """The request to prime vLLM's detokenizer with: a view whose prompt is the
    safe tail, or ``request`` itself when the prompt is short or has no safe
    tail."""
    ids = request.prompt_token_ids
    if ids is None or len(ids) <= tail_n:
        _DetokStats.short += 1
        return request
    skip_special_tokens = bool(request.sampling_params.skip_special_tokens)
    tail = safe_prompt_tail(tokenizer._tokenizer, ids, skip_special_tokens, tail_n)
    if tail is ids:
        _DetokStats.full_fallback += 1
        return request
    _DetokStats.tail += 1
    return _PromptTailView(request, tail)


def maybe_install_detok_prompt_tail() -> bool:
    global _DETOK_INSTALLED
    if _DETOK_INSTALLED:
        return False
    tail_n = DETOK_PROMPT_TAIL_TOKENS
    pname = multiprocessing.current_process().name
    # An API server is the main process, or a child vLLM's DP supervisor names
    # APIServer_DPRank_<n> (ApiServer_<n> in older vLLM).
    if not (pname == "MainProcess" or pname.lower().startswith("apiserver")):
        logger.warning(
            "%s not installed in process %r, which is not an API server",
            _LOG_PREFIX,
            pname,
        )
        return False
    try:
        import inspect

        from vllm.v1.engine import detokenizer as detok_mod

        cls = detok_mod.FastIncrementalDetokenizer
        src = inspect.getsource(cls.__init__)
    except Exception as e:
        logger.warning(
            "%s cannot import FastIncrementalDetokenizer (%r); not enabled",
            _LOG_PREFIX,
            e,
        )
        return False
    # Only patch the upstream shape we have proven equivalent: a DecodeStream
    # primed with ``ids=request.prompt_token_ids``.
    if "DecodeStream(" not in src or "ids=request.prompt_token_ids" not in src:
        logger.warning(
            "%s FastIncrementalDetokenizer.__init__ changed upstream; not enabled",
            _LOG_PREFIX,
        )
        return False

    orig_init = cls.__init__
    log_s = _DETOK_PROMPT_TAIL_LOG_S
    _DetokStats.last_log = time.monotonic()

    def __init__(self, tokenizer, request):
        primed = request
        try:
            primed = _tail_request(tokenizer, request, tail_n)
        except Exception as e:
            # vLLM's own priming with the full prompt is always correct, so a
            # request the tail cannot handle keeps it. The warning is keyed on
            # the error type alone (warning_once caches its arguments), so a
            # failure that repeats on every request logs once and keeps no
            # exception alive; each request's detail goes to debug.
            _DetokStats.error += 1
            logger.warning_once(
                "%s %s while choosing the prompt tail; priming with the full "
                "prompt (details at debug level)",
                _LOG_PREFIX,
                type(e).__name__,
            )
            logger.debug(
                "%s %s while choosing the prompt tail: %s",
                _LOG_PREFIX,
                type(e).__name__,
                e,
            )
        now = time.monotonic()
        if now - _DetokStats.last_log >= log_s:
            _DetokStats.last_log = now
            logger.debug(
                "%s stats tail=%d full_fallback=%d short=%d error=%d",
                _LOG_PREFIX,
                _DetokStats.tail,
                _DetokStats.full_fallback,
                _DetokStats.short,
                _DetokStats.error,
            )
        orig_init(self, tokenizer, primed)

    # pyrefly: ignore[missing-attribute]
    __init__.__wrapped__ = orig_init
    cls.__init__ = __init__
    _DETOK_INSTALLED = True
    logger.info(
        "%s installed FastIncrementalDetokenizer prompt-tail priming "
        "(tail=%d, pid %d, %s)",
        _LOG_PREFIX,
        tail_n,
        os.getpid(),
        pname,
    )
    return True
