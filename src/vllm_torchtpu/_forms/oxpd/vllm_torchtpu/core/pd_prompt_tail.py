# SPDX-License-Identifier: Apache-2.0
"""q38 ADM-P (engine-core half): ship only the prompt tail to PD decode workers.

On a KV-consumer engine (the PD decode) a newly scheduled request's KV arrives
by transfer, so its workers never compute the prompt positions below
``num_computed_tokens``. Yet ``NewRequestData`` carries the whole
``prompt_token_ids`` (100K+ Python ints), which the engine core pickles and
broadcasts to every worker on the admission step, and each worker unpickles
and copies (DEC-HOST-B's CPU microbench, 100K ints: 4.5 ms pickle, 7.9 ms
unpickle per worker, 9.1 ms copy).

With ``VLLM_TPU_PD_PROMPT_TAIL=W`` (W > 0) on a ``kv_consumer`` engine,
``NewRequestData.from_request`` replaces ``prompt_token_ids`` by
``PromptTail(length, start, tail)`` (DEC-HOST-B's
``vllm_torchtpu.runner.prompt_tail``), with ``start = num_computed_tokens - W``
and ``tail = prompt[start:]``. ``len()`` of a PromptTail is the full prompt
length and any read below ``start`` raises, so a reader that needs the whole
prompt fails loudly instead of reading wrong tokens.

The trim happens only when the request cannot need the earlier tokens: no
prompt embeddings, no ``prompt_is_token_ids``, no multimodal features, no
presence / frequency / repetition penalty, no prompt logprobs, no per-request
logits processors, ``start > 0``; W >= 16 (the worker's minimum window);
and only on an engine without custom logits processors and without the ngram
speculator (checked at install). Otherwise the full list is shipped as today.
The engine core's own ``Request`` keeps the full prompt (hashing, scheduling).
W = 0 (the default) installs nothing.
"""

from __future__ import annotations

from typing import Any

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_LOG_EVERY = 100
# The worker refuses a tail that starts closer than this to num_computed_tokens
# (runner/prompt_tail.py MIN_TOKENS_BELOW_COMPUTED), so a smaller W would make
# every trimmed admission raise; the engine core refuses such a W at install.
MIN_WINDOW = 16


class _Stats:
    trimmed = 0
    tokens = 0


def _prompt_tail_cls():
    from vllm_torchtpu.runner.prompt_tail import PromptTail

    return PromptTail


def eligible(nrd: Any) -> bool:
    """Whether this request's workers can run on the prompt tail alone."""
    if nrd.prompt_token_ids is None:
        return False
    if getattr(nrd, "prompt_embeds", None) is not None:
        return False
    if getattr(nrd, "prompt_is_token_ids", None) is not None:
        return False
    if getattr(nrd, "mm_features", None):
        return False
    sp = getattr(nrd, "sampling_params", None)
    if sp is None:
        return False
    if (
        getattr(sp, "presence_penalty", 0.0) != 0.0
        or getattr(sp, "frequency_penalty", 0.0) != 0.0
        or getattr(sp, "repetition_penalty", 1.0) != 1.0
        or getattr(sp, "prompt_logprobs", None) is not None
        or getattr(sp, "logits_processors", None)
    ):
        return False
    return True


def tail_new_request_data(nrd: Any, window: int, prompt_tail_cls=None) -> Any:
    """Replace one NewRequestData's prompt by its tail from num_computed - window."""
    if window <= 0 or not eligible(nrd):
        return nrd
    prompt = nrd.prompt_token_ids
    start = min(int(nrd.num_computed_tokens) - window, len(prompt))
    if start <= 0:
        return nrd
    cls = prompt_tail_cls or _prompt_tail_cls()
    nrd.prompt_token_ids = cls(len(prompt), start, list(prompt[start:]))
    _Stats.trimmed += 1
    _Stats.tokens += start
    if _Stats.trimmed % _LOG_EVERY == 0:
        logger.info(
            "PD prompt tail: trimmed %d requests, %d prompt tokens not shipped",
            _Stats.trimmed,
            _Stats.tokens,
        )
    return nrd


def _refusal(vllm_config: Any) -> str | None:
    kv = getattr(vllm_config, "kv_transfer_config", None)
    if kv is None or getattr(kv, "kv_role", None) != "kv_consumer":
        return "not a kv_consumer engine"
    mc = getattr(vllm_config, "model_config", None)
    if mc is not None and getattr(mc, "logits_processors", None):
        return "custom logits processors are configured"
    spec = getattr(vllm_config, "speculative_config", None)
    if spec is not None and getattr(spec, "method", None) == "ngram":
        return "the ngram speculator reads the prompt"
    try:
        _prompt_tail_cls()
    except ImportError:
        return "vllm_torchtpu.runner.prompt_tail (the worker half) is missing"
    return None


def patch_new_request_data_for_prompt_tail(vllm_config: Any) -> bool:
    """Wrap NewRequestData.from_request on an eligible KV-consumer engine core.
    Called from the engine-core process entry point; idempotent. Returns
    whether the wrapper is installed."""
    window = envs.VLLM_TPU_PD_PROMPT_TAIL
    if window <= 0:
        return False
    if window < MIN_WINDOW:
        logger.info("PD prompt tail: off (W < %d: VLLM_TPU_PD_PROMPT_TAIL=%d)", MIN_WINDOW, window)
        return False
    why = _refusal(vllm_config)
    if why is not None:
        logger.info("PD prompt tail: off (%s)", why)
        return False
    from vllm.v1.core.sched import output as sched_output

    cls = sched_output.NewRequestData
    if getattr(cls, "_q38_prompt_tail_patch", False):
        return True
    orig = cls.from_request.__func__
    tail_cls = _prompt_tail_cls()

    def from_request(klass, *args, **kwargs):
        return tail_new_request_data(orig(klass, *args, **kwargs), window, tail_cls)

    cls.from_request = classmethod(from_request)
    cls._q38_prompt_tail_patch = True
    logger.info(
        "PD prompt tail: on (VLLM_TPU_PD_PROMPT_TAIL=%d, kv_consumer): eligible new "
        "requests ship prompt tokens from num_computed_tokens - %d",
        window,
        window,
    )
    return True
