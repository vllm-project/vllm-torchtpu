# SPDX-License-Identifier: Apache-2.0
"""Start each request's output detokenizer from the prompt's last tokens.

Opt-in with ``VLLM_TPU_DETOK_SUFFIX_PRIME=1``.

Why: vLLM's ``FastIncrementalDetokenizer`` primes its tokenizer's
``DecodeStream`` with every prompt token. The stream's first step then decodes
the whole prompt twice (once alone, once with the new token) to find where the
new token's text starts: about 50 ms for a 150k-token agent prompt on one core.
It runs in the API server's output handler, holding the GIL, before the
request's first chunk is sent and while every other stream's outputs wait.

Why it is exact: a step returns ``decode(prefix + new)[len(decode(prefix)):]``
and, once it has returned text, keeps only the ids after the prefix. A
byte-level decode of tokens that end on a whole character is the
concatenation of their texts, so any prompt suffix that decodes to whole
characters gives the same new text as the full prompt. The stream is primed
with the shortest suffix of 4 to 24 prompt tokens whose text holds no
replacement character, as vLLM did before native prefill; a prompt without one
(it ends inside a character) keeps the full prime.
"""

from __future__ import annotations

import msgspec

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

SHORTEST_SUFFIX = 4
LONGEST_SUFFIX = 24
REPLACEMENT_CHARACTER = "�"


def prompt_suffix(tokenizer, prompt_token_ids: list[int] | None) -> list[int] | None:
    """The shortest prompt suffix of 4 to 24 tokens that decodes to whole characters, else the prompt."""
    if prompt_token_ids is None or len(prompt_token_ids) <= SHORTEST_SUFFIX:
        return prompt_token_ids
    for length in range(SHORTEST_SUFFIX, min(len(prompt_token_ids), LONGEST_SUFFIX) + 1):
        suffix = prompt_token_ids[-length:]
        if REPLACEMENT_CHARACTER not in tokenizer.decode(suffix, skip_special_tokens=False):
            return suffix
    return prompt_token_ids


def patch_detokenizer_suffix_prime() -> None:
    """Prime ``FastIncrementalDetokenizer`` with a prompt suffix (VLLM_TPU_DETOK_SUFFIX_PRIME)."""
    from vllm_torchtpu import envs

    if not envs.VLLM_TPU_DETOK_SUFFIX_PRIME:
        return
    from vllm.v1.engine.detokenizer import FastIncrementalDetokenizer

    cls = FastIncrementalDetokenizer
    if cls.__dict__.get("_tpu_detok_suffix_prime", False):
        return
    full_prime_init = cls.__init__

    def __init__(self, tokenizer, request) -> None:
        suffix = prompt_suffix(tokenizer._tokenizer, request.prompt_token_ids)
        full_prime_init(self, tokenizer, msgspec.structs.replace(request, prompt_token_ids=suffix))

    cls.__init__ = __init__
    cls._tpu_detok_suffix_prime = True
    logger.info(
        "VLLM_TPU_DETOK_SUFFIX_PRIME: output detokenizers start from the prompt's last %d-%d tokens",
        SHORTEST_SUFFIX,
        LONGEST_SUFFIX,
    )
