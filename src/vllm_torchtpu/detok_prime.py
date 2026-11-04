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
"""Prime vLLM's fast incremental detokenizer with a prompt tail, not the prompt.

Controlled by ``VLLM_TPU_DETOK_PRIME_TOKENS`` (0 = upstream behaviour).

Why: ``FastIncrementalDetokenizer`` builds ``DecodeStream(ids=prompt_token_ids)``.
The stream's first ``step()`` decodes the whole prompt twice (once to set its
prefix, once with the new token appended), on the API server's output loop, so
first-token latency grows ~27 ms per 100K prompt tokens on GLM-5.2.

Why a tail is exact: ``step()`` emits ``decode(ids)[len(prefix):]`` where
``prefix = decode(primed ids)``, and after its first emission it drops every
primed id, so the stream state is identical from then on. Until then the
emitted text depends only on the decoder's left context, which for byte-level,
byte-fallback and metaspace decoders is bounded by one UTF-8 character. The
stream treats two primed texts specially, and the tail is widened past both:
an empty text (the next token becomes sequence start, so a metaspace decoder
strips its space) and a text ending in U+FFFD (no prefix is recorded). If no
tail short of the whole prompt qualifies, the whole prompt is used.

Upstream vLLM primed with a 4-24 token suffix until #34217 (2026-02) switched
to the whole prompt; open PRs #51281 (32 tokens) and #57330 (128) restore a tail.
"""

from __future__ import annotations

from typing import Any

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_REPLACEMENT = "�"


def prime_ids(
    tokenizer: Any,
    prompt_ids: list[int] | None,
    skip_special_tokens: bool,
    num_tokens: int,
) -> list[int] | None:
    """Return the prompt tail to prime ``DecodeStream`` with (``prompt_ids``
    itself when no shorter tail is safe). ``tokenizer`` is a
    ``tokenizers.Tokenizer``."""
    if not prompt_ids or num_tokens <= 0 or len(prompt_ids) <= num_tokens:
        return prompt_ids
    n = num_tokens
    while n < len(prompt_ids):
        tail = prompt_ids[-n:]
        try:
            text = tokenizer.decode(tail, skip_special_tokens=skip_special_tokens)
        except Exception:
            return prompt_ids
        if text and not text.endswith(_REPLACEMENT):
            return tail
        n *= 2
    return prompt_ids


class _PrimedRequest:
    """Read-only view of a request with ``prompt_token_ids`` replaced."""

    __slots__ = ("_request", "prompt_token_ids")

    def __init__(self, request: Any, prompt_token_ids: list[int] | None):
        self._request = request
        self.prompt_token_ids = prompt_token_ids

    def __getattr__(self, name: str) -> Any:
        return getattr(self._request, name)


def patch_detokenizer_prime_tail() -> None:
    """Make ``FastIncrementalDetokenizer`` prime with a bounded prompt tail."""
    from vllm_torchtpu import envs

    num_tokens = envs.VLLM_TPU_DETOK_PRIME_TOKENS
    if num_tokens <= 0:
        return
    from vllm.v1.engine import detokenizer as detok_mod

    cls = detok_mod.FastIncrementalDetokenizer
    if getattr(cls, "_tpu_detok_prime_tokens", None) is not None:
        return
    orig_init = cls.__init__

    def __init__(self, tokenizer, request):
        ids = request.prompt_token_ids
        if ids is not None and len(ids) > num_tokens:
            params = request.sampling_params
            tail = prime_ids(
                tokenizer._tokenizer, ids, params.skip_special_tokens, num_tokens
            )
            if tail is not ids:
                request = _PrimedRequest(request, tail)
        orig_init(self, tokenizer, request)

    cls._tpu_detok_orig_init = orig_init
    cls.__init__ = __init__
    cls._tpu_detok_prime_tokens = num_tokens
    logger.info(
        "VLLM_TPU_DETOK_PRIME_TOKENS=%d: FastIncrementalDetokenizer primes with "
        "the prompt tail",
        num_tokens,
    )
