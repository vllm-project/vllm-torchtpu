# SPDX-License-Identifier: Apache-2.0
"""Prime the streaming detokenizer with a prompt suffix (TPU_DETOK_PROMPT_SUFFIX).

vLLM's ``FastIncrementalDetokenizer`` builds ``DecodeStream(ids=prompt_token_ids)``
with the *whole* prompt; the stream's first ``step`` then decodes all of it
(prefix text + prefix-with-new-token), in the API server's event loop, right
after the first token's TTFT stamp. On 100k+ token AgentX prompts that is tens of
ms per request between the engine producing the first token and the client
receiving it, and it stalls every other stream's chunks meanwhile.

The decoded text of new tokens depends only on a few preceding tokens (an
incomplete UTF-8 sequence spans at most 4 byte-fallback tokens; leading-space
stripping and cleanup look one token back), so priming with the last N prompt
tokens (N=TPU_DETOK_PROMPT_SUFFIX, e.g. 32) yields the same output text -- vLLM's
slow detokenizer primes with 5. Default 0 = off (stock behaviour).
"""

import os

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)
_ENV = "TPU_DETOK_PROMPT_SUFFIX"


def install() -> None:
    try:
        n = int(os.environ.get(_ENV, "0") or 0)
    except ValueError:
        n = 0
    if n <= 0:
        return
    try:
        from vllm.v1.engine import detokenizer as _dt

        cls = _dt.FastIncrementalDetokenizer
        if getattr(cls, "_tpu_prompt_suffix", 0):
            return
        orig_init = cls.__init__

        def __init__(self, tokenizer, request):
            ids = request.prompt_token_ids
            if ids is None or len(ids) <= n:
                return orig_init(self, tokenizer, request)
            # Only the DecodeStream prefill reads the prompt here; hand it the
            # suffix and restore the request before anything else sees it.
            request.prompt_token_ids = ids[-n:]
            try:
                orig_init(self, tokenizer, request)
            finally:
                request.prompt_token_ids = ids

        cls.__init__ = __init__
        cls._tpu_prompt_suffix = n
        logger.info("%s=%d: detokenizer primed with the last %d prompt tokens",
                    _ENV, n, n)
    except Exception as e:  # pragma: no cover
        logger.warning("%s: install failed: %s", _ENV, e)
