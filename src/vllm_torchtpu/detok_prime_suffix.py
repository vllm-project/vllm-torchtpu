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
"""Prime each request's decode stream with the prompt's last tokens only.

vLLM's ``FastIncrementalDetokenizer`` primes its ``DecodeStream`` with every
prompt token. The stream's first step decodes all of them to find the text in
front of the new token, so the first output token's text costs time in
proportion to the prompt, on the API server's event loop.

Only the prompt's tail can change how the next token is written: an
unfinished multi-byte character, or a decoder that writes the first token of
a text apart (a leading space dropped). A suffix whose own decode is not empty
and holds no replacement character starts and ends on whole characters, so the
text after it is the same as after the whole prompt.

Ported from vLLM's ``vllm/v1/engine/detokenizer.py`` before it primed the
stream with the whole prompt (the shortest such suffix of 4 to 23 tokens, else
the whole prompt). Changes: the suffix decode honours ``skip_special_tokens``,
a suffix that decodes to empty text is refused, and the suffix primes the
stream through ``DecodeStream(ids=...)``.
"""

from collections.abc import Sequence

from tokenizers import Tokenizer

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

REPLACEMENT_CHARACTER = "�"
# The lower bound vLLM used for the suffix.
SHORTEST_SUFFIX = 4
# Bounds the first step's decode to at most 24 tokens.
LONGEST_SUFFIX = 23


def prompt_suffix(
    tokenizer: Tokenizer, prompt_token_ids: Sequence[int], skip_special_tokens: bool
) -> Sequence[int]:
    """The shortest prompt suffix that decodes to whole characters.

    Args:
        tokenizer: the request's tokenizer.
        prompt_token_ids: the prompt's token ids.
        skip_special_tokens: the request's sampling setting.

    Returns:
        The shortest suffix of 4 to 23 tokens whose decode is not empty and
        holds no replacement character, else the whole prompt.
    """
    longest = min(len(prompt_token_ids) - 1, LONGEST_SUFFIX)
    for length in range(SHORTEST_SUFFIX, longest + 1):
        suffix = prompt_token_ids[-length:]
        text = tokenizer.decode(suffix, skip_special_tokens=skip_special_tokens)
        if text and REPLACEMENT_CHARACTER not in text:
            return suffix
    return prompt_token_ids


class _PromptSuffixRequest:
    """A request as the detokenizer reads it, with the prompt cut to its
    suffix."""

    def __init__(self, request, prompt_token_ids: Sequence[int]):
        self._request = request
        self.prompt_token_ids = prompt_token_ids

    def __getattr__(self, name):
        return getattr(self._request, name)


def patch_detokenizer_prime_suffix() -> None:
    """Prime ``FastIncrementalDetokenizer``'s stream with the prompt suffix."""
    from vllm_torchtpu import envs

    if not envs.VLLM_TPU_DETOK_PRIME_SUFFIX:
        return
    from vllm.v1.engine import detokenizer

    cls = detokenizer.FastIncrementalDetokenizer
    if getattr(cls, "_tpu_prime_suffix", False):
        return
    original_init = cls.__init__

    def __init__(self, tokenizer, request):
        prompt = request.prompt_token_ids
        if prompt is not None:
            suffix = prompt_suffix(
                tokenizer._tokenizer,
                prompt,
                request.sampling_params.skip_special_tokens,
            )
            if len(suffix) < len(prompt):
                request = _PromptSuffixRequest(request, suffix)
        original_init(self, tokenizer, request)

    cls.__init__ = __init__
    cls._tpu_prime_suffix = True
    logger.info(
        "VLLM_TPU_DETOK_PRIME_SUFFIX: FastIncrementalDetokenizer primes its "
        "decode stream with a prompt suffix of %d to %d tokens",
        SHORTEST_SUFFIX,
        LONGEST_SUFFIX,
    )
