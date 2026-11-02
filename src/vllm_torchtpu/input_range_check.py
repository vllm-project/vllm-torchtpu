# SPDX-License-Identifier: Apache-2.0
"""Skip the vocabulary range check for prompt ids the server's own tokenizer made.

Opt-in with ``VLLM_TPU_SKIP_TOKENIZER_ID_RANGE_CHECK=1``.

Why: before a request goes to the engine, vLLM's
``InputProcessor._validate_model_input`` runs Python's ``max`` and ``min`` over
every prompt id to reject ids outside the vocabulary. For a 137k-token agent
prompt that is two passes over 137k Python ints in the API server's event loop,
on every turn, between the response headers and the engine hand-off.

Why it is exact: the renderer's tokenizer returns ids of its own vocabulary
only, and those lie in ``[0, tokenizer.max_token_id]``, inside the range the
check accepts. ``BaseRenderer._build_tokens_prompt`` builds the id list of
every prompt the renderer tokenizes, the segment tokenizer's included; this
module remembers that list object. The check skips the two passes only for a
token prompt whose id list is that same object. A prompt sent as ids, one whose
ids truncation or padding replaced, and every multimodal or embeds prompt carry
another object or type and are checked as before. The prompt length check runs
for every prompt.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

REMEMBERED_PROMPTS = 64


class TokenizerOutputs:
    """The id lists the renderer's tokenizer built, kept until their range check claims them."""

    def __init__(self, capacity: int = REMEMBERED_PROMPTS) -> None:
        self.capacity = capacity
        self.lists: OrderedDict[int, list[int]] = OrderedDict()
        self.lock = threading.Lock()

    def remember(self, ids: list[int]) -> None:
        with self.lock:
            self.lists[id(ids)] = ids
            while len(self.lists) > self.capacity:
                self.lists.popitem(last=False)

    def claim(self, ids: object) -> bool:
        """True when ``ids`` is a list the tokenizer built; it is forgotten either way."""
        with self.lock:
            remembered = self.lists.pop(id(ids), None)
        return remembered is ids


TOKENIZER_OUTPUTS = TokenizerOutputs()


def patch_skip_tokenizer_id_range_check() -> None:
    """Skip the prompt id range check for the renderer's own tokenizer output."""
    from vllm_torchtpu import envs

    if not envs.VLLM_TPU_SKIP_TOKENIZER_ID_RANGE_CHECK:
        return
    from vllm.renderers.base import BaseRenderer
    from vllm.v1.engine.input_processor import InputProcessor

    if InputProcessor.__dict__.get("_tpu_skip_tokenizer_id_range_check", False):
        return
    build_tokens_prompt = BaseRenderer._build_tokens_prompt
    validate_model_input = InputProcessor._validate_model_input

    def _build_tokens_prompt(token_ids, prompt, *, offset_mapping=None):
        tokens_prompt = build_tokens_prompt(token_ids, prompt, offset_mapping=offset_mapping)
        TOKENIZER_OUTPUTS.remember(tokens_prompt["prompt_token_ids"])
        return tokens_prompt

    def _validate_model_input(self, prompt_input, prompt_type) -> None:
        if prompt_input["type"] == "token" and TOKENIZER_OUTPUTS.claim(
            prompt_input["prompt_token_ids"]
        ):
            self._validate_prompt_len(len(prompt_input["prompt_token_ids"]), prompt_type)
            return
        validate_model_input(self, prompt_input, prompt_type)

    BaseRenderer._build_tokens_prompt = staticmethod(_build_tokens_prompt)
    InputProcessor._validate_model_input = _validate_model_input
    InputProcessor._tpu_skip_tokenizer_id_range_check = True
    logger.info(
        "VLLM_TPU_SKIP_TOKENIZER_ID_RANGE_CHECK: prompt ids from the server's tokenizer skip the vocabulary range check"
    )
