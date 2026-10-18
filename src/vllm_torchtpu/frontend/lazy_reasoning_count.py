# SPDX-License-Identifier: Apache-2.0
"""Skip reasoning-token recounts that no client can observe in chat streaming.

vLLM's ``OpenAIServingChat.chat_completion_stream_generator`` calls
``parser.count_reasoning_tokens(all generated ids)`` after every streamed
delta. For a ``DelegatingParser`` over a ``ParserEngineReasoningAdapter``
whose streaming count is not valid (the prompt already ended reasoning, so
``extract_reasoning_streaming`` never ran), every call decodes all generated
ids and re-parses the whole output. That is O(n^2) per request, run on the
API-server event loop while holding the GIL.

The stream generator reads the count only in per-chunk usage, when
continuous usage stats are on, and in the final usage after the finished
delta. The patch runs the stock count in exactly those cases and returns the
previous value otherwise, so every count a client or the server can observe
is unchanged.

TODO: remove once vLLM stops recounting reasoning tokens after every
streamed delta.
"""

import functools
import inspect
import multiprocessing
from collections.abc import Sequence

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_installed = False
# True once any chat server runs with --enable-force-include-usage, which
# turns continuous usage stats on for every request.
_force_include_usage = False


def _is_api_server_process() -> bool:
    name = multiprocessing.current_process().name
    # ApiServer_<i>: multi-API-server manager. APIServer_DPRank_<r>: the
    # multi-port data-parallel supervisor.
    return name == "MainProcess" or name.startswith(("ApiServer", "APIServer"))


def find_unsupported_vllm_api() -> str | None:
    """Returns why the installed vLLM cannot take the patch, or None."""
    try:
        from vllm.entrypoints.openai.chat_completion.serving import (
            OpenAIServingChat,
        )
        from vllm.parser.abstract_parser import DelegatingParser
        from vllm.parser.engine.adapters import ParserEngineReasoningAdapter
    except ImportError as e:
        return str(e)
    if "count_reasoning_tokens" not in vars(DelegatingParser):
        return "DelegatingParser.count_reasoning_tokens is not defined"
    if (
        not {"request", "finished"}
        <= inspect.signature(DelegatingParser.parse_delta).parameters.keys()
    ):
        return "DelegatingParser.parse_delta takes no request/finished"
    if "count_reasoning_tokens" not in vars(ParserEngineReasoningAdapter):
        return "ParserEngineReasoningAdapter.count_reasoning_tokens is not defined"
    if (
        "enable_force_include_usage"
        not in inspect.signature(OpenAIServingChat.__init__).parameters
    ):
        return "OpenAIServingChat takes no enable_force_include_usage"
    return None


def maybe_install_lazy_reasoning_count() -> None:
    """Patches ``DelegatingParser`` in the API-server process. Idempotent."""
    global _installed
    if (
        _installed
        or not envs.VLLM_TPU_LAZY_REASONING_COUNT
        or not _is_api_server_process()
    ):
        return
    unsupported = find_unsupported_vllm_api()
    if unsupported is not None:
        logger.warning(
            "Lazy reasoning count not installed (%s); chat streaming keeps "
            "vLLM's per-delta reasoning-token recount.",
            unsupported,
        )
        return

    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    from vllm.entrypoints.serve.utils.api_utils import should_include_usage
    from vllm.parser.abstract_parser import DelegatingParser
    from vllm.parser.engine.adapters import ParserEngineReasoningAdapter

    orig_serving_init = OpenAIServingChat.__init__
    orig_parse_delta = DelegatingParser.parse_delta
    orig_count = DelegatingParser.count_reasoning_tokens

    @functools.wraps(orig_serving_init)
    def serving_init(self, *args, **kwargs) -> None:
        global _force_include_usage
        orig_serving_init(self, *args, **kwargs)
        _force_include_usage |= bool(self.enable_force_include_usage)

    @functools.wraps(orig_parse_delta)
    def parse_delta(self, *args, **kwargs):
        delta = orig_parse_delta(self, *args, **kwargs)
        # The chat stream generator passes these by keyword. A missing one
        # means another caller, which keeps the stock count.
        request = kwargs.get("request")
        self._tpu_skip_reasoning_count = (
            kwargs.get("finished") is False
            and request is not None
            and isinstance(request, ChatCompletionRequest)
            and not should_include_usage(request.stream_options, _force_include_usage)[
                1
            ]
        )
        return delta

    @functools.wraps(orig_count)
    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        # One-shot: only the count right after a non-final chat delta skips.
        skip = self.__dict__.pop("_tpu_skip_reasoning_count", False)
        reasoning_parser = self._reasoning_parser
        # The recount is expensive only when the streaming count is invalid.
        # The attribute is read with a default so a vLLM rename falls back
        # to the stock count; test_lazy_reasoning_count guards the name.
        if (
            skip
            and isinstance(reasoning_parser, ParserEngineReasoningAdapter)
            and not getattr(reasoning_parser, "_streaming_count_valid", True)
        ):
            return self.__dict__.get("_tpu_last_reasoning_count", 0)
        count = orig_count(self, token_ids)
        self._tpu_last_reasoning_count = count
        return count

    OpenAIServingChat.__init__ = serving_init
    DelegatingParser.parse_delta = parse_delta
    DelegatingParser.count_reasoning_tokens = count_reasoning_tokens
    _installed = True
    logger.info(
        "Lazy reasoning count installed: chat streaming recounts reasoning "
        "tokens only when the count is observable."
    )
