# SPDX-License-Identifier: Apache-2.0
"""Count reasoning tokens without re-parsing tool calls (VLLM_TPU_REASONING_COUNT_SKIP_TOOLS).

After a streamed chat completion, the server counts the response's reasoning
tokens (``usage.completion_tokens_details``). When the streaming count is not
valid, ``ParserEngineReasoningAdapter.count_reasoning_tokens`` re-parses the
whole output in a fresh parser engine with tool parsing on: every chunk of a
tool call recomputes the call's full argument JSON (regex conversion plus the
safe-prefix scan), which is quadratic in the argument length. One long tool
argument (a large file write) holds the API server's event loop, and with it
every request, for minutes (j-2514e030: 333 s, py-spy in
``_qwen3_arg_converter`` / ``_safe_arg_prefix`` under ``count_reasoning_tokens``).

The reasoning count only depends on the reasoning section, so the re-parse
runs with tool parsing skipped. Same count, linear time.
"""

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def patch_reasoning_count_skip_tools() -> None:
    if not envs.VLLM_TPU_REASONING_COUNT_SKIP_TOOLS:
        return
    try:
        from vllm.parser.engine.adapters import ParserEngineReasoningAdapter
    except ImportError:
        return
    if getattr(ParserEngineReasoningAdapter, "_tpu_count_skips_tools", False):
        return

    def count_reasoning_tokens(self, token_ids):
        if self._streaming_count_valid or not token_ids:
            return original(self, token_ids)
        if self._counting_parser_engine is None:
            self._counting_parser_engine = self._parser_engine_cls(
                self.model_tokenizer, **self._parser_engine_kwargs
            )
        engine = self._counting_parser_engine
        saved = engine.skip_tool_parsing
        engine.skip_tool_parsing = True
        try:
            engine._single_pass_parse(self.model_tokenizer.decode(token_ids), token_ids)
        finally:
            engine.skip_tool_parsing = saved
        return engine.count_reasoning_tokens(token_ids)

    original = ParserEngineReasoningAdapter.count_reasoning_tokens
    ParserEngineReasoningAdapter.count_reasoning_tokens = count_reasoning_tokens
    ParserEngineReasoningAdapter._tpu_count_skips_tools = True
    logger.info("Reasoning token count re-parses without tool parsing")
