# SPDX-License-Identifier: Apache-2.0
"""The reasoning-count re-parse runs with tool parsing skipped and restores it."""

import sys
import types

from vllm_torchtpu import envs
from vllm_torchtpu.core import reasoning_count_patch


class FakeEngine:
    def __init__(self, tokenizer, **kwargs):
        self.skip_tool_parsing = False
        self.seen = []
        self.count = 0

    def _single_pass_parse(self, text, token_ids):
        self.seen.append(self.skip_tool_parsing)
        self.count = sum(1 for t in token_ids if t < 100)  # "reasoning" ids

    def count_reasoning_tokens(self, token_ids):
        return self.count


class FakeTokenizer:
    def decode(self, ids):
        return " ".join(map(str, ids))


def make_adapter_module():
    class ParserEngineReasoningAdapter:
        _parser_engine_cls = FakeEngine

        def __init__(self):
            self.model_tokenizer = FakeTokenizer()
            self._parser_engine = FakeEngine(None)
            self._parser_engine_kwargs = {}
            self._counting_parser_engine = None
            self._streaming_count_valid = False

        def count_reasoning_tokens(self, token_ids):
            if self._streaming_count_valid:
                return -1  # the streaming engine's count
            if not token_ids:
                return 0
            raise AssertionError("unpatched re-parse")

    module = types.ModuleType("vllm.parser.engine.adapters")
    module.ParserEngineReasoningAdapter = ParserEngineReasoningAdapter
    return module


def install(monkeypatch, enabled=True):
    module = make_adapter_module()
    import vllm  # noqa: F401  (the real package; only the adapter is faked)

    for name in ("vllm.parser", "vllm.parser.engine"):
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.parser.engine.adapters", module)
    monkeypatch.setattr(envs, "VLLM_TPU_REASONING_COUNT_SKIP_TOOLS", enabled)
    reasoning_count_patch.patch_reasoning_count_skip_tools()
    return module.ParserEngineReasoningAdapter


def test_reparse_skips_tool_parsing_and_restores(monkeypatch):
    adapter = install(monkeypatch)()
    assert adapter.count_reasoning_tokens([1, 2, 300, 3]) == 3
    engine = adapter._counting_parser_engine
    assert engine.seen == [True]
    assert engine.skip_tool_parsing is False


def test_streaming_count_and_empty_keep_original(monkeypatch):
    adapter = install(monkeypatch)()
    assert adapter.count_reasoning_tokens([]) == 0
    adapter._streaming_count_valid = True
    assert adapter.count_reasoning_tokens([1, 2]) == -1
    assert adapter._counting_parser_engine is None


def test_flag_off_leaves_vllm_unpatched(monkeypatch):
    cls = install(monkeypatch, enabled=False)
    assert not getattr(cls, "_tpu_count_skips_tools", False)
