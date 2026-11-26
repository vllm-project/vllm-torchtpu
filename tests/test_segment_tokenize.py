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
"""Unit tests for segment-cached tokenization and GLM DSA auto-dispatch."""

from __future__ import annotations

import logging
import sys
import threading
import types
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from vllm_torchtpu import tpu_info
from vllm_torchtpu.core import segment_tokenize
from vllm_torchtpu.models.vllm import glm_dsa_dispatch


@pytest.fixture(autouse=True)
def _reset_state():
    prev = (
        glm_dsa_dispatch._ACTIVE_ARCH,
        glm_dsa_dispatch._ACTIVE_MAX_MODEL_LEN,
        tpu_info._v7x_model_fastpaths,
    )
    try:
        yield
    finally:
        (
            glm_dsa_dispatch._ACTIVE_ARCH,
            glm_dsa_dispatch._ACTIVE_MAX_MODEL_LEN,
            tpu_info._v7x_model_fastpaths,
        ) = prev


@dataclass
class _AddedToken:
    content: str
    normalized: bool = False
    lstrip: bool = False
    rstrip: bool = False
    single_word: bool = False


class _Encoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class _FakeBackendTokenizer:
    def __init__(self, added: dict[int, _AddedToken]) -> None:
        self._added = dict(added)
        self._vocab = {
            "<|user|>": 1001,
            "<|assistant|>": 1002,
            "<|user|>_ext": 1003,
            "ab": 200,
        }
        self.encode_calls = self.batch_calls = 0

    def get_added_tokens_decoder(self) -> dict[int, _AddedToken]:
        return dict(self._added)

    def get_vocab_size(self, with_added_tokens: bool = True) -> int:
        del with_added_tokens
        return 2048

    def no_truncation(self) -> None:
        pass

    def no_padding(self) -> None:
        pass

    def _encode_raw(self, text: str) -> list[int]:
        out, i = [], 0
        ordered = sorted(self._vocab.items(), key=lambda kv: (-len(kv[0]), kv[0]))
        while i < len(text):
            for tok, tid in ordered:
                if text.startswith(tok, i):
                    out.append(tid)
                    i += len(tok)
                    break
            else:
                out.append(ord(text[i]))
                i += 1
        return out

    def encode(self, text: str, add_special_tokens: bool = False) -> _Encoding:
        self.encode_calls += 1
        ids = self._encode_raw(text)
        return _Encoding([1, *ids] if add_special_tokens else ids)

    def encode_batch(
        self, texts: list[str], add_special_tokens: bool = False
    ) -> list[_Encoding]:
        self.batch_calls += 1
        return [
            _Encoding(
                [1, *self._encode_raw(t)] if add_special_tokens else self._encode_raw(t)
            )
            for t in texts
        ]


class _WrapperTokenizer:
    def __init__(self, backend: _FakeBackendTokenizer) -> None:
        self._tokenizer = backend


def test_segment_tokenizer_exact_parity_and_longest_added_token():
    backend = _FakeBackendTokenizer(
        {
            1001: _AddedToken("<|user|>"),
            1002: _AddedToken("<|assistant|>"),
            1003: _AddedToken("<|user|>_ext"),
        }
    )
    seg = segment_tokenize.get_segment_tokenizer(
        _WrapperTokenizer(backend), max_cached_tokens=4096
    )
    assert seg is not None
    long_sys = "system instructions: " + ("alpha beta gamma " * 20)
    turn_1 = "solve 17 * 23 step by step " + ("context payload " * 12)
    prompt_a = f"<|user|>{long_sys}<|assistant|>ok<|user|>_ext{turn_1}<|assistant|>"
    prompt_b = (
        f"<|user|>{long_sys}<|assistant|>ok<|user|>a<|user|>b"
        f"<|assistant|>{turn_1}<|assistant|>"
    )

    exp_a = backend.encode(prompt_a, add_special_tokens=False).ids
    exp_b = backend.encode(prompt_b, add_special_tokens=False).ids
    assert seg.encode(prompt_a) == exp_a and seg.encode(prompt_b) == exp_b
    assert seg.hits >= 2
    calls_before, misses_before = seg._backend.batch_calls, seg.misses
    assert seg.encode(prompt_a) == exp_a
    assert (
        seg.hits >= 4
        and seg._backend.batch_calls >= calls_before
        and seg.misses - misses_before <= 1
    )


def test_segment_tokenizer_rejects_empty_or_non_boundary_safe_added_tokens():
    for added in ({}, {1001: _AddedToken("<|user|>", lstrip=True)}):
        assert (
            segment_tokenize.get_segment_tokenizer(
                _WrapperTokenizer(_FakeBackendTokenizer(added)), max_cached_tokens=512
            )
            is None
        )


def test_segment_tokenizer_lru_eviction_respects_budget():
    seg = segment_tokenize.SegmentTokenizer(
        _FakeBackendTokenizer({1001: _AddedToken("<|user|>")}), max_cached_tokens=150
    )
    seg_a, seg_b = "A" * 100, "B" * 100
    seg.encode(f"<|user|>{seg_a}<|user|>{seg_b}")
    assert seg._cached_tokens <= 150 and seg_a not in seg._cache and seg_b in seg._cache


def test_glm_dsa_architecture_gating_strictly_isolates_other_models(monkeypatch):
    def _cfg(arch: str, mtype: str):
        return SimpleNamespace(
            max_model_len=163_840,
            hf_config=SimpleNamespace(architectures=[arch], model_type=mtype),
        )

    glm_cfg = _cfg("GlmMoeDsaForCausalLM", "glm_moe_dsa")
    monkeypatch.setattr(glm_dsa_dispatch, "is_tpu_v7x", lambda: True)
    assert glm_dsa_dispatch.is_glm_dsa_model(
        glm_cfg
    ) and glm_dsa_dispatch.glm_dsa_v7x_active(glm_cfg)
    assert (
        glm_dsa_dispatch.resolve_ep_tp_size(16, glm_cfg) == 2
        and glm_dsa_dispatch.resolve_ep_tp_size(1, glm_cfg) == 1
    )
    for arch, mtype in (
        ("DeepseekV3ForCausalLM", "deepseek_v3"),
        ("KimiK2ForCausalLM", "kimi_k2"),
        ("Gemma4ForCausalLM", "gemma4"),
        ("Qwen3MoeForCausalLM", "qwen3_moe"),
    ):
        c = _cfg(arch, mtype)
        assert (
            not glm_dsa_dispatch.is_glm_dsa_model(c)
            and not glm_dsa_dispatch.glm_dsa_v7x_active(c)
            and glm_dsa_dispatch.resolve_ep_tp_size(16, c) == 1
        )
    monkeypatch.setattr(glm_dsa_dispatch, "is_tpu_v7x", lambda: False)
    assert (
        not glm_dsa_dispatch.glm_dsa_v7x_active(glm_cfg)
        and glm_dsa_dispatch.resolve_ep_tp_size(16, glm_cfg) == 1
    )


def test_sp_shared_group_and_bucket_gate_edge():
    gib = 1 << 30
    assert glm_dsa_dispatch.select_sp_shared_group(16, 163_840, 40 * gib) == 1
    assert glm_dsa_dispatch.select_sp_shared_group(16, 1_048_576, 40 * gib) == 4
    assert glm_dsa_dispatch.select_sp_shared_group(16, 163_840, 18 * gib) == 4
    assert (
        glm_dsa_dispatch.resolve_dcp_mla_topk_cap(0, dcp_size=16) == 0
        and glm_dsa_dispatch.resolve_dcp_mla_topk_cap(512, dcp_size=16) == 512
    )
    for b, edge in ((64, 32), (128, 64), (1024, 512)):
        assert glm_dsa_dispatch.bucket_gate_edge(b) == edge


def test_patch_renderer_segment_tokenize_gates_by_model_and_length(monkeypatch):
    tok = _WrapperTokenizer(_FakeBackendTokenizer({1001: _AddedToken("<|user|>")}))
    orig_calls: list[str] = []

    class FakeBaseRenderer:
        def __init__(self, model_config, tokenizer):
            self.model_config, self._tok = model_config, tokenizer

        def get_tokenizer(self):
            return self._tok

        def _wants_offsets(self, *_a):
            return False

        def _build_tokens_prompt(self, ids, prompt, offset_mapping=None):
            return {
                "prompt_token_ids": ids,
                "prompt": prompt["prompt"],
                "offset_mapping": offset_mapping,
            }

        def _tokenize_prompt(self, prompt, params):
            orig_calls.append(prompt["prompt"][:16])
            kw = params.get_encode_kwargs()
            ids = self._tok._tokenizer.encode(
                prompt["prompt"],
                add_special_tokens=bool(kw.get("add_special_tokens", False)),
            ).ids
            return self._build_tokens_prompt(ids, prompt)

    fake_vllm, fake_logger, fake_renderers, fake_base = (
        types.ModuleType("vllm"),
        types.ModuleType("vllm.logger"),
        types.ModuleType("vllm.renderers"),
        types.ModuleType("vllm.renderers.base"),
    )
    fake_logger.init_logger = logging.getLogger  # type: ignore[attr-defined]
    fake_vllm.logger, fake_vllm.renderers = fake_logger, fake_renderers  # type: ignore[attr-defined]
    fake_renderers.base, fake_base.BaseRenderer = fake_base, FakeBaseRenderer  # type: ignore[attr-defined]
    for name, mod in (
        ("vllm", fake_vllm),
        ("vllm.logger", fake_logger),
        ("vllm.renderers", fake_renderers),
        ("vllm.renderers.base", fake_base),
    ):
        monkeypatch.setitem(sys.modules, name, mod)

    segment_tokenize.patch_renderer_segment_tokenize()
    segment_tokenize.patch_renderer_segment_tokenize()
    params = SimpleNamespace(get_encode_kwargs=lambda: {"add_special_tokens": False})
    long_text = "<|user|>" + ("ab" * 16_400)
    dsv3_cfg = SimpleNamespace(
        max_model_len=163_840,
        hf_config=SimpleNamespace(
            architectures=["DeepseekV3ForCausalLM"], model_type="deepseek_v3"
        ),
    )
    glm_cfg = SimpleNamespace(
        max_model_len=163_840,
        hf_config=SimpleNamespace(
            architectures=["GlmMoeDsaForCausalLM"], model_type="glm_moe_dsa"
        ),
    )
    out_other = FakeBaseRenderer(dsv3_cfg, tok)._tokenize_prompt(
        {"prompt": long_text}, params
    )
    assert len(orig_calls) == 1
    r_glm = FakeBaseRenderer(glm_cfg, tok)
    out_glm = r_glm._tokenize_prompt({"prompt": long_text}, params)
    assert (
        len(orig_calls) == 1
        and out_glm["prompt_token_ids"] == out_other["prompt_token_ids"]
    )
    r_glm._tokenize_prompt({"prompt": "<|user|>short"}, params)
    assert len(orig_calls) == 2


def test_segment_tokenizer_rejects_prepend_and_handles_non_weakref_and_concurrency():
    b_norm = _FakeBackendTokenizer({1001: _AddedToken("<|user|>")})
    b_norm.normalizer = SimpleNamespace(type="Prepend")  # type: ignore[attr-defined]
    b_pre = _FakeBackendTokenizer({1001: _AddedToken("<|user|>")})
    b_pre.pre_tokenizer = SimpleNamespace(prepend_scheme="first")  # type: ignore[attr-defined]
    assert (
        segment_tokenize.SegmentTokenizer.eligible(b_norm) is not None
        and segment_tokenize.SegmentTokenizer.eligible(b_pre) is not None
    )

    class _SlotsTok:
        __slots__ = ("_tokenizer",)

        def __init__(self, b):
            self._tokenizer = b

    s1 = _SlotsTok(_FakeBackendTokenizer({1001: _AddedToken("<|user|>")}))
    assert segment_tokenize.get_segment_tokenizer(s1, max_cached_tokens=512) is not None
    assert len(segment_tokenize._INSTANCES) == 0

    entered, release = threading.Event(), threading.Event()

    class _SlowBatchBackend(_FakeBackendTokenizer):
        def encode_batch(self, texts, add_special_tokens=False):
            if any("MISS" in t for t in texts):
                entered.set()
                assert release.wait(timeout=5.0)
            return super().encode_batch(texts, add_special_tokens=add_special_tokens)

    slow_seg = segment_tokenize.SegmentTokenizer(
        _SlowBatchBackend({1001: _AddedToken("<|user|>")}), max_cached_tokens=4096
    )
    hit_prompt = f"<|user|>HIT_{'x' * 80}"
    exp_hit = slow_seg.encode(hit_prompt)
    miss_err: list[BaseException] = []

    def _run_miss():
        try:
            slow_seg.encode(f"<|user|>MISS_{'y' * 80}")
        except BaseException as exc:  # noqa: BLE001
            miss_err.append(exc)

    worker = threading.Thread(target=_run_miss)
    worker.start()
    assert entered.wait(timeout=5.0) and slow_seg.encode(hit_prompt) == exp_hit
    release.set()
    worker.join(timeout=5.0)
    assert not miss_err

    glm_dsa_dispatch.set_active_model_config(
        SimpleNamespace(architectures=["GlmMoeDsaForCausalLM"], max_model_len=163_840)
    )
    assert tpu_info._v7x_model_fastpaths is True
    glm_dsa_dispatch.set_active_model_config(
        SimpleNamespace(architectures=["DeepseekV3ForCausalLM"], max_model_len=163_840)
    )
    assert tpu_info._v7x_model_fastpaths is False
