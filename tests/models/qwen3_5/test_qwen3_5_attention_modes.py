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
"""The Qwen3.5 trade-off settings native_fp8_attention and exp2_softmax: they
parse strictly, are off by default, and set the target layers' batched RPA
modes (speculative draft layers stay at full precision; see
tests/layers/adapter/test_native_fp8_draft_scope.py)."""

from types import SimpleNamespace

import pytest

from vllm_torchtpu.layers.adapter import attention, native_fp8_scope
from vllm_torchtpu.models.qwen3_5 import settings

pytestmark = pytest.mark.cpu_test

QWEN35 = "Qwen3_5MoeForConditionalGeneration"


def _vllm_config(section=None, arch=QWEN35):
    additional = {} if section is None else {"qwen3_5": section}
    return SimpleNamespace(
        model_config=SimpleNamespace(architectures=[arch], max_model_len=262144),
        additional_config=additional,
        speculative_config=None,
    )


def test_trade_offs_are_off_by_default():
    defaults = settings.parse({})
    assert defaults.native_fp8_attention is False
    assert defaults.exp2_softmax is False
    assert attention._qwen3_5_attention_modes(_vllm_config()) == native_fp8_scope.OFF


@pytest.mark.parametrize(
    "section, modes",
    [
        (
            {"native_fp8_attention": True},
            native_fp8_scope.AttentionModes(native_fp8="pvs", native_fp8_prefill="pvs"),
        ),
        ({"exp2_softmax": True}, native_fp8_scope.AttentionModes(decode_exp2=True)),
        (
            {"native_fp8_attention": True, "exp2_softmax": True},
            native_fp8_scope.AttentionModes(
                native_fp8="pvs", native_fp8_prefill="pvs", decode_exp2=True
            ),
        ),
    ],
)
def test_settings_set_the_target_modes(section, modes):
    assert attention._qwen3_5_attention_modes(_vllm_config(section)) == modes


@pytest.mark.parametrize("value", [1, "true", None])
def test_trade_offs_take_bools_only(value):
    for key in ("native_fp8_attention", "exp2_softmax"):
        with pytest.raises(ValueError, match=key):
            settings.parse({key: value})


def test_other_models_stay_off():
    other = _vllm_config(arch="LlamaForCausalLM")
    assert attention._qwen3_5_attention_modes(other) == native_fp8_scope.OFF
    with pytest.raises(ValueError, match="not Qwen3.5"):
        attention._qwen3_5_attention_modes(
            _vllm_config({"native_fp8_attention": True}, arch="LlamaForCausalLM")
        )


def test_batched_impl_reads_the_settings(monkeypatch):
    config = _vllm_config({"native_fp8_attention": True, "exp2_softmax": True})
    monkeypatch.setattr(attention, "get_current_vllm_config_or_none", lambda: config)
    monkeypatch.setattr(
        attention.PallasAttentionBackendImpl, "__init__", lambda self, *a, **k: None
    )
    impl = attention.PallasBatchedRPAAttentionBackendImpl.__new__(
        attention.PallasBatchedRPAAttentionBackendImpl
    )
    impl.head_size = 256
    impl.num_kv_heads = 1
    impl.kv_cache_quantized_dtype = None
    attention.PallasBatchedRPAAttentionBackendImpl.__init__(impl)
    assert impl.attention_modes == native_fp8_scope.AttentionModes(
        native_fp8="pvs", native_fp8_prefill="pvs", decode_exp2=True
    )
