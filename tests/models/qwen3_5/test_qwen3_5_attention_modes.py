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
modes per PD leg, in the kernels the validation entries' legs run them in
(speculative draft layers stay at full precision; see
tests/layers/adapter/test_native_fp8_draft_scope.py)."""

from types import SimpleNamespace

import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa import configs
from vllm_torchtpu.layers.adapter import attention, native_fp8_scope
from vllm_torchtpu.models.qwen3_5 import settings

pytestmark = pytest.mark.cpu_test

PVS = configs.NativeFp8(pv=configs.Fp8Operand.STATIC)

QWEN35 = "Qwen3_5MoeForConditionalGeneration"


def _vllm_config(section=None, arch=QWEN35, role="kv_consumer"):
    additional = {} if section is None else {"qwen3_5": section}
    return SimpleNamespace(
        # The text config of no measured checkpoint: the batched layer's
        # tuned-tiles rule reads it (attention.qwen3_5_tuned_tiles).
        model_config=SimpleNamespace(
            architectures=[arch], max_model_len=262144, hf_text_config=SimpleNamespace()
        ),
        additional_config=additional,
        speculative_config=None,
        kv_transfer_config=None if role is None else SimpleNamespace(kv_role=role),
    )


def test_trade_offs_are_off_by_default():
    defaults = settings.parse({})
    assert defaults.native_fp8_attention is False
    assert defaults.exp2_softmax is False
    for role in ("kv_producer", "kv_consumer", "kv_both", None):
        config = _vllm_config(role=role)
        assert attention._qwen3_5_attention_modes(config) == native_fp8_scope.OFF


BOTH = {"native_fp8_attention": True, "exp2_softmax": True}


@pytest.mark.parametrize(
    "role, section, modes",
    [
        # Prefill legs: fp8 P.V in the prefill and mixed kernels only; the
        # decode kernel keeps full precision and base e.
        (
            "kv_producer",
            {"native_fp8_attention": True},
            native_fp8_scope.AttentionModes(native_fp8_prefill=PVS),
        ),
        ("kv_producer", {"exp2_softmax": True}, native_fp8_scope.OFF),
        (
            "kv_producer",
            BOTH,
            native_fp8_scope.AttentionModes(native_fp8_prefill=PVS),
        ),
        # Decode legs: fp8 P.V and the base-2 softmax in the decode kernel
        # only; the prefill and mixed kernels keep full precision.
        (
            "kv_consumer",
            {"native_fp8_attention": True},
            native_fp8_scope.AttentionModes(native_fp8=PVS),
        ),
        (
            "kv_consumer",
            {"exp2_softmax": True},
            native_fp8_scope.AttentionModes(decode_exp2=True),
        ),
        (
            "kv_consumer",
            BOTH,
            native_fp8_scope.AttentionModes(native_fp8=PVS, decode_exp2=True),
        ),
    ],
)
def test_settings_set_the_legs_target_modes(role, section, modes):
    config = _vllm_config(section, role=role)
    assert attention._qwen3_5_attention_modes(config) == modes


@pytest.mark.parametrize("role", ["kv_both", None])
@pytest.mark.parametrize(
    "section", [{"native_fp8_attention": True}, {"exp2_softmax": True}]
)
def test_engines_that_are_not_a_pd_leg_refuse_the_trade_offs(role, section):
    with pytest.raises(ValueError, match="kv_role"):
        attention._qwen3_5_attention_modes(_vllm_config(section, role=role))


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
    config = _vllm_config(BOTH, role="kv_consumer")
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
        native_fp8=PVS, decode_exp2=True
    )
