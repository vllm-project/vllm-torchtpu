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
"""Registry entry for Qwen3.8-Flash, a segment-composed `ModelDef`."""

import pytest

from vllm_torchtpu.megakernel.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.model_def import ModelDef, find_model, register_model
from vllm_torchtpu.megakernel.models import qwen35, qwen38

# The published Qwen3.8-Flash-Next config, reduced to the keys the entry reads.
HF_CONFIG = {
    "architectures": ["Qwen4ExpForConditionalGeneration"],
    "model_type": "qwen4_exp",
    "text_config": {
        "model_type": "qwen4_exp_text",
        "num_hidden_layers": 48,
        "full_attention_interval": 4,
        "hidden_size": 2560,
        "num_attention_heads": 24,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 640,
        "shared_expert_intermediate_size": 640,
        "hc_count": 4,
        "hc_lowrank": 320,
        "rms_norm_eps": 1e-06,
        "rope_parameters": {"partial_rotary_factor": 0.25, "rope_theta": 10000000},
    },
}


def _signature(spec):
    """Slot structure and module names; modules compare by identity."""

    def mod(m):
        return type(m).__name__, m.name

    return spec.layer_type.name, [
        (s.name, s.branch_counts, [mod(b) for b in s.branches])
        if hasattr(s, "branches")
        else mod(s)
        for s in spec.layer_type.slots
    ]


def test_qwen4_exp_config_resolves_to_flash_and_builds_its_segment():
    assert find_model(HF_CONFIG) is qwen38.Qwen38Flash
    assert find_model("qwen4_exp") is qwen38.Qwen38Flash
    assert find_model("Qwen4ExpForCausalLM") is qwen38.Qwen38Flash

    g = qwen38.Qwen38Flash.derive_geometry(HF_CONFIG)
    assert g == qwen38.FULL

    spec = qwen38.Qwen38Flash.to_model_spec(HF_CONFIG, R=8)
    want = qwen38.segment_spec(qwen38.FULL, 8)
    assert _signature(spec) == _signature(want)


def test_other_checkpoints_do_not_resolve_to_flash():
    # Positive control: the Qwen3.5 config resolves, just not to Flash.
    qwen35_cfg = {
        "architectures": ["Qwen3_5MoeForCausalLM"],
        "model_type": "qwen3_5_moe",
    }
    assert find_model(qwen35_cfg) is qwen35.Qwen35
    assert find_model({"model_type": "qwen4_exp_v2"}) is None


def test_flash_has_no_library_pool_layout():
    with pytest.raises(MegakernelConfigError, match="unified pool"):
        qwen38.Qwen38Flash.pool_layout(HF_CONFIG, R=8)


def test_segment_composed_model_must_override_its_specs():
    with pytest.raises(MegakernelConfigError, match="must override to_model_spec"):

        @register_model(name="segment_composed_missing_spec")
        class _Missing(ModelDef):
            segment_composed = True

            @classmethod
            def derive_geometry(
                cls,
                hf_config=None,
                *,
                moe_tp: bool | None = None,
                e_slots: int | None = None,
            ):
                return None


def test_whole_model_driver_refuses_segment_composed_model(monkeypatch):
    pytest.importorskip("torch")
    from vllm_torchtpu.megakernel_runtime.driver import MegakernelDriver

    monkeypatch.setattr(MegakernelDriver, "_model_name", qwen38.QWEN38_FLASH_MODEL_NAME)
    with pytest.raises(MegakernelConfigError, match="segment-composed"):
        MegakernelDriver._model_def()
