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
"""`validate_pcp_manager_config`: sparse MLA + headtail and Qwen GDN hybrids +
interleaved, and the host plan / kernel `sharding_mode` agreeing."""

from types import SimpleNamespace

import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter.pcp_attention import pcp_kernel_sharding_mode
from vllm_torchtpu.runner.pcp.manager import validate_pcp_manager_config

QWEN35 = "Qwen3_5MoeForConditionalGeneration"


def _config(
    *,
    architecture=QWEN35,
    sparse_mla=False,
    tp=1,
    pcp=4,
    dcp=4,
    interleave=256,
    kv_transfer_config=None,
    head_dim=256,
    sliding_window=None,
    language_model_only=True,
    speculative_config=None,
):
    hf_config = SimpleNamespace(head_dim=head_dim)
    if sparse_mla:
        hf_config.index_topk = 2048
    return SimpleNamespace(
        model_config=SimpleNamespace(
            architecture=architecture,
            use_mla=sparse_mla,
            hf_text_config=hf_config,
            multimodal_config=SimpleNamespace(language_model_only=language_model_only),
            get_sliding_window=lambda: sliding_window,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            prefill_context_parallel_size=pcp,
            decode_context_parallel_size=dcp,
            cp_kv_cache_interleave_size=interleave,
            is_moe_model=True,
            enable_expert_parallel=True,
            pipeline_parallel_size=1,
            data_parallel_size=1,
        ),
        kv_transfer_config=kv_transfer_config,
        speculative_config=speculative_config,
        lora_config=None,
    )


@pytest.fixture
def plan(monkeypatch, request):
    monkeypatch.setattr(envs, "TPU_PCP_SHARDING_PLAN", request.param)
    monkeypatch.setattr(envs, "TPU_KV_RESHARD_TRANSPORT", "zmq")
    monkeypatch.setattr(envs, "TPU_USE_RAIDEN_KV_CACHE_MANAGER", False)
    return request.param


@pytest.mark.parametrize("plan", ["interleaved"], indirect=True)
def test_accepts_qwen_gdn_interleaved(plan):
    validate_pcp_manager_config(_config())


def _sparse_mla_config(dcp=8):
    config = _config(
        architecture="GlmMoeDsaForCausalLM", sparse_mla=True, tp=2, dcp=dcp
    )
    config.model_config.multimodal_config = None
    return config


@pytest.mark.parametrize("plan", ["headtail"], indirect=True)
def test_accepts_sparse_mla_headtail(plan):
    validate_pcp_manager_config(_sparse_mla_config())


@pytest.mark.parametrize(
    ("plan", "config", "message"),
    [
        ("headtail", _config(), "only supports"),
        ("interleaved", _sparse_mla_config(), "only supports"),
        ("interleaved", _config(architecture="KimiK3ForCausalLM"), "only supports"),
        ("interleaved", _config(architecture="LlamaForCausalLM"), "only supports"),
        ("interleaved", _config(pcp=1, dcp=1), "prefill_context_parallel_size > 1"),
        ("interleaved", _config(dcp=1), "decode_context_parallel_size =="),
        ("interleaved", _config(interleave=0), "cp_kv_cache_interleave_size > 0"),
        (
            "interleaved",
            _config(kv_transfer_config=SimpleNamespace()),
            "KV transfer",
        ),
        ("interleaved", _config(head_dim=64), "head_dim == 64"),
        ("interleaved", _config(sliding_window=4096), "full attention only"),
        ("interleaved", _config(language_model_only=False), "multimodal"),
        (
            "interleaved",
            _config(speculative_config=SimpleNamespace()),
            "speculative decoding",
        ),
        ("headtail", _sparse_mla_config(dcp=4), "dcp=4"),
    ],
    indirect=["plan"],
)
def test_rejects(plan, config, message):
    with pytest.raises(NotImplementedError, match=message):
        validate_pcp_manager_config(config)


@pytest.mark.parametrize("plan", ["interleaved"], indirect=True)
@pytest.mark.parametrize(
    "env", ["TPU_KV_RESHARD_TRANSPORT", "TPU_USE_RAIDEN_KV_CACHE_MANAGER"]
)
def test_interleaved_rejects_raiden(plan, env, monkeypatch):
    monkeypatch.setattr(
        envs, env, "raiden" if env == "TPU_KV_RESHARD_TRANSPORT" else True
    )
    with pytest.raises(NotImplementedError, match="KV transfer"):
        validate_pcp_manager_config(_config())


@pytest.mark.parametrize(
    ("sharding_plan", "sharding_mode"),
    [
        ("headtail", "head_tail"),
        ("interleaved", "interleaved"),
        ("Interleaved", "interleaved"),
    ],
)
def test_kernel_sharding_mode_follows_the_host_plan_flag(sharding_plan, sharding_mode):
    assert pcp_kernel_sharding_mode(sharding_plan) == sharding_mode
