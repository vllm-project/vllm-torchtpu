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

from types import SimpleNamespace

import pytest

from vllm_torchtpu.platforms import pp_validation
from vllm_torchtpu.platforms.pp_validation import validate_pipeline_parallel_config


def _config(
    pp=2,
    async_scheduling=False,
    speculative=None,
    kv_transfer=None,
    tp=1,
    layer_types=None,
    pcp=1,
    dp=1,
    ep=False,
):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pp,
            tensor_parallel_size=tp,
            prefill_context_parallel_size=pcp,
            data_parallel_size=dp,
            enable_expert_parallel=ep,
        ),
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        speculative_config=speculative,
        kv_transfer_config=kv_transfer,
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(layer_types=layer_types)
        ),
    )


RAIDEN = SimpleNamespace(kv_connector="TPURaidenConnector", is_kv_producer=True)
RAIDEN_CONSUMER = SimpleNamespace(
    kv_connector="TPURaidenConnector", is_kv_producer=False
)
OFFLOAD = SimpleNamespace(
    kv_connector="TPURaidenOffloadingConnector", is_kv_producer=True
)


def _multi(*children):
    return SimpleNamespace(
        kv_connector="TPUMultiConnector",
        is_kv_producer=True,
        kv_connector_extra_config={
            "connectors": [{"kv_connector": name} for name in children]
        },
    )


# Qwen3.5-0.8B: 24 layers, full attention on every fourth.
HYBRID_24 = ["full_attention" if i % 4 == 3 else "linear_attention" for i in range(24)]


def test_single_stage_accepts_everything():
    validate_pipeline_parallel_config(
        _config(pp=1, async_scheduling=True, speculative=object(), kv_transfer=object())
    )


def test_pipeline_parallel_accepts_sync_scheduling():
    validate_pipeline_parallel_config(_config(pp=2))


def test_rejects_async_scheduling_and_names_the_flag():
    with pytest.raises(
        NotImplementedError, match="async scheduling.*--no-async-scheduling"
    ):
        validate_pipeline_parallel_config(_config(pp=2, async_scheduling=True))


MTP_K1 = SimpleNamespace(method="mtp", num_speculative_tokens=1)


def test_accepts_prefill_only_mtp_k1_on_a_kv_producer():
    validate_pipeline_parallel_config(
        _config(pp=2, speculative=MTP_K1, kv_transfer=RAIDEN, layer_types=HYBRID_24)
    )


@pytest.mark.parametrize(
    "speculative, kv_transfer",
    [
        (SimpleNamespace(method="eagle3", num_speculative_tokens=1), RAIDEN),
        (SimpleNamespace(method="mtp", num_speculative_tokens=2), RAIDEN),
        # Without a consumer to hand the draft KV to, the drafts serve
        # nothing and every decode step would be refused at runtime.
        (MTP_K1, None),
        (MTP_K1, RAIDEN_CONSUMER),
    ],
)
def test_rejects_speculative_decoding_other_than_mtp_k1_on_a_producer(
    speculative, kv_transfer
):
    with pytest.raises(
        NotImplementedError, match="prefill-only MTP.*num_speculative_tokens=1"
    ):
        validate_pipeline_parallel_config(
            _config(pp=2, speculative=speculative, kv_transfer=kv_transfer)
        )


def test_rejects_kv_transfer_connectors_other_than_raiden():
    other = SimpleNamespace(kv_connector="TPUConnectorHMA")
    with pytest.raises(NotImplementedError, match="TPURaidenConnector"):
        validate_pipeline_parallel_config(_config(pp=2, kv_transfer=other))


def test_accepts_the_raiden_connector():
    validate_pipeline_parallel_config(_config(pp=2, kv_transfer=RAIDEN))


def test_accepts_the_raiden_offloading_connector():
    validate_pipeline_parallel_config(_config(pp=2, kv_transfer=OFFLOAD))


def test_accepts_a_multi_connector_of_raiden_connectors():
    validate_pipeline_parallel_config(
        _config(
            pp=2,
            kv_transfer=_multi("TPURaidenConnector", "TPURaidenOffloadingConnector"),
        )
    )


def test_rejects_a_multi_connector_with_another_child():
    with pytest.raises(NotImplementedError, match="TPUConnectorHMA"):
        validate_pipeline_parallel_config(
            _config(pp=2, kv_transfer=_multi("TPURaidenConnector", "TPUConnectorHMA"))
        )


def test_offloading_rejects_a_stage_of_linear_attention_layers_only(monkeypatch):
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    with pytest.raises(
        NotImplementedError, match=r"stage 0 holds layers 0-2, linear_attention only"
    ):
        validate_pipeline_parallel_config(
            _config(pp=8, kv_transfer=OFFLOAD, layer_types=HYBRID_24)
        )


def test_raiden_rejects_a_stage_of_linear_attention_layers_only(monkeypatch):
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    with pytest.raises(
        NotImplementedError,
        match=r"stage 0 holds layers 0-2, linear_attention "
        r"only.*VLLM_PP_LAYER_PARTITION",
    ):
        validate_pipeline_parallel_config(
            _config(pp=8, kv_transfer=RAIDEN, layer_types=HYBRID_24)
        )


def test_raiden_accepts_stages_that_each_hold_full_attention(monkeypatch):
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    validate_pipeline_parallel_config(
        _config(pp=4, kv_transfer=RAIDEN, layer_types=HYBRID_24)
    )


def test_raiden_partition_check_honors_the_layer_partition(monkeypatch):
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "4,4,4,4,4,4")
    validate_pipeline_parallel_config(
        _config(pp=6, kv_transfer=RAIDEN, layer_types=HYBRID_24)
    )


def test_linear_attention_stages_are_fine_without_kv_transfer():
    validate_pipeline_parallel_config(_config(pp=8, layer_types=HYBRID_24))


def test_rejects_tensor_parallel_stages():
    with pytest.raises(NotImplementedError, match="tensor-parallel-size 1"):
        validate_pipeline_parallel_config(_config(pp=2, tp=4))


@pytest.fixture
def stage_meshes(monkeypatch):
    """A torch_tpu that builds meshes over a stage's process group."""
    monkeypatch.setattr(pp_validation, "_torch_tpu_builds_stage_meshes", lambda: True)


def test_accepts_context_parallel_stages(stage_meshes):
    validate_pipeline_parallel_config(_config(pp=4, pcp=2))


def test_rejects_context_parallel_stages_on_world_size_meshes(monkeypatch):
    monkeypatch.setattr(pp_validation, "_torch_tpu_builds_stage_meshes", lambda: False)
    with pytest.raises(NotImplementedError, match="stages of several devices"):
        validate_pipeline_parallel_config(_config(pp=4, pcp=2, kv_transfer=RAIDEN))


def test_rejects_expert_parallel_stages_on_world_size_meshes(monkeypatch):
    monkeypatch.setattr(pp_validation, "_torch_tpu_builds_stage_meshes", lambda: False)
    with pytest.raises(NotImplementedError, match="stages of several devices"):
        validate_pipeline_parallel_config(_config(pp=4, dp=2, ep=True))


def test_pipeline_stages_of_one_device_skip_the_torch_tpu_check(monkeypatch):
    def unexpected():
        raise AssertionError("probed torch_tpu for a stage of one device")

    monkeypatch.setattr(pp_validation, "_torch_tpu_builds_stage_meshes", unexpected)
    validate_pipeline_parallel_config(_config(pp=8))
    validate_pipeline_parallel_config(_config(pp=4, dp=2))
    validate_pipeline_parallel_config(_config(pp=8, ep=True))


def test_rejects_other_kv_transfer_with_context_parallel_stages(stage_meshes):
    other = SimpleNamespace(kv_connector="TPUConnectorHMA")
    with pytest.raises(NotImplementedError, match="context-parallel stages"):
        validate_pipeline_parallel_config(_config(pp=4, pcp=2, kv_transfer=other))


def test_raiden_accepts_context_parallel_stages(stage_meshes):
    validate_pipeline_parallel_config(_config(pp=4, pcp=2, kv_transfer=RAIDEN))


def test_offloading_accepts_context_parallel_stages(stage_meshes):
    validate_pipeline_parallel_config(_config(pp=4, pcp=2, kv_transfer=OFFLOAD))


def test_rejects_the_legacy_ray_executor(monkeypatch):
    monkeypatch.setenv("TPU_MULTIHOST_BACKEND", "ray")
    monkeypatch.setenv("VLLM_USE_RAY_V2_EXECUTOR_BACKEND", "0")
    with pytest.raises(
        NotImplementedError, match="legacy Ray.*VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1"
    ):
        validate_pipeline_parallel_config(_config(pp=2))


def test_accepts_the_ray_v2_executor(monkeypatch):
    monkeypatch.setenv("TPU_MULTIHOST_BACKEND", "ray")
    monkeypatch.setenv("VLLM_USE_RAY_V2_EXECUTOR_BACKEND", "1")
    validate_pipeline_parallel_config(_config(pp=2))


def test_raiden_accepts_stages_of_sparse_attention_layers(monkeypatch):
    # GLM-5.2: every layer is sparse attention with its own latent cache.
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    validate_pipeline_parallel_config(
        _config(pp=8, kv_transfer=RAIDEN, layer_types=["deepseek_sparse_attention"] * 8)
    )
