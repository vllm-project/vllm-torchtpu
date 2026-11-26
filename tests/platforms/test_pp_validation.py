# SPDX-License-Identifier: Apache-2.0

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
    hf_config=None,
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
            hf_text_config=hf_config or SimpleNamespace(layer_types=layer_types)
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


def test_rejects_speculative_decoding():
    with pytest.raises(NotImplementedError, match="speculative decoding"):
        validate_pipeline_parallel_config(_config(pp=2, speculative=object()))


MTP1 = SimpleNamespace(method="mtp", num_speculative_tokens=1)


@pytest.fixture
def pp_mtp_flag(monkeypatch):
    from vllm_torchtpu import envs

    monkeypatch.setattr(envs, "TPU_PP_MTP_PREFILL_KV", True)


def test_admits_mtp_for_prefill_kv_on_a_producer(pp_mtp_flag):
    config = _config(pp=2, speculative=MTP1, kv_transfer=OFFLOAD)
    validate_pipeline_parallel_config(config)
    assert pp_validation.pp_mtp_prefill_kv_only(config)


def test_admits_mtp_for_prefill_kv_on_the_disaggregation_producer(pp_mtp_flag):
    validate_pipeline_parallel_config(
        _config(pp=2, speculative=MTP1, kv_transfer=RAIDEN)
    )


def test_mtp_without_the_flag_is_refused_and_names_it():
    with pytest.raises(
        NotImplementedError, match="speculative decoding.*TPU_PP_MTP_PREFILL_KV=1"
    ):
        validate_pipeline_parallel_config(
            _config(pp=2, speculative=MTP1, kv_transfer=OFFLOAD)
        )


@pytest.mark.parametrize(
    "speculative, kv_transfer, match",
    [
        (
            SimpleNamespace(method="eagle3", num_speculative_tokens=1),
            OFFLOAD,
            "method 'eagle3'",
        ),
        (
            SimpleNamespace(method="mtp", num_speculative_tokens=3),
            OFFLOAD,
            "num_speculative_tokens != 1",
        ),
        (MTP1, None, "without a KV producer connector"),
        (MTP1, RAIDEN_CONSUMER, "without a KV producer connector"),
    ],
)
def test_flagged_mtp_is_refused_outside_prefill_kv(
    pp_mtp_flag, speculative, kv_transfer, match
):
    with pytest.raises(NotImplementedError, match=match):
        validate_pipeline_parallel_config(
            _config(pp=2, speculative=speculative, kv_transfer=kv_transfer)
        )


def test_flagged_mtp_still_refuses_async_scheduling(pp_mtp_flag):
    with pytest.raises(NotImplementedError, match="async scheduling"):
        validate_pipeline_parallel_config(
            _config(pp=2, async_scheduling=True, speculative=MTP1, kv_transfer=OFFLOAD)
        )


def test_single_stage_mtp_is_not_prefill_kv_only():
    assert not pp_validation.pp_mtp_prefill_kv_only(_config(pp=1, speculative=MTP1))
    assert not pp_validation.pp_mtp_prefill_kv_only(_config(pp=2))


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


def test_rejects_tensor_parallel_stages_on_world_size_meshes(monkeypatch):
    monkeypatch.setattr(pp_validation, "_torch_tpu_builds_stage_meshes", lambda: False)
    with pytest.raises(NotImplementedError, match="tensor-parallel-size 1"):
        validate_pipeline_parallel_config(_config(pp=2, tp=4))


@pytest.fixture
def stage_meshes(monkeypatch):
    """A torch_tpu that builds meshes over a stage's process group."""
    monkeypatch.setattr(pp_validation, "_torch_tpu_builds_stage_meshes", lambda: True)


def test_accepts_tensor_parallel_stages(stage_meshes):
    validate_pipeline_parallel_config(_config(pp=2, tp=4))


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


def _glm52(**overrides):
    # GLM-5.2: 78 layers; layers 0-2 and every fourth from 6 run their indexer.
    fields = dict(
        layer_types=None,
        num_hidden_layers=78,
        index_topk=2048,
        index_topk_freq=4,
        index_skip_topk_offset=3,
        index_topk_pattern=None,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_rejects_a_stage_starting_on_a_shared_topk_layer(monkeypatch):
    # The default 39/39 split starts stage 1 at layer 39, which reuses layer
    # 38's top-k.
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    with pytest.raises(
        NotImplementedError,
        match=r"reuses the top-k \(stage 1 starts at layer 39\)"
        r".*VLLM_PP_LAYER_PARTITION",
    ):
        validate_pipeline_parallel_config(_config(pp=2, hf_config=_glm52()))


@pytest.mark.parametrize("partition", ["42,36", "38,40"])
def test_accepts_stages_starting_on_an_indexer_layer(monkeypatch, partition):
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", partition)
    validate_pipeline_parallel_config(_config(pp=2, hf_config=_glm52()))


def test_topk_pattern_overrides_the_frequency(monkeypatch):
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "2,2")
    pattern = ["F", "S", "S", "F"]
    with pytest.raises(NotImplementedError, match="starts at layer 2"):
        validate_pipeline_parallel_config(
            _config(
                pp=2,
                hf_config=_glm52(num_hidden_layers=4, index_topk_pattern=pattern),
            )
        )


def test_models_without_an_indexer_skip_the_topk_check(monkeypatch):
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    plain = SimpleNamespace(layer_types=None, num_hidden_layers=78)
    validate_pipeline_parallel_config(_config(pp=2, hf_config=plain))
