# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from vllm_torchtpu.platforms.pp_validation import validate_pipeline_parallel_config


def _config(
    pp=2,
    async_scheduling=False,
    speculative=None,
    kv_transfer=None,
    tp=1,
    layer_types=None,
):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pp, tensor_parallel_size=tp
        ),
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        speculative_config=speculative,
        kv_transfer_config=kv_transfer,
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(layer_types=layer_types)
        ),
    )


RAIDEN = SimpleNamespace(kv_connector="TPURaidenConnector")
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


def test_rejects_kv_transfer_connectors_other_than_raiden():
    other = SimpleNamespace(kv_connector="TPUConnectorHMA")
    with pytest.raises(NotImplementedError, match="TPURaidenConnector"):
        validate_pipeline_parallel_config(_config(pp=2, kv_transfer=other))


def test_accepts_the_raiden_connector():
    validate_pipeline_parallel_config(_config(pp=2, kv_transfer=RAIDEN))


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
