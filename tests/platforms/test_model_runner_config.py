# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for TPU's runner selection and native vLLM validation."""
from types import SimpleNamespace

import pytest
from vllm.config import CacheConfig, CompilationConfig, VllmConfig
from vllm.config.compilation import CompilationMode

import vllm_torchtpu as plugin

pytestmark = pytest.mark.cpu_test


def _config(device="tpu", method=None, pcp=1):
    # Exercise the native validator/property without loading a model or devices.
    config = object.__new__(VllmConfig)
    config.device_config = SimpleNamespace(device_type=device)
    config.model_config = None
    config.parallel_config = SimpleNamespace(
        prefill_context_parallel_size=pcp,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        distributed_executor_backend="mp",
        enable_batch_sharded_sampling=False,
        enable_dbo=False,
        enable_elastic_ep=False,
    )
    config.cache_config = CacheConfig()
    config.compilation_config = CompilationConfig(mode=CompilationMode.NONE)
    config.speculative_config = None if method is None else SimpleNamespace(
        method=method,
        enable_adaptive_verification=False,
        parallel_drafting=method in ("dflash", "dspark"),
        draft_model_config=SimpleNamespace(
            architectures=["DFlashDraftModel"],
            hf_config=SimpleNamespace(
                layer_types=["sliding_attention", "full_attention"]),
        ),
    )
    return config


@pytest.fixture(autouse=True)
def _install_patch(monkeypatch):
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    plugin._patch_vllm_force_v1_runner_tpu()


@pytest.mark.parametrize("device", ["tpu", "cuda"])
@pytest.mark.parametrize("method,pcp,reason", [
    (None, 2, "prefill context parallel"),
    ("dspark", 1, "dspark speculative decoding"),
    ("dflash", 1, "mixed sliding/full dflash drafts"),
])
def test_native_v1_validation_uses_tpu_capabilities(device, method, pcp,
                                                    reason):
    config = _config(device=device, method=method, pcp=pcp)
    if device == "tpu":
        config._validate_v1_model_runner()
    else:
        with pytest.raises(ValueError, match=reason):
            config._validate_v1_model_runner()


@pytest.mark.parametrize("feature,reason", [
    ("adaptive", "adaptive draft verification"),
    ("dflash2", "dflash2 drafts"),
    ("diffusion", "diffusion models"),
    ("batch_sharded", "batch-sharded sampling"),
])
def test_tpu_preserves_native_v1_rejections(feature, reason):
    config = _config(method="dflash", pcp=2)
    if feature == "adaptive":
        config.speculative_config.enable_adaptive_verification = True
    elif feature == "dflash2":
        config.speculative_config.draft_model_config.architectures = [
            "DFlash2DraftModel"
        ]
    elif feature == "diffusion":
        config.model_config = SimpleNamespace(is_diffusion=True)
    else:
        config.parallel_config.enable_batch_sharded_sampling = True
    with pytest.raises(ValueError, match=reason):
        config._validate_v1_model_runner()


@pytest.mark.parametrize("has_triton", [False, True])
@pytest.mark.parametrize("method", [None, "dflash", "dspark"])
def test_tpu_defaults_to_v1_even_when_triton_is_available(
        monkeypatch, has_triton, method):
    monkeypatch.setattr("vllm.config.vllm.HAS_TRITON", has_triton)
    config = _config(method=method)
    assert config.use_v2_model_runner is False


@pytest.mark.parametrize("value,expected", [("0", False), ("1", True)])
@pytest.mark.parametrize("device", ["tpu", "cuda"])
def test_explicit_runner_selection_keeps_native_precedence(
        monkeypatch, value, expected, device):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", value)
    assert _config(device=device).use_v2_model_runner is expected


@pytest.mark.parametrize("has_triton", [False, True])
def test_non_tpu_default_runner_selection_is_unchanged(monkeypatch,
                                                       has_triton):
    monkeypatch.setattr("vllm.config.vllm.HAS_TRITON", has_triton)
    assert _config(device="cuda").use_v2_model_runner is has_triton


def test_tpu_preserves_future_native_rejection(monkeypatch):

    class FutureVllmConfig(VllmConfig):
        _tpu_force_v1_runner_patch = False

        def _get_v1_model_runner_unsupported_features(self):
            return [
                *super()._get_v1_model_runner_unsupported_features(),
                "future unsupported feature",
            ]

    monkeypatch.setattr("vllm.config.vllm.VllmConfig", FutureVllmConfig)
    plugin._patch_vllm_force_v1_runner_tpu()
    config = _config(method="dflash")
    config.__class__ = FutureVllmConfig
    with pytest.raises(ValueError, match="future unsupported feature"):
        config._validate_v1_model_runner()


@pytest.mark.parametrize("window", [128, None])
def test_tpu_mixed_dflash_keeps_layer_attention_contract(window):
    from transformers import Qwen3Config
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.models import qwen3_dflash

    plugin._patch_dflash_bypass_v2_runner_check()
    config = _config(method="dflash")
    draft = Qwen3Config(num_hidden_layers=2,
                        layer_types=["sliding_attention", "full_attention"],
                        use_sliding_window=True,
                        sliding_window=window)
    with set_current_vllm_config(config):
        config._validate_v1_model_runner()
        assert config.use_v2_model_runner is False
        if window is None:
            with pytest.raises(ValueError, match="requires a window size"):
                qwen3_dflash._resolve_layer_attention(draft, 0)
        else:
            assert qwen3_dflash._resolve_layer_attention(draft,
                                                         0) == (128, True)
        assert qwen3_dflash._resolve_layer_attention(draft, 1) == (None, False)
        # The existing model-level capability shim must restore the selector
        # on both its successful and its exceptional paths.
        assert config.use_v2_model_runner is False
