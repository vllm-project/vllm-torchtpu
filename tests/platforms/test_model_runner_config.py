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
"""CPU contracts for TPU's runner validation and native vLLM validation."""

from types import SimpleNamespace

import pytest
from vllm.config import CacheConfig, CompilationConfig, VllmConfig
from vllm.config.compilation import CompilationMode, CUDAGraphMode

import vllm_torchtpu as plugin
from vllm_torchtpu.platforms.tpu_platform import _validate_tpu_runner_features


def _config(device="tpu", method=None, pcp=1):
    # Exercise the validators without loading a model or devices.
    config = object.__new__(VllmConfig)
    config.device_config = SimpleNamespace(device_type=device)
    config.model_config = None
    config.attention_config = SimpleNamespace(hisparse_config=None)
    config.parallel_config = SimpleNamespace(
        prefill_context_parallel_size=pcp,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        distributed_executor_backend="mp",
        enable_batch_sharded_sampling=False,
        enable_dbo=False,
        enable_elastic_ep=False,
        use_ubatching=False,
    )
    config.cache_config = CacheConfig()
    config.scheduler_config = SimpleNamespace(async_scheduling=None)
    config.compilation_config = CompilationConfig(mode=CompilationMode.NONE)
    config.speculative_config = (
        None
        if method is None
        else SimpleNamespace(
            method=method,
            enable_adaptive_verification=False,
            parallel_drafting=method in ("dflash", "dspark"),
            use_eagle=lambda: method in ("eagle", "eagle3", "mtp", "dflash", "dspark"),
            draft_lm_head_quantization=None,
            draft_model_config=SimpleNamespace(
                architectures=["DFlashDraftModel"],
                hf_config=SimpleNamespace(
                    layer_types=["sliding_attention", "full_attention"]
                ),
            ),
        )
    )
    return config


@pytest.fixture(autouse=True)
def _install_patch():
    plugin._patch_vllm_tpu_runner_validation()


@pytest.mark.parametrize("method,pcp", [(None, 2), ("dspark", 1), ("dflash", 1)])
def test_tpu_allows_what_tpu_model_runner_implements(method, pcp):
    # Prefill context parallelism, DSpark and mixed sliding/full DFlash drafts.
    _validate_tpu_runner_features(_config(method=method, pcp=pcp))


def test_tpu_allows_dspark_adaptive_verification():
    config = _config(method="dspark")
    config.speculative_config.enable_adaptive_verification = True
    _validate_tpu_runner_features(config)


@pytest.mark.parametrize(
    "feature,reason",
    [
        ("adaptive", "adaptive draft verification outside DSpark"),
        ("xpress", "xpress speculative decoding"),
        ("dflash2", "DFlash candidate-head drafts"),
        ("parse_mtp", "Parse 3.5 MTP"),
        ("lm_head_quantization", "draft lm_head quantization"),
        ("diffusion", "diffusion models"),
        ("step5", "Step5"),
        ("logits_processors", "custom logits processors"),
        ("batch_sharded", "batch-sharded sampling"),
    ],
)
def test_tpu_rejects_what_tpu_model_runner_lacks(feature, reason):
    # vLLM v0.33.0's V1 runner rejected these; v0.34.0 dropped that check.
    config = _config(method="dflash", pcp=2)
    spec = config.speculative_config
    if feature == "adaptive":
        spec.enable_adaptive_verification = True
    elif feature == "xpress":
        spec.draft_model_config.architectures = ["DFlashQwen3XPressModel"]
    elif feature == "dflash2":
        spec.draft_model_config.architectures = ["DFlash2DraftModel"]
    elif feature == "parse_mtp":
        spec.draft_model_config.architectures = ["ParseAuxMTPModel"]
    elif feature == "lm_head_quantization":
        spec.draft_lm_head_quantization = "fp8"
    elif feature == "diffusion":
        config.model_config = SimpleNamespace(
            is_diffusion=True, architecture="", logits_processors=None
        )
    elif feature == "step5":
        config.model_config = SimpleNamespace(
            is_diffusion=False, architecture="Step5ForCausalLM", logits_processors=None
        )
    elif feature == "logits_processors":
        # vLLM's frontend accepts only Model Runner V2 logits processors.
        config.model_config = SimpleNamespace(
            is_diffusion=False, architecture="", logits_processors=["pkg.mod:Proc"]
        )
    else:
        config.parallel_config.enable_batch_sharded_sampling = True
    with pytest.raises(ValueError, match=reason):
        _validate_tpu_runner_features(config)


@pytest.mark.parametrize("arch", ["DFlash2DraftModel", "Qwen3XPressModel"])
def test_tpu_allows_dflash_only_draft_architectures_under_other_methods(arch):
    # XPress refiners and DFlash candidate heads are rejected only under dflash.
    config = _config(method="eagle3")
    config.speculative_config.draft_model_config.architectures = [arch]
    _validate_tpu_runner_features(config)


def test_tpu_validate_model_runner_runs_the_tpu_runner_check():
    # vLLM's own runner check is where v0.33.0 ran the V1 one.
    config = _config()
    config.parallel_config.enable_batch_sharded_sampling = True
    with pytest.raises(ValueError, match="batch-sharded sampling"):
        config._validate_model_runner()


def test_non_tpu_validate_model_runner_skips_the_tpu_runner_check():
    config = _config(device="cuda")
    config.parallel_config.enable_batch_sharded_sampling = True
    config._validate_model_runner()


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("ngram", False),
        ("draft_model", False),
        ("pp", False),
        ("dflash", None),
        ("plain", None),
    ],
)
def test_tpu_post_init_disables_default_async_scheduling_where_unsupported(
    monkeypatch, mode, expected
):
    seen_before_upstream = []
    monkeypatch.setattr(VllmConfig, "_tpu_runner_validation_patch", False)
    # Re-installing wraps these again; restore them after the test.
    for name in ("_validate_model_runner", "_validate_adaptive_verification"):
        monkeypatch.setattr(VllmConfig, name, getattr(VllmConfig, name))
    monkeypatch.setattr(
        VllmConfig,
        "__post_init__",
        lambda self: seen_before_upstream.append(
            self.scheduler_config.async_scheduling
        ),
    )
    plugin._patch_vllm_tpu_runner_validation()

    method = mode if mode in ("ngram", "draft_model", "dflash") else None
    config = _config(method=method)
    if mode == "pp":
        config.parallel_config.pipeline_parallel_size = 2

    config.__post_init__()
    assert seen_before_upstream == [expected]

    # Explicit async_scheduling=True stays True so platform validators reject it.
    explicit = _config(method="ngram")
    explicit.scheduler_config.async_scheduling = True
    explicit.__post_init__()
    assert seen_before_upstream[-1] is True

    # Non-TPU configs keep vLLM's own default handling.
    non_tpu = _config(device="cuda", method="ngram")
    non_tpu.__post_init__()
    assert seen_before_upstream[-1] is None


@pytest.mark.parametrize("parser", [None, "cli"])
def test_pre_register_installs_patches_before_the_first_config(monkeypatch, parser):
    # vLLM calls pre_register_and_update before EngineArgs builds a VllmConfig.
    from vllm_torchtpu import patch_registry
    from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

    stages = []
    monkeypatch.setattr(patch_registry, "apply", stages.append)
    cli = SimpleNamespace(set_defaults=lambda **_: None) if parser else None
    TpuPlatform.pre_register_and_update(cli)
    assert stages == ["pre_register"]


def _adaptive_dspark_config(device):
    config = _config(device=device, method="dspark")
    config.speculative_config.enable_adaptive_verification = True
    config.lora_config = None
    config.model_config = SimpleNamespace(
        is_diffusion=False, enforce_eager=False, architecture=""
    )
    config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    return config


def test_tpu_runs_dspark_adaptive_verification():
    # TPU has no CUDA graphs; its own checks replace the native one.
    _adaptive_dspark_config("tpu")._validate_adaptive_verification()


def test_non_tpu_keeps_native_adaptive_verification_checks():
    config = _adaptive_dspark_config("cuda")
    with pytest.raises(ValueError, match="full CUDA graphs"):
        config._validate_adaptive_verification()


def test_tpu_adaptive_verification_keeps_the_lora_check():
    config = _adaptive_dspark_config("tpu")
    config.lora_config = object()
    with pytest.raises(ValueError, match="LoRA"):
        config._validate_adaptive_verification()


@pytest.mark.parametrize("window", [128, None])
def test_mixed_dflash_layer_attention_needs_no_tpu_patch(window):
    """vLLM v0.34.0 no longer limits mixed sliding/full DFlash drafts to Model
    Runner V2, so TPU resolves their layer attention with the upstream code."""
    from transformers import Qwen3Config
    from vllm.model_executor.models import qwen3_dflash

    draft = Qwen3Config(
        num_hidden_layers=2,
        layer_types=["sliding_attention", "full_attention"],
        use_sliding_window=True,
        sliding_window=window,
    )
    if window is None:
        with pytest.raises(ValueError, match="requires a window size"):
            qwen3_dflash._resolve_layer_attention(draft, 0)
    else:
        assert qwen3_dflash._resolve_layer_attention(draft, 0) == (128, True)
    assert qwen3_dflash._resolve_layer_attention(draft, 1) == (None, False)
