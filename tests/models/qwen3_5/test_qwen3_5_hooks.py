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
"""Tests for the Qwen3.5 config hook (models/qwen3_5/hooks.py)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from vllm.model_executor.models.config import (
    MODELS_CONFIG_MAP,
    Qwen3_5ForConditionalGenerationConfig,
)

from vllm_torchtpu import additional_config_sections
from vllm_torchtpu.models.qwen3_5 import settings
from vllm_torchtpu.models.qwen3_5.hooks import TPUQwen35MoeConfig
from vllm_torchtpu.models.vllm import register_models

QWEN35 = "Qwen3_5MoeForConditionalGeneration"


@pytest.fixture
def restore_config_map():
    saved = dict(MODELS_CONFIG_MAP)
    yield
    MODELS_CONFIG_MAP.clear()
    MODELS_CONFIG_MAP.update(saved)


def test_hook_is_registered_for_qwen3_5_only(restore_config_map):
    register_models()
    assert frozenset({QWEN35}) == settings.ARCHITECTURES
    assert MODELS_CONFIG_MAP[QWEN35] is TPUQwen35MoeConfig
    # vLLM's own Qwen3.5 hook stays in charge of the sibling architectures.
    for sibling in (
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5ForCausalLM",
        "Qwen3_5MoeForCausalLM",
    ):
        assert not issubclass(MODELS_CONFIG_MAP[sibling], TPUQwen35MoeConfig)


def test_hook_extends_vllms_qwen3_5_hook():
    assert issubclass(TPUQwen35MoeConfig, Qwen3_5ForConditionalGenerationConfig)


# Qwen3.5-397B-A17B's text config. hidden x moe_intermediate = 4096 x 1024 is
# what the IX record jobs' TP8 decode legs logged as their budget: 33.5e9 B plus
# 716 slot-layers x 3 x 4096 x 1024 B = 42,509,364,992 B.
HIDDEN, INTER, EXPERTS, LAYERS = 4096, 1024, 512, 60
SLOT_LAYER_BYTES = 3 * HIDDEN * INTER
# Main's placement only permutes experts: 512 / EP8 = 64 slots per rank, 16
# fewer than the 80 the budget was measured with, in each of the 60 layers.
CREDIT_EP8 = (80 - 64) * LAYERS * SLOT_LAYER_BYTES


def _vllm_config(
    arch: str = QWEN35,
    *,
    kv_role: str | None = "kv_consumer",
    kv_cache_memory_bytes: int | None = None,
    prefix_match_unit: int | None = None,
    tp: int = 8,
    dp: int = 1,
    dcp: int = 4,
    max_num_seqs: int = 8,
    additional: dict | None = None,
) -> SimpleNamespace:
    text = SimpleNamespace(
        hidden_size=HIDDEN,
        moe_intermediate_size=INTER,
        num_experts=EXPERTS,
        num_hidden_layers=LAYERS,
        mamba_ssm_dtype="float32",
        rope_parameters={"mrope_section": [11, 11, 10], "rope_theta": 1e7},
    )
    model_config = SimpleNamespace(
        architecture=arch,
        architectures=[arch],
        hf_text_config=text,
        config_updated=False,
        is_hybrid=False,
        convert_type="none",
    )
    return SimpleNamespace(
        model_config=model_config,
        cache_config=SimpleNamespace(
            kv_cache_memory_bytes=kv_cache_memory_bytes,
            prefix_match_unit=prefix_match_unit,
            prefix_cache_retention_interval=0,
            mamba_ssm_cache_dtype="auto",
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs, scheduler_cls=None),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            data_parallel_size=dp,
            decode_context_parallel_size=dcp,
        ),
        kv_transfer_config=(
            None if kv_role is None else SimpleNamespace(kv_role=kv_role)
        ),
        load_config=SimpleNamespace(load_format="auto"),
        additional_config={} if additional is None else additional,
    )


def _run_hook(cfg) -> SimpleNamespace:
    TPUQwen35MoeConfig.verify_and_update_config(cfg)
    return cfg


def test_tp8_dcp4_decode_leg_gets_the_calibrated_budget():
    cfg = _run_hook(_vllm_config())
    assert cfg.cache_config.kv_cache_memory_bytes == 33_500_000_000 + CREDIT_EP8
    assert cfg.cache_config.kv_cache_memory_bytes == 45_579_595_520


def test_dcp1_with_more_than_16_seqs_uses_the_smaller_base():
    cfg = _run_hook(_vllm_config(tp=2, dp=4, dcp=1, max_num_seqs=28))
    # EP = TP x DP = 8, so the same 64 slots per rank.
    assert cfg.cache_config.kv_cache_memory_bytes == 32_500_000_000 + CREDIT_EP8


@pytest.mark.parametrize("dcp,seqs", [(1, 16), (4, 28), (1, 1)])
def test_other_shapes_use_the_larger_base(dcp, seqs):
    cfg = _run_hook(_vllm_config(dcp=dcp, max_num_seqs=seqs))
    assert cfg.cache_config.kv_cache_memory_bytes == 33_500_000_000 + CREDIT_EP8


def test_more_slots_than_the_baseline_take_hbm():
    # EP4 holds 128 slots per rank: 48 more than the baseline.
    cfg = _run_hook(_vllm_config(tp=4, dcp=1, max_num_seqs=8))
    expected = 33_500_000_000 - (128 - 80) * LAYERS * SLOT_LAYER_BYTES
    assert cfg.cache_config.kv_cache_memory_bytes == expected


@pytest.mark.parametrize("kv_role", ["kv_consumer", None, "kv_producer", "kv_both"])
def test_explicit_kv_cache_memory_bytes_is_kept(kv_role):
    cfg = _run_hook(_vllm_config(kv_role=kv_role, kv_cache_memory_bytes=39_500_000_000))
    assert cfg.cache_config.kv_cache_memory_bytes == 39_500_000_000


def test_kv_producer_stays_on_gpu_memory_utilization():
    cfg = _run_hook(_vllm_config(kv_role="kv_producer", tp=1, dcp=1, max_num_seqs=32))
    assert cfg.cache_config.kv_cache_memory_bytes is None


@pytest.mark.parametrize("kv_role", [None, "kv_both"])
def test_mixed_engine_gets_the_budget(kv_role):
    cfg = _run_hook(_vllm_config(kv_role=kv_role))
    assert cfg.cache_config.kv_cache_memory_bytes == 33_500_000_000 + CREDIT_EP8


def test_credit_is_zero_when_the_text_config_lacks_moe_fields():
    cfg = _vllm_config()
    del cfg.model_config.hf_text_config.moe_intermediate_size
    _run_hook(cfg)
    assert cfg.cache_config.kv_cache_memory_bytes == 33_500_000_000


def test_kv_producer_gets_a_128_token_prefix_match_unit():
    cfg = _run_hook(_vllm_config(kv_role="kv_producer"))
    assert cfg.cache_config.prefix_match_unit == 128


@pytest.mark.parametrize("kv_role", ["kv_consumer", None, "kv_both"])
def test_other_roles_keep_the_default_prefix_match_unit(kv_role):
    cfg = _run_hook(_vllm_config(kv_role=kv_role))
    assert cfg.cache_config.prefix_match_unit is None


@pytest.mark.parametrize("unit", [1, 64, 256])
def test_explicit_prefix_match_unit_is_kept(unit):
    cfg = _run_hook(_vllm_config(kv_role="kv_producer", prefix_match_unit=unit))
    assert cfg.cache_config.prefix_match_unit == unit


def test_retention_interval_and_scheduler_are_not_touched():
    cfg = _vllm_config(kv_role="kv_producer")
    cfg.cache_config.prefix_cache_retention_interval = 1280
    _run_hook(cfg)
    assert cfg.cache_config.prefix_cache_retention_interval == 1280
    assert cfg.scheduler_config.scheduler_cls is None


def test_vllms_qwen3_5_check_still_runs():
    cfg = _run_hook(_vllm_config())
    assert cfg.cache_config.mamba_ssm_cache_dtype == "float32"


def test_unknown_setting_fails_at_engine_start():
    with pytest.raises(ValueError, match="Unknown Qwen3.5 setting"):
        _run_hook(_vllm_config(additional={"qwen3_5": {"typo": True}}))


def _snapshot(obj):
    if isinstance(obj, SimpleNamespace):
        return {k: _snapshot(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {k: _snapshot(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_snapshot(v) for v in obj]
    return obj


# The Qwen3.5 siblings vLLM itself registers: the closest architectures our
# registration could capture by mistake.
OTHER_ARCHS = (
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5ForCausalLM",
    "Qwen3_5MoeForCausalLM",
)


@pytest.mark.parametrize("arch", OTHER_ARCHS)
@pytest.mark.parametrize("kv_role", ["kv_consumer", "kv_producer", None])
def test_other_models_resolve_exactly_as_on_main(restore_config_map, arch, kv_role):
    """Base vs head: vLLM's own dispatch (VllmConfig.try_verify_and_update_config)
    over the config map main has (vLLM's Qwen3.5 hook for our architecture) and
    over the map after register_models(). A model that is not ours must come
    out identical, field for field."""
    from vllm.config import VllmConfig

    register_models()
    head_map = dict(MODELS_CONFIG_MAP)
    base_map = dict(head_map)
    for ours in settings.ARCHITECTURES:
        base_map[ours] = Qwen3_5ForConditionalGenerationConfig

    results = []
    for config_map in (base_map, head_map):
        MODELS_CONFIG_MAP.clear()
        MODELS_CONFIG_MAP.update(config_map)
        cfg = _vllm_config(arch, kv_role=kv_role)
        VllmConfig.try_verify_and_update_config(cfg)
        results.append(_snapshot(cfg))
    assert results[0] == results[1]


def test_our_model_resolves_through_the_hook(restore_config_map):
    from vllm.config import VllmConfig

    register_models()
    cfg = _vllm_config()
    VllmConfig.try_verify_and_update_config(cfg)
    assert cfg.cache_config.kv_cache_memory_bytes == 33_500_000_000 + CREDIT_EP8


# Names no change will register: real section names (tpu_scheduler, ...) stop
# being "unread" once their change lands in the same tree.
@pytest.mark.parametrize("key", ["tpu_typo", "tpu_not_a_section"])
def test_unread_tpu_section_fails_at_engine_start(key):
    """A tpu_* section no reader in this build claims would be ignored by
    vLLM; the hook refuses it and names the sections it reads."""
    assert key not in additional_config_sections.KNOWN_SECTIONS
    with pytest.raises(ValueError, match=rf"section\(s\) \['{key}'\].*qwen3_5"):
        _run_hook(_vllm_config(additional={key: {"x": 1}}))


def test_known_and_foreign_sections_pass():
    cfg = _vllm_config(
        additional={
            "qwen3_5": {},
            "tpu_online_fp8": {},
            # Outside the qwen3_5 / tpu_* namespaces: main's keys and other
            # plugins' keys keep vLLM's pass-through behaviour.
            "abort_on_preemption": False,
            "megakernel": {},
            "phased_profiler_decode_only_kv_len_threshold": 1,
            "some_other_plugin": {"a": 1},
        }
    )
    _run_hook(cfg)
    assert cfg.cache_config.kv_cache_memory_bytes == 33_500_000_000 + CREDIT_EP8


@pytest.mark.parametrize("arch", OTHER_ARCHS)
def test_other_models_are_not_checked(restore_config_map, arch):
    from vllm.config import VllmConfig

    register_models()
    cfg = _vllm_config(arch, additional={"tpu_typo": {}})
    VllmConfig.try_verify_and_update_config(cfg)
