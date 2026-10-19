# SPDX-License-Identifier: Apache-2.0
"""Serving policy validation, serialization and legacy conflict contracts."""

import copy
import json
import os
from types import SimpleNamespace

import pytest

from vllm_torchtpu.models.vllm.deepseek_v41.config import (
    DeepseekV41Config,
    ExecutionMode,
    MoEBackend,
    verify_and_update_config,
)

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in os.environ:
        if name.startswith(("TPU_DEEPSEEK_V41_", "MOE_", "USE_MOE_")):
            monkeypatch.delenv(name)


def config(**policy):
    return SimpleNamespace(
        additional_config={"deepseek_v41": policy, "unrelated": 17},
        engram_config=SimpleNamespace(cpu_offload=True),
        parallel_config=SimpleNamespace(
            enable_expert_parallel=True, prefill_context_parallel_size=1
        ),
        kernel_config=SimpleNamespace(moe_backend="auto"),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(expert_dtype="fp4")),
        cache_config=SimpleNamespace(
            enable_prefix_caching=True, swa_bounded_replay=True
        ),
    )


@pytest.mark.parametrize("execution", ["full", "ced"])
@pytest.mark.parametrize("moe", ["gmm", "fused_ep_w4a8"])
def test_policy_round_trips_to_worker_and_config_hash_input(execution, moe):
    cfg = config(execution=execution, moe=moe)
    verify_and_update_config(cfg)
    before = json.dumps(cfg.additional_config, sort_keys=True)
    worker = copy.deepcopy(cfg)
    verify_and_update_config(worker)
    assert json.dumps(worker.additional_config, sort_keys=True) == before
    assert DeepseekV41Config.from_vllm_config(worker) == DeepseekV41Config.parse(
        dict(execution=execution, moe=moe)
    )
    assert worker.additional_config["unrelated"] == 17


@pytest.mark.parametrize(
    "policy",
    [{"moe": "adaptive"}, {"tile": 48}, {"execution": True}, {"engram": "gpu"}],
)
def test_rejects_unknown_or_mistyped_options(policy):
    with pytest.raises((ValueError, TypeError)):
        verify_and_update_config(config(**policy))


def test_defaults_are_explicit_and_do_not_enable_approximation():
    cfg = config()
    verify_and_update_config(cfg)
    assert cfg.additional_config["deepseek_v41"] == {"execution": "full", "moe": "gmm"}


def test_legacy_values_are_frozen_before_worker_env_changes(monkeypatch):
    cfg = config()
    del cfg.additional_config["deepseek_v41"]
    monkeypatch.setenv("TPU_DEEPSEEK_V41_CED", "1")
    verify_and_update_config(cfg)
    monkeypatch.delenv("TPU_DEEPSEEK_V41_CED")
    assert DeepseekV41Config.from_vllm_config(cfg).execution == ExecutionMode.CED


def test_conflicting_legacy_env_is_rejected(monkeypatch):
    monkeypatch.setenv("TPU_DEEPSEEK_V41_CED", "1")
    with pytest.raises(ValueError, match="conflicts"):
        verify_and_update_config(config(execution="full"))


@pytest.mark.parametrize(
    "name,value",
    [
        ("MOE_FUSED_EP_KERNEL_IMPL", "adaptive"),
        ("MOE_FUSED_EP_V2_SHARDED_PLAN", "0"),
        ("MOE_FUSED_EP_V2_HEIGHT_RUNGS_OVERRIDE", "2"),
        ("MOE_REQUANTIZE_BLOCK_SIZE", "256"),
    ],
)
def test_explicit_backend_cannot_be_reconfigured_by_hidden_env(
    monkeypatch, name, value
):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="remove"):
        verify_and_update_config(config(moe="fused_ep_w4a8"))


def test_explicit_backend_requires_ep_and_fp4():
    cfg = config(moe="fused_ep_w4a8")
    cfg.parallel_config.enable_expert_parallel = False
    with pytest.raises(ValueError, match="expert-parallel"):
        verify_and_update_config(cfg)
    cfg.parallel_config.enable_expert_parallel = True
    cfg.model_config.hf_config.expert_dtype = "fp8"
    with pytest.raises(ValueError, match="FP4"):
        verify_and_update_config(cfg)


def test_ced_requires_bounded_prefix_replay():
    cfg = config(execution="ced")
    cfg.cache_config.swa_bounded_replay = False
    with pytest.raises(ValueError, match="swa-bounded-replay"):
        verify_and_update_config(cfg)


def test_backend_selection_changes_serialized_config():
    cfg = config(moe="gmm")
    verify_and_update_config(cfg)
    baseline = json.dumps(cfg.additional_config, sort_keys=True)
    cfg.additional_config["deepseek_v41"]["moe"] = MoEBackend.FUSED_EP_W4A8
    verify_and_update_config(cfg)
    assert json.dumps(cfg.additional_config, sort_keys=True) != baseline


@pytest.mark.parametrize("offload", [False, True])
def test_standard_engram_config_controls_placement(offload):
    from vllm_torchtpu.models.vllm.deepseek_v41.config import engram_cpu_offload

    cfg = config()
    cfg.engram_config.cpu_offload = offload
    verify_and_update_config(cfg)
    assert engram_cpu_offload(cfg) is offload


def test_legacy_engram_conflict_is_rejected(monkeypatch):
    monkeypatch.setenv("TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD", "0")
    with pytest.raises(ValueError, match="conflicts with --engram-config"):
        verify_and_update_config(config())
