# SPDX-License-Identifier: Apache-2.0
"""q38 HOSTTIER-D D1: the decode host-DRAM prefix tier on Qwen4Exp (CPU).

A decode (PD consumer) engine composes the model's PD connector with the Raiden
offload connector in one TPUMultiConnector, behind VLLM_TPU_DECODE_HOST_TIER
(default off). These tests pin admission, the consumer-mode group filter (the
QSA raw ring is a CircularBufferSpec, which subclasses AttentionSpec, and must
never enter the tier), and the multi connector's split / join contracts.
"""

from types import SimpleNamespace as NS

import pytest

from tests.runner.test_qwen4_exp_host_tier import (
    GiB,
    _offload_child,
    _pd_child,
    base_config,
    multi_transfer,
)
from vllm_torchtpu import envs
from vllm_torchtpu.compilation import tpu_compiler
from vllm_torchtpu.models.vllm.qwen4_exp import config as qconfig


@pytest.fixture(autouse=True)
def pooled(monkeypatch):
    monkeypatch.setenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", "1")
    monkeypatch.setenv("TPU_KV_RESHARD_TRANSPORT", "raiden")
    monkeypatch.delenv("QWEN4_EXP_PACK_COMPRESSED_K", raising=False)
    monkeypatch.delenv("QWEN4_EXP_PACK_MAIN_KV", raising=False)
    monkeypatch.delenv("QWEN4_EXP_PP_LOGICAL_BLOCKS_FACTOR", raising=False)
    monkeypatch.delenv("VLLM_TPU_DECODE_HOST_TIER", raising=False)


def served_decode(tp=8, dp=1, cpu_bytes=100 * GiB):
    """A decode engine with a host tier: FP8 KV, APC, align, PP1."""
    cfg = base_config(tp=tp, dp=dp)
    cfg.cache_config.cache_dtype = "fp8"
    cfg.cache_config.enable_prefix_caching = True
    cfg.cache_config.mamba_cache_mode = "align"
    cfg.cache_config.mamba_ssm_cache_dtype = "float32"
    cfg.kv_transfer_config = multi_transfer(
        [_pd_child("kv_consumer"), _offload_child(cpu_bytes=cpu_bytes)],
        role="kv_consumer",
    )
    return cfg


def test_flags_default_off_and_compile_ignored():
    for name in ("VLLM_TPU_DECODE_HOST_TIER", "VLLM_TPU_DECODE_HOST_TIER_VERIFY"):
        assert not envs.environment_variables[name]()
        assert name in tpu_compiler._TPU_COMPILE_ENV_IGNORED


def test_decode_tier_refused_with_flag_off():
    with pytest.raises(ValueError, match="VLLM_TPU_DECODE_HOST_TIER=1"):
        qconfig.validate_config(served_decode())


@pytest.mark.parametrize("tp,dp", [(8, 1), (2, 4)])
def test_decode_tier_admitted_with_flag_on(monkeypatch, tp, dp):
    monkeypatch.setenv("VLLM_TPU_DECODE_HOST_TIER", "1")
    cfg = served_decode(tp=tp, dp=dp)
    qconfig.validate_config(cfg)
    assert qconfig.pd_transfer_config(cfg).kv_role == "kv_consumer"
    assert qconfig.host_offload_enabled(cfg)
    assert qconfig.host_offload_transfer_config(cfg).kv_role == "kv_both"


@pytest.mark.parametrize(
    "transfer",
    [
        # top-level role must be the PD child's
        lambda: multi_transfer([_pd_child("kv_consumer"), _offload_child()], role="kv_both"),
        lambda: multi_transfer([_pd_child("kv_consumer"), _offload_child()], role="kv_producer"),
        # exactly the two children
        lambda: multi_transfer([_pd_child("kv_consumer")], role="kv_consumer"),
        lambda: multi_transfer(
            [_pd_child("kv_consumer"), _offload_child(), _offload_child()], role="kv_consumer"
        ),
    ],
)
def test_decode_tier_rejects_other_shapes_with_flag_on(monkeypatch, transfer):
    monkeypatch.setenv("VLLM_TPU_DECODE_HOST_TIER", "1")
    cfg = served_decode()
    cfg.kv_transfer_config = transfer()
    with pytest.raises(ValueError, match="PD producer"):
        qconfig.validate_config(cfg)


def test_decode_tier_keeps_offload_child_rules(monkeypatch):
    monkeypatch.setenv("VLLM_TPU_DECODE_HOST_TIER", "1")
    cfg = served_decode(cpu_bytes=0)
    with pytest.raises(ValueError, match="cpu_bytes_to_use"):
        qconfig.validate_config(cfg)


def test_flag_is_inert_on_the_prefill_producer(monkeypatch):
    from tests.runner.test_qwen4_exp_host_tier import served_prefill

    monkeypatch.setenv("VLLM_TPU_DECODE_HOST_TIER", "1")
    cfg = served_prefill()
    qconfig.validate_config(cfg)
    assert qconfig.pd_transfer_config(cfg).kv_role == "kv_producer"


def test_plain_decode_pd_unchanged_by_flag(monkeypatch):
    monkeypatch.setenv("VLLM_TPU_DECODE_HOST_TIER", "1")
    cfg = served_decode()
    cfg.kv_transfer_config = NS(**_pd_child("kv_consumer"))
    qconfig.validate_config(cfg)
    assert qconfig.pd_transfer_enabled(cfg) and not qconfig.host_offload_enabled(cfg)


@pytest.mark.parametrize("role", ["kv_both"])
def test_decode_tier_admits_only_the_consumer_role(monkeypatch, role):
    """With the flag on, the decode tier is admitted for a kv_consumer PD child only: a PD child whose role is neither
    producer nor consumer (top-level role equal, so the shape check passes) must be refused by the multi-connector rule
    itself, not left to later checks. (Kills the 'any non-producer role admitted' mutant, j-a9065f15.)"""
    monkeypatch.setenv("VLLM_TPU_DECODE_HOST_TIER", "1")
    cfg = served_decode()
    cfg.kv_transfer_config = multi_transfer([_pd_child(role), _offload_child()], role=role)
    with pytest.raises(ValueError, match="VLLM_TPU_DECODE_HOST_TIER=1"):
        qconfig.validate_multi_connector_config(cfg)
    cfg.kv_transfer_config = multi_transfer([_pd_child("kv_consumer"), _offload_child()], role="kv_consumer")
    qconfig.validate_multi_connector_config(cfg)   # the consumer shape itself still passes this rule
