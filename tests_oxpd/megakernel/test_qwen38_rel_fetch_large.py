# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_MK_REL_FETCH_LARGE enables rel_fetch on whole-step buckets B >= 48."""

from vllm_torchtpu import envs
from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wc


def test_rel_fetch_large_flag_off_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_TPU_MK_REL_FETCH_LARGE", raising=False)
    monkeypatch.delenv("VLLM_TPU_MK_GDN_INK", raising=False)
    monkeypatch.delenv("VLLM_TPU_MK_RINGS", raising=False)
    assert not envs.VLLM_TPU_MK_REL_FETCH_LARGE
    assert not wc.wholestep_tiles(48).rel_fetch
    assert not wc.wholestep_tiles(64).rel_fetch


def test_rel_fetch_large_applies_to_b48_and_above(monkeypatch):
    monkeypatch.setenv("VLLM_TPU_MK_REL_FETCH_LARGE", "1")
    monkeypatch.delenv("VLLM_TPU_MK_GDN_INK", raising=False)
    monkeypatch.delenv("VLLM_TPU_MK_RINGS", raising=False)
    assert not wc.wholestep_tiles(16).rel_fetch
    assert not wc.wholestep_tiles(32).rel_fetch
    for bucket in (48, 64, 96, 128):
        assert wc.wholestep_tiles(bucket).rel_fetch is True


def test_rel_fetch_preserves_gdn_ink_bit2(monkeypatch):
    monkeypatch.delenv("VLLM_TPU_MK_REL_FETCH_LARGE", raising=False)
    monkeypatch.setenv("VLLM_TPU_MK_GDN_INK", "2")
    assert wc.wholestep_tiles(16).rel_fetch is True
    assert wc.wholestep_tiles(48).rel_fetch is True
