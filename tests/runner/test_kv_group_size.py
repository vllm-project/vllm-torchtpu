# SPDX-License-Identifier: Apache-2.0
"""KV cache group size rule (vllm_torchtpu.kv_group_size)."""

import importlib.util
import pathlib

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[2] / "src/vllm_torchtpu/kv_group_size.py"
_spec = importlib.util.spec_from_file_location("kv_group_size", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
size = _mod.kv_cache_group_size


def test_qwen38_pipeline_prefill_groups_by_twelve():
    # 36 GDN, 1 PLE, 12 main KV, 12 compressed keys, 12 raw rings.
    assert size([36, 1, 12, 12, 12]) == 12
    assert size([36, 1, 12, 12, 12], "min") == 1


def test_qwen38_mtp_packed_decode_stays_one_group_per_layer():
    # 36 GDN, 1 PLE, 12 main KV, 2 packed compressed owners, 13 rings, 1 draft main KV.
    assert size([36, 1, 12, 2, 13, 1]) == 1


def test_qwen35_and_uniform_cases_match_upstream():
    assert size([45, 15]) == 15 == size([45, 15], "min")
    assert size([12, 13], "min") == 13
    assert size([7]) == 7
    assert size([1, 1]) == 1


def test_rule_validation():
    with pytest.raises(ValueError):
        size([], "gcd")
    with pytest.raises(ValueError):
        size([4], "max")
