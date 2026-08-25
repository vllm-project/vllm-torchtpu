# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from vllm_torchtpu.kernels.mla.v2 import tuned_params
from vllm_torchtpu.kernels.mla.v2.tuned_params import (TuningKey,
                                                       get_tuned_params)


def _k3_mixed_key() -> TuningKey:
    return TuningKey(
        case="mixed",
        max_num_tokens=4096,
        actual_num_q_heads=12,
        actual_lkv_dim=512,
        actual_r_dim=64,
        q_dtype="bfloat16",
        kv_dtype="bfloat16",
        page_size_per_kv_packing=128,
        kv_packing=2,
        max_num_seqs=8,
        pages_per_seq=128,
    )


def _k3_tp32_fp8_8k_mixed_key() -> TuningKey:
    return TuningKey(
        case="mixed",
        max_num_tokens=8192,
        actual_num_q_heads=3,
        actual_lkv_dim=512,
        actual_r_dim=64,
        q_dtype="bfloat16",
        kv_dtype="float8_e4m3fn",
        page_size_per_kv_packing=4,
        kv_packing=4,
        max_num_seqs=8,
        pages_per_seq=520,
    )


def _k3_tp32_fp8_8k_page256_mixed_key() -> TuningKey:
    return replace(
        _k3_tp32_fp8_8k_mixed_key(),
        page_size_per_kv_packing=64,
        pages_per_seq=40,
    )


def test_k3_mixed_tuned_params() -> None:
    tuned = get_tuned_params(_k3_mixed_key())

    assert tuned.num_kv_pages_per_block == 4
    assert tuned.num_queries_per_block == 256
    assert tuned.q_split == 1


def test_k3_tp32_fp8_8k_mixed_tuned_params() -> None:
    tuned = get_tuned_params(_k3_tp32_fp8_8k_mixed_key())

    assert tuned.num_kv_pages_per_block == 256
    assert tuned.num_queries_per_block == 512
    assert tuned.q_split == 4
    assert tuned.vmem_limit_bytes == 56 * 1024 * 1024


def test_k3_tp32_fp8_8k_page256_mixed_tuned_params() -> None:
    tuned = get_tuned_params(_k3_tp32_fp8_8k_page256_mixed_key())

    assert tuned.num_kv_pages_per_block == 16
    assert tuned.num_queries_per_block == 512
    assert tuned.q_split == 8
    assert tuned.vmem_limit_bytes == 52 * 1024 * 1024


def test_k3_mixed_tuned_params_require_matching_geometry(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tuned_params, "logger", MagicMock())
    mismatched_key = replace(_k3_mixed_key(), page_size_per_kv_packing=64)

    tuned = get_tuned_params(mismatched_key)

    assert tuned.num_kv_pages_per_block == 1
    assert tuned.num_queries_per_block == 16
    assert tuned.q_split == 1
