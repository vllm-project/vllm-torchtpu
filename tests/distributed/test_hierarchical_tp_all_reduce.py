# SPDX-License-Identifier: Apache-2.0
"""Tests for native SparseCore hierarchical TP all-reduce activation."""

import importlib
import os
from unittest.mock import patch

_NATIVE_FLAG = "--xla_tpu_enable_sparse_core_hierarchical_all_reduce"


def test_env_threshold_is_zero_by_default_and_accepts_tokens(monkeypatch):
    from vllm_torchtpu import envs

    name = "TPU_TP_HIERARCHICAL_ALL_REDUCE_MIN_TOKENS"
    getter = envs.environment_variables[name]
    monkeypatch.delenv(name, raising=False)
    assert getter() == 0
    monkeypatch.setenv(name, "1024")
    assert getter() == 1024


def test_positive_threshold_enables_native_libtpu_hierarchy():
    import vllm_torchtpu.env_override as env_override

    values = {
        "TPU_ACCELERATOR_TYPE": "tpu7x",
        "TPU_TP_HIERARCHICAL_ALL_REDUCE_MIN_TOKENS": "1024",
    }
    with patch.dict(os.environ, values, clear=True):
        importlib.reload(env_override)
        args = os.environ["LIBTPU_INIT_ARGS"].split()

    assert f"{_NATIVE_FLAG}=true" in args


def test_zero_threshold_does_not_enable_native_libtpu_hierarchy():
    import vllm_torchtpu.env_override as env_override

    values = {
        "TPU_ACCELERATOR_TYPE": "tpu7x",
        "TPU_TP_HIERARCHICAL_ALL_REDUCE_MIN_TOKENS": "0",
    }
    with patch.dict(os.environ, values, clear=True):
        importlib.reload(env_override)
        args = os.environ["LIBTPU_INIT_ARGS"].split()

    assert not any(arg.startswith(_NATIVE_FLAG) for arg in args)


def test_explicit_native_libtpu_setting_is_not_overwritten():
    import vllm_torchtpu.env_override as env_override

    explicit = f"{_NATIVE_FLAG}=false"
    values = {
        "LIBTPU_INIT_ARGS": explicit,
        "TPU_ACCELERATOR_TYPE": "tpu7x",
        "TPU_TP_HIERARCHICAL_ALL_REDUCE_MIN_TOKENS": "1024",
    }
    with patch.dict(os.environ, values, clear=True):
        importlib.reload(env_override)
        args = os.environ["LIBTPU_INIT_ARGS"].split()

    assert args.count(explicit) == 1
    assert f"{_NATIVE_FLAG}=true" not in args
