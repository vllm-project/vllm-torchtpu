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
"""Tests for startup TPU environment preflight and compile factors hashing."""

import ast
import os
import pathlib
from unittest.mock import patch

import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.platforms import preflight as preflight_module
from vllm_torchtpu.platforms.preflight import preflight_check


def test_type_checking_parity():
    """Verify TYPE_CHECKING declarations match registered envs 1:1."""
    source = pathlib.Path(envs.__file__).read_text()
    tree = ast.parse(source)
    tc_names = set()
    for node in tree.body:
        if (
            type(node) is ast.If
            and type(node.test) is ast.Name
            and node.test.id == "TYPE_CHECKING"
        ):
            for stmt in node.body:
                if type(stmt) is ast.AnnAssign and type(stmt.target) is ast.Name:
                    tc_names.add(stmt.target.id)
    assert tc_names == set(envs.environment_variables)
    assert envs.env_with_choices.__doc__ is not None
    assert "TPU_MOE_ROUTER_TOPK" in envs.ENV_CHOICES


def test_compile_factors_hashing_and_ignored_keys():
    """Verify compile factors partition and digest sensitivity."""
    all_keys = set(envs.environment_variables)
    factors_keys = set(envs.compile_factors())
    ignored_keys = set(envs.COMPILE_IGNORED_FACTORS)
    assert factors_keys | ignored_keys == all_keys

    base_digest = envs.compile_factors_digest()
    with envs.patch(ONEHOT_MOE_PERMUTE_THRESHOLD=1024):
        assert envs.compile_factors_digest() != base_digest
    new_version = "v2" if envs.RAGGED_GATHER_REDUCE_VERSION != "v2" else "v3"
    with envs.patch(RAGGED_GATHER_REDUCE_VERSION=new_version):
        assert envs.compile_factors_digest() != base_digest
    with envs.patch(USE_MOE_FUSED_EP_KERNEL=not envs.USE_MOE_FUSED_EP_KERNEL):
        assert envs.compile_factors_digest() != base_digest

    with envs.patch(TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="/tmp/cache_root_test"):
        assert envs.compile_factors_digest() == base_digest
    with envs.patch(USE_PHASED_PROFILER=not envs.USE_PHASED_PROFILER):
        assert envs.compile_factors_digest() == base_digest
    with envs.patch(VLLM_TPU_STRICT_ENV=True):
        assert envs.compile_factors_digest() == base_digest
    with envs.patch(VLLM_TORCHTPU_IPC_KEY="test_ipc_key"):
        assert envs.compile_factors_digest() == base_digest


def test_envs_patch_and_cache_isolation():
    """Verify envs.patch overrides and clean cache restoration."""
    envs.enable_envs_cache()
    try:
        orig_sp = envs.TPU_K3_SP_PREFILL
        orig_rg = envs.RAGGED_GATHER_REDUCE_VERSION
        orig_env_sp = os.environ.get("TPU_K3_SP_PREFILL")
        orig_env_rg = os.environ.get("RAGGED_GATHER_REDUCE_VERSION")

        with envs.patch(TPU_K3_SP_PREFILL=True, RAGGED_GATHER_REDUCE_VERSION="v3"):
            assert envs.TPU_K3_SP_PREFILL is True
            assert envs.RAGGED_GATHER_REDUCE_VERSION == "v3"
            factors = envs.compile_factors()
            assert factors["TPU_K3_SP_PREFILL"] is True
            assert factors["RAGGED_GATHER_REDUCE_VERSION"] == "v3"

        assert orig_sp == envs.TPU_K3_SP_PREFILL
        assert orig_rg == envs.RAGGED_GATHER_REDUCE_VERSION
        assert os.environ.get("TPU_K3_SP_PREFILL") == orig_env_sp
        assert os.environ.get("RAGGED_GATHER_REDUCE_VERSION") == orig_env_rg

        with pytest.raises(KeyError), envs.patch(NONEXISTENT_TPU_KNOB=1):
            pass
    finally:
        envs.disable_envs_cache()


def test_preflight_aggregates_fatal_errors(monkeypatch):
    """Verify aggregated error reporting across multiple broken env knobs."""
    monkeypatch.setenv("USE_MOE_EP_KERNEL", "1")
    monkeypatch.setenv("TPU_MOE_ROUTER_TOPK", "bogus_mode")
    monkeypatch.setenv("TPU_MOE_DENSE_EXPERT_THRESHOLD", "-5")
    monkeypatch.setenv("ONEHOT_MOE_PERMUTE_THRESHOLD", "not_an_int")

    with pytest.raises(ValueError) as excinfo:
        preflight_check()

    err_msg = str(excinfo.value)
    assert "USE_MOE_FUSED_EP_KERNEL=1" in err_msg
    assert "TPU_MOE_ROUTER_TOPK" in err_msg
    assert "TPU_MOE_DENSE_EXPERT_THRESHOLD" in err_msg
    assert "ONEHOT_MOE_PERMUTE_THRESHOLD" in err_msg


def test_preflight_unknown_tpu_flag_warns_by_default_and_fails_in_strict_mode(
    monkeypatch,
):
    """Verify unknown flags warn by default and error only in strict mode."""
    monkeypatch.setenv("TPU_FUTURE_CANARY_KNOB", "1")
    monkeypatch.setenv("TPU_MOE_FUSED_EP_KERNEL", "1")
    monkeypatch.setenv("VLLM_TPU_STRICT_ENV", "0")

    with pytest.warns(UserWarning):
        report = preflight_check()
    assert report is not None
    warn_text = " ".join(report.warnings)
    assert "TPU_FUTURE_CANARY_KNOB" in warn_text
    assert "USE_MOE_FUSED_EP_KERNEL" in warn_text

    monkeypatch.setenv("VLLM_TPU_STRICT_ENV", "1")
    with pytest.raises(ValueError, match="USE_MOE_FUSED_EP_KERNEL"):
        preflight_check()

    monkeypatch.delenv("TPU_FUTURE_CANARY_KNOB", raising=False)
    monkeypatch.delenv("TPU_MOE_FUSED_EP_KERNEL", raising=False)
    monkeypatch.setenv("TPU_VISIBLE_CHIPS", "0")
    monkeypatch.setenv("TPU_LIBRARY_PATH", "/tmp/libtpu.so")
    monkeypatch.setenv("TPU_PREMAPPED_BUFFER_SIZE", "1024")
    monkeypatch.setenv("TPU_NUM_HOSTS", "2")
    monkeypatch.setenv("TPU_BACKEND_TYPE", "jax")
    monkeypatch.setenv("VLLM_TPU_USING_PATHWAYS", "0")
    clean_report = preflight_check()
    assert len(clean_report.warnings) == 0


def test_preflight_deprecated_alias_forwarding_and_strict_mode(monkeypatch):
    """Verify deprecated aliases forward values and error under strict mode."""
    monkeypatch.setitem(
        envs.DEPRECATED_ENVS, "TPU_LEGACY_SP_PREFILL_ALIAS", "TPU_K3_SP_PREFILL"
    )
    monkeypatch.delenv("TPU_K3_SP_PREFILL", raising=False)
    monkeypatch.setenv("TPU_LEGACY_SP_PREFILL_ALIAS", "1")
    monkeypatch.setenv("VLLM_TPU_STRICT_ENV", "0")

    envs.enable_envs_cache()
    try:
        with pytest.warns(FutureWarning):
            report = preflight_check()
        assert report is not None
        assert os.environ.get("TPU_K3_SP_PREFILL") == "1"
        assert envs.TPU_K3_SP_PREFILL is True
    finally:
        envs.disable_envs_cache()

    monkeypatch.setenv("TPU_K3_SP_PREFILL", "0")
    monkeypatch.setenv("TPU_LEGACY_SP_PREFILL_ALIAS", "1")
    report2 = preflight_check()
    assert report2 is not None
    assert os.environ.get("TPU_K3_SP_PREFILL") == "0"

    monkeypatch.delenv("TPU_K3_SP_PREFILL", raising=False)
    monkeypatch.setenv("TPU_LEGACY_SP_PREFILL_ALIAS", "not_a_bool")
    with pytest.raises(ValueError, match="TPU_K3_SP_PREFILL"):
        preflight_check()

    monkeypatch.setenv("TPU_LEGACY_SP_PREFILL_ALIAS", "1")
    monkeypatch.setenv("VLLM_TPU_STRICT_ENV", "1")
    with pytest.raises(ValueError, match="TPU_LEGACY_SP_PREFILL_ALIAS"):
        preflight_check()


def test_preflight_rank_digest_mismatch_and_secret_masking(monkeypatch):
    """Verify compile factor digest mismatch and secret masking."""
    with pytest.raises(ValueError, match="digest mismatch"):
        preflight_check(expected_compile_digest="0000000000000000")

    monkeypatch.setenv("VLLM_TORCHTPU_COMPILE_FACTORS_DIGEST", "0000000000000000")
    with pytest.raises(ValueError, match="digest mismatch"):
        preflight_check()

    monkeypatch.delenv("VLLM_TORCHTPU_COMPILE_FACTORS_DIGEST", raising=False)
    current_digest = envs.compile_factors_digest()
    monkeypatch.setenv("VLLM_TORCHTPU_COMPILE_FACTORS_DIGEST", current_digest)
    matching_report = preflight_check(expected_compile_digest=current_digest)
    assert matching_report is not None

    monkeypatch.setenv("VLLM_TORCHTPU_IPC_KEY", "raw-secret-token")
    monkeypatch.setenv("RAIDEN_SHM_KEY", "raw-secret-token-shm")
    with patch.object(preflight_module.logger, "info") as mock_info:
        secret_report = preflight_check()
        configured = dict(secret_report.configured_envs)
        assert configured["VLLM_TORCHTPU_IPC_KEY"] == "***"
        assert configured["RAIDEN_SHM_KEY"] == "***"
        mock_info.assert_called_once()
        log_call_args = str(mock_info.call_args)
        assert "***" in log_call_args
        assert "raw-secret-token" not in log_call_args
