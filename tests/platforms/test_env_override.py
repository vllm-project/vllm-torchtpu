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

from vllm_torchtpu.env_override import _resolve_sc_offload_threshold_bytes


def test_resolve_sc_offload_threshold_bytes(monkeypatch):
    # 1. Explicit integer override
    monkeypatch.setenv("SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES", "33554432")
    assert _resolve_sc_offload_threshold_bytes() == 33554432

    # 2. Disabled explicitly ("0")
    monkeypatch.setenv("SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES", "0")
    assert _resolve_sc_offload_threshold_bytes() is None

    # Reset override to auto
    monkeypatch.setenv("SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES", "auto")

    # 3. TPU_ACCELERATOR_TYPE lookup
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "v7x-8")
    monkeypatch.delenv("TPU_TYPE", raising=False)
    monkeypatch.delenv("ACCELERATOR_TYPE", raising=False)
    assert _resolve_sc_offload_threshold_bytes() == 67108864

    # 4. TPU_TYPE lookup fallback
    monkeypatch.delenv("TPU_ACCELERATOR_TYPE", raising=False)
    monkeypatch.setenv("TPU_TYPE", "v6e-4")
    assert _resolve_sc_offload_threshold_bytes() == 134217728

    # 5. ACCELERATOR_TYPE lookup fallback
    monkeypatch.delenv("TPU_TYPE", raising=False)
    monkeypatch.setenv("ACCELERATOR_TYPE", "v5p-8")
    assert _resolve_sc_offload_threshold_bytes() == 67108864

    # 6. Fallback default (64 MiB) when no env var is set and metadata fails
    monkeypatch.delenv("ACCELERATOR_TYPE", raising=False)
    monkeypatch.setattr(
        "vllm_torchtpu.env_override._fetch_tpu_type_from_metadata",
        lambda: None)
    assert _resolve_sc_offload_threshold_bytes() == 67108864


def test_user_has_flag():
    import vllm_torchtpu.env_override as env_override

    monkeypatch_args = "--xla_tpu_use_dynamic_smem_negotiation=true --xla_tpu_enable_diagnostic_coordinator=false"
    env_override._existing_libtpu_args = monkeypatch_args

    assert env_override._user_has_flag(
        "--xla_tpu_use_dynamic_smem_negotiation")
    assert env_override._user_has_flag(
        "--xla_tpu_enable_diagnostic_coordinator")
    assert not env_override._user_has_flag("--xla_tpu_non_existent_flag")
