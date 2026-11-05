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
"""Tests for TPU_SKIP_STATIC_QUANT_PROFILE_RUN in profile_run and compile hash."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm_torchtpu import envs
from vllm_torchtpu.compilation.tpu_compiler import _TPU_COMPILE_ENV_IGNORED


def test_skip_static_quant_profile_run_env_default_and_compile_hash(monkeypatch):
    monkeypatch.delenv("TPU_SKIP_STATIC_QUANT_PROFILE_RUN", raising=False)
    assert envs.TPU_SKIP_STATIC_QUANT_PROFILE_RUN is True
    assert "TPU_SKIP_STATIC_QUANT_PROFILE_RUN" in _TPU_COMPILE_ENV_IGNORED

    monkeypatch.setenv("TPU_SKIP_STATIC_QUANT_PROFILE_RUN", "0")
    assert envs.TPU_SKIP_STATIC_QUANT_PROFILE_RUN is False


def test_quantized_backbone_probe_guard(monkeypatch):
    """Verify guard conditions for skipping quantized kv=0 _dummy_run."""

    unset = object()

    def _need_probe(
        *,
        skip_env: bool = True,
        gpu_mem_util: float = 0.90,
        enforce_eager: bool = False,
        quant_config: object = unset,
        lora_config=None,
        pp_size: int = 1,
    ) -> bool:
        if quant_config is unset:
            quant_config = MagicMock()
        monkeypatch.setenv(
            "TPU_SKIP_STATIC_QUANT_PROFILE_RUN", "1" if skip_env else "0"
        )
        skip_static_quant_probe = (
            quant_config is not None
            and envs.TPU_SKIP_STATIC_QUANT_PROFILE_RUN
            and gpu_mem_util <= 0.92
        )
        return (
            not skip_static_quant_probe
            or enforce_eager
            or lora_config is not None
            or pp_size > 1
        )

    assert _need_probe() is False
    assert _need_probe(skip_env=False) is True
    assert _need_probe(gpu_mem_util=0.95) is True
    assert _need_probe(enforce_eager=True) is True
    assert _need_probe(lora_config=SimpleNamespace()) is True
    assert _need_probe(pp_size=2) is True
    assert _need_probe(quant_config=None) is True
