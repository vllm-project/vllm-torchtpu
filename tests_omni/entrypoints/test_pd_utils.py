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
"""TPU tests for Prefill-Decode (PD) disaggregation helpers and AR stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from transformers import Qwen3OmniMoeConfig
from vllm import SamplingParams
from vllm_omni.engine.stage_init_utils import build_vllm_config
from vllm_omni.entrypoints.pd_utils import PDDisaggregationMixin
from vllm_omni.platforms import current_omni_platform
from vllm_omni.worker.omni_connector_validation import validate_worker_omni_connector

from tests_omni.omni_test_utils import PR_TIMEOUT_S

pytestmark = [
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

EXPECTED_PD_PAIR = (0, 1)
EXPECTED_PREFILL_MAX_TOKENS = 1


@dataclass
class _PDStageStub:
    stage_id: int
    stage_type: str
    is_prefill_only: bool
    is_decode_only: bool
    engine_input_source: list[int]
    engine_args: dict[str, Any]
    default_sampling_params: dict[str, Any]


class _PDHost(PDDisaggregationMixin):
    stage_configs: list[_PDStageStub]
    _name: str


def _write_tiny_qwen3_omni_config(model_dir: Path) -> Path:
    cfg = Qwen3OmniMoeConfig()
    cfg.architectures = ["Qwen3OmniMoeForConditionalGeneration"]
    cfg.thinker_config.text_config.num_hidden_layers = 2
    cfg.talker_config.text_config.num_hidden_layers = 2
    cfg.save_pretrained(model_dir)
    return model_dir


def _build_pd_thinker_stages() -> list[_PDStageStub]:
    base_args = {
        "model_stage": "thinker",
        "model_arch": "Qwen3OmniMoeForConditionalGeneration",
        "hf_config_name": "thinker_config",
        "worker_type": "ar",
        "skip_tokenizer_init": True,
        "enforce_eager": True,
        "max_model_len": 128,
        "max_num_seqs": 2,
    }
    prefill_stage = _PDStageStub(
        stage_id=0,
        stage_type="llm",
        is_prefill_only=True,
        is_decode_only=False,
        engine_input_source=[],
        engine_args={
            **base_args,
            "kv_transfer_config": {
                "kv_connector": "TPUConnector",
                "kv_role": "kv_producer",
                "kv_ip": "127.0.0.1",
                "kv_port": 25201,
            },
        },
        default_sampling_params={"max_tokens": 1},
    )
    decode_stage = _PDStageStub(
        stage_id=1,
        stage_type="llm",
        is_prefill_only=False,
        is_decode_only=True,
        engine_input_source=[0],
        engine_args={
            **base_args,
            "custom_process_next_stage_input_func": (
                "vllm_omni.model_executor.stage_input_processors."
                "qwen3_omni.thinker2talker_full_payload"
            ),
            "kv_transfer_config": {
                "kv_connector": "TPUConnector",
                "kv_role": "kv_consumer",
                "kv_ip": "127.0.0.1",
                "kv_port": 25202,
            },
        },
        default_sampling_params={"max_tokens": 16},
    )
    return [prefill_stage, decode_stage]


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason=(
        "TPUWorker does not provide an Omni connector model runner "
        "(tracked in vllm-torchtpu PR #2065)"
    ),
)
def test_prefill_decode_disaggregation(tmp_path: Path) -> None:
    """Prefill-Decode (PD) Disaggregation.

    - TPU path: configure a 2-stage Qwen3-Omni Thinker PD split with
      ``TPUConnector`` (``kv_producer`` -> ``kv_consumer``) and build
      ``VllmConfig`` for both AR stages under ``TpuPlatform`` and
      ``OmniTpuPlatform``.
    - Setup check: assert ``_get_pd_separation_pair() == (0, 1)`` and
      ``_prepare_prefill_sampling_params`` clamps ``max_tokens == 1`` with
      ``do_remote_decode=True``.
    - Outcome check: assert ``build_vllm_config`` preserves ``kv_role`` on both
      stages and ``validate_worker_omni_connector`` accepts ``worker_cls``.
    """
    assert current_omni_platform.device_name == "tpu", (
        f"Expected omni platform 'tpu', got {current_omni_platform.device_name!r}"
    )

    stages = _build_pd_thinker_stages()
    host = object.__new__(_PDHost)
    host.stage_configs = stages
    host._name = "test-pd-host"
    host._init_pd_state()

    pair = host._get_pd_separation_pair()
    assert pair == EXPECTED_PD_PAIR, (
        f"PD separation pair {pair} != expected {EXPECTED_PD_PAIR}"
    )

    prefill_sp = host._prepare_prefill_sampling_params(
        "req-pd-0",
        SamplingParams(max_tokens=16, stop=["END"]),
    )
    assert prefill_sp.max_tokens == EXPECTED_PREFILL_MAX_TOKENS, (
        f"Prefill max_tokens {prefill_sp.max_tokens} != {EXPECTED_PREFILL_MAX_TOKENS}"
    )
    assert prefill_sp.stop == [], (
        f"Expected empty prefill stop list, got {prefill_sp.stop}"
    )
    kv_params = (prefill_sp.extra_args or {}).get("kv_transfer_params", {})
    assert kv_params.get("do_remote_decode") is True, (
        f"Expected do_remote_decode=True in kv_transfer_params, got {kv_params!r}"
    )

    model_dir = _write_tiny_qwen3_omni_config(tmp_path / "qwen3_omni_pd")
    prefill_cfg, _ = build_vllm_config(
        stages[0],
        str(model_dir),
        stage_connector_spec={
            "name": "SharedMemoryConnector",
            "extra": {"role": "sender"},
        },
    )
    decode_cfg, _ = build_vllm_config(
        stages[1],
        str(model_dir),
        stage_connector_spec={
            "name": "SharedMemoryConnector",
            "extra": {"role": "receiver"},
        },
    )

    assert prefill_cfg.kv_transfer_config is not None, (
        "Prefill VllmConfig lost kv_transfer_config"
    )
    assert prefill_cfg.kv_transfer_config.kv_role == "kv_producer", (
        f"Prefill kv_role {prefill_cfg.kv_transfer_config.kv_role!r} != 'kv_producer'"
    )
    assert decode_cfg.kv_transfer_config is not None, (
        "Decode VllmConfig lost kv_transfer_config"
    )
    assert decode_cfg.kv_transfer_config.kv_role == "kv_consumer", (
        f"Decode kv_role {decode_cfg.kv_transfer_config.kv_role!r} != 'kv_consumer'"
    )
    validate_worker_omni_connector(
        prefill_cfg.parallel_config.worker_cls, required=True
    )
    validate_worker_omni_connector(decode_cfg.parallel_config.worker_cls, required=True)
