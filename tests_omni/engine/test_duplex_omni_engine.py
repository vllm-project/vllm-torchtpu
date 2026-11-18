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
"""TPU tests for vLLM-Omni engine session modes (turn vs duplex)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml
from torch_tpu._internal.utils import hardware
from transformers import Qwen3OmniMoeConfig
from vllm_omni.config.stage_config import load_deploy_config
from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.delivery import DuplexOutputBuffer
from vllm_omni.engine.duplex_omni_engine import DuplexOmniEngine
from vllm_omni.entrypoints.duplex.audio_encoding import encode_audio
from vllm_omni.entrypoints.openai.api_server import _should_serve_duplex
from vllm_omni.model_executor.models.qwen3_omni.duplex.plugin import (
    Qwen3OmniDuplexPlugin,
)
from vllm_omni.model_executor.models.qwen3_omni.pipeline import QWEN3_OMNI_PIPELINE

from tests_omni.omni_test_utils import NIGHTLY_TIMEOUT_S

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.nightly,
    pytest.mark.multichip,
    pytest.mark.spawns_tpu_workers,
    pytest.mark.timeout(NIGHTLY_TIMEOUT_S),
]

EXPECTED_MAX_SESSIONS = 2
OUTBOX_MAX_BYTES = 1024 * 1024
OUTBOX_MAX_EVENTS = 128


def _write_tiny_qwen3_omni_dir(model_dir: Path) -> Path:
    cfg = Qwen3OmniMoeConfig()
    cfg.architectures = ["Qwen3OmniMoeForConditionalGeneration"]
    cfg.thinker_config.text_config.num_hidden_layers = 2
    cfg.talker_config.text_config.num_hidden_layers = 2
    cfg.save_pretrained(model_dir)
    return model_dir


def _write_session_deploys(tmp_path: Path) -> tuple[Path, Path]:
    common_stage = {
        "load_format": "dummy",
        "skip_tokenizer_init": True,
        "enforce_eager": True,
        "max_model_len": 128,
        "max_num_seqs": 2,
        "gpu_memory_utilization": 0.2,
    }
    base_doc = {
        "pipeline": "qwen3_omni_moe",
        "async_chunk": True,
        "connectors": {
            "shm_conn": {
                "name": "SharedMemoryConnector",
            }
        },
        "stages": [
            {
                **common_stage,
                "stage_id": 0,
                "devices": "0",
                "output_connectors": {"to_stage_1": "shm_conn"},
            },
            {
                **common_stage,
                "stage_id": 1,
                "devices": "1",
                "input_connectors": {"from_stage_0": "shm_conn"},
                "output_connectors": {"to_stage_2": "shm_conn"},
            },
            {
                **common_stage,
                "stage_id": 2,
                "devices": "2",
                "input_connectors": {"from_stage_1": "shm_conn"},
            },
        ],
    }
    turn_path = tmp_path / "qwen3_turn.yaml"
    turn_path.write_text(
        yaml.safe_dump({**base_doc, "session_mode": "turn"}),
        encoding="utf-8",
    )
    duplex_path = tmp_path / "qwen3_duplex.yaml"
    duplex_path.write_text(
        yaml.safe_dump(
            {
                **base_doc,
                "session_mode": "duplex",
                "duplex_session": {
                    "max_sessions": EXPECTED_MAX_SESSIONS,
                    "warmup_frames": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    return turn_path, duplex_path


async def _open_and_close_duplex_session(
    engine: DuplexOmniEngine,
) -> tuple[bool, bool]:
    outbox = DuplexOutputBuffer(
        max_bytes=OUTBOX_MAX_BYTES,
        max_events=OUTBOX_MAX_EVENTS,
    )
    open_res = await engine.open_session_async(
        "tpu-duplex-sid",
        DuplexSessionConfig(),
        output_buffer=outbox,
        timeout=10.0,
    )
    close_res = await engine.close_session_async("tpu-duplex-sid", timeout=10.0)
    return open_res.ok, close_res.ok


@pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason=(
        "DuplexOmniEngine requires AR/GEN stages, and TPUWorker lacks an "
        "OmniConnectorModelRunnerMixin runner (tracked in vllm-torchtpu PR #2065)"
    ),
)
def test_engine_session_modes(tmp_path: Path) -> None:
    """Engine Session Modes (Turn vs Duplex).

    - TPU path: evaluate ``_should_serve_duplex`` on turn vs duplex configs and
      initialize ``DuplexOmniEngine`` on TPU across devices 0, 1, and 2.
    - Setup check: assert ``_should_serve_duplex`` returns ``False``/``True`` and
      ``_validate_deployment`` loads ``Qwen3OmniDuplexPlugin`` with
      ``max_sessions == 2``.
    - Outcome check: assert ``duplex_capabilities.max_sessions == 2`` and open/close
      a live duplex session via ``open_session_async`` / ``close_session_async``.
    """
    device_count = hardware.get_tpu_device_count()
    assert device_count >= 3, (
        f"Duplex 3-stage pipeline requires 3 TPU devices, found {device_count}"
    )

    model_dir = _write_tiny_qwen3_omni_dir(tmp_path / "qwen3_duplex_model")
    turn_yaml, duplex_yaml = _write_session_deploys(tmp_path)

    serves_turn = _should_serve_duplex(
        str(model_dir),
        {"deploy_config": str(turn_yaml)},
    )
    serves_duplex = _should_serve_duplex(
        str(model_dir),
        {"deploy_config": str(duplex_yaml)},
    )
    assert serves_turn is False, (
        f"Expected _should_serve_duplex=False for turn mode, got {serves_turn}"
    )
    assert serves_duplex is True, (
        f"Expected _should_serve_duplex=True for duplex mode, got {serves_duplex}"
    )

    probe = object.__new__(DuplexOmniEngine)
    probe.model = str(model_dir)
    probe.pipeline_config = QWEN3_OMNI_PIPELINE
    probe.deploy_config = load_deploy_config(duplex_yaml)
    probe._audio_encoder = encode_audio
    probe.single_stage_mode = False
    probe._validate_deployment()
    assert isinstance(probe.plugin, Qwen3OmniDuplexPlugin), (
        f"Expected Qwen3OmniDuplexPlugin, got {type(probe.plugin).__name__}"
    )
    assert probe.duplex_session_config.max_sessions == EXPECTED_MAX_SESSIONS, (
        f"Expected duplex max_sessions={EXPECTED_MAX_SESSIONS}, "
        f"got {probe.duplex_session_config.max_sessions}"
    )

    engine = DuplexOmniEngine(
        str(model_dir),
        duplex_audio_encoder=encode_audio,
        deploy_config=str(duplex_yaml),
        stage_init_timeout=120,
        startup_timeout=240,
    )
    try:
        caps = engine.duplex_capabilities
        assert caps.max_sessions == EXPECTED_MAX_SESSIONS, (
            f"duplex_capabilities.max_sessions {caps.max_sessions} "
            f"!= {EXPECTED_MAX_SESSIONS}"
        )
        opened_ok, closed_ok = asyncio.run(_open_and_close_duplex_session(engine))
        assert opened_ok is True, (
            f"Expected open_session_async ok=True, got {opened_ok}"
        )
        assert closed_ok is True, (
            f"Expected close_session_async ok=True, got {closed_ok}"
        )
    finally:
        engine.shutdown()
