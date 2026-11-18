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
"""TPU engine test for the DuplexOmni programmatic full-duplex API."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from transformers import AutoTokenizer, Qwen3OmniMoeConfig
from vllm_omni.engine.duplex.commands import AppendText, Commit, CreateResponse
from vllm_omni.engine.duplex.events import (
    DuplexEvent,
    ResponseDone,
    SessionCreated,
    TextDelta,
)
from vllm_omni.engine.duplex_omni_engine import DuplexOmniEngine
from vllm_omni.entrypoints.duplex_omni import DuplexOmni

from tests.tpu_release import wait_for_tpu_release
from tests_omni.omni_test_utils import NIGHTLY_TIMEOUT_S
from tests_omni.tiny_models import tiny_qwen_image

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.nightly,
    pytest.mark.multichip,
    pytest.mark.spawns_tpu_workers,
    pytest.mark.timeout(NIGHTLY_TIMEOUT_S),
]

STAGE_INIT_TIMEOUT_S = 120
SESSION_EVENT_TIMEOUT_S = 60.0


def _build_tiny_qwen3_omni_dir(target_dir: Path) -> str:
    """Create a local 2-layer Qwen3-Omni-MoE directory using tiny Qwen tokenizer."""
    cfg = Qwen3OmniMoeConfig(
        architectures=["Qwen3OmniMoeForConditionalGeneration"],
    )
    cfg.thinker_config.text_config.num_hidden_layers = 2
    cfg.talker_config.text_config.num_hidden_layers = 2
    cfg.code2wav_config.num_hidden_layers = 2
    cfg.save_pretrained(target_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        tiny_qwen_image(),
        subfolder="tokenizer",
    )
    tokenizer.save_pretrained(target_dir)
    return str(target_dir)


@pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason=(
        "DuplexOmni requires AR/GEN stages, and TPUWorker lacks "
        "OmniConnectorModelRunnerMixin on main (vllm-torchtpu PR #2065)"
    ),
)
def test_duplex_omni_session(tmp_path: Path) -> None:
    """Programmatic Full-Duplex API (DuplexOmni).

    - TPU path: constructs ``DuplexOmni`` with
      ``deploy_config="qwen3_omni_duplex.yaml"`` on a tiny local
      ``Qwen3OmniMoeConfig`` directory, opens a duplex session, and submits
      ``AppendText``, ``Commit``, and ``CreateResponse``.
    - Setup check: ``omni.engine`` is a ``DuplexOmniEngine`` with
      ``deploy_config.session_mode == "duplex"``.
    - Outcome check: ``session.events()`` emits ``SessionCreated`` followed by
      non-empty ``TextDelta`` output and a terminal ``ResponseDone`` event.
    """
    model_dir = _build_tiny_qwen3_omni_dir(tmp_path / "tiny_qwen3_omni")
    stage_overrides = {
        str(stage_id): {
            "load_format": "dummy",
            "max_model_len": 512,
            "enforce_eager": True,
        }
        for stage_id in (0, 1, 2)
    }

    async def _run_duplex() -> tuple[str, list[DuplexEvent]]:
        wait_for_tpu_release()
        omni = DuplexOmni(
            model=model_dir,
            deploy_config="qwen3_omni_duplex.yaml",
            stage_overrides=stage_overrides,
            stage_init_timeout=STAGE_INIT_TIMEOUT_S,
            init_timeout=STAGE_INIT_TIMEOUT_S,
        )
        try:
            assert isinstance(omni.engine, DuplexOmniEngine), (
                f"DuplexOmni engine {type(omni.engine).__name__} != DuplexOmniEngine"
            )
            assert omni.engine.deploy_config is not None, (
                "DuplexOmni deploy_config is None"
            )
            assert omni.engine.deploy_config.session_mode == "duplex", (
                "DuplexOmni session_mode "
                f"{omni.engine.deploy_config.session_mode!r} != 'duplex'"
            )

            collected: list[DuplexEvent] = []
            async with await omni.open_session({"modalities": ["text"]}) as session:
                session_id = session.session_id
                await session.submit(AppendText(text="Hello"))
                await session.submit(Commit())
                await session.submit(CreateResponse(options={"max_output_tokens": 4}))

                async def _drain() -> None:
                    async for event in session.events():
                        collected.append(event)
                        if isinstance(event, ResponseDone):
                            break

                await asyncio.wait_for(_drain(), timeout=SESSION_EVENT_TIMEOUT_S)
            return session_id, collected
        finally:
            omni.shutdown()
            wait_for_tpu_release()

    session_id, events = asyncio.run(_run_duplex())
    assert session_id.startswith("duplex-"), (
        f"Duplex session_id {session_id!r} missing 'duplex-' prefix"
    )
    assert any(isinstance(ev, SessionCreated) for ev in events), (
        f"Duplex session events {[type(ev).__name__ for ev in events]} "
        "missing SessionCreated"
    )
    assert any(isinstance(ev, ResponseDone) for ev in events), (
        f"Duplex session events {[type(ev).__name__ for ev in events]} "
        "missing ResponseDone"
    )
    text_chunks = [ev.delta for ev in events if isinstance(ev, TextDelta)]
    assert "".join(text_chunks) != "", (
        f"Duplex session emitted empty TextDelta output across {len(events)} events"
    )
