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
"""TPU test for the Mage-VL Duplex Server (WS /v1/mage-vl/duplex)."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import textwrap
from pathlib import Path

import av
import numpy as np
import pytest
import safetensors.torch
import torch
from fastapi.testclient import TestClient
from vllm_omni.experimental.fullduplex.core.session import (
    DuplexSession,
    DuplexSessionConfig,
)
from vllm_omni.experimental.fullduplex.mage_vl import MageVLCodecWindow
from vllm_omni.experimental.fullduplex.mage_vl.serving.backend import (
    MageVLTransformersBackend,
)
from vllm_omni.experimental.fullduplex.mage_vl.serving.server import (
    MageVLServingConfig,
    create_app,
)
from vllm_omni.platforms import current_omni_platform

from tests_omni.omni_test_utils import PR_TIMEOUT_S

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

AUTH_TOKEN = "mage-tpu-token"
GATE_SCORE_ATOL = 1e-2

_TINY_MAGE_CODE = textwrap.dedent(
    """\
    import torch
    from torch import nn
    from transformers import PretrainedConfig, PreTrainedModel
    from transformers.processing_utils import ProcessorMixin


    class TinyMageConfig(PretrainedConfig):
        model_type = "tiny_mage_vl"

        def __init__(self, vocab_size: int = 64, hidden_size: int = 32, **kwargs):
            super().__init__(**kwargs)
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size


    class TinyMageForCausalLM(PreTrainedModel):
        config_class = TinyMageConfig
        _supports_sdpa = True

        def __init__(self, config: TinyMageConfig):
            super().__init__(config)
            self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.gate_proj = nn.Linear(16, 2)
            self.post_init()

        def forward(self, input_ids=None, pixel_values=None, **kwargs):
            hidden = self.embed(input_ids)
            return (self.lm_head(hidden),)

        def streammind_gate_forward_segments(self, visual):
            pooled = torch.stack([seg["pixel_values"].mean(dim=0) for seg in visual])
            logits = self.gate_proj(pooled)
            bias = torch.tensor(
                [-2.0, 4.0], device=logits.device, dtype=logits.dtype
            )
            return (logits + bias,)

        @torch.no_grad()
        def generate(
            self, input_ids=None, pixel_values=None, max_new_tokens: int = 4, **kwargs
        ):
            del kwargs, max_new_tokens
            hidden = self.embed(input_ids[:, -1:])
            if pixel_values is not None:
                vis_bias = pixel_values.mean().to(hidden.dtype)
                hidden = hidden + vis_bias
            next_id = self.lm_head(hidden).argmax(dim=-1)
            step2 = ((next_id + 3) % self.config.vocab_size)
            return torch.cat([input_ids, next_id, step2], dim=1)


    class _TinyTokenizer:
        def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
            del skip_special_tokens
            ids = (
                token_ids.tolist()
                if hasattr(token_ids, "tolist")
                else list(token_ids)
            )
            return "tokens:" + ",".join(str(int(i)) for i in ids)


    class TinyMageProcessor(ProcessorMixin):
        attributes = []

        def __init__(self, **kwargs):
            del kwargs
            self.tokenizer = _TinyTokenizer()

        def apply_chat_template(
            self,
            messages,
            tokenize: bool = False,
            add_generation_prompt: bool = True,
        ) -> str:
            del tokenize, add_generation_prompt
            parts = []
            for msg in messages:
                for item in msg.get("content", []):
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(item["text"])
            return " ".join(parts)

        def __call__(self, text=None, videos=None, **kwargs):
            del videos, kwargs
            prompt = (text[0] if text else "mage")[:8]
            ids = [min(ord(ch), 60) for ch in prompt] or [1]
            return {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "pixel_values": torch.full((2, 16), 0.5, dtype=torch.float32),
                "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
            }

        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
            del pretrained_model_name_or_path, kwargs
            return cls()
    """
)


def _write_tiny_mage_checkpoint(ckpt_dir: Path) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "modeling_tiny_mage.py").write_text(_TINY_MAGE_CODE, encoding="utf-8")
    config = {
        "architectures": ["TinyMageForCausalLM"],
        "model_type": "tiny_mage_vl",
        "vocab_size": 64,
        "hidden_size": 32,
        "auto_map": {
            "AutoConfig": "modeling_tiny_mage.TinyMageConfig",
            "AutoModelForCausalLM": "modeling_tiny_mage.TinyMageForCausalLM",
            "AutoProcessor": "modeling_tiny_mage.TinyMageProcessor",
        },
    }
    (ckpt_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (ckpt_dir / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "processor_class": "TinyMageProcessor",
                "auto_map": {"AutoProcessor": "modeling_tiny_mage.TinyMageProcessor"},
            }
        ),
        encoding="utf-8",
    )
    torch.manual_seed(0)
    tensors = {
        "embed.weight": torch.randn(64, 32, dtype=torch.bfloat16),
        "lm_head.weight": torch.randn(64, 32, dtype=torch.bfloat16),
        "gate_proj.weight": torch.zeros(2, 16, dtype=torch.bfloat16),
        "gate_proj.bias": torch.zeros(2, dtype=torch.bfloat16),
    }
    safetensors.torch.save_file(tensors, str(ckpt_dir / "model.safetensors"))


def _encode_tiny_mp4_b64() -> str:
    buf = io.BytesIO()
    with av.open(buf, mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=8)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        for idx in range(2):
            arr = np.full((16, 16, 3), 60 + 40 * idx, dtype=np.uint8)
            for pkt in stream.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "MageVLTransformersBackend and mage_vl.serving.server default to "
        "device='cuda' instead of current_omni_platform on TPU"
    ),
)
def test_mage_vl_duplex_server(tmp_path: Path) -> None:
    """Mage-VL Duplex Server on TPU over WS /v1/mage-vl/duplex.

    - TPU path: ``MageVLTransformersBackend.load()`` (default device matching CLI)
      wired into ``mage_vl.serving.server.create_app`` and ``WS /v1/mage-vl/duplex``.
    - Setup check: ``current_omni_platform.device_type == "tpu"``, model loaded onto
      ``"tpu"``, and ``/health`` and ``/ready`` return HTTP 200.
    - Outcome check: ``input.append`` video window triggers proactive gate and
      generation matching the CPU ``MageVLTransformersBackend`` reference.
    """
    assert current_omni_platform.device_type == "tpu", (
        f"Expected TPU omni platform, got {current_omni_platform.device_type!r}"
    )

    ckpt_dir = tmp_path / "tiny_mage_vl"
    _write_tiny_mage_checkpoint(ckpt_dir)
    video_b64 = _encode_tiny_mp4_b64()

    cpu_backend = MageVLTransformersBackend(
        str(ckpt_dir),
        device="cpu",
        window_size=1,
        max_new_tokens=2,
        gate_threshold=0.5,
    )
    cpu_backend.load()
    dummy_session = DuplexSession("cpu-ref", DuplexSessionConfig())
    window = MageVLCodecWindow(
        segment_id="seg-0", data={"segment_id": "seg-0", "video_base64": video_b64}
    )
    ref_gate = asyncio.run(cpu_backend.gate(dummy_session, [window]))
    ref_text = asyncio.run(
        cpu_backend.generate(dummy_session, [window], None, ref_gate)
    )

    # Exercise the server's default backend configuration (no --device override).
    tpu_backend = MageVLTransformersBackend(
        str(ckpt_dir),
        window_size=1,
        max_new_tokens=2,
        gate_threshold=0.5,
    )
    tpu_backend.load()
    model_device = next(tpu_backend.model.parameters()).device.type
    assert model_device == "tpu", f"Expected Mage-VL model on tpu, got {model_device!r}"

    tpu_gate = asyncio.run(tpu_backend.gate(dummy_session, [window]))
    assert abs(tpu_gate.score - ref_gate.score) <= GATE_SCORE_ATOL, (
        f"TPU gate score {tpu_gate.score:.4f} diverged from CPU {ref_gate.score:.4f}"
    )

    app = create_app(
        tpu_backend.adapter_factory,
        MageVLServingConfig(model=str(ckpt_dir), auth_token=AUTH_TOKEN),
    )
    with TestClient(app) as client:
        health = client.get("/health")
        ready = client.get("/ready")
        assert health.status_code == 200, f"GET /health returned {health.status_code}"
        assert ready.status_code == 200, f"GET /ready returned {ready.status_code}"

        with client.websocket_connect(
            "/v1/mage-vl/duplex?session_id=tpu-sess",
            headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
        ) as ws:
            created = ws.receive_json()
            ws.send_json(
                {
                    "type": "input.append",
                    "modality": "video",
                    "data": {
                        "segment_id": "seg-0",
                        "data": {"video_base64": video_b64},
                    },
                }
            )
            resp_created = ws.receive_json()
            delta = ws.receive_json()
            done = ws.receive_json()
            ws.send_json({"type": "close"})

    assert created["type"] == "session.created", f"Unexpected first event: {created}"
    assert resp_created["type"] == "response.created", (
        f"Expected response.created, got {resp_created}"
    )
    assert delta["type"] == "response.delta", f"Expected response.delta, got {delta}"
    assert delta["data"] == ref_text, (
        f"TPU duplex text {delta['data']!r} != CPU reference {ref_text!r}"
    )
    assert done["type"] == "response.done", f"Expected response.done, got {done}"
