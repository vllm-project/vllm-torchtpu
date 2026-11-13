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
"""Unit tests for src/vllm_torchtpu/omni/worker/tpu_ar_model_runner.py."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.multimodal.utils import set_mm_embedding_modality
from vllm.sequence import IntermediateTensors
from vllm_omni.config import OmniModelConfig
from vllm_omni.distributed.omni_connectors.kv_transfer_manager import (
    OmniKVTransferManager,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

from vllm_torchtpu.omni.worker.tpu_ar_model_runner import (
    TPUARModelRunner,
    _expand_mm_dummy_batches,
)
from vllm_torchtpu.omni.worker.tpu_model_runner import OmniTPUModelRunner
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.runner.tpu_runner_async_output import AsyncTPUModelRunnerOutput

pytestmark = pytest.mark.omni


def _stage_config(**fields: Any) -> OmniModelConfig:
    defaults = dict(
        model_arch_config=SimpleNamespace(
            architectures=["Qwen3OmniMoeForConditionalGeneration"]
        ),
        model_stage="thinker",
        stage_id=0,
        requires_full_payload_input=False,
        custom_process_next_stage_input_func=None,
        stage_connector_config=None,
    )
    config = OmniModelConfig.__new__(OmniModelConfig)
    for name, value in (defaults | fields).items():
        object.__setattr__(config, name, value)
    return config


@pytest.fixture(autouse=True)
def _stub_tpu_model_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_init(self: Any, vllm_config: Any, *args: Any, **kwargs: Any) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.speculative_config = vllm_config.speculative_config

    monkeypatch.setattr(TPUModelRunner, "__init__", _fake_init)
    monkeypatch.setattr(
        "vllm_torchtpu.omni.worker.tpu_ar_model_runner.maybe_patch_qwen3_omni_thinker",
        lambda *_: False,
    )
    monkeypatch.setattr(
        "vllm_torchtpu.omni.worker.tpu_ar_model_runner.maybe_patch_qwen3_omni_talker",
        lambda *_: False,
    )


@pytest.fixture
def tpu_ar_model_runner(request: pytest.FixtureRequest) -> TPUARModelRunner:
    model_config = getattr(request, "param", None)
    if model_config is None:
        model_config = _stage_config()
    vllm_config = SimpleNamespace(
        model_config=model_config,
        speculative_config=None,
    )
    return TPUARModelRunner(vllm_config, "tpu")


def _async_output(inner: Any) -> AsyncTPUModelRunnerOutput:
    out = AsyncTPUModelRunnerOutput.__new__(AsyncTPUModelRunnerOutput)
    out._model_runner_output = inner
    return out


class _StageModel:
    def __init__(
        self,
        result: Any,
        *,
        has_preprocess: bool = False,
        has_postprocess: bool = False,
    ):
        self.result = result
        self.has_preprocess = has_preprocess
        self.has_postprocess = has_postprocess
        self.wrapped: list[Any] = []
        self.kwargs: dict[str, Any] = {}

    def __call__(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return self.result

    def make_omni_output(self, model_outputs: Any, **kwargs: Any) -> OmniOutput:
        self.wrapped.append(model_outputs)
        hidden = model_outputs[0] if isinstance(model_outputs, tuple) else model_outputs
        return OmniOutput(text_hidden_states=hidden, multimodal_outputs={"embed": {}})


@pytest.mark.parametrize(
    ("tpu_ar_model_runner", "expected"),
    [
        (_stage_config(model_stage="thinker", stage_id=0), ("thinker", 0)),
        (_stage_config(model_stage="talker", stage_id=1), ("talker", 1)),
        (SimpleNamespace(architectures=[]), (None, None)),
    ],
    indirect=["tpu_ar_model_runner"],
    ids=["thinker_stage_config", "talker_stage_config", "plain_vllm_config"],
)
def test_tpu_ar_model_runner_stage_metadata(
    tpu_ar_model_runner: TPUARModelRunner,
    expected: tuple[str | None, int | None],
) -> None:
    assert (tpu_ar_model_runner.model_stage, tpu_ar_model_runner.stage_id) == expected


def test_tpu_ar_model_runner_init_connectors_and_rejects_speculative() -> None:
    fake_kv_mgr = object()
    with (
        patch.object(
            OmniKVTransferManager, "from_vllm_config", return_value=fake_kv_mgr
        ) as mock_kv,
        patch.object(TPUARModelRunner, "init_omni_connectors") as mock_init_conn,
    ):
        conn_cfg = _stage_config(
            custom_process_next_stage_input_func="pkg.mod.thinker2talker"
        )
        vllm_cfg = SimpleNamespace(model_config=conn_cfg, speculative_config=None)
        conn_runner = TPUARModelRunner(vllm_cfg, "tpu")
        mock_kv.assert_called_once_with(conn_runner.vllm_config, conn_cfg)
        mock_init_conn.assert_called_once_with(
            model_config=conn_cfg,
            kv_transfer_manager=fake_kv_mgr,
        )

    with pytest.raises(NotImplementedError, match="speculative_config"):
        TPUARModelRunner(
            SimpleNamespace(
                model_config=_stage_config(),
                speculative_config=SimpleNamespace(method="eagle3"),
            ),
            "tpu",
        )


@pytest.mark.parametrize(
    ("has_preprocess", "has_postprocess"),
    [
        (False, False),
        (True, True),
    ],
    ids=["thinker", "talker"],
)
def test_load_model_applies_thinker_and_talker_patches(
    monkeypatch: pytest.MonkeyPatch,
    tpu_ar_model_runner: TPUARModelRunner,
    has_preprocess: bool,
    has_postprocess: bool,
) -> None:
    model = _StageModel(
        torch.zeros(1, 4),
        has_preprocess=has_preprocess,
        has_postprocess=has_postprocess,
    )
    monkeypatch.setattr(
        TPUModelRunner,
        "load_model",
        lambda self: setattr(self, "model", model),
    )
    tpu_ar_model_runner.load_model()
    assert tpu_ar_model_runner.get_model() is model


_HIDDEN = torch.arange(8.0).reshape(2, 4)
_STAGED = (_HIDDEN, {"layer_0": torch.zeros(2, 4)})
_INTERMEDIATE = IntermediateTensors({"hidden_states": _HIDDEN})


@pytest.mark.parametrize(
    ("raw_output", "intermediate_tensors", "expected_out", "expected_wrapped"),
    [
        (_HIDDEN, None, _HIDDEN, [_HIDDEN]),
        ([_HIDDEN], None, _HIDDEN, [_HIDDEN]),
        (_STAGED, None, _HIDDEN, [_STAGED]),
        (OmniOutput(text_hidden_states=_HIDDEN), None, _HIDDEN, []),
        (_INTERMEDIATE, _INTERMEDIATE, _INTERMEDIATE, []),
    ],
    ids=[
        "tensor",
        "single_elem_list",
        "staged_tuple",
        "omni_output",
        "intermediate_tensors",
    ],
)
def test_forward_model_normalizes_stage_outputs(
    tpu_ar_model_runner: TPUARModelRunner,
    raw_output: Any,
    intermediate_tensors: IntermediateTensors | None,
    expected_out: Any,
    expected_wrapped: list[Any],
) -> None:
    ids, pos = torch.tensor([1, 2]), torch.tensor([0, 1])
    model = _StageModel(raw_output)
    tpu_ar_model_runner.model = model

    out, aux = tpu_ar_model_runner.forward_model(
        ids, pos, intermediate_tensors=intermediate_tensors
    )
    assert out is expected_out
    assert aux is None
    assert model.wrapped == expected_wrapped
    if intermediate_tensors is None:
        assert "intermediate_tensors" not in model.kwargs
    else:
        assert model.kwargs["intermediate_tensors"] is intermediate_tensors


def test_tpu_ar_model_runner_mixed_dim_mm_slicing(
    tpu_ar_model_runner: TPUARModelRunner,
) -> None:
    """Verify _gather_mm_embeddings and _get_model_inputs preserve mixed widths."""
    vid_emb = set_mm_embedding_modality(torch.ones(4, 32), "video")
    aud_emb = set_mm_embedding_modality(torch.full((3, 8), 2.0), "audio")
    mask = torch.ones(10, dtype=torch.bool)

    with patch.object(
        TPUModelRunner,
        "_gather_mm_embeddings",
        return_value=([vid_emb, aud_emb], mask),
    ):
        idx_list, out_mask = tpu_ar_model_runner._gather_mm_embeddings(
            SimpleNamespace()
        )

    assert out_mask is mask
    assert len(idx_list) == 1
    assert torch.equal(idx_list[0], torch.arange(7, dtype=torch.long))

    # Simulate TPUModelRunner.execute_model slicing row indices [2:6] across the
    # video->audio boundary (last 2 rows of video + first 2 rows of audio).
    captured_mm: list[Any] = []

    def _fake_parent_get_model_inputs(self, input_ids, mm_embed_inputs):
        captured_mm.append(mm_embed_inputs)
        return None, input_ids

    with patch.object(
        TPUModelRunner, "_get_model_inputs", _fake_parent_get_model_inputs
    ):
        tpu_ar_model_runner._get_model_inputs(
            torch.zeros(10, dtype=torch.long),
            ([idx_list[0][2:6]], mask),
        )

    resolved_embeds, resolved_mask = captured_mm[0]
    assert resolved_mask is mask
    assert len(resolved_embeds) == 2
    assert resolved_embeds[0].shape == (2, 32)
    assert resolved_embeds[0].modality == "video"
    assert resolved_embeds[1].shape == (2, 8)
    assert resolved_embeds[1].modality == "audio"


def test_tpu_ar_model_runner_capture_model_precompiles_mm_buckets(
    tpu_ar_model_runner: TPUARModelRunner,
) -> None:
    runner = tpu_ar_model_runner
    runner.device = torch.device("cpu")
    runner.enforce_eager = False
    runner.supports_mm_inputs = True
    runner._pp_is_first = True
    runner.num_tokens_paddings = [16, 64]
    runner._update_num_xla_graphs = MagicMock()

    seen_modalities: list[str] = []
    seen_merges: list[tuple[int, tuple[tuple[int, ...], str] | None]] = []
    reset_cache = MagicMock()
    runner.mm_budget = SimpleNamespace(
        mm_max_toks_per_item={"video": 65536, "audio": 390, "image": 4096},
        reset_cache=reset_cache,
    )
    runner._get_mm_dummy_batch = lambda modality, count: {"modality": modality}

    def _embed_multimodal(**kwargs: Any) -> tuple[torch.Tensor, ...]:
        modality = kwargs["modality"]
        seen_modalities.append(modality)
        if modality == "empty":
            return ()
        width = 8 if modality == "audio" else 32
        return (torch.ones(4, width),)

    def _embed_input_ids(
        input_ids: torch.Tensor,
        multimodal_embeddings: Any = None,
        *,
        is_multimodal: Any = None,
    ) -> torch.Tensor:
        mm_info = (
            None
            if multimodal_embeddings is None
            else (
                tuple(multimodal_embeddings[0].shape),
                multimodal_embeddings[0].modality,
            )
        )
        seen_merges.append((int(input_ids.shape[0]), mm_info))
        return torch.zeros(input_ids.shape[0], 8)

    runner.model = SimpleNamespace(
        embed_multimodal=_embed_multimodal,
        embed_input_ids=_embed_input_ids,
    )

    with patch.object(OmniTPUModelRunner, "capture_model") as mock_super_capture:
        runner.capture_model()
        mock_super_capture.assert_called_once_with()
        assert seen_modalities == ["audio", "image"]
        reset_cache.assert_called_once_with()
        assert seen_merges == [
            (16, ((1, 8), "audio")),
            (64, ((1, 8), "audio")),
            (16, ((1, 32), "image")),
            (64, ((1, 32), "image")),
            (16, None),
        ]

        # Empty embed_multimodal logs warning_once and skips merge warmup
        seen_modalities.clear()
        seen_merges.clear()
        runner.mm_budget.mm_max_toks_per_item = {"empty": 100}
        with patch(
            "vllm_torchtpu.omni.worker.tpu_ar_model_runner.logger.warning_once"
        ) as mock_warn:
            runner.capture_model()
            mock_warn.assert_called_once()
        assert seen_modalities == ["empty"]
        assert seen_merges == []

        # Non-MM, enforce_eager, non-first PP stage, or talker stage skips MM warmup
        runner.mm_budget.mm_max_toks_per_item = {"audio": 390}
        for attr in ("supports_mm_inputs", "_pp_is_first"):
            seen_modalities.clear()
            seen_merges.clear()
            setattr(runner, attr, False)
            runner.capture_model()
            assert seen_modalities == []
            assert seen_merges == []
            setattr(runner, attr, True)

        seen_modalities.clear()
        seen_merges.clear()
        runner.model_stage = "talker"
        runner.capture_model()
        assert seen_modalities == []
        assert seen_merges == []
        runner.model_stage = "thinker"

        seen_modalities.clear()
        seen_merges.clear()
        runner.enforce_eager = True
        runner.capture_model()
        assert seen_modalities == []
        assert seen_merges == []

    # Verify _expand_mm_dummy_batches expands audio and image dummy inputs
    # across all static buckets in _AUDIO_CHUNK_BUCKETS and _VISION_PATCH_BUCKETS.
    audio_batches = _expand_mm_dummy_batches(
        "audio",
        {
            "input_features": torch.zeros(128, 3000),
            "audio_feature_lengths": torch.tensor([3000], dtype=torch.long),
        },
    )
    assert len(audio_batches) == 1
    assert audio_batches[0]["input_features"].shape == (128, 800 + 1600 + 3200)
    assert audio_batches[0]["audio_feature_lengths"].tolist() == [800, 1600, 3200]

    image_batches = _expand_mm_dummy_batches(
        "image",
        {
            "pixel_values": torch.zeros(16384, 16),
            "image_grid_thw": torch.tensor([[1, 128, 128]], dtype=torch.int32),
        },
    )
    assert [int(b["pixel_values"].shape[0]) for b in image_batches] == [
        4096,
        8192,
        16384,
    ]
    assert [b["image_grid_thw"].tolist() for b in image_batches] == [
        [[1, 64, 64]],
        [[1, 64, 128]],
        [[1, 64, 256]],
    ]


def test_tpu_ar_model_runner_talker_hooks_and_inter_stage_outputs() -> None:
    runner = TPUARModelRunner(
        SimpleNamespace(
            model_config=_stage_config(model_stage="talker", stage_id=1),
            speculative_config=None,
        ),
        "tpu",
    )
    runner.supports_mm_inputs = False
    runner.input_batch = SimpleNamespace(
        num_reqs=1,
        req_ids=["req-0"],
        num_computed_tokens_cpu=[4],
    )
    runner.query_start_loc_np = [0, 1]
    runner.requests = {"req-0": SimpleNamespace(prompt_token_ids=[10, 11, 12, 13])}
    runner._omni_chunk_req_slice = (0, 1)

    class DummyTalkerStage:
        has_preprocess = True
        has_postprocess = True
        gpu_resident_buffer_keys = {("hidden_states", "last"), ("codes", "audio")}

        def __call__(self, **kwargs: Any) -> torch.Tensor:
            return torch.full((16, 8), 9.0)

        def preprocess(
            self, input_ids: Any, input_embeds: Any = None, **req_info: Any
        ) -> tuple[Any, torch.Tensor, dict[str, Any]]:
            assert req_info["_omni_num_computed_tokens"] == 4
            assert req_info["_omni_is_prefill"] is False
            return (
                input_ids,
                torch.ones(1, 8),
                {"mtp_inputs": (torch.full((1, 8), 2.0), torch.full((1, 8), 3.0))},
            )

        def talker_mtp(
            self,
            input_ids: Any,
            input_embeds: torch.Tensor,
            last_hidden: torch.Tensor,
            text_step: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return input_embeds + last_hidden + text_step, torch.arange(16).unsqueeze(0)

        def postprocess(
            self, hidden_states: torch.Tensor, **kwargs: Any
        ) -> dict[str, Any]:
            return {"hidden_states": {"last": hidden_states[-1].detach()}}

        def make_omni_output(
            self, model_outputs: torch.Tensor, model_intermediate_buffer: Any = None
        ) -> OmniOutput:
            codes = [b["codes"]["audio"] for b in model_intermediate_buffer]
            audio_codes = torch.cat(codes, dim=0)
            return OmniOutput(
                text_hidden_states=model_outputs[: audio_codes.shape[0]],
                multimodal_outputs={"codes": {"audio": audio_codes}},
            )

    runner.model = DummyTalkerStage()
    out_ids, out_embeds = runner._get_model_inputs(
        torch.tensor([42], dtype=torch.long), None
    )
    assert out_ids is None
    assert out_embeds is not None
    assert torch.equal(out_embeds, torch.full((1, 8), 6.0))
    assert torch.equal(
        runner.model_intermediate_buffer["req-0"]["codes"]["audio"],
        torch.arange(16).unsqueeze(0),
    )

    # forward_model preserves the full padded bucket tensor (16, 8) for TPUModelRunner
    # while recording multimodal_outputs and consuming _omni_chunk_req_slice.
    fwd_out, fwd_aux = runner.forward_model(None, torch.arange(16), out_embeds)
    assert isinstance(fwd_out, torch.Tensor)
    assert fwd_out.shape == (16, 8)
    assert fwd_aux is None
    assert runner._omni_chunk_req_slice is None
    assert torch.equal(
        runner.model_intermediate_buffer["req-0"]["hidden_states"]["last"],
        torch.full((8,), 9.0),
    )

    # A subsequent warmup/_dummy_run call with _omni_chunk_req_slice=None skips
    # preprocess/postprocess and does not overwrite _omni_chunk_outputs.
    dummy_ids, dummy_embeds = runner._get_model_inputs(
        torch.tensor([99], dtype=torch.long), None
    )
    assert dummy_ids is not None
    assert torch.equal(dummy_ids, torch.tensor([99], dtype=torch.long))
    assert dummy_embeds is None
    runner.forward_model(dummy_ids, torch.arange(1), None)
    assert len(runner._omni_chunk_outputs) == 1

    inner_out = SimpleNamespace(req_ids=["req-0"], inter_stage_outputs=None)
    runner._omni_connector_initialized = True
    runner._should_accumulate_full_payload_output = MagicMock(return_value=False)
    runner.attach_omni_connector_output = MagicMock(side_effect=lambda x: x)
    with patch.object(
        OmniTPUModelRunner, "sample_tokens", return_value=_async_output(inner_out)
    ):
        runner.sample_tokens()
    assert len(inner_out.inter_stage_outputs) == 1
    assert torch.equal(
        inner_out.inter_stage_outputs[0]["codes"]["audio"],
        torch.arange(16).unsqueeze(0),
    )
    runner.attach_omni_connector_output.assert_called_once_with(inner_out)


def test_tpu_ar_model_runner_multi_chunk_inter_stage_outputs(
    tpu_ar_model_runner: TPUARModelRunner,
) -> None:
    """Multi-chunk step slices each request from its own chunk's outputs."""
    runner = tpu_ar_model_runner
    runner.input_batch = SimpleNamespace(
        num_reqs=2,
        req_ids=["req-decode", "req-prefill"],
        num_computed_tokens_cpu=[3, 0],
    )

    class MultiChunkThinker:
        has_preprocess = False
        has_postprocess = False

        def __init__(self) -> None:
            self.call_idx = 0

        def __call__(self, **kwargs: Any) -> tuple[torch.Tensor, dict[str, Any]]:
            self.call_idx += 1
            if self.call_idx == 1:
                # Chunk 0: 1 decode token for req-decode (padded to 4)
                hs = torch.full((4, 4), 10.0)
                layer = torch.arange(16, dtype=torch.float32).reshape(4, 4)
                return hs, {"hidden_states": {"layers": {"24": layer}}}
            # Chunk 1: 3 prefill tokens for req-prefill (padded to 4)
            hs = torch.full((4, 4), 20.0)
            layer = torch.arange(100, 116, dtype=torch.float32).reshape(4, 4)
            return hs, {"hidden_states": {"layers": {"24": layer}}}

        def make_omni_output(
            self, model_outputs: tuple[torch.Tensor, dict[str, Any]], **kwargs: Any
        ) -> OmniOutput:
            hs, sidecar = model_outputs
            return OmniOutput(text_hidden_states=hs, multimodal_outputs=sidecar)

    runner.model = MultiChunkThinker()

    # Chunk 0: req-decode [0:1]
    runner._omni_chunk_req_slice = (0, 1)
    runner.query_start_loc_np = [0, 1]
    runner.forward_model(torch.zeros(4, dtype=torch.int32), torch.arange(4))

    # Chunk 1: req-prefill [1:2] with 3 tokens
    runner._omni_chunk_req_slice = (1, 2)
    runner.query_start_loc_np = [0, 3]
    runner.forward_model(torch.zeros(4, dtype=torch.int32), torch.arange(4))

    inner_out = SimpleNamespace(
        req_ids=["req-decode", "req-prefill"], inter_stage_outputs=None
    )
    with patch.object(
        OmniTPUModelRunner, "sample_tokens", return_value=_async_output(inner_out)
    ):
        runner.sample_tokens()

    assert len(inner_out.inter_stage_outputs) == 2
    assert torch.equal(
        inner_out.inter_stage_outputs[0]["hidden_states"]["layers"]["24"],
        torch.arange(16, dtype=torch.float32).reshape(4, 4)[:1],
    )
    assert torch.equal(
        inner_out.inter_stage_outputs[1]["hidden_states"]["layers"]["24"],
        torch.arange(100, 116, dtype=torch.float32).reshape(4, 4)[:3],
    )


def test_tpu_ar_model_runner_full_payload_connector_lifecycle() -> None:
    runner = TPUARModelRunner(
        SimpleNamespace(
            model_config=_stage_config(
                model_stage="talker", stage_id=1, async_chunk=False
            ),
            speculative_config=None,
        ),
        "tpu",
    )
    runner._omni_connector_initialized = True
    runner._pending_full_payload_send = {"req-done", "req-evicted"}
    runner.requests = {"req-done": SimpleNamespace(prompt_token_ids=[1, 2, 3])}
    runner.register_chunk_recv = MagicMock()
    runner.recv_full_payload_inputs = MagicMock()
    runner.flush_full_payload_outputs = MagicMock()
    runner.attach_omni_connector_output = MagicMock(side_effect=lambda x: x)

    sched_out = SimpleNamespace(
        pending_input_registrations=["reg-0"],
        finished_req_ids={"req-done"},
    )
    with patch.object(OmniTPUModelRunner, "execute_model", return_value=None):
        assert runner.execute_model(sched_out) is None

    runner.register_chunk_recv.assert_called_once_with("reg-0")
    runner.recv_full_payload_inputs.assert_called_once_with(sched_out)
    runner.flush_full_payload_outputs.assert_called_once_with(
        {"req-done", "req-evicted"}
    )


def test_sync_local_stage_payloads_moves_ready_active_payloads() -> None:
    runner = TPUARModelRunner(
        SimpleNamespace(
            model_config=_stage_config(
                model_stage="talker", stage_id=1, async_chunk=False
            ),
            speculative_config=None,
        ),
        "tpu",
    )
    runner.model = SimpleNamespace(gpu_resident_buffer_keys=set())
    runner.requests = {"req-ready": SimpleNamespace(), "req-pending": SimpleNamespace()}
    runner._local_stage_payload_cache = {
        "req-ready": {"embed": {"prefill": torch.ones(2, 4)}},
        "req-pending": {"embed": {"prefill": torch.zeros(2, 4)}},
        "req-unknown": {"embed": {"prefill": torch.ones(1, 4)}},
    }
    runner._full_payload_pending_broadcast_req_ids = {"req-pending"}

    runner._sync_local_stage_payloads()

    assert "req-ready" not in runner._local_stage_payload_cache
    assert "req-pending" in runner._local_stage_payload_cache
    assert "req-unknown" in runner._local_stage_payload_cache
    assert torch.equal(
        runner.model_intermediate_buffer["req-ready"]["embed"]["prefill"],
        torch.ones(2, 4),
    )
    assert "req-pending" not in runner.model_intermediate_buffer
