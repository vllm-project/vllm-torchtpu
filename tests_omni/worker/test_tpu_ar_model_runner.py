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
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.sequence import IntermediateTensors
from vllm_omni.config import OmniModelConfig
from vllm_omni.distributed.omni_connectors.kv_transfer_manager import (
    OmniKVTransferManager,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.worker.omni_connector_model_runner_mixin import (
    OmniConnectorModelRunnerMixin,
)

from vllm_torchtpu.omni.worker.tpu_ar_model_runner import TPUARModelRunner
from vllm_torchtpu.omni.worker.tpu_model_runner import OmniTPUModelRunner
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.runner.tpu_runner_async_output import AsyncTPUModelRunnerOutput

pytestmark = pytest.mark.omni


def _stage_config(**fields) -> OmniModelConfig:
    defaults = dict(
        model_stage="thinker",
        stage_id=0,
        requires_full_payload_input=False,
        custom_process_next_stage_input_func=None,
        stage_connector_config=None,
    )
    config = OmniModelConfig.__new__(OmniModelConfig)
    for name, value in {**defaults, **fields}.items():
        object.__setattr__(config, name, value)
    return config


def _build(model_config=None, model=None, speculative_config=None) -> TPUARModelRunner:
    if model_config is None:
        model_config = _stage_config()

    def _fake_init(self, vllm_config, *args, **kwargs):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.speculative_config = vllm_config.speculative_config

    with patch.object(TPUModelRunner, "__init__", _fake_init):
        runner = TPUARModelRunner(
            SimpleNamespace(
                model_config=model_config,
                speculative_config=speculative_config,
            ),
            "tpu",
        )
    if model is not None:
        with (
            patch(
                "vllm_torchtpu.omni.worker.tpu_ar_model_runner."
                "maybe_patch_qwen3_omni_thinker"
            ) as mock_thinker_patch,
            patch(
                "vllm_torchtpu.omni.worker.tpu_ar_model_runner."
                "maybe_patch_qwen3_omni_talker"
            ) as mock_talker_patch,
            patch.object(
                TPUModelRunner, "load_model", lambda self: setattr(self, "model", model)
            ),
        ):
            runner.load_model()
            mock_thinker_patch.assert_called_once_with(runner.model_config)
            mock_talker_patch.assert_called_once_with(runner.model_config)
    return runner


def _async_output(inner):
    out = AsyncTPUModelRunnerOutput.__new__(AsyncTPUModelRunnerOutput)
    out._model_runner_output = inner
    return out


class _StageModel:
    def __init__(self, result, *, has_preprocess=False, has_postprocess=False):
        self.result = result
        self.has_preprocess = has_preprocess
        self.has_postprocess = has_postprocess
        self.wrapped = []

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return self.result

    def make_omni_output(self, model_outputs, **kwargs) -> OmniOutput:
        self.wrapped.append(model_outputs)
        hidden = model_outputs[0] if isinstance(model_outputs, tuple) else model_outputs
        return OmniOutput(text_hidden_states=hidden, multimodal_outputs={"embed": {}})


def test_tpu_ar_model_runner_init_and_inheritance():
    assert TPUARModelRunner.__bases__ == (
        OmniTPUModelRunner,
        OmniConnectorModelRunnerMixin,
    )
    assert OmniConnectorModelRunnerMixin not in OmniTPUModelRunner.__mro__

    runner = _build(_stage_config(model_stage="thinker", stage_id=0))
    assert (runner.model_stage, runner.stage_id) == ("thinker", 0)
    talker = _build(_stage_config(model_stage="talker", stage_id=1))
    assert (talker.model_stage, talker.stage_id) == ("talker", 1)
    plain = _build(SimpleNamespace())
    assert (plain.model_stage, plain.stage_id) == (None, None)

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
        conn_runner = _build(conn_cfg)
        mock_kv.assert_called_once_with(conn_runner.vllm_config, conn_cfg)
        mock_init_conn.assert_called_once_with(
            model_config=conn_cfg,
            kv_transfer_manager=fake_kv_mgr,
        )

    with pytest.raises(NotImplementedError, match="speculative_config"):
        _build(speculative_config=SimpleNamespace(method="eagle3"))


def test_load_model_applies_thinker_and_talker_patches():
    thinker = _StageModel(torch.zeros(1, 4))
    assert _build(model=thinker).get_model() is thinker

    talker = _StageModel(torch.zeros(1, 4), has_preprocess=True, has_postprocess=True)
    assert _build(model=talker).get_model() is talker


def test_forward_model_normalizes_stage_outputs():
    ids, pos = torch.tensor([1, 2]), torch.tensor([0, 1])
    hidden = torch.arange(8.0).reshape(2, 4)
    staged = (hidden, {"layer_0": torch.zeros(2, 4)})

    for raw, expected_wrapped in [
        (hidden, [hidden]),
        ([hidden], [hidden]),
        (staged, [staged]),
        (OmniOutput(text_hidden_states=hidden), []),
    ]:
        runner = _build(model=(model := _StageModel(raw)))
        assert runner.forward_model(ids, pos) == (hidden, None)
        assert model.wrapped == expected_wrapped
        assert "intermediate_tensors" not in model.kwargs

    tensors = IntermediateTensors({"hidden_states": hidden})
    runner = _build(model=(model := _StageModel(tensors)))
    assert runner.forward_model(ids, pos, intermediate_tensors=tensors) == (
        tensors,
        None,
    )
    assert model.kwargs["intermediate_tensors"] is tensors


def test_tpu_ar_model_runner_mixed_dim_mm_slicing():
    runner = _build(_stage_config(model_stage="thinker", stage_id=0))
    runner.supports_mm_inputs = True

    vid_emb, aud_emb = torch.ones(6, 32), torch.ones(4, 8)
    vid_emb.modality, aud_emb.modality = "video", "audio"
    mask = torch.ones(10, dtype=torch.bool)

    with patch.object(
        OmniTPUModelRunner,
        "_gather_mm_embeddings",
        return_value=([vid_emb, aud_emb], mask),
    ):
        idx_list, out_mask = runner._gather_mm_embeddings("sched_out")
    assert len(idx_list) == 1 and torch.equal(idx_list[0], torch.arange(10))
    assert out_mask is mask

    captured_mm = []

    def _dummy_embed_input_ids(ids, **kwargs):
        captured_mm.append(kwargs.get("multimodal_embeddings"))
        return torch.zeros(ids.shape[0], 8)

    runner.model = SimpleNamespace(embed_input_ids=_dummy_embed_input_ids)
    _, embeds_out = runner._get_model_inputs(
        torch.zeros(5, dtype=torch.long),
        ([torch.arange(4, 9, dtype=torch.int64)], torch.ones(5, dtype=torch.bool)),
    )
    assert embeds_out.shape == (5, 8) and len(captured_mm[0]) == 2
    assert captured_mm[0][0].shape == (2, 32)
    assert captured_mm[0][0].modality == "video"
    assert captured_mm[0][1].shape == (3, 8)
    assert captured_mm[0][1].modality == "audio"
    assert len(runner._omni_mm_embeds_list) == 2

    # Slicing through the final embedding row clears _omni_mm_embeds_list.
    runner._get_model_inputs(
        torch.zeros(1, dtype=torch.long),
        ([torch.arange(9, 10, dtype=torch.int64)], torch.ones(1, dtype=torch.bool)),
    )
    assert runner._omni_mm_embeds_list == []


def test_tpu_ar_model_runner_capture_model_precompiles_mm_buckets():
    runner = _build(_stage_config(model_stage="thinker", stage_id=0))
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

    def _embed_multimodal(**kwargs):
        modality = kwargs["modality"]
        seen_modalities.append(modality)
        if modality == "empty":
            return ()
        width = 8 if modality == "audio" else 32
        return (torch.ones(4, width),)

    def _embed_input_ids(input_ids, multimodal_embeddings=None, *, is_multimodal=None):
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


def test_tpu_ar_model_runner_talker_hooks_and_inter_stage_outputs():
    runner = _build(_stage_config(model_stage="talker", stage_id=1))
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

        def __call__(self, **kwargs):
            return torch.full((16, 8), 9.0)

        def preprocess(self, input_ids, input_embeds=None, **req_info):
            assert req_info["_omni_num_computed_tokens"] == 4
            assert req_info["_omni_is_prefill"] is False
            return (
                input_ids,
                torch.ones(1, 8),
                {"mtp_inputs": (torch.full((1, 8), 2.0), torch.full((1, 8), 3.0))},
            )

        def talker_mtp(self, input_ids, input_embeds, last_hidden, text_step):
            return input_embeds + last_hidden + text_step, torch.arange(16).unsqueeze(0)

        def postprocess(self, hidden_states, **kwargs):
            return {"hidden_states": {"last": hidden_states[-1].detach()}}

        def make_omni_output(self, model_outputs, model_intermediate_buffer=None):
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
    assert torch.equal(out_embeds, torch.full((1, 8), 6.0))
    assert torch.equal(
        runner.model_intermediate_buffer["req-0"]["codes"]["audio"],
        torch.arange(16).unsqueeze(0),
    )

    # forward_model preserves the full padded bucket tensor (16, 8) for TPUModelRunner
    # while recording multimodal_outputs and consuming _omni_chunk_req_slice.
    fwd_out, fwd_aux = runner.forward_model(None, torch.arange(16), out_embeds)
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


def test_tpu_ar_model_runner_multi_chunk_inter_stage_outputs():
    """Multi-chunk step slices each request from its own chunk's outputs."""
    runner = _build(_stage_config(model_stage="thinker", stage_id=0))
    runner.input_batch = SimpleNamespace(
        num_reqs=2,
        req_ids=["req-decode", "req-prefill"],
        num_computed_tokens_cpu=[3, 0],
    )

    class MultiChunkThinker:
        has_preprocess = False
        has_postprocess = False

        def __init__(self):
            self.call_idx = 0

        def __call__(self, **kwargs):
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

        def make_omni_output(self, model_outputs, **kwargs):
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


def test_tpu_ar_model_runner_full_payload_connector_lifecycle():
    runner = _build(_stage_config(model_stage="talker", stage_id=1, async_chunk=False))
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


def test_sync_local_stage_payloads_moves_ready_active_payloads():
    runner = _build(_stage_config(model_stage="talker", stage_id=1, async_chunk=False))
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
