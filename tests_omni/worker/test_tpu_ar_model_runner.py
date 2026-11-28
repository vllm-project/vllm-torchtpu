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

from tests_omni.omni_test_utils import build_qwen3_omni_thinker_stub
from vllm_torchtpu.omni.worker.tpu_ar_model_runner import (
    TPUARModelRunner,
    _expand_mm_dummy_batches,
    _slice_mm_embeds_by_row_indices,
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
    for name, value in {**defaults, **fields}.items():
        object.__setattr__(config, name, value)
    return config


@pytest.fixture(autouse=True)
def _stub_tpu_model_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_init(self: Any, vllm_config: Any, *args: Any, **kwargs: Any) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.speculative_config = vllm_config.speculative_config
        self.enforce_eager = bool(getattr(self.model_config, "enforce_eager", False))
        self.supports_mm_inputs = True
        self.model: Any = SimpleNamespace(use_deepstack=False)

    monkeypatch.setattr(TPUModelRunner, "__init__", _fake_init)
    monkeypatch.setattr(
        "vllm_torchtpu.omni.worker.tpu_ar_model_runner.maybe_patch_qwen3_omni_thinker",
        lambda *_, **__: False,
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
    return TPUARModelRunner(
        vllm_config,
        torch.device("tpu"),
        profiler_rank=0,
        profiler_world_size=1,
    )


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
        conn_runner = TPUARModelRunner(
            vllm_cfg,
            torch.device("tpu"),
            profiler_rank=0,
            profiler_world_size=1,
        )
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
            torch.device("tpu"),
            profiler_rank=0,
            profiler_world_size=1,
        )


@pytest.mark.parametrize(
    ("has_preprocess", "has_postprocess"),
    [
        (False, False),
        (True, True),
    ],
    ids=["thinker", "talker"],
)
def test_load_model_accepts_thinker_and_talker_hook_capabilities(
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


@pytest.mark.parametrize("enforce_eager", [False, True])
def test_load_model_forwards_enforce_eager_to_thinker_patch(
    monkeypatch: pytest.MonkeyPatch,
    enforce_eager: bool,
) -> None:
    seen: list[tuple[Any, bool]] = []

    def _record_patch(archs: Any, *, enforce_eager: bool = False) -> bool:
        seen.append((archs, enforce_eager))
        return False

    monkeypatch.setattr(
        "vllm_torchtpu.omni.worker.tpu_ar_model_runner.maybe_patch_qwen3_omni_thinker",
        _record_patch,
    )
    model = _StageModel(torch.zeros(1, 4))
    monkeypatch.setattr(
        TPUModelRunner,
        "load_model",
        lambda self: setattr(self, "model", model),
    )
    vllm_config = SimpleNamespace(
        model_config=_stage_config(enforce_eager=enforce_eager),
        speculative_config=None,
    )
    runner = TPUARModelRunner(
        vllm_config,
        torch.device("tpu"),
        profiler_rank=0,
        profiler_world_size=1,
    )
    runner.load_model()
    assert seen == [(["Qwen3OmniMoeForConditionalGeneration"], enforce_eager)]


_HIDDEN = torch.arange(8.0).reshape(2, 4)
_AUX = torch.ones(2, 4)
_STAGED = (_HIDDEN, {"layer_0": torch.zeros(2, 4)})
_INTERMEDIATE = IntermediateTensors({"hidden_states": _HIDDEN})


class _PlainVllmModel:
    def __init__(self, result: Any) -> None:
        self.result = result

    def __call__(self, **kwargs: Any) -> Any:
        return self.result


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


@pytest.mark.parametrize(
    ("raw_output", "expected_out", "expected_aux"),
    [
        (_HIDDEN, _HIDDEN, None),
        ([_HIDDEN], _HIDDEN, None),
        ((_HIDDEN, _AUX), _HIDDEN, _AUX),
        (_INTERMEDIATE, _INTERMEDIATE, None),
    ],
    ids=["tensor", "single_elem_list", "draft_aux_tuple", "intermediate_tensors"],
)
def test_forward_model_supports_plain_vllm_models(
    tpu_ar_model_runner: TPUARModelRunner,
    raw_output: Any,
    expected_out: Any,
    expected_aux: Any,
) -> None:
    ids, pos = torch.tensor([1, 2]), torch.tensor([0, 1])
    tpu_ar_model_runner.model = _PlainVllmModel(raw_output)

    out, aux = tpu_ar_model_runner.forward_model(ids, pos)
    assert out is expected_out
    assert aux is expected_aux


@pytest.mark.parametrize(
    ("row_indices", "match"),
    [
        pytest.param(torch.zeros((2, 2), dtype=torch.long), "1D", id="non_1d"),
        pytest.param(torch.tensor([0, 2]), "invalid", id="gap"),
        pytest.param(torch.tensor([2, 1]), "invalid", id="descending"),
        pytest.param(torch.tensor([1, 1]), "invalid", id="duplicate"),
        pytest.param(torch.tensor([0, 2, 1, 3]), "contiguous", id="permuted"),
        pytest.param(torch.tensor([-1, 0]), "invalid", id="negative"),
        pytest.param(torch.tensor([5, 6, 7]), "invalid", id="out_of_bounds"),
    ],
)
def test_slice_mm_embeds_by_row_indices_rejects_invalid_indices(
    row_indices: torch.Tensor, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _slice_mm_embeds_by_row_indices(
            [torch.zeros(4, 32), torch.zeros(3, 8)], row_indices
        )


def test_tpu_ar_model_runner_mixed_dim_mm_slicing(
    monkeypatch: pytest.MonkeyPatch,
    tpu_ar_model_runner: TPUARModelRunner,
) -> None:
    """Verify gather + multi-chunk slicing preserves rows, widths, and modalities."""
    vid = set_mm_embedding_modality(torch.arange(4 * 32.0).reshape(4, 32), "video")
    aud = set_mm_embedding_modality(
        torch.arange(1000.0, 1000.0 + 3 * 8).reshape(3, 8), "audio"
    )
    img = set_mm_embedding_modality(
        torch.arange(2000.0, 2000.0 + 2 * 32).reshape(2, 32), "image"
    )
    mask = torch.ones(10, dtype=torch.bool)
    seen_shifts: list[int] = []

    monkeypatch.setattr(
        TPUModelRunner,
        "_gather_mm_embeddings",
        lambda self, sched, shift=0: (
            seen_shifts.append(shift),
            ([vid, aud, img], mask),
        )[1],
    )
    idx_list, out_mask = tpu_ar_model_runner._gather_mm_embeddings(
        SimpleNamespace(), shift_computed_tokens=3
    )
    assert seen_shifts == [3]
    assert out_mask is mask
    assert len(idx_list) == 1 and torch.equal(idx_list[0], torch.arange(9))

    captured_mm: list[Any] = []
    monkeypatch.setattr(
        TPUModelRunner,
        "_get_model_inputs",
        lambda self, ids, mm: (captured_mm.append(mm), (None, ids))[1],
    )
    for start, end in ((0, 0), (0, 2), (2, 6), (6, 9)):
        tpu_ar_model_runner._get_model_inputs(
            torch.zeros(10, dtype=torch.long), ([idx_list[0][start:end]], mask)
        )

    assert captured_mm[0] == ([], mask)
    expected_chunks = [
        [(vid[0:2], "video")],
        [(vid[2:4], "video"), (aud[0:2], "audio")],
        [(aud[2:3], "audio"), (img[0:2], "image")],
    ]
    for (embeds, chunk_mask), exp in zip(captured_mm[1:], expected_chunks, strict=True):
        assert chunk_mask is mask
        assert len(embeds) == len(exp)
        for got, (exp_t, exp_mod) in zip(embeds, exp, strict=True):
            assert torch.equal(got, exp_t) and got.modality == exp_mod

    monkeypatch.setattr(
        TPUModelRunner, "_gather_mm_embeddings", lambda self, sched, shift=0: ([], mask)
    )
    assert tpu_ar_model_runner._gather_mm_embeddings(SimpleNamespace()) == ([], mask)
    assert tpu_ar_model_runner._omni_mm_embeds_list is None


@pytest.mark.parametrize(
    ("mm_embed_inputs", "cached"),
    [
        pytest.param(None, None, id="none"),
        pytest.param(
            ([], torch.zeros(4, dtype=torch.bool)), [torch.ones(2, 8)], id="empty"
        ),
        pytest.param(
            ([torch.ones(2, 8)], torch.ones(4, dtype=torch.bool)),
            [torch.ones(2, 8)],
            id="2d",
        ),
        pytest.param(
            ([torch.arange(2)], torch.ones(4, dtype=torch.bool)), None, id="uncached"
        ),
    ],
)
def test_get_model_inputs_passthrough(
    monkeypatch: pytest.MonkeyPatch,
    tpu_ar_model_runner: TPUARModelRunner,
    mm_embed_inputs: Any,
    cached: list[torch.Tensor] | None,
) -> None:
    tpu_ar_model_runner._omni_mm_embeds_list = cached
    captured: list[Any] = []
    monkeypatch.setattr(
        TPUModelRunner,
        "_get_model_inputs",
        lambda self, ids, mm: (captured.append(mm), (None, ids))[1],
    )
    tpu_ar_model_runner._get_model_inputs(
        torch.zeros(4, dtype=torch.long), mm_embed_inputs
    )
    assert captured == [mm_embed_inputs]


@pytest.mark.parametrize("should_raise", [False, True], ids=["normal", "exception"])
def test_execute_model_always_clears_mm_cache(
    monkeypatch: pytest.MonkeyPatch,
    tpu_ar_model_runner: TPUARModelRunner,
    should_raise: bool,
) -> None:
    sentinel = SimpleNamespace(req_ids=[], kv_connector_output=None)

    def _fake_execute_model(
        self: Any,
        scheduler_output: Any,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> Any:
        self._omni_mm_embeds_list = [torch.ones(2, 8)]
        if should_raise:
            raise RuntimeError("boom")
        return sentinel

    monkeypatch.setattr(TPUModelRunner, "execute_model", _fake_execute_model)
    if should_raise:
        with pytest.raises(RuntimeError, match="boom"):
            tpu_ar_model_runner.execute_model(SimpleNamespace())
    else:
        assert tpu_ar_model_runner.execute_model(SimpleNamespace()) is sentinel
    assert tpu_ar_model_runner._omni_mm_embeds_list is None


@pytest.mark.parametrize(
    ("step_kind", "stale_tokens", "expect_ds_sync"),
    [
        pytest.param("vision", 4, True, id="vision_updates_and_syncs"),
        pytest.param("text_only", 4, True, id="text_clears_stale_and_syncs"),
        pytest.param("text_only", 0, False, id="text_clean_no_sync"),
    ],
)
def test_get_model_inputs_deepstack_clear_and_sync(
    monkeypatch: pytest.MonkeyPatch,
    tpu_ar_model_runner: TPUARModelRunner,
    step_kind: str,
    stale_tokens: int,
    expect_ds_sync: bool,
) -> None:
    """Verify stale DeepStack clearing on entry and TPU buffer flush gating."""
    hidden_size = 8
    thinker = build_qwen3_omni_thinker_stub(hidden_size=hidden_size, max_tokens=8)
    if stale_tokens:
        for buf in thinker.deepstack_input_embeds:
            buf[:stale_tokens].fill_(7.0)
    thinker.deepstack_input_embeds_num_tokens = stale_tokens
    tpu_ar_model_runner.model = thinker

    orig_embed = thinker.embed_input_ids
    monkeypatch.setattr(
        TPUModelRunner,
        "_get_model_inputs",
        lambda self, ids, mm_in: (
            orig_embed(
                ids,
                multimodal_embeddings=(mm_in[0] if mm_in else None),
                is_multimodal=(mm_in[1] if mm_in else None),
            ),
            (None, SimpleNamespace(device=SimpleNamespace(type="tpu"))),
        )[1],
    )
    sync_calls: list[tuple[Any, bool]] = []
    monkeypatch.setattr(
        "vllm_torchtpu.omni.worker.tpu_ar_model_runner.synchronize_tensors",
        lambda tensors, wait=True: sync_calls.append((tensors, wait)),
    )

    if step_kind == "vision":
        ids = torch.tensor([1, 12, 12, 2], dtype=torch.long)
        mm_in: Any = (
            [set_mm_embedding_modality(torch.ones(2, hidden_size * 3), "image")],
            torch.tensor([False, True, True, False]),
        )
    else:
        ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        mm_in = None

    tpu_ar_model_runner._get_model_inputs(ids, mm_in)
    if step_kind == "text_only":
        assert thinker.deepstack_input_embeds_num_tokens == 0
        for buf in thinker.deepstack_input_embeds:
            assert torch.equal(buf[:4], torch.zeros(4, hidden_size))
    else:
        for buf in thinker.deepstack_input_embeds:
            assert torch.equal(buf[[0, 3]], torch.zeros(2, hidden_size))
    assert sync_calls == (
        [(thinker.deepstack_input_embeds, False)] if expect_ds_sync else []
    )


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
    assert [int(b["input_features"].shape[-1]) for b in audio_batches] == [
        800,
        1600,
        3200,
        6400,
        12800,
    ]
    assert [b["audio_feature_lengths"].tolist() for b in audio_batches] == [
        [800],
        [1600],
        [3200],
        [6400],
        [12800],
    ]

    image_batches = _expand_mm_dummy_batches(
        "image",
        {
            "pixel_values": torch.zeros(16384, 16),
            "image_grid_thw": torch.tensor([[1, 128, 128]], dtype=torch.int32),
        },
    )
    assert [int(b["pixel_values"].shape[0]) for b in image_batches] == [
        2048,
        4096,
        8192,
        16384,
    ]
    assert [b["image_grid_thw"].tolist() for b in image_batches] == [
        [[1, 64, 32]],
        [[1, 64, 64]],
        [[1, 64, 128]],
        [[1, 64, 256]],
    ]


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
    assert runner._omni_chunk_req_slice is None

    # Chunk 1: req-prefill [1:2] with 3 tokens
    runner._omni_chunk_req_slice = (1, 2)
    runner.query_start_loc_np = [0, 3]
    runner.forward_model(torch.zeros(4, dtype=torch.int32), torch.arange(4))
    assert runner._omni_chunk_req_slice is None

    # Warmup / _dummy_run with _omni_chunk_req_slice=None must not append outputs
    runner.forward_model(torch.zeros(4, dtype=torch.int32), torch.arange(4))
    assert len(runner._omni_chunk_outputs) == 2

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
        torch.device("tpu"),
        profiler_rank=0,
        profiler_world_size=1,
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
        torch.device("tpu"),
        profiler_rank=0,
        profiler_world_size=1,
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


def test_update_states_manages_intermediate_buffer_and_connector_cleanup(
    tpu_ar_model_runner: TPUARModelRunner,
) -> None:
    runner = tpu_ar_model_runner
    runner.model = SimpleNamespace(gpu_resident_buffer_keys=set())
    runner._omni_connector_initialized = True
    runner.cleanup_finished_request = MagicMock()
    runner.model_intermediate_buffer["req-old"] = {"embed": {"prefill": torch.ones(1)}}
    req_state_new = SimpleNamespace()
    runner.requests = {"req-new": req_state_new, "req-cached": SimpleNamespace()}

    sched_out = SimpleNamespace(
        finished_req_ids={"req-old"},
        scheduled_new_reqs=[
            SimpleNamespace(
                req_id="req-new",
                additional_information={"meta": {"num_processed_tokens": 4}},
                model_intermediate_buffer={"embed": {"prefill": torch.zeros(2, 4)}},
            )
        ],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["req-cached"],
            additional_information={"req-cached": {"meta": {"resumable": True}}},
        ),
    )
    with patch.object(OmniTPUModelRunner, "_update_states", return_value=True):
        assert runner._update_states(sched_out) is True

    assert "req-old" not in runner.model_intermediate_buffer
    runner.cleanup_finished_request.assert_called_once_with("req-old")
    assert runner.model_intermediate_buffer["req-new"]["meta"] == {
        "num_processed_tokens": 4
    }
    assert torch.equal(
        runner.model_intermediate_buffer["req-new"]["embed"]["prefill"],
        torch.zeros(2, 4),
    )
    assert (
        req_state_new.additional_information_cpu
        is runner.model_intermediate_buffer["req-new"]
    )
    assert runner.model_intermediate_buffer["req-cached"]["meta"] == {"resumable": True}


def test_tpu_ar_model_runner_talker_hooks_and_inter_stage_outputs() -> None:
    runner = TPUARModelRunner(
        SimpleNamespace(
            model_config=_stage_config(model_stage="talker", stage_id=1),
            speculative_config=None,
        ),
        torch.device("tpu"),
        profiler_rank=0,
        profiler_world_size=1,
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
