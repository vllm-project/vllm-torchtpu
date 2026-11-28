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

import pytest
import torch
from vllm.multimodal.utils import set_mm_embedding_modality
from vllm.sequence import IntermediateTensors
from vllm_omni.config import OmniModelConfig
from vllm_omni.model_executor.models.output_templates import OmniOutput

from tests_omni.omni_test_utils import build_qwen3_omni_thinker_stub
from vllm_torchtpu.omni.worker.tpu_ar_model_runner import (
    TPUARModelRunner,
    _slice_mm_embeds_by_row_indices,
)
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

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
        (SimpleNamespace(architectures=[]), (None, None)),
    ],
    indirect=["tpu_ar_model_runner"],
    ids=["omni_stage_config", "plain_vllm_config"],
)
def test_tpu_ar_model_runner_stage_metadata(
    tpu_ar_model_runner: TPUARModelRunner,
    expected: tuple[str | None, int | None],
) -> None:
    assert (tpu_ar_model_runner.model_stage, tpu_ar_model_runner.stage_id) == expected


@pytest.mark.parametrize(
    ("model_config", "speculative_config", "match"),
    [
        (
            _stage_config(
                custom_process_next_stage_input_func="pkg.mod.thinker2talker"
            ),
            None,
            "connectors",
        ),
        (
            _stage_config(),
            SimpleNamespace(method="eagle3"),
            "speculative_config",
        ),
    ],
    ids=["connectors", "speculative_config"],
)
def test_tpu_ar_model_runner_init_rejects_unsupported_config(
    model_config: Any,
    speculative_config: Any,
    match: str,
) -> None:
    vllm_config = SimpleNamespace(
        model_config=model_config,
        speculative_config=speculative_config,
    )
    with pytest.raises(NotImplementedError, match=match):
        TPUARModelRunner(
            vllm_config,
            torch.device("tpu"),
            profiler_rank=0,
            profiler_world_size=1,
        )


@pytest.mark.parametrize(
    ("has_preprocess", "has_postprocess", "error_match"),
    [
        (False, False, None),
        (True, False, "has_preprocess"),
        (False, True, "has_postprocess"),
        (True, True, "has_preprocess, has_postprocess"),
    ],
    ids=["thinker", "preprocess_only", "postprocess_only", "talker_both"],
)
def test_load_model_checks_per_step_hook_capabilities(
    monkeypatch: pytest.MonkeyPatch,
    tpu_ar_model_runner: TPUARModelRunner,
    has_preprocess: bool,
    has_postprocess: bool,
    error_match: str | None,
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
    if error_match is None:
        tpu_ar_model_runner.load_model()
        assert tpu_ar_model_runner.get_model() is model
    else:
        with pytest.raises(NotImplementedError, match=error_match):
            tpu_ar_model_runner.load_model()


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
