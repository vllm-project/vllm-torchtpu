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
"""CPU unit tests for the Qwen3-Omni Thinker audio encoder patch."""

from __future__ import annotations

import types
from collections.abc import Iterator
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeAudioEncoder,
    _get_feat_extract_output_lengths,
)

from vllm_torchtpu.omni.patches.qwen3omni_thinker import (
    _AUDIO_CHUNK_BUCKETS,
    _AUDIO_CHUNK_LENGTHS_PATCHED,
    _CU_SEQLENS_BLOCK_SIZE,
    _get_compiled_bucket_fn,
    maybe_patch_qwen3_omni_thinker,
)

pytestmark = pytest.mark.omni

THINKER_ARCHS = ["Qwen3OmniMoeForConditionalGeneration"]
_PATCHED_ATTRS = (
    "forward",
    "prepare_encoder_metadata",
    "forward_with_encoder_metadata",
    "forward_chunks",
    "_tpu_orig_forward",
)


@pytest.fixture(autouse=True)
def _isolate_thinker_patch_targets() -> Iterator[None]:
    """Snapshot and restore Qwen3OmniMoeAudioEncoder attributes across each test."""
    cls = Qwen3OmniMoeAudioEncoder
    saved = {attr: cls.__dict__.get(attr) for attr in _PATCHED_ATTRS}
    try:
        yield
    finally:
        for attr, val in saved.items():
            if val is None:
                if attr in cls.__dict__:
                    delattr(cls, attr)
            else:
                setattr(cls, attr, val)


class _DummyPosEmb(nn.Module):
    def __init__(self, max_len: int = 64, d_model: int = 8) -> None:
        super().__init__()
        self.register_buffer("positional_embedding", torch.zeros(max_len, d_model))


class _DummyReplicatedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.input_size = in_features
        self.output_size = out_features
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return self.linear(x), None


class _TrackedTensor(torch.Tensor):
    cpu_called: bool = False

    def cpu(self, *args: object, **kwargs: object) -> torch.Tensor:
        _TrackedTensor.cpu_called = True
        return super().cpu(*args, **kwargs)


def _make_dummy_audio_encoder(
    seen_conv_batch_sizes: list[int] | None = None,
    seen_layer_seq_lens: list[int] | None = None,
    seen_cu_seqlens: list[list[int]] | None = None,
    n_window_infer: int = 400,
) -> nn.Module:
    class _Layer(nn.Module):
        def forward(
            self,
            hidden_states: torch.Tensor,
            cu_seqlens: torch.Tensor,
            max_seqlen: torch.Tensor | None = None,
        ) -> torch.Tensor:
            del max_seqlen
            if seen_layer_seq_lens is not None:
                seen_layer_seq_lens.append(int(hidden_states.shape[0]))
            if seen_cu_seqlens is not None:
                seen_cu_seqlens.append(cu_seqlens.tolist())
            return hidden_states

    class _Conv1(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if seen_conv_batch_sizes is not None:
                seen_conv_batch_sizes.append(int(x.shape[0]))
            _, _, f, _ = x.shape
            return x[:, :, : max(1, f // 2), :13]

    class _DummyEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.n_window = 50
            self.n_window_infer = n_window_infer
            self.conv_chunksize = 500
            self.conv2d1 = _Conv1()
            self.conv2d2 = nn.Identity()
            self.conv2d3 = nn.Identity()
            self.conv_out = _DummyReplicatedLinear(8, 8)
            self.positional_embedding = _DummyPosEmb(max_len=64, d_model=8)
            self.layers = nn.ModuleList([_Layer(), _Layer()])
            self.ln_post = nn.Identity()
            self.proj1 = _DummyReplicatedLinear(8, 8)
            self.act = nn.Identity()
            self.proj2 = _DummyReplicatedLinear(8, 8)

        def _get_cnn_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
            return _get_feat_extract_output_lengths(input_lengths)

    return _DummyEncoder()


@pytest.mark.parametrize(
    "architectures",
    [
        None,
        [],
        ["LlamaForCausalLM"],
        ["Qwen3OmniMoeThinkerForConditionalGeneration"],
    ],
)
def test_non_omni_config_is_not_patched(architectures: list[str] | None) -> None:
    before = Qwen3OmniMoeAudioEncoder.forward
    assert maybe_patch_qwen3_omni_thinker(architectures) is False
    assert Qwen3OmniMoeAudioEncoder.forward is before


def test_patch_application_is_idempotent() -> None:
    cls = Qwen3OmniMoeAudioEncoder
    assert maybe_patch_qwen3_omni_thinker(THINKER_ARCHS) is True
    patched_forward = cls.forward
    assert getattr(patched_forward, _AUDIO_CHUNK_LENGTHS_PATCHED, False)
    assert hasattr(cls, "_tpu_orig_forward")
    assert maybe_patch_qwen3_omni_thinker(THINKER_ARCHS) is False
    assert cls.forward is patched_forward


def test_forward_moves_feature_lens_to_cpu() -> None:
    cls = Qwen3OmniMoeAudioEncoder
    assert maybe_patch_qwen3_omni_thinker(THINKER_ARCHS) is True
    _TrackedTensor.cpu_called = False
    enc = _make_dummy_audio_encoder()
    lens = torch.tensor([800]).as_subclass(_TrackedTensor)
    out = cls.forward(enc, torch.zeros(16, 800), lens, torch.tensor([104]))
    assert _TrackedTensor.cpu_called is True
    assert out.shape == (104, 8)


def test_metadata_api_computes_aftercnn_lens_and_encodes() -> None:
    cls = Qwen3OmniMoeAudioEncoder
    assert maybe_patch_qwen3_omni_thinker(THINKER_ARCHS) is True
    enc = _make_dummy_audio_encoder()
    metadata = cls.prepare_encoder_metadata(enc, [800, 250])
    assert metadata["aftercnn_lens"] == [104, 33]
    out_meta = cls.forward_with_encoder_metadata(enc, torch.zeros(16, 1050), metadata)
    assert out_meta.shape == (137, 8)


@pytest.mark.parametrize("clip_lens", [[], [0], [0, 0]])
def test_empty_audio_inputs_derive_output_dim_from_proj2(
    clip_lens: list[int],
) -> None:
    cls = Qwen3OmniMoeAudioEncoder
    assert maybe_patch_qwen3_omni_thinker(THINKER_ARCHS) is True
    enc = _make_dummy_audio_encoder()
    assert not hasattr(enc, "output_dim")
    metadata = cls.prepare_encoder_metadata(enc, clip_lens)
    out = cls.forward_with_encoder_metadata(enc, torch.zeros(16, 0), metadata)
    assert out.shape == (0, 8)


def test_forward_chunks_raises_not_implemented() -> None:
    cls = Qwen3OmniMoeAudioEncoder
    assert maybe_patch_qwen3_omni_thinker(THINKER_ARCHS) is True
    enc = _make_dummy_audio_encoder()
    metadata = cls.prepare_encoder_metadata(enc, [800])
    with pytest.raises(NotImplementedError, match="forward_chunks"):
        cls.forward_chunks(enc, torch.zeros(8, 16, 100), metadata)


@pytest.mark.parametrize("n_window_infer", [300, 600, 1200])
def test_unaligned_attention_window_ratio_raises(n_window_infer: int) -> None:
    cls = Qwen3OmniMoeAudioEncoder
    assert maybe_patch_qwen3_omni_thinker(THINKER_ARCHS) is True
    enc = _make_dummy_audio_encoder(n_window_infer=n_window_infer)
    lens = torch.tensor([800], dtype=torch.long)
    with pytest.raises(AssertionError, match="divisible"):
        cls.forward(enc, torch.zeros(16, 800), lens, torch.tensor([104]))


@pytest.mark.parametrize(
    ("clip_lens", "expected_buckets", "expected_cu_prefixes"),
    [
        ([250], [8], [[0, 33, 104, 105]]),
        ([250, 800], [16], [[0, 33, 39, 91, 143, 208, 209]]),
        (
            [2000, 2000],
            [64],
            [[0, 52, 104, 156, 208, 260, 312, 364, 416, 468, 520, 832, 833]],
        ),
        ([6500], [128], [[0, 52, 104, 156, 208]]),
        (
            [12801, 200],
            [128, 8],
            [[0, 52, 104, 156, 208], [0, 1, 13, 39, 104, 105]],
        ),
    ],
)
def test_audio_encoder_static_bucketing_and_packing(
    clip_lens: list[int],
    expected_buckets: list[int],
    expected_cu_prefixes: list[list[int]],
) -> None:
    """Verify power-of-2 bucketing, multi-clip packing, and >128 split layout."""
    assert maybe_patch_qwen3_omni_thinker(THINKER_ARCHS) is True
    cls = Qwen3OmniMoeAudioEncoder

    seen_conv_batch_sizes: list[int] = []
    seen_layer_seq_lens: list[int] = []
    seen_cu_seqlens: list[list[int]] = []
    enc = _make_dummy_audio_encoder(
        seen_conv_batch_sizes=seen_conv_batch_sizes,
        seen_layer_seq_lens=seen_layer_seq_lens,
        seen_cu_seqlens=seen_cu_seqlens,
    )

    with patch.object(
        nn.Module,
        "cpu",
        side_effect=AssertionError("Unexpected Module.cpu() host mutation"),
    ):
        feats = torch.zeros(16, sum(clip_lens))
        lens = torch.tensor(clip_lens, dtype=torch.long)
        aftercnn = _get_feat_extract_output_lengths(lens)
        out = cls.forward(enc, feats, lens, aftercnn)

        assert out.shape == (int(aftercnn.sum().item()), 8)
        assert seen_conv_batch_sizes == expected_buckets
        expected_seq_lens = [b * 13 for b in expected_buckets for _ in range(2)]
        assert seen_layer_seq_lens == expected_seq_lens
        for cu in seen_cu_seqlens:
            assert len(cu) == _CU_SEQLENS_BLOCK_SIZE
        pass_cu_seqlens = seen_cu_seqlens[::2]
        assert pass_cu_seqlens == seen_cu_seqlens[1::2]
        assert len(pass_cu_seqlens) == len(expected_cu_prefixes)
        for cu, expected_prefix in zip(
            pass_cu_seqlens, expected_cu_prefixes, strict=True
        ):
            assert cu[: len(expected_prefix)] == expected_prefix


def test_compiled_bucket_cache_uses_distinct_code_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify per-bucket compile caching and distinct CodeTypes across encoders."""
    seen_calls: list[tuple[types.FunctionType, dict[str, object]]] = []

    def _fake_compile(fn: types.FunctionType, **kwargs: object) -> types.FunctionType:
        seen_calls.append((fn, kwargs))
        return fn

    monkeypatch.setattr(torch, "compile", _fake_compile)
    enc_a = _make_dummy_audio_encoder()
    enc_b = _make_dummy_audio_encoder()

    first_bucket = _AUDIO_CHUNK_BUCKETS[0]
    assert _get_compiled_bucket_fn(enc_a, first_bucket) is _get_compiled_bucket_fn(
        enc_a, first_bucket
    )
    assert len(seen_calls) == 1

    for b in _AUDIO_CHUNK_BUCKETS:
        _get_compiled_bucket_fn(enc_a, b)
        _get_compiled_bucket_fn(enc_b, b)

    expected_total = 2 * len(_AUDIO_CHUNK_BUCKETS)
    assert len(seen_calls) == expected_total
    assert len({fn.__code__ for fn, _ in seen_calls}) == expected_total
    for _, kwargs in seen_calls:
        assert kwargs == {"backend": "tpu", "fullgraph": True, "dynamic": False}
