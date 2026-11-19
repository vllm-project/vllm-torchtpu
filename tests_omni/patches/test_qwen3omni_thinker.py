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

from collections.abc import Iterator
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeAudioEncoder,
    _get_feat_extract_output_lengths,
)

from tests_omni.omni_test_utils import restore_class_attrs
from vllm_torchtpu.omni.patches.qwen3omni_thinker import (
    _AUDIO_CHUNK_BUCKETS,
    _AUDIO_CHUNK_LENGTHS_PATCHED,
    _CU_SEQLENS_BLOCK_SIZE,
    _get_compiled_bucket_fn,
    _pad_cu_seqlens,
    _run_static_bucket_pass,
    maybe_patch_qwen3_omni_thinker,
)

pytestmark = pytest.mark.omni

THINKER_ARCHS = ["Qwen3OmniMoeForConditionalGeneration"]
_PATCHED_ATTRS = ("forward", "_tpu_orig_forward")


@pytest.fixture(autouse=True)
def _isolate_thinker_patch_targets() -> Iterator[None]:
    """Snapshot and restore Qwen3OmniMoeAudioEncoder attributes across each test."""
    with restore_class_attrs(Qwen3OmniMoeAudioEncoder, _PATCHED_ATTRS):
        yield


@pytest.fixture
def patched_audio_encoder() -> type[Qwen3OmniMoeAudioEncoder]:
    """Apply the Thinker audio encoder patch for tests exercising forward()."""
    assert maybe_patch_qwen3_omni_thinker(THINKER_ARCHS) is True
    return Qwen3OmniMoeAudioEncoder


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


def test_forward_moves_feature_lens_to_cpu(
    patched_audio_encoder: type[Qwen3OmniMoeAudioEncoder],
) -> None:
    _TrackedTensor.cpu_called = False
    enc = _make_dummy_audio_encoder()
    lens = torch.tensor([800]).as_subclass(_TrackedTensor)
    out = patched_audio_encoder.forward(
        enc, torch.zeros(16, 800), lens, torch.tensor([104])
    )
    assert _TrackedTensor.cpu_called is True
    assert out.shape == (104, 8)


@pytest.mark.parametrize("clip_lens", [[], [0], [0, 0]])
def test_empty_audio_inputs_derive_output_dim_from_proj2(
    patched_audio_encoder: type[Qwen3OmniMoeAudioEncoder],
    clip_lens: list[int],
) -> None:
    enc = _make_dummy_audio_encoder()
    assert not hasattr(enc, "output_dim")
    lens = torch.tensor(clip_lens, dtype=torch.long)
    out = patched_audio_encoder.forward(enc, torch.zeros(16, 0), lens)
    assert out.shape == (0, 8)


@pytest.mark.parametrize(
    ("n_window_infer", "match"),
    [
        (0, "positive multiple"),
        (150, "positive multiple"),
        (100, "exceeding _CU_SEQLENS_BLOCK_SIZE"),
        (300, "divisible"),
    ],
)
def test_invalid_attention_window_config_raises_value_error(
    patched_audio_encoder: type[Qwen3OmniMoeAudioEncoder],
    n_window_infer: int,
    match: str,
) -> None:
    enc = _make_dummy_audio_encoder(n_window_infer=n_window_infer)
    lens = torch.tensor([800], dtype=torch.long)
    with pytest.raises(ValueError, match=match):
        patched_audio_encoder.forward(
            enc, torch.zeros(16, 800), lens, torch.tensor([104])
        )


@pytest.mark.parametrize("bad_lens", [[-10], [800, -1]])
def test_negative_feature_lens_raises_value_error(
    patched_audio_encoder: type[Qwen3OmniMoeAudioEncoder],
    bad_lens: list[int],
) -> None:
    enc = _make_dummy_audio_encoder()
    with pytest.raises(ValueError, match="non-negative"):
        patched_audio_encoder.forward(
            enc, torch.zeros(16, 0), torch.tensor(bad_lens, dtype=torch.long)
        )


@pytest.mark.parametrize(
    ("clip_lens", "expected_buckets", "expected_cu_prefixes"),
    [
        ([250], [8], [[0, 33, 104]]),
        ([0, 250], [8], [[0, 33, 104]]),
        ([250, 800], [16], [[0, 33, 39, 91, 143, 208]]),
        (
            [2000, 2000],
            [64],
            [[0, 52, 104, 156, 208, 260, 312, 364, 416, 468, 520, 832]],
        ),
        (
            [12801, 200],
            [128, 8],
            [[0, 52, 104, 156, 208], [0, 1, 13, 39, 104]],
        ),
        # 65 one-window clips fit in <=128 chunks but exceed 128 cu_seqlens
        # entries (2 + 65*2 = 132 > 128), forcing a cu_seqlens-budget group split.
        ([50] * 65, [64, 8], [[0, 7, 13, 20, 26], [0, 7, 13, 20, 104]]),
    ],
)
def test_audio_encoder_static_bucketing_and_packing(
    patched_audio_encoder: type[Qwen3OmniMoeAudioEncoder],
    clip_lens: list[int],
    expected_buckets: list[int],
    expected_cu_prefixes: list[list[int]],
) -> None:
    """Verify bucketing, packing, >128/cu_seqlens splits, and valid cu_seqlens tails."""
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
        # Strided slice exercises the non-contiguous input_features path too.
        total_frames = sum(clip_lens)
        feats = torch.zeros(16, total_frames * 2)[:, ::2]
        assert not feats.is_contiguous()
        lens = torch.tensor(clip_lens, dtype=torch.long)
        aftercnn = _get_feat_extract_output_lengths(lens)
        out = patched_audio_encoder.forward(enc, feats, lens, aftercnn)

        assert out.shape == (int(aftercnn.sum().item()), 8)
        assert seen_conv_batch_sizes == expected_buckets
        expected_seq_lens = [b * 13 for b in expected_buckets for _ in range(2)]
        assert seen_layer_seq_lens == expected_seq_lens
        pass_cu_seqlens = seen_cu_seqlens[::2]
        assert pass_cu_seqlens == seen_cu_seqlens[1::2]
        assert len(pass_cu_seqlens) == len(expected_cu_prefixes)
        for bucket, cu, expected_prefix in zip(
            expected_buckets, pass_cu_seqlens, expected_cu_prefixes, strict=True
        ):
            padded_total = bucket * 13
            assert len(cu) == _CU_SEQLENS_BLOCK_SIZE
            assert cu[: len(expected_prefix)] == expected_prefix
            # Trailing padding uses padded_total so segment lengths stay non-negative
            # and sum exactly to padded_total for CPU/native SDPA splits.
            assert cu[-1] == padded_total
            diffs = [b_end - b_start for b_start, b_end in zip(cu, cu[1:])]
            assert all(d >= 0 for d in diffs)
            assert sum(diffs) == padded_total


def test_pad_cu_seqlens_rejects_overflow() -> None:
    with pytest.raises(ValueError, match="exceeds fixed _CU_SEQLENS_BLOCK_SIZE"):
        _pad_cu_seqlens(list(range(_CU_SEQLENS_BLOCK_SIZE)), 1664, torch.device("cpu"))


def test_static_bucket_pass_traces_fullgraph_and_caches_per_bucket() -> None:
    """Verify fullgraph torch.compile tracing (no mocks) and distinct bucket caches."""
    enc = _make_dummy_audio_encoder()
    x = torch.zeros(8, 1, 16, 100)
    cu = _pad_cu_seqlens([0, 33], 104, torch.device("cpu"))

    eager_out = _run_static_bucket_pass(enc, x, cu)
    compiled = torch.compile(
        lambda x_in, cu_in: _run_static_bucket_pass(enc, x_in, cu_in),
        backend="eager",
        fullgraph=True,
        dynamic=False,
    )
    traced_out = compiled(x, cu)
    assert torch.allclose(traced_out, eager_out)

    enc_b = _make_dummy_audio_encoder()
    fn_a = _get_compiled_bucket_fn(enc, _AUDIO_CHUNK_BUCKETS[0])
    assert fn_a is _get_compiled_bucket_fn(enc, _AUDIO_CHUNK_BUCKETS[0])
    fn_b = _get_compiled_bucket_fn(enc_b, _AUDIO_CHUNK_BUCKETS[0])
    assert fn_a is not fn_b
    assert set(enc._tpu_compiled_buckets) == {_AUDIO_CHUNK_BUCKETS[0]}
