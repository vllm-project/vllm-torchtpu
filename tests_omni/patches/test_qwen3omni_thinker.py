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
from itertools import pairwise
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from transformers import Qwen3OmniMoeAudioEncoderConfig
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.attention import mm_encoder_attention
from vllm.v1.attention.ops.vit_attn_wrappers import torch_sdpa_wrapper
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeAudioEncoder,
    _get_feat_extract_output_lengths,
)
from vllm_omni.model_executor.models.registry import _OMNI_MODELS

from tests_omni.omni_test_utils import restore_class_attrs
from vllm_torchtpu.omni.patches.qwen3omni_thinker import (
    _AUDIO_CHUNK_BUCKETS,
    _CU_SEQLENS_BLOCK_SIZE,
    _QWEN3_OMNI_ARCHS,
    _build_bucket_inputs,
    _get_compiled_bucket_fn,
    _pad_cu_seqlens,
    _padded_audio_forward,
    _run_static_bucket_pass,
    maybe_patch_qwen3_omni_thinker,
)

pytestmark = pytest.mark.omni

THINKER_ARCHS = ["Qwen3OmniMoeForConditionalGeneration"]
STANDALONE_THINKER_ARCHS = ["Qwen3OmniMoeThinkerForConditionalGeneration"]
_PATCHED_ATTRS = ("forward", "_tpu_enforce_eager")

TINY_AUDIO_CONFIG = Qwen3OmniMoeAudioEncoderConfig(
    num_mel_bins=128,
    d_model=32,
    encoder_layers=2,
    encoder_attention_heads=4,
    encoder_ffn_dim=64,
    downsample_hidden_size=16,
    output_dim=32,
    n_window=50,
    n_window_infer=400,
)


@pytest.fixture(autouse=True)
def _isolate_thinker_patch_targets() -> Iterator[None]:
    """Snapshot and restore Qwen3OmniMoeAudioEncoder attributes across each test."""
    with restore_class_attrs(Qwen3OmniMoeAudioEncoder, _PATCHED_ATTRS):
        yield


@pytest.fixture
def patched_audio_encoder() -> None:
    """Apply the Thinker audio encoder patch for tests exercising forward()."""
    assert maybe_patch_qwen3_omni_thinker(THINKER_ARCHS) is True


@pytest.fixture(scope="module")
def tiny_audio_encoder(omni_ctx: None) -> Iterator[Qwen3OmniMoeAudioEncoder]:
    """Build a tiny real Qwen3OmniMoeAudioEncoder running native ops on CPU."""
    torch.manual_seed(0)
    enc = Qwen3OmniMoeAudioEncoder(TINY_AUDIO_CONFIG).eval()
    for module in enc.modules():
        if isinstance(module, CustomOp):
            module._forward_method = module.forward_native
    with patch.object(
        mm_encoder_attention, "vit_torch_sdpa_wrapper", torch_sdpa_wrapper
    ):
        yield enc


@pytest.mark.parametrize(
    "architectures",
    [
        None,
        [],
        ["LlamaForCausalLM"],
        ["Qwen3OmniMoeTalkerForConditionalGeneration"],
    ],
)
def test_non_omni_config_is_not_patched(architectures: list[str] | None) -> None:
    before = Qwen3OmniMoeAudioEncoder.forward
    assert maybe_patch_qwen3_omni_thinker(architectures) is False
    assert Qwen3OmniMoeAudioEncoder.forward is before


def test_patched_architectures_match_omni_registry() -> None:
    """Cross-check _QWEN3_OMNI_ARCHS against vllm-omni's model registry."""
    thinker_modules = {"qwen3_omni", "qwen3_omni_moe_thinker"}
    registry_thinkers = {
        arch
        for arch, (pkg, mod, _) in _OMNI_MODELS.items()
        if pkg == "qwen3_omni" and mod in thinker_modules
    }
    assert registry_thinkers == _QWEN3_OMNI_ARCHS
    assert "Qwen3OmniMoeTalkerForConditionalGeneration" not in _QWEN3_OMNI_ARCHS
    assert "Qwen3OmniMoeCode2Wav" not in _QWEN3_OMNI_ARCHS


@pytest.mark.parametrize("architectures", [THINKER_ARCHS, STANDALONE_THINKER_ARCHS])
def test_patch_application_is_idempotent(architectures: list[str]) -> None:
    cls = Qwen3OmniMoeAudioEncoder
    assert maybe_patch_qwen3_omni_thinker(architectures) is True
    patched_forward = cls.forward
    assert patched_forward is _padded_audio_forward
    assert maybe_patch_qwen3_omni_thinker(architectures) is False
    assert cls.forward is patched_forward
    assert getattr(cls, "_tpu_enforce_eager", None) is False
    assert maybe_patch_qwen3_omni_thinker(architectures, enforce_eager=True) is False
    assert getattr(cls, "_tpu_enforce_eager", None) is True


@torch.no_grad()
def test_forward_evaluates_feature_lens_on_cpu(
    patched_audio_encoder: None,
    tiny_audio_encoder: Qwen3OmniMoeAudioEncoder,
) -> None:
    seen_probe_devices: list[str] = []
    orig_cnn_lens = tiny_audio_encoder._get_cnn_output_lengths

    def _spy_cnn_lens(input_lengths: torch.Tensor) -> torch.Tensor:
        seen_probe_devices.append(input_lengths.device.type)
        return orig_cnn_lens(input_lengths)

    lens_device = "tpu" if torch.accelerator.is_available() else "cpu"
    lens = torch.tensor([800], dtype=torch.long, device=lens_device)
    with patch.object(
        tiny_audio_encoder, "_get_cnn_output_lengths", side_effect=_spy_cnn_lens
    ):
        out = tiny_audio_encoder(torch.zeros(TINY_AUDIO_CONFIG.num_mel_bins, 800), lens)
    assert seen_probe_devices == ["cpu"]
    assert out.shape == (104, TINY_AUDIO_CONFIG.output_dim)


@pytest.mark.parametrize("clip_lens", [[], [0], [0, 0]])
@torch.no_grad()
def test_empty_audio_inputs_derive_output_dim_from_proj2(
    patched_audio_encoder: None,
    tiny_audio_encoder: Qwen3OmniMoeAudioEncoder,
    clip_lens: list[int],
) -> None:
    assert not hasattr(tiny_audio_encoder, "output_dim")
    lens = torch.tensor(clip_lens, dtype=torch.long)
    out = tiny_audio_encoder(torch.zeros(TINY_AUDIO_CONFIG.num_mel_bins, 0), lens)
    assert out.shape == (0, TINY_AUDIO_CONFIG.output_dim)


@pytest.mark.parametrize(
    ("n_window_infer", "match"),
    [
        (0, "positive multiple"),
        (150, "positive multiple"),
        (100, "exceeding _CU_SEQLENS_BLOCK_SIZE"),
        (300, "divisible"),
    ],
)
@torch.no_grad()
def test_invalid_attention_window_config_raises_value_error(
    patched_audio_encoder: None,
    tiny_audio_encoder: Qwen3OmniMoeAudioEncoder,
    n_window_infer: int,
    match: str,
) -> None:
    lens = torch.tensor([800], dtype=torch.long)
    with (
        patch.object(tiny_audio_encoder, "n_window_infer", n_window_infer),
        pytest.raises(ValueError, match=match),
    ):
        tiny_audio_encoder(torch.zeros(TINY_AUDIO_CONFIG.num_mel_bins, 800), lens)


@pytest.mark.parametrize("bad_lens", [[-10], [800, -1]])
@torch.no_grad()
def test_negative_feature_lens_raises_value_error(
    patched_audio_encoder: None,
    tiny_audio_encoder: Qwen3OmniMoeAudioEncoder,
    bad_lens: list[int],
) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        tiny_audio_encoder(
            torch.zeros(TINY_AUDIO_CONFIG.num_mel_bins, 0),
            torch.tensor(bad_lens, dtype=torch.long),
        )


@pytest.mark.parametrize(
    "bad_shape",
    [(TINY_AUDIO_CONFIG.num_mel_bins,), (TINY_AUDIO_CONFIG.num_mel_bins, 100)],
)
@torch.no_grad()
def test_malformed_input_features_raises_value_error(
    patched_audio_encoder: None,
    tiny_audio_encoder: Qwen3OmniMoeAudioEncoder,
    bad_shape: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="Expected 2D input_features"):
        tiny_audio_encoder(
            torch.zeros(bad_shape),
            torch.tensor([250], dtype=torch.long),
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
            [12750],
            [128],
            [[0, 52, 104, 156, 208]],
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
@torch.no_grad()
def test_audio_encoder_static_bucketing_and_packing(
    patched_audio_encoder: None,
    tiny_audio_encoder: Qwen3OmniMoeAudioEncoder,
    clip_lens: list[int],
    expected_buckets: list[int],
    expected_cu_prefixes: list[list[int]],
) -> None:
    """Verify bucketing, packing, >128/cu_seqlens splits, and valid cu_seqlens tails."""
    seen_conv_batch_sizes: list[int] = []
    seen_layer_seq_lens: list[int] = []
    seen_cu_seqlens: list[list[int]] = []

    def _record_conv(_mod: nn.Module, args: tuple[Any, ...]) -> None:
        seen_conv_batch_sizes.append(int(args[0].shape[0]))

    def _record_layer(_mod: nn.Module, args: tuple[Any, ...]) -> None:
        seen_layer_seq_lens.append(int(args[0].shape[0]))
        seen_cu_seqlens.append(args[1].tolist())

    handles = [tiny_audio_encoder.conv2d1.register_forward_pre_hook(_record_conv)]
    handles.extend(
        layer.register_forward_pre_hook(_record_layer)
        for layer in tiny_audio_encoder.layers
    )
    try:
        with patch.object(
            nn.Module,
            "cpu",
            side_effect=AssertionError("Unexpected Module.cpu() host mutation"),
        ):
            # Strided slice exercises the non-contiguous input_features path too.
            total_frames = sum(clip_lens)
            feats = torch.zeros(TINY_AUDIO_CONFIG.num_mel_bins, total_frames * 2)[
                :, ::2
            ]
            assert not feats.is_contiguous()
            lens = torch.tensor(clip_lens, dtype=torch.long)
            aftercnn = _get_feat_extract_output_lengths(lens)
            out = tiny_audio_encoder(feats, lens, aftercnn)

            assert out.shape == (
                int(aftercnn.sum().item()),
                TINY_AUDIO_CONFIG.output_dim,
            )
            assert seen_conv_batch_sizes == expected_buckets
            num_layers = len(tiny_audio_encoder.layers)
            expected_seq_lens = [
                b * 13 for b in expected_buckets for _ in range(num_layers)
            ]
            assert seen_layer_seq_lens == expected_seq_lens
            pass_cu_seqlens = seen_cu_seqlens[::num_layers]
            for layer_idx in range(1, num_layers):
                assert pass_cu_seqlens == seen_cu_seqlens[layer_idx::num_layers]
            assert len(pass_cu_seqlens) == len(expected_cu_prefixes)
            for bucket, cu, expected_prefix in zip(
                expected_buckets, pass_cu_seqlens, expected_cu_prefixes, strict=True
            ):
                padded_total = bucket * 13
                assert len(cu) == _CU_SEQLENS_BLOCK_SIZE
                assert cu[: len(expected_prefix)] == expected_prefix
                # Trailing padding uses padded_total so segment lengths stay
                # non-negative and sum to padded_total for CPU/native SDPA.
                assert cu[-1] == padded_total
                diffs = [b_end - b_start for b_start, b_end in pairwise(cu)]
                assert all(d >= 0 for d in diffs)
                assert sum(diffs) == padded_total
    finally:
        for handle in handles:
            handle.remove()


def test_pad_cu_seqlens_rejects_overflow() -> None:
    with pytest.raises(ValueError, match="exceeds fixed _CU_SEQLENS_BLOCK_SIZE"):
        _pad_cu_seqlens(list(range(_CU_SEQLENS_BLOCK_SIZE)), 1664, torch.device("cpu"))


def test_compiled_bucket_fn_caches_per_encoder_and_bucket(
    tiny_audio_encoder: Qwen3OmniMoeAudioEncoder,
) -> None:
    """Verify distinct compiled bucket entries per (encoder instance, bucket)."""
    enc_b = Qwen3OmniMoeAudioEncoder(TINY_AUDIO_CONFIG).eval()
    try:
        fn_8 = _get_compiled_bucket_fn(tiny_audio_encoder, _AUDIO_CHUNK_BUCKETS[0])
        fn_16 = _get_compiled_bucket_fn(tiny_audio_encoder, _AUDIO_CHUNK_BUCKETS[1])
        assert fn_8 is _get_compiled_bucket_fn(
            tiny_audio_encoder, _AUDIO_CHUNK_BUCKETS[0]
        )
        assert fn_8 is not fn_16
        assert fn_8 is not _get_compiled_bucket_fn(enc_b, _AUDIO_CHUNK_BUCKETS[0])
        compiled_buckets = tiny_audio_encoder.__dict__.get("_tpu_compiled_buckets")
        assert isinstance(compiled_buckets, dict)
        assert set(compiled_buckets) == {
            _AUDIO_CHUNK_BUCKETS[0],
            _AUDIO_CHUNK_BUCKETS[1],
        }
    finally:
        if hasattr(tiny_audio_encoder, "_tpu_compiled_buckets"):
            delattr(tiny_audio_encoder, "_tpu_compiled_buckets")


@pytest.mark.parametrize("enforce_eager", [False, True])
@torch.no_grad()
def test_enforce_eager_controls_tpu_bucket_compilation(
    tiny_audio_encoder: Qwen3OmniMoeAudioEncoder,
    enforce_eager: bool,
) -> None:
    """Verify TPU path compiles unless enforce_eager is set, and always flushes."""
    assert maybe_patch_qwen3_omni_thinker(THINKER_ARCHS, enforce_eager=enforce_eager)
    compiled_buckets: list[int] = []
    sync_waits: list[bool] = []
    real_conv2d1 = tiny_audio_encoder.conv2d1

    class _TpuDeviceConvProxy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = SimpleNamespace(
                device=SimpleNamespace(type="tpu"),
                dtype=real_conv2d1.weight.dtype,
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return real_conv2d1(x)

    def _fake_get_compiled(enc: Any, bucket_chunks: int) -> Any:
        compiled_buckets.append(bucket_chunks)
        return lambda x, cu: _run_static_bucket_pass(enc, x, cu)

    def _cpu_build_bucket(
        feats_u8_np: Any,
        dtype: torch.dtype,
        group: Any,
        window_size: int,
        tokens_per_chunk: int,
        window_aftercnn: int,
        _device: Any,
    ) -> Any:
        return _build_bucket_inputs(
            feats_u8_np,
            dtype,
            group,
            window_size,
            tokens_per_chunk,
            window_aftercnn,
            torch.device("cpu"),
        )

    patch_mod = "vllm_torchtpu.omni.patches.qwen3omni_thinker"
    with (
        patch.object(tiny_audio_encoder, "conv2d1", _TpuDeviceConvProxy()),
        patch(f"{patch_mod}._build_bucket_inputs", side_effect=_cpu_build_bucket),
        patch(
            f"{patch_mod}._get_compiled_bucket_fn",
            side_effect=_fake_get_compiled,
        ),
        patch(
            f"{patch_mod}.synchronize_tensors",
            side_effect=lambda _t, wait=True: sync_waits.append(wait),
        ),
    ):
        out = tiny_audio_encoder(
            torch.zeros(TINY_AUDIO_CONFIG.num_mel_bins, 250),
            torch.tensor([250], dtype=torch.long),
        )
    assert out.shape == (33, TINY_AUDIO_CONFIG.output_dim)
    assert compiled_buckets == ([] if enforce_eager else [8])
    assert sync_waits == [False]
