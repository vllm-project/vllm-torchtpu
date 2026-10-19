# SPDX-License-Identifier: Apache-2.0
"""Geometry, normalization, and weight-coverage contracts for Muse perception."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from vllm_torchtpu.models.vllm.muse_spark.audio import DepthwiseCausalConv
from vllm_torchtpu.models.vllm.muse_spark.layers import MuseRMSNorm
from vllm_torchtpu.models.vllm.muse_spark.multimodal import (
    MuseSparkForConditionalGeneration,
)
from vllm_torchtpu.models.vllm.muse_spark.processor import (
    PreprocessConfig,
    compute_video_clip_metadata,
    preprocess_image,
    preprocess_video,
)


def test_image_preprocessing_composites_alpha_and_preserves_rectangular_grid():
    rgba = np.zeros((56, 84, 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[:, :42, 3] = 255
    image = Image.fromarray(rgba)
    pixels, count = preprocess_image(image, PreprocessConfig(max_num_image_tokens=6))
    assert pixels.shape == (3, 56, 84) and count == 6
    torch.testing.assert_close(pixels[:, 0, 0], torch.tensor([1.0, -1.0, -1.0]))
    torch.testing.assert_close(pixels[:, 0, -1], torch.ones(3))


def test_video_alignment_keeps_frame_order_and_clip_timestamps():
    frame_count, aligned = 5, 4
    frames = np.stack(
        [np.full((28, 56, 3), 30 * i, dtype=np.uint8) for i in range(frame_count)]
    )
    cfg = PreprocessConfig(max_num_tokens_per_frame=2)
    pixels, counts, timestamps = preprocess_video(
        frames, [float(i) for i in range(frame_count)], cfg
    )
    assert pixels.shape == (aligned, 3, 28, 56)
    assert counts == [2] * aligned
    assert timestamps == [0.0, 1.0, 2.0, 3.0]
    # Each clip combines two adjacent frames into the same two spatial tokens.
    metadata = compute_video_clip_metadata(pixels, timestamps, cfg)
    assert metadata.num_clips == aligned // 2
    assert metadata.patches_per_clip == 2
    assert metadata.clip_timestamps == timestamps[::2]


def test_audio_causal_convolution_streaming_matches_prefill():
    conv = DepthwiseCausalConv(3, 7)
    x = torch.randn(1, 13, 3, generator=torch.Generator().manual_seed(8))
    whole, _ = conv(x)
    first, cache = conv(x[:, :5], torch.zeros(1, 3, 6))
    rest, _ = conv(x[:, 5:], cache)
    torch.testing.assert_close(torch.cat((first, rest), dim=1), whole)


def _empty_omni():
    model = MuseSparkForConditionalGeneration.__new__(MuseSparkForConditionalGeneration)
    nn.Module.__init__(model)
    model.model, model.lm_head = nn.Module(), nn.Module()
    model.model.language_model = nn.Module()
    model.tp_rank, model.tp_size = 0, 1
    model.ep_rank, model.ep_size, model.use_ep = 0, 1, False
    return model


def test_embedding_merge_normalizes_text_before_replacement():
    model = _empty_omni()
    model.model.language_model.embed_tokens = nn.Embedding(8, 4, dtype=torch.bfloat16)
    model.model.language_model.embed_tokens.weight.data.copy_(
        torch.arange(32).reshape(8, 4)
    )
    model.model.language_model.tok_embeddings_norm = MuseRMSNorm(
        4, 1e-6, has_weight=False
    )
    model.model.language_model.residual_dtype = torch.float32
    ids = torch.tensor([1, 2, 3, 4, 0])
    mask = torch.tensor([False, True, False, True, False])
    media = torch.tensor(
        [[12.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]], dtype=torch.bfloat16
    )
    actual = model.embed_input_ids(ids, [media[:1], media[1:]], is_multimodal=mask)
    expected = model.model.language_model.embed_tokens(ids).float()
    expected = expected / (expected.square().mean(-1, keepdim=True) + 1e-6).sqrt()
    # The TPU runner passes an empty list for text-only prefill/decode chunks.
    text_only = model.embed_input_ids(ids, [], is_multimodal=torch.zeros_like(mask))
    torch.testing.assert_close(text_only, expected)
    expected[mask] = media.float()
    torch.testing.assert_close(actual, expected)
    assert actual.dtype == torch.float32


def test_modality_loader_combines_native_qkv_and_loads_audio():
    model = _empty_omni()
    model.model.vision_tower = nn.Module()
    model.model.vision_tower.attn = nn.Module()
    qkv = model.model.vision_tower.attn.qkv_proj = nn.Linear(2, 6, bias=False)
    model.model.audio_tower = nn.Module()
    model.model.audio_tower.proj = nn.Linear(2, 3, bias=False)

    def load_qkv(param, tensor, shard):
        offset = {"q": 0, "k": 2, "v": 4}[shard]
        param.data[offset : offset + 2].copy_(tensor)

    qkv.weight.weight_loader = load_qkv
    weights = [
        (f"model.vision_tower.attn.{part}_proj.weight", torch.full((2, 2), float(i)))
        for i, part in enumerate(("q", "k", "v"))
    ]
    weights.append(("model.audio_tower.proj.weight", torch.full((3, 2), 4.0)))
    loaded = model.load_weights(weights)
    assert loaded == {
        "model.vision_tower.attn.qkv_proj.weight",
        "model.audio_tower.proj.weight",
    }
    torch.testing.assert_close(
        qkv.weight[:, 0], torch.tensor([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
    )
    torch.testing.assert_close(
        model.model.audio_tower.proj.weight, torch.full((3, 2), 4.0)
    )


class ReferenceAttention(nn.Module):
    def __init__(self, num_heads, head_size, scale, **kwargs):
        super().__init__()
        self.scale = scale

    def forward(self, q, k, v, cu_seqlens, **kwargs):
        positions = torch.arange(q.shape[1])
        segments = (positions[None] >= cu_seqlens[:, None]).sum(0)
        mask = segments[:, None] == segments[None, :]
        scores = q.transpose(1, 2) @ k.transpose(1, 2).transpose(-1, -2)
        weights = (scores * self.scale).masked_fill(~mask, -torch.inf).softmax(-1)
        return (weights @ v.transpose(1, 2)).transpose(1, 2)


@pytest.fixture
def cpu_vllm_config_context():
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig("cpu"))):
        yield


@pytest.fixture
def image_model(monkeypatch, cpu_vllm_config_context):
    # Keep graph specializations independent across fixtures.
    torch._dynamo.reset()
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    from vllm_torchtpu.models.vllm.muse_spark import vision
    from vllm_torchtpu.models.vllm.muse_spark.config import MuseSparkVisionConfig
    from vllm_torchtpu.models.vllm.muse_spark.multimodal import (
        MuseSparkForConditionalGeneration,
    )
    from vllm_torchtpu.models.vllm.muse_spark.vision_replay import MuseVisionReplayMixin

    monkeypatch.setattr(vision, "is_vit_use_data_parallel", lambda *args: True)
    monkeypatch.setattr(vision, "MMEncoderAttention", ReferenceAttention)
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1
    )
    config = MuseSparkVisionConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_hidden_layers=4,
        intermediate_size=64,
        pos_emb_height=4,
        pos_emb_width=4,
        patch_size=2,
        patch_temporal=2,
        merge_size=2,
        hidden_act="gelu",
        layer_norm_eps=1e-5,
        rope_parameters={"rope_type": "axial", "rope_theta": 10000.0},
        layer_types=["window_attention"] * 3 + ["full_attention"],
    )

    class Model(nn.Module, MuseVisionReplayMixin):
        _process_image_input = MuseSparkForConditionalGeneration._process_image_input
        _get_patch_count = MuseSparkForConditionalGeneration._get_patch_count
        _encode_vision = MuseSparkForConditionalGeneration._encode_vision

        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=48)
            self.model = nn.Module()
            self.model.vision_tower = vision.NaViT(config)
            self.model.vision_adapter = vision.NaViTAdapter(
                128, 48, use_data_parallel=True
            )
            self.model.vision_projection = vision.ColumnParallelLinear(
                48, 48, bias=False, disable_tp=True
            )
            self.normalize_tok_embeddings = False
            self._vision_output_dtype = torch.float32
            self._use_vision_dp = False
            self._patch_height = self._patch_width = 2
            self._downsample_factor = self._patch_temporal = 2

    torch.manual_seed(63)
    model = Model().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, 0.15)
    # This fixture bypasses the loader that installs CPU GEMM dispatch.
    for module in model.modules():
        if isinstance(module, vision.ColumnParallelLinear | vision.RowParallelLinear):
            module.cpu_linear = torch.nn.functional.linear
    return model


def make_manager(model, monkeypatch, dtype=torch.float32):
    from vllm_torchtpu.runner.mm_encoder_manager import MMEncoderManager

    graphs = []

    def counting_backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    compile_fn = torch.compile
    monkeypatch.setattr(
        torch,
        "compile",
        lambda fn, **kwargs: compile_fn(
            fn, backend=counting_backend, fullgraph=True, dynamic=False
        ),
    )
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            encoder_cudagraph_token_budgets=[4, 16, 64],
            encoder_cudagraph_max_vision_items_per_batch=4,
            encoder_cudagraph_max_frames_per_batch=0,
        ),
        model_config=SimpleNamespace(
            dtype=dtype,
            multimodal_config=SimpleNamespace(
                mm_encoder_tp_mode="weights", get_limit_per_prompt=lambda _: 0
            ),
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )
    return MMEncoderManager(config, torch.device("cpu"), model), graphs


@torch.inference_mode()
def test_bucket_replay_matches_individual_images_and_reuses_graph(
    image_model, monkeypatch
):
    manager, graphs = make_manager(image_model, monkeypatch)
    assert manager.supports_modality("image")
    assert not manager.supports_modality("video")
    assert not manager.supports_modality("audio")
    saved = None
    # Vary geometry within one bucket; isolate images, padding, and cached outputs.
    for grids in ([(2, 14), (2, 6)], [(6, 6)], [(4, 10), (2, 2)], [(10, 4)]):
        images = [torch.randn(3, h * 2, w * 2) for h, w in grids]
        expected = [
            image_model.encoder_eager_forward({"images": [im]}) for im in images
        ]
        actual = manager.execute({"images": images})
        assert len(actual) == len(expected)
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, rtol=2e-5, atol=2e-6)
        if saved is None:
            saved = (actual[0], actual[0].clone())
        torch.testing.assert_close(*saved, rtol=0, atol=0)
    assert len(graphs) == 1
    assert manager.graph_hits == 6
    assert manager.graph_misses == 0


@torch.inference_mode()
def test_fp32_encoder_retains_bf16_inputs_and_returns_bf16(image_model, monkeypatch):
    from vllm_torchtpu.models.vllm.muse_spark.vision_replay import prepare_image_buffers

    image_model.model.vision_tower.input_dtype = torch.bfloat16
    image_model._vision_output_dtype = torch.bfloat16
    manager, _ = make_manager(image_model, monkeypatch, torch.bfloat16)
    image = torch.randn(3, 12, 20)
    values = prepare_image_buffers(image_model.model.vision_tower, [image])
    rounded = prepare_image_buffers(image_model.model.vision_tower, [image.bfloat16()])
    torch.testing.assert_close(values["patches"], rounded["patches"], rtol=0, atol=0)
    torch.testing.assert_close(
        values["positions"], values["positions"].bfloat16().float(), rtol=0, atol=0
    )
    assert manager.budget_templates["default"][16]["patches"].dtype == torch.float32
    expected = image_model.encoder_eager_forward({"images": [image]})
    (actual,) = manager.execute({"images": [image]})
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@torch.inference_mode()
def test_exact_bucket_and_oversize_eager_fallback(image_model, monkeypatch):
    manager, _ = make_manager(image_model, monkeypatch)
    for grid in ((2, 2), (4, 4), (18, 16)):
        image = torch.randn(3, grid[0] * 2, grid[1] * 2)
        expected = image_model.encoder_eager_forward({"images": [image]})
        actual = manager.execute({"images": [image]})[0]
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert manager.graph_hits == 2
    assert manager.graph_misses == 1


@torch.inference_mode()
def test_precompiled_buckets_replay_new_geometry_without_recompilation(
    image_model, monkeypatch
):
    manager, graphs = make_manager(image_model, monkeypatch)
    # The counting backend executes synchronously on CPU.
    monkeypatch.setattr(
        "vllm_torchtpu.runner.mm_encoder_manager.synchronize_tensors",
        lambda *args, **kwargs: None,
    )
    manager.precompile_vision_encoder()
    assert len(graphs) == 3
    for height, width in ((2, 2), (6, 6), (8, 10), (2, 14), (10, 8)):
        image = torch.randn(3, height * 2, width * 2)
        expected = image_model.encoder_eager_forward({"images": [image]})
        (actual,) = manager.execute({"images": [image]})
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert len(graphs) == 3
    assert manager.graph_hits == 5
    assert manager.graph_misses == 0
