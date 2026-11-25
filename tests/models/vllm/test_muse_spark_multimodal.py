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
"""Geometry, normalization, and weight-coverage contracts for Muse perception."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_torchtpu.models.vllm.muse_spark.layers import MuseRMSNorm
from vllm_torchtpu.models.vllm.muse_spark.model import MuseModel
from vllm_torchtpu.models.vllm.muse_spark.multimodal import (
    MuseSparkForConditionalGeneration,
)


@pytest.fixture(scope="module", autouse=True)
def _cpu_platform_without_tpu_device():
    """CPU CI registers no TPU device type with PyTorch, so there the vLLM
    configs and the Gloo groups from `dist_ctx` must use the CPU device.

    Module-scoped and autouse so it is in place before `dist_ctx`; TPU hosts
    keep the real platform for the TPU-marked tests.
    """
    with pytest.MonkeyPatch.context() as patch:
        try:
            torch.device("tpu")
        except RuntimeError:
            from vllm.platforms import current_platform

            patch.setattr(current_platform, "device_name", "cpu")
            patch.setattr(current_platform, "device_type", "cpu")
        yield


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
    model.model.language_model.embed_input_ids = MuseModel.embed_input_ids.__get__(
        model.model.language_model
    )
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


def test_modality_loader_maps_checkpoint_vision_and_audio_names():
    class StackedQKV(nn.Linear):
        # Reads the shard id the weights mapper attaches, as QKVParallelLinear does.
        def load_weights(self, weights):
            for name, tensor in weights:
                offset = {"q": 0, "k": 2, "v": 4}[tensor.shard_id]
                getattr(self, name).data[offset : offset + 2].copy_(tensor)
                yield name

    model = _empty_omni()
    tower = model.model.vision_tower = nn.Module()
    tower.conv1_linear = nn.Linear(2, 3, bias=False)
    tower.positional_embedding_vlm = nn.Parameter(torch.zeros(4, 3))
    tower.encoder = nn.Module()
    tower.encoder.layers = nn.ModuleList([nn.Module()])
    layer = tower.encoder.layers[0]
    layer.attn = nn.Module()
    qkv = layer.attn.qkv_proj = StackedQKV(2, 6, bias=False)
    layer.attn.out_proj = nn.Linear(2, 2, bias=False)
    layer.ln_1 = nn.LayerNorm(2)
    layer.mlp = nn.Module()
    layer.mlp.c_fc = nn.Linear(2, 2, bias=False)
    tower.adapter = nn.Module()
    tower.adapter.c_proj = nn.Linear(2, 2, bias=False)
    model.model.vision_projection = nn.Linear(2, 3, bias=False)
    model.model.audio_tower = nn.Module()
    model.model.audio_tower.o_proj = nn.Linear(2, 3, bias=False)
    checkpoint = {
        "model.vision_tower.patch_embedder.patch_embedding.weight": (
            "model.vision_tower.conv1_linear.weight",
            (3, 2),
        ),
        "model.vision_tower.patch_embedder.position_embedding_table.weight": (
            "model.vision_tower.positional_embedding_vlm",
            (4, 3),
        ),
        "model.vision_tower.layers.0.attn.proj.weight": (
            "model.vision_tower.encoder.layers.0.attn.out_proj.weight",
            (2, 2),
        ),
        "model.vision_tower.layers.0.norm1.weight": (
            "model.vision_tower.encoder.layers.0.ln_1.weight",
            (2,),
        ),
        "model.vision_tower.layers.0.mlp.fc1.weight": (
            "model.vision_tower.encoder.layers.0.mlp.c_fc.weight",
            (2, 2),
        ),
        "model.vision_adapter.fc2.weight": (
            "model.vision_tower.adapter.c_proj.weight",
            (2, 2),
        ),
        "model.vision_projection.weight": ("model.vision_projection.weight", (3, 2)),
        "model.audio_tower.proj.weight": ("model.audio_tower.o_proj.weight", (3, 2)),
    }
    weights = [
        (f"model.vision_tower.layers.0.attn.{part}_proj.weight", torch.full((2, 2), i))
        for i, part in enumerate(("q", "k", "v"))
    ]
    weights += [
        (name, torch.full(shape, 7.0)) for name, (_, shape) in checkpoint.items()
    ]

    loaded = model.load_weights(weights)

    assert loaded == {
        "model.vision_tower.encoder.layers.0.attn.qkv_proj.weight",
        *(target for target, _ in checkpoint.values()),
    }
    torch.testing.assert_close(
        qkv.weight[:, 0], torch.tensor([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
    )
    params = dict(model.named_parameters())
    for target, shape in checkpoint.values():
        torch.testing.assert_close(params[target], torch.full(shape, 7.0))


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


def _build_image_model(monkeypatch, attention):
    """A small NaViT with the projection and replay hooks of the real model.

    ``attention`` replaces upstream's MMEncoderAttention; None keeps the TPU
    flash attention.
    """
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        RowParallelLinear,
    )
    from vllm.models.muse_spark.common import vision as upstream_vision
    from vllm.transformers_utils.configs.muse_spark import MuseSparkVisionConfig

    from vllm_torchtpu.models.vllm.muse_spark import vision
    from vllm_torchtpu.models.vllm.muse_spark.multimodal import (
        MuseSparkForConditionalGeneration,
    )
    from vllm_torchtpu.models.vllm.muse_spark.vision_replay import MuseVisionReplayMixin

    monkeypatch.setattr(upstream_vision, "is_vit_use_data_parallel", lambda *args: True)
    if attention is not None:
        monkeypatch.setattr(upstream_vision, "MMEncoderAttention", attention)
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

    omni = MuseSparkForConditionalGeneration

    class Model(nn.Module, MuseVisionReplayMixin):
        _process_image_input = omni._process_image_input
        _image_items = omni._image_items
        _get_patch_count = omni._get_patch_count
        _encode_vision = omni._encode_vision
        _project_vision = omni._project_vision
        _run_dp_sharded_navit = omni._run_dp_sharded_navit

        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=48)
            self.model = nn.Module()
            self.model.vision_tower = vision.NaViT(
                config, projector_hidden_size=48, max_num_image_tokens=64
            )
            self.model.vision_projection = ColumnParallelLinear(
                48, 48, bias=False, disable_tp=True
            )
            self.perception_emb_norm = MuseRMSNorm(48, eps=1e-5, has_weight=False)
            self._max_image_tokens = 64
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
        if isinstance(module, ColumnParallelLinear | RowParallelLinear):
            module.cpu_linear = torch.nn.functional.linear
    return model


@pytest.fixture
def image_model(monkeypatch, cpu_vllm_config_context):
    # Keep graph specializations independent across fixtures.
    torch._dynamo.reset()
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    return _build_image_model(monkeypatch, ReferenceAttention)


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
def test_navit_matches_upstream_forward(image_model):
    from vllm.models.muse_spark.common.vision import NaViT as UpstreamNaViT

    tower = image_model.model.vision_tower
    images = [torch.randn(3, 12, 20), torch.randn(3, 8, 8), torch.randn(4, 3, 8, 12)]
    torch.testing.assert_close(
        tower(images), UpstreamNaViT.forward(tower, images), rtol=2e-5, atol=2e-6
    )


@torch.inference_mode()
def test_bf16_replay_matches_eager(image_model, monkeypatch):
    image_model.to(torch.bfloat16)
    manager, _ = make_manager(image_model, monkeypatch, torch.bfloat16)
    assert manager.budget_templates["default"][16]["patches"].dtype == torch.bfloat16
    image = torch.randn(3, 12, 20)
    expected = image_model.encoder_eager_forward({"images": [image]})
    (actual,) = manager.execute({"images": [image]})
    assert actual.dtype == expected.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


def test_max_image_tokens_sizes_rope_table_and_replay_buckets(image_model):
    tower = image_model.model.vision_tower
    # 64 tokens of 2x2 merges: sqrt(256) + 1 = 17 rows and columns, not the
    # 129 of upstream's fixed 4096 tokens.
    assert tower.max_rope_grid_h == tower.max_rope_grid_w == 17
    assert tower.precomputed_freqs_cis.shape[:2] == (17, 17)
    assert image_model.get_encoder_cudagraph_budget_range(None) == (64, 64)


@torch.inference_mode()
def test_multi_frame_image_items_are_rejected_on_both_paths(image_model, monkeypatch):
    clip = torch.randn(2, 3, 8, 8)
    with pytest.raises(ValueError, match=r"Expected \[C, H, W\] image items"):
        image_model.encoder_eager_forward({"images": [clip]})
    with pytest.raises(ValueError, match=r"Expected \[C, H, W\] image items"):
        image_model.get_encoder_cudagraph_item_specs({"images": [clip]})


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


def _omni_vllm_config(
    *,
    image=1,
    video=1,
    audio=1,
    cudagraph_mm_encoder=False,
    compile_mm_encoder=False,
    data_parallel_size=1,
    enable_expert_parallel=False,
):
    from vllm.config.compilation import CompilationMode
    from vllm.transformers_utils.configs.muse_spark import MuseSparkVisionConfig

    text = SimpleNamespace(
        vocab_size=48,
        hidden_size=64,
        head_dim=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        rms_norm_eps=1e-5,
        post_norm_eps=1e-8,
        layer_rope_theta=[0.0],
        layer_types=["full_attention"],
        mlp_layer_types=["sparse"],
        sliding_window=4,
        qk_scale_factor=1.0,
        rope_parameters={"rope_type": "default", "rope_theta": 500000.0},
        num_experts_per_tok=2,
        num_local_experts=4,
        moe_hidden_size=32,
        moe_intermediate_size=32,
        output_multiplier=1.0,
        routed_scaling_factor=1.0,
        residual_gate_temperature=0.3,
        tie_word_embeddings=False,
        final_logit_softcapping=None,
    )
    vision = MuseSparkVisionConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_hidden_layers=2,
        intermediate_size=64,
        pos_emb_height=4,
        pos_emb_width=4,
        patch_size=2,
        patch_temporal=2,
        merge_size=2,
        hidden_act="gelu",
        layer_norm_eps=1e-5,
        rope_parameters={"rope_type": "axial", "rope_theta": 10000.0},
        layer_types=["window_attention", "full_attention"],
    )
    audio_config = SimpleNamespace(
        out_hidden_size=64,
        patch_sizes=[2, 3],
        conv_kernel_size=3,
        hidden_sizes=[4, 8],
        depths=[1, 1],
        mlp_ratio=2.0,
        rms_norm_eps=1e-6,
    )
    limits = {"image": image, "video": video, "audio": audio}
    mm_config = SimpleNamespace(get_limit_per_prompt=lambda modality: limits[modality])
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                text_config=text,
                vision_config=vision,
                audio_config=audio_config,
                projector_hidden_size=48,
            ),
            hf_text_config=text,
            dtype=torch.bfloat16,
            max_model_len=64,
            quantization=None,
            model="meta-models/muse-spark-test",
            revision=None,
            get_multimodal_config=lambda: mm_config,
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            enable_eplb=False,
            enable_expert_parallel=enable_expert_parallel,
            data_parallel_size=data_parallel_size,
            expert_placement_strategy="linear",
        ),
        load_config=SimpleNamespace(load_format="safetensors"),
        compilation_config=SimpleNamespace(
            mode=CompilationMode.NONE,
            cudagraph_mm_encoder=cudagraph_mm_encoder,
            compile_mm_encoder=compile_mm_encoder,
        ),
        cache_config=None,
        quant_config=None,
        use_v2_model_runner=False,
        kernel_config=SimpleNamespace(linear_backend="auto"),
    )


@pytest.fixture
def build_omni(dist_ctx, monkeypatch):
    """Construct MuseSparkForConditionalGeneration from a tiny config on CPU."""
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.models.muse_spark.common import audio as upstream_audio
    from vllm.models.muse_spark.common import vision as upstream_vision
    from vllm.utils.torch_utils import set_default_torch_dtype

    from vllm_torchtpu.models.vllm.muse_spark import model, multimodal

    monkeypatch.setattr(model, "Attention", lambda *args, **kwargs: nn.Module())
    monkeypatch.setattr(
        model,
        "FusedMoEFactory",
        lambda **_: SimpleNamespace(routed_experts=nn.Module()),
    )
    monkeypatch.setattr(upstream_vision, "MMEncoderAttention", ReferenceAttention)
    monkeypatch.setattr(upstream_audio, "RMSNorm", nn.RMSNorm)
    processor = {"image_processor": {"max_image_tokens": 64}}
    monkeypatch.setattr(
        multimodal, "get_processor_config", lambda *args, **kwargs: processor
    )

    def build(vllm_config):
        with (
            set_current_vllm_config(VllmConfig(device_config=DeviceConfig("cpu"))),
            set_default_torch_dtype(torch.bfloat16),
        ):
            omni = MuseSparkForConditionalGeneration(vllm_config=vllm_config)
        for module in omni.modules():
            if isinstance(module, LinearBase):
                module.cpu_linear = torch.nn.functional.linear
        return omni

    build.processor = processor
    return build


@pytest.mark.parametrize("cudagraph_mm_encoder", [False, True])
def test_init_builds_towers_in_model_dtype_without_global_precision(
    build_omni, cudagraph_mm_encoder
):
    precision = torch.get_float32_matmul_precision()
    omni = build_omni(_omni_vllm_config(cudagraph_mm_encoder=cudagraph_mm_encoder))

    assert torch.get_float32_matmul_precision() == precision
    tower = omni.model.vision_tower
    assert {p.dtype for p in tower.parameters()} == {torch.bfloat16}
    assert tower.max_rope_grid_h == 17 and omni._max_image_tokens == 64
    assert omni.model.vision_projection.weight.dtype == torch.bfloat16
    assert hasattr(tower, "adapter") and not hasattr(omni.model, "vision_adapter")
    assert omni.get_encoder_cudagraph_budget_range(None) == (64, 64)


def test_init_compiles_audio_with_static_tpu_shapes(build_omni, monkeypatch):
    calls = []
    monkeypatch.setattr(
        torch, "compile", lambda fn, **kwargs: calls.append(kwargs) or fn
    )
    build_omni(_omni_vllm_config(compile_mm_encoder=True))
    assert calls == [{"backend": "tpu", "fullgraph": True, "dynamic": False}]


def test_init_requires_max_image_tokens(build_omni):
    build_omni.processor["image_processor"].pop("max_image_tokens")
    with pytest.raises(ValueError, match="image_processor.max_image_tokens"):
        build_omni(_omni_vllm_config())


@pytest.mark.parametrize(
    ("limits", "cudagraph", "message"),
    [
        ({"video": 1, "audio": 0}, True, "supports image inputs only"),
        ({"video": 0, "audio": 1}, True, "supports image inputs only"),
        ({"video": 0, "audio": 0}, False, "require cudagraph_mm_encoder=True"),
    ],
    ids=["video", "audio", "image_without_replay"],
)
def test_dp_attention_with_ep_rejects_unsupported_inputs_at_startup(
    build_omni, limits, cudagraph, message
):
    config = _omni_vllm_config(
        cudagraph_mm_encoder=cudagraph,
        data_parallel_size=2,
        enable_expert_parallel=True,
        **limits,
    )
    with pytest.raises(ValueError, match=message):
        build_omni(config)


def test_dp_attention_with_ep_accepts_images_through_replay(build_omni):
    config = _omni_vllm_config(
        video=0,
        audio=0,
        cudagraph_mm_encoder=True,
        data_parallel_size=2,
        enable_expert_parallel=True,
    )
    assert build_omni(config).use_ep


@torch.inference_mode()
def test_video_clips_encode_through_the_tower_and_projection(build_omni):
    omni = build_omni(_omni_vllm_config())
    torch.manual_seed(5)
    with torch.no_grad():
        for parameter in omni.parameters():
            parameter.normal_(0, 0.15)
    for module in omni.modules():
        quant_method = getattr(module, "quant_method", None)
        if quant_method is not None:
            quant_method.process_weights_after_loading(module)
    clip = torch.randn(4, 3, 8, 12, dtype=torch.float32)

    (embedding,) = omni.embed_multimodal(videos=[clip])

    # Two 2-frame clips of 4x6 patches, merged 2x2: 2 * 6 tokens.
    assert embedding.shape == (12, 64) and embedding.dtype == torch.bfloat16
    expected = omni._project_vision(omni.model.vision_tower([clip]))
    torch.testing.assert_close(embedding, expected, rtol=0, atol=0)


@torch.inference_mode()
def test_dp_sharded_navit_gathers_every_rank_in_input_order(image_model, monkeypatch):
    from vllm_torchtpu.models.vllm.muse_spark import multimodal

    images = [torch.randn(3, h, w) for h, w in ((8, 8), (12, 20), (4, 4), (8, 12))]
    expected = image_model.model.vision_tower(images)
    monkeypatch.setattr(multimodal, "synchronize_tensors", lambda *args: None)
    monkeypatch.setattr(multimodal, "get_tensor_model_parallel_world_size", lambda: 2)
    local = {}

    def run(rank, gather):
        monkeypatch.setattr(multimodal, "get_tensor_model_parallel_rank", lambda: rank)
        monkeypatch.setattr(multimodal, "tensor_model_parallel_all_gather", gather)
        return image_model._run_dp_sharded_navit(images)

    for rank in (0, 1):
        run(rank, lambda x, dim, rank=rank: local.setdefault(rank, x).repeat(2, 1))
    gathered = torch.cat((local[0], local[1]))
    for rank in (0, 1):
        actual = run(rank, lambda x, dim: gathered)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_models_register_with_vllm():
    from vllm.model_executor.models import ModelRegistry

    from vllm_torchtpu.models.vllm import register_models
    from vllm_torchtpu.models.vllm.muse_spark.model import MuseSparkForCausalLM

    register_models()
    for architecture, cls in (
        ("MuseSparkForCausalLM", MuseSparkForCausalLM),
        ("MuseSparkForConditionalGeneration", MuseSparkForConditionalGeneration),
    ):
        entry = ModelRegistry.models.get(architecture)
        assert entry is not None, f"{architecture} is not registered"
        assert entry.load_model_cls() is cls


def _vision_items():
    generator = torch.Generator().manual_seed(21)
    return [
        torch.randn(3, 12, 20, generator=generator),
        torch.randn(3, 8, 8, generator=generator),
        torch.randn(4, 3, 8, 12, generator=generator),
    ]


@pytest.fixture
def tpu_and_reference_image_models(dist_ctx, monkeypatch):
    """The same BF16 weights on CPU with reference attention and on TPU with
    the flash attention kernel; the RoPE tables stay FP32, as in serving."""
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.utils.torch_utils import set_default_torch_dtype

    torch._dynamo.reset()
    with (
        set_current_vllm_config(VllmConfig(device_config=DeviceConfig("cpu"))),
        set_default_torch_dtype(torch.bfloat16),
    ):
        monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
        reference = _build_image_model(monkeypatch, ReferenceAttention)
    monkeypatch.undo()
    with set_current_vllm_config(VllmConfig()), set_default_torch_dtype(torch.bfloat16):
        model = _build_image_model(monkeypatch, None)
    model.load_state_dict(reference.state_dict())
    return reference, model.to("tpu")


@pytest.mark.tpu_test
@torch.inference_mode()
def test_tpu_tower_with_flash_attention_matches_cpu_reference(
    tpu_and_reference_image_models,
):
    """Images and a video clip through NaViT with the TPU flash attention."""
    reference, model = tpu_and_reference_image_models
    items = _vision_items()
    expected = reference.model.vision_tower(items).float()

    actual = model.model.vision_tower([x.to("tpu") for x in items]).cpu().float()

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)
    assert (actual - expected).abs().mean() < 1e-2


@pytest.mark.tpu_test
@torch.inference_mode()
def test_tpu_bucket_replay_matches_tpu_eager(tpu_and_reference_image_models):
    from vllm_torchtpu.runner.mm_encoder_manager import MMEncoderManager

    _, model = tpu_and_reference_image_models
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            encoder_cudagraph_token_budgets=[16, 64],
            encoder_cudagraph_max_vision_items_per_batch=4,
            encoder_cudagraph_max_frames_per_batch=0,
        ),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            multimodal_config=SimpleNamespace(
                mm_encoder_tp_mode="weights", get_limit_per_prompt=lambda _: 0
            ),
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )
    manager = MMEncoderManager(config, torch.device("tpu"), model)
    images = [x.to("tpu") for x in _vision_items()[:2]]
    expected = [model.encoder_eager_forward({"images": [x]}).cpu() for x in images]

    actual = [x.cpu() for x in manager.execute({"images": images})]

    assert manager.graph_hits == 2 and manager.graph_misses == 0
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got.float(), want.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.tpu_test
@torch.inference_mode()
def test_tpu_dp_sharded_navit_matches_tower(tpu_and_reference_image_models):
    """The encoder-DP path (materialize, all-gather, synchronize) on TPU."""
    _, model = tpu_and_reference_image_models
    items = [x.to("tpu") for x in _vision_items()]
    expected = model.model.vision_tower(items).cpu()

    actual = model._run_dp_sharded_navit(items).cpu()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
