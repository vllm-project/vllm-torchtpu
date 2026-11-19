# SPDX-License-Identifier: Apache-2.0
import torch

from vllm_torchtpu.model_loader_patches import ExpertParamStager, ExpertWriteTracker


def test_w13_completes_after_both_shards_of_every_expert():
    tracker = ExpertWriteTracker()
    w13 = torch.empty(4, 6, 8)
    for expert in range(4):
        assert not tracker.record(w13, expert, "w1")
        last = tracker.record(w13, expert, "w3")
        assert last == (expert == 3)


def test_w2_completes_after_every_expert():
    tracker = ExpertWriteTracker()
    w2 = torch.empty(3, 8, 6)
    assert [tracker.record(w2, e, "w2") for e in range(3)] == [False, False, True]


def test_w13_waits_for_w3_when_every_w1_arrives_first():
    tracker = ExpertWriteTracker()
    w13 = torch.empty(4, 6, 8)
    assert not any(tracker.record(w13, e, "w1") for e in range(4))
    assert [tracker.record(w13, e, "w3") for e in range(4)] == [
        False,
        False,
        False,
        True,
    ]


def test_one_local_expert_needs_both_shards():
    tracker = ExpertWriteTracker()
    w13 = torch.empty(1, 6, 8)
    assert not tracker.record(w13, 5, "w1")
    assert tracker.record(w13, 5, "w3")


def test_whole_tensor_writes_never_complete_a_parameter():
    # gpt-oss MXFP4 checkpoints copy every expert in one write with no
    # shard or expert id; such a parameter is written at the end of loading.
    tracker = ExpertWriteTracker()
    w13 = torch.empty(1, 6, 8)
    assert not tracker.record(w13, None, None)
    assert not tracker.record(w13, None, None)


def test_parameters_are_tracked_apart():
    tracker = ExpertWriteTracker()
    a, b = torch.empty(2, 4, 4), torch.empty(2, 4, 4)
    assert not tracker.record(a, 0, "w2")
    assert not tracker.record(b, 0, "w2")
    assert tracker.record(a, 1, "w2")
    assert tracker.record(b, 1, "w2")


def test_a_completed_parameter_can_be_written_again():
    tracker = ExpertWriteTracker()
    w2 = torch.empty(2, 4, 4)
    assert tracker.record(w2, 0, "w2") is False
    assert tracker.record(w2, 1, "w2") is True
    assert tracker.record(w2, 0, "w2") is False


def _param(*shape):
    return torch.nn.Parameter(torch.zeros(*shape), requires_grad=False)


def test_stager_writes_the_parameter_once_its_last_expert_lands():
    stager = ExpertParamStager()
    w2 = _param(2, 4, 4)
    host = stager.host_param(w2)
    assert stager.host_param(w2) is host
    host[0] = 1
    assert not stager.written(w2, 0, "w2")
    assert torch.equal(w2.data, torch.zeros(2, 4, 4))
    host[1] = 2
    assert stager.written(w2, 1, "w2")
    assert torch.equal(w2.data[0], torch.ones(4, 4))
    assert torch.equal(w2.data[1], torch.full((4, 4), 2.0))
    assert len(stager) == 0


def test_stand_in_keeps_the_parameter_attributes():
    stager = ExpertParamStager()
    w13 = _param(2, 8, 4)
    w13.is_transposed = True
    host = stager.host_param(w13)
    assert host.is_transposed is True
    assert host.shape == w13.shape and host.device.type == "cpu"


def test_flush_all_writes_incomplete_parameters():
    stager = ExpertParamStager()
    w13 = _param(3, 8, 4)
    stager.host_param(w13)[0] = 5
    assert not stager.written(w13, 0, "w1")
    stager.flush_all()
    assert len(stager) == 0
    assert stager.flushed == 1
    assert torch.equal(w13.data[0], torch.full((8, 4), 5.0))


def test_flush_uploads_in_chunks(monkeypatch):
    from vllm_torchtpu import model_loader_patches

    # two experts per upload
    monkeypatch.setattr(model_loader_patches, "_UPLOAD_CHUNK_BYTES", 2 * 8 * 4 * 4)
    stager = ExpertParamStager()
    w2 = _param(5, 8, 4)
    host = stager.host_param(w2)
    for expert in range(5):
        host[expert] = expert + 1
        stager.written(w2, expert, "w2")
    assert w2.shape == (5, 8, 4)
    for expert in range(5):
        assert torch.equal(w2.data[expert], torch.full((8, 4), expert + 1.0))


def test_the_oldest_parameter_is_written_early_when_too_many_are_staged(monkeypatch):
    from vllm_torchtpu import model_loader_patches

    monkeypatch.setattr(model_loader_patches, "_MAX_STAGED", 2)
    stager = ExpertParamStager()
    first, second, third = _param(2, 4, 4), _param(2, 4, 4), _param(2, 4, 4)
    stager.host_param(first)[0] = 7
    stager.host_param(second)
    assert stager.staging(first)
    stager.host_param(third)
    # staging a third parameter wrote the first one early
    assert stager.flushed == 1
    assert torch.equal(first.data[0], torch.full((4, 4), 7.0))
    assert not stager.staging(first)
    assert stager.staging(second) and stager.staging(third)
    assert len(stager) == 2


def test_interleaved_parameters_sharing_pooled_buffers_keep_their_experts():
    """Every w1 of two same-shaped parameters lands before any w3; neither
    may be written early and hand its buffer to the other."""
    stager = ExpertParamStager()
    a, b = _param(2, 4, 2), _param(2, 4, 2)
    for param, w1 in ((a, 11), (b, 21)):
        for expert in range(2):
            stager.host_param(param)[expert, :2] = w1
            assert not stager.written(param, expert, "w1")
    assert stager.flushed == 0
    for param, w3 in ((a, 13), (b, 23)):
        for expert in range(2):
            stager.host_param(param)[expert, 2:] = w3
            assert stager.written(param, expert, "w3") == (expert == 1)
    assert stager.flushed == 2 and len(stager) == 0
    for param, w1, w3 in ((a, 11, 13), (b, 21, 23)):
        assert torch.equal(param.data[:, :2], torch.full((2, 2, 2), float(w1)))
        assert torch.equal(param.data[:, 2:], torch.full((2, 2, 2), float(w3)))


def test_the_patched_loader_keeps_the_upstream_loader_attributes(monkeypatch):
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.model_loader import base_loader

    from vllm_torchtpu.model_loader_patches import patch_moe_expert_write_staging

    # The patch is process-wide; undo it after the test so later tests load
    # expert weights through the stock loader.
    monkeypatch.setattr(RoutedExperts, "weight_loader", RoutedExperts.weight_loader)
    monkeypatch.setattr(
        RoutedExperts, "_tpu_expert_staging_patch", False, raising=False
    )
    monkeypatch.setattr(
        base_loader,
        "process_weights_after_loading",
        base_loader.process_weights_after_loading,
    )
    patch_moe_expert_write_staging()
    loader = RoutedExperts.weight_loader
    assert loader.supports_moe_loading is True
    assert loader.__name__ == "weight_loader"


def _install_cpu_torch_tpu_stub(monkeypatch) -> None:
    """Minimal torch_tpu stub so VllmFp8MoEMethodTPU imports on CPU."""
    import sys
    import types

    import jax.numpy as jnp

    if "torch_tpu" in sys.modules:
        return
    pallas_mod = types.ModuleType("torch_tpu._internal.pallas")
    pallas_inner = types.SimpleNamespace(
        JaxCallable=object,
        TORCH_TO_JAX_DTYPE_MAP={
            torch.float32: jnp.float32,
            torch.bfloat16: jnp.bfloat16,
            torch.float8_e4m3fn: jnp.float8_e4m3fn,
            torch.uint8: jnp.uint8,
            torch.int32: jnp.int32,
        },
    )

    class _FakeOp:
        def __init__(self, fn):
            self._fn = fn

        def __call__(self, *args, **kwargs):
            return self._fn(*args, **kwargs)

        def register_fake(self, fn):
            return fn

    pallas_mod.pallas = pallas_inner
    pallas_mod.jax_op = lambda _name, fn, **_kw: _FakeOp(fn)
    pallas_mod.sharded_jax_op = lambda _name, fn, **_kw: _FakeOp(fn)
    internal_mod = types.ModuleType("torch_tpu._internal")
    internal_mod.pallas = pallas_mod
    root_mod = types.ModuleType("torch_tpu")
    root_mod._internal = internal_mod
    monkeypatch.setitem(sys.modules, "torch_tpu", root_mod)
    monkeypatch.setitem(sys.modules, "torch_tpu._internal", internal_mod)
    monkeypatch.setitem(sys.modules, "torch_tpu._internal.pallas", pallas_mod)
    if "tpu" not in torch._dynamo.list_backends():
        torch._dynamo.register_backend(name="tpu")(lambda gm, _inputs: gm.forward)


def _build_fake_fp8_moe_layer(
    monkeypatch,
    *,
    prefix: str = "model.layers.3.mlp.experts",
    num_experts: int = 4,
    inter: int = 512,
    hidden: int = 512,
    block: int = 128,
    num_hidden_layers: int | None = None,
):
    from types import SimpleNamespace

    from vllm.model_executor.layers.fused_moe import RoutedExperts

    _install_cpu_torch_tpu_stub(monkeypatch)
    from vllm_torchtpu.layers.adapter.quantization import fp8 as fp8_mod

    def fake_quantize_fp4(w_kmajor, block, pack=True):
        # Deterministic value-preserving packed-FP4 stand-in on CPU so bit-exact
        # equivalence between early per-layer requant and end-of-load requant is
        # sensitive to every element of w13/w2 and their scales.
        e, k, n = w_kmajor.shape
        blocks = w_kmajor.reshape(e, k // block, block, n)
        scale = (blocks.abs().amax(dim=2, keepdim=True) / 6.0).clamp(min=1e-6)
        q = ((blocks / scale).round().clamp(-6, 6) + 8).to(torch.uint8)
        q_flat = q.reshape(e, k, n)
        packed = (q_flat[:, 0::2, :] & 0x0F) | ((q_flat[:, 1::2, :] & 0x0F) << 4)
        return packed.contiguous(), scale.to(torch.float32).contiguous()

    monkeypatch.setattr(fp8_mod, "quantize_native_fp4_kmajor", fake_quantize_fp4)
    monkeypatch.setattr(fp8_mod, "prebuild_fused_moe_kernel", lambda **_kw: None)
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.fused_moe_ep.prebuild_fused_moe_ep",
        lambda *_a, **_kw: None,
    )

    quant_cfg = SimpleNamespace(
        is_checkpoint_fp8_serialized=True,
        weight_block_size=[block, block],
        activation_scheme="dynamic",
    )
    moe_cfg = SimpleNamespace(
        has_bias=False,
        experts_per_token=2,
        intermediate_size_per_partition=inter,
        intermediate_size_per_partition_unpadded=inter,
        tp_size=1,
        tp_rank=0,
        moe_parallel_config=SimpleNamespace(use_ep=False, ep_size=1, tp_size=1),
    )
    qm = object.__new__(fp8_mod.VllmFp8MoEMethodTPU)
    qm.quant_config = quant_cfg
    qm.moe = moe_cfg
    qm.weight_block_size = [block, block]
    qm.moe_block_shape = (block, block)
    qm.block_quant = True
    qm.weight_scale_name = "weight_scale_inv"
    qm.prefix = prefix
    qm._num_hidden_layers = num_hidden_layers
    qm._tpu_activation_str = None
    qm._use_adaptive_fused_moe = False
    qm._rhs_quant_dtype = None

    class FakeRoutedExperts(RoutedExperts):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.quant_config = None
            self.quant_method = qm
            self.moe_config = moe_cfg
            self.activation = "silu"
            self.renormalize = True
            self.num_experts = num_experts
            self.global_num_experts = num_experts
            self._loaded_expert_biases = set()
            self.w13_weight = torch.nn.Parameter(
                torch.zeros(num_experts, 2 * inter, hidden, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.w2_weight = torch.nn.Parameter(
                torch.zeros(num_experts, hidden, inter, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.w13_weight_scale_inv = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    2 * inter // block,
                    hidden // block,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            self.w13_weight_scale_inv.quant_method = "block"
            self.w2_weight_scale_inv = torch.nn.Parameter(
                torch.ones(
                    num_experts, hidden // block, inter // block, dtype=torch.float32
                ),
                requires_grad=False,
            )
            self.w2_weight_scale_inv.quant_method = "block"

        def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
            return expert_id

        def _load_w13(
            self, shard_id, shard_dim, loaded_weight, expert_data, tp_rank, **_kw
        ):
            half = expert_data.shape[0] // 2
            dst = expert_data[:half] if shard_id == "w1" else expert_data[half:]
            dst.copy_(loaded_weight)

        def _load_w2(self, shard_dim, loaded_weight, expert_data, tp_rank, **_kw):
            expert_data.copy_(loaded_weight)

    return FakeRoutedExperts()


def _stream_expert_shards_into_layer(layer, shards_dict) -> None:
    for e in range(layer.num_experts):
        layer.weight_loader(
            layer.w13_weight, shards_dict[("w1", e)], "w1.weight", "w1", e
        )
        layer.weight_loader(
            layer.w13_weight_scale_inv,
            shards_dict[("s1", e)],
            "w1.weight_scale_inv",
            "w1",
            e,
        )
        layer.weight_loader(
            layer.w13_weight, shards_dict[("w3", e)], "w3.weight", "w3", e
        )
        layer.weight_loader(
            layer.w13_weight_scale_inv,
            shards_dict[("s3", e)],
            "w3.weight_scale_inv",
            "w3",
            e,
        )
        layer.weight_loader(
            layer.w2_weight, shards_dict[("w2", e)], "w2.weight", "w2", e
        )
        layer.weight_loader(
            layer.w2_weight_scale_inv,
            shards_dict[("s2", e)],
            "w2.weight_scale_inv",
            "w2",
            e,
        )


def _make_random_moe_shards(seed: int, num_experts=4, inter=512, hidden=512, block=128):
    g = torch.Generator().manual_seed(seed)
    d = {}
    for e in range(num_experts):
        d[("w1", e)] = torch.randn((inter, hidden), generator=g).to(torch.float8_e4m3fn)
        d[("w3", e)] = torch.randn((inter, hidden), generator=g).to(torch.float8_e4m3fn)
        d[("w2", e)] = torch.randn((hidden, inter), generator=g).to(torch.float8_e4m3fn)
        d[("s1", e)] = torch.rand((inter // block, hidden // block), generator=g) + 0.25
        d[("s3", e)] = torch.rand((inter // block, hidden // block), generator=g) + 0.25
        d[("s2", e)] = torch.rand((hidden // block, inter // block), generator=g) + 0.25
    return d


def test_early_per_layer_fp4_requant_matches_end_of_load_and_bounds_fp8_memory(
    monkeypatch,
):
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.model_loader import base_loader

    from vllm_torchtpu import envs, model_loader_patches

    if not hasattr(torch, "float4_e2m1fn_x2"):
        import pytest

        pytest.skip("torch.float4_e2m1fn_x2 unavailable")

    monkeypatch.setenv("MOE_REQUANTIZE_WEIGHT_DTYPE", "fp4")
    monkeypatch.setenv("MOE_REQUANTIZE_BLOCK_SIZE", "512")
    monkeypatch.setenv("VLLM_TPU_MOE_REQUANT_MTP", "0")
    monkeypatch.setattr(model_loader_patches, "_STAGE_ON_CPU_FOR_TEST", True)

    stock_weight_loader = RoutedExperts.weight_loader
    monkeypatch.setattr(RoutedExperts, "weight_loader", stock_weight_loader)

    def run_pipeline(early_enabled: bool):
        monkeypatch.setenv("VLLM_TPU_MOE_EARLY_REQUANT", "1" if early_enabled else "0")
        assert envs.VLLM_TPU_MOE_EARLY_REQUANT is early_enabled
        monkeypatch.setattr(RoutedExperts, "weight_loader", stock_weight_loader)
        monkeypatch.setattr(
            RoutedExperts, "_tpu_expert_staging_patch", False, raising=False
        )

        def orig_proc(model, _mc, _td):
            for m in model:
                m.quant_method.process_weights_after_loading(m)

        monkeypatch.setattr(base_loader, "process_weights_after_loading", orig_proc)
        model_loader_patches.patch_moe_expert_write_staging()

        prefixes = [
            "model.layers.3.mlp.experts",
            "model.layers.4.mlp.experts",
            "model.layers.5.mlp.experts",
            "mtp.layers.0.mlp.experts",  # MTP layer: kept FP8 when REQUANT_MTP=0
            # A DeepSeek/GLM-style drafter MoE: its layer index is the target's
            # num_hidden_layers, so it is kept FP8 as well.
            "model.layers.78.mlp.experts",
        ]
        layers = [
            _build_fake_fp8_moe_layer(monkeypatch, prefix=p, num_hidden_layers=78)
            for p in prefixes
        ]
        peak_fp8_layers_after_stream = 0
        for idx, layer in enumerate(layers):
            _stream_expert_shards_into_layer(
                layer, _make_random_moe_shards(seed=100 + idx)
            )
            fp8_unrequanted = sum(
                1
                for m in layers[: idx + 1]
                if getattr(m, "_tpu_moe_weights_requanted", None) is None
            )
            peak_fp8_layers_after_stream = max(
                peak_fp8_layers_after_stream, fp8_unrequanted
            )

        base_loader.process_weights_after_loading(layers, None, "cpu")
        return layers, peak_fp8_layers_after_stream

    early_layers, peak_early = run_pipeline(early_enabled=True)
    late_layers, peak_late = run_pipeline(early_enabled=False)

    # With early requant on, every layer converts as soon as its 4th parameter
    # completes during shard streaming (0 unrequanted layers accumulate!).
    assert peak_early == 0
    assert peak_late == 5

    # Target layers 0..2 are packed uint8 FP4; MTP layers 3 and 4 stay FP8.
    for idx in range(3):
        assert early_layers[idx].w13_weight.dtype == torch.uint8
        assert early_layers[idx].w2_weight.dtype == torch.uint8
    for idx in (3, 4):
        assert early_layers[idx].w13_weight.dtype == torch.float8_e4m3fn
        assert early_layers[idx].w2_weight.dtype == torch.float8_e4m3fn
        assert early_layers[idx]._tpu_moe_weights_requanted[3] is True

    # Bit-for-bit identical weights and inverse scales across all 5 layers.
    for e_layer, l_layer in zip(early_layers, late_layers):
        for attr in (
            "w13_weight",
            "w2_weight",
            "w13_weight_scale_inv",
            "w2_weight_scale_inv",
        ):
            a, b = getattr(e_layer, attr), getattr(l_layer, attr)
            assert a.dtype == b.dtype and a.shape == b.shape
            assert torch.equal(a.data, b.data)


def test_early_requant_skips_written_early_layer_and_falls_back_cleanly(monkeypatch):
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.model_loader import base_loader

    from vllm_torchtpu import model_loader_patches

    if not hasattr(torch, "float4_e2m1fn_x2"):
        import pytest

        pytest.skip("torch.float4_e2m1fn_x2 unavailable")

    monkeypatch.setenv("MOE_REQUANTIZE_WEIGHT_DTYPE", "fp4")
    monkeypatch.setenv("MOE_REQUANTIZE_BLOCK_SIZE", "512")
    monkeypatch.delenv("VLLM_TPU_MOE_EARLY_REQUANT", raising=False)
    monkeypatch.setattr(model_loader_patches, "_STAGE_ON_CPU_FOR_TEST", True)
    # Force early eviction when the 3rd parameter of the layer is staged.
    monkeypatch.setattr(model_loader_patches, "_MAX_STAGED", 2)

    monkeypatch.setattr(RoutedExperts, "weight_loader", RoutedExperts.weight_loader)
    monkeypatch.setattr(
        RoutedExperts, "_tpu_expert_staging_patch", False, raising=False
    )
    monkeypatch.setattr(
        base_loader,
        "process_weights_after_loading",
        lambda model, _mc, _td: [
            m.quant_method.process_weights_after_loading(m) for m in model
        ],
    )
    model_loader_patches.patch_moe_expert_write_staging()

    layer = _build_fake_fp8_moe_layer(monkeypatch)
    _stream_expert_shards_into_layer(layer, _make_random_moe_shards(seed=7))
    # Evicted early during staging -> deferred to end-of-load requantization.
    assert getattr(layer, "_tpu_moe_weights_requanted", None) is None
    base_loader.process_weights_after_loading([layer], None, "cpu")
    assert layer.w13_weight.dtype == torch.uint8
    assert layer.w2_weight.dtype == torch.uint8


def test_early_requant_releases_params_dict_pinned_fp8_storage(monkeypatch):
    """Regression test for vLLM's load_weights ``params_dict`` pinning old
    Parameter wrappers until all checkpoint shards finish streaming."""
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.model_loader import base_loader

    from vllm_torchtpu import model_loader_patches

    if not hasattr(torch, "float4_e2m1fn_x2"):
        import pytest

        pytest.skip("torch.float4_e2m1fn_x2 unavailable")

    monkeypatch.setenv("MOE_REQUANTIZE_WEIGHT_DTYPE", "fp4")
    monkeypatch.setenv("MOE_REQUANTIZE_BLOCK_SIZE", "512")
    monkeypatch.delenv("VLLM_TPU_MOE_EARLY_REQUANT", raising=False)
    monkeypatch.setattr(model_loader_patches, "_STAGE_ON_CPU_FOR_TEST", True)

    monkeypatch.setattr(RoutedExperts, "weight_loader", RoutedExperts.weight_loader)
    monkeypatch.setattr(
        RoutedExperts, "_tpu_expert_staging_patch", False, raising=False
    )
    monkeypatch.setattr(
        base_loader,
        "process_weights_after_loading",
        lambda model, _mc, _td: [
            m.quant_method.process_weights_after_loading(m) for m in model
        ],
    )
    model_loader_patches.patch_moe_expert_write_staging()

    model = torch.nn.ModuleList(
        [
            _build_fake_fp8_moe_layer(
                monkeypatch, prefix=f"model.layers.{i}.mlp.experts"
            )
            for i in range(3, 6)
        ]
    )
    # Mirror vLLM's load_weights(): snapshot named_parameters() BEFORE streaming.
    params_dict = dict(model.named_parameters())
    assert all(p.data.numel() > 0 for p in params_dict.values())

    for idx, layer in enumerate(model):
        _stream_expert_shards_into_layer(layer, _make_random_moe_shards(seed=200 + idx))
        # Immediately after layer idx finishes streaming, every old Parameter
        # wrapper held in params_dict for layer idx must have 0-element .data!
        for name, old_p in params_dict.items():
            if name.startswith(f"{idx}."):
                assert old_p.data.numel() <= 1, f"{name} still pinned: {old_p.shape}"
        assert layer.w13_weight.dtype == torch.uint8
        assert layer.w13_weight.numel() > 0


def test_multichunk_fp4_requant_matches_single_chunk_bit_for_bit(monkeypatch):
    """Chunked packed-FP4 requantization (per_chunk < num_experts) produces
    bit-identical weights and scales to single-chunk requantization."""
    from vllm_torchtpu import model_loader_patches
    from vllm_torchtpu.layers.adapter.quantization import fp8 as fp8_mod

    if not hasattr(torch, "float4_e2m1fn_x2"):
        import pytest

        pytest.skip("torch.float4_e2m1fn_x2 unavailable")

    monkeypatch.setenv("MOE_REQUANTIZE_WEIGHT_DTYPE", "fp4")
    monkeypatch.setenv("MOE_REQUANTIZE_BLOCK_SIZE", "512")
    monkeypatch.setenv("USE_MOE_FUSED_EP_KERNEL", "0")
    monkeypatch.setattr(model_loader_patches, "_STAGE_ON_CPU_FOR_TEST", True)

    shards = _make_random_moe_shards(seed=321, num_experts=6)

    def run_with_chunk_bytes(chunk_bytes: int):
        monkeypatch.setattr(fp8_mod, "_REQUANT_CHUNK_BYTES", chunk_bytes)
        layer = _build_fake_fp8_moe_layer(monkeypatch, num_experts=6)
        for e in range(6):
            half = layer.w13_weight.shape[1] // 2
            layer.w13_weight.data[e, :half].copy_(shards[("w1", e)])
            layer.w13_weight.data[e, half:].copy_(shards[("w3", e)])
            layer.w2_weight.data[e].copy_(shards[("w2", e)])
            sh = layer.w13_weight_scale_inv.shape[1] // 2
            layer.w13_weight_scale_inv.data[e, :sh].copy_(shards[("s1", e)])
            layer.w13_weight_scale_inv.data[e, sh:].copy_(shards[("s3", e)])
            layer.w2_weight_scale_inv.data[e].copy_(shards[("s2", e)])
        assert layer.quant_method.requantize_moe_layer_weights(layer) is True
        return layer

    # 2 experts per chunk for w13 (2 * 512 * 512 * 4 B = 2 MiB of float32 per
    # expert) vs all 6 in one chunk.
    multi = run_with_chunk_bytes(2 * (2 * 512 * 512 * 4))
    single = run_with_chunk_bytes(1 << 30)
    for attr in (
        "w13_weight",
        "w2_weight",
        "w13_weight_scale_inv",
        "w2_weight_scale_inv",
    ):
        assert torch.equal(getattr(multi, attr).data, getattr(single, attr).data)


def test_release_param_storage_preserves_non_cpu_dispatch_key(monkeypatch):
    """variable.set_data rejects changing Tensor dispatch keys (e.g. TPU/Meta -> CPU);
    verify _release_param_storage keeps the parameter's device and dtype."""
    _install_cpu_torch_tpu_stub(monkeypatch)
    from vllm_torchtpu.layers.adapter.quantization.fp8 import _release_param_storage

    p = torch.nn.Parameter(
        torch.empty((32, 256, 512), dtype=torch.float8_e4m3fn, device="meta"),
        requires_grad=False,
    )
    _release_param_storage(p)
    assert p.device.type == "meta"
    assert p.dtype == torch.float8_e4m3fn
    assert p.numel() == 1
