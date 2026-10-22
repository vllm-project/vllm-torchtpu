# SPDX-License-Identifier: Apache-2.0
"""QSA checkpoint loading, producer rounding and native cache binding contracts."""

import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
import torch
from qwen4_exp_test_utils import q4exp_text_config as attention_config

from vllm_torchtpu.kernels.qwen4_exp.qsa import index_rope, main_rope
from vllm_torchtpu.models.vllm.qwen4_exp.qsa import Qwen4ExpQSA, make_rope_cache


@pytest.fixture(autouse=True)
def cpu_vllm_config_context():
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    resolve_kv_cache_layout(config, [["LBNHC", "LBHNC"]])
    with set_current_vllm_config(config):
        yield


@pytest.mark.parametrize("rotary_dim", [32, 64, 128, 192])
def test_flat_rope_rows_preserve_logical_cos_sin_and_padding(rotary_dim):
    from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding

    positions = 33
    cache = make_rope_cache(positions, rotary_dim=rotary_dim, theta=10000000.0)
    assert cache.shape == (positions, (rotary_dim + 127) // 128 * 128)
    assert cache.dtype == torch.bfloat16
    with torch.device("cpu"):
        expected = RotaryEmbedding(
            head_size=rotary_dim,
            rotary_dim=rotary_dim,
            max_position_embeddings=positions,
            base=10000000.0,
            is_neox_style=True,
            dtype=torch.bfloat16,
        ).cos_sin_cache
    torch.testing.assert_close(cache[:, :rotary_dim], expected, rtol=0, atol=0)
    assert torch.count_nonzero(cache[:, rotary_dim:]) == 0


@pytest.mark.parametrize(
    "k_name,v_name",
    [
        ("k_proj.k_scale", "v_proj.v_scale"),
        ("k_proj.output_scale", "v_proj.output_scale"),
        ("attn.k_scale", "attn.v_scale"),
        ("attn._k_scale", "attn._v_scale"),
        ("k_scale", "v_scale"),
        ("_k_scale", "_v_scale"),
    ],
)
def test_cuda_checkpoint_scale_aliases_keep_write_and_read_scales_consistent(
    k_name, v_name, monkeypatch
):
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1
    )
    with torch.device("cpu"):
        model = Qwen4ExpQSA(
            config=attention_config(hidden_size=16),
            tp_size=1,
            tp_rank=0,
            max_model_len=128,
            cache_dtype="fp8_e4m3",
        )
        assert model._k_scale.item() == model._k_scale_float == 1.0
        assert model._v_scale.item() == model._v_scale_float == 1.0
        for bad in (torch.ones(2), torch.tensor(0.0), torch.tensor(float("nan"))):
            with pytest.raises(ValueError, match="per-tensor|finite positive"):
                model.load_weights([(k_name, bad)])
        loaded = model.load_weights(
            [(k_name, torch.tensor(0.375)), (v_name, torch.tensor(2.5))]
        )
        assert loaded == {"_k_scale", "_v_scale"}
        assert model._k_scale.item() == model._k_scale_float == 0.375
        assert model._v_scale.item() == model._v_scale_float == 2.5
        assert model._k_scale.dtype == model._v_scale.dtype == torch.float32
        with pytest.raises(ValueError, match="Duplicate"):
            model.load_weights([("_k_scale", torch.tensor(1.0))])


@pytest.mark.parametrize("cache_dtype", ["auto", "fp8", "fp8_e4m3"])
def test_alternative_config_drives_checkpoint_cache_and_custom_op(
    cache_dtype, monkeypatch
):
    from torch._subclasses.fake_tensor import FakeTensorMode

    from vllm_torchtpu.models.vllm.qwen4_exp.qsa import QSAMetadata

    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1
    )
    config = attention_config(
        hidden_size=128,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=64,
        indexer_n_heads=2,
        indexer_head_dim=32,
        indexer_budget=128,
        indexer_compress_ratio=8,
        rms_norm_eps=0.01,
        partial_rotary_factor=0.5,
        rope_parameters={"rope_theta": 12345.0},
        max_position_embeddings=256,
    )
    with torch.device("cpu"):
        model = Qwen4ExpQSA(
            config=config,
            tp_size=1,
            tp_rank=0,
            max_model_len=256,
            cache_dtype=cache_dtype,
        )
        model.load_weights(
            [
                ("q_proj.weight", torch.empty(1024, 128, dtype=torch.bfloat16)),
                ("k_proj.weight", torch.empty(256, 128, dtype=torch.bfloat16)),
                ("v_proj.weight", torch.empty(256, 128, dtype=torch.bfloat16)),
                ("o_proj.weight", torch.empty(128, 512, dtype=torch.bfloat16)),
                (
                    "indexer.index_qk_proj.weight",
                    torch.empty(96, 128, dtype=torch.bfloat16),
                ),
            ]
        )
    with torch.device("cpu"), FakeTensorMode(allow_non_fake_inputs=True):
        owners = [
            torch.full(shape, fill, dtype=dtype)
            for _, shape, dtype, fill in model.get_cache_owner_layouts(4, 3)
        ]
        auxiliary_shapes = [(4, 16, 1, 128), (3, 8, 32), (3, 8, 2)]
        expected_shapes = (
            [(4, 128, 4, 2, 128)] + auxiliary_shapes
            if cache_dtype == "auto"
            else [(4, 128, 2, 4, 128)] + auxiliary_shapes
        )
        assert [tuple(owner.shape) for owner in owners] == expected_shapes
        model.bind_cache_owners(owners)
        metadata = QSAMetadata(
            torch.empty(3, dtype=torch.int32),
            torch.empty(2, dtype=torch.int32),
            torch.empty(2, dtype=torch.int32),
            torch.empty(2, 2, dtype=torch.int32),
            torch.empty(2, 2, dtype=torch.int32),
        )
        output, selected = model.forward_with_indices(
            torch.empty(7, 128, dtype=torch.bfloat16),
            torch.empty(7, dtype=torch.int64),
            metadata,
        )
        assert output.shape == (7, 128)
        assert selected.shape == (7, 135)
        assert model.rope_cache.shape == (256, 128)
        specs = model.get_auxiliary_cache_specs()
        assert specs["compressed_k"].tokens_per_state == 8
        assert specs["compressed_k"].head_size == 32
        assert specs["raw_ring"].block_size == 8
        main = model.get_kv_cache_spec(None)
        assert main.head_size == 64 and main.num_kv_heads == 4
        if cache_dtype != "auto":
            from vllm.v1.kv_cache_interface import KVQuantMode

            assert main.kv_quant_mode == KVQuantMode.FP8_PER_TENSOR
            assert main.page_size_bytes == 128 * 4 * 128 * 2


def bf(x):
    return np.asarray(x, ml_dtypes.bfloat16).astype(np.float32)


def norm(x, w, eps=1e-6):
    x = np.asarray(x, np.float32)
    return bf(
        x
        / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
        * (1 + np.asarray(w, np.float32))
    )


def rope_oracle(x, positions, cache, index=True):
    cs = np.asarray(cache, np.float32)[positions]
    half = cache.shape[-1]
    shape = (len(x),) + (1,) * (x.ndim - 2) + (half,)
    c, s = cs[:, 0].reshape(shape), cs[:, 1].reshape(shape)
    a, b = x[..., :half], x[..., half : 2 * half]
    if index:
        left = bf(
            np.subtract(
                np.multiply(a, c, dtype=np.float32), bf(b * s), dtype=np.float32
            )
        )
        right = bf(
            np.add(np.multiply(a, s, dtype=np.float32), bf(b * c), dtype=np.float32)
        )
    else:
        left, right = bf(a * c - b * s), bf(b * c + a * s)
    return np.concatenate([left, right, x[..., 2 * half :]], -1)


def compress_oracle(raw, positions, weight, trig):
    rows = []
    for begin in range(0, len(raw) - 3, 4):
        pooled = bf(np.mean(raw[begin : begin + 4], axis=0, dtype=np.float32))
        rows.append(
            rope_oracle(norm(pooled[None], weight), positions[begin : begin + 1], trig)[
                0
            ]
        )
    return np.asarray(rows, np.float32).reshape(-1, 128)


def test_independent_rope_contracts_and_compression_order():
    rng = np.random.default_rng(33)
    raw = bf(rng.normal(size=(12, 128)))
    weight = bf(rng.normal(size=128) * 0.2)
    pos = np.arange(97, 109)
    stored = make_rope_cache(128, rotary_dim=64, theta=10000000.0).float().numpy()
    trig = stored[:, :64].reshape(128, 2, 32)
    normalized = norm(raw, weight)
    args = (
        jnp.asarray(normalized, jnp.bfloat16),
        jnp.asarray(pos),
        jnp.asarray(stored, jnp.bfloat16),
        64,
    )
    np.testing.assert_array_equal(
        np.asarray(index_rope(*args)).astype(np.float32),
        rope_oracle(normalized, pos, trig),
    )
    np.testing.assert_array_equal(
        np.asarray(main_rope(*args)).astype(np.float32),
        rope_oracle(normalized, pos, trig, False),
    )
    expected = compress_oracle(raw, pos, weight, trig)
    wrong = bf(rope_oracle(normalized, pos, trig).reshape(3, 4, 128).mean(1))
    assert np.max(np.abs(expected - wrong)) > 0.1


@pytest.mark.parametrize("dtype", ["auto", "fp8", "fp8_e4m3"])
def test_model_cache_binding_reaches_real_custom_op_schema(dtype, monkeypatch):
    # Kernel-only tests cannot detect the model passing the wrong number or
    # order of persistent owners. Fake dispatch checks the real op schema.
    from torch._subclasses.fake_tensor import FakeTensorMode

    from vllm_torchtpu.models.vllm.qwen4_exp.qsa import QSAMetadata

    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1
    )
    with torch.device("cpu"), FakeTensorMode():
        model = Qwen4ExpQSA(
            config=attention_config(),
            tp_size=1,
            tp_rank=0,
            max_model_len=8192,
            cache_dtype=dtype,
        )
        auxiliary = [
            torch.empty(4, 32, 1, 128, dtype=torch.bfloat16),
            torch.empty(3, 8, 128, dtype=torch.bfloat16),
            torch.empty(3, 8, 2, dtype=torch.int32),
        ]
        if dtype == "auto":
            owners = [
                torch.empty(4, 128, 2, 2, 256, dtype=torch.bfloat16),
                *auxiliary,
            ]
        else:
            owners = [
                torch.empty(4, 128, 1, 4, 256, dtype=torch.float8_e4m3fn),
                *auxiliary,
            ]
        model.bind_cache_owners(owners)
        metadata = QSAMetadata(
            torch.empty(3, dtype=torch.int32),
            torch.empty(2, dtype=torch.int32),
            torch.empty(2, dtype=torch.int32),
            torch.empty(2, 2, dtype=torch.int32),
            torch.empty(2, 2, dtype=torch.int32),
        )
        output, selected = model.forward_with_indices(
            torch.empty(7, 2560, dtype=torch.bfloat16),
            torch.empty(7, dtype=torch.int64),
            metadata,
        )
        assert output.shape == (7, 2560) and output.dtype == torch.bfloat16
        assert selected.shape == (7, 2051)


@pytest.mark.parametrize(
    "tp,rank", [(1, 0), (2, 1), (4, 1), (4, 2), (8, 3), (8, 4), (8, 7)]
)
def test_checkpoint_head_packing_and_kv_replication(tp, rank, monkeypatch):
    # Only replace distributed environment discovery; real upstream loaders
    # still perform every tested slice and copy. No CPU process group needed.
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: rank
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: tp
    )
    with torch.device("cpu"):
        module = Qwen4ExpQSA(
            config=attention_config(), tp_size=tp, tp_rank=rank, max_model_len=8192
        )
        q = (
            torch.arange(12288, dtype=torch.float32)[:, None]
            .expand(-1, 2560)
            .to(torch.bfloat16)
        )
        k = (
            torch.arange(512, dtype=torch.float32)[:, None]
            .expand(-1, 2560)
            .to(torch.bfloat16)
        )
        index = (
            torch.arange(640, dtype=torch.float32)[:, None]
            .expand(-1, 2560)
            .to(torch.bfloat16)
        )
        o = (
            torch.arange(6144, dtype=torch.float32)[None]
            .expand(2560, -1)
            .to(torch.bfloat16)
        )
        module.load_weights(
            [
                ("q_proj.weight", q),
                ("k_proj.weight", k),
                ("v_proj.weight", k),
                ("indexer.index_qk_proj.weight", index),
                ("o_proj.weight", o),
            ]
        )
        width = 12288 // tp
        torch.testing.assert_close(
            module.q_proj.weight, q[rank * width : (rank + 1) * width], rtol=0, atol=0
        )
        kv_rank = rank // max(tp // 2, 1)
        kw = max(512 // tp, 256)
        torch.testing.assert_close(
            module.k_proj.weight, k[kv_rank * kw : (kv_rank + 1) * kw], rtol=0, atol=0
        )
        torch.testing.assert_close(
            module.indexer.index_qk_proj.weight, index, rtol=0, atol=0
        )
        torch.testing.assert_close(
            module.o_proj.weight,
            o[:, rank * width // 2 : (rank + 1) * width // 2],
            rtol=0,
            atol=0,
        )
        assert module.get_attn_backend().get_min_page_size(None) == 128
        assert module.get_attn_backend().is_sparse()
        assert module.get_auxiliary_cache_specs()["compressed_k"].tokens_per_state == 4
        with pytest.raises(ValueError, match="Missing"):
            module.finalize_weights()


@pytest.mark.parametrize(
    "updates,tp,match",
    [
        ({}, 0, "positive tp_size"),
        ({}, -1, "positive tp_size"),
        ({"num_attention_heads": 0}, 1, "num_attention_heads"),
        ({"num_attention_heads": 7}, 2, "num_attention_heads"),
        ({"num_key_value_heads": 0}, 1, "num_key_value_heads"),
        ({"num_key_value_heads": 3}, 2, "num_key_value_heads"),
        ({"indexer_kv_heads": 2}, 1, "indexer_kv_heads"),
        ({"head_dim": 0}, 1, "head_dim"),
        ({"indexer_head_dim": 0}, 1, "indexer_head_dim"),
        ({"indexer_n_heads": 0}, 1, "indexer_n_heads"),
    ],
)
def test_invalid_geometry_fails_before_projection_allocation(
    updates, tp, match, monkeypatch
):
    import vllm_torchtpu.models.vllm.qwen4_exp.qsa as module

    def forbidden(*args, **kwargs):
        pytest.fail("invalid geometry reached weight allocation")

    monkeypatch.setattr(module, "ColumnParallelLinear", forbidden)
    monkeypatch.setattr(module, "ReplicatedLinear", forbidden)
    with pytest.raises(ValueError, match=match):
        Qwen4ExpQSA(
            config=attention_config(hidden_size=16, **updates),
            tp_rank=0,
            tp_size=tp,
            max_model_len=128,
            device="cpu",
        )


@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
def test_actual_native_factory_fake_dtype_and_fullgraph(position_dtype):
    from torch._subclasses.fake_tensor import FakeTensorMode

    from vllm_torchtpu.layers.adapter.custom_ops.qsa_op import build_qsa_op

    build_qsa_op()
    # Call the registered native op directly, retaining the original position
    # dtype rather than relying on the wrapper's normalization to int32.
    op = torch.ops.qwen4_exp.qsa
    graphs = []

    def record(gm, inputs):
        graphs.append(gm.code)
        return gm.forward

    with FakeTensorMode():

        def bf(*shape):
            return torch.empty(shape, dtype=torch.bfloat16)

        def ii(*shape):
            return torch.empty(shape, dtype=torch.int32)

        args = (
            bf(4, 512),
            bf(4, 128),
            bf(4, 128),
            bf(4, 384),
            bf(128),
            bf(128),
            bf(128),
            bf(128),
            torch.empty(4, dtype=position_dtype),
            bf(128, 128),
            bf(4, 128, 1, 2, 128),
            bf(4, 32, 1, 128),
            bf(3, 8, 128),
            ii(3, 8, 2),
            ii(3),
            ii(2),
            ii(2),
            ii(2, 2),
            ii(2, 2),
            4,
            64,
            1e-6,
            32,
        )
        for fn in (
            op,
            torch.compile(op, backend=record, fullgraph=True, dynamic=False),
        ):
            result = fn(*args)
            assert result[-1].shape == (4, 67)
            assert result[-1].dtype == torch.int32
            assert result[-2].shape == (4, 256)
            assert result[3].shape == (3, 8, 2) and result[3].dtype == torch.int32
    assert len(graphs) == 1


def test_native_tag_planes_are_donated_and_written_to_the_same_owner(monkeypatch):
    from vllm_torchtpu.layers.adapter.custom_ops import qsa_op

    calls = []

    class RecordingOp:
        def register_fake(self, fn):
            return fn

        def __call__(self, *args):
            calls.append(args)
            planes = args[13].clone()
            # Include signed/unsigned low-plane boundaries and independent
            # high bits. This tests transport, not a claim of int64 positions.
            planes[0, 0] = torch.tensor([-2147483648, 2147483647])
            planes[0, 1] = torch.tensor([2147483647, -2147483648])
            return (
                *args[10:13],
                planes,
                args[0].new_zeros((4, 256)),
                args[8].new_full((4, 67), -1),
            )

    def factory(name, fn, *, donate_argnums):
        assert donate_argnums == (10, 11, 12, 13)
        return RecordingOp()

    monkeypatch.setattr(qsa_op.pallas, "jax_op", factory)
    qsa_op.build_qsa_op.cache_clear()
    try:
        op = qsa_op.build_qsa_op()
        tags = torch.full((3, 8, 2), -1, dtype=torch.int32)
        owners = (
            torch.zeros(4, 128, 1, 2, 128, dtype=torch.bfloat16),
            torch.zeros(4, 32, 1, 128, dtype=torch.bfloat16),
            torch.zeros(3, 8, 128, dtype=torch.bfloat16),
            tags,
        )
        identities = [(id(x), x.data_ptr()) for x in owners]

        def bf(*shape):
            return torch.zeros(shape, dtype=torch.bfloat16)

        def ii(*shape):
            return torch.zeros(shape, dtype=torch.int32)

        args = (
            bf(4, 512),
            bf(4, 128),
            bf(4, 128),
            bf(4, 384),
            bf(128),
            bf(128),
            bf(128),
            bf(128),
            ii(4),
            bf(128, 128),
            *owners,
            ii(3),
            ii(2),
            ii(2),
            ii(2, 2),
            ii(2, 2),
            4,
            64,
            1e-6,
            32,
        )
        for _ in range(2):
            op(*args)
            assert calls[-1][13] is tags
            assert identities == [(id(x), x.data_ptr()) for x in owners]
            torch.testing.assert_close(
                tags[0, :2],
                torch.tensor(
                    [[-2147483648, 2147483647], [2147483647, -2147483648]],
                    dtype=torch.int32,
                ),
                rtol=0,
                atol=0,
            )
            assert torch.all(tags[1:] == -1)
            tags.fill_(-1)  # external slot reset/reuse preserves both planes
            assert torch.all(tags == -1)
    finally:
        qsa_op.build_qsa_op.cache_clear()
