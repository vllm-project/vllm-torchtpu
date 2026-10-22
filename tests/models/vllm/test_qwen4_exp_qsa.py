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
"""QSA checkpoint loading, producer rounding and native cache binding contracts."""

import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
import torch
from qwen4_exp_test_utils import (
    q4exp_text_config as attention_config,
)
from qwen4_exp_test_utils import (
    qsa_forward_context,
    qsa_test_pools,
)

from vllm_torchtpu.kernels.qwen4_exp.qsa import index_rope, main_rope
from vllm_torchtpu.models.vllm.qwen4_exp.qsa import (
    Qwen4ExpQSA,
    get_qsa_cache_specs,
    make_rope_cache,
)


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
        owners, layout = qsa_test_pools(model, 4)
        assert owners[0].shape[0] == owners[1].shape[0] == 4
        assert owners[1].dtype == torch.bfloat16
        assert owners[0].shape[1] == layout.block_size
        model.bind_pool(*owners)
        metadata = QSAMetadata(
            torch.empty(3, dtype=torch.int32),
            torch.empty(2, dtype=torch.int32),
            torch.empty(2, dtype=torch.int32),
            torch.empty(2, 2, dtype=torch.int32),
            torch.empty(2, 2, dtype=torch.int32),
        )
        qsa_forward_context(monkeypatch, model, metadata)
        output, selected = model.forward_with_indices(
            torch.empty(7, 128, dtype=torch.bfloat16),
            torch.empty(7, dtype=torch.int64),
            metadata,
        )
        assert output.shape == (7, 128)
        assert selected.shape == (7, 135)
        assert model.rope_cache.shape == (256, 128)
        specs = get_qsa_cache_specs(model.config, model.tp_size, model.cache_dtype)[1]
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
        owners, layout = qsa_test_pools(model, 4)
        model.bind_pool(*owners)
        metadata = QSAMetadata(
            torch.empty(3, dtype=torch.int32),
            torch.empty(2, dtype=torch.int32),
            torch.empty(2, dtype=torch.int32),
            torch.empty(2, 2, dtype=torch.int32),
            torch.empty(2, 2, dtype=torch.int32),
        )
        qsa_forward_context(monkeypatch, model, metadata)
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
        assert (
            module.get_attn_backend().get_supported_kernel_block_sizes()[0].base == 128
        )
        assert module.get_attn_backend().__name__ == "Qwen4ExpPooledQSABackend"
        assert (
            get_qsa_cache_specs(module.config, module.tp_size, module.cache_dtype)[1][
                "compressed_k"
            ].tokens_per_state
            == 4
        )
        with pytest.raises(ValueError, match="Missing"):
            module.finalize_weights()


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize(
    "factory_name,donated",
    [
        ("build_compact_pooled_qsa_op", (10, 11)),
    ],
)
def test_static_indexer_backend_cache_and_owner_contract(
    monkeypatch, factory_name, donated, fp8
):
    from tests.kernels.test_qwen4_exp_indexer_routing import ReferenceTopK
    from vllm_torchtpu.layers.adapter.custom_ops import qsa_op

    captured = []

    class RecordedOp:
        def register_fake(self, function):
            return function

        def __call__(self, *args, **kwargs):
            main, index = args[10:12]
            return (
                torch.full(main.shape, 2, dtype=torch.bfloat16).to(main.dtype),
                torch.full_like(index, 3),
                torch.zeros((1, 128), dtype=torch.bfloat16),
                torch.full((1, 67), -1, dtype=torch.int32),
            )

    def register(name, function, **kwargs):
        captured.append((name, function, kwargs))
        return RecordedOp()

    cached = getattr(qsa_op, "_" + factory_name)
    cached.cache_clear()
    try:
        monkeypatch.setattr(qsa_op.pallas, "jax_op", register)
        factory = getattr(qsa_op, factory_name)
        assert factory() is factory(topk_backend=None)
        backend = ReferenceTopK()
        first = factory(topk_backend=backend, topk_backend_key="reference-v1/forward")
        assert (
            factory(
                topk_backend=ReferenceTopK(), topk_backend_key="reference-v1/forward"
            )
            is first
        )
        assert (
            factory(
                topk_backend=ReferenceTopK(reverse=True),
                topk_backend_key="reference-v1/reverse",
            )
            is not first
        )
        main = torch.zeros(
            (3, 128, 1, 4 if fp8 else 2, 256),
            dtype=torch.float8_e4m3fn if fp8 else torch.bfloat16,
        )
        index = torch.zeros((3, 16, 1, 2, 128), dtype=torch.bfloat16)
        aliases = (main, index)
        storages = [owner.untyped_storage()._cdata for owner in aliases]
        producer = [torch.zeros((1, 256), dtype=torch.bfloat16) for _ in range(10)]
        metadata = [torch.zeros((1,), dtype=torch.int64) for _ in range(5)]
        factory()(
            *producer,
            main,
            index,
            *metadata,
            compress_ratio=4,
            token_topk=64,
            norm_eps=1e-6,
            rotary_dim=64,
        )
        assert main is aliases[0] and index is aliases[1]
        assert [owner.untyped_storage()._cdata for owner in aliases] == storages
        assert torch.all(main.float() == 2) and torch.all(index == 3)
        assert index.dtype == torch.bfloat16
        assert len(captured) == 3
        assert len({item[0] for item in captured}) == 3
        assert captured[1][1].keywords["topk_backend"] is backend
        assert all(item[2]["donate_argnums"] == donated for item in captured)
        with pytest.raises(TypeError, match="unhashable"):
            factory(topk_backend={})
        with pytest.raises(ValueError, match="requires a stable"):
            factory(topk_backend=backend)
        with pytest.raises(ValueError, match="different backend"):
            factory(
                topk_backend=ReferenceTopK(reverse=True),
                topk_backend_key="reference-v1/forward",
            )
        with pytest.raises(ValueError, match="requires an explicit"):
            factory(topk_backend_key="orphan-key")
        assert captured[0][0] == "qwen4_exp::" + factory_name[6:-3]
        assert len(captured) == 3
    finally:
        cached.cache_clear()


def test_actual_model_keeps_static_backend_when_binding_compact_owners(monkeypatch):
    import gc
    import weakref

    from tests.kernels.test_qwen4_exp_indexer_routing import ReferenceTopK
    from vllm_torchtpu.models.vllm.qwen4_exp import qsa

    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1
    )
    seen = []
    backend = ReferenceTopK()

    def factory(**kwargs):
        seen.append((kwargs["topk_backend"], kwargs["topk_backend_key"]))
        return lambda *args, **kw: None

    monkeypatch.setattr(qsa, "build_compact_pooled_qsa_op", factory)
    layer = Qwen4ExpQSA(
        config=attention_config(hidden_size=16),
        tp_size=1,
        tp_rank=0,
        max_model_len=128,
        topk_backend=backend,
        topk_backend_key="reference-v1/forward",
    )
    main = torch.zeros((2, 128, 1, 2, 256), dtype=torch.bfloat16)
    index = torch.zeros((2, 16, 1, 2, 128), dtype=torch.bfloat16)
    with pytest.raises(TypeError):
        layer.bind_pool(main)
    layer.bind_pool(main, index)
    assert seen == [(backend, "reference-v1/forward")]
    assert layer.cache_owners == (main, index)
    assert "topk_backend" not in layer._op.keywords
    assert "topk_backend_key" not in layer._op.keywords
    ref = weakref.ref(layer)
    operation = layer._op
    del layer
    gc.collect()
    assert ref() is None
    assert operation is not None


def test_compact_selector_forwards_logical_ids_and_existing_resource_budget(
    monkeypatch,
):
    import jax

    from tests.kernels.test_qwen4_exp_indexer_routing import ReferenceTopK
    from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa

    seen = []
    backend = ReferenceTopK()

    def select(query, cache, lengths, table, req, **kwargs):
        seen.append((table, kwargs))
        return jnp.full((len(query), 16), -1, jnp.int32)

    monkeypatch.setattr(pooled_qsa, "select_index_blocks", select)
    with jax.disable_jit():
        pooled_qsa.select_blocks(
            jnp.ones((4, 1, 128), jnp.bfloat16),
            jnp.zeros((12, 32, 1, 2, 128), jnp.bfloat16),
            jnp.array([4, 5, 0, 0], jnp.int32),
            jnp.array([[3, 4], [0, 0]], jnp.int32),
            jnp.array([0, 0, 1, 1], jnp.int32),
            jnp.array([0, 2, 2], jnp.int32),
            jnp.array([3, 0], jnp.int32),
            ratio=4,
            block_topk=16,
            logical_block_size=256,
            vmem_limit_bytes=123456,
            topk_backend=backend,
        )
    np.testing.assert_array_equal(seen[0][0], [[6, 7, 8, 9], [0, 1, 0, 1]])
    assert seen[0][1]["topk_backend"] is backend
    assert seen[0][1]["vmem_limit_bytes"] == 123456


@torch.inference_mode()
def test_output_projection_tp2_partial_sum_matches_tp1(monkeypatch):
    config = attention_config(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        indexer_n_heads=2,
        indexer_head_dim=128,
    )
    checkpoint = torch.zeros(32, 512, dtype=torch.bfloat16)
    rows = torch.arange(32)
    checkpoint[rows, rows] = 0.5
    checkpoint[rows, rows + 256] = 0.25
    attended = ((torch.arange(3 * 512).reshape(3, 512) % 7) / 4).to(torch.bfloat16)
    partials = []
    for size, rank in [(1, 0), (2, 0), (2, 1)]:
        monkeypatch.setattr(
            "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
            lambda rank=rank: rank,
        )
        monkeypatch.setattr(
            "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
            lambda size=size: size,
        )
        layer = Qwen4ExpQSA(
            config=config, tp_rank=rank, tp_size=size, max_model_len=128, device="cpu"
        )
        layer.load_weights([("o_proj.weight", checkpoint)])
        width = attended.shape[-1] // size
        output = layer.o_proj(attended[:, rank * width : (rank + 1) * width])
        if size == 1:
            expected = output
        else:
            partials.append(output)
    # Binary-exact inputs isolate rank slicing/reduction from BF16 sum rounding.
    torch.testing.assert_close(partials[0] + partials[1], expected, rtol=0, atol=0)
