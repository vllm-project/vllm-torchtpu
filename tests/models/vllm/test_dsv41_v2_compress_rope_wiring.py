# SPDX-License-Identifier: Apache-2.0
"""CPU contracts; real JAX bridges, stubbed device kernel and Torch registration."""

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention as v2_attn
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import compress_rope_op as bridge
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import csa2_adapter
from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import PagedCacheRows


def _cache(width, dtype, *, banks=1, scale=False):
    owners = tuple(torch.zeros(256, 128, dtype=torch.int32) for _ in range(banks))
    scales = tuple(torch.zeros(256) for _ in range(banks)) if scale else None
    # Nonzero offset and padded page stride exercise physical record mapping.
    size = width * dtype.itemsize
    return PagedCacheRows(
        owners[0] if banks == 1 else owners,
        ((4, 8, width), dtype, 16 * size, 2 * size),
        scale=scales[0] if scale and banks == 1 else scales,
    )


def _fixture(monkeypatch, ratio, banks, replay):
    attn = object.__new__(v2_attn.DeepseekV41Attention)
    torch.nn.Module.__init__(attn)
    attn.compress_ratio = ratio
    attn._compress_rope_enabled = True
    attn.prefix = "main"
    attn.window_size = 128
    attn._config = {"index_topk": 1}
    attn._max_model_len = 1024
    attn.hidden_size = 16
    attn._tpu_rope_cache = torch.ones(64, 64)
    attn._rotate = lambda values, positions: values
    attn._quantize_indexer = lambda values: values
    state = _cache(1024, torch.float32, banks=banks) if ratio == 2 else None
    main = _cache(512, torch.uint8, banks=banks, scale=True)
    index = _cache(128, torch.bfloat16, banks=banks)
    attn.kv_cache = main
    attn.compressor = SimpleNamespace(
        compress_ratio=ratio,
        rms_norm_eps=1e-5,
        fused_wkv_wgate=SimpleNamespace(
            weight=torch.zeros(16, ratio * 512, dtype=torch.bfloat16)
        ),
        norm=SimpleNamespace(weight=torch.ones(512, dtype=torch.bfloat16)),
        state_cache=SimpleNamespace(prefix="state", kv_cache=state),
        project=MagicMock(side_effect=AssertionError("unfused projection")),
    )
    attn.indexer = SimpleNamespace(
        k_cache=SimpleNamespace(prefix="index", compress_ratio=ratio, kv_cache=index),
        wk=lambda values: values,
        k_norm=lambda values: values,
    )
    # Two real requests and repeated padded boundary; positions are not row IDs.
    positions = torch.tensor([9, 10, 11, 20, 21, 0], dtype=torch.int32)
    metadata = {
        key: SimpleNamespace(
            seq_lens=torch.tensor([12, 22, 0]),
            query_start_loc=torch.tensor([0, 3, 5, 5], dtype=torch.int32),
            block_tables=torch.tensor([[0, 0, 0], [1, 1, 1], [0, 0, 0]]),
        )
        for key in ("main", "index", "state")
    }
    ctx = SimpleNamespace(
        attn_metadata=metadata,
        additional_kwargs={"deepseek_v41_window_starts": torch.tensor([9, 0, 0])}
        if replay
        else {},
    )
    monkeypatch.setattr(v2_attn, "get_forward_context", lambda: ctx)
    monkeypatch.setattr(
        v2_attn, "get_rope_rows_op", lambda: lambda cache, pos: cache[pos]
    )
    return attn, positions, state, main


@pytest.mark.parametrize("ratio,banks", [(1, 1), (1, 2), (2, 1), (2, 2)])
@pytest.mark.parametrize("defer", [False, True])
@pytest.mark.parametrize("replay", [False, True])
def test_fused_compressor_wiring(monkeypatch, ratio, banks, defer, replay):
    attn, positions, state, main = _fixture(monkeypatch, ratio, banks, replay)
    captured = {}
    kernel_banks = banks if ratio == 2 else 0

    def kernel(x, weight, gamma, rows, hp, valid, pool, write, read, *, ratio, eps):
        assert hp.shape == valid.shape == (6, 1)
        assert hp.dtype == valid.dtype == jnp.int32
        assert eps == 1e-5
        assert weight.shape == (16, ratio * 512)
        captured.update(hp=np.asarray(hp), valid=np.asarray(valid))
        if ratio == 2:
            captured.update(write=np.asarray(write), read=np.asarray(read))
            pools = pool if isinstance(pool, tuple) else (pool,)
            assert len(pools) == kernel_banks
            updated = tuple(p + i + 7 for i, p in enumerate(pools))
            pool = updated if isinstance(pool, tuple) else updated[0]
        return (
            jnp.ones((6, 512), jnp.bfloat16),
            jnp.zeros((6, 128), jnp.int32),
            jnp.ones((6, 1), jnp.float32),
            pool,
        )

    monkeypatch.setattr(bridge.compress_rope, "fused_compress_rope", kernel)

    def get_op(ratio, banks, *, eps):
        fn = bridge._bridge(ratio, banks, eps)

        def call(*args):
            arrays = [
                jnp.asarray(a.float().numpy(), jnp.bfloat16)
                if a.dtype == torch.bfloat16
                else jnp.asarray(a.numpy())
                for a in args
            ]
            result = fn(*arrays)
            return tuple(
                torch.from_numpy(np.array(a, dtype=np.float32)).bfloat16()
                if a.dtype == jnp.bfloat16
                else torch.from_numpy(np.array(a))
                for a in result
            )

        return call

    monkeypatch.setattr(v2_attn, "get_compress_rope_op", get_op)
    publish = MagicMock()
    insert = MagicMock()
    monkeypatch.setattr(csa2_adapter, "publish_main_encoded", publish)
    monkeypatch.setattr(v2_attn, "_insert", insert)
    with v2_attn.stage_attention_cache_scope():
        result = attn._fused_compress_global_kv(
            torch.zeros(6, 16, dtype=torch.bfloat16),
            positions,
            defer_index_write=defer,
        )
        assert (
            v2_attn._active_stage_attn_cache[("candidate_index_version", "index")] == 1
        )
    valid = np.array([True, ratio == 1, True, ratio == 1, True, False])
    if replay:
        valid[:3] = False
    np.testing.assert_array_equal(captured["valid"][:, 0], valid)
    expected_slots = (positions // ratio) % 8 + torch.tensor([0, 0, 0, 8, 8, 0])
    expected = main.row_indices(expected_slots).masked_fill(~torch.tensor(valid), -1)
    assert torch.equal(publish.call_args.args[1], expected)
    if ratio == 2:
        np.testing.assert_array_equal(captured["hp"][:, 0], [0, 1, 1, 0, 1, 0])
        np.testing.assert_array_equal(captured["write"], [3, 4, 5, 22, 23, -1])
        np.testing.assert_array_equal(captured["read"], [2, -1, -1, 21, -1, -1])
        assert torch.all(state.raw == 7)
        if banks == 2:
            assert torch.all(state.raw_hi == 8)
    if defer:
        insert.assert_not_called()
        assert torch.equal(
            result[0],
            attn.indexer.k_cache.kv_cache.row_indices(expected_slots).masked_fill(
                ~torch.tensor(valid), -1
            ),
        )
    else:
        assert result is None
        insert.assert_called_once()
    attn.compressor.project.assert_not_called()


@pytest.mark.parametrize("ratio,banks", [(1, 0), (2, 1), (2, 2)])
def test_empty_real_bridge_and_fake_contract(ratio, banks):
    fn = bridge._bridge(ratio, banks)
    shapes = [(0, 16), (16, ratio * 512), (512,), (0, 64), (0, 1), (0, 1)]
    dtypes = [
        jnp.bfloat16,
        jnp.bfloat16,
        jnp.bfloat16,
        jnp.float32,
        jnp.int32,
        jnp.int32,
    ]
    if banks:
        shapes += [(64, 128)] * banks + [(0,), (0,)]
        dtypes += [jnp.int32] * (banks + 2)
    args = [jnp.zeros(shape, dtype) for shape, dtype in zip(shapes, dtypes)]
    actual = fn(*args)
    abstract = jax.eval_shape(fn, *args)
    torch_args = [
        torch.zeros(
            shape,
            dtype=torch.bfloat16
            if dtype == jnp.bfloat16
            else torch.int32
            if dtype == jnp.int32
            else torch.float32,
        )
        for shape, dtype in zip(shapes, dtypes)
    ]
    fake = bridge._fake(ratio, banks)(*torch_args)
    assert len(inspect.signature(fn).parameters) == len(args)
    assert len(actual) == len(fake) == 3 + banks
    for real, traced, expected in zip(actual, abstract, fake):
        assert real.shape == traced.shape == tuple(expected.shape)
        assert str(real.dtype) == str(expected.dtype).removeprefix("torch.")
    for i in range(banks):
        assert actual[3 + i] is args[6 + i]


def test_reject_invalid_bank_mode_before_registration():
    with pytest.raises(ValueError, match="one/two banks"):
        bridge.get_compress_rope_op(2, 0)


def test_group_and_token_rows_share_independent_stage_entries(monkeypatch):
    attn, positions, _, _ = _fixture(monkeypatch, 2, 1, False)
    gather = MagicMock(side_effect=lambda cache, pos: cache[pos])
    monkeypatch.setattr(v2_attn, "get_rope_rows_op", lambda: gather)
    with v2_attn.stage_attention_cache_scope():
        token = attn._rope_rows(positions)
        grouped = attn._group_positions(positions)
        group = attn._rope_rows(grouped, role="group")
        assert attn._group_positions(positions) is grouped
        assert attn._rope_rows(positions) is token
        assert attn._rope_rows(grouped, role="group") is group
        assert gather.call_count == 2


def test_register_only_matching_signatures_and_donations(monkeypatch):
    registered = []

    def register(name, fn, **kwargs):
        count = len(inspect.signature(fn).parameters)
        assert count == len(kwargs["input_partition_specs"])
        registered.append((name, count, kwargs["donate_argnums"]))
        return SimpleNamespace(register_fake=lambda fake: None)

    monkeypatch.setattr(bridge, "_ops", {})
    monkeypatch.setattr(bridge, "get_pallas_op_mesh", lambda: None)
    monkeypatch.setattr(bridge.pallas, "jax_op", register)
    for ratio, banks in ((1, 0), (2, 1), (2, 2)):
        bridge.get_compress_rope_op(ratio, banks, eps=1e-5)
    assert [entry[1:] for entry in registered] == [(6, ()), (9, (6,)), (10, (6, 7))]
    with pytest.raises(ValueError):
        bridge.get_compress_rope_op(2)


def test_new_preprocessing_flag_defaults_off(monkeypatch):
    from vllm_torchtpu import envs

    monkeypatch.delenv("VLLM_TPU_FUSED_DSV41_ATTENTION_PREPROCESS", raising=False)
    assert not envs.environment_variables["VLLM_TPU_FUSED_DSV41_ATTENTION_PREPROCESS"]()


def test_legacy_qb_fallback_does_not_enter_new_kernel(monkeypatch):
    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import projection_rope as module

    monkeypatch.setattr(module, "is_supported", lambda *args: False)
    fused = MagicMock(side_effect=AssertionError("new path without shared rows"))
    monkeypatch.setattr(module.qb_rope, "fused_qb_rope", fused)
    monkeypatch.setattr(
        module, "xla_quantized_matmul", lambda *a: jnp.zeros((1, 32768))
    )
    monkeypatch.setattr(module, "rope", lambda x, *a: x)
    assert module.projection_rope(
        jnp.zeros((1, 1280)), None, None, None, None
    ).shape == (1, 64, 512)
    fused.assert_not_called()


@pytest.mark.parametrize("banks", [1, 2])
def test_encoded_publication_preserves_banks_without_encoding(monkeypatch, banks):
    from vllm_torchtpu.kernels.deepseek_v41.csa2 import cache as kernels

    encoded = MagicMock(side_effect=AssertionError("encoded twice"))
    monkeypatch.setattr(kernels, "encode_csa2_kv", encoded)
    pool = jnp.zeros((8, 128), jnp.int32)
    scales = jnp.zeros((8,), jnp.float32)
    if banks == 2:
        pool = (pool, pool)
        scales = (scales, scales)
    result = kernels.publish_csa2_encoded.__wrapped__(
        pool,
        scales,
        jnp.zeros((0,), jnp.int32),
        jnp.zeros((0, 128), jnp.int32),
        jnp.zeros((0, 1), jnp.float32),
    )
    assert result[0] is pool and result[1] is scales
    encoded.assert_not_called()
    with pytest.raises(ValueError, match="incompatible shapes"):
        kernels.publish_csa2_encoded.__wrapped__(
            pool,
            scales,
            jnp.zeros((1, 1), jnp.int32),
            jnp.zeros((1, 128), jnp.int32),
            jnp.zeros((1,), jnp.float32),
        )
