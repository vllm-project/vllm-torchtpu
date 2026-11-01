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
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import csa2_adapter, preprocess_ops
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
    attn._csa2 = attn._fused_preprocess = True
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
        preprocess_ops, "rope_rows", lambda: lambda cache, pos: cache[pos]
    )
    return attn, positions, state, main


def _torch_op(fn):
    """Run a JAX bridge on Torch CPU tensors, as the registered op would."""

    def call(*args):
        arrays = [
            jnp.asarray(a.float().numpy(), jnp.bfloat16)
            if a.dtype == torch.bfloat16
            else jnp.asarray(a.numpy())
            for a in args
        ]
        return tuple(
            torch.from_numpy(np.array(a, dtype=np.float32)).bfloat16()
            if a.dtype == jnp.bfloat16
            else torch.from_numpy(np.array(a))
            for a in fn(*arrays)
        )

    return call


@pytest.mark.parametrize("ratio,banks", [(1, 1), (1, 2), (2, 1), (2, 2)])
@pytest.mark.parametrize("replay", [False, True])
def test_fused_compressor_wiring(monkeypatch, ratio, banks, replay):
    attn, positions, state, main = _fixture(monkeypatch, ratio, banks, replay)
    captured = {}
    kernel_banks = banks if ratio == 2 else 0

    def kernel(
        x,
        weight,
        gamma,
        rows,
        hp,
        valid,
        pool=None,
        write=None,
        read=None,
        *,
        ratio,
        eps,
    ):
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
        else:
            assert pool is write is read is None
        return (
            jnp.ones((6, 512), jnp.bfloat16),
            jnp.zeros((6, 128), jnp.int32),
            jnp.ones((6, 1), jnp.float32),
            pool,
        )

    monkeypatch.setattr(preprocess_ops.compress_kernel, "fused_compress_rope", kernel)
    monkeypatch.setattr(
        preprocess_ops,
        "compress_rope",
        lambda ratio, banks, eps: _torch_op(
            preprocess_ops._compress_bridge(ratio, banks, eps)
        ),
    )
    publish = MagicMock()
    insert = MagicMock()
    monkeypatch.setattr(csa2_adapter, "publish_main", publish)
    monkeypatch.setattr(v2_attn, "_insert", insert)
    with v2_attn.stage_attention_cache_scope():
        result = attn._write_csa2_global_kv(
            torch.zeros(6, 16, dtype=torch.bfloat16), positions
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
    assert publish.call_args.args[0] is main
    assert torch.equal(publish.call_args.args[1], expected)
    if ratio == 2:
        np.testing.assert_array_equal(captured["hp"][:, 0], [0, 1, 1, 0, 1, 0])
        np.testing.assert_array_equal(captured["write"], [3, 4, 5, 22, 23, -1])
        np.testing.assert_array_equal(captured["read"], [2, -1, -1, 21, -1, -1])
        assert torch.all(state.raw == 7)
        if banks == 2:
            assert torch.all(state.raw_hi == 8)
    else:
        np.testing.assert_array_equal(captured["hp"][:, 0], [1] * 6)
    # Index K is written immediately from the pre-RoPE latent.
    assert result is None
    insert.assert_called_once()
    cache, slots, _, index_valid = insert.call_args.args
    assert cache is attn.indexer.k_cache.kv_cache
    assert torch.equal(
        slots.masked_fill(~index_valid, -1),
        expected_slots.masked_fill(~torch.tensor(valid), -1),
    )
    attn.compressor.project.assert_not_called()


def test_encoder_global_kv_uses_fused_compressor(monkeypatch):
    monkeypatch.setattr(
        v2_attn,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"main": object()}),
    )
    attn = SimpleNamespace(
        _fused_preprocess=True,
        kv_cache=torch.ones(1),
        compressor=SimpleNamespace(project=MagicMock(side_effect=AssertionError)),
        _write_csa2_global_kv=MagicMock(),
    )
    hidden, positions = torch.zeros(2, 16), torch.arange(2, dtype=torch.int32)
    v2_attn.DeepseekV41Attention.write_encoder_global_kv(attn, hidden, positions)
    attn._write_csa2_global_kv.assert_called_once_with(hidden, positions)


def test_csa2_main_cache_rejects_latent_writes():
    main = _cache(512, torch.uint8, scale=True)
    with pytest.raises(NotImplementedError, match="fused compressor"):
        main.write(torch.tensor([0, 1]), torch.zeros(2, 512, dtype=torch.bfloat16))


@pytest.mark.parametrize("ratio,banks", [(1, 0), (2, 1), (2, 2)])
def test_empty_real_bridge_and_fake_contract(ratio, banks):
    fn = preprocess_ops._compress_bridge(ratio, banks, 1e-6)
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
    fake = preprocess_ops._compress_fake(banks)(*torch_args)
    assert len(inspect.signature(fn).parameters) == len(args)
    assert len(actual) == len(fake) == 3 + banks
    for real, traced, expected in zip(actual, abstract, fake):
        assert real.shape == traced.shape == tuple(expected.shape)
        assert str(real.dtype) == str(expected.dtype).removeprefix("torch.")
    for i in range(banks):
        assert actual[3 + i] is args[6 + i]


def test_group_and_token_rows_share_independent_stage_entries(monkeypatch):
    attn, positions, _, _ = _fixture(monkeypatch, 2, 1, False)
    gather = MagicMock(side_effect=lambda cache, pos: cache[pos])
    monkeypatch.setattr(preprocess_ops, "rope_rows", lambda: gather)
    with v2_attn.stage_attention_cache_scope():
        token = attn._rope_rows(positions)
        grouped = attn._group_positions(positions)
        group = attn._rope_rows(grouped, 2)
        assert attn._group_positions(positions) is grouped
        assert attn._rope_rows(positions) is token
        assert attn._rope_rows(grouped, 2) is group
        assert gather.call_count == 2


def test_register_matching_signatures_donations_and_names(monkeypatch):
    registered = {}

    def register(name, fn, **kwargs):
        count = len(inspect.signature(fn).parameters)
        assert count == len(kwargs["input_partition_specs"])
        registered[name] = (count, kwargs["donate_argnums"])
        return SimpleNamespace(register_fake=lambda fake: None)

    monkeypatch.setattr(preprocess_ops, "_ops", {})
    monkeypatch.setattr(preprocess_ops, "get_pallas_op_mesh", lambda: None)
    monkeypatch.setattr(preprocess_ops.pallas, "jax_op", register)
    with pytest.raises(ValueError, match="one/two banks"):
        preprocess_ops.compress_rope(2, 0, 1e-5)
    assert not registered
    for ratio, banks in ((1, 0), (2, 1), (2, 2)):
        preprocess_ops.compress_rope(ratio, banks, 1e-5)
    preprocess_ops.rope_rows()
    preprocess_ops.qb_rope()
    for emit_qr in (True, False):
        for eps in (1e-6, 1e-5):
            preprocess_ops.qa_rope(emit_qr, eps)
            preprocess_ops.qa_rope(emit_qr, eps)  # cached, not re-registered
    compress = {k: v for k, v in registered.items() if "compress" in k}
    assert sorted(compress.values()) == [(6, ()), (9, (6,)), (10, (6, 7))]
    assert registered["pallas::dsv41_rope_rows"] == (2, ())
    assert registered["pallas::dsv41_qb_rope"] == (4, ())
    qa = [k for k in registered if "qa_kv_norm_rope" in k]
    assert len(qa) == 4 and len(registered) == 9


def test_csa2_bridges_have_schema_compatible_signatures(monkeypatch):
    # Torch infers each op schema from the bridge signature: varargs are
    # rejected at model construction, and every parameter needs a spec.
    from torch_tpu._internal import pallas

    from vllm_torchtpu.models.vllm import vllm_model_wrapper_context as context

    registered = {}

    def register(name, fn, **kwargs):
        params = list(inspect.signature(fn).parameters.values())
        assert all(p.kind is p.POSITIONAL_OR_KEYWORD for p in params), name
        assert len(params) == len(kwargs["input_partition_specs"]), name
        registered[name] = kwargs["donate_argnums"]
        return SimpleNamespace(register_fake=lambda fake: None)

    monkeypatch.setattr(csa2_adapter, "_ops", {})
    monkeypatch.setattr(pallas, "jax_op", register)
    monkeypatch.setattr(context, "get_pallas_op_mesh", lambda: None)
    monkeypatch.setattr(
        context, "get_vllm_model_wrapper_context", lambda: SimpleNamespace(mesh=None)
    )
    csa2_adapter.initialize_csa2_ops()
    prefix = "pallas::dsv41v2_csa2_"
    assert registered[prefix + "publish_swa_planned"] == (0,)
    assert registered[prefix + "publish_swa_planned_banked"] == (0, 1)
    assert registered[prefix + "plan_swa_banked"] == ()


@pytest.mark.parametrize("banks", [1, 2])
def test_publication_preserves_banks_and_rejects_bad_shapes(banks):
    from vllm_torchtpu.kernels.deepseek_v41.csa2 import cache as kernels

    pool = jnp.zeros((8, 128), jnp.int32)
    scales = jnp.zeros((8,), jnp.float32)
    if banks == 2:
        pool = (pool, pool)
        scales = (scales, scales)
    result = kernels.publish_csa2_kv.__wrapped__(
        pool,
        scales,
        jnp.zeros((0,), jnp.int32),
        jnp.zeros((0, 128), jnp.int32),
        jnp.zeros((0, 1), jnp.float32),
    )
    assert result[0] is pool and result[1] is scales
    with pytest.raises(ValueError, match="incompatible shapes"):
        kernels.publish_csa2_kv.__wrapped__(
            pool,
            scales,
            jnp.zeros((1, 1), jnp.int32),
            jnp.zeros((1, 128), jnp.int32),
            jnp.zeros((1,), jnp.float32),
        )
