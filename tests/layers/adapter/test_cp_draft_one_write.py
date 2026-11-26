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
"""One-op context K/V write of every DSpark draft layer under DCP
(`cp_draft_attention.write_layers_op`, TPU_MINIMAX_M3_DSPARK_ONE_KV_WRITE)."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.layers.adapter import cp_draft_attention as cda

PAGE = 128
HEAD_DIM = 128


def _cache(pages, heads, dtype):
    packing = 4 // jnp.dtype(dtype).itemsize
    words = -(-2 * heads // packing)
    return jnp.zeros((pages, PAGE, words, packing, HEAD_DIM), dtype)


@pytest.mark.parametrize("dcp", [2, 8])
def test_write_layers_equals_per_layer(dcp):
    rng = np.random.default_rng(dcp)
    ctx = np.array([500, 0, 129], np.int32)
    q_len = np.array([20, 9, 300], np.int32)
    total = ctx + q_len
    logical = -(-int(total.max()) // (PAGE * dcp)) + 1
    perm = rng.permutation(3 * logical + 1)[: 3 * logical]
    tables = jnp.asarray(perm, jnp.int32)
    cu = jnp.asarray(np.concatenate([[0], np.cumsum(q_len)]), jnp.int32)
    num_tokens = int(q_len.sum()) + 4
    keys = jnp.asarray(rng.standard_normal((3, num_tokens, 1, HEAD_DIM)), jnp.bfloat16)
    values = jnp.asarray(
        rng.standard_normal((3, num_tokens, 1, HEAD_DIM)), jnp.bfloat16
    )
    caches = [_cache(3 * logical + 1, dcp // 2, jnp.float8_e4m3fn) for _ in range(3)]
    for rank in range(dcp):
        kw = dict(
            dcp=dcp,
            head_ranks=2,
            rank=jnp.int32(rank),
            k_scale=0.7,
            v_scale=0.3,
            interpret=True,
        )
        got = cda.write_layers(
            caches, keys, values, jnp.asarray(total), tables, cu, **kw
        )
        for i in range(3):
            want = cda.write_kv(
                caches[i], keys[i], values[i], jnp.asarray(total), tables, cu, **kw
            )
            np.testing.assert_array_equal(
                np.asarray(got[i].astype(jnp.float32)),
                np.asarray(want.astype(jnp.float32)),
            )


def _model(dspark, layers, hidden, seen):
    class _Norm(torch.nn.Module):
        def forward(self, x):
            return x * 0.5

    class _Rope:
        def __call__(self, positions, k, q):
            return k + positions[:, None].to(k.dtype), q

    def make_layer(i):
        attn = torch.nn.Module()
        w = torch.randn(hidden, 3 * HEAD_DIM)
        attn.qkv_proj = lambda x, w=w: (x @ w, None)
        attn.q_size, attn.kv_size, attn.head_dim = HEAD_DIM, HEAD_DIM, HEAD_DIM
        attn.k_norm = _Norm()
        attn.rotary_emb = _Rope()

        def write(layer, k, v, cache, md, i=i):
            seen["per_layer"].append((i, k.clone(), v.clone()))

        impl = SimpleNamespace(
            sharded_draft_dcp=dict(dcp=8, kv_head_replicas=2, k_scale=1.0, v_scale=1.0),
            num_kv_heads=1,
            head_size=HEAD_DIM,
            write_sharded_draft_kv=write,
        )
        attn.attn = SimpleNamespace(
            impl=impl,
            kv_cache=torch.zeros(1),
            draft_dcp_rank=torch.tensor([0], dtype=torch.int32),
        )
        return SimpleNamespace(self_attn=attn)

    model = SimpleNamespace(
        layers=[make_layer(i) for i in range(layers)],
        fc=lambda x: x[:, :hidden],
        hidden_norm=_Norm(),
        _one_kv_write_op=None,
        context_kv_weight=None,
    )
    for name in (
        "prepare_draft_kv_write",
        "_layers_context_kv",
        "_write_context_kv_one_op",
        "store_context_kv",
    ):
        setattr(model, name, getattr(dspark.MiniMaxM3DSparkModel, name).__get__(model))
    return model


@pytest.mark.parametrize("enabled", [False, True])
def test_model_one_write_matches_per_layer(monkeypatch, enabled):
    """`store_context_kv` with the all-layer op hands the op each layer's K/V
    exactly as the per-layer path writes it; env off keeps the per-layer path."""
    from vllm_torchtpu import envs
    from vllm_torchtpu.models.minimax_m3 import dspark

    monkeypatch.setattr(
        envs, "TPU_MINIMAX_M3_DSPARK_ONE_KV_WRITE", enabled, raising=False
    )
    monkeypatch.setattr(dspark, "get_tensor_model_parallel_world_size", lambda: 1)
    torch.manual_seed(0)
    hidden, num_tokens, layers = 64, 9, 3
    seen = {"per_layer": [], "op": None}
    model = _model(dspark, layers, hidden, seen)
    monkeypatch.setattr(
        dspark.cp_draft_attention, "write_layers_op", lambda *cfg: ("op", cfg)
    )

    def fake_write_layers(*, keys, values, kv_caches, attn_metadata, dcp_rank, op):
        seen["op"] = (op, keys.clone(), values.clone())

    monkeypatch.setattr(
        dspark.cp_draft_attention, "write_layers_with_dcp", fake_write_layers
    )
    aux = torch.randn(num_tokens, hidden)
    positions = torch.arange(num_tokens)
    md = SimpleNamespace()
    model.store_context_kv(aux, positions, (md,) * layers)
    assert len(seen["per_layer"]) == layers
    model.prepare_draft_kv_write()
    if not enabled:
        assert model._one_kv_write_op is None
        seen["per_layer"].clear()
        model.store_context_kv(aux, positions, (md,) * layers)
        assert len(seen["per_layer"]) == layers and seen["op"] is None
        return
    assert model._one_kv_write_op == ("op", (8, 2, 1.0, 1.0, layers))
    model.store_context_kv(aux, positions, (md,) * layers)
    op, keys, values = seen["op"]
    assert keys.shape == (layers, num_tokens, 1, HEAD_DIM)
    for i, k, v in seen["per_layer"]:
        torch.testing.assert_close(
            keys[i], k.reshape(num_tokens, 1, HEAD_DIM), rtol=0, atol=0
        )
        torch.testing.assert_close(
            values[i], v.reshape(num_tokens, 1, HEAD_DIM), rtol=0, atol=0
        )
    # Distinct metadata per layer: the per-layer path.
    seen["per_layer"].clear()
    model.store_context_kv(
        aux, positions, tuple(SimpleNamespace() for _ in range(layers))
    )
    assert len(seen["per_layer"]) == layers


@pytest.mark.parametrize("layers", [1, 5])
@pytest.mark.parametrize("dcp", [2, 8])
def test_write_layers_op_fake_symbolic(dcp, layers):
    """The all-layer write op's fake fits its schema (one layer: one bare
    Tensor) with a symbolic token count, as Dynamo traces it on TPU."""
    pytest.importorskip("torch_tpu")
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import (
        DimDynamic,
        ShapeEnv,
        StatelessSymbolicContext,
    )

    op = cda.write_layers_op(dcp, 2, None, None, layers)
    assert len(op._opoverload._schema.returns) == layers
    cache = torch.empty((4096, PAGE, 4, 2, HEAD_DIM), dtype=torch.bfloat16)
    meta = (
        torch.empty((16,), dtype=torch.int32),
        torch.empty((16 * 64,), dtype=torch.int32),
        torch.empty((17,), dtype=torch.int32),
        torch.empty((1,), dtype=torch.int32),
    )
    mode = FakeTensorMode(shape_env=ShapeEnv())
    seed = mode.from_tensor(
        torch.empty(72),
        symbolic_context=StatelessSymbolicContext(dynamic_sizes=[DimDynamic.DYNAMIC]),
    )
    caches = [mode.from_tensor(cache) for _ in range(layers)]
    meta_f = [mode.from_tensor(m) for m in meta]
    with mode:
        n = seed.shape[0]
        assert isinstance(n, torch.SymInt)
        keys = torch.empty((layers, n, 1, HEAD_DIM), dtype=torch.bfloat16)
        outs = op(*caches, keys, keys, *meta_f)
    outs = outs if isinstance(outs, tuple) else (outs,)
    assert len(outs) == layers
    assert all(o.shape == cache.shape for o in outs)


def test_context_kv_cat_matches_per_layer(monkeypatch):
    """TPU_MINIMAX_M3_DSPARK_CONTEXT_KV_CAT: one matmul against the layers'
    concatenated K/V columns gives each layer's context K/V exactly."""
    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
    from vllm_torchtpu.models.minimax_m3 import dspark

    monkeypatch.setattr(envs, "TPU_MINIMAX_M3_DSPARK_ONE_KV_WRITE", True, raising=False)
    monkeypatch.setattr(envs, "TPU_MINIMAX_M3_DSPARK_CONTEXT_KV_CAT", True, raising=False)
    monkeypatch.setattr(
        dspark.cp_draft_attention, "write_layers_op", lambda *cfg: ("op", cfg)
    )
    torch.manual_seed(0)
    hidden, layers = 64, 3
    model = _model(dspark, layers, hidden, {"per_layer": [], "op": None})
    for layer in model.layers:
        linear = torch.nn.Module()
        linear.weight = torch.nn.Parameter(
            torch.randn(hidden, 3 * HEAD_DIM).to(torch.bfloat16), requires_grad=False
        )
        linear.bias = None
        setattr(linear, WEIGHT_FLIPPED_ATTR, True)
        layer.self_attn.qkv_proj = linear
    context = torch.randn(9, hidden).to(torch.bfloat16)
    want = model._layers_context_kv(context)
    model.prepare_draft_kv_write()
    assert model.context_kv_weight is not None
    assert model.context_kv_weight.shape == (hidden, layers * 2 * HEAD_DIM)
    got = model._layers_context_kv(context)
    for (gk, gv), (wk, wv) in zip(got, want, strict=True):
        torch.testing.assert_close(gk, wk, rtol=0, atol=0)
        torch.testing.assert_close(gv, wv, rtol=0, atol=0)
