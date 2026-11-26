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
"""Spec tests for split absorb_q wiring and searchsorted compare_all fast path."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.mla import dcp_token_parallel, kv_cache_utils
from vllm_torchtpu.layers.adapter import attention as attention_adapter
from vllm_torchtpu.layers.adapter import dcp_token_parallel_op


def _make_layout(dcp_size: int = 4, split_rows: int = 8):
    return dcp_token_parallel.TokenParallelLayout(
        dcp_size=dcp_size,
        interleave_size=1,
        page_size=16,
        split_rows=split_rows,
        max_pages=4,
    )


class _FakeLayer:
    def __init__(
        self, num_heads: int = 4, kv_lora_rank: int = 32, v_head_dim: int = 16
    ) -> None:
        self.layer_name = "model.layers.0.self_attn.attn"
        self.use_sparse = True
        self.kv_cache_quantized_dtype = None
        self.scale = 0.125
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.v_head_dim = v_head_dim
        self.W_UV = torch.zeros(
            (num_heads, kv_lora_rank, v_head_dim), dtype=torch.bfloat16
        )
        self.W_UV_tp_all = torch.zeros(
            (num_heads * 4, kv_lora_rank, v_head_dim), dtype=torch.bfloat16
        )


def _make_fake_layer(num_heads: int = 4, kv_lora_rank: int = 32, v_head_dim: int = 16):
    return _FakeLayer(num_heads, kv_lora_rank, v_head_dim)


def _capture_built_ops(monkeypatch, *, q_replicated: bool, local_oproj: bool):
    captured: list[dict] = []

    def fake_pcp_op(name, fn, **kwargs):
        record = {"name": name, "fn": fn, "kwargs": kwargs, "calls": [], "fake": None}

        def op_call(*args):
            record["calls"].append(args)
            return record["fake"](*args)

        def register_fake(fake_fn):
            record["fake"] = fake_fn

        op_call.register_fake = register_fake
        captured.append(record)
        return op_call

    monkeypatch.setattr(dcp_token_parallel_op, "pcp_streaming_jax_op", fake_pcp_op)
    monkeypatch.setattr(
        dcp_token_parallel_op,
        "_sharded_rank_ids",
        lambda: torch.zeros((1,), dtype=torch.int32),
    )
    monkeypatch.setattr(
        "vllm.distributed.get_tensor_model_parallel_world_size",
        lambda: 4,
    )
    monkeypatch.setattr(
        dcp_token_parallel_op,
        "dcp_rank_operand",
        lambda: torch.zeros((1,), dtype=torch.int32),
    )
    monkeypatch.setattr(
        "vllm_torchtpu.distributed.dcp.dcp_rank_operand",
        lambda: torch.zeros((1,), dtype=torch.int32),
    )
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.cp_mla_attention._retain_rank",
        lambda x: x,
    )

    layout = _make_layout(dcp_size=4, split_rows=8)
    layer = _make_fake_layer()
    impl = SimpleNamespace(
        _dcp_mesh=f"mesh_{q_replicated}_{local_oproj}_{id(captured)}",
        dcp_size=4,
        dcp_interleave_size=1,
    )
    tp_op = dcp_token_parallel_op.build_attention_op(
        layer, impl, layout, q_replicated=q_replicated, local_oproj=local_oproj
    )
    layer.sparse_mla_tp_op = tp_op
    return layer, tp_op, captured


def test_should_accept_split_q_tuple_and_preserve_fake_shapes(monkeypatch):
    layer, tp_op, captured = _capture_built_ops(
        monkeypatch, q_replicated=True, local_oproj=True
    )
    assert tp_op.split_q_inputs is True
    built_kinds = [record["name"].rsplit("_", 1)[0] for record in captured]
    assert built_kinds == [
        "pallas::sparse_mla_tp",
        "pallas::sparse_mla_tp_oproj",
        "pallas::sparse_mla_tp_oproj_sharded_kv",
        "pallas::dcp_qc_head_shift",
        "pallas::dcp_gather_owned_k",
        "pallas::dcp_indexer_owned_rows",
    ]

    num_tokens, s, rows = 64, 8, 16
    nope_cache = torch.zeros((4, 16, 32), dtype=torch.bfloat16)
    rope_cache = torch.zeros((4, 16, 8), dtype=torch.bfloat16)
    q_head = torch.zeros((s, 16, 32), dtype=torch.bfloat16)
    q_tail = torch.zeros((rows, 16, 32), dtype=torch.bfloat16)
    q_pe = torch.zeros((s + rows, 16, 8), dtype=torch.bfloat16)
    kv_c = torch.zeros((num_tokens, 32), dtype=torch.bfloat16)
    k_pe = torch.zeros((num_tokens, 8), dtype=torch.bfloat16)
    topk = torch.zeros((s + rows, 4), dtype=torch.int32)
    md = SimpleNamespace(
        seq_lens=torch.zeros((2,), dtype=torch.int32),
        block_tables=torch.zeros((4,), dtype=torch.int32),
        query_start_loc=torch.zeros((3,), dtype=torch.int32),
        request_distribution=torch.zeros((3,), dtype=torch.int32),
    )

    out_base = dcp_token_parallel_op.attention(
        layer,
        (nope_cache, rope_cache),
        (q_head, q_tail),
        q_pe,
        kv_c,
        k_pe,
        topk,
        md,
        q_replicated=True,
    )
    assert tuple(out_base.shape) == (num_tokens, layer.num_heads, layer.v_head_dim)
    assert captured[0]["calls"][0][2] is q_head
    assert captured[0]["calls"][0][3] is q_tail

    hidden = 48
    w_o = (
        torch.zeros((layer.num_heads * layer.v_head_dim, hidden), dtype=torch.bfloat16),
        torch.ones((hidden,), dtype=torch.float32),
    )
    out_oproj = dcp_token_parallel_op.attention(
        layer,
        (nope_cache, rope_cache),
        (q_head, q_tail),
        q_pe,
        kv_c,
        k_pe,
        topk,
        md,
        q_replicated=True,
        w_o=w_o,
    )
    assert tuple(out_oproj.shape) == (num_tokens // 4, hidden)
    assert captured[1]["calls"][0][2] is q_head
    assert captured[1]["calls"][0][3] is q_tail

    # Legacy contiguous tensor input still splits automatically.
    q_contig = torch.cat([q_head, q_tail], dim=0)
    out_contig = dcp_token_parallel_op.attention(
        layer,
        (nope_cache, rope_cache),
        q_contig,
        q_pe,
        kv_c,
        k_pe,
        topk,
        md,
        q_replicated=True,
    )
    assert tuple(out_contig.shape) == (num_tokens, layer.num_heads, layer.v_head_dim)
    assert tuple(captured[0]["calls"][1][2].shape) == (s, 16, 32)
    assert tuple(captured[0]["calls"][1][3].shape) == (rows, 16, 32)


def test_should_reject_mismatched_split_q_row_counts(monkeypatch):
    layer, _, _ = _capture_built_ops(monkeypatch, q_replicated=True, local_oproj=False)
    num_tokens = 64
    nope_cache = torch.zeros((4, 16, 32), dtype=torch.bfloat16)
    rope_cache = torch.zeros((4, 16, 8), dtype=torch.bfloat16)
    bad_head = torch.zeros((7, 16, 32), dtype=torch.bfloat16)
    good_tail = torch.zeros((16, 16, 32), dtype=torch.bfloat16)
    q_pe = torch.zeros((24, 16, 8), dtype=torch.bfloat16)
    kv_c = torch.zeros((num_tokens, 32), dtype=torch.bfloat16)
    k_pe = torch.zeros((num_tokens, 8), dtype=torch.bfloat16)
    topk = torch.zeros((24, 4), dtype=torch.int32)
    md = SimpleNamespace(
        seq_lens=torch.zeros((2,), dtype=torch.int32),
        block_tables=torch.zeros((4,), dtype=torch.int32),
        query_start_loc=torch.zeros((3,), dtype=torch.int32),
        request_distribution=torch.zeros((3,), dtype=torch.int32),
    )
    with pytest.raises(ValueError, match="split q has"):
        dcp_token_parallel_op.attention(
            layer,
            (nope_cache, rope_cache),
            (bad_head, good_tail),
            q_pe,
            kv_c,
            k_pe,
            topk,
            md,
            q_replicated=True,
        )


def test_should_compute_exact_fake_shapes_for_absorb_q_split():
    q_nope = torch.empty((1088, 64, 192), dtype=torch.bfloat16)
    w_uk_t = torch.empty((64, 192, 512), dtype=torch.bfloat16)
    head, tail = attention_adapter._fake_absorb_q_split(q_nope, w_uk_t, 64)  # noqa: SLF001 -- verifying registered torch fake kernel shape contract directly on CPU
    assert tuple(head.shape) == (64, 64, 512)
    assert tuple(tail.shape) == (1024, 64, 512)


def test_should_route_oproj_op_through_overlapped_lead_attention_and_project_from_v(
    monkeypatch,
):
    dcp = 4
    if len(jax.devices()) < dcp:
        pytest.skip(f"needs {dcp} devices")
    mesh = Mesh(np.array(jax.devices()[:dcp]), ("dcp",))
    shd = NamedSharding(mesh, P("dcp"))
    _capture_built_ops(monkeypatch, q_replicated=True, local_oproj=True)
    captured_real: list[dict] = []

    def fake_pcp_op(name, fn, **kwargs):
        rec = {"name": name, "fn": fn, "kwargs": kwargs}

        def rec_call(*args):
            return args

        def register_fake(fake_fn):
            del fake_fn

        rec_call.register_fake = register_fake
        captured_real.append(rec)
        return rec_call

    monkeypatch.setattr(dcp_token_parallel_op, "pcp_streaming_jax_op", fake_pcp_op)
    layout = _make_layout(dcp_size=dcp, split_rows=8)
    layer = _make_fake_layer()
    nope_spec = kv_cache_utils.SparseMLAKVCacheSpec.create(
        kv_cache_utils.KVCacheType.NOPE,
        kv_cache_utils.KVCacheLayout.TENSORCORE,
        4,
        16,
        512,
        1,
    )
    rope_spec = kv_cache_utils.SparseMLAKVCacheSpec.create(
        kv_cache_utils.KVCacheType.ROPE,
        kv_cache_utils.KVCacheLayout.TENSORCORE,
        4,
        16,
        64,
        1,
    )
    layer.mla_kv_spec = (nope_spec, rope_spec)
    impl = SimpleNamespace(_dcp_mesh=mesh, dcp_size=dcp, dcp_interleave_size=1)

    calls: list[str] = []

    def fake_write(nope, rope, *a, **kw):
        return nope, rope

    def fake_dcp_attn(q_dcp, pe_dcp, kvc, kpe, nope, rope, *a, **kw):
        calls.append("lead_dcp")
        part = jax.device_put(jnp.zeros((dcp * 8, 16, 32), jnp.bfloat16), shd)
        lse = jax.device_put(jnp.zeros((dcp * 8, 16), jnp.float32), shd)
        return nope, rope, part, lse

    def fake_with_lead_oproj(lay, m, ax, nope, rope, *a, **kw):
        calls.append(f"with_lead_oproj:3d={kw.get('epilogue_3d')}")
        out = jax.device_put(jnp.zeros((dcp * 16, 32), jnp.bfloat16), shd)
        return nope, rope, out

    def fake_plain_tp(lay, m, ax, nope, rope, *a, **kw):
        calls.append("plain_tp")
        out = jax.device_put(jnp.zeros((dcp * 64, 4, 16), jnp.bfloat16), shd)
        return nope, rope, out

    monkeypatch.setattr(
        dcp_token_parallel_op, "update_sparse_mla_kv_cache_dcp", fake_write
    )
    monkeypatch.setattr(
        "vllm_torchtpu.layers.core.attention_interface.sparse_mla_attention_dcp",
        fake_dcp_attn,
    )
    monkeypatch.setattr(
        dcp_token_parallel, "sparse_mla_attention_with_lead_oproj", fake_with_lead_oproj
    )
    monkeypatch.setattr(dcp_token_parallel, "sparse_mla_attention", fake_plain_tp)
    dcp_token_parallel_op.build_attention_op(layer, impl, layout, q_replicated=True)

    nope = jax.device_put(jnp.zeros((dcp * 4, 16, 4, 32), jnp.uint8), shd)
    rope = jax.device_put(jnp.zeros((dcp * 4, 4, 4, 8), jnp.uint8), shd)
    q_head = jax.device_put(jnp.zeros((dcp * 8, 16, 32), jnp.bfloat16), shd)
    q_tail = jax.device_put(jnp.zeros((dcp * 16, 16, 32), jnp.bfloat16), shd)
    q_pe = jax.device_put(jnp.zeros((dcp * 24, 16, 8), jnp.bfloat16), shd)
    kv_c = jnp.zeros((64, 32), jnp.bfloat16)
    k_pe = jnp.zeros((64, 8), jnp.bfloat16)
    topk = jax.device_put(jnp.zeros((dcp * 24, 4), jnp.int32), shd)
    sl = jnp.asarray([64, 0], jnp.int32)
    bt = jnp.zeros((8,), jnp.int32)
    qsl = jnp.asarray([0, 64, 64], jnp.int32)
    dist = jnp.asarray([0, 0, 1], jnp.int32)
    ranks = jax.device_put(jnp.arange(dcp, dtype=jnp.int32), shd)
    w_uv = jnp.zeros((16, 32, 16), jnp.bfloat16)
    w_o = jax.device_put(jnp.zeros((dcp * 128, 32), jnp.float8_e4m3fn), shd)
    s_o = jax.device_put(jnp.ones((dcp * 32,), jnp.float32), shd)

    oproj_fn = captured_real[1]["fn"]
    base_fn = captured_real[0]["fn"]
    oproj_fn(
        nope,
        rope,
        q_head,
        q_tail,
        q_pe,
        kv_c,
        k_pe,
        topk,
        sl,
        bt,
        qsl,
        dist,
        ranks,
        w_uv,
        w_o,
        s_o,
    )
    assert calls == ["lead_dcp", "with_lead_oproj:3d=True"]

    calls.clear()
    w_uv_shd = jax.device_put(jnp.zeros((dcp * 4, 32, 16), jnp.bfloat16), shd)
    base_fn(
        nope,
        rope,
        q_head,
        q_tail,
        q_pe,
        kv_c,
        k_pe,
        topk,
        sl,
        bt,
        qsl,
        dist,
        ranks,
        w_uv_shd,
    )
    assert calls == ["lead_dcp", "plain_tp"]
