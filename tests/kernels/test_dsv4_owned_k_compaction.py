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

from __future__ import annotations

import inspect
import itertools
import types
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch._dynamo as torch_dynamo
import vllm.distributed as vllm_dist
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.compilation import shape_variants
from vllm_torchtpu.distributed import dcp as dcp_mod
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    DCP_AXIS_NAME,
    KVLayout,
    get_dcp_page_and_slot,
    get_dst_rows,
    get_kv_cache_shape,
    kv_cache_append,
)
from vllm_torchtpu.kernels.mla import dcp_token_parallel as tp
from vllm_torchtpu.layers.adapter import dcp_token_parallel_op as tp_op_mod
from vllm_torchtpu.layers.adapter import dcp_unified_idx_qb_mla_op as unified_qb_mod
from vllm_torchtpu.layers.adapter import dcp_unified_indexer_mla_op as unified_op_mod
from vllm_torchtpu.layers.adapter import moe_prefill_sp as sp_mod
from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op as mla_op


@pytest.mark.parametrize("zigzag_bands", [1, 4])
def test_should_compact_owned_row_indices_and_invert_zigzag_bands_exactly(
    zigzag_bands: int,
):
    dcp, interleave, page_size, num_pages = 4, 4, 64, 16
    num_tokens = 1024
    num_seqs = 4
    seq_lens = jnp.array([500, 920, 0, 0], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 360, 980, 980, 980], dtype=jnp.int32)
    page_indices = jnp.arange(num_seqs * 8, dtype=jnp.int32).reshape(num_seqs, 8)

    req_major_k = (
        torch.arange(num_tokens * 8, dtype=torch.float32).reshape(num_tokens, 8) * 0.125
        + 3.0
    )
    req_positions = torch.arange(num_tokens, dtype=torch.int64) * 3 + 17
    zigzag_global_k = sp_mod._to_zigzag_rank_major(  # noqa: SLF001 -- verifies exact inverse of module's zigzag helper
        req_major_k, dcp, zigzag_bands
    )
    ranks_gathered = []
    for r in range(dcp):
        rk_t = torch.tensor([[r]], dtype=torch.int32)
        ranks_gathered.append(sp_mod.own_rows(zigzag_global_k, rk_t, dcp))
    gathered_raw_k = jnp.asarray(torch.cat(ranks_gathered, dim=0).numpy())
    unzigzagged_k = jnp.asarray(
        sp_mod._from_zigzag_rank_major(  # noqa: SLF001 -- golden unzigzag reference
            torch.from_numpy(np.array(gathered_raw_k, copy=True)),
            num_tokens,
            dcp,
            zigzag_bands,
        ).numpy()
    )
    np.testing.assert_array_equal(np.asarray(unzigzagged_k), req_major_k.numpy())

    for r in range(dcp):
        page, slot = get_dcp_page_and_slot(
            num_tokens,
            seq_lens,
            page_indices,
            cu_q_lens,
            r,
            page_size=page_size,
            num_pages=num_pages,
            dcp_size=dcp,
            interleave_size=interleave,
        )
        dst_rows = get_dst_rows(page, slot, page_size)
        own_idx, dst_c = kv_cache_append.compact_owned_row_indices(
            dst_rows,
            num_seqs=num_seqs,
            dcp_size=dcp,
            interleave_size=interleave,
        )
        ref_k_own, ref_dst_c = kv_cache_append.compact_owned_rows(
            unzigzagged_k,
            dst_rows,
            num_seqs=num_seqs,
            dcp_size=dcp,
            interleave_size=interleave,
        )
        raw_own = kv_cache_append.zigzag_to_rank_major_idx(
            own_idx,
            num_tokens=num_tokens,
            dcp_size=dcp,
            zigzag_bands=zigzag_bands,
        )
        got_k_own = jnp.take(gathered_raw_k, raw_own, axis=0)
        np.testing.assert_array_equal(np.asarray(dst_c), np.asarray(ref_dst_c))
        np.testing.assert_array_equal(np.asarray(got_k_own), np.asarray(ref_k_own))

        got_pos = req_positions.index_select(
            0, torch.from_numpy(np.array(own_idx, copy=True)).long()
        )
        assert got_pos.dtype == torch.int64
        np.testing.assert_array_equal(
            got_pos.numpy(), req_positions.numpy()[np.asarray(own_idx)]
        )

        if zigzag_bands > 1:
            wrong_raw = kv_cache_append.zigzag_to_rank_major_idx(
                own_idx,
                num_tokens=num_tokens,
                dcp_size=dcp,
                zigzag_bands=1,
            )
            wrong_k_own = jnp.take(gathered_raw_k, wrong_raw, axis=0)
            valid = np.asarray(dst_c) >= 0
            assert not np.array_equal(
                np.asarray(wrong_k_own)[valid], np.asarray(ref_k_own)[valid]
            )


def test_should_return_identity_indices_when_owned_bound_exceeds_num_tokens():
    dst = jnp.array([4, -1, 5, -1], dtype=jnp.int32)
    src = jnp.arange(16, dtype=jnp.uint8).reshape(4, 4)
    idx, dst_out = kv_cache_append.compact_owned_row_indices(
        dst, num_seqs=4, dcp_size=4, interleave_size=4
    )
    src_out, dst_out2 = kv_cache_append.compact_owned_rows(
        src, dst, num_seqs=4, dcp_size=4, interleave_size=4
    )
    np.testing.assert_array_equal(np.asarray(idx), np.arange(4, dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(dst_out), np.asarray(dst))
    np.testing.assert_array_equal(np.asarray(src_out), np.asarray(src))
    np.testing.assert_array_equal(np.asarray(dst_out2), np.asarray(dst))


def test_should_reject_invalid_args_in_owned_kv_append_helpers():
    cache = jnp.zeros((2, 16, 4, 128), dtype=jnp.uint8)
    rows = jnp.zeros((8, 128), dtype=jnp.uint8)
    dst = jnp.full((8,), -1, dtype=jnp.int32)
    with pytest.raises(ValueError, match="Expected writer"):
        kv_cache_append.scatter_owned_rows_dcp(cache, rows, dst, writer="invalid")
    with pytest.raises(ValueError, match="Expected .* owned K rows"):
        kv_cache_append.coalesced_owned_kv_append_dcp(
            cache,
            rows,
            jnp.array([32, 0], dtype=jnp.int32),
            jnp.zeros((2, 4), dtype=jnp.int32),
            jnp.array([0, 32, 32], dtype=jnp.int32),
            0,
            num_tokens=256,
            dcp_size=4,
            interleave_size=4,
            kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
            writer="xla",
        )


def test_should_scatter_owned_rows_identically_across_dcp_ranks():
    dcp = 4
    if jax.device_count() < dcp:
        pytest.skip(f"requires {dcp} devices")
    mesh = Mesh(np.array(jax.devices()[:dcp]), (DCP_AXIS_NAME,))
    writer = "pallas" if jax.default_backend() == "tpu" else "xla"
    kv_layout = (
        KVLayout.SEQ_ALONG_LANE if writer == "pallas" else KVLayout.HEAD_ALONG_SUBLANE
    )
    page_size, head_dim, interleave, num_pages = 128, 128, 4, 24
    num_tokens, num_seqs = 512, 3
    rng = np.random.default_rng(77)
    k_packed_np = rng.integers(1, 250, size=(num_tokens, head_dim + 1), dtype=np.uint8)

    cache_shape = get_kv_cache_shape(num_pages, page_size, head_dim, kv_layout)
    shd = NamedSharding(mesh, P(DCP_AXIS_NAME))
    rep = NamedSharding(mesh, P())
    cache_init = jax.device_put(
        jnp.zeros((dcp * num_pages, *cache_shape[1:]), dtype=jnp.uint8), shd
    )
    seq_lens = jax.device_put(jnp.array([220, 410, 0], dtype=jnp.int32), rep)
    page_indices = jax.device_put(
        jnp.arange(num_seqs * 8, dtype=jnp.int32).reshape(num_seqs, 8), rep
    )
    cu_q_lens = jax.device_put(jnp.array([0, 180, 460, 460], dtype=jnp.int32), rep)
    rank_ids = jax.device_put(jnp.arange(dcp, dtype=jnp.int32), shd)

    want_cache = mla_op._cp_scatter_indexer_k(  # noqa: SLF001 -- exercises adapter scatter entrypoint directly
        cache_init,
        jax.device_put(jnp.asarray(k_packed_np), rep),
        seq_lens,
        page_indices,
        cu_q_lens,
        mesh=mesh,
        cp_size=dcp,
        interleave_size=interleave,
        axis_name=DCP_AXIS_NAME,
        writer=writer,
        kv_layout=kv_layout,
        rank_ids=rank_ids,
        compact=True,
    )

    def _prep_owned(kp_rep, sl, pi, cql, rk):
        page, slot = get_dcp_page_and_slot(
            num_tokens,
            sl,
            pi,
            cql,
            rk[0],
            page_size=page_size,
            num_pages=num_pages,
            dcp_size=dcp,
            interleave_size=interleave,
        )
        dst_rows = get_dst_rows(page, slot, page_size)
        own_idx, _ = kv_cache_append.compact_owned_row_indices(
            dst_rows,
            num_seqs=num_seqs,
            dcp_size=dcp,
            interleave_size=interleave,
        )
        return jnp.take(kp_rep, own_idx, axis=0)

    kp_owned = jax.shard_map(
        _prep_owned,
        mesh=mesh,
        in_specs=(P(), P(), P(), P(), P(DCP_AXIS_NAME)),
        out_specs=P(DCP_AXIS_NAME),
        check_vma=False,
    )(
        jax.device_put(jnp.asarray(k_packed_np), rep),
        seq_lens,
        page_indices,
        cu_q_lens,
        rank_ids,
    )

    got_cache = mla_op._cp_scatter_indexer_k(  # noqa: SLF001 -- exercises adapter scatter entrypoint with owned K shard
        cache_init,
        kp_owned,
        seq_lens,
        page_indices,
        cu_q_lens,
        mesh=mesh,
        cp_size=dcp,
        interleave_size=interleave,
        axis_name=DCP_AXIS_NAME,
        writer=writer,
        kv_layout=kv_layout,
        rank_ids=rank_ids,
        k_sharded=True,
        step_tokens=num_tokens,
    )
    np.testing.assert_array_equal(np.asarray(got_cache), np.asarray(want_cache))


def test_should_cache_step_owned_k_plan_across_indexer_layers_and_respect_guards(
    monkeypatch: pytest.MonkeyPatch,
):
    tp, num_tokens, rows_local = 4, 1024, 256
    s, q_lora, kv_dim, k_dim, w_dim = 16, 32, 24, 16, 8
    positions = torch.arange(num_tokens, dtype=torch.int64) * 2 + 5
    seq_lens = torch.tensor([300, 600, 0, 0], dtype=torch.int32)
    block_tables = torch.arange(32, dtype=torch.int32).reshape(4, 8)
    query_start_loc = torch.tensor([0, 200, 700, 700, 700], dtype=torch.int32)

    plan_calls: list[int] = []
    gather_calls: list[int] = []
    shift_calls: list[int] = []

    def fake_owned_rows_op(sl, bt, qsl, tok_probe, rk):
        del sl, bt, qsl, rk
        n_tok = int(tok_probe.shape[0]) * tp
        plan_calls.append(n_tok)
        cap = kv_cache_append.owned_rows_bound(
            n_tok, num_seqs=4, dcp_size=tp, interleave_size=4
        )
        idx = torch.arange(cap, dtype=torch.int32) % n_tok
        return idx, (idx + 3) % n_tok

    def fake_gather_owned_k(
        k_local: torch.Tensor, raw_own: torch.Tensor
    ) -> torch.Tensor:
        gather_calls.append(int(raw_own.shape[0]))
        full = k_local.repeat(tp, 1)
        return full.index_select(0, raw_own.long())

    def fake_qc_shift(qcw_local: torch.Tensor, rk: torch.Tensor) -> torch.Tensor:
        del rk
        shift_calls.append(int(qcw_local.shape[1]))
        return torch.cat([qcw_local[:s], qcw_local], dim=0)

    def _make_mla(live_meta: Any):
        tp_op = types.SimpleNamespace(
            q_replicated=True,
            rank=torch.zeros((1, 1), dtype=torch.int32),
            layout=types.SimpleNamespace(dcp_size=tp, split_rows=s, interleave_size=4),
            serves_sharded_kv=True,
            qc_head_shift_op=fake_qc_shift,
            gather_owned_k_op=fake_gather_owned_k,
            indexer_owned_rows_op=fake_owned_rows_op,
        )
        indexer_op = types.SimpleNamespace(
            topk_op_tp=types.SimpleNamespace(q_own_rows=True, serves_sharded_k=True),
            live_cache_step_metadata=lambda: live_meta,
        )
        indexer = types.SimpleNamespace(
            head_dim=k_dim,
            indexer_op=indexer_op,
            wk_weights_proj=lambda x_in: (
                torch.ones((x_in.shape[0], k_dim + w_dim), dtype=x_in.dtype) * 2.0,
                None,
            ),
        )
        mla_attn = types.SimpleNamespace(
            sparse_mla_tp_op=tp_op,
            sparse_mla_unified_idx_qb_op=types.SimpleNamespace(serves_sharded_k=True),
            sparse_mla_unified_tp_op=types.SimpleNamespace(serves_sharded_k=True),
        )
        return types.SimpleNamespace(
            fused_qkv_a_proj=lambda x_in: (
                torch.ones((x_in.shape[0], q_lora + kv_dim), dtype=x_in.dtype),
                None,
            ),
            indexer=indexer,
            skip_topk=False,
            mla_attn=mla_attn,
            gate_is_fused=False,
            kimi_preprocess=None,
            q_lora_rank=q_lora,
            is_sparse=True,
            g_proj=None,
            o_proj=object(),
        )

    monkeypatch.setattr(sp_mod, "glm_dsa_v7x_active", lambda *_a, **_kw: True)
    monkeypatch.setattr(sp_mod, "get_tensor_model_parallel_world_size", lambda: tp)
    monkeypatch.setattr(sp_mod.dcp_token_parallel_op, "applies", lambda _n: True)
    monkeypatch.setattr(
        sp_mod.dcp_token_parallel_op, "serves_sequence_row_oproj", lambda _o: True
    )
    monkeypatch.setattr(
        sp_mod.fused_moe_ep,
        "rank_operand",
        lambda: torch.zeros((1, 1), dtype=torch.int32),
    )

    gatherer = sp_mod._ShardedKVGatherProjections(  # noqa: SLF001 -- exercises per-step gatherer directly
        positions, tp, zigzag_bands=4
    )
    x_local = torch.ones((rows_local, 16), dtype=torch.bfloat16)
    live_meta = (seq_lens, block_tables, query_start_loc)
    mla_l0 = _make_mla(live_meta)
    mla_l1 = _make_mla(live_meta)

    pre_qkv0, pre_kw0 = gatherer.project_gather(mla_l0, x_local, num_tokens)
    pre_qkv1, pre_kw1 = gatherer.project_gather(mla_l1, x_local, num_tokens)

    assert isinstance(pre_qkv0, tuple) and len(pre_qkv0) == 3
    assert isinstance(pre_kw0, tuple) and len(pre_kw0) == 3
    assert isinstance(pre_kw1, tuple) and len(pre_kw1) == 3
    assert plan_calls == [num_tokens]
    assert len(gather_calls) == 2
    assert shift_calls == [q_lora + w_dim, q_lora + w_dim]
    assert pre_kw0[2].dtype == torch.int64
    assert torch.equal(pre_kw0[2], pre_kw1[2])

    monkeypatch.setattr(
        sp_mod,
        "tensor_model_parallel_all_gather",
        lambda t, _d: t.repeat(tp, 1),
    )
    mla_profile = _make_mla(None)
    _, pre_kw_profile = gatherer.project_gather(mla_profile, x_local, num_tokens)
    assert isinstance(pre_kw_profile, torch.Tensor)
    assert pre_kw_profile.shape == (num_tokens, k_dim + w_dim)


_OP_COUNTER = itertools.count()


class _StopAfterScatterError(RuntimeError):
    pass


def test_should_preserve_symbolic_token_dim_under_dynamo_tracing(
    monkeypatch: pytest.MonkeyPatch,
):
    tp_size = 4
    s, q_lora, kv_dim, k_dim, w_dim = 64, 256, 24, 16, 8
    layout = tp.TokenParallelLayout(
        tp_size,
        4,
        64,
        split_rows=s,
        max_pages=32,
        gather_caps=(4, 8, 16, 32),
        zigzag_bands=4,
    )
    mesh = Mesh(np.array([jax.devices()[0]] * tp_size), (DCP_AXIS_NAME,))

    monkeypatch.setattr(sp_mod, "glm_dsa_v7x_active", lambda *_a, **_kw: True)
    monkeypatch.setattr(
        vllm_dist, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    monkeypatch.setattr(sp_mod, "get_tensor_model_parallel_world_size", lambda: tp_size)
    monkeypatch.setattr(sp_mod.dcp_token_parallel_op, "applies", lambda _n: True)
    monkeypatch.setattr(
        sp_mod.dcp_token_parallel_op, "serves_sequence_row_oproj", lambda _o: True
    )
    monkeypatch.setattr(
        sp_mod.fused_moe_ep,
        "rank_operand",
        lambda: torch.zeros((1, 1), dtype=torch.int32),
    )
    monkeypatch.setattr(
        tp_op_mod,
        "_sharded_rank_ids",
        lambda: torch.zeros((1,), dtype=torch.int32),
    )
    monkeypatch.setattr(
        dcp_mod,
        "dcp_rank_operand",
        lambda: torch.zeros((1, 1), dtype=torch.int32),
    )
    monkeypatch.setattr(
        tp_op_mod,
        "get_shared_jax_op",
        lambda kind, _key, build: build(f"vllm_tpu_sym::{kind}_{next(_OP_COUNTER)}"),
    )

    class _LayerStub:
        kv_cache_quantized_dtype = None
        scale = 0.125
        num_heads = 4
        v_head_dim = 8
        kv_lora_rank = 16

    layer_stub = _LayerStub()
    impl_stub = types.SimpleNamespace(
        _dcp_mesh=mesh, dcp_size=tp_size, dcp_interleave_size=4
    )
    monkeypatch.setattr(
        tp_op_mod, "dcp_rank_operand", lambda: torch.zeros((1, 1), dtype=torch.int32)
    )
    tp_op = tp_op_mod.build_attention_op(
        layer_stub, impl_stub, layout, q_replicated=True
    )
    tp_op.rank = torch.zeros((1, 1), dtype=torch.int32)  # type: ignore[attr-defined]

    seq_lens = torch.tensor([300, 600, 0, 0], dtype=torch.int32)
    block_tables = torch.arange(32, dtype=torch.int32).reshape(4, 8)
    query_start_loc = torch.tensor([0, 200, 700, 700, 700], dtype=torch.int32)
    live_meta = (seq_lens, block_tables, query_start_loc)

    class _FakeProj(torch.nn.Module):
        def __init__(self, in_dim: int, out_dim: int) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.ones((out_dim, in_dim), dtype=torch.bfloat16) * 0.01,
                requires_grad=False,
            )

        def forward(
            self, x_in: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            return x_in @ self.weight.t(), None

    class _FakeMLA(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fused_qkv_a_proj = _FakeProj(16, q_lora + kv_dim)
            self.wk_proj = _FakeProj(16, k_dim + w_dim)
            self.skip_topk = False
            self.gate_is_fused = False
            self.kimi_preprocess = None
            self.q_lora_rank = q_lora
            self.is_sparse = True
            self.g_proj = None
            self.o_proj = torch.nn.Identity()
            self.indexer = types.SimpleNamespace(
                head_dim=k_dim,
                wk_weights_proj=self.wk_proj,
                indexer_op=types.SimpleNamespace(
                    topk_op_tp=types.SimpleNamespace(
                        q_own_rows=True, serves_sharded_k=True
                    ),
                    live_cache_step_metadata=lambda: live_meta,
                ),
            )
            self.mla_attn = types.SimpleNamespace(
                sparse_mla_tp_op=tp_op,
                sparse_mla_unified_idx_qb_op=types.SimpleNamespace(
                    serves_sharded_k=True
                ),
                sparse_mla_unified_tp_op=types.SimpleNamespace(serves_sharded_k=True),
            )

    class _TwoLayerModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.l0 = _FakeMLA()
            self.l1 = _FakeMLA()
            self.register_buffer("rk", torch.zeros((1, 1), dtype=torch.int32))

        def forward(
            self, full_x: torch.Tensor, positions: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            num_tokens = positions.shape[0]
            x_local = sp_mod.own_rows(full_x, self.rk, tp_size)
            gatherer = sp_mod._ShardedKVGatherProjections(  # noqa: SLF001 -- traces exact step gatherer under Dynamo
                positions, tp_size, zigzag_bands=4
            )
            qkv0, kw0 = gatherer.project_gather(self.l0, x_local, num_tokens)
            _, kw1 = gatherer.project_gather(self.l1, x_local, num_tokens)
            assert isinstance(qkv0, tuple) and len(qkv0) == 3
            assert isinstance(kw0, tuple) and len(kw0) == 3
            assert isinstance(kw1, tuple) and len(kw1) == 3
            step_tok = qkv0[1].shape[0] * tp_size
            return kw0[0].sum() + kw1[0].sum(), kw0[2], positions[:step_tok]

    graphs: list[torch.fx.GraphModule] = []

    def _capture_backend(gm: torch.fx.GraphModule, _example_inputs: Any):
        graphs.append(gm)
        return lambda *_a: (
            torch.zeros((), dtype=torch.bfloat16),
            torch.zeros((1,), dtype=torch.int64),
            torch.zeros((1,), dtype=torch.int64),
        )

    torch_dynamo.reset()
    model_ut = _TwoLayerModel()
    full_x_in = torch.ones((16384, 16), dtype=torch.bfloat16)
    pos_in = torch.arange(16384, dtype=torch.int64)
    torch_dynamo.mark_dynamic(full_x_in, 0)
    torch_dynamo.mark_dynamic(pos_in, 0)
    compiled = torch.compile(model_ut, backend=_capture_backend, fullgraph=True)
    compiled(full_x_in, pos_in)

    assert len(graphs) == 1
    env = shape_variants.trace_shape_env(graphs[0])
    supported = [
        sz
        for sz in (2048, 4096, 8192, 16384)
        if shape_variants.unsupported_reason(env, sz) is None
    ]
    assert supported == [2048, 4096, 8192, 16384]

    # Verify jax.export retains all 5 @main arguments (no unused-arg pruning).
    target_op_fn = tp_op.indexer_owned_rows_op._init_fn  # noqa: SLF001 -- inspects wrapped JaxCallable exported module
    exp_ok = target_op_fn.exported(
        jnp.zeros((4,), dtype=jnp.int32),
        jnp.zeros((4, 8), dtype=jnp.int32),
        jnp.zeros((5,), dtype=jnp.int32),
        jnp.zeros((2048, 1), dtype=jnp.bfloat16),
        jnp.zeros((tp_size, 1), dtype=jnp.int32),
    )
    assert "func.func public @main(%arg0:" in exp_ok.mlir_module()
    assert "%arg4:" in exp_ok.mlir_module()

    # Positive control: dropping sentinel lets jax.export prune %arg4 (4 args).
    def _dead_sentinel_fn(sl, pi, cql, sentinel, rk):
        del sentinel
        return sl[:1] + pi[0, :1] + cql[:1] + rk[:1, 0]

    exp_dead = jax.export.export(jax.jit(_dead_sentinel_fn))(
        jnp.zeros((4,), dtype=jnp.int32),
        jnp.zeros((4, 8), dtype=jnp.int32),
        jnp.zeros((5,), dtype=jnp.int32),
        jnp.zeros((2048, 1), dtype=jnp.bfloat16),
        jnp.zeros((tp_size, 1), dtype=jnp.int32),
    )
    assert "%arg4:" not in exp_dead.mlir_module()

    # Positive control: int(x.shape[0]) in register_fake violates mark_dynamic.
    bad_op_name = f"vllm_tpu_sym::bad_fake_{next(_OP_COUNTER)}"

    def _bad_eager(x: torch.Tensor) -> torch.Tensor:
        return x.new_empty((x.shape[0],))

    bad_op = torch.library.custom_op(bad_op_name, _bad_eager, mutates_args=())
    bad_op.register_fake(lambda x: x.new_empty((int(x.shape[0]),)))

    def _specializing_fn(
        pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = bad_op(pos)
        return out.sum().to(torch.bfloat16), pos[:1], out

    pos_bad = torch.arange(16384, dtype=torch.int64)
    torch_dynamo.mark_dynamic(pos_bad, 0)
    with pytest.raises(torch.fx.experimental.symbolic_shapes.ConstraintViolationError):
        torch.compile(_specializing_fn, backend=_capture_backend, fullgraph=True)(
            pos_bad
        )


@pytest.mark.parametrize("use_in_branch_qb", [True, False])
def test_should_forward_step_tokens_and_compacted_k_to_unified_scatter_callbacks(
    monkeypatch: pytest.MonkeyPatch,
    use_in_branch_qb: bool,
):
    dcp = 4
    mesh = Mesh(np.array([jax.devices()[0]] * dcp), (DCP_AXIS_NAME,))
    layout = tp.TokenParallelLayout(
        dcp,
        4,
        32,
        split_rows=16,
        max_pages=8,
        gather_caps=(2, 4, 6, 8),
        zigzag_bands=4,
    )
    registered: list[tuple[str, Any]] = []

    class _FakeBuiltOp:
        def register_fake(self, _fn: Any) -> None:
            return None

    def _capture_op(
        name: str,
        fn: Any,
        *,
        donate_argnums: Any,
        mesh: Any,
        input_partition_specs: Any,
        output_partition_specs: Any,
    ) -> _FakeBuiltOp:
        del donate_argnums, mesh, input_partition_specs, output_partition_specs
        registered.append((name, fn))
        return _FakeBuiltOp()

    target_mod = unified_qb_mod if use_in_branch_qb else unified_op_mod
    monkeypatch.setattr(target_mod, "pcp_streaming_jax_op", _capture_op)
    monkeypatch.setattr(target_mod, "get_tensor_model_parallel_world_size", lambda: dcp)
    monkeypatch.setattr(
        target_mod,
        "dcp_rank_operand",
        lambda: torch.zeros((1,), dtype=torch.int32),
    )
    monkeypatch.setattr(
        target_mod,
        "dcp_lane_operand",
        lambda: torch.zeros((1,), dtype=torch.int32),
    )
    monkeypatch.setattr(
        target_mod,
        "get_shared_jax_op",
        lambda kind, _key, build: build(f"u::{kind}"),
    )

    spec_stub = types.SimpleNamespace(page_size=32, layout="test_layout")

    class _UnifiedLayerStub:
        kv_cache_quantized_dtype = None
        scale = 0.125
        num_heads = 4
        v_head_dim = 6
        use_sparse = True
        mla_kv_spec = (spec_stub, spec_stub)

    layer_stub = _UnifiedLayerStub()
    impl_stub = types.SimpleNamespace(
        _dcp_mesh=mesh, dcp_size=dcp, dcp_interleave_size=4
    )
    indexer_op_stub = types.SimpleNamespace(
        topk_tokens=12, k_cache=types.SimpleNamespace(prefix="layer0.idx")
    )

    mode_shapes: list[tuple[int, ...]] = []
    scatter_calls: list[tuple[tuple[int, ...], int | None, int | None, bool]] = []

    def _resolve_mode(_idx_c: jax.Array, k_for_mode: jax.Array, *, mode: Any):
        del mode
        mode_shapes.append(tuple(k_for_mode.shape))
        return ("pallas", KVLayout.SEQ_ALONG_LANE)

    def _scatter_k(*args: Any, **kwargs: Any) -> jax.Array:
        scatter_calls.append(
            (
                tuple(args[1].shape),
                kwargs.get("step_tokens"),
                kwargs.get("zigzag_bands"),
                bool(kwargs.get("k_sharded")),
            )
        )
        raise _StopAfterScatterError("verified scatter args")

    cb = unified_op_mod.UnifiedIndexerCallbacks(
        resolve_kv_cache_mode=_resolve_mode,
        scatter_indexer_k=_scatter_k,
        streamidx_block_sizes=lambda *a, **kw: ((1, 1, 1), (1, 1, 1)),
        dcp_indexer_a2a_for_bucket=lambda *a: False,
        register_indexer_layer=lambda _prefix: 0,
    )
    monkeypatch.setattr(unified_qb_mod, "glm_dsa_v7x_active", lambda *_a, **_kw: True)
    monkeypatch.setattr(unified_op_mod, "glm_dsa_v7x_active", lambda *_a, **_kw: True)
    built_op = (
        unified_qb_mod.build_unified_indexer_in_branch_qb_op(
            layer_stub,
            impl_stub,
            indexer_op_stub,
            layout,
            q_replicated=True,
            callbacks=cb,
        )
        if use_in_branch_qb
        else unified_op_mod.build_unified_indexer_attention_op(
            layer_stub,
            impl_stub,
            indexer_op_stub,
            layout,
            q_replicated=True,
            callbacks=cb,
        )
    )
    assert built_op is not None
    sharded_k_fns = [fn for name, fn in registered if "sharded_kv_k" in name]
    assert len(sharded_k_fns) == 1

    step_tokens = 2048
    compacted_global_k_rows = dcp * 600
    idx_c = jnp.zeros((16, 3, 4, 32), dtype=jnp.uint8)
    nope_c = jnp.zeros((16, 32, 8), dtype=jnp.bfloat16)
    rope_c = jnp.zeros((16, 8, 16), dtype=jnp.bfloat16)
    q_idx = jnp.zeros((dcp * 528, 2, 12), dtype=jnp.uint8)
    w_idx = jnp.ones((dcp * 528, 2), dtype=jnp.bfloat16)
    k_packed = jnp.zeros((compacted_global_k_rows, 13), dtype=jnp.uint8)
    kv_c_normed = jnp.zeros((step_tokens, 8), dtype=jnp.bfloat16)
    k_pe = jnp.zeros((step_tokens, 4), dtype=jnp.bfloat16)
    tail_common = (
        kv_c_normed,
        k_pe,
        jnp.zeros((8,), dtype=jnp.int32),
        jnp.zeros((16,), dtype=jnp.int32),
        jnp.zeros((16,), dtype=jnp.int32),
        jnp.zeros((9,), dtype=jnp.int32),
        jnp.zeros((3,), dtype=jnp.int32),
        jnp.zeros((3,), dtype=jnp.int32),
        jnp.arange(dcp, dtype=jnp.int32),
        jnp.ones((4, 8, 6), dtype=jnp.bfloat16),
        jnp.zeros((dcp * 24, 32), dtype=jnp.float8_e4m3fn),
        jnp.ones((dcp * 32,), dtype=jnp.float32),
    )
    sig_len = len(inspect.signature(sharded_k_fns[0]).parameters)
    if use_in_branch_qb:
        mid_args: tuple[Any, ...] = (
            jnp.zeros((dcp * 528, 12), dtype=jnp.bfloat16),
            jnp.zeros((12, 48), dtype=jnp.float8_e4m3fn),
            jnp.ones((48,), dtype=jnp.float32),
            jnp.ones((4, 8, 8), dtype=jnp.bfloat16),
            jnp.zeros((dcp * 528, 4), dtype=jnp.bfloat16),
        )
        base_arity = 23
    else:
        mid_args = (
            jnp.zeros((dcp * 16, 4, 8), dtype=jnp.bfloat16),
            jnp.zeros((dcp * 512, 4, 8), dtype=jnp.bfloat16),
            jnp.zeros((dcp * 528, 4, 4), dtype=jnp.bfloat16),
        )
        base_arity = 21
    extra_lane = (jnp.zeros((dcp,), dtype=jnp.int32),) if sig_len > base_arity else ()

    with pytest.raises(_StopAfterScatterError, match="verified scatter args"):
        sharded_k_fns[0](
            idx_c,
            nope_c,
            rope_c,
            q_idx,
            w_idx,
            k_packed,
            *mid_args,
            *tail_common,
            *extra_lane,
        )

    assert mode_shapes == [(600, 13)]
    assert scatter_calls == [((compacted_global_k_rows, 13), step_tokens, 4, True)]
