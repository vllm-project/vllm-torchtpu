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
"""Unit and numerical parity tests for DeepSeek-V4.1 local-query PCP."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch.nn as nn
import vllm.distributed.parallel_state as ps_mod
import vllm.forward_context as fc_mod

from tests.models.vllm.deepseek_v41 import test_attention as attn_test
from tests.models.vllm.deepseek_v41 import test_engram as engram_test
from vllm_torchtpu import patch_registry
from vllm_torchtpu.kernels.experimental.deepseek_v41.cache import (
    gather_cache_rows,
    pack_fp8_kv_rows,
    write_sub_rows,
)
from vllm_torchtpu.kernels.experimental.deepseek_v41.indexer import paged_indexer_topk
from vllm_torchtpu.kernels.experimental.deepseek_v41.sparse_mla import (
    gather_sparse_fp8_kv,
    sparse_mla_swa_attention,
)
from vllm_torchtpu.layers.adapter.custom_ops import deepseek_v41 as dsv41_ops
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.layers.core.pcp_sequence_layout import (
    PCP_STREAMING_SEQUENCE_LAYOUT_DESCRIPTOR,
    PcpSequenceLayoutEligibility,
    PcpSequenceLayoutPlanner,
    prepare_pcp_sequence_layout,
)
from vllm_torchtpu.layers.core.sequence_layout import (
    DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR,
    AllSequenceLayoutPlanner,
    SequenceLayoutPlan,
)
from vllm_torchtpu.models.vllm import _TPUDeepseekV41Config
from vllm_torchtpu.models.vllm.deepseek_v41 import (
    DeepseekV41Attention,
    DeepseekV41DecoderLayer,
    DeepseekV41ForCausalLM,
    DeepseekV41Model,
)
from vllm_torchtpu.models.vllm.deepseek_v41 import engram as engram_mod
from vllm_torchtpu.models.vllm.deepseek_v41 import model as model_mod
from vllm_torchtpu.models.vllm.deepseek_v41 import pcp as pcp_mod
from vllm_torchtpu.models.vllm.deepseek_v41.engram import capture_host_engram_inputs
from vllm_torchtpu.models.vllm.deepseek_v41.pcp import PcpStep
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform


@contextmanager
def _mock_forward_context(
    monkeypatch: pytest.MonkeyPatch,
    attn_metadata: Any,
    additional_kwargs: dict[str, Any] | None = None,
    dp_metadata: Any = None,
):
    ctx = SimpleNamespace(
        attn_metadata=attn_metadata,
        additional_kwargs={} if additional_kwargs is None else additional_kwargs,
        dp_metadata=dp_metadata,
        batch_descriptor=None,
    )
    for mod in (fc_mod, engram_mod.upstream, model_mod.upstream):
        monkeypatch.setattr(mod, "get_forward_context", lambda: ctx)
        monkeypatch.setattr(mod, "is_forward_context_available", lambda: True)
    yield ctx


def _pcp_config(
    *,
    pcp_size: int = 1,
    tp_size: int = 1,
    dcp_size: int = 1,
    dp_size: int = 1,
    enable_ep: bool = False,
    window_size: int = 16,
    block_size: int = 16,
    architecture: str = "DeepseekV41ForCausalLM",
) -> Any:
    cfg = attn_test._make_vllm_config(
        max_model_len=64,
        compress_ratios=[2, 2, 2, 1, 0],
        static_ctx={},
    )
    hf = cfg.model_config.hf_config
    hf.sliding_window = window_size
    hf.max_position_embeddings = 128
    hf.num_hidden_layers = 5
    hf.num_attention_heads = 16
    hf.o_groups = 2
    hf.o_lora_rank = 128
    hf.index_n_heads = 8
    hf.index_topk = 16
    hf.kv_source_layer_ids = [0, 3]
    hf.index_source_layer_ids = [0, 1, 3]
    hf.candidate_source_layer_id = 1
    hf.candidate_topk_blocks = 4
    hf.candidate_block_size = 4
    hf.vocab_size = 256
    hf.hc_mult = 4
    hf.hc_eps = 1e-6
    hf.hc_sinkhorn_iters = 2
    hf.n_routed_experts = 384
    hf.n_shared_experts = 1
    hf.expert_dtype = "fp8"
    hf.engram_layer_ids = []
    cfg.model_config.architecture = architecture
    cfg.model_config.hf_text_config = hf
    cfg.model_config.use_mla = True
    cfg.cache_config.block_size = block_size
    cfg.cache_config.enable_prefix_caching = False
    cfg.compilation_config.compile_sizes = [8, 16, 32]
    cfg.engram_config = None
    cfg.scheduler_config.max_num_batched_tokens = 32
    cfg.scheduler_config.max_num_seqs = 8
    cfg.scheduler_config.async_scheduling = False
    cfg.parallel_config = SimpleNamespace(
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=1,
        data_parallel_size=dp_size,
        decode_context_parallel_size=dcp_size,
        prefill_context_parallel_size=pcp_size,
        enable_expert_parallel=enable_ep,
        enable_elastic_ep=False,
        pcp_shard_decode_requests=False,
        cp_kv_cache_interleave_size=1,
    )
    attn_test._ACTIVE_VLLM_CONFIG[:] = [cfg]
    return cfg


@contextmanager
def _pallas_interpret_ops(monkeypatch: pytest.MonkeyPatch):
    """Route DSV4.1 custom ops through the real Pallas kernels with interpret=True."""
    attn_test._patch_vllm_attention_deps(monkeypatch)

    def _j(x: Any) -> Any:
        if not isinstance(x, torch.Tensor):
            return x
        arr = x.detach().cpu()
        if arr.dtype == torch.bfloat16:
            return jnp.asarray(arr.float().numpy()).astype(jnp.bfloat16)
        return jnp.asarray(arr.numpy())

    def _t(x: Any) -> Any:
        if isinstance(x, tuple):
            return tuple(_t(v) for v in x)
        if x.dtype == jnp.bfloat16:
            return torch.from_numpy(
                np.array(x.astype(jnp.float32), copy=True)
            ).bfloat16()
        return torch.from_numpy(np.array(x, copy=True))

    def _cpu_o_proj(
        o: torch.Tensor,
        pos: torch.Tensor,
        cs: torch.Tensor,
        w: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        pos_i32 = pos.clamp(0, cs.shape[0] - 1).int()
        cs_inv = cs.clone()
        cs_inv[:, cs.shape[1] // 2 :] = -cs_inv[:, cs.shape[1] // 2 :]
        o_rot = attn_test._cpu_kv_rope(o, pos_i32, cs_inv).float()
        t, g_cnt = o.shape[0], s.shape[0] // 128
        z = torch.einsum(
            "tgd,dgr->tgr", o_rot.view(t, g_cnt, -1), w.float().view(-1, g_cnt, 128)
        )
        return (z.reshape(t, -1) * s.float()).bfloat16()

    monkeypatch.setattr(
        DeepseekV41Attention, "o_proj_op", property(lambda _s: _cpu_o_proj)
    )

    mesh = jax.sharding.AbstractMesh(
        (), (), abstract_device=jax.sharding.AbstractDevice("TPU7x", 1, "tpu")
    )
    with (
        jax.default_device(jax.devices("cpu")[0]),
        jax.sharding.use_abstract_mesh(mesh),
    ):
        monkeypatch.setattr(
            dsv41_ops.cache_op,
            "dsv41_write_sub_rows",
            lambda c, u, s, off=0, after=None, page_tokens=16: _t(
                write_sub_rows(
                    _j(c),
                    _j(u),
                    _j(s),
                    slot_offset=int(off),
                    after=_j(after),
                    page_tokens=int(page_tokens),
                    interpret=True,
                )
            ),
        )
        monkeypatch.setattr(
            dsv41_ops.cache_op,
            "dsv41_gather_cache_rows",
            lambda c, idx, hi=None: _t(
                gather_cache_rows(_j(c), _j(idx), pool_hi=_j(hi), interpret=True)
            ),
        )
        monkeypatch.setattr(
            dsv41_ops.cache_op,
            "dsv41_pack_fp8_kv_rows",
            lambda lat: _t(pack_fp8_kv_rows(_j(lat), interpret=True)),
        )
        monkeypatch.setattr(
            dsv41_ops.indexer_op,
            "dsv41_paged_indexer_topk",
            lambda q, w, kl, bt, sl, cu, tk, kh=None, ci=None, cr=1: _t(
                paged_indexer_topk(
                    _j(q),
                    _j(w),
                    _j(kl),
                    _j(bt),
                    _j(sl),
                    _j(cu),
                    topk=int(tk),
                    k_pool_hi=_j(kh),
                    candidate_indices=_j(ci),
                    compress_ratio=int(cr),
                    interpret=True,
                )
            ),
        )
        monkeypatch.setattr(
            dsv41_ops.attention_op,
            "dsv41_gather_sparse_fp8_kv",
            lambda kl, rl, tk, bt, cu, ps, kh=None, rh=None: _t(
                gather_sparse_fp8_kv(
                    _j(kl),
                    _j(rl),
                    _j(tk),
                    _j(bt),
                    _j(cu),
                    page_size=int(ps),
                    kv_pool_hi=_j(kh),
                    rope_pool_hi=_j(rh),
                    interpret=True,
                )
            ),
        )
        monkeypatch.setattr(
            dsv41_ops.attention_op,
            "dsv41_sparse_mla_swa_attention",
            lambda q, gn, gr, tk, sk, sl, cu, sc, win, swa=None, cr=1, mp=None: _t(
                sparse_mla_swa_attention(
                    _j(q),
                    _j(gn),
                    _j(gr),
                    _j(tk),
                    _j(sk),
                    _j(sl),
                    _j(cu),
                    sm_scale=float(sc),
                    swa_window=int(win),
                    swa_kv=_j(swa),
                    kv_compress_ratio=int(cr),
                    swa_min_pos=_j(mp),
                    interpret=True,
                )
            ),
        )
        yield


def _bind_5layer_stack(
    vcfg: Any, max_tokens: int = 32, seed: int = 77
) -> list[DeepseekV41Attention]:
    attn_test._ACTIVE_VLLM_CONFIG[:] = [vcfg]
    vcfg.compilation_config.static_forward_context.clear()
    torch.manual_seed(seed)
    topk_buf = torch.full((max_tokens, 16), -1, dtype=torch.int32)
    cand_buf = torch.full((max_tokens, 4), -1, dtype=torch.int32)
    layers = [
        DeepseekV41Attention(
            vcfg,
            f"model.layers.{idx}.self_attn",
            topk_indices_buffer=topk_buf,
            candidate_block_buffer=cand_buf,
        )
        for idx in range(5)
    ]
    g_w = torch.Generator().manual_seed(seed + 100)
    num_blocks, bs = 16, vcfg.cache_config.block_size
    for attn in layers:
        out_rows = attn.wo_a.weight.shape[0]
        in_cols = attn.wo_a.weight.shape[1]
        w_fp8 = (
            torch.randn((out_rows, in_cols), generator=g_w, dtype=torch.float32) * 0.05
        ).to(torch.float8_e4m3fn)
        attn.wo_a.weight = nn.Parameter(w_fp8, requires_grad=False)
        attn.wo_a.weight_scale = nn.Parameter(
            torch.full((out_rows,), 0.05, dtype=torch.float32), requires_grad=False
        )
        attn.process_weights_after_loading(torch.bfloat16)
        attn.swa_cache_layer.bind_kv_cache(
            torch.zeros((num_blocks, bs, 512), dtype=torch.bfloat16),
            raw_cache=torch.zeros((num_blocks * bs * 2, 128), dtype=torch.int32),
        )
        if attn.compressor is not None and attn.compressor.state_cache is not None:
            attn.compressor.state_cache.bind_kv_cache(
                torch.zeros((num_blocks, bs, 1024), dtype=torch.float32),
                raw_cache=torch.zeros((num_blocks * bs * 8, 128), dtype=torch.int32),
            )
        if attn.indexer is not None and attn.indexer.owns_k:
            attn.indexer.k_cache.bind_kv_cache(
                torch.zeros((num_blocks, bs, 128), dtype=torch.bfloat16),
                raw_cache=torch.zeros((num_blocks * bs // 2, 128), dtype=torch.int32),
            )
        if attn.is_kv_source:
            attn.bind_kv_cache(
                torch.zeros((num_blocks, bs, 512), dtype=torch.uint8),
                raw_cache=torch.zeros((num_blocks * bs, 128), dtype=torch.int32),
                rope_pool=torch.zeros(
                    (num_blocks * max(bs // 4, 1), 128), dtype=torch.int32
                ),
            )
    return layers


def _make_step_meta(
    layers: list[DeepseekV41Attention],
    seq_lens: list[int],
    query_lens: list[int],
    *,
    pcp_size: int = 1,
) -> dict[str, AttentionMetadata]:
    num_reqs = len(seq_lens)
    bt = torch.arange(1, num_reqs * 4 + 1, dtype=torch.int32).reshape(num_reqs, 4)
    qsl = [0, *np.cumsum(query_lens).tolist()]
    desc = (
        PCP_STREAMING_SEQUENCE_LAYOUT_DESCRIPTOR
        if pcp_size > 1
        else DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR
    )
    meta = AttentionMetadata(
        input_positions=torch.zeros(max(qsl[-1], 1), dtype=torch.int32),
        block_tables=bt,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        query_start_loc=torch.tensor(qsl, dtype=torch.int32),
        sequence_layout_kind=desc.kind.value,
        sequence_layout_protocol=desc.protocol,
        sequence_layout_version=desc.version,
    )
    out: dict[str, AttentionMetadata] = {}
    for idx, attn in enumerate(layers):
        out[f"model.layers.{idx}.self_attn"] = meta
        out[attn.swa_cache_layer.prefix] = meta
        if attn.compressor is not None and attn.compressor.state_cache is not None:
            out[attn.compressor.state_cache.prefix] = meta
        if attn.indexer is not None:
            out[attn.indexer.k_cache.prefix] = meta
    return out


def _run_stack_step(
    monkeypatch: pytest.MonkeyPatch,
    layers: list[DeepseekV41Attention],
    hidden_global: torch.Tensor,
    pos_global: torch.Tensor,
    seq_lens: list[int],
    query_lens: list[int],
    *,
    pcp_size: int = 1,
    local_padded: int = 8,
    window_starts: list[int] | None = None,
    simulate_compiling: bool = False,
) -> torch.Tensor:
    total = int(hidden_global.shape[0])
    attn_meta = _make_step_meta(layers, seq_lens, query_lens, pcp_size=pcp_size)
    ws_tensor = (
        torch.zeros(len(seq_lens), dtype=torch.int32)
        if window_starts is None
        else torch.tensor(window_starts, dtype=torch.int32)
    )
    extra = {
        "deepseek_v41_window_starts": ws_tensor,
        "deepseek_v41_replay_starts": ws_tensor,
    }
    if pcp_size == 1:
        x = hidden_global.clone()
        with _mock_forward_context(monkeypatch, attn_meta, extra):
            for attn in layers:
                x = x + attn(pos_global, x, None)
        return x

    interleave = int(
        pcp_mod.split_size(torch.tensor(total, dtype=torch.int64), pcp_size).item()
    )
    rank_x: list[torch.Tensor] = []
    rank_pos: list[torch.Tensor] = []
    for r in range(pcp_size):
        s = min(r * interleave, total)
        n_loc = min(s + interleave, total) - s
        xh = torch.zeros(
            (local_padded, hidden_global.shape[1]), dtype=hidden_global.dtype
        )
        ph = torch.zeros(local_padded, dtype=torch.int32)
        if n_loc > 0:
            xh[:n_loc] = hidden_global[s : s + n_loc]
            ph[:n_loc] = pos_global[s : s + n_loc]
        rank_x.append(xh)
        rank_pos.append(ph)

    gathered_pos = torch.cat(rank_pos, dim=0)
    monkeypatch.setattr(
        pcp_mod, "all_gather_equal_tokens", lambda _t, dim=0: gathered_pos
    )
    steps = [
        PcpStep.build(
            attn_meta,
            rank_pos[r],
            torch.tensor(r, dtype=torch.int32),
            pcp_size,
            window_starts=ws_tensor,
        )
        for r in range(pcp_size)
    ]
    if simulate_compiling:
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    shared_topk = layers[0].topk_indices_buffer
    shared_cand = layers[0].candidate_block_buffer
    assert shared_topk is not None and shared_cand is not None
    rank_topk = [torch.full_like(shared_topk, -1) for _ in range(pcp_size)]
    rank_cand = [torch.full_like(shared_cand, -1) for _ in range(pcp_size)]

    with _mock_forward_context(monkeypatch, attn_meta, extra):
        for attn in layers:
            kv_rot_locals: list[torch.Tensor] = []
            kv_scores: list[torch.Tensor | None] = []
            for r in range(pcp_size):
                qr_kv, kv_sc, _ = attn._run_parallel_input_projections(rank_x[r])
                _, _, kv_lat = attn._split_qkv_and_norm(qr_kv)
                kv_rot_locals.append(attn.rotate(kv_lat, rank_pos[r]))
                kv_scores.append(kv_sc)

            gathered_kv_rot = torch.cat(kv_rot_locals, dim=0)
            gathered_kv_sc = (
                torch.cat([cast(torch.Tensor, t) for t in kv_scores], dim=0)
                if kv_scores[0] is not None
                else None
            )
            monkeypatch.setattr(
                pcp_mod,
                "all_gather_equal_tokens",
                lambda local_t,
                dim=0,
                *,
                hd=attn.head_dim,
                g_rot=gathered_kv_rot,
                g_sc=gathered_kv_sc: (
                    g_rot if local_t.shape[-1] == hd else cast(torch.Tensor, g_sc)
                ),
            )
            swa_pcr = attn.swa_cache_layer.paged_cache_rows
            assert swa_pcr is not None
            cache_banks: list[torch.Tensor] = [swa_pcr.raw]
            if attn.compressor is not None and attn.compressor.state_cache is not None:
                assert attn.compressor.state_cache.paged_cache_rows is not None
                cache_banks.append(attn.compressor.state_cache.paged_cache_rows.raw)
            if attn.indexer is not None and attn.indexer.owns_k:
                assert attn.indexer.k_cache.paged_cache_rows is not None
                cache_banks.append(attn.indexer.k_cache.paged_cache_rows.raw)
            if attn.is_kv_source:
                assert (
                    attn.paged_cache_rows is not None
                    and attn.paged_cache_rows.rope is not None
                )
                cache_banks.extend(
                    [attn.paged_cache_rows.raw, attn.paged_cache_rows.rope]
                )

            pre_step = [b.clone() for b in cache_banks]
            rank0_post: list[torch.Tensor] = []
            for r in range(pcp_size):
                if r > 0:
                    for b, pre in zip(cache_banks, pre_step):
                        b.copy_(pre)
                shared_topk.copy_(rank_topk[r])
                shared_cand.copy_(rank_cand[r])
                with pcp_mod.step_scope(steps[r]):
                    proj = attn(rank_pos[r], rank_x[r], None)
                rank_topk[r].copy_(shared_topk)
                rank_cand[r].copy_(shared_cand)
                if simulate_compiling:
                    assert swa_pcr._last_read is None
                if r == 0:
                    rank0_post = [b.clone() for b in cache_banks]
                else:
                    for b, ref in zip(cache_banks, rank0_post):
                        assert torch.equal(b, ref)
                rank_x[r] = rank_x[r] + proj

    return torch.cat(
        [
            rank_x[r][: min((r + 1) * interleave, total) - min(r * interleave, total)]
            for r in range(pcp_size)
            if min(r * interleave, total) < total
        ],
        dim=0,
    )


def test_pcp_step_compaction_and_idle_ranks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify PcpStep request compaction, local_window_starts, and idle ranks."""
    meta = AttentionMetadata(
        input_positions=torch.zeros(9, dtype=torch.int32),
        block_tables=torch.tensor([[1, 2], [3, 4], [5, 6], [0, 0]], dtype=torch.int32),
        seq_lens=torch.tensor([10, 6, 15, 0], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1, 6, 9, 9], dtype=torch.int32),
        sequence_layout_kind="partial",
        sequence_layout_protocol="pcp_streaming",
        sequence_layout_version=1,
    )
    ws = torch.tensor([8, 0, 4, 0], dtype=torch.int32)
    pos_global = torch.tensor(
        [9, 0, 1, 2, 3, 4, 12, 13, 14, 0, 0, 0], dtype=torch.int32
    )
    monkeypatch.setattr(
        pcp_mod, "all_gather_equal_tokens", lambda _t, dim=0: pos_global
    )

    s0 = PcpStep.build(
        {"l0": meta}, pos_global[0:3], torch.tensor(0), 4, window_starts=ws
    )
    m0 = s0.local(meta)
    assert s0.local(meta) is m0
    assert cast(torch.Tensor, m0.query_start_loc).tolist() == [0, 1, 3, 3, 3]
    assert cast(torch.Tensor, m0.seq_lens).tolist() == [10, 3, 0, 0]
    assert s0.local_window_starts is not None and s0.local_window_starts.tolist() == [
        8,
        0,
        0,
        0,
    ]

    s2 = PcpStep.build(
        {"l0": meta}, pos_global[6:9], torch.tensor(2), 4, window_starts=ws
    )
    m2 = s2.local(meta)
    assert cast(torch.Tensor, m2.query_start_loc).tolist() == [0, 3, 3, 3, 3]
    assert cast(torch.Tensor, m2.seq_lens).tolist() == [15, 0, 0, 0]
    assert cast(torch.Tensor, m2.block_tables)[0].tolist() == [5, 6]
    assert s2.local_window_starts is not None and s2.local_window_starts.tolist() == [
        4,
        0,
        0,
        0,
    ]

    s3 = PcpStep.build(
        {"l0": meta}, pos_global[9:12], torch.tensor(3), 4, window_starts=ws
    )
    m3 = s3.local(meta)
    assert int(s3.num_local.item()) == 0
    assert cast(torch.Tensor, m3.seq_lens).tolist() == [0, 0, 0, 0]
    assert cast(torch.Tensor, m3.query_start_loc).tolist() == [0, 0, 0, 0, 0]


@pytest.mark.parametrize("pcp_size", [2, 4])
def test_multirank_attention_parity(
    pcp_size: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify CP=1 vs PCP=2/4 parity across 5 layers, odd chunks, SWA, and decode."""
    with _pallas_interpret_ops(monkeypatch):
        gen = torch.Generator().manual_seed(2026 + pcp_size)
        cfg_ref = _pcp_config(pcp_size=1)
        cfg_pcp = _pcp_config(pcp_size=pcp_size)
        layers_ref = _bind_5layer_stack(cfg_ref, seed=91)
        layers_pcp = _bind_5layer_stack(cfg_pcp, seed=91)

        steps = [
            ([21, 3, 2], [21, 3, 2], None, 16, True),
            ([22, 7], [1, 4], [6, 0], 8, False),
        ]
        for s_lens, q_lens, ws, loc_pad, compiling in steps:
            h = torch.randn(
                (sum(q_lens), 64), generator=gen, dtype=torch.float32
            ).bfloat16()
            pos = torch.cat(
                [
                    torch.arange(sl - ql, sl, dtype=torch.int32)
                    for sl, ql in zip(s_lens, q_lens)
                ],
                dim=0,
            )
            attn_test._ACTIVE_VLLM_CONFIG[:] = [cfg_ref]
            out_ref = _run_stack_step(
                monkeypatch,
                layers_ref,
                h,
                pos,
                s_lens,
                q_lens,
                pcp_size=1,
                window_starts=ws,
            )
            attn_test._ACTIVE_VLLM_CONFIG[:] = [cfg_pcp]
            out_pcp = _run_stack_step(
                monkeypatch,
                layers_pcp,
                h,
                pos,
                s_lens,
                q_lens,
                pcp_size=pcp_size,
                local_padded=loc_pad,
                window_starts=ws,
                simulate_compiling=compiling,
            )
            torch.testing.assert_close(out_pcp, out_ref, atol=1e-2, rtol=1e-2)


def test_host_engram_pcp_parity_and_multichunk_async_indexing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify host 4-gram Engram hashing under PCP=4 and multi-chunk async indexing."""
    state = engram_test._make_ngram_state(monkeypatch)
    tok_hist = np.zeros((2, 16), dtype=np.int32)
    tok_hist[0, :7] = [10, 11, 12, 13, 14, 15, 16]
    tok_hist[1, :5] = [20, 21, 22, 29, 0]

    ref_tokens = torch.tensor([13, 14, 15, 16, 31], dtype=torch.int32)
    ref_pos = torch.tensor([3, 4, 5, 6, 4], dtype=torch.int32)
    ref_qsl = torch.tensor([0, 4, 5], dtype=torch.int32)
    ref_lb = torch.tensor([[12, 11, 10], [29, 22, 21]], dtype=torch.int32)
    ref_hashes = state.hash_on_host(
        ref_tokens,
        ref_pos,
        ref_qsl,
        torch.zeros_like(ref_tokens, dtype=torch.bool),
        ref_lb,
        torch.zeros_like(ref_lb, dtype=torch.bool),
    )

    pcp_size = 4
    vcfg = _pcp_config(pcp_size=pcp_size)
    collected: list[torch.Tensor] = []
    for r in range(pcp_size):
        monkeypatch.setattr(
            engram_mod,
            "get_pcp_group",
            lambda rank=r: SimpleNamespace(rank_in_group=rank, world_size=pcp_size),
        )
        # Simulate multi-chunk AsyncPreResults where chunk 0 had 1 discarded partial
        # prefill request (padded to 8 in req_id_to_index_copy) and chunk 1 had "r1"
        # at unpadded index 1 in sampled_token_ids_cpu (padded offset 8 in TPU tensor).
        runner = SimpleNamespace(
            vllm_config=vcfg,
            parallel_config=vcfg.parallel_config,
            num_tokens_paddings=[4, 8],
            max_num_tokens=8,
            _dp_target_bucket=None,
            uses_mrope=False,
            supports_mm_inputs=False,
            arange_np=np.arange(16, dtype=np.int64),
            input_ids_cpu=torch.tensor([13, 14, 15, 16, 0, 0, 0, 0], dtype=torch.int32),
            positions_cpu=torch.tensor([3, 4, 5, 6, 4, 0, 0, 0], dtype=torch.int32),
            query_start_loc_np=np.array([0, 4, 5, 5], dtype=np.int32),
            seq_lens_np=np.array([7, 5, 0], dtype=np.int32),
            input_batch=SimpleNamespace(
                num_reqs=2,
                req_ids=["r0", "r1"],
                req_id_to_index={"r0": 0, "r1": 1},
                num_computed_tokens_cpu=np.array([3, 4], dtype=np.int32),
                num_prompt_tokens=np.array([7, 3], dtype=np.int32),
                num_tokens_no_spec=np.array([7, 5], dtype=np.int32),
                token_ids_cpu=tok_hist.copy(),
            ),
            _pre_async_results=SimpleNamespace(
                req_id_to_index_copy={"r0": 0, "r1": 8},
                request_seq_lens=[(0, None, 3, "r0"), (1, None, 5, "r1")],
                discard_sampled_tokens_req_indices=[0],
                wait_for_copy=lambda: torch.tensor([[99], [31]], dtype=torch.int32),
            ),
        )
        runner.positions_np = runner.positions_cpu.numpy()
        prepared = prepare_pcp_sequence_layout(
            runner,
            SimpleNamespace(num_scheduled_tokens={"r0": 4, "r1": 1}),
            0,
            2,
            np.array([4, 1], dtype=np.int32),
            5,
            True,
            2,
            2,
            get_native_pcp_rank=lambda rank=r: rank,
            get_native_pcp_world_size=lambda: pcp_size,
        )
        runner._last_sequence_layout_plan = SequenceLayoutPlan(
            descriptor=PCP_STREAMING_SEQUENCE_LAYOUT_DESCRIPTOR,
            token_slice=prepared.local_token_slice,
            global_num_tokens=5,
            global_padded_num_tokens=prepared.padded_total_num_scheduled_tokens,
            local_num_tokens=prepared.local_total_num_scheduled_tokens,
            local_padded_num_tokens=prepared.local_padded_total_num_scheduled_tokens,
            _packed_to_request_major_token_indices=(
                prepared.packed_to_request_major_token_indices
            ),
            _request_major_to_packed_token_indices=(
                prepared.request_major_to_packed_token_indices
            ),
        )
        host_in = capture_host_engram_inputs(runner, 2, 0, state.lookback_depth)
        assert host_in is not None
        if host_in.num_local > 0:
            dead = model_mod.upstream.image_sentinel_mask(host_in.tokens)
            lb_dead = model_mod.upstream.image_sentinel_mask(host_in.lookback)
            all_h = state.hash_on_host(
                host_in.tokens,
                host_in.positions,
                host_in.query_start_loc,
                dead,
                host_in.lookback,
                lb_dead,
            )
            collected.append(
                all_h[host_in.local_start : host_in.local_start + host_in.num_local]
            )
    assert torch.equal(torch.cat(collected, dim=0), ref_hashes)

    dummy_plan = PcpSequenceLayoutPlanner.from_vllm_config(vcfg).prepare_dummy(
        num_tokens=4, num_reqs=2, kv_cache_initialized=True
    )
    assert dummy_plan.is_dummy is True
    dummy_in = capture_host_engram_inputs(
        SimpleNamespace(_last_sequence_layout_plan=dummy_plan), 2, 0, 3
    )
    assert (
        dummy_in is not None and dummy_in.num_local == 0 and dummy_in.local_tokens == 4
    )


def test_device_engram_hashes_pcp_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _device_engram_hashes fallback under PCP=2 matches CP=1 hashes."""
    state = engram_test._make_ngram_state(monkeypatch)
    state.use_slot_cache = False
    state.swa_cache_module.paged_cache_rows = None

    ref_tokens = torch.tensor([13, 14, 15, 16, 31], dtype=torch.int32)
    ref_pos = torch.tensor([3, 4, 5, 6, 4], dtype=torch.int32)
    ref_qsl = torch.tensor([0, 4, 5], dtype=torch.int32)
    ref_lb = torch.tensor([[12, 11, 10], [29, 22, 21]], dtype=torch.int32)
    ref_hashes = state.hash_on_host(
        ref_tokens,
        ref_pos,
        ref_qsl,
        torch.zeros_like(ref_tokens, dtype=torch.bool),
        ref_lb,
        torch.zeros_like(ref_lb, dtype=torch.bool),
    )

    meta = AttentionMetadata(
        input_positions=torch.zeros(5, dtype=torch.int32),
        block_tables=torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
        seq_lens=torch.tensor([7, 5], dtype=torch.int32),
        query_start_loc=ref_qsl,
        sequence_layout_kind="partial",
    )
    packed_ids = torch.tensor([13, 14, 15, 16, 31, 0], dtype=torch.int32)
    packed_pos = torch.tensor([3, 4, 5, 6, 4, 0], dtype=torch.int32)
    monkeypatch.setattr(
        pcp_mod,
        "all_gather_equal_tokens",
        lambda x, dim=0: packed_ids if x[0].item() in (13, 16) else packed_pos,
    )
    dev_slices: list[torch.Tensor] = []
    for r, sl in ((0, slice(0, 3)), (1, slice(3, 6))):
        stub = SimpleNamespace(
            engram_hash=state,
            _pcp_size=2,
            _pcp_rank=torch.tensor(r, dtype=torch.int32),
        )
        stub._pcp_step = lambda pos, s=stub: DeepseekV41Model._pcp_step(
            cast(Any, s), pos
        )
        with _mock_forward_context(monkeypatch, {"swa": meta}):
            h_r, m_r = DeepseekV41Model._device_engram_hashes(
                cast(Any, stub), packed_ids[sl], packed_pos[sl], meta, ref_lb
            )
        dev_slices.append(h_r[m_r])
    assert torch.equal(torch.cat(dev_slices, dim=0), ref_hashes)


def test_prepare_host_lookback_step_gating(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify CP=1 HBM Engram skips host capture while PCP or staged Engram runs it."""
    captured: list[int] = []
    monkeypatch.setattr(
        engram_mod,
        "capture_host_engram_inputs",
        lambda _r, n, _s, _d: (captured.append(n), SimpleNamespace(num_local=n))[1],
    )
    hbm_layer = DeepseekV41DecoderLayer.__new__(DeepseekV41DecoderLayer)
    hbm_layer.engram = cast(Any, SimpleNamespace(is_staged=lambda: False))
    staged_layer = DeepseekV41DecoderLayer.__new__(DeepseekV41DecoderLayer)
    staged_layer.engram = cast(Any, SimpleNamespace(is_staged=lambda: True))

    runner = SimpleNamespace(speculative_config=None)
    cp1_hbm = SimpleNamespace(
        model=SimpleNamespace(
            engram_hash=SimpleNamespace(lookback_depth=3),
            _pcp_size=1,
            layers=[hbm_layer],
            _host_engram_inputs=None,
        )
    )
    assert (
        DeepseekV41ForCausalLM.prepare_host_lookback_step(cast(Any, cp1_hbm), runner, 2)
        is False
    )
    assert captured == []

    pcp_hbm = SimpleNamespace(
        model=SimpleNamespace(
            engram_hash=SimpleNamespace(lookback_depth=3),
            _pcp_size=8,
            layers=[hbm_layer],
            _host_engram_inputs=None,
        )
    )
    assert (
        DeepseekV41ForCausalLM.prepare_host_lookback_step(cast(Any, pcp_hbm), runner, 2)
        is True
    )
    cp1_staged = SimpleNamespace(
        model=SimpleNamespace(
            engram_hash=SimpleNamespace(lookback_depth=3),
            _pcp_size=1,
            layers=[staged_layer],
            _host_engram_inputs=None,
        )
    )
    assert (
        DeepseekV41ForCausalLM.prepare_host_lookback_step(
            cast(Any, cp1_staged), runner, 3
        )
        is True
    )
    assert captured == [2, 3]


def test_engram_dp_group_pcp_patch_and_qwen4_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify _patch_vllm_engram_dp_group_pcp extends DSV4.1 EDP and skips Qwen4Exp."""
    import vllm.config as vllm_cfg_pkg
    from vllm.config.engram import EngramConfig

    monkeypatch.setattr(
        vllm_cfg_pkg,
        "get_current_vllm_config_or_none",
        lambda: attn_test._ACTIVE_VLLM_CONFIG[0],
    )
    monkeypatch.setattr(
        EngramConfig, "verify_model_config", EngramConfig.verify_model_config
    )
    monkeypatch.setattr(
        EngramConfig, "resolve_dp_shared_memory", EngramConfig.resolve_dp_shared_memory
    )
    monkeypatch.setattr(
        EngramConfig, "verify_parallel_config", EngramConfig.verify_parallel_config
    )
    monkeypatch.setattr(ps_mod, "_tpu_engram_dp_pcp_patch", False, raising=False)
    monkeypatch.setattr(
        ps_mod, "_engram_dp_shard_size", lambda w, d, r: min(d, max(w // max(r, 1), 1))
    )
    built_edp_ranks: list[list[list[int]]] = []
    seen_defer_edp: list[int] = []
    monkeypatch.setattr(
        ps_mod,
        "initialize_model_parallel",
        lambda **_kw: seen_defer_edp.append(ps_mod._engram_dp_shard_size(8, 1, 8)),
    )
    monkeypatch.setattr(patch_registry, "_applied", set())
    patch_registry.apply("engine_core")

    cfg_pcp8 = _pcp_config(pcp_size=8, tp_size=1, dp_size=1)
    cfg_pcp8.model_config.hf_text_config.engram_layer_ids = [1]
    cfg_pcp8.engram_config = EngramConfig()
    cfg_pcp8.engram_config.verify_model_config(cfg_pcp8.model_config)
    cfg_pcp8.engram_config.resolve_dp_shared_memory(cfg_pcp8.parallel_config)
    assert cfg_pcp8.engram_config.dp_shared_memory is True
    cfg_pcp8.engram_config.verify_parallel_config(cfg_pcp8.parallel_config)
    assert ps_mod._engram_dp_shard_size(8, 1, 8) == 8

    # Qwen4Exp PLE uses EngramConfig without sharing DSV4.1's PCP EDP group.
    cfg_qwen4 = _pcp_config(
        pcp_size=8, tp_size=1, dp_size=1, architecture="Qwen4ExpForCausalLM"
    )
    cfg_qwen4.model_config.hf_text_config.ple_layer_ids = [0]
    cfg_qwen4.engram_config = EngramConfig()
    cfg_qwen4.engram_config.verify_model_config(cfg_qwen4.model_config)
    cfg_qwen4.engram_config.resolve_dp_shared_memory(cfg_qwen4.parallel_config)
    assert cfg_qwen4.engram_config.dp_shared_memory is False
    assert ps_mod._engram_dp_shard_size(8, 1, 8) == 1

    attn_test._ACTIVE_VLLM_CONFIG[:] = [cfg_pcp8]
    monkeypatch.setattr(ps_mod, "_ENGRAM_DP", None)
    monkeypatch.setattr(
        ps_mod,
        "get_world_group",
        lambda: SimpleNamespace(world_size=8, local_rank=0, device_group="g"),
    )
    monkeypatch.setattr(
        ps_mod,
        "init_model_parallel_group",
        lambda group_ranks, *_a, group_name="", **_kw: (
            built_edp_ranks.append(group_ranks) if group_name == "edp" else None,
            SimpleNamespace(group_ranks=group_ranks, group_name=group_name),
        )[1],
    )
    ps_mod.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        prefill_context_model_parallel_size=8,
        decode_context_model_parallel_size=1,
        backend="gloo",
    )
    assert seen_defer_edp == [1] and built_edp_ranks == [[list(range(8))]]


def test_engram_cpu_hash_gather_and_row_slice(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify engram_token_slot fallback, CPU hash all_gather, and row slicing."""
    cfg_pcp8 = _pcp_config(pcp_size=8, tp_size=1, dp_size=1)
    dp_meta = SimpleNamespace(
        num_tokens_across_dp_cpu=torch.tensor([3, 6], dtype=torch.int32)
    )
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *_a, **_kw: self)
    extra_kw = TpuPlatform.set_additional_forward_context(
        attn_metadata={}, vllm_config=cfg_pcp8, dp_metadata=dp_meta
    )
    assert extra_kw["tpu_num_tokens_across_dp_cpu"] == (3, 6)

    with _mock_forward_context(monkeypatch, {}, {}, dp_metadata=dp_meta):
        assert engram_mod.engram_token_slot(5) == 5

    cpu_ag_calls: list[int] = []

    def _fake_dist_ag(
        chunks: list[torch.Tensor], tensor: torch.Tensor, group: Any = None
    ) -> None:
        assert group == "fake_cpu_group" and tensor.device.type == "cpu"
        cpu_ag_calls.append(tensor.shape[0])
        for idx, ch in enumerate(chunks):
            ch.copy_(tensor + idx * 100)

    monkeypatch.setattr(engram_mod.dist, "all_gather", _fake_dist_ag)
    monkeypatch.setattr(
        engram_mod,
        "get_engram_dp_group",
        lambda: SimpleNamespace(world_size=2, cpu_group="fake_cpu_group"),
    )
    with _mock_forward_context(monkeypatch, {}, extra_kw, dp_metadata=dp_meta):
        assert engram_mod.engram_token_slot(2) == 6
        gathered = engram_mod.gather_engram_hashes(
            torch.ones((3, 2, 4), dtype=torch.int32), dp_shared_memory=False
        )
        assert cpu_ag_calls == [6] and gathered.shape == (12, 2, 4)

    dp_size, slot, local_heads, width = 4, 6, 2, 8
    per_head_shards = [
        torch.arange(dp_size * slot * local_heads * width, dtype=torch.float32).reshape(
            dp_size * slot, local_heads, width
        )
        + h_shard * 1000.0
        for h_shard in range(dp_size)
    ]
    monkeypatch.setattr(
        engram_mod,
        "get_engram_dp_group",
        lambda: SimpleNamespace(
            world_size=dp_size,
            rank_in_group=3,
            all_gather=lambda _x, dim: torch.cat(per_head_shards, dim=dim),
        ),
    )
    rank3_rows = engram_mod._dp_gather_engram_rows(per_head_shards[0], num_tokens=5)
    expected_rank3 = torch.cat(
        [shard[3 * slot : 3 * slot + 5] for shard in per_head_shards], dim=1
    )
    torch.testing.assert_close(rank3_rows, expected_rank3, atol=0.0, rtol=0.0)


def test_pcp_config_validation_and_layout_caps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify PCP config guards, CED auto-disable, bucket caps, and EP*PCP sharding."""
    import vllm_torchtpu.models.vllm as models_init_mod

    monkeypatch.setattr(
        models_init_mod.DeepseekV4ForCausalLMConfig,
        "verify_and_update_config",
        staticmethod(lambda _c: None),
    )
    with pytest.raises(NotImplementedError, match="--tensor-parallel-size=1"):
        _TPUDeepseekV41Config.verify_and_update_config(
            _pcp_config(pcp_size=4, tp_size=2, enable_ep=True)
        )
    with pytest.raises(NotImplementedError, match="does not support DCP"):
        _TPUDeepseekV41Config.verify_and_update_config(
            _pcp_config(pcp_size=4, tp_size=1, dcp_size=2, enable_ep=True)
        )
    with pytest.raises(NotImplementedError, match="--data-parallel-size=1"):
        _TPUDeepseekV41Config.verify_and_update_config(
            _pcp_config(pcp_size=4, tp_size=1, dp_size=2, enable_ep=True)
        )
    bad_spec = _pcp_config(pcp_size=4, tp_size=1, enable_ep=True)
    bad_spec.speculative_config = SimpleNamespace(method="dspark")
    with pytest.raises(NotImplementedError, match="speculative decoding"):
        _TPUDeepseekV41Config.verify_and_update_config(bad_spec)
    with pytest.raises(NotImplementedError, match="--enable-expert-parallel"):
        _TPUDeepseekV41Config.verify_and_update_config(
            _pcp_config(pcp_size=4, tp_size=1, enable_ep=False)
        )

    good_pcp = _pcp_config(pcp_size=8, tp_size=1, enable_ep=True)
    _TPUDeepseekV41Config.verify_and_update_config(good_pcp)

    assert model_mod._ced_config(_pcp_config(pcp_size=1)) == (True, 3)
    assert model_mod._ced_config(good_pcp) == (False, 3)

    elig_v41 = PcpSequenceLayoutEligibility.from_vllm_config(good_pcp)
    assert elig_v41.contiguous_split is True
    planner_v41 = PcpSequenceLayoutPlanner(elig_v41)
    assert planner_v41.retains_request_replay_starts is True
    assert planner_v41.cap_local_token_paddings(
        [16, 64, 256, 512, 1024, 2048, 8192], max_batched_tokens=8192
    ) == [16, 64, 256, 512, 1024]

    non_mla = _pcp_config(pcp_size=2)
    non_mla.model_config.use_mla = False
    assert (
        PcpSequenceLayoutPlanner.from_vllm_config(non_mla).retains_request_replay_starts
        is False
    )
    assert (
        AllSequenceLayoutPlanner()
        .prepare_dummy(num_tokens=8, num_reqs=1, kv_cache_initialized=True)
        .is_dummy
    )

    with pytest.raises(NotImplementedError, match="--enable-expert-parallel"):
        DeepseekV41ForCausalLM(
            vllm_config=_pcp_config(pcp_size=8, tp_size=1, enable_ep=False)
        )
    with pytest.raises(NotImplementedError, match="DP=1"):
        DeepseekV41ForCausalLM(
            vllm_config=_pcp_config(pcp_size=8, tp_size=1, dp_size=2, enable_ep=True)
        )

    fake_self = SimpleNamespace(
        config=good_pcp.model_config.hf_config,
        model=SimpleNamespace(
            parallel_config=good_pcp.parallel_config,
            layers=[SimpleNamespace(ffn=SimpleNamespace(experts="exp"))],
        ),
    )
    DeepseekV41ForCausalLM.set_moe_parameters(cast(Any, fake_self))
    assert fake_self.num_local_physical_experts == 384 // 8
