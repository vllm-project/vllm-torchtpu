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

from contextlib import ExitStack, contextmanager
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
from vllm.config import ModelConfig

import vllm_torchtpu as torchtpu_root
from tests.models.vllm.deepseek_v41 import test_attention as attn_test
from tests.models.vllm.deepseek_v41 import test_engram as engram_test
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
    PCP_CONTIGUOUS_SEQUENCE_LAYOUT_DESCRIPTOR,
    PcpSequenceLayoutPlanner,
)
from vllm_torchtpu.layers.core.sequence_layout import (
    DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR,
)
from vllm_torchtpu.models.vllm import (
    _TPUDeepseekV41Config,
    install_pcp_model_capabilities,
)
from vllm_torchtpu.models.vllm.deepseek_v41 import (
    DeepseekV41Attention,
    DeepseekV41ForCausalLM,
    DeepseekV41Model,
)
from vllm_torchtpu.models.vllm.deepseek_v41 import engram as engram_mod
from vllm_torchtpu.models.vllm.deepseek_v41 import model as model_mod
from vllm_torchtpu.models.vllm.deepseek_v41 import pcp as pcp_mod
from vllm_torchtpu.models.vllm.deepseek_v41.pcp import PcpStep
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform
from vllm_torchtpu.utils import HostSequenceLookbackInputs


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
    architecture: str = "DeepseekV41ForCausalLM",
) -> Any:
    cfg = attn_test._make_vllm_config(max_model_len=64, static_ctx={})
    hf = cfg.model_config.hf_config
    hf.__dict__.update(
        sliding_window=16,
        max_position_embeddings=128,
        num_attention_heads=16,
        o_lora_rank=128,
        index_n_heads=8,
        index_topk=16,
        candidate_topk_blocks=4,
        candidate_block_size=4,
        vocab_size=256,
        hc_mult=4,
        hc_eps=1e-6,
        n_routed_experts=384,
        n_shared_experts=1,
        engram_layer_ids=[],
    )
    is_v41 = architecture == "DeepseekV41ForCausalLM"
    cfg.model_config.__dict__.update(
        architecture=architecture,
        hf_text_config=hf,
        use_mla=True,
        uses_contiguous_pcp_layout=is_v41,
        shares_engram_host_table_across_pcp=is_v41,
    )
    cfg.cache_config.block_size = 16
    cfg.cache_config.enable_prefix_caching = False
    cfg.compilation_config.compile_sizes = [8, 16, 32]
    cfg.engram_config = None
    cfg.scheduler_config.max_num_batched_tokens = 32
    cfg.scheduler_config.async_scheduling = False
    cfg.parallel_config.__dict__.update(
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=1,
        data_parallel_size=dp_size,
        decode_context_parallel_size=dcp_size,
        prefill_context_parallel_size=pcp_size,
        enable_expert_parallel=enable_ep,
        enable_elastic_ep=False,
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
        return (
            jnp.asarray(arr.float().numpy()).astype(jnp.bfloat16)
            if arr.dtype == torch.bfloat16
            else jnp.asarray(arr.numpy())
        )

    def _t(x: Any) -> Any:
        if isinstance(x, tuple):
            return tuple(_t(v) for v in x)
        return (
            torch.from_numpy(np.array(x.astype(jnp.float32), copy=True)).bfloat16()
            if x.dtype == jnp.bfloat16
            else torch.from_numpy(np.array(x, copy=True))
        )

    def _cpu_o_proj(o: Any, pos: Any, cs: Any, w: Any, s: Any) -> torch.Tensor:
        cs_inv = cs.clone()
        cs_inv[:, cs.shape[1] // 2 :] *= -1
        o_rot = attn_test._cpu_kv_rope(o, pos.clamp(0, cs.shape[0] - 1).int(), cs_inv)
        t, g_cnt = o.shape[0], s.shape[0] // 128
        z = torch.einsum(
            "tgd,dgr->tgr",
            o_rot.float().view(t, g_cnt, -1),
            w.float().view(-1, g_cnt, 128),
        )
        return (z.reshape(t, -1) * s.float()).bfloat16()

    def _bind(mod: Any, attr: str, fn: Any, n_pos: int, kw_spec: tuple[Any, ...]):
        spec_map = dict(kw_spec)

        def _call(*args: Any, **call_kw: Any) -> Any:
            pos_args = [_j(a) for a in args[:n_pos]]
            kw: dict[str, Any] = {"interpret": True}
            for (k_name, conv), val in zip(kw_spec, args[n_pos:]):
                kw[k_name] = _j(val) if conv is None else conv(val)
            for k_name, val in call_kw.items():
                conv = spec_map.get(k_name)
                kw[k_name] = _j(val) if conv is None else conv(val)
            return _t(fn(*pos_args, **kw))

        monkeypatch.setattr(mod, attr, _call)

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
        _bind(
            dsv41_ops.cache_op,
            "dsv41_write_sub_rows",
            write_sub_rows,
            3,
            (("slot_offset", int), ("after", None), ("page_tokens", int)),
        )
        _bind(
            dsv41_ops.cache_op,
            "dsv41_gather_cache_rows",
            gather_cache_rows,
            2,
            (("pool_hi", None),),
        )
        _bind(dsv41_ops.cache_op, "dsv41_pack_fp8_kv_rows", pack_fp8_kv_rows, 1, ())
        _bind(
            dsv41_ops.indexer_op,
            "dsv41_paged_indexer_topk",
            paged_indexer_topk,
            6,
            (
                ("topk", int),
                ("k_pool_hi", None),
                ("candidate_indices", None),
                ("compress_ratio", int),
            ),
        )
        _bind(
            dsv41_ops.attention_op,
            "dsv41_gather_sparse_fp8_kv",
            gather_sparse_fp8_kv,
            5,
            (("page_size", int), ("kv_pool_hi", None), ("rope_pool_hi", None)),
        )
        _bind(
            dsv41_ops.attention_op,
            "dsv41_sparse_mla_swa_attention",
            sparse_mla_swa_attention,
            7,
            (
                ("sm_scale", float),
                ("swa_window", int),
                ("swa_kv", None),
                ("kv_compress_ratio", int),
                ("swa_min_pos", None),
            ),
        )
        yield


def _bind_stack(
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
        for idx in range(vcfg.model_config.hf_config.num_hidden_layers)
    ]
    g_w = torch.Generator().manual_seed(seed + 100)
    num_blocks, bs = 16, vcfg.cache_config.block_size
    for attn in layers:
        out_rows, in_cols = attn.wo_a.weight.shape
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


def _run_stack_step(
    monkeypatch: pytest.MonkeyPatch,
    layers: list[DeepseekV41Attention],
    hidden_global: torch.Tensor,
    pos_global: torch.Tensor,
    seq_lens: list[int],
    query_lens: list[int],
    *,
    num_decodes: int = 0,
    pcp_size: int = 1,
    local_padded: int = 8,
    window_starts: list[int] | None = None,
    simulate_compiling: bool = False,
) -> torch.Tensor:
    total, num_reqs = int(hidden_global.shape[0]), len(seq_lens)
    qsl = [0, *np.cumsum(query_lens).tolist()]
    desc = (
        PCP_CONTIGUOUS_SEQUENCE_LAYOUT_DESCRIPTOR
        if pcp_size > 1
        else DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR
    )
    meta = AttentionMetadata(
        input_positions=torch.zeros(max(qsl[-1], 1), dtype=torch.int32),
        block_tables=torch.arange(1, num_reqs * 4 + 1, dtype=torch.int32).reshape(
            num_reqs, 4
        ),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        query_start_loc=torch.tensor(qsl, dtype=torch.int32),
        request_distribution=torch.tensor(
            [num_decodes, 0, num_reqs], dtype=torch.int32
        ),
        sequence_layout_kind=desc.kind.value,
        sequence_layout_protocol=desc.protocol,
        sequence_layout_version=desc.version,
    )
    attn_meta: dict[str, AttentionMetadata] = {}
    for idx, attn in enumerate(layers):
        attn_meta[f"model.layers.{idx}.self_attn"] = meta
        attn_meta[attn.swa_cache_layer.prefix] = meta
        if attn.compressor is not None and attn.compressor.state_cache is not None:
            attn_meta[attn.compressor.state_cache.prefix] = meta
        if attn.indexer is not None:
            attn_meta[attn.indexer.k_cache.prefix] = meta

    ws_tensor = (
        torch.zeros(num_reqs, dtype=torch.int32)
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
    rank_x, rank_pos = [], []
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
    with ExitStack() as stack:
        mp = stack.enter_context(monkeypatch.context())
        mp.setattr(pcp_mod, "all_gather_equal_tokens", lambda _t, dim=0: gathered_pos)
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
            mp.setattr(torch.compiler, "is_compiling", lambda: True)
        else:
            assert torch.compiler.is_compiling() is False

        shared_topk = layers[0].topk_indices_buffer
        shared_cand = layers[0].candidate_block_buffer
        assert shared_topk is not None and shared_cand is not None
        rank_topk = [torch.full_like(shared_topk, -1) for _ in range(pcp_size)]
        rank_cand = [torch.full_like(shared_cand, -1) for _ in range(pcp_size)]

        stack.enter_context(_mock_forward_context(mp, attn_meta, extra))
        for attn in layers:
            kv_rot_locals, kv_scores = [], []
            for r in range(pcp_size):
                qr_kv, kv_sc, _ = attn._run_parallel_input_projections(rank_x[r])
                _, _, kv_lat = attn._split_qkv_and_norm(qr_kv)
                kv_rot_locals.append(attn.rotate(kv_lat, rank_pos[r]))
                kv_scores.append(kv_sc)

            g_rot = torch.cat(kv_rot_locals, dim=0)
            g_sc = (
                torch.cat([cast(torch.Tensor, t) for t in kv_scores], dim=0)
                if kv_scores[0] is not None
                else None
            )
            mp.setattr(
                pcp_mod,
                "all_gather_equal_tokens",
                lambda t, dim=0, *, hd=attn.head_dim, r=g_rot, s=g_sc: (
                    r if t.shape[-1] == hd else cast(torch.Tensor, s)
                ),
            )
            swa_pcr = attn.swa_cache_layer.paged_cache_rows
            assert swa_pcr is not None
            banks: list[torch.Tensor] = [swa_pcr.raw]
            if attn.compressor is not None and attn.compressor.state_cache is not None:
                assert attn.compressor.state_cache.paged_cache_rows is not None
                banks.append(attn.compressor.state_cache.paged_cache_rows.raw)
            if attn.indexer is not None and attn.indexer.owns_k:
                assert attn.indexer.k_cache.paged_cache_rows is not None
                banks.append(attn.indexer.k_cache.paged_cache_rows.raw)
            if attn.is_kv_source:
                assert (
                    attn.paged_cache_rows is not None
                    and attn.paged_cache_rows.rope is not None
                )
                banks.extend([attn.paged_cache_rows.raw, attn.paged_cache_rows.rope])

            pre_step = [b.clone() for b in banks]
            rank0_post: list[torch.Tensor] = []
            for r in range(pcp_size):
                if r > 0:
                    for b, pre in zip(banks, pre_step):
                        b.copy_(pre)
                shared_topk.copy_(rank_topk[r])
                shared_cand.copy_(rank_cand[r])
                proj = attn(rank_pos[r], rank_x[r], None, pcp_step=steps[r])
                rank_topk[r].copy_(shared_topk)
                rank_cand[r].copy_(shared_cand)
                if simulate_compiling:
                    assert swa_pcr._last_read is None
                if r == 0:
                    rank0_post = [b.clone() for b in banks]
                else:
                    for b, ref in zip(banks, rank0_post):
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
    """Verify PcpStep request compaction, idle ranks, and missing-distribution error."""
    meta = AttentionMetadata(
        input_positions=torch.zeros(9, dtype=torch.int32),
        block_tables=torch.tensor([[1, 2], [3, 4], [5, 6], [0, 0]], dtype=torch.int32),
        seq_lens=torch.tensor([10, 6, 15, 0], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1, 6, 9, 9], dtype=torch.int32),
        request_distribution=torch.tensor([1, 0, 3], dtype=torch.int32),
        sequence_layout_kind="partial",
        sequence_layout_protocol="pcp_streaming",
        sequence_layout_version=2,
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
    assert s0.local_window_starts is not None
    assert s0.local_window_starts.tolist() == [8, 0, 0, 0]

    s2 = PcpStep.build(
        {"l0": meta}, pos_global[6:9], torch.tensor(2), 4, window_starts=ws
    )
    m2 = s2.local(meta)
    assert cast(torch.Tensor, m2.query_start_loc).tolist() == [0, 3, 3, 3, 3]
    assert cast(torch.Tensor, m2.seq_lens).tolist() == [15, 0, 0, 0]
    assert cast(torch.Tensor, m2.block_tables)[0].tolist() == [5, 6]
    assert s2.local_window_starts is not None
    assert s2.local_window_starts.tolist() == [4, 0, 0, 0]

    s3 = PcpStep.build(
        {"l0": meta}, pos_global[9:12], torch.tensor(3), 4, window_starts=ws
    )
    assert int(s3.num_local.item()) == 0
    assert cast(torch.Tensor, s3.local(meta).seq_lens).tolist() == [0, 0, 0, 0]

    no_dist = AttentionMetadata(
        input_positions=meta.input_positions,
        block_tables=meta.block_tables,
        seq_lens=meta.seq_lens,
        query_start_loc=meta.query_start_loc,
        request_distribution=None,
    )
    with pytest.raises(ValueError, match="request_distribution"):
        PcpStep.build({"l0": no_dist}, pos_global[0:3], torch.tensor(0), 4)


@pytest.mark.parametrize("pcp_size", [2, 4])
def test_multirank_attention_parity(
    pcp_size: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify CP=1 vs PCP=2/4 parity across 4 layers, odd chunks, SWA, and decode."""
    with _pallas_interpret_ops(monkeypatch):
        gen = torch.Generator().manual_seed(2026 + pcp_size)
        cfg_ref, cfg_pcp = _pcp_config(pcp_size=1), _pcp_config(pcp_size=pcp_size)
        layers_ref = _bind_stack(cfg_ref, seed=91)
        layers_pcp = _bind_stack(cfg_pcp, seed=91)

        steps = [
            ([21, 3, 2], [21, 3, 2], 0, None, 16, True),
            ([22, 7], [1, 4], 1, [6, 0], 8, False),
        ]
        for s_lens, q_lens, n_dec, ws, loc_pad, compiling in steps:
            h = torch.randn(
                (sum(q_lens), 64), generator=gen, dtype=torch.float32
            ).bfloat16()
            pos = torch.cat(
                [
                    torch.arange(sl - ql, sl, dtype=torch.int32)
                    for sl, ql in zip(s_lens, q_lens)
                ]
            )
            attn_test._ACTIVE_VLLM_CONFIG[:] = [cfg_ref]
            out_ref = _run_stack_step(
                monkeypatch,
                layers_ref,
                h,
                pos,
                s_lens,
                q_lens,
                num_decodes=n_dec,
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
                num_decodes=n_dec,
                pcp_size=pcp_size,
                local_padded=loc_pad,
                window_starts=ws,
                simulate_compiling=compiling,
            )
            torch.testing.assert_close(out_pcp, out_ref, atol=1e-2, rtol=1e-2)
        assert torch.compiler.is_compiling() is False


def test_host_sequence_lookback_pcp_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify runner host lookback + _prepare_engram parity, DEAD_ID pad, and guards."""
    import vllm_torchtpu.layers.core.pcp_sequence_layout as layout_mod
    import vllm_torchtpu.runner.tpu_runner as runner_mod
    from tests.runner.runner_input_test_utils import _make_runner

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
    planner = PcpSequenceLayoutPlanner.from_vllm_config(vcfg)
    backbone = SimpleNamespace(
        engram_hash=state,
        engram_dp_shared_memory=True,
        _pcp_size=pcp_size,
        _ced_boundary=4,
        start_layer=0,
        end_layer=0,
        layers=[],
    )
    collected: list[torch.Tensor] = []
    for r in range(pcp_size):
        monkeypatch.setattr(layout_mod, "_get_native_pcp_rank", lambda rank=r: rank)
        monkeypatch.setattr(layout_mod, "_get_native_pcp_world_size", lambda: pcp_size)
        monkeypatch.setattr(
            runner_mod,
            "get_pcp_group",
            lambda rank=r: SimpleNamespace(rank_in_group=rank, world_size=pcp_size),
        )
        runner = _make_runner(
            num_computed_tokens=[3, 4],
            prompt_tokens=[7, 3],
            scheduled_tokens=[4, 1],
            token_paddings=[4, 8],
            pcp_size=pcp_size,
            max_num_reqs=2,
        )
        runner.vllm_config = vcfg
        runner.parallel_config = vcfg.parallel_config
        runner._dp_target_bucket = None
        runner._replay_starts = {}
        runner.model = cast(
            Any, SimpleNamespace(token_lookback_depth=state.lookback_depth)
        )
        runner.input_ids_cpu[:8] = torch.tensor([13, 14, 15, 16, 0, 0, 0, 0])
        runner.positions_cpu[:8] = torch.tensor([3, 4, 5, 6, 4, 0, 0, 0])
        runner.query_start_loc_np[:4] = [0, 4, 5, 5]
        runner.seq_lens_np[:3] = [7, 5, 0]
        runner.input_batch = cast(
            Any,
            SimpleNamespace(
                num_reqs=2,
                req_ids=["r0", "r1"],
                req_id_to_index={"r0": 0, "r1": 1},
                num_computed_tokens_cpu=np.array([3, 4], dtype=np.int32),
                num_prompt_tokens=np.array([7, 3], dtype=np.int32),
                num_tokens_no_spec=np.array([7, 5], dtype=np.int32),
                token_ids_cpu=tok_hist.copy(),
            ),
        )
        runner._pre_async_results = cast(
            Any,
            SimpleNamespace(
                request_seq_lens=[(0, None, 3, "r0"), (1, None, 5, "r1")],
                discard_sampled_tokens_req_indices=[0],
                wait_for_copy=lambda: torch.tensor([[99], [31]], dtype=torch.int32),
            ),
        )
        runner._last_sequence_layout_plan = planner.prepare_real(
            runner=runner,
            scheduler_output=SimpleNamespace(num_scheduled_tokens={"r0": 4, "r1": 1}),
            start_index=0,
            num_reqs=2,
            num_scheduled_tokens_per_req=np.array([4, 1], dtype=np.int32),
            total_num_scheduled_tokens=5,
            use_max_model_len=True,
            target_num_reqs=2,
            padded_num_reqs=2,
        )
        host_in = runner._prepare_host_sequence_lookback_inputs(2, 0)
        assert host_in is not None
        z4 = torch.zeros(4, dtype=torch.int32)
        hashes, _ = DeepseekV41Model._prepare_engram(
            cast(Any, backbone), z4, z4, None, "full", host_in
        )
        assert hashes is not None
        assert (hashes[host_in.num_local :] == engram_mod.upstream.DEAD_ID).all()
        if host_in.num_local > 0:
            collected.append(hashes[: host_in.num_local])
    assert torch.equal(torch.cat(collected, dim=0), ref_hashes)

    z2 = torch.zeros(2, dtype=torch.int32)
    overflow = HostSequenceLookbackInputs(
        ref_tokens, ref_pos, ref_qsl, ref_lb, local_start=0, num_local=5
    )
    with pytest.raises(ValueError, match="exceeds device bucket"):
        DeepseekV41Model._prepare_engram(
            cast(Any, backbone), z2, z2, None, "full", overflow
        )
    monkeypatch.setattr(state, "ensure_cache", lambda: True)
    with (
        _mock_forward_context(monkeypatch, {"swa": SimpleNamespace()}),
        pytest.raises(ValueError, match="host_lookback_inputs is required"),
    ):
        DeepseekV41Model._prepare_engram(
            cast(Any, backbone), z2, z2, None, "full", None
        )


def test_uses_host_sequence_lookback_gating_and_hbm_no_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify CP=1 never enables host sequence lookback and HBM Engram skips sync."""
    for pcp_size, expected in ((1, False), (8, True)):
        stub = SimpleNamespace(
            model=SimpleNamespace(engram_hash=object(), _pcp_size=pcp_size)
        )
        assert (
            DeepseekV41ForCausalLM.uses_host_sequence_lookback(cast(Any, stub))
            is expected
        )

    hbm_engram = engram_mod.Engram.__new__(engram_mod.Engram)
    hbm_engram._staged_raw = None
    hbm_engram._active_staged = None
    hbm_engram._active_raw = None
    hbm_engram.layer_hash_index = 0
    hbm_engram.embed_tokens = cast(Any, SimpleNamespace(dp_size=1))
    hbm_engram.staged_rows = torch.zeros((4, 2, 8), dtype=torch.bfloat16)
    hbm_engram.prepare_embeddings = lambda _h: None  # type: ignore[method-assign]
    assert hbm_engram.staged_device_tensors() == ()

    sync_calls: list[bool] = []
    monkeypatch.setattr(
        model_mod,
        "synchronize_tensors",
        lambda _t, wait=False: sync_calls.append(wait),
    )
    state = engram_test._make_ngram_state(monkeypatch)
    backbone = SimpleNamespace(
        engram_hash=state,
        engram_dp_shared_memory=False,
        _pcp_size=1,
        _ced_boundary=1,
        start_layer=0,
        end_layer=1,
        layers=[SimpleNamespace(engram=hbm_engram)],
    )
    ids = torch.zeros(4, dtype=torch.int32)
    DeepseekV41Model._prepare_engram(cast(Any, backbone), ids, ids, None, "full")
    assert sync_calls == []


def test_engram_dp_group_pcp_patch_and_qwen4_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify _patch_vllm_engram_dp_group_pcp extends DSV4.1 EDP and skips others."""
    import vllm.config as vllm_cfg_pkg
    from vllm.config.engram import EngramConfig

    monkeypatch.setattr(
        vllm_cfg_pkg,
        "get_current_vllm_config_or_none",
        lambda: attn_test._ACTIVE_VLLM_CONFIG[0],
    )
    for name in (
        "verify_model_config",
        "resolve_dp_shared_memory",
        "verify_parallel_config",
    ):
        monkeypatch.setattr(EngramConfig, name, getattr(EngramConfig, name))
    monkeypatch.setattr(ps_mod, "_tpu_engram_dp_pcp_patch", False, raising=False)
    monkeypatch.setattr(
        ps_mod, "_engram_dp_shard_size", lambda w, d, r: min(d, max(w // max(r, 1), 1))
    )
    built_edp: list[list[list[int]]] = []
    monkeypatch.setattr(ps_mod, "initialize_model_parallel", lambda **_kw: None)
    torchtpu_root._patch_vllm_engram_dp_group_pcp()

    cfg_pcp8 = _pcp_config(pcp_size=8)
    unverified = EngramConfig()
    unverified.resolve_dp_shared_memory(cfg_pcp8.parallel_config)
    assert unverified.dp_shared_memory is False

    cfg_pcp8.model_config.hf_text_config.engram_layer_ids = [1]
    cfg_pcp8.engram_config = EngramConfig()
    cfg_pcp8.engram_config.verify_model_config(cfg_pcp8.model_config)
    cfg_pcp8.engram_config.resolve_dp_shared_memory(cfg_pcp8.parallel_config)
    assert cfg_pcp8.engram_config.dp_shared_memory is True
    cfg_pcp8.engram_config.verify_parallel_config(cfg_pcp8.parallel_config)
    assert ps_mod._engram_dp_shard_size(8, 1, 8) == 8

    cfg_qwen4 = _pcp_config(pcp_size=8, architecture="Qwen4ExpForCausalLM")
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
        lambda ranks, *_a, group_name="", **_kw: (
            built_edp.append(ranks) if group_name == "edp" else None,
            SimpleNamespace(group_ranks=ranks, group_name=group_name),
        )[1],
    )
    ps_mod.initialize_model_parallel(1, 1, 8, 1, backend="gloo")
    assert built_edp == [[list(range(8))]]


def test_engram_hash_gather_validation_and_dead_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify engram_token_slot error check, overflow error, and DEAD_ID padding."""

    class _DpCountsStub:
        device = torch.device("cpu")

        def tolist(self) -> list[int]:
            return [3, 6]

        def to(self, target: str) -> str:
            assert target == "tpu"
            return "tpu_counts"

    cfg = _pcp_config(pcp_size=8)
    dp_meta = SimpleNamespace(num_tokens_across_dp_cpu=_DpCountsStub())
    extra_kw = TpuPlatform.set_additional_forward_context(
        attn_metadata={}, vllm_config=cfg, dp_metadata=cast(Any, dp_meta)
    )
    assert extra_kw["tpu_num_tokens_across_dp_cpu"] == (3, 6)
    assert dp_meta.num_tokens_across_dp_cpu == "tpu_counts"

    with (
        _mock_forward_context(monkeypatch, {}, {}, dp_metadata=dp_meta),
        pytest.raises(RuntimeError, match="tpu_num_tokens_across_dp_cpu"),
    ):
        engram_mod.engram_token_slot(5)

    padded_seen: list[torch.Tensor] = []
    monkeypatch.setattr(
        engram_mod.dist,
        "all_gather",
        lambda chunks, tensor, group=None: (
            padded_seen.append(tensor.clone()),
            [ch.copy_(tensor + idx * 100) for idx, ch in enumerate(chunks)],
        ),
    )
    monkeypatch.setattr(
        engram_mod,
        "get_engram_dp_group",
        lambda: SimpleNamespace(world_size=2, cpu_group="fake_cpu_group"),
    )
    with _mock_forward_context(monkeypatch, {}, extra_kw, dp_metadata=dp_meta):
        gathered = engram_mod.gather_engram_hashes(
            torch.ones((3, 2, 4), dtype=torch.int32), dp_shared_memory=False
        )
        assert gathered.shape == (12, 2, 4)
        assert (padded_seen[0][3:] == engram_mod.upstream.DEAD_ID).all()
        with pytest.raises(ValueError, match="exceed coordinated slot"):
            engram_mod.gather_engram_hashes(
                torch.ones((7, 2, 4), dtype=torch.int32), dp_shared_memory=False
            )


def test_pcp_config_validation_and_model_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify PCP config validation guards and ModelConfig capability properties."""
    import vllm_torchtpu.models.vllm as models_init_mod

    install_pcp_model_capabilities()
    mc = ModelConfig.__new__(ModelConfig)
    monkeypatch.setattr(ModelConfig, "architecture", "DeepseekV41ForCausalLM")
    assert mc.uses_contiguous_pcp_layout and mc.shares_engram_host_table_across_pcp
    monkeypatch.setattr(ModelConfig, "architecture", "DeepseekV3ForCausalLM")
    assert not mc.uses_contiguous_pcp_layout
    assert not mc.shares_engram_host_table_across_pcp

    monkeypatch.setattr(
        models_init_mod.DeepseekV4ForCausalLMConfig,
        "verify_and_update_config",
        staticmethod(lambda _c: None),
    )
    bad_spec = _pcp_config(pcp_size=4, enable_ep=True)
    bad_spec.speculative_config = SimpleNamespace(method="dspark")
    cases = [
        (_pcp_config(pcp_size=4, tp_size=2, enable_ep=True), "--tensor-parallel-size"),
        (_pcp_config(pcp_size=4, dcp_size=2, enable_ep=True), "does not support DCP"),
        (_pcp_config(pcp_size=4, dp_size=2, enable_ep=True), "--data-parallel-size"),
        (bad_spec, "speculative decoding"),
        (_pcp_config(pcp_size=4, enable_ep=False), "--enable-expert-parallel"),
    ]
    for bad_cfg, pat in cases:
        with pytest.raises(NotImplementedError, match=pat):
            _TPUDeepseekV41Config.verify_and_update_config(bad_cfg)
    _TPUDeepseekV41Config.verify_and_update_config(
        _pcp_config(pcp_size=8, enable_ep=True)
    )


def test_ced_disabled_with_warning_under_pcp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _ced_config disables two-stage execution with warning_once at PCP > 1."""
    warnings: list[str] = []
    monkeypatch.setattr(
        model_mod.logger, "warning_once", lambda msg, *_a: warnings.append(msg)
    )
    assert model_mod._ced_config(_pcp_config(pcp_size=1)) == (True, 2)
    assert warnings == []
    assert model_mod._ced_config(_pcp_config(pcp_size=8, enable_ep=True)) == (False, 2)
    assert len(warnings) == 1 and "Disabling two-stage CED" in warnings[0]


def test_pcp_local_topk_buffer_and_moe_ep_sharding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify DeepseekV41Model local top-k buffer sizing and DP*PCP*TP EP sharding."""
    from collections import Counter

    from vllm.config import set_current_vllm_config

    cfg = _pcp_config(pcp_size=8, enable_ep=True)
    cfg.model_config.hf_config.num_hidden_layers = 0
    cfg.compilation_config.__dict__.update(
        mode=0, custom_ops=["none"], enabled_custom_ops=Counter()
    )
    monkeypatch.setattr(
        model_mod, "get_pcp_group", lambda: SimpleNamespace(rank_in_group=0)
    )
    monkeypatch.setattr(
        model_mod,
        "VocabParallelEmbedding",
        lambda n, d, **_kw: nn.Embedding(n, d),
    )
    with set_current_vllm_config(cfg):
        backbone = DeepseekV41Model(vllm_config=cfg)
    assert backbone.topk_indices_buffer.shape == (8, 16)
    assert backbone.candidate_block_buffer is not None
    assert backbone.candidate_block_buffer.shape == (8, 4)

    fake_self = SimpleNamespace(
        config=cfg.model_config.hf_config,
        model=SimpleNamespace(parallel_config=cfg.parallel_config, layers=[]),
    )
    DeepseekV41ForCausalLM.set_moe_parameters(cast(Any, fake_self))
    assert fake_self.num_local_physical_experts == 384 // 8
