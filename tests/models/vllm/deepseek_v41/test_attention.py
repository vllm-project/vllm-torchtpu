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
"""Unit and TPU7x parity tests for DeepSeek-V4.1 attention and SWACache layers."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
import vllm.forward_context as fc_mod
from vllm.ir.ops import rms_norm
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbeddingBase
from vllm.v1.core import kv_cache_utils

import vllm_torchtpu as torchtpu_root
from vllm_torchtpu.models.vllm.deepseek_v41 import (
    AttentionBackend,
    DeepseekCompressor,
    DeepseekV41Attention,
    DeepseekV41Indexer,
    PagedCacheRows,
    SWACache,
)
from vllm_torchtpu.models.vllm.deepseek_v41 import (
    attention as attn_mod,
)
from vllm_torchtpu.models.vllm.deepseek_v41 import (
    compressor as compressor_mod,
)
from vllm_torchtpu.models.vllm.deepseek_v41 import (
    indexer as indexer_mod,
)
from vllm_torchtpu.models.vllm.deepseek_v41.cache import write_cache

_ACTIVE_VLLM_CONFIG: list[Any] = []


class _DummyRope(RotaryEmbeddingBase):
    def __init__(self, max_pos: int = 32, rope_dim: int = 64) -> None:
        nn.Module.__init__(self)
        half = rope_dim // 2
        steps = torch.arange(max_pos, dtype=torch.float32).unsqueeze(1)
        freqs = 0.05 * torch.arange(1, half + 1, dtype=torch.float32).unsqueeze(0)
        angles = steps * freqs
        self.register_buffer(
            "cos_sin_cache",
            torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1),
            persistent=False,
        )


def _cpu_kv_rope(
    x: torch.Tensor, pos: torch.Tensor, cos_sin_cache: torch.Tensor
) -> torch.Tensor:
    rot_dim = cos_sin_cache.shape[-1]
    cs = cos_sin_cache[pos.long()].float()
    if x.ndim == 3:
        cs = cs.unsqueeze(1)
    cos, sin = cs.chunk(2, dim=-1)
    cos = cos.repeat_interleave(2, dim=-1)
    sin = sin.repeat_interleave(2, dim=-1)
    tail = x[..., -rot_dim:].float()
    rot = torch.stack((-tail[..., 1::2], tail[..., 0::2]), dim=-1).flatten(-2)
    out_tail = (tail * cos + rot * sin).to(x.dtype)
    return torch.cat((x[..., :-rot_dim], out_tail), dim=-1)


def _make_meta(
    block_tables: torch.Tensor | None = None,
    seq_lens: list[int] | None = None,
    query_start_loc: list[int] | None = None,
    device: torch.device | str = "cpu",
) -> Any:
    bt = (
        torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device=device)
        if block_tables is None
        else block_tables.to(device)
    )
    sl = torch.tensor(
        [4, 4] if seq_lens is None else seq_lens, dtype=torch.int32, device=device
    )
    qsl = torch.tensor(
        [0, 2, 4] if query_start_loc is None else query_start_loc,
        dtype=torch.int32,
        device=device,
    )
    return SimpleNamespace(block_tables=bt, seq_lens=sl, query_start_loc=qsl)


@contextmanager
def _mock_forward_context(
    monkeypatch: pytest.MonkeyPatch,
    attn_metadata: Any,
    additional_kwargs: dict[str, Any] | None = None,
):
    ctx = SimpleNamespace(
        attn_metadata=attn_metadata,
        additional_kwargs={} if additional_kwargs is None else additional_kwargs,
    )
    monkeypatch.setattr(fc_mod, "get_forward_context", lambda: ctx)
    yield ctx


def _make_vllm_config(
    *,
    max_model_len: int = 16,
    compress_ratios: list[int] | None = None,
    static_ctx: dict[str, Any] | None = None,
) -> Any:
    hf_config = SimpleNamespace(
        hidden_size=64,
        num_attention_heads=4,
        q_lora_rank=16,
        o_lora_rank=16,
        head_dim=512,
        qk_rope_head_dim=64,
        o_groups=2,
        sliding_window=4,
        rms_norm_eps=1e-6,
        max_position_embeddings=32,
        num_hidden_layers=4,
        compress_ratios=[2, 2, 1, 0] if compress_ratios is None else compress_ratios,
        kv_source_layer_ids=[0, 2],
        index_source_layer_ids=[0, 1, 2],
        candidate_source_layer_id=0,
        candidate_topk_blocks=2,
        candidate_block_size=2,
        index_n_heads=4,
        index_head_dim=128,
        index_topk=4,
        rope_theta=10000.0,
        compress_rope_theta=10000.0,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        rope_scaling=None,
    )
    cache_config = SimpleNamespace(
        block_size=4,
        cache_dtype="auto",
        swa_bounded_replay=False,
    )
    comp_config = SimpleNamespace(
        static_forward_context={} if static_ctx is None else static_ctx,
        custom_ops=["none"],
        enabled_custom_ops=Counter(),
        disabled_custom_ops=Counter(),
    )
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config, max_model_len=max_model_len),
        quant_config=None,
        cache_config=cache_config,
        compilation_config=comp_config,
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=8,
            max_num_seqs=4,
            disable_hybrid_kv_cache_manager=False,
        ),
        attention_config=SimpleNamespace(
            indexer_sparse_logits=False,
            indexer_kv_dtype="fp8",
            hisparse_config=None,
        ),
        kernel_config=SimpleNamespace(
            enable_jit_warmup=False, sparse_indexer_topk_backend="auto"
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            pcp_shard_decode_requests=False,
            cp_kv_cache_interleave_size=1,
        ),
        speculative_config=None,
        num_lookahead_tokens=0,
        num_speculative_tokens=0,
    )
    _ACTIVE_VLLM_CONFIG[:] = [cfg]
    return cfg


class _DummyLinear(nn.Module):
    def __init__(self, in_f: int, out_f: int, *, return_bias: bool = True) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(in_f * 1009 + out_f)
        w = torch.randn((out_f, in_f), generator=gen, dtype=torch.float32) * 0.05
        self.weight = nn.Parameter(w.bfloat16())
        self.return_bias = return_bias
        self.is_bmm = False
        self.bmm_batch_size = 1

    def forward(self, x: torch.Tensor) -> Any:
        out = x.to(self.weight.dtype) @ self.weight.T
        return (out, None) if self.return_bias else out


class _DummyNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight.to(x.dtype), self.variance_epsilon)


def _patch_vllm_attention_deps(
    monkeypatch: pytest.MonkeyPatch, *, patch_rope_op: bool = True
) -> None:
    import vllm.config.vllm as vllm_cfg_mod
    import vllm.v1.attention.backends.mla.sparse_swa as swa_mod

    if patch_rope_op:
        for cls in (DeepseekCompressor, DeepseekV41Indexer, DeepseekV41Attention):
            monkeypatch.setattr(cls, "kv_rope_op", property(lambda _s: _cpu_kv_rope))

    default_cfg = _make_vllm_config(static_ctx={})
    monkeypatch.setattr(vllm_cfg_mod, "_current_vllm_config", default_cfg)
    monkeypatch.setattr(
        vllm_cfg_mod, "get_current_vllm_config", lambda: _ACTIVE_VLLM_CONFIG[0]
    )
    for mod in (
        attn_mod.upstream,
        indexer_mod.upstream,
        compressor_mod.compressor_upstream,
        swa_mod,
    ):
        monkeypatch.setattr(
            mod, "get_current_vllm_config", lambda: _ACTIVE_VLLM_CONFIG[0]
        )
    monkeypatch.setattr(attn_mod, "get_tensor_model_parallel_world_size", lambda: 1)

    def _merged_col(
        in_f: int,
        sizes: list[int],
        *,
        bias: bool,
        quant_config: Any,
        prefix: str,
        disable_tp: bool,
    ) -> nn.Module:
        assert (
            bias is False and disable_tp is True and prefix.endswith(".fused_wqa_wkv")
        )
        del quant_config
        return _DummyLinear(in_f, sum(sizes), return_bias=True)

    def _col(in_f: int, out_f: int, **_kw: Any) -> nn.Module:
        return _DummyLinear(in_f, out_f, return_bias=_kw.get("return_bias", True))

    def _row(in_f: int, out_f: int, **_kw: Any) -> nn.Module:
        return _DummyLinear(in_f, out_f, return_bias=_kw.get("return_bias", True))

    monkeypatch.setattr(attn_mod.vllm_lin, "MergedColumnParallelLinear", _merged_col)
    monkeypatch.setattr(attn_mod.vllm_lin, "ColumnParallelLinear", _col)
    monkeypatch.setattr(attn_mod.vllm_lin, "RowParallelLinear", _row)
    monkeypatch.setattr(
        attn_mod.upstream,
        "build_deepseek_v4_rope",
        lambda _cfg,
        *,
        head_dim,
        rope_head_dim,
        max_position_embeddings,
        compress_ratio: _DummyRope(max_position_embeddings, rope_head_dim),
    )

    def _compressor_super_init(
        self: Any,
        vllm_config: Any,
        compress_ratio: int,
        hidden_size: int,
        head_dim: int,
        rotate: bool = False,
        prefix: str = "",
        k_cache_prefix: str = "",
    ) -> None:
        del rotate
        nn.Module.__init__(self)
        self.prefix = prefix
        self.k_cache_prefix = k_cache_prefix
        self.head_dim = head_dim
        self.compress_ratio = compress_ratio
        self.block_size = vllm_config.cache_config.block_size
        self.rms_norm_eps = 1e-6
        self._static_forward_context = (
            vllm_config.compilation_config.static_forward_context
        )
        self.fused_wkv_wgate = _DummyLinear(
            hidden_size, 2 * self.head_dim, return_bias=False
        )
        self.norm = _DummyNorm(self.head_dim, 1e-6)
        self.state_cache = None

    monkeypatch.setattr(
        compressor_mod.compressor_upstream.DeepseekCompressor,
        "__init__",
        _compressor_super_init,
    )

    def _indexer_super_init(
        self: Any,
        vllm_config: Any,
        config: Any,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: Any,
        cache_config: Any,
        topk_indices_buffer: Any,
        compress_ratio: int,
        prefix: str,
        *,
        owns_k: bool,
        k_cache: Any,
        main_head_dim: int,
        candidate_block_buffer: Any = None,
        candidate_block_size: int = 0,
        candidate_write: bool = False,
    ) -> None:
        del vllm_config, quant_config, cache_config
        nn.Module.__init__(self)
        self.n_head = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.topk_tokens = config.index_topk
        self.softmax_scale = self.head_dim**-0.5
        self.topk_indices_buffer = topk_indices_buffer
        self.compress_ratio = compress_ratio
        self.owns_k = owns_k
        self.k_cache = k_cache
        self.wk = _DummyLinear(main_head_dim, self.head_dim, return_bias=True)
        self.wq_b = _DummyLinear(
            q_lora_rank, self.n_head * self.head_dim, return_bias=True
        )
        self.weights_proj = _DummyLinear(hidden_size, self.n_head, return_bias=True)
        self.k_norm = _DummyNorm(self.head_dim, 1e-6)
        self.indexer_op = self.attn_cls(
            self.k_cache,
            128,
            "ue8m0",
            self.topk_tokens,
            self.head_dim,
            16,
            16,
            self.topk_indices_buffer,
            prefix=prefix,
            compress_ratio=self.compress_ratio,
            candidate_blocks=candidate_block_buffer,
            candidate_block_size=candidate_block_size,
            candidate_write=candidate_write,
        )

    monkeypatch.setattr(
        indexer_mod.upstream.DeepseekV4Indexer, "__init__", _indexer_super_init
    )


def test_swa_cache_and_attention_init_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify SWACache validation/specs and DeepseekV41Attention init branches."""
    _patch_vllm_attention_deps(monkeypatch)
    vcfg = _make_vllm_config()
    swa_cache = SWACache(512, 4, torch.bfloat16, "l.swa", vcfg.cache_config, 4)
    assert swa_cache.get_attn_backend() is AttentionBackend
    swa_spec = swa_cache.get_kv_cache_spec(vcfg)
    assert swa_spec.dtype == torch.bfloat16 and swa_spec.page_size_padded is None
    with pytest.raises(ValueError, match="SWACache requires dtype=torch.bfloat16"):
        SWACache(512, 4, torch.uint8, "l.bad_swa", vcfg.cache_config, 4)

    monkeypatch.setattr(
        attn_mod.upstream.DeepseekV4Attention,
        "get_kv_cache_spec",
        lambda self, _v: (
            None
            if not self.is_kv_source
            else attn_mod.MLAAttentionSpec(
                block_size=4,
                num_kv_heads=1,
                head_size=512,
                dtype=torch.float8_e4m3fn,
                page_size_padded=512,
            )
        ),
    )
    s_ctx: dict[str, Any] = {}
    vcfg_short = _make_vllm_config(max_model_len=8, static_ctx=s_ctx)
    cand_buf = torch.full((4, 2), -1, dtype=torch.int32)
    l0 = DeepseekV41Attention(
        vcfg_short, "model.layers.0.self_attn", candidate_block_buffer=cand_buf
    )
    assert l0.indexer is not None and l0.indexer.indexer_op.candidate_blocks is cand_buf
    s_ctx["model.layers.0.self_attn.indexer.k_cache"] = l0.indexer.k_cache
    l1 = DeepseekV41Attention(
        vcfg_short, "model.layers.1.self_attn", candidate_block_buffer=cand_buf
    )
    assert l1.indexer is not None and l1.indexer.indexer_op.candidate_blocks is cand_buf

    vcfg_no_cand = _make_vllm_config(max_model_len=64, static_ctx={})
    vcfg_no_cand.model_config.hf_config.candidate_source_layer_id = -1
    l0_no_cand = DeepseekV41Attention(
        vcfg_no_cand, "model.layers.0.self_attn", candidate_block_buffer=cand_buf
    )
    assert (
        l0_no_cand.indexer is not None
        and l0_no_cand.indexer.indexer_op.candidate_blocks is None
    )
    l_mtp = DeepseekV41Attention(vcfg_no_cand, "model.layers.5.self_attn")
    assert l0.get_kv_cache_spec(vcfg_short) is not None
    assert l1.get_kv_cache_spec(vcfg_short) is None
    assert l_mtp.compress_ratio == 0

    bad_ratio_cfg = _make_vllm_config(compress_ratios=[3])
    with pytest.raises(ValueError, match="Unsupported compress_ratio"):
        DeepseekV41Attention(bad_ratio_cfg, "model.layers.0.self_attn")

    no_cache_cfg = _make_vllm_config()
    no_cache_cfg.cache_config = None
    with pytest.raises(ValueError, match="requires vllm_config.cache_config"):
        DeepseekV41Attention(no_cache_cfg, "model.layers.0.self_attn")


def test_attention_encoder_and_forward_kv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify encoder global KV publishing, forward_kv, and SWA write window cap."""
    _patch_vllm_attention_deps(monkeypatch)
    swa_keeps: list[torch.Tensor] = []

    def _rec_write(
        cache: Any, _s: Any, _v: Any, valid: torch.Tensor, after_read: bool = False
    ) -> None:
        if cache.shape[-1] == 512 and cache.dtype == torch.bfloat16:
            assert after_read is False
            swa_keeps.append(valid.clone())

    for mod in (attn_mod, compressor_mod, indexer_mod):
        monkeypatch.setattr(mod, "write_cache", _rec_write)
    monkeypatch.setattr(attn_mod, "sp_all_gather", lambda x: x)
    s_ctx: dict[str, Any] = {}
    vcfg = _make_vllm_config(static_ctx=s_ctx)
    l0 = DeepseekV41Attention(vcfg, "model.layers.0.self_attn")
    assert l0.indexer is not None
    s_ctx["model.layers.0.self_attn.indexer.k_cache"] = l0.indexer.k_cache
    l1 = DeepseekV41Attention(vcfg, "model.layers.1.self_attn")
    l3 = DeepseekV41Attention(vcfg, "model.layers.3.self_attn")
    l0.swa_cache_layer.kv_cache = torch.zeros((4, 4, 512), dtype=torch.bfloat16)
    l1.swa_cache_layer.kv_cache = torch.zeros((4, 4, 512), dtype=torch.bfloat16)
    l3.swa_cache_layer.kv_cache = torch.zeros((4, 4, 512), dtype=torch.bfloat16)
    l0.kv_cache = torch.zeros((4, 4, 512), dtype=torch.uint8)
    l0.indexer.k_cache.kv_cache = torch.zeros((4, 4, 128), dtype=torch.bfloat16)
    h = torch.ones((2, 64), dtype=torch.bfloat16)
    pos = torch.tensor([0, 1], dtype=torch.int32)
    qr0, kvs0, wi0 = l0._run_parallel_input_projections(h)
    _, kvs1, wi1 = l1._run_parallel_input_projections(h)
    _, kvs3, wi3 = l3._run_parallel_input_projections(h)
    assert qr0.shape == (2, 528) and kvs0 is not None and wi0 is not None
    assert kvs1 is None and wi1 is not None and kvs3 is None and wi3 is None

    with _mock_forward_context(monkeypatch, None):
        l3.forward_kv(pos, h)
    with _mock_forward_context(monkeypatch, _make_meta()):
        l1.write_encoder_global_kv(h, pos)
        l0.write_encoder_global_kv(h, pos, sp_sharded=True)
        l0.write_encoder_global_kv(h, pos, sp_sharded=False)
        l0.indexer.owns_k = False
        l0.write_encoder_global_kv(h, pos, sp_sharded=False)
        l0.forward_kv(pos, h)
        l0.indexer.owns_k = True
        l0.forward_kv(pos, h)
        l1.forward_kv(pos, h)
        l3.forward_kv(pos, h)

    meta10 = _make_meta(
        block_tables=torch.tensor([[0, 1, 2]], dtype=torch.int32),
        seq_lens=[10],
        query_start_loc=[0, 10],
    )
    with _mock_forward_context(monkeypatch, meta10):
        l3.forward_kv(
            torch.arange(10, dtype=torch.int32),
            torch.ones((10, 64), dtype=torch.bfloat16),
        )
    assert not swa_keeps[-1][:6].any() and swa_keeps[-1][6:].all()


def test_attention_prepare_and_forward_mqa(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _prepare_and_attn SWA prefix-before-write order and forward_mqa paths."""
    _patch_vllm_attention_deps(monkeypatch)
    swa_afters: list[bool] = []

    def _spy_write_cache(
        cache: Any,
        slots: torch.Tensor,
        values: torch.Tensor,
        valid: torch.Tensor,
        after_read: bool = False,
        *,
        records: torch.Tensor | None = None,
    ) -> None:
        if values.shape[-1] == 512 and values.dtype == torch.bfloat16:
            swa_afters.append(after_read)
        write_cache(cache, slots, values, valid, after_read, records=records)

    monkeypatch.setattr(attn_mod, "write_cache", _spy_write_cache)
    monkeypatch.setattr(
        indexer_mod.indexer_op,
        "dsv41_paged_indexer_topk",
        lambda q, *_a: torch.zeros((q.shape[0], 4), dtype=torch.int32),
    )
    monkeypatch.setattr(
        attn_mod.ops.attention_op,
        "dsv41_gather_sparse_fp8_kv",
        lambda _kl, _rl, tk, *_a: (
            torch.zeros((tk.shape[0], tk.shape[1], 128), dtype=torch.int32),
            torch.zeros((tk.shape[0], tk.shape[1] // 4, 128), dtype=torch.int32),
        ),
    )
    mla_calls: list[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]
    ] = []

    def _mock_mla(
        q: torch.Tensor,
        _gn: torch.Tensor,
        _gr: torch.Tensor,
        _tk: torch.Tensor,
        _sk: torch.Tensor,
        sl: torch.Tensor,
        cu: torch.Tensor,
        _sc: float,
        _win: int,
        swa: torch.Tensor | None,
        _cr: int = 1,
        swa_min_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mla_calls.append(
            (
                sl.clone(),
                cu.clone(),
                None if swa is None else swa.clone(),
                None if swa_min_pos is None else swa_min_pos.clone(),
            )
        )
        return torch.ones_like(q)

    monkeypatch.setattr(
        attn_mod.ops.attention_op, "dsv41_sparse_mla_swa_attention", _mock_mla
    )
    s_ctx: dict[str, Any] = {}
    vcfg = _make_vllm_config(static_ctx=s_ctx)
    topk_buf = torch.zeros((4, 4), dtype=torch.int32)
    l0 = DeepseekV41Attention(
        vcfg, "model.layers.0.self_attn", topk_indices_buffer=topk_buf
    )
    l3 = DeepseekV41Attention(vcfg, "model.layers.3.self_attn")
    l3.swa_cache_layer.kv_cache = torch.ones((4, 4, 512), dtype=torch.bfloat16)

    h = torch.ones((2, 64), dtype=torch.bfloat16)
    qr = torch.ones((2, 16), dtype=torch.bfloat16)
    kv = torch.full((2, 512), 2.0, dtype=torch.bfloat16)
    kv_s = torch.ones((2, 1024), dtype=torch.float32)
    idx_w = torch.ones((2, 4), dtype=torch.bfloat16)
    pos = torch.tensor([0, 1], dtype=torch.int32)
    out = torch.empty((2, 4, 512), dtype=torch.bfloat16)
    with _mock_forward_context(monkeypatch, None):
        l0._prepare_and_attn(h, qr, kv, None, kv_s, idx_w, pos, out)
        assert (out == 0).all()

    l0.swa_cache_layer.kv_cache = torch.ones((4, 4, 512), dtype=torch.bfloat16)
    with _mock_forward_context(
        monkeypatch,
        _make_meta(),
        {"deepseek_v41_replay_starts": torch.tensor([0, 0], dtype=torch.int32)},
    ):
        l0._prepare_and_attn(h, qr, kv, None, kv_s, idx_w, pos, out)
        assert mla_calls[0][2] is not None and (mla_calls[0][2][2:4] == 1).all()
        l3.forward_mqa(torch.ones((2, 4, 512), dtype=torch.bfloat16), kv, pos, out)
        l0.swa_cache_layer.kv_cache.fill_(1.0)
        l0.kv_cache = torch.zeros((2, 4, 4, 128), dtype=torch.int32)
        l0.ced_global_from_encoder = True
        assert l0.project_context_kv(h).shape == (2, 512)
        assert l0.indexer is not None
        monkeypatch.setattr(
            l0.indexer,
            "forward",
            lambda *_a: (
                torch.ones((2, 4, 128), dtype=torch.bfloat16),
                torch.ones((2, 4), dtype=torch.float32),
                torch.ones((2, 4, 1), dtype=torch.float32),
            ),
        )
        l0._prepare_and_attn(h, qr, kv, None, kv_s, idx_w, pos, out)

        # 2-bank PagedCacheRows with null_block=True (mapped_bt is not None)
        l0.swa_cache_layer.kv_cache.fill_(1.0)
        l0.bind_kv_cache(
            torch.zeros((2, 4, 512), dtype=torch.uint8),
            raw_cache=(
                torch.zeros((8, 128), dtype=torch.int32),
                torch.zeros((8, 128), dtype=torch.int32),
            ),
            layout=((2, 4, 512), torch.uint8, 2048, 0),
            rope_pool=(
                torch.zeros((2, 128), dtype=torch.int32),
                torch.zeros((2, 128), dtype=torch.int32),
            ),
        )
        assert l0.paged_cache_rows is not None and l0.paged_cache_rows.null_block
        l0.indexer = None
        l0._prepare_and_attn(h, qr, kv, None, None, None, pos, out)

        # 1-bank PagedCacheRows without null_block (mapped_bt is None)
        l0.swa_cache_layer.kv_cache.fill_(1.0)
        standalone_pcr = PagedCacheRows(
            torch.zeros((8, 128), dtype=torch.int32),
            ((2, 4, 512), torch.uint8, 2048, 0),
            torch.zeros((2, 128), dtype=torch.int32),
            null_block=False,
        )
        with monkeypatch.context() as mp_pcr:
            mp_pcr.setattr(
                DeepseekV41Attention,
                "paged_cache_rows",
                property(lambda _s: standalone_pcr),
            )
            l0.forward_mqa(torch.ones((2, 4, 512), dtype=torch.bfloat16), kv, pos, out)
        l0.dspark_block_size = 2
        l0.non_causal_multi_token_decode = True
        l0.forward_mqa(torch.ones((2, 4, 512), dtype=torch.bfloat16), kv, pos, out)

    with _mock_forward_context(monkeypatch, _make_meta()):
        l0.window_size = 0
        l0.dspark_block_size = 0
        l0.non_causal_multi_token_decode = False
        l0.forward_mqa(torch.ones((2, 4, 512), dtype=torch.bfloat16), kv, pos, out)
        assert mla_calls[-1][2] is None
        l0.non_causal_multi_token_decode = True
        l0.forward_mqa(torch.ones((2, 4, 512), dtype=torch.bfloat16), kv, pos, out)
        assert mla_calls[-1][2] is None
        l0.window_size = 4
        l0.non_causal_multi_token_decode = False
        l0.forward_mqa(torch.ones((2, 4, 512), dtype=torch.bfloat16), kv, pos, out)
        l0.dspark_block_size = 2
        l0.non_causal_multi_token_decode = True
        l0._prepare_and_attn(h, qr, kv, None, None, None, pos, out)
        l0.dspark_block_size = 0
        l0.forward_mqa(torch.ones((2, 4, 512), dtype=torch.bfloat16), kv, pos, out)
        l0.non_causal_multi_token_decode = False

    assert swa_afters == [True, True, True]
    causal_sl, causal_cu, causal_swa, causal_mp = mla_calls[4]
    nc_sl, nc_cu, nc_swa, nc_mp = mla_calls[5]
    assert causal_swa is not None and nc_swa is not None
    assert causal_mp is not None and torch.equal(
        causal_mp, torch.tensor([0, 0], dtype=torch.int32)
    )
    assert nc_mp is not None and torch.equal(
        nc_mp, torch.tensor([0, 0], dtype=torch.int32)
    )
    assert mla_calls[-2][3] is None and mla_calls[-1][3] is None
    assert torch.equal(causal_sl, torch.tensor([4, 4], dtype=torch.int32))
    assert torch.equal(causal_cu, torch.tensor([0, 2, 4], dtype=torch.int32))
    assert causal_swa.shape == (10, 512)
    assert (causal_swa[:2] == 0).all() and (causal_swa[2:4] == 1).all()
    assert (causal_swa[4:6] == 0).all() and (causal_swa[6:8] == 1).all()
    assert torch.equal(causal_swa[8:10], kv)
    assert torch.equal(nc_sl, torch.tensor([4, 4], dtype=torch.int32))
    assert torch.equal(nc_cu, torch.tensor([0, 1, 2], dtype=torch.int32))
    assert nc_swa.shape == (2, 6, 512)
    assert (
        (nc_swa[:, :2] == 1).all()
        and (nc_swa[:, 2:4] == 2).all()
        and (nc_swa[:, 4:] == 0).all()
    )
    assert (out == 1).all()

    # Physical page capacity divisor (16 != block_size 64) and replay_starts > 0.
    l0.swa_cache_layer.block_size = 64
    swa_16 = torch.zeros((4, 16, 512), dtype=torch.bfloat16)
    swa_16[2].fill_(3.0)
    swa_16[3].fill_(7.0)
    l0.swa_cache_layer.kv_cache = swa_16
    bt_16 = torch.tensor([[2, 3]], dtype=torch.int32)
    meta_16 = _make_meta(block_tables=bt_16, seq_lens=[20], query_start_loc=[0, 2])
    with _mock_forward_context(
        monkeypatch,
        meta_16,
        {"deepseek_v41_replay_starts": torch.tensor([16], dtype=torch.int32)},
    ):
        l0.window_size = 4
        l0.forward_mqa(
            torch.ones((2, 4, 512), dtype=torch.bfloat16),
            kv,
            torch.tensor([18, 19], dtype=torch.int32),
            out,
        )
    _, _, swa_p16, mp_p16 = mla_calls[-1]
    assert swa_p16 is not None and mp_p16 is not None
    assert torch.equal(mp_p16, torch.tensor([16], dtype=torch.int32))
    assert (swa_p16[:2] == 0).all() and (swa_p16[2:4] == 7).all()

    # Verify window_starts vs replay_starts under ced_decoder=False vs True.
    with _mock_forward_context(
        monkeypatch,
        meta_16,
        {
            "deepseek_v41_window_starts": torch.tensor([0], dtype=torch.int32),
            "deepseek_v41_replay_starts": torch.tensor([16], dtype=torch.int32),
        },
    ):
        q_2 = torch.ones((2, 4, 512), dtype=torch.bfloat16)
        pos_2 = torch.tensor([18, 19], dtype=torch.int32)
        l0.ced_decoder = False
        l0.forward_mqa(q_2, kv, pos_2, out)
        _, _, swa_enc, mp_enc = mla_calls[-1]
        assert swa_enc is not None and mp_enc is not None
        assert torch.equal(mp_enc, torch.tensor([0], dtype=torch.int32))
        assert (swa_enc[:2] == 3).all() and (swa_enc[2:4] == 7).all()
        l0.ced_decoder = True
        l0.forward_mqa(q_2, kv, pos_2, out)
        _, _, swa_dec, mp_dec = mla_calls[-1]
        assert swa_dec is not None and mp_dec is not None
        assert torch.equal(mp_dec, torch.tensor([16], dtype=torch.int32))
        assert (swa_dec[:2] == 0).all() and (swa_dec[2:4] == 7).all()


def test_replay_masks_global_compressor_and_indexer_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify write_encoder_global_kv masks replayed compressor and indexer slots."""
    _patch_vllm_attention_deps(monkeypatch)
    masks: list[tuple[int, list[bool]]] = []

    def _rec_write(
        cache: Any, _s: Any, _v: Any, valid: torch.Tensor, after_read: bool = False
    ) -> None:
        if not after_read:
            masks.append((int(cache.shape[-1]), valid.tolist()))

    monkeypatch.setattr(compressor_mod, "write_cache", _rec_write)
    monkeypatch.setattr(indexer_mod, "write_cache", _rec_write)
    s_ctx: dict[str, Any] = {}
    vcfg = _make_vllm_config(static_ctx=s_ctx)
    l0 = DeepseekV41Attention(vcfg, "model.layers.0.self_attn")
    assert l0.compressor is not None and l0.indexer is not None
    assert l0.compressor.state_cache is not None
    l0.kv_cache = torch.zeros((8, 4, 512), dtype=torch.uint8)
    l0.indexer.k_cache.kv_cache = torch.zeros((8, 4, 128), dtype=torch.bfloat16)
    l0.compressor.state_cache.kv_cache = torch.zeros((8, 4, 1024), dtype=torch.float32)
    bt = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32)
    meta = _make_meta(block_tables=bt, seq_lens=[10, 4], query_start_loc=[0, 6, 10])
    pos = torch.tensor([4, 5, 6, 7, 8, 9, 0, 1, 2, 3], dtype=torch.int32)
    h = torch.ones((10, 64), dtype=torch.bfloat16)
    replays = {"deepseek_v41_replay_starts": torch.tensor([4, 0], dtype=torch.int32)}
    with _mock_forward_context(monkeypatch, meta, replays):
        l0.write_encoder_global_kv(h, pos)
    expected = [False, False, False, False, False, True, False, True, False, True]
    assert masks == [(512, expected), (128, expected)]


def test_attention_native_null_block_retained_swa_and_nc_replay_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify null_block=0 isolation, cross-dtype NaN safety, SWA retention, and nc."""
    _patch_vllm_attention_deps(monkeypatch)
    s_ctx: dict[str, Any] = {}
    vcfg = _make_vllm_config(static_ctx=s_ctx)
    l0 = DeepseekV41Attention(vcfg, "model.layers.0.self_attn")
    assert l0.compressor is not None and l0.indexer is not None

    # 1. SWACache + _patch_dflash_sliding_window_retained_tokens retain lookahead + 1.
    torchtpu_root._patch_dflash_sliding_window_retained_tokens()
    spec_plain = l0.swa_cache_layer.get_kv_cache_spec(vcfg)
    assert spec_plain.extra_retained_tokens == 0
    vcfg.speculative_config = SimpleNamespace(method="dspark")
    vcfg.num_lookahead_tokens = 5
    spec_map = {"l0.swa": l0.swa_cache_layer.get_kv_cache_spec(vcfg)}
    kv_cache_utils.get_kv_cache_groups(vcfg, spec_map)
    assert spec_map["l0.swa"].extra_retained_tokens == 6

    # 2. Bind 2-bank NativeCacheView pools poisoned with f32->bf16 & FP8 NaN bits
    # in null_block=0 and recycled blocks, plus non-zero layer row_offset.
    nan_w = int(torch.tensor([0x7FC07FC0], dtype=torch.uint32).view(torch.int32)[0])
    fp8_nan = int(torch.tensor([0x7F7F7F7F], dtype=torch.uint32).view(torch.int32)[0])
    lo = torch.full((32, 128), nan_w, dtype=torch.int32)
    hi = torch.full((32, 128), fp8_nan, dtype=torch.int32)

    def _gather_mock(
        b_lo: torch.Tensor, idx: torch.Tensor, b_hi: torch.Tensor | None = None
    ) -> torch.Tensor:
        n_lo = b_lo.shape[0]
        valid = (idx >= 0) & (idx < n_lo + (0 if b_hi is None else b_hi.shape[0]))
        safe = idx.masked_fill(~valid, 0).long()
        if b_hi is None:
            res = b_lo.index_select(0, safe)
        else:
            in_hi = safe >= n_lo
            v_lo = b_lo.index_select(0, safe.masked_fill(in_hi, 0))
            v_hi = b_hi.index_select(0, (safe - n_lo).masked_fill(~in_hi, 0))
            res = torch.where(in_hi.unsqueeze(-1), v_hi, v_lo)
        return res.masked_fill(~valid.unsqueeze(-1), 0)

    monkeypatch.setattr(attn_mod.ops.cache_op, "dsv41_gather_cache_rows", _gather_mock)
    l0.swa_cache_layer.bind_kv_cache(
        torch.zeros((4, 4, 512), dtype=torch.bfloat16),
        raw_cache=(lo, hi),
        layout=((4, 4, 512), torch.bfloat16, 8192, 2048),
    )
    l0.indexer.k_cache.bind_kv_cache(
        torch.zeros((4, 4, 128), dtype=torch.bfloat16),
        raw_cache=(lo, hi),
        layout=((4, 4, 128), torch.bfloat16, 8192, 4096),
    )
    swa_pcr = l0.swa_cache_layer.paged_cache_rows
    idx_pcr = l0.indexer.k_cache.paged_cache_rows
    assert swa_pcr is not None and idx_pcr is not None
    f2_w = int(torch.tensor([2.0, 2.0], dtype=torch.bfloat16).view(torch.int32)[0])
    hi[20:28].fill_(f2_w)

    bt = torch.tensor([[0, 3]], dtype=torch.int32)
    meta = _make_meta(block_tables=bt, seq_lens=[7], query_start_loc=[0, 2])
    pos_pad = torch.tensor([1, 5, 6, 0], dtype=torch.int32)
    slots = attn_mod.cache_slots(swa_pcr, meta, pos_pad)
    assert slots.tolist() == [-1, 13, -1, -1]

    seen_idx_bt: list[torch.Tensor] = []

    def _rec_topk(*args: Any) -> torch.Tensor:
        seen_idx_bt.append(args[3].clone())
        return torch.full((args[0].shape[0], 4), -1, dtype=torch.int32)

    monkeypatch.setattr(indexer_mod.indexer_op, "dsv41_paged_indexer_topk", _rec_topk)
    seen_mla: list[tuple[torch.Tensor | None, torch.Tensor | None]] = []

    def _rec_mla(*args: Any) -> torch.Tensor:
        seen_mla.append((args[9], args[11]))
        return torch.zeros_like(args[0])

    monkeypatch.setattr(
        attn_mod.ops.attention_op, "dsv41_sparse_mla_swa_attention", _rec_mla
    )
    replays = {"deepseek_v41_replay_starts": torch.tensor([4], dtype=torch.int32)}
    with _mock_forward_context(monkeypatch, meta, replays):
        l0.indexer.select_topk(
            torch.ones((2, 4, 128), dtype=torch.bfloat16),
            torch.ones((2, 4, 1), dtype=torch.float32),
            pos_pad[:2],
        )
        assert seen_idx_bt[-1].tolist() == [[-1, 28]]
        l0.window_size = 4
        l0.dspark_block_size = 2
        l0.non_causal_multi_token_decode = True
        out = torch.zeros((2, 4, 512), dtype=torch.bfloat16)
        l0.forward_mqa(
            torch.ones((2, 4, 512), dtype=torch.bfloat16),
            torch.ones((2, 512), dtype=torch.bfloat16),
            torch.tensor([5, 6], dtype=torch.int32),
            out,
        )
    swa_kv, swa_mp = seen_mla[-1]
    assert swa_kv is not None and not torch.isnan(swa_kv).any()
    assert swa_mp is not None and swa_mp.tolist() == [4, 4]
    assert bool(
        (swa_kv[:, :3] == 0.0).all()
        and (swa_kv[:, 3:4] == 2.0).all()
        and (swa_kv[:, 4:] == 1.0).all()
    )


def test_dspark_non_causal_swa_boundary_splicing_and_idle_zero_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify DSpark draft SWA splicing at null_block=0 boundaries and idle ranks."""
    _patch_vllm_attention_deps(monkeypatch)
    s_ctx: dict[str, Any] = {}
    vcfg = _make_vllm_config(static_ctx=s_ctx)
    l0 = DeepseekV41Attention(vcfg, "model.layers.0.self_attn")
    l0.indexer = None
    l0.window_size = 4
    l0.dspark_block_size = 3
    l0.non_causal_multi_token_decode = True

    swa_buf = torch.zeros((4, 4, 512), dtype=torch.bfloat16)
    for t_i in range(4):
        swa_buf[1, t_i].fill_(10.0 + t_i)
        swa_buf[2, t_i].fill_(20.0 + t_i)
    l0.swa_cache_layer.kv_cache = swa_buf
    swa_snap = swa_buf.clone()

    seen_swa: list[torch.Tensor | None] = []

    def _rec_mla(*args: Any) -> torch.Tensor:
        seen_swa.append(None if args[9] is None else args[9].clone())
        return torch.zeros_like(args[0])

    monkeypatch.setattr(
        attn_mod.ops.attention_op, "dsv41_sparse_mla_swa_attention", _rec_mla
    )

    bt_boundary = torch.tensor([[1, 0], [0, 0]], dtype=torch.int32)
    meta_boundary = _make_meta(
        block_tables=bt_boundary, seq_lens=[7, 3], query_start_loc=[0, 3, 6]
    )
    kv_live = torch.zeros((8, 512), dtype=torch.bfloat16)
    for i, val in enumerate((40.0, 41.0, 42.0, 50.0, 51.0, 52.0, 99.0, 99.0)):
        kv_live[i].fill_(val)
    pos_live = torch.tensor([4, 5, 6, 0, 1, 2, 0, 0], dtype=torch.int32)
    out = torch.zeros((8, 4, 512), dtype=torch.bfloat16)
    h = torch.ones((8, 64), dtype=torch.bfloat16)
    qr = torch.ones((8, 16), dtype=torch.bfloat16)
    with _mock_forward_context(monkeypatch, meta_boundary):
        l0._prepare_and_attn(h, qr, kv_live, None, None, None, pos_live, out)
    swa_b = seen_swa[-1]
    assert swa_b is not None and swa_b.shape == (8, 7, 512)
    for tok_i in range(3):
        assert swa_b[tok_i, :, 0].tolist() == [10.0, 11.0, 12.0, 13.0, 40.0, 41.0, 42.0]
    for tok_i in range(3, 6):
        assert swa_b[tok_i, :, 0].tolist() == [50.0, 51.0, 52.0, 0.0, 0.0, 0.0, 0.0]

    bt_idle = torch.tensor([[2, 1], [1, 2]], dtype=torch.int32)
    meta_idle = _make_meta(
        block_tables=bt_idle, seq_lens=[3, 3], query_start_loc=[0, 3, 6]
    )
    with _mock_forward_context(monkeypatch, meta_idle):
        l0._prepare_and_attn(
            h,
            qr,
            torch.full((8, 512), -7.0, dtype=torch.bfloat16),
            None,
            None,
            None,
            torch.zeros(8, dtype=torch.int32),
            out,
        )
    assert torch.equal(swa_buf, swa_snap)


@pytest.mark.tpu_test
def test_attention_tpu_forward_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # pragma: no cover
    """Verify 5-layer V4.1 attention TPU7x parity across ratios, ring cap, DSpark."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    import torch_tpu  # noqa: F401

    from vllm_torchtpu.kernels.experimental.deepseek_v41.cache import (
        pack_fp8_kv_rows,
        write_sub_rows,
    )
    from vllm_torchtpu.kernels.experimental.deepseek_v41.indexer import (
        paged_indexer_topk,
    )
    from vllm_torchtpu.kernels.experimental.deepseek_v41.sparse_mla import (
        gather_sparse_fp8_kv,
        sparse_mla_swa_attention,
    )
    from vllm_torchtpu.layers.adapter.custom_ops import deepseek_v41 as dsv41_ops
    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        set_vllm_model_wrapper_context,
    )
    from vllm_torchtpu.runner.utils import relayout_rope_caches

    _patch_vllm_attention_deps(monkeypatch, patch_rope_op=False)

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

    gen = torch.Generator().manual_seed(1234)
    n_tok = 24
    h_cpu = torch.randn((n_tok, 64), generator=gen, dtype=torch.float32).bfloat16()
    qr_cpu = torch.randn((n_tok, 16), generator=gen, dtype=torch.float32).bfloat16()
    kv_cpu = torch.randn((n_tok, 512), generator=gen, dtype=torch.float32).bfloat16()
    kv_s_cpu = torch.randn((n_tok, 1024), generator=gen, dtype=torch.float32)
    idx_w_cpu = torch.randn((n_tok, 8), generator=gen, dtype=torch.float32).bfloat16()

    def _run_on_device(dev: torch.device) -> tuple[list[torch.Tensor], torch.Tensor]:
        s_ctx: dict[str, Any] = {}
        vcfg = _make_vllm_config(
            max_model_len=64, compress_ratios=[0, 2, 2, 1, 1], static_ctx=s_ctx
        )
        hf = vcfg.model_config.hf_config
        hf.num_hidden_layers = 5
        hf.max_position_embeddings = 64
        hf.num_attention_heads = 8
        hf.index_n_heads = 8
        hf.sliding_window = 128
        hf.index_topk = 32
        hf.kv_source_layer_ids = [1, 3]
        hf.index_source_layer_ids = [1, 2, 3, 4]
        hf.candidate_source_layer_id = 3
        hf.candidate_topk_blocks = 64
        hf.candidate_block_size = 2
        vcfg.cache_config.block_size = 16
        vcfg.scheduler_config.max_num_batched_tokens = n_tok
        topk_buf = torch.full((n_tok, 32), -1, dtype=torch.int32, device=dev)
        cand_buf = torch.full((n_tok, 64), -1, dtype=torch.int32, device=dev)
        layers: list[DeepseekV41Attention] = []
        for i in range(5):
            lyr = DeepseekV41Attention(
                vcfg,
                f"model.layers.{i}.self_attn",
                topk_indices_buffer=topk_buf,
                candidate_block_buffer=cand_buf,
            ).to(dev)
            if lyr.is_kv_source and lyr.indexer is not None:
                s_ctx[f"model.layers.{i}.self_attn.indexer.k_cache"] = (
                    lyr.indexer.k_cache
                )
            if dev.type == "tpu":
                relayout_rope_caches(lyr)
            lyr.swa_cache_layer.kv_cache = torch.zeros(
                (8, 16, 512), dtype=torch.bfloat16, device=dev
            )
            layers.append(lyr)
        layers[0].non_causal_multi_token_decode = True
        layers[0].dspark_block_size = 5
        assert layers[1].compressor is not None
        assert layers[1].compressor.state_cache is not None
        layers[1].compressor.state_cache.kv_cache = torch.zeros(
            (8, 16, 1024), dtype=torch.float32, device=dev
        )
        for src in (layers[1], layers[3]):
            src.bind_kv_cache(
                torch.zeros((8, 16, 512), dtype=torch.uint8, device=dev),
                raw_cache=torch.zeros((128, 128), dtype=torch.int32, device=dev),
                layout=((8, 16, 512), torch.uint8, 8192, 0),
                rope_pool=torch.zeros((32, 128), dtype=torch.int32, device=dev),
            )
            assert src.indexer is not None
            src.indexer.k_cache.bind_kv_cache(
                torch.zeros((8, 16, 128), dtype=torch.bfloat16, device=dev),
                raw_cache=torch.zeros((64, 128), dtype=torch.int32, device=dev),
                layout=((8, 16, 128), torch.bfloat16, 4096, 0),
            )
        bt = torch.full((2, 10), -1, dtype=torch.int32, device=dev)
        bt[0, :2] = torch.tensor([1, 2], dtype=torch.int32, device=dev)
        bt[1, :2] = torch.tensor([3, 4], dtype=torch.int32, device=dev)
        meta = _make_meta(
            block_tables=bt, seq_lens=[20, 4], query_start_loc=[0, 20, 24], device=dev
        )
        h = h_cpu.to(dev)
        qr = qr_cpu.to(dev)
        kv = kv_cpu.to(dev)
        kv_s = kv_s_cpu.to(dev)
        idx_w = idx_w_cpu.to(dev)
        pos = torch.cat((torch.arange(20), torch.arange(4))).to(
            device=dev, dtype=torch.int32
        )
        outs: list[torch.Tensor] = []
        with (
            set_vllm_model_wrapper_context(mesh=None),
            _mock_forward_context(monkeypatch, meta),
        ):
            for lyr in layers:
                out_i = torch.zeros((n_tok, 8, 512), dtype=torch.bfloat16, device=dev)
                use_c = kv_s if lyr.compressor is not None else None
                use_w = idx_w if lyr.indexer is not None else None
                lyr._prepare_and_attn(h, qr, kv, None, use_c, use_w, pos, out_i)
                outs.append(out_i.cpu())
        return outs, cand_buf.cpu()

    tpu7x_mesh = jax.sharding.AbstractMesh(
        (), (), abstract_device=jax.sharding.AbstractDevice("TPU7x", 1, "tpu")
    )
    with (
        monkeypatch.context() as mp,
        jax.default_device(jax.devices("cpu")[0]),
        jax.sharding.use_abstract_mesh(tpu7x_mesh),
    ):
        for cls in (DeepseekCompressor, DeepseekV41Indexer, DeepseekV41Attention):
            mp.setattr(cls, "kv_rope_op", property(lambda _s: _cpu_kv_rope))
        mp.setattr(
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
        mp.setattr(
            dsv41_ops.cache_op,
            "dsv41_pack_fp8_kv_rows",
            lambda lat: _t(pack_fp8_kv_rows(_j(lat), interpret=True)),
        )
        mp.setattr(
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
        mp.setattr(
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
        mp.setattr(
            dsv41_ops.attention_op,
            "dsv41_sparse_mla_swa_attention",
            lambda q,
            gn,
            gr,
            tk,
            sk,
            sl,
            cu,
            sc,
            win,
            swa=None,
            cr=1,
            swa_min_pos=None: _t(
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
                    swa_min_pos=_j(swa_min_pos),
                    interpret=True,
                )
            ),
        )
        outs_cpu, cand_cpu = _run_on_device(torch.device("cpu"))

    outs_tpu, cand_tpu = _run_on_device(torch.device("tpu:0"))
    assert cand_tpu.shape == (n_tok, 64) and (cand_tpu[:, 10:] == -1).all()
    assert torch.equal(cand_tpu, cand_cpu)
    for out_tpu, out_cpu in zip(outs_tpu, outs_cpu, strict=True):
        assert torch.allclose(out_tpu, out_cpu, rtol=5e-2, atol=5e-2)
