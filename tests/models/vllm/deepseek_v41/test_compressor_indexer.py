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
"""Unit and parity tests for DeepSeek-V4.1 metadata, compressor, and indexer."""

from __future__ import annotations

import inspect
from collections import Counter
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import vllm.forward_context as fc_mod
from vllm.ir.ops import rms_norm
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbeddingBase
from vllm.v1.attention.backend import MultipleOf

from vllm_torchtpu.models.vllm.deepseek_v41 import (
    compressor as compressor_mod,
)
from vllm_torchtpu.models.vllm.deepseek_v41 import (
    indexer as indexer_mod,
)
from vllm_torchtpu.models.vllm.deepseek_v41 import (
    metadata as metadata_mod,
)
from vllm_torchtpu.models.vllm.deepseek_v41.cache import (
    NativeCacheView,
    PagedCacheRows,
    write_cache,
)
from vllm_torchtpu.models.vllm.deepseek_v41.compressor import (
    AttentionBackend,
    CompressorStateCache,
    DeepseekCompressor,
    save_compress_norm,
)
from vllm_torchtpu.models.vllm.deepseek_v41.indexer import (
    INDEX_HEAD_DIM,
    DeepseekV41Indexer,
    IndexerCache,
    TpuSparseAttnIndexer,
    publish_candidate_blocks,
)
from vllm_torchtpu.models.vllm.deepseek_v41.metadata import (
    active_cache,
    block_table,
    cache_slots,
    layer_meta,
    not_replayed,
    replay_starts,
    rotate_rope,
    token_requests,
)


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


_ACTIVE_VLLM_CONFIG: list[Any] = []


def _make_vllm_config(
    *,
    max_model_len: int = 16,
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
        compress_ratios=[2, 2, 1, 0],
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
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8, max_num_seqs=4),
        attention_config=SimpleNamespace(
            indexer_sparse_logits=False, indexer_kv_dtype="fp8"
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


def _patch_upstream_deps(
    monkeypatch: pytest.MonkeyPatch, *, patch_rope_op: bool = True
) -> None:
    import vllm.config.vllm as vllm_cfg_mod

    if patch_rope_op:
        for cls in (DeepseekCompressor, DeepseekV41Indexer):
            monkeypatch.setattr(cls, "kv_rope_op", property(lambda _s: _cpu_kv_rope))

    default_cfg = _make_vllm_config(static_ctx={})
    monkeypatch.setattr(vllm_cfg_mod, "_current_vllm_config", default_cfg)
    monkeypatch.setattr(
        vllm_cfg_mod, "get_current_vllm_config", lambda: _ACTIVE_VLLM_CONFIG[0]
    )
    for mod in (indexer_mod.upstream, compressor_mod.compressor_upstream):
        monkeypatch.setattr(
            mod,
            "get_current_vllm_config",
            lambda: _ACTIVE_VLLM_CONFIG[0],
        )

    def _merged_col(
        in_f: int,
        sizes: list[int],
        *,
        bias: bool,
        return_bias: bool = True,
        quant_config: Any = None,
        disable_tp: bool = False,
        prefix: str = "",
    ) -> nn.Module:
        del bias, quant_config, disable_tp, prefix
        return _DummyLinear(in_f, sum(sizes), return_bias=return_bias)

    def _replicated(
        in_f: int,
        out_f: int,
        *,
        bias: bool = False,
        quant_config: Any = None,
        prefix: str = "",
    ) -> nn.Module:
        del bias, quant_config, prefix
        return _DummyLinear(in_f, out_f, return_bias=True)

    monkeypatch.setattr(
        compressor_mod.compressor_upstream,
        "MergedColumnParallelLinear",
        _merged_col,
    )
    monkeypatch.setattr(
        compressor_mod.compressor_upstream,
        "RMSNorm",
        _DummyNorm,
    )
    monkeypatch.setattr(indexer_mod.upstream, "ReplicatedLinear", _replicated)
    monkeypatch.setattr(indexer_mod.upstream, "RMSNorm", _DummyNorm)
    monkeypatch.setattr(indexer_mod.upstream, "dsa_indexer_uses_fp4", lambda _c: False)
    monkeypatch.setattr(
        indexer_mod.upstream, "get_max_prefill_buffer_size", lambda _c: 32
    )


def test_c4_and_c5_contract_surface() -> None:
    """Verify exported public surfaces and descriptive parameter names."""
    assert not hasattr(DeepseekV41Indexer, "produce_k")
    assert hasattr(DeepseekV41Indexer, "_produce_k")
    assert not hasattr(indexer_mod, "_publish_candidate_blocks")
    assert callable(publish_candidate_blocks)
    for helper_name in (
        "rotate_rope",
        "active_cache",
        "layer_meta",
        "block_table",
        "query_token_requests",
        "token_requests",
        "cache_slots",
        "replay_starts",
        "not_replayed",
    ):
        assert hasattr(metadata_mod, helper_name)
    assert tuple(inspect.signature(rotate_rope).parameters) == (
        "inputs",
        "positions",
        "rotary_emb",
        "kv_rope_op",
    )
    assert tuple(inspect.signature(cache_slots).parameters) == (
        "cache",
        "meta",
        "positions",
        "compress_ratio",
        "ring",
    )
    assert tuple(inspect.signature(not_replayed).parameters) == (
        "meta",
        "positions",
        "window_size",
    )
    assert tuple(inspect.signature(publish_candidate_blocks).parameters) == (
        "topk_indices",
        "positions",
        "query_end",
        "compress_ratio",
        "candidate_block_size",
        "candidate_topk_blocks",
    )
    assert "layer.paged_cache_rows" in inspect.getsource(active_cache)


def test_rotate_rope_and_active_cache() -> None:
    rope = _DummyRope(max_pos=8, rope_dim=4)
    pos = torch.tensor([-3, 0, 2, 99], dtype=torch.int64)
    x2d = torch.randn((6, 4), dtype=torch.bfloat16).T
    assert not x2d.is_contiguous()
    x3d = torch.randn((4, 2, 6), dtype=torch.bfloat16)

    seen: list[tuple[bool, torch.dtype, list[int]]] = []

    def _spy_op(x: torch.Tensor, p: torch.Tensor, cs: torch.Tensor) -> torch.Tensor:
        assert cs is rope.cos_sin_cache
        seen.append((x.is_contiguous(), p.dtype, p.tolist()))
        return _cpu_kv_rope(x, p, cs)

    out2d = rotate_rope(x2d, pos, rope, _spy_op)
    out3d = rotate_rope(x3d, pos, rope, _spy_op)
    assert out2d.shape == (4, 6) and out3d.shape == (4, 2, 6)
    assert seen == [
        (True, torch.int32, [0, 0, 2, 7]),
        (True, torch.int32, [0, 0, 2, 7]),
    ]

    native = NativeCacheView()
    dense_cache = torch.zeros((3, 4), dtype=torch.bfloat16)
    native.bind_kv_cache(dense_cache)
    assert active_cache(native) is dense_cache
    native.bind_kv_cache(
        dense_cache,
        raw_cache=torch.zeros((2, 128), dtype=torch.int32),
        layout=((1, 4, 128), torch.bfloat16, 1024, 0),
    )
    assert native.paged_cache_rows is not None
    assert active_cache(native) is native.paged_cache_rows


def test_metadata_cache_slots_and_replay_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    meta_1d = _make_meta(
        block_tables=torch.tensor([0, 2, 1, 3], dtype=torch.int32),
        seq_lens=[5, 4],
        query_start_loc=[0, 2, 4],
    )
    with _mock_forward_context(monkeypatch, {"l0": meta_1d}):
        assert layer_meta("l0") is meta_1d
        assert layer_meta("missing") is None
    with _mock_forward_context(monkeypatch, meta_1d):
        assert layer_meta("l0") is meta_1d

    bt_2d = block_table(meta_1d)
    assert bt_2d.shape == (2, 2)
    pos = torch.tensor([0, 4, 1, 3, 0], dtype=torch.int32)
    assert token_requests(meta_1d, pos).tolist() == [0, 0, 1, 1, 2]

    plain_cache = torch.zeros((4, 2, 8), dtype=torch.bfloat16)
    slots_plain = cache_slots(plain_cache, meta_1d, pos, compress_ratio=1, ring=False)
    slots_ring = cache_slots(plain_cache, meta_1d, pos, compress_ratio=1, ring=True)
    # Req 0 block_tables=[0, 2]: plain allows block 0 (slot 0), clamps col=2->1
    # (block 2); trailing padded token 4 (>= qsl[-1]=4) is masked to -1.
    assert slots_plain.tolist() == [0, 4, 3, 7, -1]
    assert slots_ring.tolist() == [0, 0, 3, 3, -1]

    paged_null = PagedCacheRows(
        torch.zeros((8, 128), dtype=torch.int32),
        ((4, 2, 128), torch.bfloat16, 512, 0),
        null_block=True,
    )
    # With null_block=True, block 0 on token 0 is masked to -1.
    assert cache_slots(paged_null, meta_1d, pos).tolist() == [-1, 4, 3, 7, -1]
    paged_no_null = PagedCacheRows(
        torch.zeros((8, 128), dtype=torch.int32),
        ((4, 2, 128), torch.bfloat16, 512, 0),
        null_block=False,
    )
    assert cache_slots(paged_no_null, meta_1d, pos).tolist() == [0, 4, 3, 7, -1]

    empty_meta = _make_meta(seq_lens=[], query_start_loc=[0])
    assert (cache_slots(plain_cache, empty_meta, pos) == -1).all()

    dev = torch.device("cpu")
    with _mock_forward_context(
        monkeypatch,
        meta_1d,
        {
            "deepseek_v41_window_starts": torch.tensor([], dtype=torch.int32),
            "deepseek_v41_replay_starts": "not_a_tensor",
        },
    ):
        assert replay_starts(2, dev, ced_decoder=False) is None
        assert replay_starts(2, dev, ced_decoder=True) is None
        assert not_replayed(meta_1d, pos, 2).all()

    w_t = torch.tensor([2, 6], dtype=torch.int32)
    r_t = torch.tensor([4, 1], dtype=torch.int32)
    with _mock_forward_context(
        monkeypatch,
        meta_1d,
        {"deepseek_v41_window_starts": w_t, "deepseek_v41_replay_starts": r_t},
    ):
        assert replay_starts(3, dev, ced_decoder=False).tolist() == [2, 6, 6]
        assert replay_starts(3, dev, ced_decoder=True).tolist() == [4, 6, 6]

    with _mock_forward_context(
        monkeypatch, meta_1d, {"deepseek_v41_window_starts": w_t}
    ):
        assert replay_starts(2, dev, ced_decoder=True).tolist() == [2, 6]

    with _mock_forward_context(
        monkeypatch, meta_1d, {"deepseek_v41_replay_starts": r_t}
    ):
        assert replay_starts(2, dev, ced_decoder=False).tolist() == [4, 1]
        assert replay_starts(2, dev, ced_decoder=True).tolist() == [4, 1]

    # Req 0 replay_start=4, win=4 masks [4..7] False while keeping suffix [8..9] True;
    # Req 1 replay_start=0 keeps [0..3] True.
    rep_meta = _make_meta(seq_lens=[10, 4], query_start_loc=[0, 6, 10])
    rep_pos = torch.tensor([4, 5, 6, 7, 8, 9, 0, 1, 2, 3], dtype=torch.int32)
    with _mock_forward_context(
        monkeypatch,
        rep_meta,
        {"deepseek_v41_replay_starts": torch.tensor([4, 0], dtype=torch.int32)},
    ):
        assert not_replayed(rep_meta, rep_pos, 4).tolist() == [
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
        ]


def test_backend_and_cache_classes(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_upstream_deps(monkeypatch)
    vcfg = _make_vllm_config()
    assert AttentionBackend.get_supported_kernel_block_sizes() == [MultipleOf(1)]
    assert AttentionBackend.get_page_size(vcfg) == 4
    vcfg.cache_config.block_size = 32
    assert AttentionBackend.get_page_size(vcfg) == 32
    vcfg.cache_config.block_size = None
    assert AttentionBackend.get_page_size(vcfg) == 16
    vcfg.cache_config.block_size = 4
    assert AttentionBackend.get_min_page_size(vcfg) == 1

    state_cache = CompressorStateCache(1024, torch.float32, "l0.comp.state")
    assert state_cache.get_attn_backend() is AttentionBackend

    idx_cache = IndexerCache(
        INDEX_HEAD_DIM, torch.bfloat16, "l0.idx.k", vcfg.cache_config, 2
    )
    assert idx_cache.get_attn_backend() is AttentionBackend
    idx_spec = idx_cache.get_kv_cache_spec(vcfg)
    assert idx_spec.dtype == torch.bfloat16 and idx_spec.page_size_padded is None

    with pytest.raises(ValueError, match="IndexerCache requires head_dim=128"):
        IndexerCache(64, torch.bfloat16, "l0.bad_hd", vcfg.cache_config, 2)
    with pytest.raises(ValueError, match="IndexerCache requires head_dim=128"):
        IndexerCache(128, torch.float32, "l0.bad_dt", vcfg.cache_config, 2)


def test_save_compress_norm_and_c1_dirty_pool_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_upstream_deps(monkeypatch)
    published_masks: list[torch.Tensor] = []
    after_flags: list[bool] = []
    monkeypatch.setattr(
        compressor_mod,
        "write_cache",
        lambda _c, _s, _v, m, after_read=False: (
            published_masks.append(m.clone()),
            after_flags.append(after_read),
        ),
    )

    kv_score = torch.ones((3, 8), dtype=torch.float32)
    pos = torch.tensor([1, 2, 3], dtype=torch.int32)
    weight = torch.ones(4, dtype=torch.float32)
    state = torch.ones((2, 4, 8), dtype=torch.float32)
    slots = torch.tensor([0, 1, 2], dtype=torch.int32)

    out_cr1 = save_compress_norm(kv_score, pos, state, slots, weight, 1e-6, 1)
    out_none_slots = save_compress_norm(kv_score, pos, state, None, weight, 1e-6, 2)
    out_empty_state = save_compress_norm(
        kv_score, pos, torch.tensor([]), slots, weight, 1e-6, 2
    )
    assert torch.equal(out_cr1, out_none_slots)
    assert torch.equal(out_cr1, out_empty_state)
    assert out_cr1.shape == (3, 4) and out_cr1.dtype == torch.bfloat16

    out_no_qsl = save_compress_norm(kv_score, pos, state, slots, weight, 1e-6, 2)
    out_qsl1 = save_compress_norm(
        kv_score,
        pos,
        state,
        slots,
        weight,
        1e-6,
        2,
        torch.tensor([0], dtype=torch.int32),
    )
    out_qsl2 = save_compress_norm(
        kv_score,
        pos,
        state,
        slots,
        weight,
        1e-6,
        2,
        torch.tensor([0, 3], dtype=torch.int32),
    )
    assert torch.equal(out_no_qsl, out_qsl1)
    assert torch.equal(out_no_qsl, out_qsl2)

    # 24-token prefill chunk exceeding a 16-row ring buffer publishes only trailing 16.
    pos24 = torch.arange(24, dtype=torch.int32)
    slots24 = pos24 % 16
    state16 = torch.zeros((1, 16, 8), dtype=torch.float32)
    qsl24 = torch.tensor([0, 24], dtype=torch.int32)
    kv24 = torch.randn((24, 8), dtype=torch.float32)
    save_compress_norm(kv24, pos24, state16, slots24, weight, 1e-6, 2, qsl24)
    pub24 = published_masks[-1]
    assert not pub24[:8].any() and pub24[8::2].all() and not pub24[9::2].any()
    assert after_flags == [True, True, True, True]

    # Dirty NaN in state_cache must not poison even or in-chunk paired rows
    # (torch.where avoids 0 * NaN), MUST surface as NaN on unzeroed cross-step odd
    # rows (zero fail-silent element-wise isfinite masking), and becomes clean after
    # CompressorStateCache.zero_blocks clears the newly claimed block.
    dirty_state = torch.full((2, 4, 8), float("nan"), dtype=torch.float32)
    kv1 = torch.ones((1, 8), dtype=torch.float32)
    qsl_1 = torch.tensor([0, 1], dtype=torch.int32)
    out_even = save_compress_norm(
        kv1,
        torch.tensor([0], dtype=torch.int32),
        dirty_state,
        torch.tensor([0], dtype=torch.int32),
        weight,
        1e-6,
        2,
        qsl_1,
    )
    out_pair = save_compress_norm(
        torch.ones((2, 8), dtype=torch.float32),
        torch.tensor([0, 1], dtype=torch.int32),
        dirty_state,
        torch.tensor([0, 1], dtype=torch.int32),
        weight,
        1e-6,
        2,
        torch.tensor([0, 2], dtype=torch.int32),
    )
    out_unzeroed_odd = save_compress_norm(
        kv1,
        torch.tensor([5], dtype=torch.int32),
        dirty_state,
        torch.tensor([5], dtype=torch.int32),
        weight,
        1e-6,
        2,
        qsl_1,
    )
    assert torch.isfinite(out_even).all()
    assert torch.isfinite(out_pair).all()
    assert torch.isnan(out_unzeroed_odd).any()

    comp_cache = CompressorStateCache(8, torch.float32, "l0.state_zero")
    comp_cache.kv_cache = dirty_state
    comp_cache.zero_blocks([1])
    assert torch.isfinite(dirty_state[1]).all() and (dirty_state[1] == 0.0).all()
    out_zeroed_odd = save_compress_norm(
        kv1,
        torch.tensor([5], dtype=torch.int32),
        dirty_state,
        torch.tensor([5], dtype=torch.int32),
        weight,
        1e-6,
        2,
        qsl_1,
    )
    assert torch.isfinite(out_zeroed_odd).all()


def test_deepseek_compressor_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_upstream_deps(monkeypatch)
    writes: list[tuple[torch.Tensor, list[bool]]] = []
    monkeypatch.setattr(
        compressor_mod,
        "write_cache",
        lambda _c, s, _v, keep, after_read=False: (
            None if after_read else writes.append((s.clone(), keep.tolist()))
        ),
    )
    s_ctx: dict[str, Any] = {}
    vcfg = _make_vllm_config(static_ctx=s_ctx)
    k_holder = NativeCacheView()
    k_holder.kv_cache = torch.tensor([])
    s_ctx["l0"] = k_holder

    comp1 = DeepseekCompressor(
        vcfg,
        compress_ratio=1,
        hidden_size=64,
        head_dim=512,
        rotate=True,
        prefix="l0.c1",
        k_cache_prefix="l0",
    )
    comp2 = DeepseekCompressor(
        vcfg,
        compress_ratio=2,
        hidden_size=64,
        head_dim=512,
        rotate=True,
        prefix="l0.c2",
        k_cache_prefix="l0",
    )
    assert comp1.state_cache is None
    assert isinstance(comp2.state_cache, CompressorStateCache)
    assert s_ctx["l0.c2.state_cache"] is comp2.state_cache

    h = torch.ones((4, 64), dtype=torch.bfloat16)
    pos = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    rope = _DummyRope(max_pos=16, rope_dim=64)
    kv_s1 = comp1.project(h)
    kv_s2 = comp2.project(h)
    assert kv_s1.shape == (4, 512) and kv_s2.shape == (4, 1024)

    # meta=None, latent=None, and empty k_cache / state_cache branches.
    with _mock_forward_context(monkeypatch, None):
        assert comp1(kv_s1, pos).shape == (4, 512)
        assert comp2(kv_s2, pos).shape == (4, 512)
        comp2.insert_cache(None, pos, rope)
        comp2.insert_cache(torch.ones((4, 512), dtype=torch.bfloat16), pos, rope)
    assert len(writes) == 0

    with _mock_forward_context(monkeypatch, _make_meta()):
        # Empty k_cache still skips write_cache.
        comp2.insert_cache(torch.ones((4, 512), dtype=torch.bfloat16), pos, rope)
        assert len(writes) == 0
        comp2.state_cache.kv_cache = torch.zeros((4, 8, 1024), dtype=torch.float32)
        k_holder.kv_cache = torch.zeros((4, 4, 512), dtype=torch.bfloat16)
        lat1 = comp1(kv_s1, pos)
        lat2 = comp2(kv_s2, pos)
        comp1.insert_cache(lat1, pos, rope)
        comp2.insert_cache(lat2, pos, rope)

    assert len(writes) == 2
    assert writes[0][1] == [True, True, True, True]
    assert writes[1][1] == [False, True, False, True]


def test_publish_candidate_blocks_and_indexer_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_upstream_deps(monkeypatch)
    # Direct publish_candidate_blocks branch tests: pad_cols > 0 and pad_cols == 0.
    topk_narrow = torch.tensor([[0, 0], [4, 2], [2, 2]], dtype=torch.int32)
    pos_3 = torch.tensor([3, 15, 0], dtype=torch.int32)
    q_end_2 = torch.tensor(2, dtype=torch.int32)
    # c_k=5 > 1 + topk.shape[1]=3 -> pad_cols=2 > 0; row 2 >= q_end_2 -> all -1.
    pub_pad = publish_candidate_blocks(topk_narrow, pos_3, q_end_2, 2, 2, 5)
    assert pub_pad[0].tolist() == [0, -1, -1, -1, -1]
    assert pub_pad[1].tolist() == [0, 1, 2, 3, -1]
    assert pub_pad[2].tolist() == [-1, -1, -1, -1, -1]
    # n_blks=4 > c_k=2 with topk width 4 -> pad_cols=0, pins tail=3 + sorted unique.
    topk_wide = torch.tensor([[7, 2, 2, 5]], dtype=torch.int32)
    pub_dedup = publish_candidate_blocks(
        topk_wide,
        torch.tensor([15], dtype=torch.int32),
        torch.tensor(1, dtype=torch.int32),
        2,
        2,
        3,
    )
    assert pub_dedup[0].tolist() == [3, 1, 2]

    # Unbound TpuSparseAttnIndexer._select_topk fails fast with RuntimeError.
    s_ctx: dict[str, Any] = {}
    vcfg = _make_vllm_config(static_ctx=s_ctx)
    k_cache = IndexerCache(
        INDEX_HEAD_DIM, torch.bfloat16, "l0.idx.k", vcfg.cache_config, 2
    )
    raw_op = TpuSparseAttnIndexer(k_cache)
    with pytest.raises(RuntimeError, match="callback is unbound"):
        raw_op(
            torch.ones((1, 64)),
            torch.ones((1, 4, 128)),
            torch.zeros(1, dtype=torch.int32),
            torch.ones((1, 4, 1)),
        )

    writes: list[list[bool]] = []
    monkeypatch.setattr(
        indexer_mod,
        "write_cache",
        lambda _c, _s, _v, keep, after_read=False: writes.append(keep.tolist()),
    )
    seen_calls: list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]] = []

    def _mock_topk(
        q: torch.Tensor,
        _w: torch.Tensor,
        _kl: torch.Tensor,
        bt: torch.Tensor,
        _sl: torch.Tensor,
        _cu: torch.Tensor,
        tk: int,
        kh: torch.Tensor | None = None,
        ci: torch.Tensor | None = None,
        _cr: int = 1,
    ) -> torch.Tensor:
        seen_calls.append((bt.clone(), kh, None if ci is None else ci.clone()))
        base = torch.tensor([0, 0, 2, 2], dtype=torch.int32)
        return base.unsqueeze(0).expand(q.shape[0], tk).clone()

    monkeypatch.setattr(indexer_mod.indexer_op, "dsv41_paged_indexer_topk", _mock_topk)

    topk_buf = torch.full((4, 4), -1, dtype=torch.int32)
    cand_wide = torch.full((4, 8), -9, dtype=torch.int32)
    idx = DeepseekV41Indexer(
        vcfg,
        vcfg.model_config.hf_config,
        64,
        16,
        None,
        vcfg.cache_config,
        topk_buf,
        2,
        "l0.idx",
        owns_k=True,
        k_cache=k_cache,
        main_head_dim=512,
        candidate_block_buffer=cand_wide,
        candidate_block_size=2,
        candidate_write=True,
    )

    qr = torch.ones((2, 16), dtype=torch.bfloat16)
    lat = torch.ones((2, 512), dtype=torch.bfloat16)
    w = torch.ones((2, 4), dtype=torch.bfloat16)
    pos = torch.tensor([3, 15], dtype=torch.int32)
    rope = _DummyRope(max_pos=32, rope_dim=64)

    # Calling select_topk before forward() without positions raises ValueError.
    with pytest.raises(ValueError, match="requires positions"):
        idx.select_topk(
            torch.ones((2, 4, 128), dtype=torch.bfloat16),
            torch.ones((2, 4, 1), dtype=torch.float32),
        )

    # meta=None and latent=None paths.
    with _mock_forward_context(monkeypatch, None):
        idx._produce_k(None, pos, rope)
        idx._produce_k(lat, pos, rope)
        q_s, q_scale, wt = idx(qr, None, w, pos, rope)
        assert q_s is not None and q_scale is None and wt is not None
        assert (idx.select_topk(q_s, wt) == -1).all()
    assert len(writes) == 0

    # Plain tensor k_cache, candidate_write=True, tuple q_quant via indexer_op.
    k_cache.kv_cache = torch.zeros((4, 4, 128), dtype=torch.int32)
    with _mock_forward_context(
        monkeypatch, _make_meta(seq_lens=[4, 16], query_start_loc=[0, 1, 2])
    ):
        q_s, _, wt = idx(qr, lat, w, pos, rope)
        assert q_s is not None and wt is not None
        assert writes == [[True, True]]
        idx.indexer_op(qr, (q_s, None), None, wt)
        assert cand_wide[0].tolist() == [0, -1, -1, -1, -1, -1, -1, -1]
        assert cand_wide[1].tolist() == [0, 1, 2, 3, -1, -1, -1, -1]
        assert seen_calls[-1][1] is None and seen_calls[-1][2] is None

        # Narrow candidate writer (c_k=2 < n_blks[1]=4) pins tail=3.
        cand_narrow = torch.full((4, 2), -9, dtype=torch.int32)
        idx.indexer_op.candidate_blocks = cand_narrow
        idx.indexer_op(qr, q_s, pos, wt)
        assert cand_narrow[0].tolist() == [0, -1]
        assert cand_narrow[1].tolist() == [3, 0]

        # Non-owning consumer with 2-bank PagedCacheRows, null_block=True, non-zero
        # Non-owning consumer with 2-bank PagedCacheRows, null_block=True, non-zero
        # row_offset (page_stride//r = 4, row_offset//r = 2): 0 -> -1, 1 -> 6,
        # 2 -> 10, 3 -> 14.
        idx_consumer = DeepseekV41Indexer(
            vcfg,
            vcfg.model_config.hf_config,
            64,
            16,
            None,
            vcfg.cache_config,
            None,
            2,
            "l1.idx",
            owns_k=False,
            k_cache=k_cache,
            main_head_dim=512,
            candidate_block_buffer=cand_wide,
            candidate_block_size=2,
            candidate_write=False,
        )
        idx_consumer._produce_k(lat, pos, rope)
        k_cache.bind_kv_cache(
            torch.zeros((4, 4, 128), dtype=torch.bfloat16),
            raw_cache=(
                torch.zeros((32, 128), dtype=torch.int32),
                torch.zeros((32, 128), dtype=torch.int32),
            ),
            layout=((4, 4, 128), torch.bfloat16, 4096, 2048),
        )
        # Wide candidate buffer (8 * 2 = 16 >= max_keys = 2 * 2 * 2 = 8) skips cand_idx.
        idx_consumer(qr, lat, w, pos, rope)
        idx_consumer.select_topk(q_s, wt, pos)
        bt_mapped, kh_mapped, ci_wide = seen_calls[-1]
        assert bt_mapped.tolist() == [[-1, 6], [10, 14]]
        assert kh_mapped is not None and ci_wide is None

        # Narrow candidate buffer (2 * 2 = 4 < max_keys = 8) expands cand_idx;
        # also test PagedCacheRows with null_block=False and 1 bank.
        k_cache.bind_kv_cache(
            torch.zeros((4, 4, 128), dtype=torch.bfloat16),
            raw_cache=torch.zeros((16, 128), dtype=torch.int32),
            layout=((4, 4, 128), torch.bfloat16, 1024, 0),
        )
        assert k_cache.paged_cache_rows is not None
        k_cache.paged_cache_rows.null_block = False
        cand_consumer = torch.tensor(
            [[0, -1], [3, 0], [0, 0], [0, 0]], dtype=torch.int32
        )
        idx_consumer.indexer_op.candidate_blocks = cand_consumer
        out = idx_consumer.select_topk(q_s, wt, pos)
        bt_1b, kh_1b, ci_narrow = seen_calls[-1]
        assert bt_1b.tolist() == [[0, 1], [2, 3]]
        assert kh_1b is None and ci_narrow is not None
        assert ci_narrow.tolist() == [[0, 1, -1, -1], [6, 7, 0, 1]]
        assert out.shape == (2, 4)


def test_regression_pin_compressor_and_candidate_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin deterministic outputs of save_compress_norm and candidate blocks."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from vllm_torchtpu.kernels.experimental.deepseek_v41.cache import (
        pack_fp8_kv_rows,
        write_sub_rows,
    )
    from vllm_torchtpu.kernels.experimental.deepseek_v41.indexer import (
        paged_indexer_topk,
    )
    from vllm_torchtpu.layers.adapter.custom_ops import deepseek_v41 as dsv41_ops

    _patch_upstream_deps(monkeypatch)

    def _to_jax(x: Any) -> Any:
        if not isinstance(x, torch.Tensor):
            return x
        arr = x.detach().cpu()
        if arr.dtype == torch.bfloat16:
            return jnp.asarray(arr.float().numpy()).astype(jnp.bfloat16)
        return jnp.asarray(arr.numpy())

    def _to_torch(x: Any) -> Any:
        if isinstance(x, tuple):
            return tuple(_to_torch(v) for v in x)
        if x.dtype == jnp.bfloat16:
            return torch.from_numpy(
                np.array(x.astype(jnp.float32), copy=True)
            ).bfloat16()
        return torch.from_numpy(np.array(x, copy=True))

    tpu7x_mesh = jax.sharding.AbstractMesh(
        (), (), abstract_device=jax.sharding.AbstractDevice("TPU7x", 1, "tpu")
    )
    with (
        jax.default_device(jax.devices("cpu")[0]),
        jax.sharding.use_abstract_mesh(tpu7x_mesh),
    ):
        monkeypatch.setattr(
            dsv41_ops.cache_op,
            "dsv41_write_sub_rows",
            lambda c, u, s, off=0, after=None, page_tokens=16: _to_torch(
                write_sub_rows(
                    _to_jax(c),
                    _to_jax(u),
                    _to_jax(s),
                    slot_offset=int(off),
                    after=_to_jax(after),
                    page_tokens=int(page_tokens),
                    interpret=True,
                )
            ),
        )
        monkeypatch.setattr(
            dsv41_ops.cache_op,
            "dsv41_pack_fp8_kv_rows",
            lambda lat: _to_torch(pack_fp8_kv_rows(_to_jax(lat), interpret=True)),
        )
        monkeypatch.setattr(
            dsv41_ops.indexer_op,
            "dsv41_paged_indexer_topk",
            lambda q, w, kl, bt, sl, cu, tk, kh=None, ci=None, cr=1: _to_torch(
                paged_indexer_topk(
                    _to_jax(q),
                    _to_jax(w),
                    _to_jax(kl),
                    _to_jax(bt),
                    _to_jax(sl),
                    _to_jax(cu),
                    topk=int(tk),
                    k_pool_hi=_to_jax(kh),
                    candidate_indices=_to_jax(ci),
                    compress_ratio=int(cr),
                    interpret=True,
                )
            ),
        )

        # Standalone reference functions for exact bitwise parity checks.
        def _ref_save_compress_norm(
            kv_score: torch.Tensor,
            positions: torch.Tensor,
            state_cache: PagedCacheRows | torch.Tensor,
            slot_mapping: torch.Tensor | None,
            norm_weight: torch.Tensor,
            eps: float,
            compress_ratio: int,
            query_start_loc: torch.Tensor | None = None,
        ) -> torch.Tensor:
            head_dim = norm_weight.shape[0]
            kv_f32, w_f32 = kv_score[:, :head_dim].float(), norm_weight.float()
            if compress_ratio == 1 or slot_mapping is None or not state_cache.numel():
                return rms_norm(kv_f32, w_f32, eps).to(torch.bfloat16)
            pos_i64, valid = positions.to(torch.int64), slot_mapping >= 0
            cap, n_tok = state_cache.shape[1], positions.shape[0]
            rows = torch.arange(n_tok, device=positions.device)
            if query_start_loc is not None and query_start_loc.numel() > 1:
                q_ends = query_start_loc[1:].to(positions.device)
                reqs = (rows.unsqueeze(1) >= q_ends.unsqueeze(0)).sum(-1)
                ends = q_ends[reqs.clamp(max=q_ends.shape[0] - 1)].to(torch.int64)
            else:
                reqs, ends = torch.zeros_like(rows), torch.full_like(pos_i64, n_tok)
            p_idx = (rows - 1).clamp_min(0)
            in_chunk = (
                (rows > 0) & (positions[p_idx] == positions - 1) & (reqs[p_idx] == reqs)
            )
            is_second = ((pos_i64 % 2 == 1) & valid).unsqueeze(-1)
            prev_blocks = (slot_mapping.clamp_min(0) // cap).long()
            prev_slot = (pos_i64 - 1).clamp_min(0) % cap
            raw_prev = state_cache[prev_blocks, prev_slot].to(kv_score.dtype)
            use_prev = is_second & ~in_chunk.unsqueeze(-1)
            prev_rows = torch.where(use_prev, raw_prev, 0)
            paired = torch.where(in_chunk.unsqueeze(-1), kv_score[p_idx], prev_rows)
            pair_kv = torch.stack((paired[:, :head_dim].float(), kv_f32), dim=1)
            s0, s1 = paired[:, head_dim:].float(), kv_score[:, head_dim:].float()
            pooled = (pair_kv * torch.softmax(torch.stack((s0, s1), dim=1), dim=1)).sum(
                dim=1
            )
            pub = (pos_i64 % 2 == 0) & valid & (rows >= ends - cap)
            write_cache(state_cache, slot_mapping, kv_score, pub, after_read=True)
            normed = rms_norm(torch.where(is_second, pooled, kv_f32), w_f32, eps)
            return normed.to(torch.bfloat16)

        def _ref_publish_candidate_blocks(
            topk: torch.Tensor,
            pos: torch.Tensor,
            q_end: torch.Tensor,
            cr: int,
            c_bs: int,
            c_k: int,
        ) -> torch.Tensor:
            n_tok, dev = topk.shape[0], topk.device
            active = (torch.arange(n_tok, device=dev) < q_end.to(dev)) & (pos >= 0)
            c_len = torch.where(active, (pos.int() + 1) // cr, 0)
            n_blks = (c_len + c_bs - 1) // c_bs
            iota = torch.arange(c_k, dtype=torch.int32, device=dev).unsqueeze(0)
            all_b = torch.where(iota < n_blks.unsqueeze(1), iota, -1)
            tail = torch.where(c_len > 0, (c_len - 1) // c_bs, -1).unsqueeze(1)
            raw_b = torch.where(topk >= 0, topk // c_bs, -1)
            sent = 0x7FFFFFFF
            b_s, _ = torch.where((raw_b >= 0) & (raw_b != tail), raw_b, sent).sort(
                dim=1
            )
            dup = F.pad(b_s[:, 1:] == b_s[:, :-1], (1, 0), value=False)
            u_s, _ = torch.where(dup, sent, b_s).sort(dim=1)
            uniq = torch.where(u_s < sent, u_s, -1)
            pad_n = max(0, c_k - 1 - uniq.shape[1])
            extra = (uniq.new_full((n_tok, pad_n), -1),) if pad_n > 0 else ()
            dedup = torch.cat((tail, uniq, *extra), dim=1)[:, :c_k]
            return torch.where((n_blks <= c_k).unsqueeze(1), all_b, dedup)

        gen = torch.Generator().manual_seed(2026)
        kv_score = torch.randn((10, 1024), generator=gen, dtype=torch.float32)
        pos = torch.tensor([0, 1, 2, 3, 4, 5, 1, 2, 3, 4], dtype=torch.int32)
        slots = torch.tensor([0, 1, 2, 3, 4, 5, 9, 10, 11, 12], dtype=torch.int32)
        qsl = torch.tensor([0, 6, 10], dtype=torch.int32)
        norm_w = torch.randn(512, generator=gen, dtype=torch.float32).abs() + 0.5
        s_new = torch.randn((2, 8, 1024), generator=gen, dtype=torch.float32)
        s_ref = s_new.clone()

        out_new = save_compress_norm(kv_score, pos, s_new, slots, norm_w, 1e-6, 2, qsl)
        out_ref = _ref_save_compress_norm(
            kv_score, pos, s_ref, slots, norm_w, 1e-6, 2, qsl
        )
        assert (out_new.float() - out_ref.float()).abs().max().item() == 0.0
        assert (s_new - s_ref).abs().max().item() == 0.0

        # End-to-end compressor + indexer + candidate publisher parity on
        # PagedCacheRows.
        s_ctx: dict[str, Any] = {}
        vcfg = _make_vllm_config(max_model_len=64, static_ctx=s_ctx)
        vcfg.cache_config.block_size = 16
        vcfg.model_config.hf_config.index_topk = 8
        vcfg.model_config.hf_config.candidate_topk_blocks = 3
        vcfg.model_config.hf_config.candidate_block_size = 2

        k_owner = NativeCacheView()
        k_owner.bind_kv_cache(
            torch.zeros((4, 16, 512), dtype=torch.uint8),
            raw_cache=torch.zeros((64, 128), dtype=torch.int32),
            layout=((4, 16, 512), torch.uint8, 8192, 0),
            rope_pool=torch.zeros((16, 128), dtype=torch.int32),
        )
        s_ctx["l0"] = k_owner

        comp = DeepseekCompressor(
            vcfg,
            compress_ratio=2,
            hidden_size=64,
            head_dim=512,
            rotate=True,
            prefix="l0.comp",
            k_cache_prefix="l0",
        )
        assert comp.state_cache is not None
        comp.state_cache.kv_cache = torch.zeros((4, 8, 1024), dtype=torch.float32)

        idx_cache = IndexerCache(
            INDEX_HEAD_DIM, torch.bfloat16, "l0.idx.k", vcfg.cache_config, 2
        )
        idx_cache.bind_kv_cache(
            torch.zeros((4, 16, 128), dtype=torch.bfloat16),
            raw_cache=torch.zeros((32, 128), dtype=torch.int32),
            layout=((4, 16, 128), torch.bfloat16, 4096, 0),
        )
        topk_buf = torch.full((10, 8), -1, dtype=torch.int32)
        cand_buf = torch.full((10, 3), -1, dtype=torch.int32)
        indexer = DeepseekV41Indexer(
            vcfg,
            vcfg.model_config.hf_config,
            64,
            16,
            None,
            vcfg.cache_config,
            topk_buf,
            2,
            "l0.idx",
            owns_k=True,
            k_cache=idx_cache,
            main_head_dim=512,
            candidate_block_buffer=cand_buf,
            candidate_block_size=2,
            candidate_write=True,
        )

        bt = torch.tensor([[1, 2], [3, 0]], dtype=torch.int32)
        meta = _make_meta(block_tables=bt, seq_lens=[18, 6], query_start_loc=[0, 6, 10])
        pos_step = torch.tensor([12, 13, 14, 15, 16, 17, 0, 1, 2, 3], dtype=torch.int32)
        h = torch.randn((10, 64), generator=gen, dtype=torch.float32).bfloat16()
        qr = torch.randn((10, 16), generator=gen, dtype=torch.float32).bfloat16()
        w_in = torch.randn((10, 4), generator=gen, dtype=torch.float32).bfloat16()
        rope = _DummyRope(max_pos=64, rope_dim=64)

        with _mock_forward_context(monkeypatch, meta):
            kv_s = comp.project(h)
            lat = comp(kv_s, pos_step)
            comp.insert_cache(lat, pos_step, rope)
            q_s, _, wt = indexer(qr, lat, w_in, pos_step, rope)
            assert q_s is not None and wt is not None
            topk_out = indexer.select_topk(q_s, wt, pos_step)

        ref_cand = _ref_publish_candidate_blocks(
            topk_out, pos_step, meta.query_start_loc[-1], 2, 2, 3
        )
        assert torch.equal(cand_buf, ref_cand)
        assert torch.equal(topk_buf, topk_out)


@pytest.mark.tpu_test
def test_compressor_indexer_tpu_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify DeepseekCompressor and DeepseekV41Indexer TPU7x vs CPU parity."""
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
    from vllm_torchtpu.layers.adapter.custom_ops import deepseek_v41 as dsv41_ops
    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        set_vllm_model_wrapper_context,
    )
    from vllm_torchtpu.runner.utils import relayout_rope_caches

    _patch_upstream_deps(monkeypatch, patch_rope_op=False)

    def _to_jax(x: Any) -> Any:
        if not isinstance(x, torch.Tensor):
            return x
        arr = x.detach().cpu()
        if arr.dtype == torch.bfloat16:
            return jnp.asarray(arr.float().numpy()).astype(jnp.bfloat16)
        return jnp.asarray(arr.numpy())

    def _to_torch(x: Any) -> Any:
        if isinstance(x, tuple):
            return tuple(_to_torch(v) for v in x)
        if x.dtype == jnp.bfloat16:
            return torch.from_numpy(
                np.array(x.astype(jnp.float32), copy=True)
            ).bfloat16()
        return torch.from_numpy(np.array(x, copy=True))

    gen = torch.Generator().manual_seed(4242)
    n_tok = 24
    h_cpu = torch.randn((n_tok, 64), generator=gen, dtype=torch.float32).bfloat16()
    qr_cpu = torch.randn((n_tok, 16), generator=gen, dtype=torch.float32).bfloat16()
    w_cpu = torch.randn((n_tok, 8), generator=gen, dtype=torch.float32).bfloat16()

    def _run_on_device(
        dev: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s_ctx: dict[str, Any] = {}
        vcfg = _make_vllm_config(max_model_len=64, static_ctx=s_ctx)
        hf = vcfg.model_config.hf_config
        hf.index_n_heads = 8
        hf.index_topk = 32
        hf.candidate_topk_blocks = 8
        hf.candidate_block_size = 2
        vcfg.cache_config.block_size = 16
        vcfg.scheduler_config.max_num_batched_tokens = n_tok

        k_owner = NativeCacheView()
        k_owner.bind_kv_cache(
            torch.zeros((8, 16, 512), dtype=torch.uint8, device=dev),
            raw_cache=torch.zeros((128, 128), dtype=torch.int32, device=dev),
            layout=((8, 16, 512), torch.uint8, 8192, 0),
            rope_pool=torch.zeros((32, 128), dtype=torch.int32, device=dev),
        )
        s_ctx["l0"] = k_owner

        comp = DeepseekCompressor(
            vcfg,
            compress_ratio=2,
            hidden_size=64,
            head_dim=512,
            rotate=True,
            prefix="l0.comp",
            k_cache_prefix="l0",
        ).to(dev)
        assert comp.state_cache is not None
        comp.state_cache.kv_cache = torch.zeros(
            (8, 16, 1024), dtype=torch.float32, device=dev
        )

        idx_cache = IndexerCache(
            INDEX_HEAD_DIM, torch.bfloat16, "l0.idx.k", vcfg.cache_config, 2
        )
        idx_cache.bind_kv_cache(
            torch.zeros((8, 16, 128), dtype=torch.bfloat16, device=dev),
            raw_cache=torch.zeros((64, 128), dtype=torch.int32, device=dev),
            layout=((8, 16, 128), torch.bfloat16, 4096, 0),
        )
        topk_buf = torch.full((n_tok, 32), -1, dtype=torch.int32, device=dev)
        cand_buf = torch.full((n_tok, 8), -1, dtype=torch.int32, device=dev)
        indexer = DeepseekV41Indexer(
            vcfg,
            hf,
            64,
            16,
            None,
            vcfg.cache_config,
            topk_buf,
            2,
            "l0.idx",
            owns_k=True,
            k_cache=idx_cache,
            main_head_dim=512,
            candidate_block_buffer=cand_buf,
            candidate_block_size=2,
            candidate_write=True,
        ).to(dev)

        rope = _DummyRope(max_pos=64, rope_dim=64).to(dev)
        if dev.type == "tpu":
            relayout_rope_caches(rope)

        bt = torch.full((2, 8), -1, dtype=torch.int32, device=dev)
        bt[0, :2] = torch.tensor([1, 2], dtype=torch.int32, device=dev)
        bt[1, :2] = torch.tensor([3, 4], dtype=torch.int32, device=dev)
        meta = _make_meta(
            block_tables=bt, seq_lens=[20, 4], query_start_loc=[0, 20, 24], device=dev
        )
        pos = torch.cat((torch.arange(20), torch.arange(4))).to(
            device=dev, dtype=torch.int32
        )
        with (
            set_vllm_model_wrapper_context(mesh=None),
            _mock_forward_context(monkeypatch, meta),
        ):
            kv_s = comp.project(h_cpu.to(dev))
            lat = comp(kv_s, pos)
            comp.insert_cache(lat, pos, rope)
            q_s, _, wt = indexer(qr_cpu.to(dev), lat, w_cpu.to(dev), pos, rope)
            assert q_s is not None and wt is not None
            indexer.select_topk(q_s, wt, pos)
        return lat.cpu(), topk_buf.cpu(), cand_buf.cpu()

    tpu7x_mesh = jax.sharding.AbstractMesh(
        (), (), abstract_device=jax.sharding.AbstractDevice("TPU7x", 1, "tpu")
    )
    with (
        monkeypatch.context() as mp,
        jax.default_device(jax.devices("cpu")[0]),
        jax.sharding.use_abstract_mesh(tpu7x_mesh),
    ):
        for cls in (DeepseekCompressor, DeepseekV41Indexer):
            mp.setattr(cls, "kv_rope_op", property(lambda _s: _cpu_kv_rope))
        mp.setattr(
            dsv41_ops.cache_op,
            "dsv41_write_sub_rows",
            lambda c, u, s, off=0, after=None, page_tokens=16: _to_torch(
                write_sub_rows(
                    _to_jax(c),
                    _to_jax(u),
                    _to_jax(s),
                    slot_offset=int(off),
                    after=_to_jax(after),
                    page_tokens=int(page_tokens),
                    interpret=True,
                )
            ),
        )
        mp.setattr(
            dsv41_ops.cache_op,
            "dsv41_pack_fp8_kv_rows",
            lambda lat: _to_torch(pack_fp8_kv_rows(_to_jax(lat), interpret=True)),
        )
        mp.setattr(
            dsv41_ops.indexer_op,
            "dsv41_paged_indexer_topk",
            lambda q, w, kl, bt, sl, cu, tk, kh=None, ci=None, cr=1: _to_torch(
                paged_indexer_topk(
                    _to_jax(q),
                    _to_jax(w),
                    _to_jax(kl),
                    _to_jax(bt),
                    _to_jax(sl),
                    _to_jax(cu),
                    topk=int(tk),
                    k_pool_hi=_to_jax(kh),
                    candidate_indices=_to_jax(ci),
                    compress_ratio=int(cr),
                    interpret=True,
                )
            ),
        )
        lat_cpu, topk_cpu, cand_cpu = _run_on_device(torch.device("cpu"))

    lat_tpu, topk_tpu, cand_tpu = _run_on_device(torch.device("tpu:0"))
    assert torch.allclose(lat_tpu, lat_cpu, rtol=2e-2, atol=2e-2)
    assert torch.equal(cand_tpu, cand_cpu)
    for row_tpu, row_cpu in zip(topk_tpu.tolist(), topk_cpu.tolist(), strict=True):
        valid_tpu = sorted(x for x in row_tpu if x >= 0)
        valid_cpu = sorted(x for x in row_cpu if x >= 0)
        assert valid_tpu == valid_cpu
