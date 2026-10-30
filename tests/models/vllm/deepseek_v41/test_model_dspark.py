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
"""Unit and TPU7x parity tests for DeepSeek-V4.1 model assembly and DSpark."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import torch.nn as nn
import vllm.forward_context as fc_mod
from vllm.config import VllmConfig
from vllm.model_executor.kernels.mhc.torch import mhc_post_torch
from vllm.v1.core import kv_cache_utils

import vllm_torchtpu as torchtpu_root
from vllm_torchtpu.models.vllm.deepseek_v41 import (
    DeepseekV41DecoderLayer,
    DeepseekV41ForCausalLM,
    DeepseekV41Model,
    DSparkDeepseekV41ForCausalLM,
    DSparkDeepseekV41Model,
)
from vllm_torchtpu.models.vllm.deepseek_v41 import (
    dspark as dspark_mod,
)
from vllm_torchtpu.models.vllm.deepseek_v41 import (
    model as model_mod,
)
from vllm_torchtpu.models.vllm.deepseek_v41.cache import (
    NativeCacheView,
    PagedCacheRows,
)
from vllm_torchtpu.models.vllm.deepseek_v41.ced import DecoderReplayPlan
from vllm_torchtpu.models.vllm.deepseek_v41.model import (
    DeepseekV41DecoderStage,
    DeepseekV41EncoderStage,
    collapse,
)
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)
from vllm_torchtpu.utils import RpaSmemPageTableModel


class _DummyCache(NativeCacheView):
    def __init__(self, prefix: str) -> None:
        super().__init__()
        self.prefix = prefix
        self.kv_cache = torch.tensor([])


class _DummyAttention(nn.Module):
    def __init__(
        self,
        vllm_config: Any,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        indexer_k_cache: Any = None,
        candidate_block_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        del topk_indices_buffer, indexer_k_cache, candidate_block_buffer
        cfg = vllm_config.model_config.hf_config
        self.hidden_size = cfg.hidden_size
        self.window_size = cfg.sliding_window
        self.swa_cache_layer = _DummyCache(f"{prefix}.swa")
        self.compressor_state = _DummyCache(f"{prefix}.comp")
        self.indexer = SimpleNamespace(k_cache=_DummyCache(f"{prefix}.idx"))
        self.wo_b = SimpleNamespace(reduce_results=True)
        self.non_causal_multi_token_decode = False
        self.dspark_block_size = 0
        self.query_block_size = 1
        self.ced_global_from_encoder = False
        self.ced_decoder = False
        self.global_kv_writes = 0

    def project_context_kv(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones((x.shape[0], 512), dtype=x.dtype, device=x.device)

    def rotate(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        del positions
        return x

    def write_encoder_global_kv(self, x: torch.Tensor, positions: torch.Tensor) -> None:
        del x, positions
        self.global_kv_writes += 1

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor, kv_cache: Any = None
    ) -> torch.Tensor:
        del positions, kv_cache
        return hidden_states


class _DummyMoE(nn.Module):
    def __init__(self, vllm_config: Any, prefix: str = "") -> None:
        super().__init__()
        del prefix
        cfg = vllm_config.model_config.hf_config
        self.hidden_size = cfg.hidden_size
        self.use_native_mega_moe = False
        self.experts = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        del input_ids
        return hidden_states


class _DummyEngram(nn.Module):
    def __init__(
        self,
        cfg: Any,
        quant_config: Any,
        layout: Any,
        layer_hash_index: int,
        use_sequence_parallel: bool,
        prefix: str,
    ) -> None:
        super().__init__()
        del cfg, quant_config, layout, prefix
        assert use_sequence_parallel is False
        self.layer_hash_index = layer_hash_index
        self.prepared: list[torch.Tensor] = []
        self._staged_raw: torch.Tensor | None = None
        self._active_staged: torch.Tensor | None = None
        self._active_raw: torch.Tensor | None = None

    def is_staged(self) -> bool:
        return self._staged_raw is not None

    def staged_device_tensors(self) -> tuple[torch.Tensor, ...]:
        return tuple(
            t
            for t in (self._active_staged, self._active_raw)
            if t is not None and t.numel() > 0
        )

    def prepare_embeddings(self, hashes: torch.Tensor) -> None:
        self.prepared.append(hashes)

    def forward(
        self, residual: torch.Tensor, hashes: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        del hashes, mask
        return residual + 0.5


class _DummyHashState(nn.Module):
    def __init__(self, vllm_config: Any, layout: Any, swa_cache_module: Any) -> None:
        super().__init__()
        del vllm_config, layout
        self.swa_cache_module = swa_cache_module
        self.lookback_depth = 2
        self.cache_ok = True

    def ensure_cache(self) -> bool:
        return self.cache_ok

    def dummy_hashes(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dev = ids.device if isinstance(ids.device, torch.device) else "cpu"
        return (
            torch.full((ids.shape[0], 2, 2), -1, dtype=torch.int64, device=dev),
            torch.zeros(ids.shape[0], dtype=torch.bool, device=dev),
        )

    def forward(self, ids: torch.Tensor, *args: Any) -> torch.Tensor:
        del args
        dev = ids.device if isinstance(ids.device, torch.device) else "cpu"
        return torch.zeros((ids.shape[0], 2, 2), dtype=torch.int64, device=dev)


def _zero_hc_params(layers: nn.ModuleList) -> None:
    for raw_lyr in layers:
        lyr = cast(Any, raw_lyr)
        lyr.hc_attn_fn.data.zero_()
        lyr.hc_ffn_fn.data.zero_()
        lyr.hc_attn_base.data.zero_()
        lyr.hc_ffn_base.data.zero_()
        lyr.hc_attn_scale.data.zero_()
        lyr.hc_ffn_scale.data.zero_()


def _patch_submodules(monkeypatch: pytest.MonkeyPatch) -> None:
    import vllm.config.vllm as vllm_cfg_mod
    import vllm.distributed.parallel_state as ps_mod

    dummy_grp = SimpleNamespace(
        world_size=1,
        rank_in_group=0,
        is_first_rank=True,
        is_last_rank=True,
        cpu_group=None,
    )
    for g in ("_TP", "_PP", "_DP", "_EP"):
        monkeypatch.setattr(ps_mod, g, dummy_grp)
    monkeypatch.setattr(vllm_cfg_mod, "_current_vllm_config", _make_vllm_config())
    monkeypatch.setattr(model_mod, "get_mhc_post_op", lambda: mhc_post_torch)
    monkeypatch.setattr(model_mod, "DeepseekV41Attention", _DummyAttention)
    monkeypatch.setattr(model_mod, "DeepseekV41MoE", _DummyMoE)
    monkeypatch.setattr(model_mod, "Engram", _DummyEngram)
    monkeypatch.setattr(model_mod, "NgramHashState", _DummyHashState)
    monkeypatch.setattr(
        model_mod.upstream, "can_share_engram_tables", lambda layout: True
    )
    monkeypatch.setattr(
        model_mod.upstream,
        "gather_engram_hashes",
        lambda h, *, dp_shared_memory=False: h,
    )
    monkeypatch.setattr(
        model_mod,
        "VocabParallelEmbedding",
        lambda num_embeddings, embedding_dim, **kw: nn.Embedding(
            num_embeddings, embedding_dim
        ),
    )
    monkeypatch.setattr(
        model_mod.upstream,
        "ParallelLMHead",
        lambda num_embeddings, embedding_dim, **kw: nn.Linear(
            embedding_dim, num_embeddings, bias=False
        ),
    )
    monkeypatch.setattr(
        model_mod.upstream,
        "LogitsProcessor",
        lambda vocab_size, **kw: lambda head, hidden, **lkw: head(hidden),
    )

    class _StubReplicatedLinear(nn.Linear):
        def __init__(
            self,
            in_f: int,
            out_f: int,
            bias: bool = True,
            *,
            return_bias: bool = True,
            **_kw: Any,
        ) -> None:
            super().__init__(in_f, out_f, bias=bias)
            self.return_bias = return_bias

        def forward(self, input: torch.Tensor) -> Any:
            out = super().forward(input)
            return (out, self.bias) if self.return_bias else out

    assert isinstance(_StubReplicatedLinear(4, 2)(torch.ones(1, 4)), tuple)
    monkeypatch.setattr(dspark_mod, "ReplicatedLinear", _StubReplicatedLinear)
    monkeypatch.setattr(
        dspark_mod,
        "DSparkMarkovHead",
        lambda vocab, draft_vocab, rank, **kw: SimpleNamespace(
            markov_w1=nn.Linear(vocab, rank, bias=False),
            markov_w2=nn.Linear(rank, draft_vocab, bias=False),
            embed=lambda ids: torch.ones((ids.shape[0], rank), dtype=torch.float32),
            bias=lambda emb, lp: torch.zeros(
                (emb.shape[0], draft_vocab), dtype=torch.float32
            ),
        ),
    )
    monkeypatch.setattr(
        dspark_mod,
        "DSparkConfidenceHead",
        lambda in_dim, prefix="", **kw: lambda h, m: torch.zeros((h.shape[0], 1)),
    )
    monkeypatch.setattr(
        dspark_mod,
        "LogitsProcessor",
        lambda vocab_size, **kw: lambda head, hidden, **lkw: head(hidden),
    )


def _make_vllm_config(
    *,
    enable_prefix_caching: bool = False,
    swa_bounded_replay: bool = False,
    dp_size: int = 1,
    pp_size: int = 1,
    enable_ep: bool = False,
    spec_method: str | None = None,
    num_spec: int = 2,
    cand_src: int = 0,
    with_engram: bool = True,
) -> Any:
    hf_config = SimpleNamespace(
        hidden_size=16,
        vocab_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        n_group=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=None,
        dspark_n_routed_experts=4,
        dspark_num_experts_per_tok=2,
        dspark_target_layer_ids=[1, 2],
        dspark_markov_rank=4,
        dspark_block_size=2,
        num_nextn_predict_layers=2,
        hc_mult=2,
        hc_eps=1e-4,
        hc_sinkhorn_iters=2,
        rms_norm_eps=1e-6,
        sliding_window=4,
        index_topk=4,
        candidate_source_layer_id=cand_src,
        candidate_topk_blocks=2,
        kv_source_layer_ids=[],
        expert_dtype="fp8",
        engram_layer_ids=[0] if with_engram else [],
    )
    layout = SimpleNamespace(layer_ids=(0,)) if with_engram else None
    hf_config._engram_layout = layout
    spec_cfg = (
        None
        if spec_method is None
        else SimpleNamespace(
            method=spec_method,
            num_speculative_tokens=num_spec,
            draft_model_config=SimpleNamespace(hf_config=hf_config),
            use_dspark=lambda: spec_method == "dspark",
            use_dflash=lambda: False,
            use_eagle=lambda: False,
            uses_draft_model=lambda: True,
        )
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config, dtype=torch.float32),
        quant_config=None,
        cache_config=SimpleNamespace(
            block_size=4,
            cache_dtype="auto",
            enable_prefix_caching=enable_prefix_caching,
            swa_bounded_replay=swa_bounded_replay,
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pp_size,
            tensor_parallel_size=1,
            data_parallel_size=dp_size,
            enable_expert_parallel=enable_ep,
            prefill_context_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=8,
            max_num_seqs=4,
            disable_hybrid_kv_cache_manager=False,
        ),
        attention_config=SimpleNamespace(
            indexer_sparse_logits=False,
            hisparse_config=None,
        ),
        compilation_config=SimpleNamespace(
            mode=0,
            compile_sizes=[2, 4, "bad"],
            static_forward_context={},
            custom_ops=["none"],
            enabled_custom_ops=Counter(),
            disabled_custom_ops=Counter(),
        ),
        engram_config=SimpleNamespace(dp_shared_memory=True) if with_engram else None,
        speculative_config=spec_cfg,
        num_speculative_tokens=0 if spec_method is None else num_spec,
        num_lookahead_tokens=0 if spec_method is None else num_spec,
    )


@contextmanager
def _forward_ctx(monkeypatch: pytest.MonkeyPatch, meta: Any):
    ctx = SimpleNamespace(
        attn_metadata=meta,
        dp_metadata=None,
        batch_descriptor=None,
        additional_kwargs={},
    )
    with (
        monkeypatch.context() as mp,
        fc_mod.override_forward_context(cast(Any, ctx)),
    ):
        mp.setattr(
            model_mod.upstream,
            "is_forward_context_available",
            fc_mod.is_forward_context_available,
        )
        yield ctx


def test_ced_config_and_collapse() -> None:
    """Verify _ced_config, collapse, and _mean_hc_streams."""
    vcfg = _make_vllm_config(enable_prefix_caching=True, swa_bounded_replay=False)
    assert model_mod._ced_config(vcfg) == (False, 1)
    vcfg_ok = _make_vllm_config(enable_prefix_caching=True, swa_bounded_replay=True)
    assert model_mod._ced_config(vcfg_ok) == (True, 1)
    vcfg_kv = _make_vllm_config(enable_prefix_caching=True, swa_bounded_replay=True)
    vcfg_kv.model_config.hf_config.num_hidden_layers = 5
    vcfg_kv.model_config.hf_config.kv_source_layer_ids = [1, 3]
    assert model_mod._ced_config(vcfg_kv) == (True, 3)
    vcfg_kv.model_config.hf_config.kv_source_layer_ids = [5]
    assert model_mod._ced_config(vcfg_kv) == (False, 5)

    h = torch.ones((3, 2, 4), dtype=torch.float32)
    pre = torch.tensor([[0.25, 0.75], [0.5, 0.5], [1.0, 0.0]], dtype=torch.float32)
    assert collapse(h, pre).shape == (3, 4)
    assert model_mod._mean_hc_streams(h).shape == (3, 4)

    h4 = torch.randn((3, 4, 8), dtype=torch.bfloat16)
    pre4 = torch.softmax(torch.randn((3, 4), dtype=torch.float32), dim=-1)
    ref_c4 = (h4.float() * pre4.unsqueeze(-1)).sum(dim=1).to(torch.bfloat16)
    ref_m4 = h4.float().mean(dim=1).to(torch.bfloat16)
    assert torch.allclose(collapse(h4, pre4), ref_c4, atol=1e-2, rtol=1e-2)
    assert torch.allclose(model_mod._mean_hc_streams(h4), ref_m4, atol=1e-2, rtol=1e-2)


def test_mhc_post_cpu_and_meta(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify mhc_post invokes get_mhc_post_op unconditionally."""
    with set_vllm_model_wrapper_context(mesh=cast(Any, None)):
        xm = torch.empty((3, 4), dtype=torch.bfloat16, device="meta")
        hm = torch.empty((3, 2, 4), dtype=torch.bfloat16, device="meta")
        pm_m = torch.empty((3, 2, 1), dtype=torch.float32, device="meta")
        cm_m = torch.empty((3, 2, 2), dtype=torch.float32, device="meta")
        out_m = model_mod.mhc_post(xm, hm, pm_m, cm_m)
        assert out_m.shape == (3, 2, 4) and out_m.device.type == "meta"

    monkeypatch.setattr(model_mod, "get_mhc_post_op", lambda: mhc_post_torch)
    h = torch.ones((3, 2, 4), dtype=torch.float32)
    x = torch.zeros((3, 4), dtype=torch.float32)
    pm = torch.ones((3, 2, 1), dtype=torch.float32)
    cm = torch.eye(2, dtype=torch.float32).unsqueeze(0).expand(3, 2, 2).contiguous()
    assert model_mod.mhc_post(x, h, pm, cm).shape == (3, 2, 4)


def test_decoder_layer_and_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify DeepseekV41DecoderLayer forward branches and stage wrappers."""
    _patch_submodules(monkeypatch)
    vcfg = _make_vllm_config()
    layout = cast(Any, SimpleNamespace(layer_ids=(0,)))
    l0 = DeepseekV41DecoderLayer(vcfg, prefix="model.layers.0", engram_layout=layout)
    l1 = DeepseekV41DecoderLayer(vcfg, prefix="model.layers.1", engram_layout=layout)
    _zero_hc_params(nn.ModuleList([l0, l1]))
    l0.hc_attn_fn_broadcast = torch.zeros((8, 16), dtype=torch.float32)

    pos = torch.arange(3, dtype=torch.int32)
    ids = torch.tensor([1, 2, 3], dtype=torch.int32)
    x2d = torch.ones((3, 16), dtype=torch.float32)
    h, res, pm, rm, f_pre, aux0 = l0(x2d, pos, ids)
    assert h.shape == (3, 16) and res.shape == (3, 2, 16) and aux0 is None

    x3d = torch.ones((3, 2, 16), dtype=torch.float32)
    h3, res3, pm3, rm3, f_pre3, _ = l1(x3d, pos, ids, pre_mix=f_pre)
    assert h3.shape == (3, 16)

    ehash = torch.zeros((3, 1, 2), dtype=torch.int64)
    emask = torch.ones(3, dtype=torch.bool)
    _, _, _, _, _, aux_eng = l0(
        h, pos, ids, f_pre, pm, rm, res, ehash, emask, capture_previous_aux=True
    )
    assert aux_eng is not None and aux_eng.shape == (3, 16)

    _, _, _, _, _, aux_no_eng = l1(
        h3, pos, ids, f_pre3, pm3, rm3, res3, capture_previous_aux=False
    )
    assert aux_no_eng is None

    owner = cast(Any, nn.Module())
    st_no_tag = DeepseekV41EncoderStage(owner, "p.full", "full")
    assert st_no_tag.prefix == "p.full"
    st_dec = DeepseekV41DecoderStage(owner, "p.dec", "decoder", "v41_decoder")
    assert st_dec._compile_prefix == "v41_decoder"

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        l_bf16 = DeepseekV41DecoderLayer(vcfg, prefix="model.layers.0")
    finally:
        torch.set_default_dtype(prev_dtype)
    assert l_bf16.hc_attn_fn.dtype == torch.float32
    assert l_bf16.hc_ffn_fn.dtype == torch.float32
    assert l_bf16.hc_attn_base.dtype == torch.float32
    assert l_bf16.hc_ffn_base.dtype == torch.float32
    assert l_bf16.hc_attn_scale.dtype == torch.float32
    assert l_bf16.hc_ffn_scale.dtype == torch.float32
    _zero_hc_params(nn.ModuleList([l_bf16]))
    res_bf16 = torch.ones((3, 2, 16), dtype=torch.bfloat16)
    fn_bf16 = torch.zeros((8, 32), dtype=torch.bfloat16)
    l_bf16.hc_attn_scale = nn.Parameter(
        torch.zeros(3, dtype=torch.bfloat16), requires_grad=False
    )
    l_bf16.hc_attn_base = nn.Parameter(
        torch.zeros(8, dtype=torch.bfloat16), requires_grad=False
    )
    pm_out, rm_out, normed_out, d_pre = l_bf16._mhc_pre(res_bf16, fn_bf16, False)
    assert pm_out.dtype == torch.float32 and rm_out.dtype == torch.float32
    assert normed_out.shape == (3, 16) and d_pre.dtype == torch.float32


def test_deepseek_v41_model_full_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify DeepseekV41Model full-stage execution with and without aux states."""
    _patch_submodules(monkeypatch)
    monkeypatch.setattr(
        model_mod.upstream.EngramLayout,
        "from_config",
        classmethod(lambda cls, cfg: cfg._engram_layout),
    )
    vcfg_full = _make_vllm_config(
        enable_prefix_caching=True,
        swa_bounded_replay=False,
        cand_src=-1,
        with_engram=False,
    )
    m_full = DeepseekV41Model(vllm_config=vcfg_full, prefix="model")
    assert callable(m_full.mhc_post_op)
    _zero_hc_params(m_full.layers)
    m_full.finalize_mhc_broadcast_weights()

    ids = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    pos = torch.arange(4, dtype=torch.int32)
    monkeypatch.setattr(
        model_mod.upstream, "is_forward_context_available", lambda: False
    )
    out_full = m_full(ids, pos)
    assert isinstance(out_full, torch.Tensor) and out_full.shape == (4, 16)

    m_full.aux_hidden_state_layers = (1, 2)
    out_full_aux, aux_list = cast(
        tuple[torch.Tensor, list[torch.Tensor]],
        m_full(ids, pos, inputs_embeds=torch.ones((4, 16), dtype=torch.float32)),
    )
    assert out_full_aux.shape == (4, 16) and len(aux_list) == 2

    seen_full_replay: list[torch.Tensor] = []
    orig_full = m_full.full_stage.forward

    def _spy_full(*a: Any, **kw: Any) -> Any:
        c_now = model_mod.upstream.get_forward_context()
        seen_full_replay.append(
            c_now.additional_kwargs["deepseek_v41_replay_starts"].clone()
        )
        return orig_full(*a, **kw)

    monkeypatch.setattr(m_full.full_stage, "forward", _spy_full)
    f_meta = SimpleNamespace(
        seq_lens=torch.tensor([4], dtype=torch.int32),
        replay_starts_cpu=(2,),
    )
    with _forward_ctx(monkeypatch, {"l0": f_meta}):
        out_ctx, _ = cast(tuple[torch.Tensor, list[torch.Tensor]], m_full(ids, pos))
    assert out_ctx.shape == (4, 16)
    assert torch.equal(seen_full_replay[-1], torch.tensor([2], dtype=torch.int32))
    with pytest.raises(AssertionError, match="Invalid CED stage"):
        m_full._forward_stage(ids, pos, stage="invalid_stage")


def test_deepseek_v41_model_ced_and_engram(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify DeepseekV41Model CED identity, narrowed replay, zero-token, and engram."""
    _patch_submodules(monkeypatch)
    monkeypatch.setattr(
        model_mod.upstream.EngramLayout,
        "from_config",
        classmethod(lambda cls, cfg: cfg._engram_layout),
    )
    vcfg_ced = _make_vllm_config(dp_size=2, enable_ep=True)
    vcfg_ced.model_config.hf_config.num_hidden_layers = 4
    m_ced = DeepseekV41Model(vllm_config=vcfg_ced, prefix="model")
    ced_layers = [cast(Any, lyr) for lyr in m_ced.layers]
    assert ced_layers[2].attn.ced_global_from_encoder is True
    assert [lyr.attn.ced_decoder for lyr in ced_layers] == [
        False,
        False,
        True,
        True,
    ]
    assert m_ced.engram_dp_shared_memory is True
    _zero_hc_params(m_ced.layers)
    m_ced.finalize_mhc_broadcast_weights()
    monkeypatch.setattr(
        model_mod, "get_dp_group", lambda: SimpleNamespace(cpu_group=None)
    )

    ids = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    pos = torch.arange(4, dtype=torch.int32)
    monkeypatch.setattr(
        model_mod.upstream, "is_forward_context_available", lambda: False
    )
    raw = torch.zeros((8, 128), dtype=torch.int32)
    pcr = PagedCacheRows(raw, ((2, 4, 256), torch.bfloat16, 2048, 0))
    ced_layers[0].attn.swa_cache_layer._tpu_paged_cache_rows = pcr
    swa_meta = SimpleNamespace(
        seq_lens=torch.tensor([4], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        block_tables=torch.tensor([[0, 1]], dtype=torch.int32),
        slot_mapping=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        query_start_loc_cpu=None,
        seq_lens_cpu=None,
        prompt_lens_cpu=None,
        replay_starts_cpu=None,
    )
    assert m_ced._prepare_engram(None, pos, None, "full") == (None, None)
    dh0, dm0 = m_ced._prepare_engram(ids, pos, None, "full")
    assert dh0 is not None and dm0 is not None and not dm0.any()

    with _forward_ctx(monkeypatch, "not_a_dict"):
        dh1, dm1 = m_ced._prepare_engram(ids, pos, None, "encoder")
        assert dh1 is not None and dm1 is not None and not dm1.any()
        assert cast(torch.Tensor, m_ced(ids, pos)).shape == (4, 16)

    engram_hash = cast(Any, m_ced.engram_hash)
    assert engram_hash is not None
    engram_hash.cache_ok = False
    with _forward_ctx(monkeypatch, {m_ced.engram_swa_prefix: swa_meta}):
        dh2, dm2 = m_ced._prepare_engram(ids, pos, None, "encoder")
        assert dh2 is not None and dm2 is not None and not dm2.any()
    engram_hash.cache_ok = True

    monkeypatch.setattr(
        model_mod.upstream, "can_share_engram_tables", lambda layout: False
    )
    vcfg_noshm = _make_vllm_config(dp_size=2, enable_ep=True)
    m_noshm = DeepseekV41Model(vllm_config=vcfg_noshm, prefix="model")
    assert m_noshm.engram_dp_shared_memory is False
    with _forward_ctx(monkeypatch, "not_a_dict") as ctx_ns:
        dh_ns, _ = m_noshm._prepare_engram(ids, pos, None, "full")
        assert dh_ns is not None
        ctx_ns.dp_metadata = SimpleNamespace()
        dh_ns2, _ = m_noshm._prepare_engram(ids, pos, None, "full")
        assert dh_ns2 is not None

    with _forward_ctx(monkeypatch, {m_ced.engram_swa_prefix: swa_meta}):
        hashes, mask = m_ced._prepare_engram(ids, pos, None, "encoder")
        assert hashes is not None and mask is not None
        lb = torch.ones((1, 2), dtype=torch.int32)
        hashes2, _ = m_ced._prepare_engram(ids, pos, lb, "full")
        assert hashes2 is not None

        seen_enc_replay: list[torch.Tensor] = []
        seen_replay_starts: list[torch.Tensor] = []
        seen_dec_pad_dp: list[
            tuple[list[bool] | None, list[int] | None, int | None]
        ] = []
        orig_enc_forward = m_ced.encoder_stage.forward
        orig_dec_forward = m_ced.decoder_stage.forward

        def _spy_enc(*a: Any, **kw: Any) -> Any:
            ctx_now = model_mod.upstream.get_forward_context()
            seen_enc_replay.append(
                ctx_now.additional_kwargs["deepseek_v41_replay_starts"].clone()
            )
            return orig_enc_forward(*a, **kw)

        seen_dec_win_starts: list[torch.Tensor] = []
        seen_dec_cand: list[torch.Tensor] = []

        def _spy_dec(*a: Any, **kw: Any) -> Any:
            ctx_now = model_mod.upstream.get_forward_context()
            seen_dec_win_starts.append(
                ctx_now.additional_kwargs["deepseek_v41_window_starts"].clone()
            )
            seen_replay_starts.append(
                ctx_now.additional_kwargs["deepseek_v41_replay_starts"].clone()
            )
            if m_ced.candidate_block_buffer is not None:
                seen_dec_cand.append(m_ced.candidate_block_buffer[:4].clone())
            ps_now = model_mod.token_padding._padding_state
            pad_now = (
                None
                if ps_now is None
                else ps_now.get_local_padding_mask(a[1].shape[0]).tolist()
            )
            dp_now = (
                None
                if ctx_now.dp_metadata is None
                else ctx_now.dp_metadata.num_tokens_across_dp_cpu.tolist()
            )
            bd_now = (
                None
                if ctx_now.batch_descriptor is None
                else ctx_now.batch_descriptor.num_tokens
            )
            seen_dec_pad_dp.append((pad_now, dp_now, bd_now))
            return orig_dec_forward(*a, **kw)

        monkeypatch.setattr(m_ced.encoder_stage, "forward", _spy_enc)
        monkeypatch.setattr(m_ced.decoder_stage, "forward", _spy_dec)
        assert cast(torch.Tensor, m_ced(ids, pos)).shape == (4, 16)
        assert torch.equal(seen_replay_starts[-1], torch.tensor([0], dtype=torch.int32))

        monkeypatch.setattr(
            model_mod.ced_mod,
            "plan_decoder_replay",
            lambda *a: DecoderReplayPlan((0, 1, 2, 3), (0, 4), (0,), 4, True),
        )
        assert cast(torch.Tensor, m_ced(ids, pos)).shape == (4, 16)
        assert torch.equal(seen_replay_starts[-1], torch.tensor([0], dtype=torch.int32))

        monkeypatch.setattr(
            model_mod.ced_mod,
            "plan_decoder_replay",
            lambda *a: DecoderReplayPlan((0, 1, 2, 3), (0, 4), (2,), 4, True),
        )
        assert cast(torch.Tensor, m_ced(ids, pos)).shape == (4, 16)
        assert torch.equal(seen_replay_starts[-1], torch.tensor([2], dtype=torch.int32))

        monkeypatch.setattr(
            model_mod.ced_mod,
            "plan_decoder_replay",
            lambda *a: DecoderReplayPlan((), (0, 0), (0,), 0),
        )
        assert cast(torch.Tensor, m_ced(ids, pos)).shape == (4, 16)
        m_ced.aux_hidden_state_layers = (1,)
        z_out, z_aux = cast(tuple[torch.Tensor, list[torch.Tensor]], m_ced(ids, pos))
        assert z_out.shape == (4, 16) and len(z_aux) == 1

        monkeypatch.setattr(
            model_mod.ced_mod,
            "plan_decoder_replay",
            lambda *a: DecoderReplayPlan((1, 3), (0, 2), (2,), 4),
        )
        m_ced.aux_hidden_state_layers = (1, 2, 4)
        rep_out, rep_aux = cast(
            tuple[torch.Tensor, list[torch.Tensor]], m_ced(ids, pos)
        )
        assert rep_out.shape == (4, 16) and len(rep_aux) == 3
        assert torch.equal(seen_replay_starts[-1], torch.tensor([2], dtype=torch.int32))

        m_ced._ced_dp_sync = False
        m_ced.config.candidate_source_layer_id = 3
        rep_out_no_ids, _ = cast(
            tuple[torch.Tensor, list[torch.Tensor]],
            m_ced(None, pos, inputs_embeds=torch.ones((4, 16))),
        )
        assert rep_out_no_ids.shape == (4, 16)

        # Verify distinct encoder/decoder replay_starts and narrowed token_padding.
        swa_meta.replay_starts_cpu = (4,)
        m_ced.config.candidate_source_layer_id = 1
        assert m_ced.candidate_block_buffer is not None
        m_ced.candidate_block_buffer[:4] = torch.tensor(
            [[10, 11], [20, 21], [30, 31], [40, 41]], dtype=torch.int32
        )
        monkeypatch.setattr(
            model_mod.ced_mod,
            "plan_decoder_replay",
            lambda *a: DecoderReplayPlan((1, 3), (0, 2), (8,), 3),
        )
        pad_state = model_mod.token_padding.TokenPaddingState.create(8, "cpu")
        model_mod.token_padding.set_padding_state(pad_state)
        try:
            model_mod.token_padding.mark_padding(3, 4)
            ctx_cur = model_mod.upstream.get_forward_context()
            ctx_cur.dp_metadata = model_mod.DPMetadata(
                torch.tensor([4, 4], dtype=torch.int32)
            )
            ctx_cur.batch_descriptor = fc_mod.BatchDescriptor(num_tokens=4)
            narrow_out, _ = cast(
                tuple[torch.Tensor, list[torch.Tensor]], m_ced(ids, pos)
            )
            assert narrow_out.shape == (4, 16)
            assert torch.equal(
                seen_enc_replay[-1], torch.tensor([4], dtype=torch.int32)
            )
            assert torch.equal(
                seen_dec_win_starts[-1], torch.tensor([4], dtype=torch.int32)
            )
            assert torch.equal(
                seen_replay_starts[-1], torch.tensor([8], dtype=torch.int32)
            )
            assert seen_dec_cand[-1][:3].tolist() == [[20, 21], [40, 41], [40, 41]]
            assert seen_dec_pad_dp[-1] == ([False, False, True], [3, 3], 3)
            assert pad_state.local_padding_mask[:4].tolist() == [
                False,
                False,
                False,
                True,
            ]
        finally:
            model_mod.token_padding.set_padding_state(None)

        # Verify TPU synchronize_tensors(wait=True) for staged Engram root + view.
        synced: list[tuple[int, bool]] = []
        monkeypatch.setattr(
            model_mod,
            "synchronize_tensors",
            lambda ts, wait=True: synced.append((len(ts), wait)),
        )
        eng0 = cast(Any, ced_layers[0].engram)
        assert eng0 is not None
        eng0._staged_raw = torch.ones((8,), dtype=torch.int32)
        eng0._active_staged = torch.ones((8,), dtype=torch.int32)
        eng0._active_raw = torch.ones((4, 2), dtype=torch.int32)
        tpu_ids = SimpleNamespace(
            shape=(4,),
            device=SimpleNamespace(type="tpu"),
            new_full=lambda sh, v: torch.full(sh, v, dtype=torch.int32),
        )
        monkeypatch.setattr(
            model_mod.upstream,
            "image_sentinel_mask",
            lambda t: torch.zeros(t.shape, dtype=torch.bool),
        )
        m_ced._prepare_engram(cast(Any, tpu_ids), pos, None, "encoder")
        assert synced == [(2, True)]

        # Verify eager embed_input_ids + synchronize_tensors on TPU when DP > 1.
        synced.clear()
        seen_enc_embeds: list[Any] = []
        monkeypatch.setattr(
            m_ced,
            "embed_input_ids",
            lambda _ids: torch.full((4, 16), 3.0, dtype=torch.float32),
        )
        monkeypatch.setattr(
            m_ced.encoder_stage,
            "forward",
            lambda *a, **kw: (
                seen_enc_embeds.append(a[3]),
                orig_enc_forward(ids, pos, a[2], a[3], None, None),
            )[1],
        )
        tpu_pos = SimpleNamespace(shape=(4,), device=SimpleNamespace(type="tpu"))
        monkeypatch.setattr(
            model_mod.ced_mod,
            "plan_decoder_replay",
            lambda *a: DecoderReplayPlan((), (0, 0), (0,), 0),
        )
        orig_starts_t = model_mod._starts_tensor
        monkeypatch.setattr(
            model_mod,
            "_starts_tensor",
            lambda s, n, _d: orig_starts_t(s, n, pos.device),
        )
        eng0._staged_raw = None
        eng0._active_staged = None
        eng0._active_raw = None
        engram_hash.cache_ok = False
        m_ced(cast(Any, tpu_ids), cast(Any, tpu_pos))
        assert seen_enc_embeds[-1] is not None and (seen_enc_embeds[-1] == 3.0).all()
        assert synced == [(1, False)]


def test_deepseek_v41_for_causal_lm_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify DeepseekV41ForCausalLM rejects unsupported PP/DP/spec configurations."""
    _patch_submodules(monkeypatch)
    with pytest.raises(NotImplementedError, match="PP=1"):
        DeepseekV41ForCausalLM(vllm_config=_make_vllm_config(pp_size=2))
    with pytest.raises(NotImplementedError, match="DP with EP"):
        DeepseekV41ForCausalLM(
            vllm_config=_make_vllm_config(dp_size=2, enable_ep=False)
        )
    with pytest.raises(NotImplementedError, match="Unsupported spec"):
        DeepseekV41ForCausalLM(vllm_config=_make_vllm_config(spec_method="mtp"))


def test_deepseek_v41_for_causal_lm_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify DeepseekV41ForCausalLM MoE setup, DSpark aux shift, and forward pass."""
    _patch_submodules(monkeypatch)
    monkeypatch.setattr(
        model_mod.upstream.EngramLayout,
        "from_config",
        classmethod(lambda cls, cfg: cfg._engram_layout),
    )
    vcfg = _make_vllm_config(spec_method="dspark", enable_prefix_caching=True)
    vcfg.model_config.hf_config.expert_dtype = "fp4"
    vcfg.model_config.hf_config.weight_block_size = [32, 32]
    lm = DeepseekV41ForCausalLM(vllm_config=vcfg)
    assert isinstance(lm, RpaSmemPageTableModel)
    assert lm.uses_rpa_smem_page_table is False
    _zero_hc_params(lm.model.layers)
    lm.process_weights_after_loading()
    assert lm.num_moe_layers == 2
    assert lm.token_lookback_depth == 2
    assert lm.hf_to_vllm_mapper._map_name("layers.0.ffn.gate.bias_vl") is None
    assert (
        lm.hf_to_vllm_mapper._map_name("layers.0.attn.wo_a.scale")
        == "model.layers.0.attn.wo_a.weight_scale_inv"
    )
    assert (
        lm.hf_to_vllm_mapper._map_name("layers.0.attn.wo_b.weight")
        == "model.layers.0.attn.wo_b.weight"
    )

    lm.set_aux_hidden_state_layers((2, 3))
    assert lm.model.aux_hidden_state_layers == (1, 2)
    vcfg_e3 = _make_vllm_config(spec_method="dspark", enable_prefix_caching=True)
    dhf = vcfg_e3.speculative_config.draft_model_config.hf_config
    dhf.eagle_aux_hidden_state_layer_ids = [1, 2]
    lm_e3 = DeepseekV41ForCausalLM(vllm_config=vcfg_e3)
    lm_e3.set_aux_hidden_state_layers((1, 2))
    assert lm_e3.model.aux_hidden_state_layers == (1, 2)

    vcfg_plain = _make_vllm_config(enable_prefix_caching=True, with_engram=False)
    lm_plain = DeepseekV41ForCausalLM(vllm_config=vcfg_plain)
    _zero_hc_params(lm_plain.model.layers)
    lm_plain.process_weights_after_loading()
    lm_plain.set_aux_hidden_state_layers((1, 2))
    assert lm_plain.model.aux_hidden_state_layers == (1, 2)

    monkeypatch.setattr(
        model_mod.upstream, "is_forward_context_available", lambda: False
    )
    ids = torch.tensor([1, 2], dtype=torch.int32)
    pos = torch.tensor([0, 1], dtype=torch.int32)
    out, aux = cast(tuple[torch.Tensor, list[torch.Tensor]], lm(ids, pos))
    assert out.shape == (2, 16) and len(aux) == 2
    out_p, _ = cast(tuple[torch.Tensor, list[torch.Tensor]], lm_plain(ids, pos))
    assert out_p.shape == (2, 16)

    # Verify 4-layer model with dspark_target_layer_ids=[1, 2, 3]: DFlashProposer
    # passes 1-based (2, 3, 4) to set_aux_hidden_state_layers, which shifts back
    # to (1, 2, 3) so delayed mhc_post captures the layer inputs to layers 1, 2, 3
    # in both full-stage and CED-split (encoder+decoder) modes.
    for ced_on in (False, True):
        vcfg4 = _make_vllm_config(
            spec_method="dspark",
            enable_prefix_caching=True,
            swa_bounded_replay=ced_on,
            with_engram=False,
        )
        vcfg4.model_config.hf_config.num_hidden_layers = 4
        vcfg4.model_config.hf_config.kv_source_layer_ids = [1, 2]
        vcfg4.model_config.hf_config.dspark_target_layer_ids = [1, 2, 3]
        dhf4 = vcfg4.speculative_config.draft_model_config.hf_config
        dhf4.dspark_target_layer_ids = [1, 2, 3]
        torch.manual_seed(1234)
        lm4 = DeepseekV41ForCausalLM(vllm_config=vcfg4)
        assert lm4.model._ced_enabled is ced_on
        assert lm4.model._ced_boundary == 2
        _zero_hc_params(lm4.model.layers)
        for l_i, raw_l in enumerate(lm4.model.layers):
            lyr_i = cast(DeepseekV41DecoderLayer, raw_l)
            lyr_i.hc_attn_base.data.fill_(float(l_i + 1) * 0.1)
            lyr_i.hc_ffn_base.data.fill_(float(l_i + 1) * 0.2)
        lm4.process_weights_after_loading()
        target_ids_0based = (1, 2, 3)
        aux_ids_1based = tuple(idx + 1 for idx in target_ids_0based)
        assert aux_ids_1based == (2, 3, 4)
        lm4.set_aux_hidden_state_layers(aux_ids_1based)
        assert lm4.model.aux_hidden_state_layers == (1, 2, 3)

        manual_layer_inputs: dict[int, torch.Tensor] = {}
        h_cur: torch.Tensor = lm4.model.embed_input_ids(ids)
        res_cur: torch.Tensor | None = None
        pm_cur: torch.Tensor | None = None
        rm_cur: torch.Tensor | None = None
        pre_cur: torch.Tensor | None = None
        for l_i, raw_l in enumerate(lm4.model.layers):
            lyr_i = cast(DeepseekV41DecoderLayer, raw_l)
            if res_cur is not None:
                assert pm_cur is not None and rm_cur is not None
                manual_layer_inputs[l_i] = model_mod._mean_hc_streams(
                    model_mod.mhc_post(h_cur, res_cur, pm_cur, rm_cur)
                )
            h_cur, res_cur, pm_cur, rm_cur, pre_cur, _ = lyr_i(
                h_cur, pos, ids, pre_cur, pm_cur, rm_cur, res_cur
            )
        assert set(manual_layer_inputs) == {1, 2, 3}
        assert all(torch.isfinite(t).all() for t in manual_layer_inputs.values())
        assert not torch.equal(manual_layer_inputs[1], manual_layer_inputs[2])
        assert not torch.equal(manual_layer_inputs[2], manual_layer_inputs[3])

        with _forward_ctx(monkeypatch, "not_a_dict"):
            out4, aux4 = cast(tuple[torch.Tensor, list[torch.Tensor]], lm4(ids, pos))
        assert out4.shape == (2, 16)
        assert len(aux4) == 3
        for captured_t, target_idx in zip(aux4, target_ids_0based, strict=True):
            assert torch.equal(captured_t, manual_layer_inputs[target_idx])


def test_dspark_name_remapping_and_embed_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify _remap_dspark_name and draft forward on dflash-shaped inputs."""
    _patch_submodules(monkeypatch)
    dm = DSparkDeepseekV41ForCausalLM(
        vllm_config=_make_vllm_config(spec_method="dspark", num_spec=2)
    )
    assert dm._remap_dspark_name("layers.0.weight") is None
    assert dm._remap_dspark_name("mtp.0.confidence_head.w") == "model.confidence_head.w"
    assert (
        dm._remap_dspark_name("mtp.0.markov_head.embed.weight")
        == "model.markov_head.markov_w1.weight"
    )
    assert (
        dm._remap_dspark_name("mtp.0.markov_head.head.weight")
        == "model.markov_head.markov_w2.weight"
    )
    assert dm._remap_dspark_name("mtp.0.main_proj.weight") == "model.main_proj.weight"
    assert dm._remap_dspark_name("mtp.1.ffn.gate.bias_vl") is None
    assert (
        dm._remap_dspark_name("mtp.1.attn.wo_b.weight")
        == "model.layers.1.attn.wo_b.weight"
    )
    dm.model.confidence_head = None
    assert dm._remap_dspark_name("mtp.0.confidence_head.w") is None
    assert "_embed_draft_blocks" not in dir(dspark_mod)

    # DSpark draft forward on dflash-shaped [16] input_ids with distinct base,
    # MASK_TOKEN, and trailing padded token IDs must never inject NaNs.
    dm.embed_tokens = nn.Embedding(64, 16)
    dm.model.embed_tokens = dm.embed_tokens
    _zero_hc_params(dm.model.layers)
    dflash_ids = torch.tensor(
        [10, 31, 31, 31, 31, 20, 31, 31, 31, 31, 30, 31, 31, 31, 31, 7],
        dtype=torch.int32,
    )
    dflash_pos = torch.zeros(16, dtype=torch.int32)
    out_dflash = dm(dflash_ids, dflash_pos)
    assert out_dflash.shape == (16, 16)
    assert torch.isfinite(out_dflash).all()


def test_dspark_model_and_kv_insertion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify DSparkDeepseekV41ForCausalLM forward, KV precompute, and loading."""
    _patch_submodules(monkeypatch)
    vcfg_bad = _make_vllm_config(spec_method="dspark", num_spec=8)
    with pytest.raises(ValueError, match="exceeds sliding window"):
        DSparkDeepseekV41Model(vllm_config=vcfg_bad)

    vcfg = _make_vllm_config(spec_method="dspark", num_spec=2)
    dm = DSparkDeepseekV41ForCausalLM(vllm_config=vcfg)
    dm.embed_tokens = nn.Embedding(32, 16)
    dm.model.embed_tokens = dm.embed_tokens
    dm.lm_head = nn.Linear(16, 32, bias=False)
    _zero_hc_params(dm.model.layers)

    assert dm.get_draft_attn_causal() == [False, False]
    assert len(dm.get_draft_kv_cache_layer_names()) == 2
    assert (
        dm.map_draft_to_target(torch.tensor([1, 2], dtype=torch.int64)).dtype
        == torch.int32
    )
    assert callable(dm.model.mhc_post_op)

    pos = torch.arange(4, dtype=torch.int32)
    ids = torch.tensor([1, 9, 2, 9], dtype=torch.int32)
    assert dm(ids, pos).shape == (4, 16)
    dm.process_weights_after_loading()
    assert dm(ids, pos).shape == (4, 16)
    assert dm(ids, pos, inputs_embeds=torch.ones((4, 16))).shape == (4, 16)

    for raw_lyr in dm.model.layers:
        cast(Any, raw_lyr).attn.swa_cache_layer.kv_cache = torch.zeros(
            (4, 4, 512), dtype=torch.float32
        )

    aux_cat = torch.ones((4, 32), dtype=torch.float32)
    assert dm.precompute_and_store_context_kv(aux_cat, pos, None).shape == (4, 16)
    slots_t = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    dm.precompute_and_store_context_kv(torch.ones((4, 16)), pos, [slots_t, slots_t])

    meta_nonempty = cast(
        Any,
        SimpleNamespace(
            seq_lens=torch.tensor([4], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
            block_tables=torch.tensor([[0, 1]], dtype=torch.int32),
        ),
    )
    meta_empty = cast(
        Any,
        SimpleNamespace(
            seq_lens=torch.zeros((0,), dtype=torch.int32),
            query_start_loc=torch.tensor([0], dtype=torch.int32),
            block_tables=torch.tensor([[0, 1]], dtype=torch.int32),
        ),
    )
    dm.precompute_and_store_context_kv(
        torch.ones((4, 16)), pos, [meta_nonempty, meta_empty]
    )

    dm.model.end_layer = 0
    dm.model.finalize_mhc_broadcast_weights()
    dm.model.end_layer = 2

    loaded = dm.load_weights(
        [
            ("layers.0.weight", torch.ones((16, 16))),
            ("mtp.0.main_proj.weight", torch.zeros((16, 32))),
            ("mtp.0.norm.weight", torch.ones((16,))),
        ]
    )
    assert "model.main_proj.weight" in loaded
    assert dm.model.confidence_head is None

    # Verify VllmConfig.num_lookahead_tokens (sample_from_anchor=False -> num_spec + 1)
    # and bfloat16 construction when outer default dtype is float32.
    vcfg_sfa = _make_vllm_config(spec_method="dspark", num_spec=2)
    vcfg_sfa.model_config.dtype = torch.bfloat16
    vcfg_sfa.model_config.hf_config.dspark_block_size = 5
    vcfg_sfa.model_config.hf_config.sample_from_anchor = False
    lookahead_fn = VllmConfig.num_lookahead_tokens.fget
    assert lookahead_fn is not None
    vcfg_sfa.num_lookahead_tokens = lookahead_fn(cast(Any, vcfg_sfa))
    assert vcfg_sfa.num_lookahead_tokens == 3
    assert torch.get_default_dtype() == torch.float32
    dm_sfa = DSparkDeepseekV41ForCausalLM(vllm_config=vcfg_sfa)
    assert torch.get_default_dtype() == torch.float32
    assert dm_sfa.model.query_block_size == 3
    assert cast(Any, dm_sfa.model.layers[0]).attn.dspark_block_size == 3
    assert dm_sfa.model.main_proj.weight.dtype == torch.bfloat16
    assert dm_sfa.model.norm.weight.dtype == torch.bfloat16


def test_cpu_ced_replay_dspark_verify_rollback_and_dirty_pool_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify CED replay, 6-token verify rollback, SWA retain, and 2-bank NaN pool."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    import vllm.config.vllm as vllm_cfg_mod
    import vllm.distributed.parallel_state as ps_mod
    import vllm.v1.attention.backends.mla.sparse_swa as swa_mod
    from vllm.model_executor.kernels.mhc.torch import mhc_post_torch
    from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager

    from tests.models.vllm.deepseek_v41.test_attention import (
        _ACTIVE_VLLM_CONFIG,
        _cpu_kv_rope,
        _patch_vllm_attention_deps,
    )
    from vllm_torchtpu.kernels.experimental.deepseek_v41.cache import (
        gather_cache_rows,
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
    from vllm_torchtpu.models.vllm.deepseek_v4 import moe as v4_moe_mod
    from vllm_torchtpu.models.vllm.deepseek_v41 import (
        attention as attn_mod,
    )
    from vllm_torchtpu.models.vllm.deepseek_v41 import (
        compressor as compressor_mod,
    )
    from vllm_torchtpu.models.vllm.deepseek_v41 import (
        indexer as indexer_mod,
    )
    from vllm_torchtpu.models.vllm.deepseek_v41.attention import (
        DeepseekCompressor,
        DeepseekV41Attention,
        DeepseekV41Indexer,
    )
    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        set_vllm_model_wrapper_context,
    )

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
        o_rot = _cpu_kv_rope(o, pos_i32, cs_inv).float()
        t, g_cnt = o.shape[0], s.shape[0] // 128
        z = torch.einsum(
            "tgd,dgr->tgr", o_rot.view(t, g_cnt, -1), w.float().view(-1, g_cnt, 128)
        )
        return (z.reshape(t, -1) * s.float()).bfloat16()

    class _Gate(nn.Module):
        def __init__(self, input_size: int, output_size: int, **_kw: Any) -> None:
            super().__init__()
            g = torch.Generator().manual_seed(input_size * 97 + output_size)
            self.weight = nn.Parameter(
                (torch.randn((output_size, input_size), generator=g) * 0.05).bfloat16()
            )
            self.e_score_correction_bias = self.tid2eid = None

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
            return (x.float() @ self.weight.float().T), None

    class _Experts(nn.Module):
        def __init__(
            self,
            num_experts: int,
            top_k: int,
            hidden_size: int,
            intermediate_size: int,
            **_kw: Any,
        ) -> None:
            super().__init__()
            self.top_k, self.gate = top_k, None
            self.moe_config = SimpleNamespace(
                dp_size=1,
                pcp_size=1,
                moe_parallel_config=SimpleNamespace(use_ep=False),
            )
            self.routed_experts = SimpleNamespace(
                quant_method=SimpleNamespace(
                    is_monolithic=True, supports_internal_mk=False
                ),
                custom_routing_function=None,
            )
            g = torch.Generator().manual_seed(hidden_size * 31 + num_experts)
            self.w1 = nn.Parameter(
                (
                    torch.randn(
                        (num_experts, intermediate_size, hidden_size), generator=g
                    )
                    * 0.05
                ).bfloat16()
            )
            self.w2 = nn.Parameter(
                (
                    torch.randn(
                        (num_experts, hidden_size, intermediate_size), generator=g
                    )
                    * 0.05
                ).bfloat16()
            )

        def forward(
            self,
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
            input_ids: Any = None,
        ) -> torch.Tensor:
            del input_ids
            probs = torch.softmax(router_logits.float(), dim=-1)
            vals, idx = torch.topk(probs, self.top_k, dim=-1)
            vals = vals / vals.sum(dim=-1, keepdim=True)
            h, w1, w2 = hidden_states.float(), self.w1.float(), self.w2.float()
            out = torch.zeros_like(h)
            for k in range(self.top_k):
                e = idx[:, k]
                g_o = torch.bmm(h.unsqueeze(1), w1[e].transpose(1, 2)).squeeze(1)
                act = torch.nn.functional.silu(g_o)
                proj = torch.bmm(act.unsqueeze(1), w2[e].transpose(1, 2)).squeeze(1)
                out = out + vals[:, k : k + 1] * proj
            return out.to(hidden_states.dtype)

    class _RepLinear(nn.Linear):
        def __init__(
            self,
            in_f: int,
            out_f: int,
            bias: bool = True,
            *,
            return_bias: bool = True,
            **_kw: Any,
        ) -> None:
            super().__init__(in_f, out_f, bias=bias, dtype=torch.bfloat16)
            self.return_bias = return_bias

        def forward(self, x: torch.Tensor) -> Any:
            out = super().forward(x)
            return (out, self.bias) if self.return_bias else out

    dummy_grp = SimpleNamespace(
        world_size=1,
        rank_in_group=0,
        is_first_rank=True,
        is_last_rank=True,
        cpu_group=None,
    )
    for g_name in ("_TP", "_PP", "_DP", "_EP"):
        monkeypatch.setattr(ps_mod, g_name, dummy_grp)
    _patch_vllm_attention_deps(monkeypatch, patch_rope_op=False)
    monkeypatch.setattr(v4_moe_mod, "GateLinear", _Gate)
    monkeypatch.setattr(v4_moe_mod, "FusedMoEFactory", _Experts)
    monkeypatch.setattr(v4_moe_mod, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(v4_moe_mod, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        model_mod,
        "VocabParallelEmbedding",
        lambda n, d, **kw: nn.Embedding(n, d, dtype=torch.bfloat16),
    )
    monkeypatch.setattr(
        model_mod.upstream,
        "ParallelLMHead",
        lambda n, d, **kw: nn.Linear(d, n, bias=False, dtype=torch.bfloat16),
    )
    monkeypatch.setattr(
        model_mod.upstream,
        "LogitsProcessor",
        lambda v, **kw: lambda head, hidden, **lkw: head(hidden),
    )
    monkeypatch.setattr(dspark_mod, "ReplicatedLinear", _RepLinear)
    monkeypatch.setattr(
        dspark_mod,
        "DSparkMarkovHead",
        lambda vocab, draft_vocab, rank, **kw: SimpleNamespace(
            markov_w1=nn.Linear(vocab, rank, bias=False, dtype=torch.bfloat16),
            markov_w2=nn.Linear(rank, draft_vocab, bias=False, dtype=torch.bfloat16),
        ),
    )
    monkeypatch.setattr(
        dspark_mod,
        "DSparkConfidenceHead",
        lambda in_dim, prefix="", **kw: nn.Linear(
            in_dim, 1, bias=False, dtype=torch.bfloat16
        ),
    )
    monkeypatch.setattr(
        dspark_mod,
        "LogitsProcessor",
        lambda v, **kw: lambda head, hidden, **lkw: head(hidden),
    )
    monkeypatch.setattr(
        model_mod.upstream.EngramLayout,
        "from_config",
        classmethod(lambda cls, cfg: None),
    )
    orig_mm = torch.mm
    monkeypatch.setattr(
        torch,
        "mm",
        lambda a, b, *, out_dtype=None, **kw: (
            orig_mm(a.to(out_dtype), b.to(out_dtype), **kw)
            if out_dtype is not None
            else orig_mm(a, b, **kw)
        ),
    )
    monkeypatch.setattr(model_mod, "get_mhc_post_op", lambda: mhc_post_torch)
    for cls in (DeepseekCompressor, DeepseekV41Indexer, DeepseekV41Attention):
        monkeypatch.setattr(cls, "kv_rope_op", property(lambda _s: _cpu_kv_rope))
    monkeypatch.setattr(
        DeepseekV41Attention, "o_proj_op", property(lambda _s: _cpu_o_proj)
    )

    s_ctx: dict[str, Any] = {}
    hf = SimpleNamespace(
        hidden_size=256,
        vocab_size=64,
        num_hidden_layers=4,
        num_attention_heads=16,
        q_lora_rank=16,
        o_lora_rank=128,
        head_dim=512,
        qk_rope_head_dim=64,
        o_groups=2,
        sliding_window=32,
        rms_norm_eps=1e-6,
        max_position_embeddings=256,
        compress_ratios=[0, 2, 1, 1, 0],
        kv_source_layer_ids=[1, 2],
        index_source_layer_ids=[1, 2, 3],
        candidate_source_layer_id=1,
        candidate_topk_blocks=4,
        candidate_block_size=4,
        index_n_heads=8,
        index_head_dim=128,
        index_topk=16,
        rope_theta=10000.0,
        compress_rope_theta=10000.0,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        rope_scaling=None,
        n_group=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=None,
        moe_intermediate_size=64,
        swiglu_limit=None,
        norm_topk_prob=True,
        scoring_func="softmax",
        routed_scaling_factor=1.0,
        num_hash_layers=0,
        dspark_n_routed_experts=4,
        dspark_num_experts_per_tok=2,
        dspark_target_layer_ids=[2, 3],
        dspark_markov_rank=4,
        dspark_block_size=5,
        num_nextn_predict_layers=1,
        hc_mult=4,
        hc_eps=1e-4,
        hc_sinkhorn_iters=2,
        expert_dtype="fp8",
        engram_layer_ids=[],
    )
    vcfg = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=hf, dtype=torch.bfloat16, max_model_len=256
        ),
        quant_config=None,
        cache_config=SimpleNamespace(
            block_size=8,
            cache_dtype="fp8",
            enable_prefix_caching=True,
            swa_bounded_replay=True,
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
            data_parallel_size=1,
            enable_expert_parallel=False,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            pcp_shard_decode_requests=False,
            cp_kv_cache_interleave_size=1,
            enable_eplb=False,
        ),
        kernel_config=SimpleNamespace(moe_backend="auto"),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=64,
            max_num_seqs=4,
            disable_hybrid_kv_cache_manager=False,
        ),
        attention_config=SimpleNamespace(
            indexer_sparse_logits=False,
            sparse_mla_prefill_workspace_factor=5,
            hisparse_config=None,
        ),
        compilation_config=SimpleNamespace(
            mode=0,
            compile_sizes=[16, 64],
            static_forward_context=s_ctx,
            custom_ops=["none"],
            enabled_custom_ops=Counter(),
            disabled_custom_ops=Counter(),
        ),
        engram_config=None,
        speculative_config=SimpleNamespace(
            method="dspark",
            num_speculative_tokens=5,
            draft_model_config=SimpleNamespace(hf_config=hf),
        ),
        num_speculative_tokens=5,
        num_lookahead_tokens=5,
    )

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

        _ACTIVE_VLLM_CONFIG[:] = [vcfg]
        for mod in (
            vllm_cfg_mod,
            attn_mod.upstream,
            indexer_mod.upstream,
            compressor_mod.compressor_upstream,
            swa_mod,
        ):
            monkeypatch.setattr(mod, "get_current_vllm_config", lambda: vcfg)
        with vllm_cfg_mod.set_current_vllm_config(cast(Any, vcfg)):
            torch.manual_seed(2026)
            lm = DeepseekV41ForCausalLM(vllm_config=cast(Any, vcfg))
            lm.set_aux_hidden_state_layers((3, 4))
            lm.process_weights_after_loading()
            torch.manual_seed(2027)
            dspark = DSparkDeepseekV41ForCausalLM(vllm_config=cast(Any, vcfg))
            dspark.embed_tokens = lm.model.embed_tokens
            dspark.model.embed_tokens = lm.model.embed_tokens
            dspark.process_weights_after_loading()

        # Bind ALL target + draft caches to ONE unified 2-bank (pool_lo, pool_hi) via
        # NativeCacheView.bind_kv_cache (null_block=True, 16 blocks/bank = 32 total,
        # stride_b = 4 * 32 KiB = 131072 B = 256 rows/block with layer row offsets).
        stride_b, n_blk_bank, n_blk = 131072, 16, 32
        pool_lo = torch.zeros((n_blk_bank * 256, 128), dtype=torch.int32)
        pool_hi = torch.zeros((n_blk_bank * 256, 128), dtype=torch.int32)
        rope_lo = torch.zeros((n_blk_bank * 64, 128), dtype=torch.int32)
        rope_hi = torch.zeros((n_blk_bank * 64, 128), dtype=torch.int32)
        raw_2b, rope_2b = (pool_lo, pool_hi), (rope_lo, rope_hi)

        for is_draft, m_obj in ((False, lm.model), (True, dspark.model)):
            g_w = torch.Generator().manual_seed(999)
            for i_l, raw_lyr in enumerate(m_obj.layers):
                lyr = cast(DeepseekV41DecoderLayer, raw_lyr)
                for p in (
                    lyr.hc_attn_fn,
                    lyr.hc_ffn_fn,
                    lyr.hc_attn_base,
                    lyr.hc_ffn_base,
                    lyr.hc_attn_scale,
                    lyr.hc_ffn_scale,
                ):
                    p.data.copy_(
                        torch.randn(p.shape, generator=g_w, dtype=torch.float32) * 0.02
                    )
                attn = lyr.attn
                w_fp8 = (
                    torch.randn((256, 4096), generator=g_w, dtype=torch.float32) * 0.05
                ).to(torch.float8_e4m3fn)
                attn.wo_a.weight = nn.Parameter(w_fp8, requires_grad=False)
                attn.wo_a.weight_scale = nn.Parameter(
                    torch.full((256,), 0.05, dtype=torch.float32), requires_grad=False
                )
                attn.process_weights_after_loading(torch.bfloat16)
                swa_off = 32768 if is_draft else i_l * 32768
                attn.swa_cache_layer.bind_kv_cache(
                    torch.zeros((1, 32, 512), dtype=torch.bfloat16).expand(
                        n_blk, -1, -1
                    ),
                    raw_cache=raw_2b,
                    layout=((n_blk, 32, 512), torch.bfloat16, stride_b, swa_off),
                )
                if attn.compressor is not None:
                    mla_off = 0 if i_l == 1 else 65536
                    attn.bind_kv_cache(
                        torch.zeros((1, 64, 512), dtype=torch.uint8).expand(
                            n_blk, -1, -1
                        ),
                        raw_cache=raw_2b,
                        layout=((n_blk, 64, 512), torch.uint8, stride_b, mla_off),
                        rope_pool=rope_2b,
                    )
                    s_ctx[
                        attn.compressor.k_cache_prefix
                    ]._tpu_paged_cache_rows = attn._tpu_paged_cache_rows
                    if attn.compressor.state_cache is not None:
                        attn.compressor.state_cache.bind_kv_cache(
                            torch.zeros((1, 8, 1024), dtype=torch.float32).expand(
                                n_blk, -1, -1
                            ),
                            raw_cache=raw_2b,
                            layout=((n_blk, 8, 1024), torch.float32, stride_b, 65536),
                        )
                if attn.indexer is not None and attn.indexer.owns_k:
                    idx_off = (i_l - 1) * 32768
                    attn.indexer.k_cache.bind_kv_cache(
                        torch.zeros((1, 128, 128), dtype=torch.bfloat16).expand(
                            n_blk, -1, -1
                        ),
                        raw_cache=raw_2b,
                        layout=((n_blk, 128, 128), torch.bfloat16, stride_b, idx_off),
                    )
            m_obj.finalize_mhc_broadcast_weights()

        # Verify SWACache + get_kv_cache_groups extra_retained_tokens=6 prevents
        # premature eviction at 63!
        monkeypatch.setattr(
            kv_cache_utils, "get_kv_cache_groups", kv_cache_utils.get_kv_cache_groups
        )
        torchtpu_root._patch_dflash_sliding_window_retained_tokens()
        dspark_l0 = cast(DeepseekV41DecoderLayer, dspark.model.layers[0])
        swa_spec = dspark_l0.attn.swa_cache_layer.get_kv_cache_spec(cast(Any, vcfg))
        spec_map = {"dspark.l0.swa": swa_spec}
        kv_cache_utils.get_kv_cache_groups(cast(Any, vcfg), spec_map)
        assert spec_map["dspark.l0.swa"].extra_retained_tokens == 6
        mgr = cast(Any, SimpleNamespace(sliding_window=32, extra_retained_tokens=6))
        assert SlidingWindowManager.get_num_skipped_tokens(mgr, 63) // 32 == 0
        assert SlidingWindowManager.get_num_skipped_tokens(mgr, 69) // 32 == 1

        def _build_md(
            *,
            n_reqs_padded: int,
            seq_lens: list[int],
            qsl: list[int],
            prompt_lens: list[int],
            replay_starts: list[int] | None,
            swa_bts: list[list[int]],
            comp_bts: list[int],
            mla1_bts: list[list[int]],
            mla2_bts: list[list[int]],
            idx1_bts: list[list[int]],
            idx2_bts: list[list[int]],
            idx3_bts: list[list[int]],
            dspark_bts: list[list[int]],
        ) -> dict[str, Any]:
            n_live = len(seq_lens)
            sl_p = tuple(seq_lens + [0] * (n_reqs_padded - n_live))
            qsl_p = tuple(qsl + [qsl[-1]] * (n_reqs_padded - n_live))
            pl_p = tuple(prompt_lens + [0] * (n_reqs_padded - n_live))
            rs_p = (
                None
                if replay_starts is None
                else tuple(replay_starts + [0] * (n_reqs_padded - n_live))
            )

            def _mk(rows_2d: list[list[int]], width: int) -> Any:
                bt = torch.zeros((n_reqs_padded, width), dtype=torch.int32)
                for r_i, row in enumerate(rows_2d):
                    for c_i, b_id in enumerate(row):
                        bt[r_i, c_i] = b_id
                return SimpleNamespace(
                    block_tables=bt,
                    seq_lens=torch.tensor(sl_p, dtype=torch.int32),
                    query_start_loc=torch.tensor(qsl_p, dtype=torch.int32),
                    query_start_loc_cpu=qsl_p,
                    seq_lens_cpu=sl_p,
                    prompt_lens_cpu=pl_p,
                    replay_starts_cpu=rs_p,
                )

            md: dict[str, Any] = {}
            for i_l, raw_lyr in enumerate(lm.model.layers):
                attn = cast(DeepseekV41DecoderLayer, raw_lyr).attn
                m_swa = _mk(swa_bts, 4)
                md[attn.swa_cache_layer.prefix] = md[attn.prefix] = m_swa
                if attn.compressor is not None:
                    m_mla = _mk(mla2_bts if attn.compress_ratio == 2 else mla1_bts, 4)
                    md[attn.compressor.k_cache_prefix] = m_mla
                    if attn.compressor.state_cache is not None:
                        md[attn.compressor.state_cache.prefix] = _mk(
                            [[b_id] for b_id in comp_bts], 2
                        )
                if attn.indexer is not None and attn.indexer.owns_k:
                    idx_rows = (
                        idx2_bts
                        if attn.indexer.compress_ratio == 2
                        else (idx1_bts if i_l == 2 else idx3_bts)
                    )
                    md[attn.indexer.k_cache.prefix] = _mk(idx_rows, 4)
            for raw_lyr in dspark.model.layers:
                attn = cast(DeepseekV41DecoderLayer, raw_lyr).attn
                md[attn.swa_cache_layer.prefix] = md[attn.prefix] = _mk(dspark_bts, 4)
            return md

        # 1. Warmup _dummy_run (block_tables = 0 = null_block, padded 16 > 4):
        #    null_block=0 in pool_lo & rope_lo MUST remain 100% zero!
        z4 = [[0], [0], [0], [0]]
        dummy_md = _build_md(
            n_reqs_padded=4,
            seq_lens=[1, 1, 1, 1],
            qsl=[0, 1, 2, 3, 4],
            prompt_lens=[1, 1, 1, 1],
            replay_starts=None,
            swa_bts=z4,
            comp_bts=[0, 0, 0, 0],
            mla1_bts=z4,
            mla2_bts=z4,
            idx1_bts=z4,
            idx2_bts=z4,
            idx3_bts=z4,
            dspark_bts=z4,
        )
        with (
            set_vllm_model_wrapper_context(mesh=cast(Any, None)),
            _forward_ctx(monkeypatch, dummy_md),
        ):
            _, aux_w = cast(
                tuple[torch.Tensor, list[torch.Tensor]],
                lm(
                    torch.zeros(16, dtype=torch.int32),
                    torch.zeros(16, dtype=torch.int32),
                ),
            )
            d_metas_w = [
                dummy_md[cast(DeepseekV41DecoderLayer, lyr).attn.swa_cache_layer.prefix]
                for lyr in dspark.model.layers
            ]
            dspark.precompute_and_store_context_kv(
                torch.cat(aux_w, dim=-1), torch.zeros(16, dtype=torch.int32), d_metas_w
            )
            dspark.model(
                torch.zeros(16, dtype=torch.int32), torch.zeros(16, dtype=torch.int32)
            )
        assert int((pool_lo[:256] != 0).sum()) == 0
        assert int((rope_lo[:64] != 0).sum()) == 0

        # Construct cross-dtype NaN bit-pattern blocks (Compressor f32 + MLA FP8+UE8M0):
        g_nan = torch.Generator().manual_seed(42)
        comp_blk = (
            (torch.randn((8, 1024), generator=g_nan, dtype=torch.float32) * 3.0)
            .view(torch.int32)
            .reshape(64, 128)
            .repeat(4, 1)
            .clone()
        )
        lat_outlier = (
            torch.randn((64, 512), generator=g_nan, dtype=torch.float32) * 140.0
        ).bfloat16()
        nope_w, _ = _t(pack_fp8_kv_rows(_j(lat_outlier), interpret=True))
        mla_blk = nope_w.repeat(4, 1)
        assert torch.isnan(comp_blk.view(torch.bfloat16)).any()
        assert torch.isnan(mla_blk.view(torch.bfloat16)).any()

        def _poison_pools() -> None:
            for b_i in range(n_blk_bank):
                src = comp_blk if (b_i % 2 == 0) else mla_blk
                pool_lo[b_i * 256 : (b_i + 1) * 256] = src
                pool_hi[b_i * 256 : (b_i + 1) * 256] = src

        torch.manual_seed(77)
        plen = 59
        prompt_ids = torch.randint(1, 50, (plen,), dtype=torch.int32)

        def _step(
            tok_ids: list[int],
            tok_pos: list[int],
            seq_l: int,
            r_start: int,
            bucket: int,
            *,
            run_dspark_ctx: bool = False,
        ) -> torch.Tensor:
            ids_b = torch.zeros(bucket, dtype=torch.int32)
            ids_b[: len(tok_ids)] = torch.tensor(tok_ids, dtype=torch.int32)
            pos_b = torch.zeros(bucket, dtype=torch.int32)
            pos_b[: len(tok_pos)] = torch.tensor(tok_pos, dtype=torch.int32)
            num_comp = seq_l - len(tok_ids)
            skip_blks = SlidingWindowManager.get_num_skipped_tokens(mgr, num_comp) // 32
            swa_row = [
                0 if c < skip_blks else b_id for c, b_id in enumerate([1, 17, 18, 0])
            ]
            ds_row = [
                0 if c < skip_blks else b_id for c, b_id in enumerate([2, 19, 20, 0])
            ]
            md = _build_md(
                n_reqs_padded=4,
                seq_lens=[seq_l],
                qsl=[0, len(tok_ids)],
                prompt_lens=[plen],
                replay_starts=[r_start],
                swa_bts=[swa_row],
                comp_bts=[16],
                mla1_bts=[[4, 22, 0, 0]],
                mla2_bts=[[3, 21, 0, 0]],
                idx1_bts=[[23, 0, 0, 0]],
                idx2_bts=[[5, 0, 0, 0]],
                idx3_bts=[[24, 0, 0, 0]],
                dspark_bts=[ds_row],
            )
            with (
                set_vllm_model_wrapper_context(mesh=cast(Any, None)),
                _forward_ctx(monkeypatch, md),
            ):
                h_out, aux_out = cast(
                    tuple[torch.Tensor, list[torch.Tensor]], lm(ids_b, pos_b)
                )
                assert not torch.isnan(h_out).any()
                if run_dspark_ctx:
                    d_metas = [
                        md[
                            cast(
                                DeepseekV41DecoderLayer, lyr
                            ).attn.swa_cache_layer.prefix
                        ]
                        for lyr in dspark.model.layers
                    ]
                    dspark.precompute_and_store_context_kv(
                        torch.cat(aux_out, dim=-1), pos_b, d_metas
                    )
                logits = lm.compute_logits(h_out[: len(tok_ids)])
                assert logits is not None
                return logits.float().detach().clone()

        # 2. Clean reference: fresh prefill [0..58] + 1-token causal decodes
        #    [59..61] and [62..67], plus DSpark context writes!
        pool_lo.zero_()
        pool_hi.zero_()
        rope_lo.zero_()
        rope_hi.zero_()
        _step(prompt_ids.tolist(), list(range(plen)), plen, 0, 64, run_dspark_ctx=True)
        step1_draft = [7, 8, 9, 21, 22, 23]  # 21..23 rejected on verify!
        step2_true = [31, 32, 33, 34, 35, 36]
        s1_ref = torch.cat(
            [
                _step([t_id], [59 + k], 60 + k, 0, 16, run_dspark_ctx=True)
                for k, t_id in enumerate(step1_draft[:3])
            ],
            dim=0,
        )
        s2_ref = torch.cat(
            [
                _step([t_id], [62 + k], 63 + k, 0, 16, run_dspark_ctx=True)
                for k, t_id in enumerate(step2_true)
            ],
            dim=0,
        )

        # 3. Candidate trajectory on NaN-poisoned 2-bank pool (INCLUDING null_block=0!):
        _poison_pools()
        pool_hi[:256] = mla_blk
        assert not torch.isfinite(pool_hi[:256].view(torch.float32)).all()
        comp_sc = next(
            cast(DeepseekV41DecoderLayer, lyr).attn.compressor.state_cache
            for lyr in lm.model.layers
            if cast(DeepseekV41DecoderLayer, lyr).attn.compressor is not None
            and cast(Any, lyr).attn.compressor.state_cache is not None
        )
        assert comp_sc is not None
        comp_sc.zero_blocks([16])
        assert torch.isfinite(pool_hi[:256].view(torch.float32)).all()
        assert int((pool_hi[:256] != 0).sum()) == 0
        _step(prompt_ids[:48].tolist(), list(range(48)), 48, 0, 64)
        pool_lo[:256] = comp_blk
        pool_hi[:256] = mla_blk
        comp_sc.zero_blocks([16])
        assert torch.isfinite(pool_hi[:256].view(torch.float32)).all()
        for b_id in (1, 2):
            pool_lo[b_id * 256 : (b_id + 1) * 256] = mla_blk
        for b_id in (17, 18, 19, 20):
            b_hi = b_id - n_blk_bank
            pool_hi[b_hi * 256 : (b_hi + 1) * 256] = comp_blk
        _step(
            prompt_ids[16:].tolist(),
            list(range(16, plen)),
            plen,
            16,
            64,
            run_dspark_ctx=True,
        )
        s1_ver_all = _step(
            step1_draft, list(range(59, 65)), 65, 16, 16, run_dspark_ctx=True
        )
        s2_ver = _step(step2_true, list(range(62, 68)), 68, 16, 16, run_dspark_ctx=True)
        s1_ver = s1_ver_all[:3]

        assert not torch.isnan(s1_ref).any() and not torch.isnan(s2_ref).any()
        assert not torch.isnan(s1_ver).any() and not torch.isnan(s2_ver).any()
        assert (s1_ref - s1_ver).abs().max().item() < 5e-2
        assert (s2_ref - s2_ver).abs().max().item() < 5e-2
        assert torch.equal(s1_ref.argmax(-1), s1_ver.argmax(-1))
        assert torch.equal(s2_ref.argmax(-1), s2_ver.argmax(-1))

        # Verify DSpark non-causal draft forward at num_computed=63 (retained),
        # num_computed=64 (exact 32*k null_block=0 boundary), AND num_computed=70
        # (col 0 evicted to poisoned null_block=0, replay_starts=[16]) with distinct
        # trailing token IDs and zero uncommitted SWACache mutation:
        _step([41, 42], [68, 69], 70, 16, 16, run_dspark_ctx=True)
        pool_lo_snap, pool_hi_snap = pool_lo.clone(), pool_hi.clone()
        for base_p in (63, 64, 70):
            skip_d = SlidingWindowManager.get_num_skipped_tokens(mgr, base_p) // 32
            n_comp_blks = (base_p + 31) // 32
            ds_b = [
                0 if (c < skip_d or c >= n_comp_blks) else b_id
                for c, b_id in enumerate([2, 19, 20, 0])
            ]
            d_ids = torch.full((16,), 63, dtype=torch.int32)
            d_ids[0], d_ids[5], d_ids[10], d_ids[15] = 9, 11, 13, 7
            d_pos = torch.zeros(16, dtype=torch.int32)
            d_pos[:5] = torch.arange(base_p, base_p + 5, dtype=torch.int32)
            d_md = _build_md(
                n_reqs_padded=3,
                seq_lens=[base_p + 5],
                qsl=[0, 5],
                prompt_lens=[plen],
                replay_starts=[16],
                swa_bts=[[1, 17, 18, 0]],
                comp_bts=[16],
                mla1_bts=[[4, 22, 0, 0]],
                mla2_bts=[[3, 21, 0, 0]],
                idx1_bts=[[23, 0, 0, 0]],
                idx2_bts=[[5, 0, 0, 0]],
                idx3_bts=[[24, 0, 0, 0]],
                dspark_bts=[ds_b],
            )
            with (
                set_vllm_model_wrapper_context(mesh=cast(Any, None)),
                _forward_ctx(monkeypatch, d_md),
            ):
                d_out = dspark.model(input_ids=d_ids, positions=d_pos)
                assert torch.isfinite(d_out).all()
        assert torch.equal(pool_lo, pool_lo_snap)
        assert torch.equal(pool_hi, pool_hi_snap)


@pytest.mark.tpu_test
def test_model_and_dspark_tpu_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # pragma: no cover
    """Run real 4-layer DeepseekV41ForCausalLM and DSpark on TPU7x."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    import torch_tpu as _torch_tpu
    import vllm.config.vllm as vllm_cfg_mod
    import vllm.distributed.parallel_state as ps_mod
    import vllm.v1.attention.backends.mla.sparse_swa as swa_mod

    from tests.models.vllm.deepseek_v41.test_attention import (
        _ACTIVE_VLLM_CONFIG,
        _cpu_kv_rope,
        _make_meta,
        _patch_vllm_attention_deps,
    )
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
    from vllm_torchtpu.models.vllm.deepseek_v4 import moe as v4_moe_mod
    from vllm_torchtpu.models.vllm.deepseek_v41 import (
        attention as attn_mod,
    )
    from vllm_torchtpu.models.vllm.deepseek_v41 import (
        compressor as compressor_mod,
    )
    from vllm_torchtpu.models.vllm.deepseek_v41 import (
        indexer as indexer_mod,
    )
    from vllm_torchtpu.models.vllm.deepseek_v41.attention import (
        DeepseekCompressor,
        DeepseekV41Attention,
        DeepseekV41Indexer,
    )
    from vllm_torchtpu.runner.utils import relayout_rope_caches

    assert _torch_tpu is not None

    class _RealGateLinear(nn.Module):
        def __init__(self, input_size: int, output_size: int, **_kw: Any) -> None:
            super().__init__()
            gen = torch.Generator().manual_seed(input_size * 97 + output_size)
            w = torch.randn((output_size, input_size), generator=gen) * 0.05
            self.weight = nn.Parameter(w.bfloat16())
            self.e_score_correction_bias = None
            self.tid2eid = None

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
            return (x.float() @ self.weight.float().T), None

    class _RealSwiGLUExperts(nn.Module):
        def __init__(
            self,
            num_experts: int,
            top_k: int,
            hidden_size: int,
            intermediate_size: int,
            **_kw: Any,
        ) -> None:
            super().__init__()
            self.top_k = top_k
            self.gate = None
            self.moe_config = SimpleNamespace(
                dp_size=1,
                pcp_size=1,
                moe_parallel_config=SimpleNamespace(use_ep=False),
            )
            self.routed_experts = SimpleNamespace(
                quant_method=SimpleNamespace(
                    is_monolithic=True, supports_internal_mk=False
                ),
                custom_routing_function=None,
            )
            gen = torch.Generator().manual_seed(hidden_size * 31 + num_experts)
            w1 = torch.randn(
                (num_experts, intermediate_size, hidden_size), generator=gen
            )
            w2 = torch.randn(
                (num_experts, hidden_size, intermediate_size), generator=gen
            )
            self.w1 = nn.Parameter((w1 * 0.05).bfloat16())
            self.w2 = nn.Parameter((w2 * 0.05).bfloat16())

        def forward(
            self,
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
            input_ids: Any = None,
        ) -> torch.Tensor:
            del input_ids
            probs = torch.softmax(router_logits.float(), dim=-1)
            vals, idx = torch.topk(probs, self.top_k, dim=-1)
            vals = vals / vals.sum(dim=-1, keepdim=True)
            h = hidden_states.float()
            w1, w2 = self.w1.float(), self.w2.float()
            out = torch.zeros_like(h)
            for k in range(self.top_k):
                e = idx[:, k]
                g_out = torch.bmm(h.unsqueeze(1), w1[e].transpose(1, 2)).squeeze(1)
                act = torch.nn.functional.silu(g_out)
                proj = torch.bmm(act.unsqueeze(1), w2[e].transpose(1, 2)).squeeze(1)
                out = out + vals[:, k : k + 1] * proj
            return out.to(hidden_states.dtype)

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
        o_rot = _cpu_kv_rope(o, pos_i32, cs_inv).float()
        t, g_cnt = o.shape[0], s.shape[0] // 128
        z = torch.einsum(
            "tgd,dgr->tgr", o_rot.view(t, g_cnt, -1), w.float().view(-1, g_cnt, 128)
        )
        return (z.reshape(t, -1) * s.float()).bfloat16()

    dummy_grp = SimpleNamespace(
        world_size=1,
        rank_in_group=0,
        is_first_rank=True,
        is_last_rank=True,
        cpu_group=None,
    )
    for g in ("_TP", "_PP", "_DP", "_EP"):
        monkeypatch.setattr(ps_mod, g, dummy_grp)
    _patch_vllm_attention_deps(monkeypatch, patch_rope_op=False)
    monkeypatch.setattr(v4_moe_mod, "GateLinear", _RealGateLinear)
    monkeypatch.setattr(v4_moe_mod, "FusedMoEFactory", _RealSwiGLUExperts)
    monkeypatch.setattr(v4_moe_mod, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(v4_moe_mod, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        model_mod,
        "VocabParallelEmbedding",
        lambda num_embeddings, embedding_dim, **kw: nn.Embedding(
            num_embeddings, embedding_dim, dtype=torch.bfloat16
        ),
    )
    monkeypatch.setattr(
        model_mod.upstream,
        "ParallelLMHead",
        lambda num_embeddings, embedding_dim, **kw: nn.Linear(
            embedding_dim, num_embeddings, bias=False, dtype=torch.bfloat16
        ),
    )
    monkeypatch.setattr(
        model_mod.upstream,
        "LogitsProcessor",
        lambda vocab_size, **kw: lambda head, hidden, **lkw: head(hidden),
    )

    class _TpuStubReplicatedLinear(nn.Linear):
        def __init__(
            self,
            in_f: int,
            out_f: int,
            bias: bool = True,
            *,
            return_bias: bool = True,
            **_kw: Any,
        ) -> None:
            super().__init__(in_f, out_f, bias=bias, dtype=torch.bfloat16)
            self.return_bias = return_bias

        def forward(self, input: torch.Tensor) -> Any:
            out = super().forward(input)
            return (out, self.bias) if self.return_bias else out

    monkeypatch.setattr(dspark_mod, "ReplicatedLinear", _TpuStubReplicatedLinear)
    monkeypatch.setattr(
        dspark_mod,
        "DSparkMarkovHead",
        lambda vocab, draft_vocab, rank, **kw: SimpleNamespace(
            markov_w1=nn.Linear(vocab, rank, bias=False, dtype=torch.bfloat16),
            markov_w2=nn.Linear(rank, draft_vocab, bias=False, dtype=torch.bfloat16),
        ),
    )
    monkeypatch.setattr(
        dspark_mod,
        "DSparkConfidenceHead",
        lambda in_dim, prefix="", **kw: nn.Linear(
            in_dim, 1, bias=False, dtype=torch.bfloat16
        ),
    )
    monkeypatch.setattr(
        dspark_mod,
        "LogitsProcessor",
        lambda vocab_size, **kw: lambda head, hidden, **lkw: head(hidden),
    )
    monkeypatch.setattr(
        model_mod.upstream.EngramLayout,
        "from_config",
        classmethod(lambda cls, cfg: None),
    )

    def _build_real_vcfg(s_ctx: dict[str, Any]) -> Any:
        hf = SimpleNamespace(
            hidden_size=256,
            vocab_size=64,
            num_hidden_layers=4,
            num_attention_heads=16,
            q_lora_rank=16,
            o_lora_rank=128,
            head_dim=512,
            qk_rope_head_dim=64,
            o_groups=2,
            sliding_window=128,
            rms_norm_eps=1e-6,
            max_position_embeddings=32,
            compress_ratios=[0, 2, 1, 0, 0],
            kv_source_layer_ids=[1, 2],
            index_source_layer_ids=[1, 2],
            candidate_source_layer_id=1,
            candidate_topk_blocks=16,
            candidate_block_size=2,
            index_n_heads=8,
            index_head_dim=128,
            index_topk=32,
            rope_theta=10000.0,
            compress_rope_theta=10000.0,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
            rope_scaling=None,
            n_group=1,
            n_routed_experts=4,
            num_experts_per_tok=2,
            n_shared_experts=None,
            moe_intermediate_size=64,
            swiglu_limit=None,
            norm_topk_prob=True,
            scoring_func="softmax",
            routed_scaling_factor=1.0,
            num_hash_layers=0,
            dspark_n_routed_experts=4,
            dspark_num_experts_per_tok=2,
            dspark_target_layer_ids=[2, 3],
            dspark_markov_rank=4,
            dspark_block_size=2,
            num_nextn_predict_layers=1,
            hc_mult=4,
            hc_eps=1e-4,
            hc_sinkhorn_iters=2,
            expert_dtype="fp8",
            engram_layer_ids=[],
        )
        return SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=hf, dtype=torch.bfloat16, max_model_len=16
            ),
            quant_config=None,
            cache_config=SimpleNamespace(
                block_size=4,
                cache_dtype="auto",
                enable_prefix_caching=True,
                swa_bounded_replay=True,
            ),
            parallel_config=SimpleNamespace(
                pipeline_parallel_size=1,
                tensor_parallel_size=1,
                data_parallel_size=1,
                enable_expert_parallel=False,
                prefill_context_parallel_size=1,
                decode_context_parallel_size=1,
                pcp_shard_decode_requests=False,
                cp_kv_cache_interleave_size=1,
                enable_eplb=False,
            ),
            kernel_config=SimpleNamespace(moe_backend="auto"),
            scheduler_config=SimpleNamespace(
                max_num_batched_tokens=8,
                max_num_seqs=4,
                disable_hybrid_kv_cache_manager=False,
            ),
            attention_config=SimpleNamespace(
                indexer_sparse_logits=False,
                sparse_mla_prefill_workspace_factor=5,
                hisparse_config=None,
            ),
            compilation_config=SimpleNamespace(
                mode=0,
                compile_sizes=[8],
                static_forward_context=s_ctx,
                custom_ops=["none"],
                enabled_custom_ops=Counter(),
                disabled_custom_ops=Counter(),
            ),
            engram_config=None,
            speculative_config=SimpleNamespace(
                method="dspark",
                num_speculative_tokens=2,
                draft_model_config=SimpleNamespace(hf_config=hf),
            ),
            num_speculative_tokens=2,
            num_lookahead_tokens=2,
        )

    def _init_layer_weights_and_caches(
        model: DeepseekV41Model | DSparkDeepseekV41Model, dev: torch.device
    ) -> dict[str, Any]:
        t_buf = model.topk_indices_buffer.to(dev)
        model.topk_indices_buffer = t_buf
        c_buf: torch.Tensor | None = None
        if isinstance(model, DeepseekV41Model):
            c_buf_raw = model.candidate_block_buffer
            c_buf = None if c_buf_raw is None else c_buf_raw.to(dev)
            model.candidate_block_buffer = c_buf
        meta = _make_meta(
            block_tables=torch.tensor([[0], [1]], dtype=torch.int32, device=dev),
            seq_lens=[4, 4],
            query_start_loc=[0, 4, 8],
            device=dev,
        )
        meta.query_start_loc_cpu = torch.tensor([0, 4, 8], dtype=torch.int32)
        meta.seq_lens_cpu = torch.tensor([4, 4], dtype=torch.int32)
        meta.prompt_lens_cpu = torch.tensor([4, 4], dtype=torch.int32)
        meta.replay_starts_cpu = None
        meta_dict: dict[str, Any] = {}
        gen = torch.Generator().manual_seed(999)
        for raw_lyr in model.layers:
            lyr = cast(DeepseekV41DecoderLayer, raw_lyr)
            for param in (
                lyr.hc_attn_fn,
                lyr.hc_ffn_fn,
                lyr.hc_attn_base,
                lyr.hc_ffn_base,
                lyr.hc_attn_scale,
                lyr.hc_ffn_scale,
            ):
                vals = torch.randn(param.shape, generator=gen, dtype=torch.float32)
                param.data.copy_((vals * 0.02).to(dev))
            attn = lyr.attn
            attn.topk_indices_buffer = t_buf
            if attn.candidate_block_buffer is not None:
                attn.candidate_block_buffer = c_buf
            w_fp8 = (
                torch.randn((256, 4096), generator=gen, dtype=torch.float32) * 0.05
            ).to(torch.float8_e4m3fn)
            attn.wo_a.weight = nn.Parameter(w_fp8.to(dev), requires_grad=False)
            attn.wo_a.weight_scale = nn.Parameter(
                torch.full((256,), 0.05, dtype=torch.float32, device=dev),
                requires_grad=False,
            )
            attn.process_weights_after_loading(torch.bfloat16)
            if dev.type == "tpu":
                relayout_rope_caches(attn)
            attn.swa_cache_layer.kv_cache = torch.zeros(
                (4, 16, 512), dtype=torch.bfloat16, device=dev
            )
            meta_dict[attn.swa_cache_layer.prefix] = meta
            meta_dict[attn.prefix] = meta
            if attn.compressor is not None:
                attn._tpu_paged_cache_rows = PagedCacheRows(
                    torch.zeros((64, 128), dtype=torch.int32, device=dev),
                    ((4, 16, 512), torch.uint8, 8192, 0),
                    torch.zeros((16, 128), dtype=torch.int32, device=dev),
                )
                meta_dict[attn.compressor.k_cache_prefix] = meta
                if attn.compressor.state_cache is not None:
                    attn.compressor.state_cache.kv_cache = torch.zeros(
                        (4, 16, 1024), dtype=torch.float32, device=dev
                    )
                    meta_dict[attn.compressor.state_cache.prefix] = meta
            if attn.indexer is not None:
                attn.indexer.topk_indices_buffer = t_buf
                attn.indexer.indexer_op.topk_indices_buffer = t_buf
                if attn.indexer.indexer_op.candidate_blocks is not None:
                    attn.indexer.indexer_op.candidate_blocks = c_buf
                attn.indexer.k_cache._tpu_paged_cache_rows = PagedCacheRows(
                    torch.zeros((32, 128), dtype=torch.int32, device=dev),
                    ((4, 16, 128), torch.bfloat16, 4096, 0),
                )
                meta_dict[attn.indexer.k_cache.prefix] = meta
        return meta_dict

    def _run_suite(
        dev: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s_ctx: dict[str, Any] = {}
        vcfg = _build_real_vcfg(s_ctx)
        _ACTIVE_VLLM_CONFIG[:] = [vcfg]
        monkeypatch.setattr(vllm_cfg_mod, "_current_vllm_config", vcfg)
        for mod in (
            vllm_cfg_mod,
            attn_mod.upstream,
            indexer_mod.upstream,
            compressor_mod.compressor_upstream,
            swa_mod,
        ):
            monkeypatch.setattr(mod, "get_current_vllm_config", lambda: vcfg)
        torch.manual_seed(2026)
        lm = DeepseekV41ForCausalLM(vllm_config=cast(Any, vcfg)).to(dev)
        lm.set_aux_hidden_state_layers((2, 3))
        meta_dict = _init_layer_weights_and_caches(lm.model, dev)
        lm.process_weights_after_loading()

        torch.manual_seed(2027)
        dspark = DSparkDeepseekV41ForCausalLM(vllm_config=cast(Any, vcfg)).to(dev)
        dspark.embed_tokens = lm.model.embed_tokens
        dspark.model.embed_tokens = lm.model.embed_tokens
        dspark_meta = _init_layer_weights_and_caches(dspark.model, dev)
        dspark.process_weights_after_loading()
        meta_dict.update(dspark_meta)

        ids = torch.tensor([1, 2, 3, 2, 5, 2, 7, 2], dtype=torch.int32, device=dev)
        pos = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.int32, device=dev)
        with (
            set_vllm_model_wrapper_context(mesh=cast(Any, None)),
            _forward_ctx(monkeypatch, meta_dict),
        ):
            out, aux = cast(tuple[torch.Tensor, list[torch.Tensor]], lm(ids, pos))
            aux_cat = torch.cat(aux, dim=-1)
            d_metas = [
                meta_dict[
                    cast(DeepseekV41DecoderLayer, lyr).attn.swa_cache_layer.prefix
                ]
                for lyr in dspark.model.layers
            ]
            dspark.precompute_and_store_context_kv(aux_cat, pos, d_metas)
            d_out = dspark(ids, pos)
            d_out_kw = dspark.model(input_ids=ids, positions=pos)
        return out.cpu(), aux[0].cpu(), d_out.cpu(), d_out_kw.cpu()

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

    tpu7x_mesh = jax.sharding.AbstractMesh(
        (), (), abstract_device=jax.sharding.AbstractDevice("TPU7x", 1, "tpu")
    )
    with (
        monkeypatch.context() as mp,
        jax.default_device(jax.devices("cpu")[0]),
        jax.sharding.use_abstract_mesh(tpu7x_mesh),
    ):
        orig_mm = torch.mm
        mp.setattr(
            torch,
            "mm",
            lambda a, b, *, out_dtype=None, **kw: (
                orig_mm(a.to(out_dtype), b.to(out_dtype), **kw)
                if out_dtype is not None
                else orig_mm(a, b, **kw)
            ),
        )
        mp.setattr(model_mod, "get_mhc_post_op", lambda: mhc_post_torch)
        for cls in (DeepseekCompressor, DeepseekV41Indexer, DeepseekV41Attention):
            mp.setattr(cls, "kv_rope_op", property(lambda _s: _cpu_kv_rope))
        mp.setattr(DeepseekV41Attention, "o_proj_op", property(lambda _s: _cpu_o_proj))
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
            swa_min_pos=None: (
                _t(
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
                )
            ),
        )
        out_cpu, aux_cpu, dspark_cpu, dspark_kw_cpu = _run_suite(torch.device("cpu"))

    out_tpu, aux_tpu, dspark_tpu, dspark_kw_tpu = _run_suite(torch.device("tpu:0"))
    torch.testing.assert_close(out_tpu, out_cpu, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(aux_tpu, aux_cpu, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(dspark_tpu, dspark_cpu, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(dspark_kw_tpu, dspark_kw_cpu, rtol=5e-2, atol=5e-2)
