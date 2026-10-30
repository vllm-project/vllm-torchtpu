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
"""Unit and TPU tests for DeepSeek-V4.1 cache, CED replay, and MoE modules."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from vllm_torchtpu.models.vllm.deepseek_v41 import (
    DecoderReplayPlan,
    DeepseekV41MoE,
    NativeCacheView,
    PagedCacheRows,
    build_decoder_inverse_rows,
)
from vllm_torchtpu.models.vllm.deepseek_v41 import (
    cache as cache_mod,
)
from vllm_torchtpu.models.vllm.deepseek_v41 import (
    ced as ced_mod,
)
from vllm_torchtpu.models.vllm.deepseek_v41 import (
    moe as moe_mod,
)


@pytest.fixture
def mock_cache_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock Pallas cache custom ops on CPU and simulate buffer donation on writes."""

    def _gather(
        lo: torch.Tensor, idx: torch.Tensor, hi: torch.Tensor | None = None
    ) -> torch.Tensor:
        n_lo = lo.shape[0]
        valid = (idx >= 0) & (idx < n_lo + (0 if hi is None else hi.shape[0]))
        safe = idx.masked_fill(~valid, 0).long()
        if hi is None:
            out = lo.index_select(0, safe)
        else:
            in_hi = safe >= n_lo
            v_lo = lo.index_select(0, safe.masked_fill(in_hi, 0))
            v_hi = hi.index_select(0, (safe - n_lo).masked_fill(~in_hi, 0))
            out = torch.where(in_hi.unsqueeze(-1), v_hi, v_lo)
        return out.masked_fill(~valid.unsqueeze(-1), 0)

    def _write(
        bank: torch.Tensor,
        updates: torch.Tensor,
        slots: torch.Tensor,
        off: int = 0,
        after: torch.Tensor | None = None,
        page_tokens: int = 16,
    ) -> torch.Tensor:
        del after, page_tokens
        out = bank.clone()
        sub = out.view(-1, updates.shape[1])
        loc = slots.long() - int(off)
        valid = (loc >= 0) & (loc < sub.shape[0])
        sub[loc[valid]] = updates[valid]
        return out

    def _pack_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = x.shape[0]
        return (
            torch.ones((n, 128), dtype=torch.int32),
            torch.ones((n, 32), dtype=torch.int32) * 2,
        )

    monkeypatch.setattr(cache_mod.cache_op, "dsv41_gather_cache_rows", _gather)
    monkeypatch.setattr(cache_mod.cache_op, "dsv41_write_sub_rows", _write)
    monkeypatch.setattr(cache_mod.cache_op, "dsv41_pack_fp8_kv_rows", _pack_fp8)


def test_paged_cache_rows_rejects_invalid_bank_or_layout() -> None:
    """Verify PagedCacheRows layout and bank validation errors."""
    raw = torch.zeros((8, 128), dtype=torch.int32)
    with pytest.raises(ValueError, match="Invalid native cache bank"):
        PagedCacheRows(
            torch.zeros((8, 64), dtype=torch.int32),
            ((2, 4, 128), torch.bfloat16, 1024, 0),
        )
    with pytest.raises(ValueError, match="FP8 records"):
        PagedCacheRows(raw, ((2, 4, 512), torch.uint8, 2048, 0), rope=None)
    with pytest.raises(ValueError, match="FP8 records"):
        PagedCacheRows(
            torch.zeros((6, 128), dtype=torch.int32),
            ((1, 6, 512), torch.uint8, 3072, 0),
            rope=torch.zeros((1, 128), dtype=torch.int32),
        )
    with pytest.raises(ValueError, match="word/row aligned"):
        PagedCacheRows(raw, ((2, 4, 128), torch.bfloat16, 1024, 128))


def test_paged_cache_rows_pages_alignment() -> None:
    """Verify PagedCacheRows.pages returns page indices only when block-aligned."""
    raw = torch.zeros((8, 128), dtype=torch.int32)
    pcr_unaligned = PagedCacheRows(raw, ((2, 4, 128), torch.bfloat16, 2048, 512))
    assert pcr_unaligned.numel() == 1024 and pcr_unaligned.device.type == "cpu"
    assert pcr_unaligned.pages(torch.tensor([[0, 1]], dtype=torch.int32)) is None

    pcr_aligned = PagedCacheRows(raw, ((2, 4, 256), torch.bfloat16, 2048, 0))
    assert torch.equal(
        pcr_aligned.pages(torch.tensor([[0, 1]], dtype=torch.int64)),
        torch.tensor([[0, 1]], dtype=torch.int32),
    )

    pcr_bad_r = PagedCacheRows(
        torch.zeros((6, 128), dtype=torch.int32), ((2, 3, 256), torch.bfloat16, 1536, 0)
    )
    assert pcr_bad_r.pages(torch.tensor([[0]], dtype=torch.int32)) is None


def test_paged_cache_rows_sub_row_bf16_updates_persistent_banks_across_steps(
    mock_cache_ops: None,
) -> None:
    """Verify sub-row BF16 writes update persistent pools in-place across steps."""
    raw_lo = torch.zeros((8, 128), dtype=torch.int32)
    raw_hi = torch.zeros((8, 128), dtype=torch.int32)
    layout = ((2, 4, 128), torch.bfloat16, 2048, 512)
    pcr_step1 = PagedCacheRows((raw_lo, raw_hi), layout)
    vals = torch.randn(3, 128, dtype=torch.bfloat16)
    pcr_step1.write(torch.tensor([0, 4, -1], dtype=torch.int64), vals.float())
    assert pcr_step1.raw is raw_lo and pcr_step1.raw_hi is raw_hi
    pcr_step2 = PagedCacheRows((raw_lo, raw_hi), layout)
    fetched = pcr_step2[torch.tensor([0, 1]), torch.tensor([0, 0])]
    assert torch.allclose(fetched[0], vals[0])
    assert torch.allclose(fetched[1], vals[1])


def test_paged_cache_rows_multi_row_float32_read_write(mock_cache_ops: None) -> None:
    """Verify multi-row float32 records expand across consecutive 128-word rows."""
    raw_f32 = torch.zeros((8, 128), dtype=torch.int32)
    pcr_f32 = PagedCacheRows(raw_f32, ((1, 4, 256), torch.float32, 4096, 0))
    vals_f32 = torch.randn(2, 256, dtype=torch.float32)
    pcr_f32.write(
        torch.tensor([0, 1], dtype=torch.int32),
        vals_f32,
        valid=torch.tensor([True, True]),
    )
    got_f32 = pcr_f32[torch.tensor([0, 0]), torch.tensor([0, 1])]
    assert torch.allclose(got_f32, vals_f32)


def test_paged_cache_rows_fp8_write_updates_persistent_raw_and_rope_banks(
    mock_cache_ops: None,
) -> None:
    """Verify FP8 writes update persistent raw and RoPE 2-bank pools in place."""
    raw_lo = torch.zeros((8, 128), dtype=torch.int32)
    raw_hi = torch.zeros((8, 128), dtype=torch.int32)
    rope_lo = torch.zeros((2, 128), dtype=torch.int32)
    rope_hi = torch.zeros((2, 128), dtype=torch.int32)
    pcr_fp8 = PagedCacheRows(
        (raw_lo, raw_hi),
        ((2, 4, 512), torch.uint8, 2048, 0),
        rope=(rope_lo, rope_hi),
    )
    lat = torch.randn(2, 512, dtype=torch.bfloat16)
    pcr_fp8.write(
        torch.tensor([0, 1], dtype=torch.int32),
        lat,
        records=torch.tensor([0, 8], dtype=torch.int64),
    )
    assert pcr_fp8.raw is raw_lo and pcr_fp8.raw_hi is raw_hi
    assert pcr_fp8.rope is rope_lo and pcr_fp8.rope_hi is rope_hi
    assert torch.all(raw_lo[0] == 1) and torch.all(raw_hi[0] == 1)
    assert torch.all(rope_lo[0, :32] == 2)
    assert torch.all(rope_hi[0, :32] == 2)


def test_write_cache_dispatches_paged_rows_and_dense_tensors(
    mock_cache_ops: None,
) -> None:
    """Verify write_cache helper handles PagedCacheRows, 2D tensors, and 3D tensors."""
    pcr = PagedCacheRows(
        torch.zeros((8, 128), dtype=torch.int32), ((1, 4, 256), torch.float32, 4096, 0)
    )
    vals = torch.randn(2, 256, dtype=torch.float32)
    valid = torch.tensor([True, False], dtype=torch.bool)
    cache_mod.write_cache(pcr, torch.tensor([0, 1]), vals, valid)
    cache_mod.write_cache(
        pcr, torch.tensor([0, 1]), vals, valid, records=torch.tensor([0, 1])
    )
    t_2d = torch.zeros((4, 256), dtype=torch.float32)
    cache_mod.write_cache(t_2d, torch.tensor([2, 3]), vals, valid)
    assert torch.allclose(t_2d[2], vals[0]) and torch.all(t_2d[3] == 0)
    t_3d = torch.zeros((2, 2, 256), dtype=torch.float32)
    cache_mod.write_cache(t_3d, torch.tensor([3, 0]), vals, valid)
    assert torch.allclose(t_3d[1, 1], vals[0])


def test_paged_cache_rows_after_dependency_and_reset(
    mock_cache_ops: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify _last_read capture during compile, after= forwarding, and reset."""
    seen_after: list[torch.Tensor | None] = []
    seen_page_tokens: list[int] = []
    orig_write = cache_mod.cache_op.dsv41_write_sub_rows

    def _spy_write(
        bank: torch.Tensor,
        updates: torch.Tensor,
        slots: torch.Tensor,
        off: int = 0,
        after: torch.Tensor | None = None,
        page_tokens: int = 16,
    ) -> torch.Tensor:
        seen_after.append(None if after is None else after.clone())
        seen_page_tokens.append(page_tokens)
        return orig_write(
            bank, updates, slots, off, after=after, page_tokens=page_tokens
        )

    monkeypatch.setattr(cache_mod.cache_op, "dsv41_write_sub_rows", _spy_write)
    raw = torch.full((8, 128), 42, dtype=torch.int32)
    pcr = PagedCacheRows(raw, ((2, 4, 128), torch.bfloat16, 2048, 0))
    vals = torch.ones((2, 128), dtype=torch.bfloat16)
    valid = torch.tensor([True, True], dtype=torch.bool)
    slots = torch.tensor([0, 1], dtype=torch.int32)

    # Eager mode: _last_read is never retained and after= stays None.
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    _ = pcr[torch.tensor([0]), torch.tensor([0])]
    assert pcr._last_read is None
    cache_mod.write_cache(pcr, slots, vals, valid, after_read=True)
    assert seen_after[-1] is None

    # Compile mode: __getitem__ records raw int32 gather word, write consumes & resets.
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    _ = pcr[torch.tensor([1]), torch.tensor([0])]
    assert pcr._last_read is not None and int(pcr._last_read.item()) == 42
    cache_mod.write_cache(pcr, slots, vals, valid, after_read=True)
    assert seen_after[-1] is not None and int(seen_after[-1].item()) == 42
    assert pcr._last_read is None

    # Compile mode with after_read=False: _last_read is cleared without forwarding.
    _ = pcr[torch.tensor([1]), torch.tensor([1])]
    assert pcr._last_read is not None
    cache_mod.write_cache(pcr, slots, vals, valid, after_read=False)
    assert seen_after[-1] is None and pcr._last_read is None

    # Compile mode with after_read=True when no prior read occurred: after= stays None.
    cache_mod.write_cache(pcr, slots, vals, valid, after_read=True)
    assert seen_after[-1] is None
    assert seen_page_tokens == [4, 4, 4, 4]


def test_paged_cache_rows_fullgraph_compile_trace() -> None:
    """Verify Dynamo fullgraph trace wires gather -> dsv41_write_sub_rows_after."""
    raw = torch.zeros((8, 128), dtype=torch.int32)
    pcr = PagedCacheRows(raw, ((2, 4, 256), torch.bfloat16, 2048, 0))
    vals = torch.ones((2, 256), dtype=torch.bfloat16)
    slots = torch.tensor([0, 1], dtype=torch.int32)
    valid = torch.tensor([True, True], dtype=torch.bool)
    captured_ops: list[str] = []

    def _capture(gm: torch.fx.GraphModule, _inputs: list[torch.Tensor]) -> Any:
        captured_ops.extend(
            str(n.target) for n in gm.graph.nodes if n.op == "call_function"
        )
        return lambda *_a: (torch.zeros((1, 256), dtype=torch.bfloat16),)

    def _read_then_write(
        vals_in: torch.Tensor, s_in: torch.Tensor, v_in: torch.Tensor
    ) -> torch.Tensor:
        pre = pcr[s_in[:1], s_in[:1]]
        cache_mod.write_cache(pcr, s_in, vals_in, v_in, after_read=True)
        return pre

    torch._dynamo.reset()
    compiled_fn = torch.compile(
        _read_then_write, backend=_capture, fullgraph=True, dynamic=False
    )
    out = compiled_fn(vals, slots, valid)
    assert out.shape == (1, 256)
    assert pcr._last_read is None
    assert "pallas.dsv41_gather_cache_rows.default" in captured_ops
    assert "pallas.dsv41_write_sub_rows_after.default" in captured_ops
    assert captured_ops.index(
        "pallas.dsv41_gather_cache_rows.default"
    ) < captured_ops.index("pallas.dsv41_write_sub_rows_after.default")


def test_native_cache_view_binding_and_compile_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify NativeCacheView initializes fields and switches kv_cache on compile."""
    layer = NativeCacheView()
    assert layer.paged_cache_rows is None and layer.kv_cache is None
    dense = torch.zeros((2, 4, 256), dtype=torch.bfloat16)
    raw = torch.zeros((8, 128), dtype=torch.int32)
    layer.kv_cache = dense
    assert layer.kv_cache is dense
    layer.bind_kv_cache(dense, raw_cache=None)
    assert layer.paged_cache_rows is None
    layer.bind_kv_cache(dense, raw_cache=raw)
    assert isinstance(layer.paged_cache_rows, PagedCacheRows)
    with pytest.raises(NotImplementedError):
        layer.get_kv_cache_spec(MagicMock())

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    assert isinstance(layer.kv_cache, PagedCacheRows)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    assert layer.kv_cache is dense


def test_native_cache_view_materializes_single_and_two_bank_pools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify NativeCacheView materializes empty, 1-bank, and 2-bank KV/RoPE pools."""
    layer = NativeCacheView()
    monkeypatch.setattr(cache_mod, "allocate_upstream_kv_cache", lambda *a: {})
    empty_mat = layer.materialize_kv_caches(
        MagicMock(), torch.device("cpu"), MagicMock()
    )
    assert empty_mat.kv_caches == {} and empty_mat.raw_tensors == []

    meta_fp8 = torch.empty((4, 4, 512), dtype=torch.uint8, device="meta")
    meta_bf16 = torch.empty((4, 4, 256), dtype=torch.bfloat16, device="meta")
    monkeypatch.setattr(
        cache_mod,
        "allocate_upstream_kv_cache",
        lambda *a: {"l0": meta_fp8, "l1": meta_bf16},
    )
    monkeypatch.setattr(
        cache_mod, "layer_to_pool_index", lambda cfg: {"l0": 0, "l1": 0}
    )
    cfg = SimpleNamespace(
        num_blocks=4,
        kv_cache_tensors=[SimpleNamespace(size=8192)],
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=SimpleNamespace(page_size_bytes=2048))
        ],
    )
    mat1 = layer.materialize_kv_caches(
        cfg, torch.device("cpu"), SimpleNamespace(is_layer_compact=True)
    )
    assert "l0" in mat1.kv_caches and "l0" in mat1.rope_pools

    monkeypatch.setattr(cache_mod, "SPARSECORE_MAX_POOL_BYTES", 4096)
    mat2 = layer.materialize_kv_caches(
        cfg, torch.device("cpu"), SimpleNamespace(is_layer_compact=False)
    )
    assert isinstance(mat2.native_raw_cache["l0"], tuple)
    assert isinstance(mat2.rope_pools["l0"], tuple)

    monkeypatch.setattr(
        cache_mod, "allocate_upstream_kv_cache", lambda *a: {"l1": meta_bf16}
    )
    mat_bf16 = layer.materialize_kv_caches(
        cfg, torch.device("cpu"), SimpleNamespace(is_layer_compact=False)
    )
    assert mat_bf16.rope_pools == {}


def test_native_cache_view_reserves_companion_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify reserve_companion_memory reserves 20% for RoPE and caps at 2 banks."""
    layer = NativeCacheView()
    monkeypatch.setattr(cache_mod, "SPARSECORE_MAX_POOL_BYTES", 4096)

    def _vcfg(arch: str, dtype: str) -> Any:
        return SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(architectures=[arch])
            ),
            cache_config=SimpleNamespace(cache_dtype=dtype),
        )

    assert layer.reserve_companion_memory(_vcfg("Other", "fp8"), 1000) == 1000
    assert (
        layer.reserve_companion_memory(_vcfg("DeepseekV41ForCausalLM", "auto"), 1000)
        == 1000
    )
    vcfg_fp8 = _vcfg("DeepseekV41ForCausalLM", "fp8")
    assert layer.reserve_companion_memory(vcfg_fp8, 4000, auto=True) == 3174
    assert layer.reserve_companion_memory(vcfg_fp8, 20000, auto=False) == 8192


def test_ced_prefix_replay_and_plan_validation() -> None:
    """Verify prefix_replay_enabled and plan_decoder_replay input validation."""
    vcfg = SimpleNamespace(
        cache_config=SimpleNamespace(
            enable_prefix_caching=True, swa_bounded_replay=True
        ),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1),
    )
    assert ced_mod.prefix_replay_enabled(vcfg)
    assert not ced_mod.prefix_replay_enabled(vcfg, ced_enabled=False)

    assert (
        ced_mod.plan_decoder_replay(
            SimpleNamespace(
                query_start_loc_cpu=None, seq_lens_cpu=None, prompt_lens_cpu=None
            ),
            4,
            8,
            (4, 8),
        )
        is None
    )
    bad_meta = SimpleNamespace(
        query_start_loc_cpu=[0, 10],
        seq_lens_cpu=[4],
        prompt_lens_cpu=[4],
        replay_starts_cpu=None,
    )
    with pytest.raises(ValueError, match="Invalid decoder replay scheduling"):
        ced_mod.plan_decoder_replay(bad_meta, 4, 8, (4, 8))
    bad_q = SimpleNamespace(
        query_start_loc_cpu=[0, 5],
        seq_lens_cpu=[3],
        prompt_lens_cpu=[3],
        replay_starts_cpu=[0],
    )
    with pytest.raises(ValueError, match="Invalid scheduled query"):
        ced_mod.plan_decoder_replay(bad_q, 4, 8, (4, 8))


def test_ced_plan_sync_and_conflict_free_scatter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify CED replay planning, DP synchronization, and gather-based scatter."""
    meta = SimpleNamespace(
        query_start_loc_cpu=torch.tensor([0, 6, 7], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([6, 10], dtype=torch.int32),
        prompt_lens_cpu=torch.tensor([6, 9], dtype=torch.int32),
        replay_starts_cpu=None,
        seq_lens=torch.tensor([6, 10], dtype=torch.int32),
    )
    plan = ced_mod.plan_decoder_replay(
        meta, window=4, num_tokens=8, compile_sizes=[4, 8]
    )
    assert plan is not None and plan.indices == (2, 3, 4, 5, 6)
    assert plan.padded_num_tokens == 8 and not plan.is_identity

    meta_id = SimpleNamespace(
        query_start_loc_cpu=[0, 4],
        seq_lens_cpu=[4],
        prompt_lens_cpu=[4],
        replay_starts_cpu=[0],
    )
    plan_id = ced_mod.plan_decoder_replay(meta_id, 4, 4, [4])
    assert plan_id is not None and plan_id.is_identity

    assert ced_mod.sync_decoder_replay_plan(plan, meta, 8, None) is plan
    s_fast = ced_mod.sync_decoder_replay_plan(
        plan, meta, 4, object(), window=4, prefix_replay=False, compile_sizes=(4, 8)
    )
    assert s_fast is not None and s_fast.padded_num_tokens == 4

    def _fake_all_reduce(buf: torch.Tensor, op: Any = None, group: Any = None) -> None:
        if buf[0].item() == 0:
            buf[0], buf[1] = 1, 8

    monkeypatch.setattr(ced_mod.dist, "all_reduce", _fake_all_reduce)
    s_none = ced_mod.sync_decoder_replay_plan(
        None, meta, 8, object(), prefix_replay=True
    )
    assert isinstance(s_none, DecoderReplayPlan) and s_none.padded_num_tokens == 8
    assert (
        ced_mod.sync_decoder_replay_plan(plan, meta, 8, object(), prefix_replay=True)
        is plan
    )

    monkeypatch.setattr(ced_mod.dist, "all_reduce", lambda buf, **kw: buf.zero_())
    assert (
        ced_mod.sync_decoder_replay_plan(None, meta, 8, object(), prefix_replay=True)
        is None
    )

    inv = build_decoder_inverse_rows((1, 3), padded=4, total=5, device="cpu")
    assert inv.tolist() == [4, 0, 4, 1, 4]
    narrow = torch.arange(4 * 2, dtype=torch.float32).reshape(4, 2) + 1.0
    scattered = ced_mod.scatter_decoder_outputs(narrow, inv)
    assert torch.equal(scattered[0], torch.zeros(2))
    assert torch.equal(scattered[1], narrow[0])
    assert torch.equal(scattered[3], narrow[1])


def test_deepseek_v41_moe_init_and_gate_slicing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify DeepseekV41MoE subclasses TPU DeepseekV4MoE and slices EP gate inputs."""

    def _make_cfg(eplb: bool = False, backend: str = "auto") -> Any:
        return SimpleNamespace(
            parallel_config=SimpleNamespace(enable_eplb=eplb),
            kernel_config=SimpleNamespace(moe_backend=backend),
            model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        )

    with pytest.raises(NotImplementedError, match="EPLB"):
        DeepseekV41MoE(_make_cfg(eplb=True))
    with pytest.raises(NotImplementedError, match="native V4 MoE"):
        DeepseekV41MoE(_make_cfg(backend="deep_gemm_mega_moe"))

    seen_hash_layers: list[int] = []

    def _fake_super_init(self: Any, vllm_config: Any, prefix: str = "") -> None:
        torch.nn.Module.__init__(self)
        seen_hash_layers.append(vllm_config.model_config.hf_config.num_hash_layers)
        self.experts = MagicMock()
        self.experts.gate = object()

    monkeypatch.setattr(moe_mod.DeepseekV4MoE, "__init__", _fake_super_init)
    cfg = _make_cfg()
    cfg.model_config.hf_config.num_hash_layers = 4
    m = DeepseekV41MoE(cfg)
    assert seen_hash_layers == [0]
    assert cfg.model_config.hf_config.num_hash_layers == 4
    assert m.experts.gate is None

    m.gate = lambda x: (x[:, :2], None)
    m.experts.moe_config = SimpleNamespace(
        dp_size=1, pcp_size=1, moe_parallel_config=SimpleNamespace(use_ep=True)
    )
    m.experts.routed_experts = SimpleNamespace(
        quant_method=SimpleNamespace(is_monolithic=True, supports_internal_mk=False),
        custom_routing_function=None,
    )
    m.experts.return_value = torch.ones((4, 8), dtype=torch.bfloat16)
    monkeypatch.setattr(
        moe_mod, "get_ep_group", lambda: SimpleNamespace(world_size=2, rank_in_group=1)
    )
    x = torch.arange(32, dtype=torch.bfloat16).reshape(4, 8)
    assert torch.equal(m._maybe_slice_gate_input(x), x[2:4])
    assert m(x).shape == (4, 8)

    with pytest.raises(ValueError, match="divisible by group size"):
        m._maybe_slice_gate_input(x[:3])

    m.experts.moe_config.dp_size = 2
    monkeypatch.setattr(
        moe_mod, "get_tp_group", lambda: SimpleNamespace(world_size=1, rank_in_group=0)
    )
    assert torch.equal(m._maybe_slice_gate_input(x), x)


def test_native_cache_view_null_block_and_two_bank_nan_isolation(
    mock_cache_ops: None,
) -> None:
    """Verify NativeCacheView enforces null_block=0 isolation across 2-bank pools."""
    del mock_cache_ops

    class _DummyNativeCache(cache_mod.NativeCacheView):
        pass

    # 4 pages total across 2 banks (2 pages per bank), 2 layers per page
    # (page_stride = 8 rows = 4096B, layer 1 at row_offset = 4 rows = 2048B).
    nan_word = int(torch.tensor([0x7FC07FC0], dtype=torch.uint32).view(torch.int32)[0])
    lo = torch.full((16, 128), nan_word, dtype=torch.int32)
    hi = torch.full((16, 128), nan_word, dtype=torch.int32)
    dummy_view = torch.zeros((4, 4, 256), dtype=torch.bfloat16)
    layer1 = _DummyNativeCache()
    layer1.bind_kv_cache(
        dummy_view,
        raw_cache=(lo, hi),
        layout=((4, 4, 256), torch.bfloat16, 4096, 2048),
    )
    pcr = layer1.paged_cache_rows
    assert pcr is not None and pcr.null_block is True

    # pages() must mask both -1 and null_block=0 to -1 even when row_offset > 0,
    # while mapping valid blocks 1 (lo bank) and 3 (hi bank) accurately.
    bt = torch.tensor([[-1, 0, 1, 3]], dtype=torch.int32)
    mapped_pages = pcr.pages(bt)
    assert mapped_pages is not None
    assert mapped_pages.tolist() == [[-1, -1, 3, 7]]

    # Writes targeting null_block=0 (slots 0..3) or negative slots must be dropped,
    # leaving block 0 untouched, whereas block 3 (slots 12..15, hi bank) updates.
    lo_blk0_before = lo[:8].clone()
    vals = torch.full((4, 256), 5.0, dtype=torch.bfloat16)
    cache_mod.write_cache(
        pcr,
        torch.tensor([-1, 0, 3, 13], dtype=torch.int32),
        vals,
        torch.ones(4, dtype=torch.bool),
    )
    assert torch.equal(lo[:8], lo_blk0_before)

    # Gathering from block -1 or null_block=0 must return clean zeros (0 NaNs!),
    # never reading poisoned NaN bit patterns from block 0 or neighboring layer 0.
    gathered = pcr[
        torch.tensor([-1, 0, 3], dtype=torch.int32),
        torch.tensor([0, 2, 1], dtype=torch.int32),
    ]
    assert not torch.isnan(gathered).any()
    assert bool((gathered[0] == 0).all() and (gathered[1] == 0).all())
    assert bool((gathered[2] == 5.0).all())


def test_native_cache_view_zero_blocks_cpu_two_bank_and_tpu_buckets(
    mock_cache_ops: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify NativeCacheView.zero_blocks clears target pages across banks & pools."""
    del mock_cache_ops

    class _DummyNativeCache(cache_mod.NativeCacheView):
        pass

    nan_w = int(torch.tensor([0x7FC07FC0], dtype=torch.uint32).view(torch.int32)[0])
    f32_nan = int(torch.tensor([0x7F800001], dtype=torch.uint32).view(torch.int32)[0])

    # Direct _zero_pools_tpu.__wrapped__ verification for 1-bank and 2-bank pools.
    w_lo = torch.full((4, 128), nan_w, dtype=torch.int32)
    w_hi = torch.full((4, 128), f32_nan, dtype=torch.int32)
    w_single = torch.full((4, 128), nan_w, dtype=torch.int32)
    z_rows = torch.zeros((2, 128), dtype=torch.int32)
    rec_1b = torch.tensor([1, -1], dtype=torch.int32)
    rec_2b = torch.tensor([2, 5], dtype=torch.int32)
    cache_mod._zero_pools_tpu.__wrapped__([w_single], [None], rec_1b, z_rows, 4)
    assert (w_single[1] == 0).all() and (w_single[0] == nan_w).all()
    cache_mod._zero_pools_tpu.__wrapped__([w_lo], [w_hi], rec_2b, z_rows, 4)
    assert (w_lo[2] == 0).all() and (w_hi[1] == 0).all()

    # Unbound NativeCacheView is a safe no-op.
    _DummyNativeCache().zero_blocks([1])

    # 1. Materialized 2-bank raw + rope pools across 2 layer-compact regions.
    #    4 pages total (2 in lo bank, 2 in hi bank), 4 rows/page (2048B stride).
    r0_lo = torch.full((8, 128), nan_w, dtype=torch.int32)
    r0_hi = torch.full((8, 128), f32_nan, dtype=torch.int32)
    r1_lo = torch.full((8, 128), f32_nan, dtype=torch.int32)
    r1_hi = torch.full((8, 128), nan_w, dtype=torch.int32)
    rp_lo = torch.full((2, 128), nan_w, dtype=torch.int32)
    rp_hi = torch.full((2, 128), f32_nan, dtype=torch.int32)

    owner = _DummyNativeCache()
    owner._tpu_zero_groups = (
        ([(r0_lo, r0_hi), (r1_lo, r1_hi)], 4),
        ([(rp_lo, rp_hi)], 4),
    )
    # Null-block-only call is a no-op across all groups.
    owner.zero_blocks([0])
    assert (r0_lo == nan_w).all()
    # Out-of-range and null_block=0 must be ignored; blocks 1 (lo) & 3 (hi) zeroed.
    owner.zero_blocks([-1, 0, 1, 3, 1, 99])
    assert (r0_lo[:4] == nan_w).all() and (r0_lo[4:8] == 0).all()
    assert (r0_hi[:4] == f32_nan).all() and (r0_hi[4:8] == 0).all()
    assert (r1_lo[:4] == f32_nan).all() and (r1_lo[4:8] == 0).all()
    assert (r1_hi[:4] == nan_w).all() and (r1_hi[4:8] == 0).all()
    assert (rp_lo[:1] == nan_w).all() and (rp_lo[1:2] == 0).all()
    assert (rp_hi[:1] == f32_nan).all() and (rp_hi[1:2] == 0).all()

    # 2. Standalone bind_kv_cache fallback (single-bank raw_cache + rope_pool,
    #    BF16 raw_cache without rope_pool, and plain _bound_cache tensor).
    single = _DummyNativeCache()
    s_raw = torch.full((12, 128), nan_w, dtype=torch.int32)
    s_rope = torch.full((3, 128), f32_nan, dtype=torch.int32)
    single.bind_kv_cache(
        torch.zeros((3, 4, 512), dtype=torch.uint8),
        raw_cache=s_raw,
        layout=((3, 4, 512), torch.uint8, 2048, 0),
        rope_pool=s_rope,
    )
    single.zero_blocks([0, 2])
    assert (s_raw[:8] == nan_w).all() and (s_raw[8:12] == 0).all()
    assert (s_rope[:2] == f32_nan).all() and (s_rope[2:3] == 0).all()

    bf16_single = _DummyNativeCache()
    bf16_raw = torch.full((6, 128), nan_w, dtype=torch.int32)
    bf16_single.bind_kv_cache(
        torch.zeros((3, 2, 256), dtype=torch.bfloat16),
        raw_cache=bf16_raw,
        layout=((3, 2, 256), torch.bfloat16, 1024, 0),
    )
    bf16_single.zero_blocks([1])
    assert (bf16_raw[:2] == nan_w).all() and (bf16_raw[2:4] == 0).all()

    plain = _DummyNativeCache()
    p_buf = torch.full((4, 2, 8), float("nan"), dtype=torch.float32)
    plain.kv_cache = p_buf
    plain.zero_blocks([0, 2])
    assert torch.isnan(p_buf[0]).all() and torch.isnan(p_buf[1]).all()
    assert (p_buf[2] == 0).all() and torch.isnan(p_buf[3]).all()

    # 3. TPU path (>256 rows -> 2048 bucket) via monkeypatched _zero_pools_tpu.
    class _FakeTpuPool:
        def __init__(self, rows: int) -> None:
            self.shape = (rows, 128)
            self.device = SimpleNamespace(type="tpu")

    tpu_calls: list[tuple[int, int, int, int, int]] = []
    sync_calls: list[int] = []
    orig_tensor = cache_mod.torch.tensor
    orig_zeros = cache_mod.torch.zeros
    monkeypatch.setattr(
        cache_mod.torch,
        "zeros",
        lambda sh, *, dtype=None, device=None: orig_zeros(sh, dtype=dtype),
    )
    monkeypatch.setattr(
        cache_mod.torch,
        "tensor",
        lambda data, *, dtype=None, device=None: orig_tensor(data, dtype=dtype),
    )
    monkeypatch.setattr(
        cache_mod,
        "_zero_pools_tpu",
        lambda los, his, rec, z, off, page_tokens=16: tpu_calls.append(
            (
                len(los),
                int(rec.shape[0]),
                int((rec >= 0).sum().item()),
                off,
                page_tokens,
            )
        ),
    )
    monkeypatch.setattr(
        cache_mod.utils,
        "synchronize_tensors",
        lambda ts: sync_calls.append(len(ts)),
    )
    tpu_owner = _DummyNativeCache()
    tpu_owner._tpu_zero_groups = (([(_FakeTpuPool(1024), _FakeTpuPool(1024))], 4),)
    # 1 block * 512 rows/block = 512 rows > 256 -> 2048 bucket.
    tpu_owner.zero_blocks([2])
    assert tpu_calls == [(1, 2048, 512, 1024, 512)]
    assert sync_calls == [2]


@pytest.mark.tpu_test
def test_tpu_paged_cache_rows_and_scatter() -> None:  # pragma: no cover
    """Execute PagedCacheRows (with donated buffer rebinding) and scatter on TPU."""
    import torch_tpu  # noqa: F401

    dev = torch.device("tpu:0")
    raw = torch.zeros((8, 128), dtype=torch.int32, device=dev)
    rope = torch.zeros((2, 128), dtype=torch.int32, device=dev)
    pcr = PagedCacheRows(raw, ((2, 4, 512), torch.uint8, 2048, 0), rope=rope)
    lat = torch.randn(4, 512, dtype=torch.bfloat16, device=dev)
    slots = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=dev)
    pcr.write(slots, lat)
    pcr.write(torch.tensor([4, 5, 6, 7], dtype=torch.int32, device=dev), lat)
    raw_cpu = pcr.raw.cpu()
    assert raw_cpu[:4].abs().sum().item() > 0
    assert raw_cpu[4:8].abs().sum().item() > 0

    bf16_raw = torch.zeros((8, 128), dtype=torch.int32, device=dev)
    bf16_pcr = PagedCacheRows(bf16_raw, ((2, 4, 256), torch.bfloat16, 2048, 0))
    bf16_vals = torch.ones((2, 256), dtype=torch.bfloat16, device=dev)
    bf16_slots = torch.tensor([0, 1], dtype=torch.int32, device=dev)
    bf16_valid = torch.tensor([True, True], dtype=torch.bool, device=dev)
    bf16_pcr.write(
        bf16_slots[:1], torch.full((1, 256), 7.0, dtype=torch.bfloat16, device=dev)
    )

    def _read_then_write(
        vals_in: torch.Tensor, s_in: torch.Tensor, v_in: torch.Tensor
    ) -> torch.Tensor:
        pre = bf16_pcr[s_in[:1], s_in[:1]]
        cache_mod.write_cache(bf16_pcr, s_in, vals_in, v_in, after_read=True)
        return pre

    torch._dynamo.reset()
    compiled_rtw = torch.compile(
        _read_then_write, backend="tpu", fullgraph=True, dynamic=False
    )
    pre_out = compiled_rtw(bf16_vals, bf16_slots, bf16_valid)
    assert bool(torch.all(pre_out.cpu() == 7.0))
    assert bool(torch.all(bf16_pcr[bf16_slots[:1], bf16_slots[:1]].cpu() == 1.0))
    assert bf16_pcr._last_read is None

    inv = build_decoder_inverse_rows((0, 2), padded=2, total=4, device=dev)
    narrow = torch.ones((2, 16), dtype=torch.bfloat16, device=dev)
    out = ced_mod.scatter_decoder_outputs(narrow, inv).cpu()
    assert out.shape == (4, 16)
    assert out[0].sum().item() == 16.0 and out[1].sum().item() == 0.0
