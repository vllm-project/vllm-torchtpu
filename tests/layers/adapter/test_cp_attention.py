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
"""CPU tests for DCP layout propagation and kernel registry isolation.

Real TPU pass correctness is covered by test_attention_dcp_hnd.py. Here the
kernel builder and collectives are replaced so both adapter entry points and
the registry can be checked together, including reuse across layers/layouts.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.config import (
    DeviceConfig,
    VllmConfig,
    get_current_vllm_config,
    set_current_vllm_config,
)
from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

from vllm_torchtpu.distributed.dcp_comm import (
    cp_lse_ag_out_rs,
    cp_lse_ag_out_scale,
)
from vllm_torchtpu.kernels.experimental.batched_rpa.configs import (
    AttentionScope,
    KVLayout,
)
from vllm_torchtpu.layers.adapter import attention, cp_attention
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)


@pytest.fixture
def cpu_vllm_config_context():
    # CPU CI does not register torch's TPU device type. These tests only
    # exercise layout propagation with CPU tensors and mocked kernels.
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    resolve_kv_cache_layout(config, [["LBNHC", "LBHNC"]])
    with set_current_vllm_config(config):
        yield


def _impl(cls=attention.PallasBatchedRPAAttentionBackendImpl, head_size=256):
    return cls(
        num_heads=8,
        head_size=head_size,
        scale=head_size**-0.5,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="fp8",
    )


def _config():
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=8,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=2,
        )
    )


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("compiled", [False, True])
def test_dcp_layout_survives_prebuild_forward_and_registry_reuse(
    monkeypatch, rank, compiled, cpu_vllm_config_context
):
    built, calls = [], []

    def build_op(_name, fn):
        built.append(fn)

        def kernel(_cache, query, *args):
            scope = fn.keywords["attention_scope"]
            calls.append((fn.keywords["kv_layout"], scope, query.shape))
            value = 1 if scope == AttentionScope.CACHE_ONLY else 3
            return torch.full_like(query, value), torch.zeros(query.shape[:2])

        return kernel

    group = SimpleNamespace(
        world_size=2,
        rank_in_group=rank,
        all_gather=lambda tensor, dim: torch.cat([tensor, tensor], dim=dim),
        reduce_scatter=lambda tensor, dim: (2 * tensor).chunk(2, dim)[rank],
    )
    monkeypatch.setattr(attention, "_get_dcp_group", lambda: group)
    monkeypatch.setattr(cp_attention, "get_dcp_group", lambda: group)
    monkeypatch.setattr(cp_attention, "_DCP_KERNEL_REGISTRY", {})
    monkeypatch.setattr(cp_attention, "_build_dcp_kernel_op", build_op)
    monkeypatch.setattr(cp_attention, "synchronize_tensors", lambda *a, **k: None)
    layer = SimpleNamespace(_k_scale_float=0.5, _v_scale_float=0.25)
    implementations = []
    # The third layer must reuse NHD's ops after HND has populated the registry.
    for layout in ("LBNHC", "LBHNC", "LBNHC"):
        get_current_vllm_config().cache_config.kv_cache_layout = layout
        impl = _impl()
        with set_vllm_model_wrapper_context(mesh=None, vllm_config=_config()):
            impl.initialize_kernel(layer)
        implementations.append(impl)

    assert len(built) == 4  # Two layouts, two scopes; shared by matching layers.
    assert (
        implementations[0].rpa_dcp_cache_kernel
        is implementations[2].rpa_dcp_cache_kernel
    )
    assert (
        implementations[0].rpa_dcp_new_kernel is implementations[2].rpa_dcp_new_kernel
    )
    assert (
        implementations[0].rpa_dcp_cache_kernel
        is not implementations[1].rpa_dcp_cache_kernel
    )
    assert (
        implementations[0].rpa_dcp_new_kernel
        is not implementations[1].rpa_dcp_new_kernel
    )

    metadata = SimpleNamespace(
        seq_lens=torch.tensor([2, 3, 4], dtype=torch.int32),
        block_tables=torch.zeros(3, 2, dtype=torch.int32),
        query_start_loc=torch.arange(4, dtype=torch.int32),
        request_distribution=torch.tensor([3, 3, 3], dtype=torch.int32),
    )
    query = torch.ones(3, 8, 256)
    key = torch.ones(3, 1, 256)
    for impl, expected_layout in zip(
        implementations,
        (
            KVLayout.HEAD_ALONG_SUBLANE,
            KVLayout.SEQ_ALONG_LANE,
            KVLayout.HEAD_ALONG_SUBLANE,
        ),
    ):
        # The per-layer layout must survive even if the current config changes.
        get_current_vllm_config().cache_config.kv_cache_layout = "LBHNC"
        forward = impl._run_dcp_forward
        if compiled:
            forward = torch.compile(forward, backend="eager", fullgraph=True)
        output = forward(layer, query, key, key, torch.empty(1), metadata)
        assert calls[-2:] == [
            (expected_layout, AttentionScope.CACHE_ONLY, (3, 16, 256)),
            (expected_layout, AttentionScope.NEW_TOKENS_ONLY, (3, 8, 256)),
        ]
        # Two cached partials (value=1, weight=1 each) and one new partial
        # (value=3, weight=1), including both production LSE merges.
        torch.testing.assert_close(output, torch.full_like(query, 5 / 3))
    assert len(built) == 4  # Forward must reuse the prebuilt ops for each layout.
    assert all(type(fn.keywords["kv_layout"]) is KVLayout for fn in built)


def test_dcp_kernel_default_layout_reuses_explicit_nhd(monkeypatch):
    monkeypatch.setattr(cp_attention, "_DCP_KERNEL_REGISTRY", {})
    monkeypatch.setattr(cp_attention, "_build_dcp_kernel_op", lambda name, fn: fn)
    kwargs = dict(
        sliding_window=None,
        sm_scale=0.0625,
        logits_soft_cap=None,
        q_scale=None,
        k_scale=None,
        v_scale=None,
        cp_group_size=2,
        cp_rank=0,
    )
    implicit = cp_attention.build_dcp_kernels(**kwargs)
    explicit = cp_attention.build_dcp_kernels(
        **kwargs, kv_layout=KVLayout.HEAD_ALONG_SUBLANE
    )
    assert all(a is b for a, b in zip(implicit, explicit))


@pytest.mark.parametrize(
    "cls,head_size",
    [
        (attention.PallasAttentionBackendImpl, 256),
        (attention.PallasBatchedRPAAttentionBackendImpl, 64),
    ],
)
def test_dcp_hnd_rejects_backends_that_allocate_nhd(
    cls, head_size, cpu_vllm_config_context
):
    get_current_vllm_config().cache_config.kv_cache_layout = "LBHNC"
    impl = _impl(cls, head_size)
    layer = SimpleNamespace(_k_scale_float=0.5, _v_scale_float=0.25)
    with (
        set_vllm_model_wrapper_context(mesh=None, vllm_config=_config()),
        pytest.raises(NotImplementedError, match="DCP with HND requires"),
    ):
        impl.initialize_kernel(layer)


@pytest.mark.parametrize("dcp_size", [2, 4])
@pytest.mark.parametrize("include_inf_and_nan", [False, True])
def test_cp_lse_ag_out_scale_and_merge_dcp_new_into_ctx_parity(
    dcp_size, include_inf_and_nan
):
    seq_tokens = 3
    heads_per_rank = 4
    head_dim = 64
    group_heads = heads_per_rank * dcp_size

    rng = torch.Generator().manual_seed(42)
    ctx_outputs = [
        torch.randn(
            seq_tokens, group_heads, head_dim, generator=rng, dtype=torch.float32
        )
        for _ in range(dcp_size)
    ]
    ctx_lses = [
        torch.randn(seq_tokens, group_heads, generator=rng, dtype=torch.float32)
        for _ in range(dcp_size)
    ]
    new_outputs = [
        torch.randn(
            seq_tokens, heads_per_rank, head_dim, generator=rng, dtype=torch.float32
        )
        for _ in range(dcp_size)
    ]
    new_lses = [
        torch.randn(seq_tokens, heads_per_rank, generator=rng, dtype=torch.float32)
        for _ in range(dcp_size)
    ]

    if include_inf_and_nan:
        ctx_lses[0].fill_(-float("inf"))
        ctx_outputs[0].fill_(float("nan"))

    class MockGroup:
        def __init__(self, rank_idx, world_size):
            self.rank_in_group = rank_idx
            self.world_size = world_size

        def all_gather(self, tensor, dim=0):
            tensors_to_gather = [
                ctx_lses[idx].unsqueeze(0) for idx in range(self.world_size)
            ]
            return torch.cat(tensors_to_gather, dim=dim)

        def reduce_scatter(self, tensor, dim=1):
            summed = torch.zeros_like(tensor)
            for other_idx in range(self.world_size):
                other_lse = ctx_lses[other_idx]
                all_lses = torch.stack(ctx_lses, dim=0)
                merged_lse = torch.logsumexp(all_lses, dim=0)
                other_weight = torch.exp(other_lse - merged_lse).unsqueeze(-1)
                other_weighted = ctx_outputs[other_idx] * other_weight
                other_weighted = torch.where(
                    other_weight > 0, other_weighted, torch.zeros_like(other_weighted)
                )
                summed = summed + other_weighted
            return summed.chunk(self.world_size, dim=dim)[self.rank_in_group]

    baseline_rank_outputs = []
    for rank_idx in range(dcp_size):
        mock_grp = MockGroup(rank_idx, dcp_size)
        ctx_rs, lse_rs = cp_lse_ag_out_rs(
            ctx_outputs[rank_idx], ctx_lses[rank_idx], mock_grp, return_lse=True
        )
        merged_out, _ = cp_attention.merge_attn_results(
            ctx_rs, lse_rs, new_outputs[rank_idx], new_lses[rank_idx]
        )
        baseline_rank_outputs.append(merged_out)

    baseline_full = torch.cat(baseline_rank_outputs, dim=1)

    class MockScaleGroup:
        def __init__(self, rank_idx, world_size, gathered_lses):
            self.rank_in_group = rank_idx
            self.world_size = world_size
            self.gathered_lses = gathered_lses

        def all_gather(self, tensor, dim=0):
            return torch.cat(self.gathered_lses, dim=dim)

    merged_ctx_outs = []
    merged_ctx_lses = []
    for rank_idx in range(dcp_size):
        merged_out, merged_lse = cp_attention.merge_dcp_new_into_ctx(
            ctx_outputs[rank_idx],
            ctx_lses[rank_idx],
            new_outputs[rank_idx],
            new_lses[rank_idx],
            dcp_rank=rank_idx,
            dcp_world_size=dcp_size,
        )
        merged_ctx_outs.append(merged_out)
        merged_ctx_lses.append(merged_lse)

    gathered_lses = [lse.unsqueeze(0) for lse in merged_ctx_lses]

    dup_rank_outputs = []
    for rank_idx in range(dcp_size):
        mock_scale_grp = MockScaleGroup(rank_idx, dcp_size, gathered_lses)
        scaled = cp_lse_ag_out_scale(
            merged_ctx_outs[rank_idx],
            merged_ctx_lses[rank_idx],
            mock_scale_grp,
            return_lse=False,
        )
        dup_rank_outputs.append(scaled)

    dup_summed = sum(dup_rank_outputs)
    torch.testing.assert_close(dup_summed, baseline_full, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("pre_gathered_q", [False, True])
def test_dcp_duplicate_o_proj_forward_skips_reduce_scatter_and_matches_baseline(
    monkeypatch, compiled, pre_gathered_q, cpu_vllm_config_context
):
    seq_tokens = 2
    own_heads = 4
    head_dim = 128
    dcp_size = 2
    group_heads = own_heads * dcp_size

    rng = torch.Generator().manual_seed(20261014)
    base_queries = [
        torch.randn(seq_tokens, own_heads, head_dim, generator=rng)
        for _ in range(dcp_size)
    ]
    full_query = torch.cat(base_queries, dim=1)
    ctx_outputs = [
        torch.randn(seq_tokens, group_heads, head_dim, generator=rng)
        for _ in range(dcp_size)
    ]
    ctx_lses = [
        torch.randn(seq_tokens, group_heads, generator=rng) for _ in range(dcp_size)
    ]
    new_outputs = [
        torch.randn(seq_tokens, own_heads, head_dim, generator=rng)
        for _ in range(dcp_size)
    ]
    new_lses = [
        torch.randn(seq_tokens, own_heads, generator=rng) for _ in range(dcp_size)
    ]

    def build_kernels_for_rank(**kwargs):
        cp_rank = kwargs["cp_rank"]

        def fake_cache_kernel(_cache, query_tensor, *args):
            assert query_tensor.shape == full_query.shape
            return (
                ctx_outputs[cp_rank] + (query_tensor - full_query),
                ctx_lses[cp_rank],
            )

        def fake_new_kernel(_cache, query_tensor, *args):
            assert query_tensor.shape == base_queries[cp_rank].shape
            return (
                new_outputs[cp_rank] + (query_tensor - base_queries[cp_rank]),
                new_lses[cp_rank],
            )

        return fake_cache_kernel, fake_new_kernel

    monkeypatch.setattr(cp_attention, "_DCP_KERNEL_REGISTRY", {})
    monkeypatch.setattr(cp_attention, "build_dcp_kernels", build_kernels_for_rank)
    monkeypatch.setattr(
        cp_attention, "synchronize_tensors", lambda *args, **kwargs: None
    )

    layer = SimpleNamespace(_k_scale_float=None, _v_scale_float=None)
    metadata = SimpleNamespace(
        seq_lens=torch.tensor([2, 2], dtype=torch.int32),
        block_tables=torch.zeros(2, 2, dtype=torch.int32),
        query_start_loc=torch.arange(3, dtype=torch.int32),
        request_distribution=torch.tensor([2, 2, 2], dtype=torch.int32),
    )
    kv_cache = torch.empty(1)

    class BaselineGroup:
        def __init__(self, rank_idx):
            self.rank_in_group = rank_idx
            self.world_size = dcp_size

        def all_gather(self, tensor, dim=0):
            if dim == 1:
                return full_query
            return torch.stack(ctx_lses, dim=0)

        def reduce_scatter(self, tensor, dim=1):
            merged_lse = torch.logsumexp(torch.stack(ctx_lses, dim=0), dim=0)
            summed = torch.zeros_like(tensor)
            for idx in range(self.world_size):
                w = torch.exp(ctx_lses[idx] - merged_lse).unsqueeze(-1)
                summed = summed + ctx_outputs[idx] * w
            return summed.chunk(self.world_size, dim=dim)[self.rank_in_group]

    baseline_outs = []
    for rank_idx in range(dcp_size):
        grp = BaselineGroup(rank_idx)
        monkeypatch.setattr(cp_attention, "get_dcp_group", lambda g=grp: g)
        out = cp_attention.forward_with_dcp(
            layer=layer,
            query=base_queries[rank_idx],
            key=torch.ones(seq_tokens, 1, head_dim),
            value=torch.ones(seq_tokens, 1, head_dim),
            kv_cache=kv_cache,
            attn_metadata=metadata,
            sliding_window=None,
            sm_scale=0.125,
            logits_soft_cap=None,
            kv_cache_quantized_dtype=None,
            dcp_world_size=dcp_size,
            dcp_rank=rank_idx,
            dcp_duplicate_o_proj=False,
        )
        baseline_outs.append(out)

    baseline_concat = torch.cat(baseline_outs, dim=1)

    expected_merged_lses = []
    for rank_idx in range(dcp_size):
        _, merged_lse = cp_attention.merge_dcp_new_into_ctx(
            ctx_outputs[rank_idx],
            ctx_lses[rank_idx],
            new_outputs[rank_idx],
            new_lses[rank_idx],
            dcp_rank=rank_idx,
            dcp_world_size=dcp_size,
        )
        expected_merged_lses.append(merged_lse)

    class GroupTracker:
        def __init__(self, rank_idx):
            self.rank_in_group = rank_idx
            self.world_size = dcp_size
            self.reduce_scatter_called = False
            self.q_all_gather_called = False

        def all_gather(self, tensor, dim=0):
            if dim == 1:
                self.q_all_gather_called = True
                return full_query
            expected_own = expected_merged_lses[self.rank_in_group].unsqueeze(0)
            assert tensor.shape == expected_own.shape
            return torch.stack(expected_merged_lses, dim=0) + (tensor - expected_own)

        def reduce_scatter(self, tensor, dim=1):
            self.reduce_scatter_called = True
            raise AssertionError(
                "reduce_scatter must not be called when dcp_duplicate_o_proj=True"
            )

    dup_outs = []
    for rank_idx in range(dcp_size):
        tracker = GroupTracker(rank_idx)
        monkeypatch.setattr(cp_attention, "get_dcp_group", lambda t=tracker: t)
        forward_fn = cp_attention.forward_with_dcp
        if compiled:
            forward_fn = torch.compile(forward_fn, backend="eager", fullgraph=True)

        if pre_gathered_q:
            query_arg = full_query
            q_dcp_replicated = True
        else:
            query_arg = base_queries[rank_idx]
            q_dcp_replicated = False

        dup_out = forward_fn(
            layer=layer,
            query=query_arg,
            key=torch.ones(seq_tokens, 1, head_dim),
            value=torch.ones(seq_tokens, 1, head_dim),
            kv_cache=kv_cache,
            attn_metadata=metadata,
            sliding_window=None,
            sm_scale=0.125,
            logits_soft_cap=None,
            kv_cache_quantized_dtype=None,
            dcp_world_size=dcp_size,
            dcp_rank=rank_idx,
            q_dcp_replicated=q_dcp_replicated,
            dcp_duplicate_o_proj=True,
        )
        assert not tracker.reduce_scatter_called
        assert tracker.q_all_gather_called is (not pre_gathered_q)
        dup_outs.append(dup_out)

    dup_sum = sum(dup_outs)
    torch.testing.assert_close(dup_sum, baseline_concat, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dcp_size", [2, 8])
def test_dcp_group_row_parallel_linear_weight_loading_and_forward(
    monkeypatch, dcp_size
):
    from vllm.model_executor.parameter import BlockQuantScaleParameter

    tp_world_size = 8
    seq_tokens = 8
    hidden_size = 512
    input_size = 512
    eff_tp_size = tp_world_size // dcp_size
    part_input_size = input_size // eff_tp_size

    rng = torch.Generator().manual_seed(1234 + dcp_size)
    weight_full = torch.randn(hidden_size, input_size, generator=rng)
    bias_full = torch.randn(hidden_size, generator=rng)
    block_size = 64
    scale_full = torch.randn(
        hidden_size // block_size, input_size // block_size, generator=rng
    )

    cfg = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp_world_size,
            decode_context_parallel_size=dcp_size,
        )
    )
    monkeypatch.setattr(cp_attention, "get_current_vllm_config_or_none", lambda: cfg)

    layers = []
    sp_layers = []
    for rank_idx in range(tp_world_size):
        monkeypatch.setattr(
            cp_attention, "get_tensor_model_parallel_rank", lambda r=rank_idx: r
        )
        monkeypatch.setattr(
            cp_attention, "get_tensor_model_parallel_world_size", lambda: tp_world_size
        )
        monkeypatch.setattr(
            "vllm.distributed.parallel_state.get_tensor_model_parallel_rank",
            lambda r=rank_idx: r,
        )
        monkeypatch.setattr(
            "vllm.distributed.parallel_state.get_tensor_model_parallel_world_size",
            lambda: tp_world_size,
        )
        monkeypatch.setattr(
            "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
            lambda r=rank_idx: r,
        )
        monkeypatch.setattr(
            "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
            lambda: tp_world_size,
        )

        if rank_idx == 0:
            with pytest.raises(ValueError, match="requires input_is_parallel=True"):
                cp_attention.DCPGroupRowParallelLinear(
                    input_size=input_size,
                    output_size=hidden_size,
                    input_is_parallel=False,
                    params_dtype=torch.float32,
                )

        layer = cp_attention.DCPGroupRowParallelLinear(
            input_size=input_size,
            output_size=hidden_size,
            bias=True,
            params_dtype=torch.float32,
        )
        assert layer.group_size == dcp_size
        assert layer.tp_rank == rank_idx // dcp_size
        assert layer.tp_size == eff_tp_size
        assert layer.input_size_per_partition == part_input_size

        layer.weight_loader_v2(layer.weight, weight_full)
        expected_shard = weight_full.narrow(
            1, (rank_idx // dcp_size) * part_input_size, part_input_size
        )
        torch.testing.assert_close(layer.weight.data, expected_shard)
        layer.weight_loader(layer.bias, bias_full)
        torch.testing.assert_close(layer.bias.data, bias_full)

        block_scale = BlockQuantScaleParameter(
            data=torch.empty(
                hidden_size // block_size,
                part_input_size // block_size,
                dtype=torch.float32,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=layer.weight_loader_v2,
        )
        layer.register_parameter("weight_scale_inv", block_scale)
        layer.update_param_tp_status()
        layer.weight_loader_v2(block_scale, scale_full)
        expected_scale_shard = scale_full.narrow(
            1,
            (rank_idx // dcp_size) * (part_input_size // block_size),
            part_input_size // block_size,
        )
        torch.testing.assert_close(block_scale.data, expected_scale_shard)
        delattr(layer, "weight_scale_inv")

        layers.append(layer)

        sp_layer = cp_attention.DCPGroupRowParallelLinear(
            input_size=input_size,
            output_size=hidden_size,
            bias=False,
            reduce_results=False,
            params_dtype=torch.float32,
        )
        sp_layer.weight_loader(sp_layer.weight, weight_full)
        sp_layers.append(sp_layer)

    full_attn_out = torch.randn(seq_tokens, input_size, generator=rng)
    full_gate = torch.randn(seq_tokens, input_size, generator=rng)
    expected_full_out = (
        full_attn_out * torch.sigmoid(full_gate)
    ) @ weight_full.t() + bias_full
    expected_sp_out = (full_attn_out * torch.sigmoid(full_gate)) @ weight_full.t()

    rank_inputs = []
    for g in range(eff_tp_size):
        group_attn = full_attn_out.narrow(1, g * part_input_size, part_input_size)
        group_gate = full_gate.narrow(1, g * part_input_size, part_input_size)
        partials = [
            torch.randn(seq_tokens, part_input_size, generator=rng)
            for _ in range(dcp_size - 1)
        ]
        last_partial = group_attn - sum(partials)
        partials.append(last_partial)
        for p in partials:
            rank_inputs.append(p * torch.sigmoid(group_gate))

    unreduced_parallels = [
        layers[r].quant_method.apply(layers[r], rank_inputs[r], None)
        for r in range(tp_world_size)
    ]
    total_reduced = sum(unreduced_parallels)
    monkeypatch.setattr(
        cp_attention,
        "tensor_model_parallel_all_reduce",
        lambda x: total_reduced,
    )
    for r in range(tp_world_size):
        out_r, bias_r = layers[r].forward(rank_inputs[r])
        assert bias_r is None
        torch.testing.assert_close(out_r, expected_full_out, atol=1e-4, rtol=1e-4)

    sp_outs = [sp_layers[r].forward(rank_inputs[r])[0] for r in range(tp_world_size)]
    sp_summed = sum(sp_outs)
    for r in range(tp_world_size):
        sp_rs_r = sp_summed.chunk(tp_world_size, dim=0)[r]
        torch.testing.assert_close(
            sp_rs_r,
            expected_sp_out.chunk(tp_world_size, dim=0)[r],
            atol=1e-4,
            rtol=1e-4,
        )


def test_qwen3_next_attention_dcp_duplicate_o_proj_patch_end_to_end(monkeypatch):
    from vllm.model_executor.models import qwen3_next

    from vllm_torchtpu.models.vllm import qwen3_next_patch

    cfg_enabled = SimpleNamespace(
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=2,
            prefill_context_parallel_size=1,
            dcp_duplicate_o_proj=True,
        )
    )
    assert cp_attention.is_dcp_duplicate_o_proj_enabled(cfg_enabled)

    cfg_disabled = SimpleNamespace(
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=2,
            prefill_context_parallel_size=2,
            dcp_duplicate_o_proj=True,
        )
    )
    assert not cp_attention.is_dcp_duplicate_o_proj_enabled(cfg_disabled)

    qwen3_next_patch.maybe_patch_qwen3_next()
    monkeypatch.setattr(
        qwen3_next_patch, "is_dcp_duplicate_o_proj_enabled", lambda: True
    )
    recorded_linear_cls = []

    def fake_upstream_init(self, *args, **kwargs):
        recorded_linear_cls.append(qwen3_next.RowParallelLinear)
        self.o_proj = SimpleNamespace(orep_active=True, group_size=2)
        self.attn = SimpleNamespace(impl=SimpleNamespace())
        self.use_fused_qk_norm_rope_gate = True

    monkeypatch.setattr(qwen3_next_patch, "UPSTREAM_ATTENTION_INIT", fake_upstream_init)
    orig_row_cls = qwen3_next.RowParallelLinear
    dummy_attn = SimpleNamespace()
    qwen3_next_patch.patched_attention_init(dummy_attn)
    assert recorded_linear_cls == [cp_attention.DCPGroupRowParallelLinear]
    assert qwen3_next.RowParallelLinear is orig_row_cls
    assert dummy_attn.dcp_duplicate_o_proj is True
    assert dummy_attn.attn.dcp_duplicate_o_proj is True
    assert dummy_attn.attn.impl.dcp_duplicate_o_proj is True
    assert dummy_attn.use_fused_qk_norm_rope_gate is False

    seq_tokens = 3
    num_heads = 4
    head_dim = 64
    dcp_size = 2
    rng = torch.Generator().manual_seed(9876)
    q_by_rank = [
        torch.randn(seq_tokens, num_heads * head_dim, generator=rng)
        for _ in range(dcp_size)
    ]
    k_by_rank = [
        torch.randn(seq_tokens, head_dim, generator=rng) for _ in range(dcp_size)
    ]
    v_by_rank = [
        torch.randn(seq_tokens, head_dim, generator=rng) for _ in range(dcp_size)
    ]
    gate_by_rank = [
        torch.randn(seq_tokens, num_heads * head_dim, generator=rng)
        for _ in range(dcp_size)
    ]

    expected_q_full = torch.cat(
        [q_r.view(seq_tokens, num_heads, head_dim) for q_r in q_by_rank], dim=1
    ).reshape(seq_tokens, num_heads * dcp_size * head_dim)
    expected_gate_full = torch.cat(
        [g_r.view(seq_tokens, num_heads, head_dim) for g_r in gate_by_rank], dim=1
    ).reshape(seq_tokens, num_heads * dcp_size * head_dim)

    for rank_idx in range(dcp_size):
        attn_layer = SimpleNamespace(
            num_heads=num_heads,
            head_dim=head_dim,
            dcp_duplicate_o_proj=True,
        )

        def all_gather_qg(tensor, dim=1):
            assert dim == 1
            if tensor.shape[0] == 2 * seq_tokens:
                t0 = torch.cat(
                    [
                        q_by_rank[0].view(seq_tokens, num_heads, head_dim),
                        gate_by_rank[0].view(seq_tokens, num_heads, head_dim),
                    ],
                    dim=0,
                )
                t1 = torch.cat(
                    [
                        q_by_rank[1].view(seq_tokens, num_heads, head_dim),
                        gate_by_rank[1].view(seq_tokens, num_heads, head_dim),
                    ],
                    dim=0,
                )
                return torch.cat([t0, t1], dim=1)
            return torch.cat(
                [
                    q_by_rank[0].view(seq_tokens, num_heads, head_dim),
                    q_by_rank[1].view(seq_tokens, num_heads, head_dim),
                ],
                dim=1,
            )

        dcp_group = SimpleNamespace(
            world_size=dcp_size,
            rank_in_group=rank_idx,
            all_gather=all_gather_qg,
        )
        monkeypatch.setattr(qwen3_next_patch, "get_dcp_group", lambda g=dcp_group: g)
        monkeypatch.setattr(
            qwen3_next_patch,
            "UPSTREAM_PROJECT_QKV_GATE",
            lambda self, qkv, positions, r=rank_idx: (
                q_by_rank[r],
                k_by_rank[r],
                v_by_rank[r],
                gate_by_rank[r],
            ),
        )
        q_out, k_out, v_out, gate_out = qwen3_next_patch.patched_project_qkv_gate(
            attn_layer, torch.empty(1), torch.empty(1)
        )
        assert q_out.is_contiguous()
        assert gate_out.is_contiguous()
        torch.testing.assert_close(q_out, expected_q_full)
        torch.testing.assert_close(gate_out, expected_gate_full)
        torch.testing.assert_close(k_out, k_by_rank[rank_idx])
        torch.testing.assert_close(v_out, v_by_rank[rank_idx])

        monkeypatch.setattr(
            qwen3_next_patch,
            "UPSTREAM_PROJECT_QKV_GATE",
            lambda self, qkv, positions, r=rank_idx: (
                q_by_rank[r],
                k_by_rank[r],
                v_by_rank[r],
                None,
            ),
        )
        q_ng, _, _, gate_ng = qwen3_next_patch.patched_project_qkv_gate(
            attn_layer, torch.empty(1), torch.empty(1)
        )
        assert gate_ng is None
        assert q_ng.is_contiguous()
        torch.testing.assert_close(q_ng, expected_q_full)

    partial_attn_0 = torch.randn(
        seq_tokens, num_heads * dcp_size * head_dim, generator=rng
    )
    partial_attn_1 = torch.randn(
        seq_tokens, num_heads * dcp_size * head_dim, generator=rng
    )
    full_attn_group = partial_attn_0 + partial_attn_1
    w_o_group = torch.randn(128, num_heads * dcp_size * head_dim, generator=rng)
    expected_group_proj = (full_attn_group * torch.sigmoid(expected_gate_full)) @ (
        w_o_group.t()
    )

    rank_proj_outs = []
    for rank_idx, partial_attn in enumerate([partial_attn_0, partial_attn_1]):
        o_proj_fn = lambda item: (item @ w_o_group.t(), None)  # noqa: E731
        o_proj_fn.group_size = dcp_size
        o_proj_fn.orep_active = True
        attn_layer = SimpleNamespace(
            num_heads=num_heads,
            head_dim=head_dim,
            dcp_duplicate_o_proj=True,
            qkv_proj=lambda hidden: (hidden, None),
            _project_qkv_gate=lambda qkv, pos, r=rank_idx: (
                expected_q_full,
                k_by_rank[r],
                v_by_rank[r],
                expected_gate_full,
            ),
            attn=lambda q, k, v, output_shape=None, p=partial_attn: (
                p.view(output_shape)
            ),
            o_proj=o_proj_fn,
        )
        out_r = qwen3_next_patch.patched_attention_forward(
            attn_layer,
            torch.zeros(seq_tokens, dtype=torch.int32),
            torch.randn(seq_tokens, 128),
        )
        rank_proj_outs.append(out_r)

    torch.testing.assert_close(
        sum(rank_proj_outs), expected_group_proj, atol=1e-4, rtol=1e-4
    )
