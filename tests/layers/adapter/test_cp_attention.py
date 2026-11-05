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


@pytest.mark.parametrize(
    "dcp_size,pre_gathered_q,include_inf",
    [(2, False, False), (2, True, True), (4, True, False)],
)
def test_cp_lse_ag_out_scale_and_forward_with_dcp_parity(
    monkeypatch, dcp_size, pre_gathered_q, include_inf, cpu_vllm_config_context
):
    tokens, own_heads, dim = 3, 4, 64
    group_heads = own_heads * dcp_size
    rng = torch.Generator().manual_seed(42 + dcp_size)
    queries = [
        torch.randn(tokens, own_heads, dim, generator=rng) for _ in range(dcp_size)
    ]
    full_q = torch.cat(queries, dim=1)
    ctx_outs = [
        torch.randn(tokens, group_heads, dim, generator=rng) for _ in range(dcp_size)
    ]
    ctx_lses = [
        torch.randn(tokens, group_heads, generator=rng) for _ in range(dcp_size)
    ]
    new_outs = [
        torch.randn(tokens, own_heads, dim, generator=rng) for _ in range(dcp_size)
    ]
    new_lses = [torch.randn(tokens, own_heads, generator=rng) for _ in range(dcp_size)]
    if include_inf:
        ctx_lses[0].fill_(-float("inf"))
        ctx_outs[0].fill_(float("nan"))

    monkeypatch.setattr(cp_attention, "_DCP_KERNEL_REGISTRY", {})
    monkeypatch.setattr(
        cp_attention,
        "build_dcp_kernels",
        lambda **kw: (
            lambda _c, q, *_a, r=kw["cp_rank"]: (
                ctx_outs[r] + (q - full_q),
                ctx_lses[r],
            ),
            lambda _c, q, *_a, r=kw["cp_rank"]: (
                new_outs[r] + (q - queries[r]),
                new_lses[r],
            ),
        ),
    )
    monkeypatch.setattr(cp_attention, "synchronize_tensors", lambda *a, **k: None)
    common_kw = dict(
        layer=SimpleNamespace(_k_scale_float=None, _v_scale_float=None),
        key=torch.ones(tokens, 1, dim),
        value=torch.ones(tokens, 1, dim),
        kv_cache=torch.empty(1),
        attn_metadata=SimpleNamespace(
            seq_lens=torch.full((tokens,), 2, dtype=torch.int32),
            block_tables=torch.zeros(tokens, 2, dtype=torch.int32),
            query_start_loc=torch.arange(tokens + 1, dtype=torch.int32),
            request_distribution=torch.tensor([tokens] * 3, dtype=torch.int32),
        ),
        sliding_window=None,
        sm_scale=0.125,
        logits_soft_cap=None,
        kv_cache_quantized_dtype=None,
        dcp_world_size=dcp_size,
    )
    ctx_lse_stack = torch.stack(ctx_lses, dim=0)
    ctx_global_lse = torch.logsumexp(ctx_lse_stack, dim=0)
    rs_total = sum(
        torch.where(
            torch.exp(ctx_lses[i] - ctx_global_lse).unsqueeze(-1) > 0,
            ctx_outs[i] * torch.exp(ctx_lses[i] - ctx_global_lse).unsqueeze(-1),
            0.0,
        )
        for i in range(dcp_size)
    )

    base_outs = []
    for r in range(dcp_size):
        grp = SimpleNamespace(
            rank_in_group=r,
            world_size=dcp_size,
            all_gather=lambda t, dim=0: full_q if dim == 1 else ctx_lse_stack,
            reduce_scatter=lambda t, dim=1, r=r: rs_total.chunk(dcp_size, dim=1)[r],
        )
        monkeypatch.setattr(cp_attention, "get_dcp_group", lambda g=grp: g)
        base_outs.append(
            cp_attention.forward_with_dcp(
                query=queries[r], dcp_rank=r, dcp_duplicate_o_proj=False, **common_kw
            )
        )

    merged_lses = torch.stack(
        [
            cp_attention.merge_dcp_new_into_ctx(
                ctx_outs[r], ctx_lses[r], new_outs[r], new_lses[r], r, dcp_size
            )[1]
            for r in range(dcp_size)
        ],
        dim=0,
    )
    dup_outs = []
    fn = torch.compile(cp_attention.forward_with_dcp, backend="eager", fullgraph=True)
    for r in range(dcp_size):
        q_calls = []
        grp = SimpleNamespace(
            rank_in_group=r,
            world_size=dcp_size,
            all_gather=lambda t, dim=0, q_c=q_calls: (q_c.append(1) or full_q)
            if dim == 1
            else merged_lses,
            reduce_scatter=lambda *_a, **_k: (_ for _ in ()).throw(
                AssertionError("reduce_scatter called")
            ),
        )
        monkeypatch.setattr(cp_attention, "get_dcp_group", lambda g=grp: g)
        dup_outs.append(
            fn(
                query=full_q if pre_gathered_q else queries[r],
                dcp_rank=r,
                q_dcp_replicated=pre_gathered_q,
                dcp_duplicate_o_proj=True,
                **common_kw,
            )
        )
        assert bool(q_calls) is (not pre_gathered_q)

    torch.testing.assert_close(
        sum(dup_outs), torch.cat(base_outs, dim=1), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("use_fp8", [False, True])
def test_dcp_group_row_parallel_linear_bf16_and_fp8(monkeypatch, use_fp8):
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU

    class _Fp8Cfg(Fp8Config):
        def get_quant_method(self, layer, prefix):
            return VllmFp8LinearMethodTPU(self, prefix=prefix)

    tp_size, dcp_size, tokens, hidden, in_size, blk = 4, 2, 4, 256, 512, 128
    eff_tp, part_in = tp_size // dcp_size, in_size // (tp_size // dcp_size)
    cfg = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp_size,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=dcp_size,
        )
    )
    monkeypatch.setattr(cp_attention, "get_current_vllm_config_or_none", lambda: cfg)
    q_cfg = (
        _Fp8Cfg(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[blk, blk],
        )
        if use_fp8
        else None
    )
    dtype = torch.bfloat16 if use_fp8 else torch.float32
    rng = torch.Generator().manual_seed(2026)
    w_full = (torch.randn(hidden, in_size, generator=rng) * 0.25).to(
        torch.float8_e4m3fn if use_fp8 else dtype
    )
    s_full = torch.rand(hidden // blk, in_size // blk, generator=rng) + 0.5
    b_full = torch.randn(hidden, generator=rng, dtype=dtype)

    layers = []
    for r in range(tp_size):
        monkeypatch.setattr(
            "vllm.distributed.parallel_state._TP",
            SimpleNamespace(world_size=tp_size, rank_in_group=r),
        )
        for mod in (
            "vllm_torchtpu.layers.adapter.cp_attention",
            "vllm.distributed.parallel_state",
            "vllm.model_executor.parameter",
        ):
            monkeypatch.setattr(f"{mod}.get_tensor_model_parallel_rank", lambda r=r: r)
            monkeypatch.setattr(
                f"{mod}.get_tensor_model_parallel_world_size", lambda: tp_size
            )
        if r == 0 and not use_fp8:
            with pytest.raises(ValueError, match="requires input_is_parallel=True"):
                cp_attention.DCPGroupRowParallelLinear(
                    in_size, hidden, input_is_parallel=False, params_dtype=dtype
                )
        layer = cp_attention.DCPGroupRowParallelLinear(
            in_size, hidden, bias=True, params_dtype=dtype, quant_config=q_cfg
        )
        assert (layer.group_size, layer.tp_rank, layer.tp_size) == (
            dcp_size,
            r // dcp_size,
            eff_tp,
        )
        layer.weight.weight_loader(layer.weight, w_full)
        layer.bias.weight_loader(layer.bias, b_full)
        if use_fp8:
            layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, s_full)
            torch.testing.assert_close(
                layer.weight_scale_inv.data,
                s_full.narrow(1, (r // dcp_size) * (part_in // blk), part_in // blk),
            )
            layer.quant_method.process_weights_after_loading(layer)
            assert layer.weight.shape == (part_in, hidden)
        layers.append(layer)

    for g in range(eff_tp):
        torch.testing.assert_close(layers[g * 2].weight, layers[g * 2 + 1].weight)

    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.quantization.fp8.quantized_matmul",
        lambda x, w, s, **_k: (x.float() @ (w.float() * s[None, :])).to(x.dtype),
    )
    x_full = torch.randn(tokens, in_size, generator=rng, dtype=dtype)
    rank_xs, group_xs = [], []
    for g in range(eff_tp):
        sl = x_full.narrow(1, g * part_in, part_in)
        p0 = (sl.float() * 0.5).to(dtype)
        p1 = (sl.float() - p0.float()).to(dtype)
        group_xs.append((p0.float() + p1.float()).to(dtype))
        rank_xs.extend([p0, p1])

    total = sum(
        layers[r].quant_method.apply(layers[r], rank_xs[r], None)
        for r in range(tp_size)
    )
    monkeypatch.setattr(
        cp_attention, "tensor_model_parallel_all_reduce", lambda _x: total
    )
    ref = (
        sum(
            layers[g * 2].quant_method.apply(layers[g * 2], group_xs[g], None)
            for g in range(eff_tp)
        )
        + b_full
    )
    out_0, bias_0 = layers[0].forward(rank_xs[0])
    assert bias_0 is None
    torch.testing.assert_close(out_0, ref, atol=5e-2, rtol=5e-2)


def test_qwen3_next_dcp_duplicate_o_proj_end_to_end(monkeypatch, caplog):
    import logging

    from vllm.model_executor.models import qwen3_next

    from vllm_torchtpu import envs
    from vllm_torchtpu.models.vllm import qwen3_next_patch

    caplog.set_level(logging.WARNING)

    def mk_cfg(dcp, pcp):
        return SimpleNamespace(
            parallel_config=SimpleNamespace(
                decode_context_parallel_size=dcp, prefill_context_parallel_size=pcp
            )
        )

    monkeypatch.setattr(envs, "TPU_DCP_DUPLICATE_O_PROJ", False)
    assert not cp_attention.is_dcp_duplicate_o_proj_enabled(mk_cfg(2, 1))
    monkeypatch.setattr(envs, "TPU_DCP_DUPLICATE_O_PROJ", True)
    assert cp_attention.is_dcp_duplicate_o_proj_enabled(mk_cfg(2, 1))
    assert not cp_attention.is_dcp_duplicate_o_proj_enabled(mk_cfg(2, 2))
    assert any("prefill_context_parallel_size > 1" in r.message for r in caplog.records)

    for cls, attrs in (
        (qwen3_next.Qwen3NextAttention, ("__init__", "_project_qkv_gate", "forward")),
        (qwen3_next.Qwen3NextDecoderLayer, ("forward",)),
        (qwen3_next.Qwen3NextModel, ("forward",)),
    ):
        for attr in attrs:
            monkeypatch.setattr(cls, attr, getattr(cls, attr))
    qwen3_next_patch.maybe_patch_qwen3_next()
    monkeypatch.setattr(
        qwen3_next_patch, "is_dcp_duplicate_o_proj_enabled", lambda: True
    )
    seen_cls = []

    def fake_init(self, *args, num_kv_heads=1, **kwargs):
        seen_cls.append(qwen3_next.RowParallelLinear)
        self.num_kv_heads = num_kv_heads
        self.o_proj = SimpleNamespace(orep_active=True, group_size=2)
        self.attn = SimpleNamespace(impl=SimpleNamespace())
        self.use_fused_qk_norm_rope_gate = True

    monkeypatch.setattr(qwen3_next_patch, "UPSTREAM_ATTENTION_INIT", fake_init)
    orig_row = qwen3_next.RowParallelLinear
    dummy = SimpleNamespace()
    qwen3_next_patch.patched_attention_init(dummy)
    assert (
        seen_cls == [cp_attention.DCPGroupRowParallelLinear]
        and qwen3_next.RowParallelLinear is orig_row
    )
    assert (dummy.dcp_duplicate_o_proj, dummy.use_fused_qk_norm_rope_gate) == (
        True,
        False,
    )
    with pytest.raises(NotImplementedError, match="num_kv_heads == 1"):
        qwen3_next_patch.patched_attention_init(SimpleNamespace(), num_kv_heads=2)

    tokens, heads, dim, dcp = 3, 4, 64, 2
    rng = torch.Generator().manual_seed(9876)
    qs = [torch.randn(tokens, heads * dim, generator=rng) for _ in range(dcp)]
    ks = [torch.randn(tokens, dim, generator=rng) for _ in range(dcp)]
    vs = [torch.randn(tokens, dim, generator=rng) for _ in range(dcp)]
    gs = [torch.randn(tokens, heads * dim, generator=rng) for _ in range(dcp)]
    ps = [torch.randn(tokens, heads * dcp * dim, generator=rng) for _ in range(dcp)]
    w_o = torch.randn(128, heads * dcp * dim, generator=rng)
    q_full = torch.cat([q.view(tokens, heads, dim) for q in qs], dim=1).reshape(
        tokens, -1
    )
    g_full = torch.cat([g.view(tokens, heads, dim) for g in gs], dim=1).reshape(
        tokens, -1
    )
    qg_cat = torch.cat(
        [
            torch.cat([qs[i].view(tokens, heads, dim), gs[i].view(tokens, heads, dim)])
            for i in range(dcp)
        ],
        dim=1,
    )
    q_cat = torch.cat([qs[i].view(tokens, heads, dim) for i in range(dcp)], dim=1)

    def o_proj_fn(item):
        return item @ w_o.t(), None

    o_proj_fn.group_size, o_proj_fn.orep_active = dcp, True

    outs = []
    for r in range(dcp):
        layer = SimpleNamespace(
            num_heads=heads,
            head_dim=dim,
            dcp_duplicate_o_proj=True,
            qkv_proj=lambda h: (h, None),
            attn=lambda q, k, v, output_shape=None, p=ps[r]: p.view(output_shape),
            o_proj=o_proj_fn,
        )
        grp = SimpleNamespace(
            world_size=dcp,
            rank_in_group=r,
            all_gather=lambda t, dim=1: qg_cat if t.shape[0] == 2 * tokens else q_cat,
        )
        monkeypatch.setattr(qwen3_next_patch, "get_dcp_group", lambda g=grp: g)
        for gate_val in (None, gs[r]):
            monkeypatch.setattr(
                qwen3_next_patch,
                "UPSTREAM_PROJECT_QKV_GATE",
                lambda self, qkv, pos, r=r, gv=gate_val: (qs[r], ks[r], vs[r], gv),
            )
            qo, ko, _vo, go = qwen3_next_patch.patched_project_qkv_gate(
                layer, torch.empty(1), torch.empty(1)
            )
            torch.testing.assert_close(qo, q_full)
            torch.testing.assert_close(ko, ks[r])
            assert (go is None) if gate_val is None else torch.allclose(go, g_full)
        layer._project_qkv_gate = lambda qkv, pos, lyr=layer: (
            qwen3_next_patch.patched_project_qkv_gate(lyr, qkv, pos)
        )
        outs.append(
            qwen3_next_patch.patched_attention_forward(
                layer, torch.zeros(tokens, dtype=torch.int32), torch.randn(tokens, 128)
            )
        )
    torch.testing.assert_close(
        sum(outs),
        ((ps[0] + ps[1]) * torch.sigmoid(g_full)) @ w_o.t(),
        atol=1e-4,
        rtol=1e-4,
    )


def test_dcp_duplicate_o_proj_guards_and_empty_cache(
    monkeypatch, cpu_vllm_config_context
):
    from vllm_torchtpu import envs
    from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

    assert "TPU_DCP_DUPLICATE_O_PROJ" in TpuPlatform.additional_env_vars
    monkeypatch.delenv("TPU_DCP_DUPLICATE_O_PROJ", raising=False)
    assert envs.environment_variables["TPU_DCP_DUPLICATE_O_PROJ"]() is False
    monkeypatch.setenv("TPU_DCP_DUPLICATE_O_PROJ", "1")
    assert envs.environment_variables["TPU_DCP_DUPLICATE_O_PROJ"]() is True
    monkeypatch.setattr(envs, "TPU_DCP_DUPLICATE_O_PROJ", True)

    warned = []
    monkeypatch.setattr(
        attention.logger, "warning_once", lambda msg, *a: warned.append(msg)
    )
    monkeypatch.setattr(
        attention,
        "_get_dcp_group",
        lambda: SimpleNamespace(world_size=2, rank_in_group=0),
    )
    monkeypatch.setattr(cp_attention, "_DCP_KERNEL_REGISTRY", {})
    monkeypatch.setattr(cp_attention, "_build_dcp_kernel_op", lambda name, fn: fn)

    impl = _impl()
    with set_vllm_model_wrapper_context(mesh=None, vllm_config=_config()):
        impl.initialize_kernel(SimpleNamespace(_k_scale_float=0.5, _v_scale_float=0.25))
    assert len(warned) == 1 and "TPU_DCP_DUPLICATE_O_PROJ" in warned[0]

    patched = SimpleNamespace(
        _k_scale_float=0.5, _v_scale_float=0.25, dcp_duplicate_o_proj=True
    )
    empty_out = impl.forward(
        patched,
        torch.zeros(6, 8, 256),
        torch.zeros(3, 1, 256),
        torch.zeros(3, 1, 256),
        torch.empty(0),
        SimpleNamespace(),
        output=None,
    )
    assert empty_out.shape == (3, 16, 256)

    bad_impl = attention.PallasBatchedRPAAttentionBackendImpl(
        num_heads=8,
        head_size=256,
        scale=256**-0.5,
        num_kv_heads=2,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="fp8",
    )
    with (
        set_vllm_model_wrapper_context(mesh=None, vllm_config=_config()),
        pytest.raises(NotImplementedError, match="num_kv_heads == 1"),
    ):
        bad_impl.initialize_kernel(patched)
