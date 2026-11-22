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
import os
import types

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax import lax
from jax.sharding import Mesh

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.mla import dcp_token_parallel as dtp
from vllm_torchtpu.layers.adapter import dcp_nonidx_qb_mla_op as in_branch_mod
from vllm_torchtpu.layers.adapter import dcp_token_parallel_op as tp_op_mod
from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op as mla_op


def _vllm_rmsnorm_ref_np(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    x_t = torch.from_numpy(x.copy()).to(torch.bfloat16)
    w_t = torch.from_numpy(w.copy()).to(torch.bfloat16)
    xf = x_t.to(torch.float32)
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    out = (xf * torch.rsqrt(var + eps)).to(w_t.dtype) * w_t
    return out.to(torch.float32).numpy()


def _bit_equal_bf16(a: jax.Array, b: jax.Array) -> bool:
    au = np.asarray(lax.bitcast_convert_type(a, jnp.uint16))
    bu = np.asarray(lax.bitcast_convert_type(b, jnp.uint16))
    return bool(np.array_equal(au, bu))


def test_should_match_vllm_rmsnorm_and_positive_control_when_in_branch_q_ln_runs() -> (
    None
):
    rng = np.random.default_rng(2026)
    d = 64
    x_f = rng.normal(0.0, 1.5, size=(12, d)).astype(np.float32)
    w_f = rng.uniform(0.25, 2.0, size=(d,)).astype(np.float32)
    eps = 1e-6

    # Verify JAX core accepts raw q_c + q_ln_w and matches eager pre-normed input
    # identically inside sparse_mla_attention_with_in_branch_qb.
    devs = jax.devices()
    if len(devs) < 2:
        pytest.skip(f"needs >= 2 JAX devices, got {len(devs)}")
    mesh = Mesh(np.array(devs[:2]), ("dcp",))

    dcp = 2
    s = 2
    rows = 4
    num_tokens = dcp * rows
    heads = 2
    rank = 8
    d_pe = 4
    n_pages = 2
    page_size = 4
    layout = dtp.TokenParallelLayout(
        dcp_size=dcp,
        interleave_size=1,
        page_size=page_size,
        split_rows=s,
        max_pages=n_pages,
        gather_caps=(n_pages,),
        zigzag_bands=1,
    )

    qc_raw_np = rng.normal(0.0, 1.2, size=(dcp * (s + rows), d)).astype(np.float32)
    qc_raw_j = jnp.asarray(qc_raw_np, dtype=jnp.bfloat16)
    w_ln_j = jnp.asarray(w_f, dtype=jnp.bfloat16)

    # Pre-norm with the exact PyTorch/vLLM formula (cast to weight.dtype then * w).
    xf = qc_raw_j.astype(jnp.float32)
    var = jnp.mean(xf * xf, axis=-1, keepdims=True)
    qc_prenorm_j = (
        (xf * lax.rsqrt(var + jnp.float32(eps))).astype(w_ln_j.dtype) * w_ln_j
    ).astype(qc_raw_j.dtype)

    # Confirm our PyTorch reference matches closely (<= 1 bf16 ULP across CPU rsqrt).
    pt_ref = _vllm_rmsnorm_ref_np(np.asarray(qc_raw_j.astype(jnp.float32)), w_f, eps)
    np.testing.assert_allclose(
        np.asarray(qc_prenorm_j.astype(jnp.float32)), pt_ref, rtol=2e-2, atol=2e-2
    )

    nope = jnp.ones((dcp * n_pages, page_size, rank), jnp.bfloat16)
    rope = jnp.ones((dcp * n_pages, page_size, d_pe), jnp.bfloat16)
    w_qb = jnp.asarray(rng.normal(0.0, 0.2, size=(d, heads * rank)), jnp.bfloat16)
    s_qb = jnp.ones((heads * rank,), jnp.float32)
    w_uk = jnp.ones((heads, 2, rank), jnp.bfloat16)
    cos_sin = jnp.ones((dcp * (s + rows), d_pe), jnp.bfloat16)
    head_topk = jnp.zeros((dcp * s, 2), jnp.int32)
    flat_topk = jnp.zeros((dcp * rows, 2), jnp.int32)
    seq_lens = jnp.asarray([4, 0], jnp.int32)
    bt = jnp.arange(2 * n_pages, dtype=jnp.int32)
    qsl = jnp.asarray([0, num_tokens, num_tokens], jnp.int32)
    dist = jnp.asarray([0, 0, 1], jnp.int32)
    w_uv = jnp.ones((heads, rank, rank), jnp.bfloat16)
    rank_ids = jnp.arange(dcp, dtype=jnp.int32)

    def fake_fused_qb(qc, wqb, _sqb, _wuk, _cs, split_rows):
        proj = jnp.dot(qc, wqb, preferred_element_type=jnp.float32).astype(qc.dtype)
        proj = proj.reshape(qc.shape[0], heads, rank)
        qpe = jnp.zeros((qc.shape[0], heads, d_pe), qc.dtype)
        return proj[:split_rows], proj[split_rows:], qpe[:split_rows], qpe[split_rows:]

    def fake_lead_attend(ql_h, _qph, _n, _r, _ht, _bt, _sl, _qsl, _dist):
        return ql_h, jnp.zeros((ql_h.shape[0], heads), jnp.float32)

    def fake_tail_attend(q_tok, _gn, _gr, _ft, _nr, _qpe=None):
        return q_tok

    def call_in_branch(qc_arr: jax.Array, ln_w: jax.Array | None):
        return dtp.sparse_mla_attention_with_in_branch_qb(
            layout,
            mesh,
            "dcp",
            nope,
            rope,
            qc_arr,
            w_qb,
            s_qb,
            w_uk,
            cos_sin,
            head_topk,
            flat_topk,
            seq_lens,
            bt,
            qsl,
            dist,
            fused_qb=fake_fused_qb,
            lead_attend=fake_lead_attend,
            tail_attend=fake_tail_attend,
            w_uv=w_uv,
            num_tokens=num_tokens,
            rank_ids=rank_ids,
            w_uv_replicated=True,
            flat_topk_cap_strided=True,
            q_ln_w=ln_w,
            q_ln_eps=eps,
        )

    out_ref, lead_ref = call_in_branch(qc_prenorm_j, None)
    out_fused, lead_fused = call_in_branch(qc_raw_j, w_ln_j)
    assert _bit_equal_bf16(out_fused, out_ref)
    assert _bit_equal_bf16(lead_fused, lead_ref)

    # Positive control: scaled norm weight must break parity.
    out_bad, _ = call_in_branch(qc_raw_j, w_ln_j * jnp.asarray(2.0, jnp.bfloat16))
    assert not _bit_equal_bf16(out_bad, out_ref)

    # Direct layout/sentinel check: nonzero unnormed scale differs from normed.
    assert not _bit_equal_bf16(qc_raw_j, qc_prenorm_j)
    del x_f


def test_should_keep_in_branch_qb_inputs_optional_q_ln_defaults() -> None:
    t = torch.zeros((2, 4), dtype=torch.bfloat16)
    base = in_branch_mod.InBranchQbInputs(q_c=t, w_qb=t, s_qb=t, w_uk_t=t, cos_sin=t)
    assert base.q_ln_weight is None
    assert base.q_ln_eps is None

    w = torch.ones((4,), dtype=torch.bfloat16)
    with_ln = in_branch_mod.InBranchQbInputs(
        q_c=t,
        w_qb=t,
        s_qb=t,
        w_uk_t=t,
        cos_sin=t,
        q_ln_weight=w,
        q_ln_eps=1e-6,
    )
    assert with_ln.q_ln_weight is w
    assert with_ln.q_ln_eps == 1e-6


def test_should_default_env_flag_on() -> None:
    assert envs.TPU_DCP_TP_IN_BRANCH_QB_LAYERNORM is True


@pytest.mark.parametrize("split_w_uk", [False, True])
def test_should_register_valid_qln_jax_op_schemas_and_route_run(
    monkeypatch: pytest.MonkeyPatch, split_w_uk: bool
) -> None:
    mesh = object()
    layout = dtp.TokenParallelLayout(4, 4, 32, split_rows=8, max_pages=4)
    monkeypatch.setenv("TPU_DCP_TP_IN_BRANCH_QB_MLA", "1")
    monkeypatch.setenv("TPU_DCP_TP_IN_BRANCH_QB_LAYERNORM", "1")
    monkeypatch.setenv("TPU_DCP_TP_PREFILL_LOCAL_OPROJ", "1")
    monkeypatch.setenv("TPU_DCP_RANK_OPERAND", "1")
    monkeypatch.setenv("TPU_DCP_TP_SHARDED_KV_WRITE", "1")
    monkeypatch.setattr(
        in_branch_mod, "get_tensor_model_parallel_world_size", lambda: 4
    )
    monkeypatch.setattr(
        in_branch_mod, "dcp_rank_operand", lambda: torch.zeros((1,), dtype=torch.int32)
    )

    built_ops: list[tuple[str, int, int]] = []
    invoked: list[str] = []

    class _FakeOp:
        def __init__(self, name: str, fn) -> None:
            self.name = name
            self.fn = fn
            self.fake = None

        def register_fake(self, fn):
            self.fake = fn
            return fn

        def __call__(self, *args):
            invoked.append(self.name)
            assert self.fake is not None
            params = inspect.signature(self.fn).parameters
            assert len(args) == len(params), (self.name, len(args), len(params))
            return self.fake(*args)

    def _capture(
        name: str,
        fn,
        *,
        donate_argnums,
        mesh,
        input_partition_specs,
        output_partition_specs,
    ):
        del donate_argnums, mesh, output_partition_specs
        sig = inspect.signature(fn)
        assert all(
            p.kind
            not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
            for p in sig.parameters.values()
        )
        n_params = len(sig.parameters)
        assert len(input_partition_specs) == n_params, (
            name,
            len(input_partition_specs),
            n_params,
        )
        built_ops.append((name, n_params, len(input_partition_specs)))
        return _FakeOp(name, fn)

    monkeypatch.setattr(in_branch_mod, "pcp_streaming_jax_op", _capture)

    class _Layer:
        kv_cache_quantized_dtype = None
        num_heads = 4
        v_head_dim = 8
        layer_name = "layer.test"
        use_sparse = True
        in_branch_q_ln_eps = 1e-6
        scale = 0.1

    layer = _Layer()
    if split_w_uk:
        layer.W_UK_T_dcp_qrep_lo = torch.zeros((2, 4, 8), dtype=torch.bfloat16)
        layer.W_UK_T_dcp_qrep_hi = torch.zeros((2, 4, 8), dtype=torch.bfloat16)
    impl = types.SimpleNamespace(
        _dcp_mesh=mesh, dcp_size=4, dcp_interleave_size=4, scale=0.1
    )

    run = in_branch_mod.build_in_branch_qb_attention_op(
        layer, impl, layout, q_replicated=True
    )
    assert run is not None
    assert run.has_q_ln(1e-6, False) is True
    assert run.has_q_ln(1e-6, True) is True
    assert run.has_q_ln(1e-5, True) is False

    # Base arity is 18 (or 19 with split_w_uk); q_ln adds +1 (qln_w).
    expected_base = 19 if split_w_uk else 18
    qln_arities = [n for nm, n, _ in built_ops if "qln" in nm]
    assert qln_arities == [expected_base + 1, expected_base + 1]

    # Route a plain and a q_ln call through fake-registered custom ops.
    t_nope = torch.zeros((4, 4, 8), dtype=torch.bfloat16)
    t_rope = torch.zeros((4, 4, 4), dtype=torch.bfloat16)
    t_qc = torch.zeros((16, 16), dtype=torch.bfloat16)
    t_wqb = torch.zeros((16, 8), dtype=torch.bfloat16)
    t_sqb = torch.ones((8,), dtype=torch.float32)
    t_wuk = torch.zeros((2 if split_w_uk else 4, 4, 8), dtype=torch.bfloat16)
    t_cs = torch.zeros((16, 4), dtype=torch.bfloat16)
    t_kv = torch.zeros((8, 8), dtype=torch.bfloat16)
    t_kp = torch.zeros((8, 4), dtype=torch.bfloat16)
    t_tk = torch.zeros((16, 2), dtype=torch.int32)
    t_sl = torch.zeros((2,), dtype=torch.int32)
    t_bt = torch.zeros((8,), dtype=torch.int32)
    t_qsl = torch.zeros((3,), dtype=torch.int32)
    t_dist = torch.zeros((3,), dtype=torch.int32)
    t_uv = torch.zeros((4, 8, 8), dtype=torch.bfloat16)
    t_wo = (
        torch.zeros((8, 12), dtype=torch.bfloat16),
        torch.ones((12,), dtype=torch.float32),
    )
    qb_plain = in_branch_mod.InBranchQbInputs(
        q_c=t_qc,
        w_qb=t_wqb,
        s_qb=t_sqb,
        w_uk_t=t_wuk,
        cos_sin=t_cs,
        w_uk_t_hi=t_wuk if split_w_uk else None,
    )
    out_plain = run(
        (t_nope, t_rope),
        qb_plain,
        t_kv,
        t_kp,
        t_tk,
        t_sl,
        t_bt,
        t_qsl,
        t_dist,
        t_uv,
        t_wo,
        kv_sharded=True,
    )
    assert out_plain.shape == (8, 12)
    assert "qln" not in invoked[-1]

    qb_qln = in_branch_mod.InBranchQbInputs(
        q_c=t_qc,
        w_qb=t_wqb,
        s_qb=t_sqb,
        w_uk_t=t_wuk,
        cos_sin=t_cs,
        w_uk_t_hi=t_wuk if split_w_uk else None,
        q_ln_weight=torch.ones((16,), dtype=torch.bfloat16),
        q_ln_eps=1e-6,
    )
    out_qln = run(
        (t_nope, t_rope),
        qb_qln,
        t_kv,
        t_kp,
        t_tk,
        t_sl,
        t_bt,
        t_qsl,
        t_dist,
        t_uv,
        t_wo,
        kv_sharded=True,
    )
    assert out_qln.shape == (8, 12)
    assert "qln" in invoked[-1]

    qb_bad = in_branch_mod.InBranchQbInputs(
        q_c=t_qc,
        w_qb=t_wqb,
        s_qb=t_sqb,
        w_uk_t=t_wuk,
        cos_sin=t_cs,
        w_uk_t_hi=t_wuk if split_w_uk else None,
        q_ln_weight=torch.ones((16,), dtype=torch.bfloat16),
        q_ln_eps=1e-5,
    )
    with pytest.raises(RuntimeError, match="q_ln op was not registered"):
        run(
            (t_nope, t_rope),
            qb_bad,
            t_kv,
            t_kp,
            t_tk,
            t_sl,
            t_bt,
            t_qsl,
            t_dist,
            t_uv,
            t_wo,
            kv_sharded=True,
        )


@pytest.mark.parametrize(
    ("flag", "skip_topk", "has_indexer", "has_q_ln_op", "expect_defer_ln"),
    [
        ("1", True, True, True, True),
        ("1", False, False, True, True),
        ("0", True, True, True, False),
        ("1", True, True, False, False),
        ("1", False, True, True, False),
    ],
)
def test_should_defer_q_a_layernorm_only_when_nonidx_and_registered(
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
    skip_topk: bool,
    has_indexer: bool,
    has_q_ln_op: bool,
    expect_defer_ln: bool,
) -> None:
    wrapper = mla_op.VllmTPUMultiHeadLatentAttentionWrapper.__new__(
        mla_op.VllmTPUMultiHeadLatentAttentionWrapper
    )
    torch.nn.Module.__init__(wrapper)
    wrapper.is_sparse = True
    wrapper.skip_topk = skip_topk

    class _FakeIndexer:
        indexer_op = types.SimpleNamespace(
            topk_op_tp=types.SimpleNamespace(q_own_rows=True),
            has_live_unified_cache=lambda: False,
        )

        def __call__(self, *args, **kwargs):
            del args, kwargs
            return torch.zeros((64, 12), dtype=torch.int32)

    wrapper.indexer = _FakeIndexer() if has_indexer else None
    wrapper.indexer_rope_emb = None
    wrapper.kimi_preprocess = None
    wrapper.q_lora_rank = 16
    wrapper.kv_lora_rank = 8
    wrapper.qk_nope_head_dim = 8
    wrapper.qk_rope_head_dim = 4
    wrapper.qk_head_dim = 12
    wrapper.num_heads = 4
    wrapper.v_head_dim = 6
    wrapper.dcp_group_size = 4
    wrapper.dcp_q_replicate = True
    wrapper.gate_is_fused = False
    wrapper.g_proj = None
    wrapper.rotary_emb = object()
    wrapper.topk_indices_buffer = torch.zeros((64, 12), dtype=torch.int32)

    ln_calls: list[torch.Tensor] = []
    ln_weight = torch.linspace(0.5, 1.5, 16, dtype=torch.bfloat16)

    class _FakeNorm:
        weight = ln_weight
        variance_epsilon = 1e-6
        has_weight = True
        variance_size_override = None

        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            ln_calls.append(x)
            return x + 1.0

    wrapper.q_a_layernorm = _FakeNorm()
    wrapper.kv_a_layernorm = lambda x: x
    wrapper.fused_qkv_a_proj = object()
    wrapper.q_b_proj = object()
    wrapper.o_proj = types.SimpleNamespace(
        weight=torch.zeros((24, 32), dtype=torch.float8_e4m3fn),
        weight_scale=torch.ones((32,), dtype=torch.float32),
    )

    captured_mla: list[dict[str, Any]] = []

    class _FakeMlaAttn:
        sparse_mla_tp_op = types.SimpleNamespace(
            q_replicated=True,
            serves_oproj=True,
            rank=torch.zeros((1,), dtype=torch.int32),
        )
        sparse_mla_unified_tp_op = None
        sparse_mla_in_branch_qb_op = types.SimpleNamespace(
            serves_sharded_kv=True,
            has_q_ln=lambda eps, kv_sharded=False: (
                has_q_ln_op and eps == 1e-6 and kv_sharded is True
            ),
        )
        W_UV = torch.zeros((4, 8, 6), dtype=torch.bfloat16)
        W_O_tp_half = None

        def __call__(self, q, kv_c, k_pe, **kw):
            del q, kv_c, k_pe
            captured_mla.append(kw)
            return torch.zeros((16, 32), dtype=torch.bfloat16)

    wrapper.mla_attn = _FakeMlaAttn()

    raw_qc = torch.arange(24 * 16, dtype=torch.bfloat16).reshape(24, 16)
    w_qb = torch.zeros((16, 48), dtype=torch.float8_e4m3fn)
    s_qb = torch.ones((48,), dtype=torch.float32)
    w_uk = torch.ones((4, 8, 8), dtype=torch.bfloat16)
    cos_sin_cache = torch.ones((64, 4), dtype=torch.bfloat16)

    monkeypatch.setenv("TPU_DCP_TP_PREFILL_LOCAL_OPROJ", "1")
    monkeypatch.setenv("TPU_DCP_TP_IN_BRANCH_QB_LAYERNORM", flag)
    monkeypatch.setattr(
        mla_op,
        "_FUSED_QB_ROPE_ABSORB_SPLIT_OP",
        lambda qc, *_a: (
            torch.zeros((8, 4, 8), dtype=qc.dtype),
            torch.zeros((qc.shape[0] - 8, 4, 8), dtype=qc.dtype),
            torch.zeros((qc.shape[0], 4, 4), dtype=qc.dtype),
        ),
    )
    monkeypatch.setattr(mla_op, "rope_half_split", lambda _r, _p, q_, k_: (q_, k_))
    monkeypatch.setattr(mla_op, "_retain_buffers", lambda x, *_a: x)
    monkeypatch.setattr(tp_op_mod, "applies", lambda n: n >= 64)
    monkeypatch.setattr(tp_op_mod, "serves_sequence_row_oproj", lambda _o: True)
    monkeypatch.setattr(
        tp_op_mod, "q_rows", lambda _op, _n: torch.arange(24, dtype=torch.long)
    )
    monkeypatch.setattr(
        wrapper,
        "_fused_qb_rope_absorb_operands",
        lambda *_a, **_kw: (w_qb, s_qb, w_uk, 8, cos_sin_cache),
    )
    prefill_group = types.SimpleNamespace(
        project_gather=lambda _w, _h, _n: (
            (
                raw_qc,
                torch.zeros((16, 12), dtype=torch.bfloat16),
                torch.arange(16, dtype=torch.long),
            ),
            None,
        )
    )
    wrapper.forward(
        torch.arange(64, dtype=torch.long),
        torch.zeros((16, 32), dtype=torch.bfloat16),
        prefill_group=prefill_group,
    )
    assert len(captured_mla) == 1
    if expect_defer_ln:
        assert len(ln_calls) == 0
        qb_in = captured_mla[0]["in_branch_qb_inputs"]
        assert qb_in is not None
        assert torch.equal(qb_in.q_c, raw_qc)
        assert qb_in.q_ln_weight is ln_weight
        assert qb_in.q_ln_eps == 1e-6
    else:
        assert len(ln_calls) == 1
        qb_in = captured_mla[0]["in_branch_qb_inputs"]
        if qb_in is not None:
            assert torch.equal(qb_in.q_c, raw_qc + 1.0)
            assert qb_in.q_ln_weight is None
            assert qb_in.q_ln_eps is None
