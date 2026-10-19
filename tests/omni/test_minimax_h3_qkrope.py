# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the MiniMax-H3 fused q/k RMSNorm + RoPE TPU patch.

The vLLM-Omni transformer module is a stand-in exposing a
``MiniMaxH3Attention`` whose forward follows upstream's
(minimax_h3_transformer.py:655-739 at vllm-omni e1d77a3) without the VDN and
VSA branches, registered in ``sys.modules`` under the upstream name. TPU
dispatch is exercised by forcing ``minimax_h3._is_tpu``, and the Pallas kernel
is replaced by a PyTorch transcription of its math; everything runs on CPU.
"""

import sys
import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm_torchtpu.omni import minimax_h3

pytestmark = pytest.mark.omni

HEAD_DIM, ROT_DIM, HEADS, HIDDEN, EPS = 128, 96, 2, 64, 1e-6


def _install(monkeypatch, module: types.ModuleType) -> types.ModuleType:
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def _force_tpu(monkeypatch) -> None:
    """Dispatch every tensor as a torch_tpu tensor (the math stays on CPU)."""
    monkeypatch.setattr(minimax_h3, "_is_tpu", lambda value: True)


def _apply_rope_table(x, rope_table, rotary_dim):
    """fused_qk_norm_rope.py:50-103, half-split mode, x-dtype arithmetic."""
    half = rotary_dim // 2
    cos = rope_table[..., :half].to(x.dtype).unsqueeze(1)
    sin = rope_table[..., half:].to(x.dtype).unsqueeze(1)
    first, second = x[..., :half], x[..., half:rotary_dim]
    return torch.cat(
        (first * cos - second * sin, second * cos + first * sin, x[..., rotary_dim:]),
        dim=-1,
    )


def _upstream_qk_norm_rope(q, k, wq, wk, rope_table, eps):
    """_eager_qk_norm_rope (fused_qk_norm_rope.py:301-320), the TPU path."""
    return tuple(
        _apply_rope_table(F.rms_norm(x, (HEAD_DIM,), w, eps), rope_table, ROT_DIM)
        for x, w in ((q, wq), (k, wk))
    )


def _half_split_rotate(head_dim, rotary_dim, dtype):
    """Same matrix as qk_norm_rope._half_split_rotate."""
    p = torch.zeros(head_dim, head_dim, dtype=dtype)
    first = torch.arange(rotary_dim // 2)
    p[first + rotary_dim // 2, first] = -1.0
    p[first, first + rotary_dim // 2] = 1.0
    return p


def _kernel_math(qkv, wq, wk, cos, sin, eps, head_dim, rotary_dim):
    """PyTorch transcription of qk_norm_rope._per_head_kernel."""
    rows, width = qkv.shape
    dim = width // 3
    p = _half_split_rotate(head_dim, rotary_dim, torch.float32)
    outs = []
    for j, w in ((0, wq), (1, wk)):
        x = qkv[:, j * dim : (j + 1) * dim].float().reshape(rows, -1, head_dim)
        inv_rms = torch.rsqrt(x.square().sum(-1, keepdim=True) / head_dim + eps)
        y = (x * inv_rms * w.float()).to(torch.bfloat16).float()
        out = y * cos.float()[:, None] + (y @ p) * sin.float()[:, None]
        outs.append(out.to(qkv.dtype).reshape(rows, dim))
    return tuple(outs)


def _rope_table(rows, seed=1):
    gen = torch.Generator().manual_seed(seed)
    theta = torch.randint(0, 64, (rows, 1), generator=gen).float() * (
        1.0 / 10000 ** (torch.arange(ROT_DIM // 2) / (ROT_DIM // 2))
    )
    return torch.cat((theta.cos(), theta.sin()), -1).to(torch.bfloat16)


class _Norm:
    def __init__(self, weight):
        self.weight = weight.to(torch.bfloat16)
        self.variance_epsilon = EPS


def _fake_transformer_module(monkeypatch) -> types.ModuleType:
    module = types.ModuleType(minimax_h3.TRANSFORMER_MODULE)

    class MiniMaxH3Attention:
        def __init__(self):
            gen = torch.Generator().manual_seed(0)
            inner = HEADS * HEAD_DIM
            self.num_heads = self.num_kv_heads = HEADS
            self.head_dim, self.rot_dim = HEAD_DIM, ROT_DIM
            self.w_qkv = (torch.randn(HIDDEN, 3 * inner, generator=gen) / 4).to(
                torch.bfloat16
            )
            self.w_out = (torch.randn(inner, HIDDEN, generator=gen) / 16).to(
                torch.bfloat16
            )
            self.q_norm = _Norm(1 + 0.1 * torch.randn(HEAD_DIM, generator=gen))
            self.k_norm = _Norm(1 + 0.1 * torch.randn(HEAD_DIM, generator=gen))
            self.rope = SimpleNamespace(is_neox_style=True)
            self.vdn = None
            self.to_gate_compress = None
            self.attention_calls = []

        def qkv_proj(self, x):
            return x @ self.w_qkv, None

        def out_proj(self, x):
            return x @ self.w_out, None

        def _run_packed_attention(self, q, k, v, **kwargs):
            self.attention_calls.append((q, k, v, kwargs))
            return q + k + v

        def forward(
            self,
            x,
            *,
            rope_table,
            cu_seqlens,
            max_seqlen,
            packed_total=None,
            num_requests=1,
            sp_seq_lens=None,
            video_layout=None,
            vsa_prefix_segments=(),
            vdn_window=None,
        ):
            total = x.shape[0]
            qkv, _ = self.qkv_proj(x)
            size = self.num_heads * self.head_dim
            q, k, v = qkv.split([size, size, size], dim=-1)
            q = q.view(total, self.num_heads, self.head_dim)
            k = k.view(total, self.num_kv_heads, self.head_dim)
            v = v.view(total, self.num_kv_heads, self.head_dim)
            if rope_table is None:
                q = F.rms_norm(q, (HEAD_DIM,), self.q_norm.weight, EPS)
                k = F.rms_norm(k, (HEAD_DIM,), self.k_norm.weight, EPS)
            else:
                q, k = _upstream_qk_norm_rope(
                    q,
                    k,
                    self.q_norm.weight,
                    self.k_norm.weight,
                    rope_table,
                    self.q_norm.variance_epsilon,
                )
            out = self._run_packed_attention(
                q,
                k,
                v,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                packed_total=packed_total if packed_total is not None else q.shape[0],
                num_requests=num_requests,
                video_layout=video_layout,
                vsa_prefix_segments=vsa_prefix_segments,
                gate_compress=None,
                vdn_window=vdn_window,
            )
            out = out.reshape(total, self.num_heads * self.head_dim)
            out, _ = self.out_proj(out)
            return out

    module.MiniMaxH3Attention = MiniMaxH3Attention
    return _install(monkeypatch, module)


def _inputs(rows=40):
    gen = torch.Generator().manual_seed(2)
    x = torch.randn(rows, HIDDEN, generator=gen).to(torch.bfloat16)
    kwargs = dict(cu_seqlens=torch.tensor([0, rows - 3, rows]), max_seqlen=rows - 3)
    return x, _rope_table(rows), kwargs


@pytest.fixture
def fused_calls(monkeypatch):
    calls = []

    def fused(*args):
        calls.append(args)
        return _kernel_math(*args)

    monkeypatch.setattr(minimax_h3, "_fused_h3_qk_norm_rope", fused)
    return calls


def test_rope_tables_rotate_like_upstream_half_split():
    rows = 33
    gen = torch.Generator().manual_seed(3)
    y = torch.randn(rows, HEADS, HEAD_DIM, generator=gen, dtype=torch.float64)
    table = _rope_table(rows).double()
    cos, sin = minimax_h3.h3_rope_tables(table, HEAD_DIM)
    assert cos.shape == sin.shape == (rows, HEAD_DIM)
    assert torch.equal(cos[:, ROT_DIM:], torch.ones(rows, HEAD_DIM - ROT_DIM).double())
    assert torch.equal(sin[:, ROT_DIM:], torch.zeros(rows, HEAD_DIM - ROT_DIM).double())
    p = _half_split_rotate(HEAD_DIM, ROT_DIM, torch.float64)
    rotated = y * cos[:, None] + (y @ p) * sin[:, None]
    # One +-1 term per permuted lane: the same products and sums as upstream.
    assert torch.equal(rotated, _apply_rope_table(y, table, ROT_DIM))


def test_rope_tables_cached_per_table_object():
    first = _rope_table(8)
    tables = minimax_h3.h3_rope_tables(first, HEAD_DIM)
    assert minimax_h3.h3_rope_tables(first, HEAD_DIM) is tables
    second = first.clone()
    rebuilt = minimax_h3.h3_rope_tables(second, HEAD_DIM)
    assert rebuilt is not tables
    assert torch.equal(rebuilt[0], tables[0]) and torch.equal(rebuilt[1], tables[1])
    del second, rebuilt
    # A dead entry never matches a new table that reuses its id.
    assert minimax_h3.h3_rope_tables(first, HEAD_DIM) is not tables

    del first
    assert minimax_h3._H3_ROPE_TABLES == [None, None]


def test_supported_gate(monkeypatch):
    module = _fake_transformer_module(monkeypatch)
    attn = module.MiniMaxH3Attention()
    x, table, kwargs = _inputs()
    supported = minimax_h3.h3_fused_qk_norm_rope_supported
    # CPU tensors keep the upstream forward.
    assert not supported(attn, x, table, kwargs)
    _force_tpu(monkeypatch)
    assert supported(attn, x, table, kwargs)
    assert supported(attn, x, table, {**kwargs, "packed_total": 64, "num_requests": 1})
    assert not supported(attn, x, None, kwargs)  # token refiner
    assert not supported(attn, x, table, {**kwargs, "vdn_window": object()})
    assert not supported(attn, x, table, {**kwargs, "new_upstream_arg": 1})
    assert not supported(attn, x, table, {"cu_seqlens": kwargs["cu_seqlens"]})
    assert not supported(attn, x.float(), table, kwargs)
    assert not supported(attn, x, table.float(), kwargs)
    assert not supported(attn, x, table[:-1], kwargs)
    assert not supported(attn, x, _rope_table(40)[:, :64], kwargs)
    assert not supported(attn, x[:0], table[:0], kwargs)
    for name, value in (
        ("vdn", object()),
        ("to_gate_compress", object()),
        ("head_dim", 64),
        ("rot_dim", 128),
        ("num_kv_heads", 1),
        ("rope", SimpleNamespace(is_neox_style=False)),
        ("q_norm", _Norm(torch.ones(HEADS * HEAD_DIM))),
        ("q_norm", SimpleNamespace(weight=torch.ones(HEAD_DIM), variance_epsilon=EPS)),
        ("k_norm", SimpleNamespace(weight=None)),
    ):
        changed = module.MiniMaxH3Attention()
        setattr(changed, name, value)
        assert not supported(changed, x, table, kwargs), name


def test_patch_installs_idempotent_and_noop(monkeypatch):
    module = _fake_transformer_module(monkeypatch)
    cls = module.MiniMaxH3Attention
    original = cls.forward
    assert minimax_h3.patch_attention_fused_qk_norm_rope()
    assert cls._tpu_orig_forward is original
    assert cls.forward is not original and cls.forward.__wrapped__ is original
    patched = cls.forward
    assert not minimax_h3.patch_attention_fused_qk_norm_rope()
    assert cls.forward is patched
    monkeypatch.setitem(sys.modules, minimax_h3.TRANSFORMER_MODULE, None)
    assert minimax_h3.patch_attention_fused_qk_norm_rope() is False
    empty = types.ModuleType(minimax_h3.TRANSFORMER_MODULE)
    _install(monkeypatch, empty)
    assert minimax_h3.patch_attention_fused_qk_norm_rope() is False


def test_cpu_and_unsupported_calls_keep_upstream_forward(monkeypatch, fused_calls):
    module = _fake_transformer_module(monkeypatch)
    cls = module.MiniMaxH3Attention
    original = cls.forward
    assert minimax_h3.patch_attention_fused_qk_norm_rope()
    attn = cls()
    x, table, kwargs = _inputs()
    for rope_table in (table, None):
        want = original(attn, x, rope_table=rope_table, **kwargs)
        assert torch.equal(attn.forward(x, rope_table=rope_table, **kwargs), want)
    _force_tpu(monkeypatch)
    for rope_table, extra in ((None, {}), (table, {"vdn_window": object()})):
        want = original(attn, x, rope_table=rope_table, **kwargs, **extra)
        got = cls.forward(attn, x, rope_table=rope_table, **kwargs, **extra)
        assert torch.equal(got, want)
    assert fused_calls == []


def test_tpu_forward_uses_fused_kernel(monkeypatch, fused_calls):
    module = _fake_transformer_module(monkeypatch)
    cls = module.MiniMaxH3Attention
    original = cls.forward
    assert minimax_h3.patch_attention_fused_qk_norm_rope()
    _force_tpu(monkeypatch)
    attn = cls()
    x, table, kwargs = _inputs()
    video_layout = object()
    call_kwargs = dict(**kwargs, video_layout=video_layout, vsa_prefix_segments=(3,))
    want = original(attn, x, rope_table=table, **call_kwargs)
    ref_q, ref_k, ref_v, ref_kwargs = attn.attention_calls.pop()
    got = cls.forward(attn, x, rope_table=table, **call_kwargs)
    q, k, v, got_kwargs = attn.attention_calls.pop()

    (args,) = fused_calls
    qkv, wq, wk, cos, sin, eps, head_dim, rot_dim = args
    assert qkv.shape == (x.shape[0], 3 * HEADS * HEAD_DIM)
    assert wq is attn.q_norm.weight and wk is attn.k_norm.weight
    tables = minimax_h3.h3_rope_tables(table, HEAD_DIM)
    assert cos is tables[0] and sin is tables[1]
    assert (eps, head_dim, rot_dim) == (EPS, HEAD_DIM, ROT_DIM)
    assert got_kwargs == ref_kwargs
    assert got_kwargs["packed_total"] == x.shape[0]
    assert got_kwargs["video_layout"] is video_layout
    assert got.shape == want.shape
    assert torch.equal(v, ref_v)
    # f32 rotate with one rounding vs upstream's bf16 eager chain.
    for out, ref in ((q, ref_q), (k, ref_k)):
        assert out.shape == ref.shape and out.dtype == ref.dtype
        diff = (out.float() - ref.float()).abs()
        assert diff.max() <= 2 * 2.0**-7 * ref.float().abs().max(), diff.max()
        assert diff.mean() < 1e-2, diff.mean()


def test_apply_installs_fused_qk_norm_rope_patch(monkeypatch):
    # Minimal pipeline_minimax_h3 stand-in (as in test_minimax_h3_opt.py).
    pipeline = types.ModuleType(minimax_h3.PIPELINE_MODULE)
    pipeline.dist = torch.distributed

    def _broadcast_tensor(tensor, *, dtype, device):
        raise AssertionError("unpatched _broadcast_tensor")

    class MiniMaxH3Pipeline:
        def _encoder_group_broadcast_tensor(self, tensor, *, dtype, device):
            raise AssertionError("unpatched")

        def _distribute_encode_inputs(self, ids, vision_kwargs):
            raise AssertionError("unpatched")

    pipeline._broadcast_tensor = _broadcast_tensor
    pipeline.MiniMaxH3Pipeline = MiniMaxH3Pipeline
    _install(monkeypatch, pipeline)
    monkeypatch.setitem(sys.modules, minimax_h3.INDEXED_MODULATION_MODULE, None)
    for name in (
        minimax_h3.ADALN_CACHE_MODULE,
        minimax_h3.DENOISE_LOOP_MODULE,
    ):
        monkeypatch.setitem(sys.modules, name, None)
    module = _fake_transformer_module(monkeypatch)
    original = module.MiniMaxH3Attention.forward
    assert minimax_h3.apply_minimax_h3_patches()
    assert module.MiniMaxH3Attention._tpu_orig_forward is original
    assert not minimax_h3.apply_minimax_h3_patches()
