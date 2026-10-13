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
"""TPU_MLA_ROPE_NEOX_PERMUTE: half-split rope columns + NeoX rope == GPT-J rope.

The claim is that permuting each head's rope output columns of the q
projection and the k_pe columns of the kv_a projection by the same P, then
running the rope NeoX-style, leaves every MLA score -- and so the attention
output -- unchanged. Checked at two levels:

* the wrapper (torch, any device): real projections in their processed
  runtime layouts (bf16 flipped / unflipped, FP8 with a blockwise scale), the
  real vLLM `RotaryEmbedding`, q-replicate on and off, decode and prefill
  shapes, the MTP-drafter configuration; plus a positive control that permutes
  q but not k and must fail.
* the sparse-MLA kernels (JAX, TPU only): the wrapper's q_pe / k_pe written
  into the paged caches -- TensorCore and SparseCore layouts, the latter being
  the rope-pipe SC gather -- and attended, unsharded and DCP position-sharded.
"""

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import vllm.model_executor.custom_op as custom_op_mod
from vllm.config.compilation import CompilationConfig, CompilationMode
from vllm.model_executor.layers.linear import DCPGroupColumnParallelLinear
from vllm.model_executor.layers.mla import MLAModules
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op as mla_op
from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
    VllmMultiHeadLatentAttentionWrapper,
    mla_rope_half_split_perm,
    permute_linear_output_channels,
)
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR

ROPE = 64
NOPE = 192
KV_LORA = 512
Q_LORA = 128
V_DIM = 256
HIDDEN = 256
QK = NOPE + ROPE
MAX_POS = 4096
FP8_BLOCK = 128
P = mla_rope_half_split_perm(ROPE)


def _rope(is_neox_style=False):
    """A real `RotaryEmbedding` as `get_rope` builds it for MLA (GPT-J). The
    TPU dispatch target is `forward_native`, which a bare config binds too."""
    cfg = CompilationConfig(mode=CompilationMode.NONE, custom_ops=["none"])
    with patch.object(custom_op_mod, "get_cached_compilation_config", lambda: cfg):
        return RotaryEmbedding(
            head_size=ROPE,
            rotary_dim=ROPE,
            max_position_embeddings=MAX_POS,
            base=10000.0,
            is_neox_style=is_neox_style,
            dtype=torch.bfloat16,
        )


class _Linear(torch.nn.Module):
    """A dense linear in one of the processed runtime layouts the TPU linear
    methods leave behind: bf16 flipped `[n_in, n_out]`, bf16 unflipped
    `[n_out, n_in]`, or FP8 flipped with a gmm_v2 `[1, n_blocks, 1, n_out]`
    scale. Forward dequantizes and contracts in f32."""

    def __init__(self, n_in, n_out, layout, gen):
        super().__init__()
        w = torch.randn(n_in, n_out, generator=gen) / math.sqrt(n_in)
        self.layout = layout
        if layout == "fp8":
            nb = n_in // FP8_BLOCK
            blocks = w.view(nb, FP8_BLOCK, n_out)
            scale = blocks.abs().amax(1) / 448.0  # [nb, n_out]
            q = (blocks / scale[:, None, :]).to(torch.float8_e4m3fn)
            self.weight = torch.nn.Parameter(q.view(n_in, n_out), requires_grad=False)
            self.weight_scale = torch.nn.Parameter(
                scale[None, :, None, :].contiguous(), requires_grad=False
            )
            setattr(self, WEIGHT_FLIPPED_ATTR, True)
        elif layout == "flipped":
            self.weight = torch.nn.Parameter(w.bfloat16(), requires_grad=False)
            setattr(self, WEIGHT_FLIPPED_ATTR, True)
        else:
            self.weight = torch.nn.Parameter(
                w.t().contiguous().bfloat16(), requires_grad=False
            )

    def dense(self):
        if self.layout == "fp8":
            s = self.weight_scale[0, :, 0, :].repeat_interleave(FP8_BLOCK, 0)
            return self.weight.float() * s
        if self.layout == "flipped":
            return self.weight.float()
        return self.weight.float().t()

    def forward(self, x):
        return (x.float() @ self.dense()).to(torch.bfloat16), None


class _QRepLinear(_Linear):
    """`q_b_proj` under query replication: returns the DCP group's heads."""

    _local_view = DCPGroupColumnParallelLinear._local_view

    def __init__(self, *args, group_size, rank_in_group, **kwargs):
        super().__init__(*args, **kwargs)
        self.group_size = group_size
        self.rank_in_group = rank_in_group
        self.qrep_active = group_size > 1


def _wrapper(
    *,
    heads,
    group_size=1,
    rank_in_group=0,
    layout="flipped",
    seed=0,
    rope=None,
    is_sparse=True,
):
    """A wrapper with real projections, built the way the existing wrapper
    tests do (bypassing the attention constructor); `mla_attn` records what
    it is handed."""
    gen = torch.Generator().manual_seed(seed)
    w = VllmMultiHeadLatentAttentionWrapper.__new__(VllmMultiHeadLatentAttentionWrapper)
    torch.nn.Module.__init__(w)
    w.kimi_preprocess = None
    w.q_lora_rank = Q_LORA
    w.kv_lora_rank = KV_LORA
    w.qk_nope_head_dim = NOPE
    w.qk_rope_head_dim = ROPE
    w.qk_head_dim = QK
    w.num_heads = heads
    w.v_head_dim = V_DIM
    w.dcp_q_replicate = group_size > 1
    w.dcp_group_size = group_size
    w.fused_qkv_a_proj = _Linear(HIDDEN, Q_LORA + KV_LORA + ROPE, layout, gen)
    w.q_b_proj = _QRepLinear(
        Q_LORA,
        heads * group_size * QK,
        layout,
        gen,
        group_size=group_size,
        rank_in_group=rank_in_group,
    )
    w.kv_a_proj_with_mqa = None
    w.q_proj = None
    w.q_a_layernorm = torch.nn.Identity()
    w.kv_a_layernorm = torch.nn.Identity()
    w.rotary_emb = rope if rope is not None else _rope()
    w.is_sparse = is_sparse
    w.indexer = None
    w.skip_topk = True
    w.topk_indices_buffer = torch.zeros((4096, 8), dtype=torch.int32)
    w.g_proj = None
    w.gate_is_fused = False
    w._rope_neox_permuted = False
    w.o_proj = lambda value: (value, None)
    return w


def _run(w, hidden, positions):
    """Forward; returns what the attention layer was handed."""
    w.mla_attn = MagicMock(
        return_value=torch.zeros((hidden.shape[0], w.num_heads * V_DIM))
    )
    VllmMultiHeadLatentAttentionWrapper.forward(
        w, positions=positions, hidden_states=hidden
    )
    (q_nope, q_pe), kv_c, k_pe = w.mla_attn.call_args.args[:3]
    rep = w.mla_attn.call_args.kwargs.get("q_dcp_replicated")
    return dict(q_nope=q_nope, q_pe=q_pe, kv_c=kv_c, k_pe=k_pe, rep=rep)


def _inputs(num_tokens, seed=1, start=0):
    gen = torch.Generator().manual_seed(seed)
    hidden = torch.randn(num_tokens, HIDDEN, generator=gen).bfloat16()
    positions = start + torch.randperm(MAX_POS - start, generator=gen)[:num_tokens]
    return hidden, positions


def _scores(out, w_uk):
    """Full MLA scores `[T, H, T]`: absorbed nope part + rope part, in f32."""
    ql = torch.einsum("thd,hdl->thl", out["q_nope"].float(), w_uk)
    nope = torch.einsum("thl,sl->ths", ql, out["kv_c"].float())
    rope = torch.einsum("thr,sr->ths", out["q_pe"].float(), out["k_pe"][:, 0].float())
    return nope + rope


def _attention(out, w_uk, scale):
    """Causal-free MLA attention output over the step's own keys, in f32."""
    p = torch.softmax(_scores(out, w_uk) * scale, dim=-1)
    return torch.einsum("ths,sl->thl", p, out["kv_c"].float())


def _assert_equivalent(off, on, heads):
    # Untouched channels are bit-identical.
    assert torch.equal(on["q_nope"], off["q_nope"])
    assert torch.equal(on["kv_c"], off["kv_c"])
    # The rope channels are exactly P of the old ones: the matmul columns are
    # the same dot products and NeoX(P x) == P GPT-J(x) elementwise.
    torch.testing.assert_close(on["q_pe"], off["q_pe"][..., P], rtol=0, atol=0)
    torch.testing.assert_close(on["k_pe"], off["k_pe"][..., P], rtol=0, atol=0)
    if off["rep"] is not None:
        for a, b in zip(off["rep"], on["rep"]):
            assert a.shape[-2] == b.shape[-2]
        torch.testing.assert_close(on["rep"][0], off["rep"][0], rtol=0, atol=0)
        torch.testing.assert_close(on["rep"][1], off["rep"][1][..., P], rtol=0, atol=0)
    gen = torch.Generator().manual_seed(7)
    w_uk = torch.randn(heads, NOPE, KV_LORA, generator=gen) / math.sqrt(NOPE)
    s_off, s_on = _scores(off, w_uk), _scores(on, w_uk)
    torch.testing.assert_close(s_on, s_off, rtol=1e-5, atol=1e-4)
    scale = 1.0 / math.sqrt(QK)
    torch.testing.assert_close(
        _attention(on, w_uk, scale), _attention(off, w_uk, scale), rtol=1e-5, atol=1e-5
    )


DECODE_TOKENS = 16
PREFILL_TOKENS = 96


@pytest.mark.parametrize("layout", ["flipped", "unflipped", "fp8"])
@pytest.mark.parametrize(
    "num_tokens", [DECODE_TOKENS, PREFILL_TOKENS], ids=["decode16", "prefill96"]
)
@pytest.mark.parametrize(
    "heads,group_size,rank", [(16, 1, 0), (4, 4, 2)], ids=["qrep_off", "qrep_on"]
)
def test_flag_on_matches_flag_off(layout, num_tokens, heads, group_size, rank):
    off = _wrapper(
        heads=heads, group_size=group_size, rank_in_group=rank, layout=layout
    )
    on = _wrapper(heads=heads, group_size=group_size, rank_in_group=rank, layout=layout)
    on.permute_rope_to_half_split()
    assert on._rope_neox_permuted
    hidden, positions = _inputs(num_tokens)
    r_off, r_on = _run(off, hidden, positions), _run(on, hidden, positions)
    _assert_equivalent(r_off, r_on, heads)
    if group_size > 1:
        # The replicated q spans the group; its scores agree head for head too.
        gh = heads * group_size
        rep_off = dict(r_off, q_nope=r_off["rep"][0], q_pe=r_off["rep"][1])
        rep_on = dict(r_on, q_nope=r_on["rep"][0], q_pe=r_on["rep"][1])
        gen = torch.Generator().manual_seed(9)
        w_uk = torch.randn(gh, NOPE, KV_LORA, generator=gen) / math.sqrt(NOPE)
        torch.testing.assert_close(
            _scores(rep_on, w_uk), _scores(rep_off, w_uk), rtol=1e-5, atol=1e-4
        )


def test_drafter_configuration():
    """The MTP draft layer is the same wrapper class, built replicated (TP 1,
    no q-replicate) and reusing the shared top-k buffer, with its own weights
    and its own cache rows -- its own permutation is self-consistent."""
    off = _wrapper(heads=64, seed=3)
    on = _wrapper(heads=64, seed=3)
    on.permute_rope_to_half_split()
    for n in (DECODE_TOKENS, 48):
        hidden, positions = _inputs(n, seed=n)
        _assert_equivalent(
            _run(off, hidden, positions), _run(on, hidden, positions), 64
        )


def test_deepseek_scaling_rope_variant():
    """The other rope class the wrapper accepts takes its own NeoX branch."""
    cfg = CompilationConfig(mode=CompilationMode.NONE, custom_ops=["none"])
    with patch.object(custom_op_mod, "get_cached_compilation_config", lambda: cfg):
        rope = mla_op.VllmDeepseekScalingRotaryEmbedding(
            head_size=ROPE,
            rotary_dim=ROPE,
            max_position_embeddings=MAX_POS,
            base=10000.0,
            is_neox_style=False,
            scaling_factor=4.0,
            dtype=torch.bfloat16,
        )
    off = _wrapper(heads=8, rope=rope)
    on = _wrapper(heads=8, rope=rope)
    on.permute_rope_to_half_split()
    hidden, positions = _inputs(DECODE_TOKENS)
    r_off, r_on = _run(off, hidden, positions), _run(on, hidden, positions)
    torch.testing.assert_close(r_on["q_pe"], r_off["q_pe"][..., P], rtol=0, atol=0)
    torch.testing.assert_close(r_on["k_pe"], r_off["k_pe"][..., P], rtol=0, atol=0)


def test_positive_control_q_only_permutation_fails():
    """Permute q but not k: the scores must move, or the test above proves
    nothing about consistency."""
    off = _wrapper(heads=16)
    bad = _wrapper(heads=16)
    q_index = mla_op.mla_rope_output_index(16 * QK, ROPE, QK, NOPE, 16)
    permute_linear_output_channels(bad.q_b_proj, q_index)
    bad._rope_neox_permuted = True
    hidden, positions = _inputs(DECODE_TOKENS)
    r_off, r_bad = _run(off, hidden, positions), _run(bad, hidden, positions)
    gen = torch.Generator().manual_seed(7)
    w_uk = torch.randn(16, NOPE, KV_LORA, generator=gen) / math.sqrt(NOPE)
    diff = (_scores(r_bad, w_uk) - _scores(r_off, w_uk)).abs().max().item()
    assert diff > 0.1, f"q-only permutation left the scores unchanged ({diff})"


def test_permutation_is_idempotent_and_reaches_every_parameter():
    w = _wrapper(heads=4, group_size=4, layout="fp8")
    before = {
        n: p.detach().clone()
        for n, p in w.q_b_proj.named_parameters()
        if n in ("weight", "weight_scale")
    }
    w.permute_rope_to_half_split()
    once = {n: p.detach().clone() for n, p in w.q_b_proj.named_parameters()}
    w.permute_rope_to_half_split()
    for n, p in w.q_b_proj.named_parameters():
        assert torch.equal(
            p.view(torch.uint8) if p.dtype.itemsize == 1 else p,
            (once[n].view(torch.uint8) if once[n].dtype.itemsize == 1 else once[n]),
        )
    # Both the fp8 weight and its per-column scale moved together.
    idx = mla_op.mla_rope_output_index(16 * QK, ROPE, QK, NOPE, 16)
    assert torch.equal(
        once["weight"].view(torch.uint8), before["weight"].view(torch.uint8)[:, idx]
    )
    assert torch.equal(once["weight_scale"], before["weight_scale"][..., idx])


def test_unknown_parameter_layout_raises():
    layer = _Linear(Q_LORA, 2 * QK, "flipped", torch.Generator().manual_seed(0))
    layer.odd = torch.nn.Parameter(torch.ones(3, 5), requires_grad=False)
    with pytest.raises(NotImplementedError):
        permute_linear_output_channels(
            layer, mla_op.mla_rope_output_index(2 * QK, ROPE, QK, NOPE, 2)
        )


class _Attn(torch.nn.Module):
    def forward(self, *args, **kwargs):
        return torch.zeros(1)


def _modules(rope):
    return MLAModules(
        kv_a_layernorm=torch.nn.Identity(),
        kv_b_proj=torch.nn.Identity(),
        rotary_emb=rope,
        o_proj=torch.nn.Identity(),
        fused_qkv_a_proj=torch.nn.Identity(),
        kv_a_proj_with_mqa=None,
        q_a_layernorm=torch.nn.Identity(),
        q_b_proj=torch.nn.Identity(),
        q_proj=None,
        indexer=None,
        is_sparse=False,
        topk_indices_buffer=None,
    )


def _construct(rope):
    with patch.object(mla_op, "VllmTPUMLAAttention", return_value=_Attn()):
        return VllmMultiHeadLatentAttentionWrapper(
            hidden_size=HIDDEN,
            num_heads=4,
            scale=1.0,
            qk_nope_head_dim=NOPE,
            qk_rope_head_dim=ROPE,
            v_head_dim=V_DIM,
            q_lora_rank=Q_LORA,
            kv_lora_rank=KV_LORA,
            mla_modules=_modules(rope),
        )


def test_flag_registers_the_post_load_hook(monkeypatch):
    monkeypatch.setattr(envs, "TPU_MLA_ROPE_NEOX_PERMUTE", True)
    w = _construct(_rope())
    assert not w._rope_neox_permuted
    assert w.mla_attn.post_load_hooks == [w.permute_rope_to_half_split]


def test_flag_off_registers_nothing(monkeypatch):
    monkeypatch.setattr(envs, "TPU_MLA_ROPE_NEOX_PERMUTE", False)
    w = _construct(_rope())
    assert not getattr(w.mla_attn, "post_load_hooks", [])


def test_flag_rejects_a_neox_rope(monkeypatch):
    monkeypatch.setenv("TPU_MLA_ROPE_NEOX_PERMUTE", "1")
    monkeypatch.setattr(envs, "TPU_MLA_ROPE_NEOX_PERMUTE", True)
    with pytest.raises(NotImplementedError):
        _construct(_rope(is_neox_style=True))


def test_default_skips_a_neox_rope(monkeypatch):
    monkeypatch.delenv("TPU_MLA_ROPE_NEOX_PERMUTE", raising=False)
    monkeypatch.setattr(envs, "TPU_MLA_ROPE_NEOX_PERMUTE", True)
    w = _construct(_rope(is_neox_style=True))
    assert not getattr(w.mla_attn, "post_load_hooks", [])


# --------------------------------------------------------------------------
# Kernel level (TPU): the wrapper's tensors through the real caches + kernels.
# --------------------------------------------------------------------------

jax = pytest.importorskip("jax")
jnp = jax.numpy

from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk  # noqa: E402
from vllm_torchtpu.kernels.mla.kv_cache_utils import (  # noqa: E402
    KVCacheLayout,
    KVCacheType,
    SparseMLAKVCacheSpec,
)
from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_kernel  # noqa: E402
from vllm_torchtpu.layers.core import attention_interface  # noqa: E402

K_HEADS = 16
PAGE_SIZE = 32
PAGES_PER_SEQ = 4
TOTAL_PAGES = 64
TOKEN_PAD = 16
K_SCALE = 1.5
SM_SCALE = 1.0 / math.sqrt(QK)


def _require_tpu():
    if jax.devices()[0].platform != "tpu":
        pytest.skip("sparse MLA kernels require a TPU (SparseCore)")


def _specs(layout, total_pages=TOTAL_PAGES):
    packing = sparse_kernel.get_dtype_packing(jnp.float8_e4m3fn)
    return tuple(
        SparseMLAKVCacheSpec.create(t, layout, total_pages, PAGE_SIZE, d, packing)
        for t, d in ((KVCacheType.NOPE, KV_LORA), (KVCacheType.ROPE, ROPE))
    )


def _empty(specs):
    return tuple(jnp.zeros(s.shape, s.jax_dtype) for s in specs)


def _fp8(t: torch.Tensor):
    return jnp.asarray(t.float().numpy() / K_SCALE).astype(jnp.float8_e4m3fn)


def _bf16(a):
    return jnp.asarray(np.asarray(a, np.float32), dtype=jnp.bfloat16)


def _causal_topk(positions, topk):
    rows = np.full((len(positions), topk), -1, np.int32)
    for i, pos in enumerate(positions):
        rows[i, : pos + 1] = np.arange(pos + 1)
    return rows


def _pad_rows(x, n):
    return (
        x
        if x.shape[0] == n
        else jnp.concatenate([x, jnp.zeros((n - x.shape[0], *x.shape[1:]), x.dtype)])
    )


class _Step:
    """Wrapper-produced kernel inputs for one model step, flag off and on."""

    def __init__(self, heads, group_size, hidden, positions, w_uk):
        self.r = {}
        for mode in ("off", "on"):
            w = _wrapper(heads=heads, group_size=group_size, rank_in_group=1)
            if mode == "on":
                w.permute_rope_to_half_split()
            out = _run(w, hidden, positions)
            q_nope, q_pe = (
                out["rep"] if group_size > 1 else (out["q_nope"], out["q_pe"])
            )
            ql = torch.einsum("thd,hdl->thl", q_nope.float(), w_uk)
            self.r[mode] = dict(
                ql_nope=_bf16(ql),
                q_pe=_bf16(q_pe.float()),
                kv_c=_fp8(out["kv_c"]),
                k_pe=_fp8(out["k_pe"][:, 0]),
            )


def _w_uk(heads):
    gen = torch.Generator().manual_seed(11)
    return torch.randn(heads, NOPE, KV_LORA, generator=gen) / math.sqrt(NOPE)


def _attend(caches, step, topk_rows, seq_lens, qsl, dist, block_tables, specs, n):
    mesh = jax.sharding.Mesh(np.array(jax.local_devices()[:1]), ("x",))
    nope, rope, out = attention_interface.sparse_mla_attention(
        _pad_rows(step["ql_nope"], n),
        _pad_rows(step["q_pe"], n),
        _pad_rows(step["kv_c"], n),
        _pad_rows(step["k_pe"], n),
        caches[0],
        caches[1],
        jnp.asarray(topk_rows, jnp.int32),
        jnp.asarray(seq_lens, jnp.int32),
        block_tables,
        jnp.asarray(qsl, jnp.int32),
        jnp.asarray(dist, jnp.int32),
        mesh,
        *specs,
        sm_scale=SM_SCALE,
        k_scale=K_SCALE,
    )
    return (nope, rope), out


def _prefill_then_decode(layout, heads, group_size, *, q_only=False):
    """Prefill chunk for 8 sequences, then a 16-token decode bucket (8 real
    decode tokens + padding). Returns {mode: (prefill_out, decode_out)}."""
    specs = _specs(layout)
    rng = np.random.default_rng(5)
    prior = [int(x) for x in rng.integers(8, 60, size=8)]
    num_seqs = len(prior)
    gh = heads * group_size
    w_uk = _w_uk(gh)
    block_tables = jnp.asarray(
        rng.permutation(TOTAL_PAGES)[: num_seqs * PAGES_PER_SEQ], jnp.int32
    )
    positions = [p for n in prior for p in range(n)]
    total = len(positions)
    n_pref = math.ceil(total / TOKEN_PAD) * TOKEN_PAD
    hidden_p = torch.randn(
        total, HIDDEN, generator=torch.Generator().manual_seed(21)
    ).bfloat16()
    pref = _Step(heads, group_size, hidden_p, torch.tensor(positions), w_uk)
    hidden_d = torch.randn(
        num_seqs, HIDDEN, generator=torch.Generator().manual_seed(22)
    ).bfloat16()
    dec = _Step(heads, group_size, hidden_d, torch.tensor(prior), w_uk)
    if q_only:  # positive control: q from the flag-on run, k from flag-off
        for st in (pref, dec):
            st.r["on"] = dict(st.r["off"], q_pe=st.r["on"]["q_pe"])
    topk = 128
    pref_topk = np.concatenate(
        [_causal_topk(positions, topk), _causal_topk([0] * (n_pref - total), topk)]
    )
    dec_topk = np.concatenate(
        [_causal_topk(prior, topk), _causal_topk([0] * (TOKEN_PAD - num_seqs), topk)]
    )
    qsl = np.concatenate([[0], np.cumsum(prior)])
    results = {}
    for mode in ("off", "on"):
        caches, out_p = _attend(
            _empty(specs),
            pref.r[mode],
            pref_topk,
            prior,
            qsl,
            [0, num_seqs, num_seqs],
            block_tables,
            specs,
            n_pref,
        )
        _, out_d = _attend(
            caches,
            dec.r[mode],
            dec_topk,
            [n + 1 for n in prior],
            list(range(num_seqs + 1)),
            [num_seqs, num_seqs, num_seqs],
            block_tables,
            specs,
            TOKEN_PAD,
        )
        results[mode] = (
            np.asarray(out_p.astype(jnp.float32))[:total],
            np.asarray(out_d.astype(jnp.float32))[:num_seqs],
        )
    return results


LAYOUTS = [KVCacheLayout.TENSORCORE, KVCacheLayout.SPARSECORE]


@pytest.mark.parametrize("layout", LAYOUTS, ids=lambda x: x.value)
@pytest.mark.parametrize(
    "heads,group_size", [(16, 1), (4, 4)], ids=["qrep_off", "qrep_on"]
)
def test_kernel_output_flag_on_matches_flag_off(layout, heads, group_size):
    _require_tpu()
    res = _prefill_then_decode(layout, heads, group_size)
    for phase, a, b in zip(("prefill", "decode"), res["off"], res["on"]):
        assert np.isfinite(a).all() and np.abs(a).max() > 0, phase
        np.testing.assert_allclose(b, a, rtol=2e-2, atol=2e-2, err_msg=phase)
        print(
            f"{layout.value} g{group_size} {phase}: max|on-off| = "
            f"{np.abs(b - a).max():.3e}, exact={np.array_equal(a, b)}"
        )


@pytest.mark.parametrize("layout", LAYOUTS, ids=lambda x: x.value)
def test_kernel_positive_control_q_only_fails(layout):
    _require_tpu()
    res = _prefill_then_decode(layout, 16, 1, q_only=True)
    worst = max(np.abs(b - a).max() for a, b in zip(res["off"], res["on"]))
    assert worst > 5e-2, f"q-only permutation did not move the output ({worst})"


def _local_topk(global_rows, dcp_size, interleave):
    """Global top-k rows -> per-rank rank-local cache indices (the
    `streamindex_topk_dcp` contract): virtual page `p // (ps * D)`, slot
    `(off // (D * I)) * I + off % I`, owner `(off % (D * I)) // I`."""
    t, k = global_rows.shape
    out = np.full((dcp_size, t, k), -1, np.int32)
    for i in range(t):
        row = global_rows[i][global_rows[i] >= 0]
        vps = PAGE_SIZE * dcp_size
        cycle = dcp_size * interleave
        off = row % vps
        owner = (off % cycle) // interleave
        local = (
            (row // vps) * PAGE_SIZE + (off // cycle) * interleave + off % interleave
        )
        for r in range(dcp_size):
            sel = local[owner == r]
            out[r, i, : sel.size] = sel
    return out.reshape(dcp_size * t, k)


def _merge(out, lse, dcp_size):
    out = out.reshape(dcp_size, -1, *out.shape[1:])
    lse = lse.reshape(dcp_size, -1, lse.shape[-1])
    m = np.max(lse, 0)
    m = np.where(np.isfinite(m), m, 0.0)
    wts = np.exp(lse - m[None])
    wts = wts / np.maximum(wts.sum(0, keepdims=True), 1e-30)
    # A rank owning none of a token's top-k has weight 0 and an unspecified
    # output; keep it out of the sum rather than risk 0 * NaN.
    return np.where(wts[..., None] > 0, wts[..., None] * out, 0.0).sum(0)


def _dcp_prefill_then_decode(layout, dcp_size, interleave=8):
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as Ps

    axis = streamindex_topk.DCP_AXIS_NAME
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:dcp_size]), (axis,))
    specs = _specs(layout, TOTAL_PAGES)
    rng = np.random.default_rng(6)
    prior = [int(x) for x in rng.integers(16, 100, size=4)]
    num_seqs = len(prior)
    w_uk = _w_uk(16)
    pages_per_seq = 2  # virtual pages of PAGE_SIZE * dcp_size positions
    block_tables = jnp.asarray(
        rng.permutation(TOTAL_PAGES)[: num_seqs * pages_per_seq], jnp.int32
    )
    positions = [p for n in prior for p in range(n)]
    total = len(positions)
    n_pref = math.ceil(total / TOKEN_PAD) * TOKEN_PAD
    gen = torch.Generator()
    pref = _Step(
        4,
        4,
        torch.randn(total, HIDDEN, generator=gen.manual_seed(31)).bfloat16(),
        torch.tensor(positions),
        w_uk,
    )
    dec = _Step(
        4,
        4,
        torch.randn(num_seqs, HIDDEN, generator=gen.manual_seed(32)).bfloat16(),
        torch.tensor(prior),
        w_uk,
    )
    topk = 128
    pref_rows = np.concatenate(
        [_causal_topk(positions, topk), _causal_topk([0] * (n_pref - total), topk)]
    )
    dec_rows = np.concatenate(
        [_causal_topk(prior, topk), _causal_topk([0] * (TOKEN_PAD - num_seqs), topk)]
    )
    qsl = np.concatenate([[0], np.cumsum(prior)])
    shard = NamedSharding(mesh, Ps(axis))

    def caches():
        return tuple(
            jax.device_put(
                jnp.zeros((dcp_size * s.shape[0], *s.shape[1:]), s.jax_dtype), shard
            )
            for s in specs
        )

    def call(c, st, rows, seq_lens, q, dist, n):
        return attention_interface.sparse_mla_attention_dcp(
            _pad_rows(st["ql_nope"], n),
            _pad_rows(st["q_pe"], n),
            _pad_rows(st["kv_c"], n),
            _pad_rows(st["k_pe"], n),
            c[0],
            c[1],
            jnp.asarray(_local_topk(rows, dcp_size, interleave), jnp.int32),
            jnp.asarray(seq_lens, jnp.int32),
            block_tables,
            jnp.asarray(q, jnp.int32),
            jnp.asarray(dist, jnp.int32),
            mesh,
            *specs,
            sm_scale=SM_SCALE,
            k_scale=K_SCALE,
            dcp_size=dcp_size,
            interleave_size=interleave,
        )

    results = {}
    for mode in ("off", "on"):
        n0, r0, out_p, lse_p = call(
            caches(),
            pref.r[mode],
            pref_rows,
            prior,
            qsl,
            [0, num_seqs, num_seqs],
            n_pref,
        )
        _, _, out_d, lse_d = call(
            (n0, r0),
            dec.r[mode],
            dec_rows,
            [n + 1 for n in prior],
            list(range(num_seqs + 1)),
            [num_seqs, num_seqs, num_seqs],
            TOKEN_PAD,
        )
        results[mode] = (
            _merge(np.asarray(out_p.astype(jnp.float32)), np.asarray(lse_p), dcp_size)[
                :total
            ],
            _merge(np.asarray(out_d.astype(jnp.float32)), np.asarray(lse_d), dcp_size)[
                :num_seqs
            ],
        )
    return results


@pytest.mark.parametrize("layout", LAYOUTS, ids=lambda x: x.value)
def test_dcp_kernel_output_flag_on_matches_flag_off(layout):
    """The DCP path: per-rank masked cache writes of the permuted k_pe, the
    rank-local gather, and the LSE merge of position-sharded partials."""
    _require_tpu()
    dcp_size = min(4, len(jax.devices()))
    if dcp_size < 2:
        pytest.skip("DCP needs at least 2 devices")
    res = _dcp_prefill_then_decode(layout, dcp_size)
    for phase, a, b in zip(("prefill", "decode"), res["off"], res["on"]):
        assert np.isfinite(a).all() and np.abs(a).max() > 0, phase
        np.testing.assert_allclose(b, a, rtol=2e-2, atol=2e-2, err_msg=phase)
        print(
            f"dcp{dcp_size} {layout.value} {phase}: max|on-off| = "
            f"{np.abs(b - a).max():.3e}, exact={np.array_equal(a, b)}"
        )


# --- TPU_MLA_DECODE_PROLOGUE: the served wiring ------------------------------


def _kernel_rope_emulated(q_raw, tab):
    """The kernel's `rope_q_block` on the raw head, in torch: x*C + rot(x)*S
    over q_pe = q_raw[..., 192:] with each product rounded to bf16 first."""
    x = q_raw[..., NOPE:].float()  # [T, H, 64]
    c = tab[:, None, 0, 64:]  # [T, 1, 64] = [cos, cos]
    s = tab[:, None, 1, 64:]  # [-sin, sin]
    rot = torch.cat([x[..., 32:], x[..., :32]], dim=-1)
    a = (x * c).bfloat16().float()
    b = (rot * s).bfloat16().float()
    return (a + b).bfloat16()


@pytest.mark.parametrize("num_tokens", [DECODE_TOKENS, PREFILL_TOKENS])
@pytest.mark.parametrize("group_size,heads", [(4, 4), (2, 8)])
def test_decode_prologue_wrapper_hands_raw_q_and_table(num_tokens, group_size, heads):
    """Flag on: the qrep sparse path gets (q_nope, raw q head, rope table);
    k_pe / kv_c / q_nope are bit-identical to flag off, and the kernel's rope
    of (raw q, table) is bit-identical to flag off's q_pe."""
    hidden, positions = _inputs(num_tokens, seed=5)
    outs = {}
    # One thread: a multi-threaded CPU GEMM may reduce in a different order
    # call to call, and this compares two forwards bit for bit.
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for flag in (False, True, False):
            w = _wrapper(heads=heads, group_size=group_size, rank_in_group=1)
            w.permute_rope_to_half_split()
            w.decode_prologue = flag
            outs.setdefault(flag, []).append(_run(w, hidden, positions))
    finally:
        torch.set_num_threads(threads)
    off, on = outs[False][0], outs[True][0]
    # The harness itself is deterministic (flag off twice).
    assert torch.equal(outs[False][1]["k_pe"], off["k_pe"])
    assert torch.equal(on["k_pe"], off["k_pe"])
    assert torch.equal(on["kv_c"], off["kv_c"])
    assert len(off["rep"]) == 2 and len(on["rep"]) == 3
    q_nope_on, q_raw, tab = on["rep"]
    assert torch.equal(q_nope_on, off["rep"][0])
    assert q_raw.shape == (num_tokens, heads * group_size, QK)
    assert tab.shape == (num_tokens, 2, 128) and tab.dtype == torch.float32
    # The raw head's nope part is q_nope itself.
    assert torch.equal(q_raw[..., :NOPE], off["rep"][0])
    got = _kernel_rope_emulated(q_raw, tab)
    assert torch.equal(got, off["rep"][1].contiguous())
    # Positive control: a table built from other positions moves it.
    w = _wrapper(heads=heads, group_size=group_size, rank_in_group=1)
    bad = mla_op.mla_q_rope_table(w.rotary_emb, (positions + 1) % MAX_POS, q_raw)
    assert not torch.equal(_kernel_rope_emulated(q_raw, bad), got)


def test_decode_prologue_needs_permuted_rope():
    """Without the NeoX half-split layout the wrapper keeps the torch rope."""
    hidden, positions = _inputs(DECODE_TOKENS, seed=6)
    w = _wrapper(heads=4, group_size=4, rank_in_group=0)
    w.decode_prologue = True  # but _rope_neox_permuted is False
    out = _run(w, hidden, positions)
    assert len(out["rep"]) == 2


def test_decode_prologue_impl_routes_to_rope_op(monkeypatch):
    """`PallasMLAttentionBackendImpl.forward` with a 3-tuple q_dcp_replicated
    calls the rope op with the flag-off op's operands, the raw q head in place
    of q_pe and the table appended."""
    try:
        from vllm_torchtpu.layers.adapter import attention as attn_mod
    except Exception as e:  # noqa: BLE001 - needs torch_tpu
        pytest.skip(f"attention backend unavailable: {e}")
    t, h, g = 16, 4, 4
    gen = torch.Generator().manual_seed(3)
    layer = MagicMock()
    layer.num_heads = h
    layer.kv_lora_rank = KV_LORA
    layer.qk_rope_head_dim = ROPE
    layer.v_head_dim = V_DIM
    layer.kv_cache_quantized_dtype = None
    layer.W_UK_T_dcp_qrep = torch.randn(h * g, NOPE, KV_LORA, generator=gen).bfloat16()
    layer.W_UK_T_dcp_qrep_scale = None
    layer.W_UV = torch.randn(h, KV_LORA, V_DIM, generator=gen).bfloat16()
    del layer.W_UV_scale
    for name in ("sparse_mla_dcp_op", "sparse_mla_dcp_rope_op"):
        setattr(
            layer,
            name,
            MagicMock(
                return_value=(
                    torch.zeros(g * t, h * g, KV_LORA, dtype=torch.bfloat16),
                    torch.zeros(g * t, h * g),
                )
            ),
        )
    impl = attn_mod.PallasMLAttentionBackendImpl.__new__(
        attn_mod.PallasMLAttentionBackendImpl
    )
    impl.dcp_size = g
    monkeypatch.setattr(
        attn_mod,
        "merge_lse_partials_scatter_heads",
        lambda partial, lse, **kw: partial[:t, :h],
    )
    monkeypatch.setattr(
        impl, "_get_kv_scales", lambda layer: (None, None, None), raising=False
    )
    q = torch.randn(t, h * g, QK, generator=gen).bfloat16()
    q_nope, q_pe = q[..., :NOPE], q[..., NOPE:]
    tab = torch.randn(t, 2, 128, generator=gen)
    kv_c = torch.randn(t, KV_LORA, generator=gen).bfloat16()
    k_pe = torch.randn(t, 1, ROPE, generator=gen).bfloat16()
    kv_cache = (torch.zeros(4, 4), torch.zeros(4, 4))
    md = MagicMock()
    topk = torch.zeros(g * t, 8, dtype=torch.int32)
    common = dict(
        layer=layer,
        kv_c_normed=kv_c,
        k_pe=k_pe,
        kv_cache=kv_cache,
        attn_metadata=md,
        topk_indices=topk,
    )
    impl.forward(
        q=(q_nope[:, :h], q_pe[:, :h]), q_dcp_replicated=(q_nope, q_pe), **common
    )
    impl.forward(
        q=(q_nope[:, :h], q_pe[:, :h]), q_dcp_replicated=(q_nope, q, tab), **common
    )
    base = layer.sparse_mla_dcp_op.call_args.args
    rope = layer.sparse_mla_dcp_rope_op.call_args.args
    assert len(rope) == len(base) + 1 and rope[-1] is tab
    for i, (a, b) in enumerate(zip(base, rope)):
        if i == 2:  # q_pe slot: the raw head view
            assert b.shape == (t, h * g, QK) and torch.equal(b, q)
            assert torch.equal(a, q_pe)
        elif isinstance(a, torch.Tensor):
            assert torch.equal(a, b), i
        else:
            assert a is b, i
