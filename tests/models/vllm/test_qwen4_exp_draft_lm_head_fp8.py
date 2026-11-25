# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_DRAFT_LMHEAD_FP8: the Qwen4Exp MTP draft's own FP8 lm_head copy.

NOT bitwise by design (draft tokens only; target verify unchanged). Checks: flag off
binds nothing; flag on keeps the target head object, weight and logits bit-identical;
draft logits stay close to bf16 (top-1 agreement on random inputs); the flag is in
the compile-cache key; the draft head's traced shapes (warmup keys) are unchanged.
"""

from types import SimpleNamespace as NS
from unittest.mock import patch

import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
import torch
from torch import nn
from vllm.model_executor.layers import logits_processor as lp_module
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.quantized_matmul.util import xla_quantized_matmul
from vllm_torchtpu.models.vllm.qwen4_exp import draft_lm_head as dlh

H, V = 1024, 4096  # CPU-sized shard (served: H 2560, V 31040 per rank)


def _to_jax(t: torch.Tensor):
    if t.dtype == torch.float8_e4m3fn:
        return jnp.asarray(t.view(torch.uint8).numpy().view(ml_dtypes.float8_e4m3fn))
    if t.dtype == torch.bfloat16:
        return jnp.asarray(t.float().numpy()).astype(jnp.bfloat16)
    return jnp.asarray(t.numpy())


def _xla_qmm(x, w_q, w_s):
    """The production per-channel FP8 path (`_quantized_matmul_jax` with a 1-D
    scale = `xla_quantized_matmul`), run in JAX on CPU."""
    out = xla_quantized_matmul(_to_jax(x.reshape(-1, x.shape[-1])), _to_jax(w_q), _to_jax(w_s))
    out = torch.from_numpy(np.asarray(out.astype(jnp.float32))).to(x.dtype)
    return out.reshape(*x.shape[:-1], w_q.shape[-1])


def _processor():
    with patch.object(
        lp_module, "get_current_vllm_config", return_value=NS(model_config=None)
    ):
        return LogitsProcessor(V)


class _Head(nn.Module):
    """A one-rank stand-in for the target's ParallelLMHead."""

    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = None
        self.tp_size, self.tp_rank = 1, 0
        self.org_vocab_size = self.org_vocab_size_padded = V
        self.num_embeddings = self.num_embeddings_padded = V
        self.num_embeddings_per_partition, self.embedding_dim = V, H
        self.shard_indices = NS(org_vocab_start_index=0)
        self.quant_method = UnquantizedEmbeddingMethod()


class _Draft(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.lm_head = head
        self.logits_processor = _processor()

    def compute_logits(self, h):
        return self.logits_processor(self.lm_head, h)


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setattr(dlh, "quantized_matmul", _xla_qmm)
    g = torch.Generator().manual_seed(0)
    # Served-like head: rows of different norms (per-row scales matter).
    w = torch.randn(V, H, generator=g) * 0.02
    w = w * (0.5 + torch.rand(V, 1, generator=g))
    target = _Head(w.to(torch.bfloat16))
    draft = _Draft(target)
    hidden = (torch.randn(256, H, generator=g) * 2.0).to(torch.bfloat16)
    return target, draft, hidden


def _bind(monkeypatch, draft, on):
    monkeypatch.setattr(envs, "VLLM_TPU_DRAFT_LMHEAD_FP8", on)
    from vllm_torchtpu.models.vllm.qwen4_exp.mtp import Qwen4ExpMTP

    return Qwen4ExpMTP.bind_draft_lm_head(draft)  # the model's own hook


def test_off_by_default():
    assert envs.environment_variables["VLLM_TPU_DRAFT_LMHEAD_FP8"]() is False


def test_flag_off_binds_nothing(monkeypatch, setup):
    target, draft, hidden = setup
    base = draft.compute_logits(hidden)
    assert _bind(monkeypatch, draft, False) is False
    assert draft.lm_head is target and dlh.bf16_lm_head(draft) is target
    assert not hasattr(draft, "bf16_lm_head")
    assert torch.equal(draft.compute_logits(hidden), base)  # bitwise


def test_flag_on_keeps_target_head_bitwise(monkeypatch, setup):
    target, draft, hidden = setup
    w0 = target.weight.detach().clone()
    lp = _processor()
    t0 = lp(target, hidden)
    params0 = {n for n, _ in draft.named_parameters()}
    assert _bind(monkeypatch, draft, True) is True
    assert isinstance(draft.lm_head, dlh.Fp8DraftLMHead)
    assert draft.lm_head is not target
    assert draft.bf16_lm_head is target and dlh.bf16_lm_head(draft) is target
    assert isinstance(target.quant_method, UnquantizedEmbeddingMethod)
    assert target.weight.dtype == torch.bfloat16 and torch.equal(target.weight, w0)
    assert torch.equal(lp(target, hidden), t0)  # target logits bit-identical
    # The target's parameter is not re-registered under the draft.
    params1 = {n for n, _ in draft.named_parameters()}
    assert params0 == {"lm_head.weight"}
    assert params1 == {"lm_head.w_q", "lm_head.w_s"}
    # Idempotent: a second bind keeps the same copy and the same bf16 head.
    fp8 = draft.lm_head
    assert _bind(monkeypatch, draft, True) is True
    assert draft.lm_head is fp8 and draft.bf16_lm_head is target


def test_quantized_weight_layout_and_error(setup):
    target, _, _ = setup
    w_q, w_s = dlh.quantize_head_weight(target.weight.detach())
    assert w_q.shape == (H, V) and w_q.dtype == torch.float8_e4m3fn
    assert w_s.shape == (V,) and w_s.dtype == torch.float32
    w = target.weight.float()
    torch.testing.assert_close(w_s, w.abs().amax(dim=1) / 448, rtol=1e-6, atol=0)
    deq = w_q.float().t() * w_s[:, None]
    # E4M3: 3 mantissa bits -> relative rounding <= 2^-4 of each element.
    assert ((deq - w).abs() <= w.abs() / 16 + w_s[:, None] * 2**-9).all()


@pytest.mark.parametrize("family", ["gaussian", "peaked"])
def test_flag_on_draft_logits_close_and_top1(monkeypatch, setup, capsys, family):
    """gaussian: random hidden, top logits nearly tie (worst case); peaked: hidden
    near one head row (a confident draft, as accepted positions are)."""
    target, draft, hidden = setup
    if family == "peaked":
        g = torch.Generator().manual_seed(1)
        rows = torch.randint(0, V, (len(hidden),), generator=g)
        hidden = (target.weight[rows].float() * 40 + hidden.float() / 2).to(torch.bfloat16)
    ref = draft.compute_logits(hidden).float()
    _bind(monkeypatch, draft, True)
    got = draft.compute_logits(hidden)
    assert got.shape == ref.shape and got.dtype == torch.bfloat16
    got = got.float()
    rel = ((got - ref).norm(dim=-1) / ref.norm(dim=-1)).max().item()
    agree = (got.argmax(-1) == ref.argmax(-1)).float().mean().item()
    with capsys.disabled():
        print(f"\n[fp8 draft lm_head] {family} rows={len(ref)} "
              f"top1_agreement={agree:.4f} max_rel_l2={rel:.4f}")
    assert rel < 0.06
    assert agree >= (0.99 if family == "peaked" else 0.9)


def test_head_attrs_copied_for_shard_argmax(monkeypatch, setup):
    target, draft, _ = setup
    _bind(monkeypatch, draft, True)
    for name in dlh._HEAD_ATTRS:
        assert getattr(draft.lm_head, name) == getattr(target, name)


@pytest.mark.parametrize("rows", [4, 8, 16, 24, 32, 48, 64])
def test_traced_shapes_unchanged(monkeypatch, setup, rows):
    """Warmup keys follow the draft head's input/output shapes; both heads map
    [rows, H] bf16 to [rows, V] bf16, so no compile key is added or removed."""
    target, draft, hidden = setup
    ref = draft.compute_logits(hidden[:rows])
    _bind(monkeypatch, draft, True)
    got = draft.compute_logits(hidden[:rows])
    assert (got.shape, got.dtype) == (ref.shape, ref.dtype)


def test_flag_in_compile_key(monkeypatch):
    from vllm_torchtpu.compilation import tpu_compiler

    assert "VLLM_TPU_DRAFT_LMHEAD_FP8" not in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    monkeypatch.delenv("VLLM_TPU_DRAFT_LMHEAD_FP8", raising=False)
    off = tpu_compiler._tpu_compile_env_factors()
    monkeypatch.setenv("VLLM_TPU_DRAFT_LMHEAD_FP8", "1")
    on = tpu_compiler._tpu_compile_env_factors()
    assert off["VLLM_TPU_DRAFT_LMHEAD_FP8"] is False
    assert on["VLLM_TPU_DRAFT_LMHEAD_FP8"] is True
    assert {k for k in on if on[k] != off.get(k)} == {"VLLM_TPU_DRAFT_LMHEAD_FP8"}


def test_scale_is_an_owned_buffer(monkeypatch, setup):
    """quantize_tensor's scale is a view (of a [V, 1, 1] tensor), which costs torch_tpu
    an execute_as_strided program before every compiled call that reads it. The head's
    scale owns its storage and holds the same values."""
    from vllm_torchtpu.layers.core.quantization import quantize_tensor

    target, draft, _ = setup
    w = target.weight.detach()
    _, s = quantize_tensor(w.to(torch.float32).t(), torch.float8_e4m3fn, axis=0)
    view = s.reshape(V).contiguous()  # the scale as bound before
    assert view._base is not None
    _, w_s = dlh.quantize_head_weight(w)
    assert w_s._base is None
    assert w_s.untyped_storage().data_ptr() != view.untyped_storage().data_ptr()
    assert w_s.shape == (V,) and w_s.dtype == torch.float32 and w_s.is_contiguous()
    assert torch.equal(w_s, view)
    _bind(monkeypatch, draft, True)
    assert torch.equal(draft.lm_head.w_s, view)
