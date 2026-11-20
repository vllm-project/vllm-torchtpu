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
"""The e4m3 LM head (TPU_FP8_LM_HEAD): kernel numerics in interpret mode,
the per-row quantization, the jax_op's fake under a symbolic row count and
under Dynamo, and the method's apply/guards."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import (
    DimDynamic,
    ShapeEnv,
    StatelessSymbolicContext,
)

from vllm_torchtpu.kernels import fp8_lm_head as kernel

HIDDEN, VOCAB = 256, 4096


def _weights(seed=0):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((VOCAB, HIDDEN)).astype(np.float32)
    w *= rng.uniform(0.01, 2.0, size=(VOCAB, 1)).astype(np.float32)
    return w


@pytest.mark.parametrize("rows", [1, 8, 16, 600])
def test_kernel_matches_reference(rows):
    w = _weights()
    amax = np.abs(w).max(axis=1, keepdims=True)
    scale = (amax / kernel.E4M3_MAX).astype(np.float32)
    q = jnp.asarray(w / scale).astype(jnp.float8_e4m3fn)
    x = jnp.asarray(
        np.random.default_rng(1).standard_normal((rows, HIDDEN)), jnp.bfloat16
    )
    out = kernel.fp8_lm_head(x, q, jnp.asarray(scale.reshape(1, -1)), interpret=True)
    assert out.shape == (rows, VOCAB) and out.dtype == jnp.bfloat16
    ref = (
        jnp.dot(x.astype(jnp.float32), q.astype(jnp.float32).T)
        * jnp.asarray(scale.reshape(1, -1))
    )
    np.testing.assert_allclose(
        np.asarray(out, np.float32), np.asarray(ref), rtol=2e-2, atol=2e-2
    )
    # Against the exact bf16 head: e4m3 weight rounding only (~3-6 %).
    exact = jnp.dot(x.astype(jnp.float32), jnp.asarray(w).T)
    err = np.linalg.norm(np.asarray(out, np.float32) - exact) / np.linalg.norm(exact)
    assert err < 0.06


def test_quantize_per_row():
    from vllm_torchtpu.layers.adapter.quantization.fp8_lm_head import quantize

    w = torch.from_numpy(_weights()).to(torch.bfloat16)
    q, scale = quantize(w)
    assert q.dtype == torch.float8_e4m3fn and q.shape == w.shape
    assert scale.shape == (1, VOCAB) and scale.dtype == torch.float32
    # Each row's amax lands on e4m3's largest finite value.
    np.testing.assert_allclose(
        q.float().abs().amax(dim=1).numpy(), kernel.E4M3_MAX, rtol=0
    )
    back = q.float() * scale.view(-1, 1)
    rel = (back - w.float()).norm() / w.float().norm()
    assert rel < 0.05
    # All-zero rows keep a finite scale.
    q0, s0 = quantize(torch.zeros(4, HIDDEN))
    assert torch.isfinite(s0).all() and (q0.float() == 0).all()


def _op():
    pytest.importorskip("torch_tpu")
    from vllm_torchtpu.layers.adapter.quantization import fp8_lm_head as method

    return method._op


def test_op_fake_symbolic_rows():
    op = _op()
    mode = FakeTensorMode(shape_env=ShapeEnv())
    seed = mode.from_tensor(
        torch.empty(16),
        symbolic_context=StatelessSymbolicContext(dynamic_sizes=[DimDynamic.DYNAMIC]),
    )
    with mode:
        rows = seed.shape[0]
        assert isinstance(rows, torch.SymInt)
        x = torch.empty((rows, HIDDEN), dtype=torch.bfloat16)
        w = torch.empty((VOCAB, HIDDEN), dtype=torch.float8_e4m3fn)
        s = torch.empty((1, VOCAB), dtype=torch.float32)
        out = op(x, w, s)
    assert out.shape == (rows, VOCAB) and out.dtype == torch.bfloat16
    assert isinstance(out.shape[0], torch.SymInt)


def test_op_dynamo():
    op = _op()
    x = torch.empty((16, HIDDEN), dtype=torch.bfloat16, device="meta")
    w = torch.empty((VOCAB, HIDDEN), dtype=torch.float8_e4m3fn, device="meta")
    s = torch.empty((1, VOCAB), dtype=torch.float32, device="meta")
    torch._dynamo.mark_dynamic(x, 0)
    torch._dynamo.reset()
    out = torch.compile(lambda *a: op(*a) + 1, backend="eager", fullgraph=True)(
        x, w, s
    )
    assert out.shape == (16, VOCAB)


def test_method_apply_reshapes_and_guards(monkeypatch):
    pytest.importorskip("torch_tpu")
    from vllm_torchtpu.layers.adapter.quantization import fp8_lm_head as method

    calls = []

    def fake_op(rows, w, s):
        calls.append(tuple(rows.shape))
        return torch.zeros((rows.shape[0], w.shape[0]), dtype=rows.dtype)

    monkeypatch.setattr(method, "_op", fake_op)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.zeros(VOCAB, HIDDEN, dtype=torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(torch.ones(1, VOCAB), requires_grad=False)
    m = method.Fp8LMHeadMethod()
    out = m.apply(layer, torch.zeros(2, 3, HIDDEN, dtype=torch.bfloat16))
    assert out.shape == (2, 3, VOCAB) and calls == [(6, HIDDEN)]
    with pytest.raises(NotImplementedError):
        m.apply(layer, torch.zeros(1, HIDDEN), bias=torch.zeros(VOCAB))
    with pytest.raises(NotImplementedError):
        m.embedding(layer, torch.zeros(1, dtype=torch.long))
    # Reload paths re-enter the hook on an already e4m3 weight: no-op.
    monkeypatch.setattr(
        method.UnquantizedEmbeddingMethod,
        "process_weights_after_loading",
        lambda self, layer: None,
    )
    before = layer.weight
    m.process_weights_after_loading(layer)
    assert layer.weight is before


def test_env_default_off(monkeypatch):
    from vllm_torchtpu import envs

    monkeypatch.delenv("TPU_FP8_LM_HEAD", raising=False)
    monkeypatch.delenv("TPU_MINIMAX_M3_FP8_DRAFT_LINEARS", raising=False)
    assert envs.TPU_FP8_LM_HEAD is False
    assert envs.TPU_MINIMAX_M3_FP8_DRAFT_LINEARS is False
