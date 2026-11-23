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
"""TPU_LM_HEAD_E4M3: e4m3 LM head with a per-row scale."""

import torch
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)

from vllm_torchtpu.layers.adapter import lm_head_e4m3


class _Head(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.weight = torch.nn.Parameter(w, requires_grad=False)
        self.quant_method = UnquantizedEmbeddingMethod()


class _Model(torch.nn.Module):
    def __init__(self, head):
        super().__init__()
        self.lm_head = head
        self.logits_processor = type("P", (), {"head_dtype": None})()


def _w(rows=96, hidden=128, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(rows, hidden, generator=g) * 0.02
    w[3] *= 50  # rows with very different ranges
    w[5] = 0  # an all-zero row
    return w.to(torch.bfloat16)


def test_quantize_rows_roundtrip():
    w = _w()
    q, s = lm_head_e4m3.quantize_rows(w)
    assert q.dtype == torch.float8_e4m3fn and s.dtype == torch.float32
    back = q.float() * s[:, None]
    err = (back - w.float()).abs()
    # e4m3 has 3 mantissa bits: relative rounding <= 2^-4 of each row's amax.
    assert torch.all(err <= w.float().abs().amax(1, keepdim=True) * 2**-4 + 1e-12)
    assert torch.all(back[5] == 0)


def test_arm_and_logits_close():
    w = _w()
    model = _Model(_Head(w.clone()))
    x = (torch.randn(8, 128, generator=torch.Generator().manual_seed(1))).to(
        torch.bfloat16
    )
    ref = model.lm_head.quant_method.apply(model.lm_head, x)
    draft = type("D", (), {"lm_head": model.lm_head, "logits_processor": None})()
    assert lm_head_e4m3.arm(model, draft)
    assert model.lm_head.weight.dtype == torch.float8_e4m3fn
    out = model.lm_head.quant_method.apply(model.lm_head, x)
    assert out.dtype == ref.dtype and out.shape == ref.shape
    rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
    assert rel < 0.05
    # exact against the dequantized weight in f32
    deq = model.lm_head.weight.float() * model.lm_head.weight_row_scale[:, None]
    exact = (x.float() @ deq.t()).to(torch.bfloat16)
    assert ((out.float() - exact.float()).abs().max() <= exact.float().abs().max() * 2**-7)


def test_arm_refuses_own_draft_head():
    model = _Model(_Head(_w()))
    draft = type("D", (), {"lm_head": _Head(_w(seed=2)), "logits_processor": None})()
    assert not lm_head_e4m3.arm(model, draft)
    assert model.lm_head.weight.dtype == torch.bfloat16


def test_arm_refuses_head_dtype():
    model = _Model(_Head(_w()))
    model.logits_processor.head_dtype = torch.float32
    assert not lm_head_e4m3.arm(model, None)


def test_arm_accepts_bf16_head_dtype():
    model = _Model(_Head(_w()))
    model.logits_processor.head_dtype = torch.bfloat16
    assert lm_head_e4m3.arm(model, None)


def test_split_logits_matches_upcast_weight():
    """The split-e4m3 fp8 matmul equals the upcast-weight matmul up to the
    bf16 output rounding; the high term alone (per-token W8A8) does not."""
    import jax.numpy as jnp
    import numpy as np

    w = _w(rows=512, hidden=768)
    q, s = lm_head_e4m3.quantize_rows(w)
    g = torch.Generator().manual_seed(3)
    x = torch.randn(16, 768, generator=g)
    x[:, :8] *= 20  # outlier channels
    x = x.to(torch.bfloat16)
    deq = q.double() * s.double()[:, None]
    oracle = x.double() @ deq.t()
    upcast = (x.float() @ deq.float().t()).to(torch.bfloat16).double()

    def rel(a):
        return ((a - oracle).norm() / oracle.norm()).item()

    qj = jnp.asarray(q.view(torch.uint8).numpy()).view(jnp.float8_e4m3fn)
    xj = jnp.asarray(x.float().numpy()).astype(jnp.bfloat16)
    out = lm_head_e4m3.split_e4m3_logits(xj, qj, jnp.asarray(s.numpy()))
    out = torch.from_numpy(np.asarray(out.astype(jnp.float32))).double()
    assert rel(out) <= 2 * rel(upcast) + 1e-4
    # Positive control: dropping the low term is a 4-bit activation.
    xf = x.double()
    sc = torch.exp2(torch.ceil(torch.log2(xf.abs().amax(1, keepdim=True) / 448)))
    hi = (xf / sc).float().to(torch.float8_e4m3fn).double() * sc
    assert rel(hi @ deq.t()) > 5 * rel(out)
