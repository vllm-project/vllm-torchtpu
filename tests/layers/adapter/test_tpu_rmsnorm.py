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
"""TPU_FUSED_ADD_RMSNORM torch bridge: the OOT RMSNorm vs the incumbent.

The incumbent is upstream ``forward_native`` compiled by the TPU backend --
the graph the served model runs -- so the comparison covers whatever bf16
roundings XLA keeps or drops there. Importing ``layers.adapter.rmsnorm``
registers the OOT class for the whole process.
"""

import pytest

torch = pytest.importorskip("torch")

from vllm_torchtpu.layers.adapter import rmsnorm as tpu_rmsnorm  # noqa: E402

pytestmark = pytest.mark.tpu_test

MAX_DIFF_FRACTION = 1e-3


def _bits_diff(a, b):
    return (a.view(torch.int16).int() - b.view(torch.int16).int()).abs()


def _norm(hidden=6144, eps=1e-5):
    from vllm.model_executor.layers.layernorm import RMSNorm

    n = RMSNorm(hidden, eps=eps)
    assert isinstance(n, tpu_rmsnorm.TpuRMSNorm)
    with torch.no_grad():
        n.weight.copy_(1 + 0.2 * torch.randn(hidden))
    return n.to(torch.bfloat16).to("tpu")


def _assert_close(got, ref, has_res):
    if has_res:
        assert torch.equal(got[1].cpu(), ref[1].cpu()), "residual must be bit-identical"
        d = _bits_diff(got[0].cpu(), ref[0].cpu())
    else:
        d = _bits_diff(got.cpu(), ref.cpu())
    frac = (d > 0).float().mean().item()
    assert d.max().item() <= 2 and frac <= MAX_DIFF_FRACTION, (d.max().item(), frac)


@pytest.mark.parametrize("rows", [1024, 4096])
@pytest.mark.parametrize("has_res", [True, False])
def test_oot_forward_matches_compiled_native(vllm_config_context, rows, has_res):
    torch.manual_seed(rows)
    torch._dynamo.reset()
    n = _norm()
    x = (0.7 * torch.randn(rows, 6144)).to(torch.bfloat16).to("tpu")
    r = (4 * torch.randn(rows, 6144)).to(torch.bfloat16).to("tpu") if has_res else None
    assert n._use_pallas(x, r)
    got = n.forward(x, r)

    def native(a, b):
        return n.forward_native(a, b)

    compiled = torch.compile(native, fullgraph=True, dynamic=False)
    ref = compiled(x.clone(), None if r is None else r.clone())
    _assert_close(got, ref, has_res)


@pytest.mark.parametrize("has_res", [True, False])
def test_oot_forward_matches_eager_native(vllm_config_context, has_res):
    torch.manual_seed(3)
    n = _norm()
    x = (0.7 * torch.randn(2048, 6144)).to(torch.bfloat16).to("tpu")
    r = (4 * torch.randn(2048, 6144)).to(torch.bfloat16).to("tpu") if has_res else None
    got = n.forward(x, r)
    ref = n.forward_native(x.clone(), None if r is None else r.clone())
    _assert_close(got, ref, has_res)


def test_decode_sized_batches_keep_upstream_path(vllm_config_context):
    n = _norm()
    x = torch.randn(16, 6144).to(torch.bfloat16).to("tpu")
    assert not n._use_pallas(x, x.clone())
    # a strided slice (the q/kv latent norm pattern) is not taken either
    wide = torch.randn(1024, 6144 + 512).to(torch.bfloat16).to("tpu")
    assert not n._use_pallas(wide[:, :6144], None)


def test_small_hidden_not_taken(vllm_config_context):
    n = _norm(hidden=512)
    x = torch.randn(4096, 512).to(torch.bfloat16).to("tpu")
    assert not n._pallas_ok and not n._use_pallas(x, None)


def test_keep_upstream_path_pins_a_traced_once_model(vllm_config_context):
    model = torch.nn.Sequential(_norm(), _norm())
    assert tpu_rmsnorm.keep_upstream_path(model) == 2
    x = torch.randn(4096, 6144).to(torch.bfloat16).to("tpu")
    assert not any(m._use_pallas(x, None) for m in model)
