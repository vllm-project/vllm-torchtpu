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
"""TPU_SAMPLER_TOP_P_KERNEL: the dense-row top-p recovery kernel (interpret
mode) against `rejection_sampler._recover_tokens`'s XLA path."""

import numpy as np
import pytest
import torch

jnp = pytest.importorskip("jax.numpy")

from vllm_torchtpu.kernels import top_p_sample as kernel  # noqa: E402
from vllm_torchtpu.layers.adapter.sample import rejection_sampler as rs  # noqa: E402


def _kernel_tokens(logits, drafts, temps, top_p, u):
    rows, vocab = logits.shape
    shape = (rows, vocab // 128, 128)
    params = np.stack([temps.numpy(), top_p.numpy()]).astype(np.float32)
    out = kernel.top_p_rows(
        jnp.asarray(logits.float().numpy().reshape(shape)).astype(jnp.bfloat16),
        jnp.asarray(u.numpy().reshape(shape)),
        jnp.asarray(params),
        jnp.asarray(drafts.numpy().astype(np.int32)),
        interpret=True,
    )
    return torch.from_numpy(np.asarray(out)[:rows, 0].copy())


@pytest.mark.parametrize("rows", [1, 5, 8])
def test_kernel_matches_xla_recovery(rows):
    torch.manual_seed(rows)
    vocab = 4 * 128 + 128
    logits = (torch.randn(rows, vocab) * 3).bfloat16()
    drafts = torch.randint(-1, vocab, (rows,))
    temps = torch.tensor([0.0, 0.7, 1.0, 1.3, 0.5, 2.0, 0.9, 1.1][:rows])
    top_p = torch.tensor([1.0, 0.9, 0.95, 0.5, 1.0, 0.8, 0.99, 0.7][:rows])
    u = torch.rand(rows, vocab)
    want = rs._recover_tokens(
        logits.float(),
        drafts,
        temps.view(-1, 1),
        torch.zeros(rows, 1, dtype=torch.int32),
        top_p.view(-1, 1),
        u,
        0,
        True,
        None,
        apply_top_k=False,
    ).to(torch.int32)
    got = _kernel_tokens(logits, drafts, temps, top_p, u)
    assert torch.equal(got, want.view(-1)), (got, want)


def test_dispatch_off_by_default(monkeypatch):
    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.adapter.sample import top_p_sample

    monkeypatch.delenv("TPU_SAMPLER_TOP_P_KERNEL", raising=False)
    assert envs.TPU_SAMPLER_TOP_P_KERNEL is False
    assert top_p_sample.arm() is False
    assert not top_p_sample.applies(torch.zeros(2, 256))
