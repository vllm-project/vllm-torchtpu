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
"""TPU_DCP_MERGE_VSPACE_FUSED: the chunked merge with its operand
built in V space equals the latent merge followed by W_UV."""

import numpy as np
import pytest
import torch

from vllm_torchtpu.layers.adapter import cp_mla_attention as cp_mla

D, T, LOCAL, C, V = 4, 16, 2, 32, 16
H = D * LOCAL


def _operand(o, lse_r, lses, w_uv):
    """Rank r's heads-major V-space operand [H, T, V] in f64."""
    merged = torch.logsumexp(lses.double(), dim=0)
    w = torch.exp(lse_r.double() - merged)  # [T, H]
    v = torch.einsum("thc,hcv->htv", o.double(), w_uv.double())
    wt = w.transpose(0, 1).unsqueeze(-1)
    return torch.where(wt > 0, v * wt, torch.zeros_like(v))


class _Group:
    """all_gather by row match; reduce_scatter checks this rank's operand and
    returns its head block of the sum over every rank's operand."""

    def __init__(self, outputs, lses, w_uv, rank):
        self.world_size, self.rank_in_group = D, rank
        self.outputs, self.lses, self.w_uv = outputs, lses, w_uv
        self.done = 0  # rows already reduce-scattered (chunks run in order)

    def all_gather(self, tensor, dim=0):
        n = tensor.shape[1]
        own = self.lses[self.rank_in_group]
        for off in range(0, T - n + 1):
            if torch.equal(own[off : off + n], tensor[0]):
                return self.lses[:, off : off + n].clone()
        raise AssertionError("lse rows not found")

    def reduce_scatter(self, tensor, dim=0):
        assert dim == 0
        n = tensor.shape[1]
        rows = slice(self.done, self.done + n)
        self.done += n
        lses = self.lses[:, rows]
        ops = [
            _operand(self.outputs[r][rows], self.lses[r][rows], lses, self.w_uv)
            for r in range(D)
        ]
        torch.testing.assert_close(
            tensor.double(), ops[self.rank_in_group], rtol=1e-5, atol=1e-5
        )
        total = sum(ops).to(tensor.dtype)
        return total[self.rank_in_group * LOCAL : (self.rank_in_group + 1) * LOCAL]


@pytest.mark.parametrize("lse_ag", [False, True])
def test_vspace_merge_is_the_latent_merge_then_w_uv(monkeypatch, lse_ag):
    rng = np.random.default_rng(7)
    outputs = torch.from_numpy(rng.standard_normal((D, T, H, C)).astype(np.float32))
    lses = torch.from_numpy((rng.standard_normal((D, T, H)) * 3).astype(np.float32))
    lses[1, 3] = float("-inf")
    w_uv = torch.from_numpy(rng.standard_normal((H, C, V)).astype(np.float32))
    for k, val in {
        "TPU_DCP_MERGE_VSPACE_FUSED": True,
        "TPU_DCP_LSE_MERGE_A2A": False,
        "TPU_DCP_MERGE_CHUNKED": 2,
        "TPU_DCP_MERGE_CHUNKED_MIN_TOKENS": 0,
        "TPU_DCP_MERGE_CHUNKED_SMALL": 0,
        "TPU_DCP_MERGE_CHUNKED_LSE_AG": lse_ag,
        "TPU_DCP_MERGE_CHUNKED_LSE_AG_MAX_TOKENS": 0,
        "TPU_DCP_MLA_MERGE_PIECES": 0,
        "TPU_DCP_MERGE_RS_BF16": 0,
        "TPU_DCP_RS_OUT_PRECISION": False,
    }.items():
        monkeypatch.setattr(cp_mla.envs, k, val, raising=False)
    merged = torch.logsumexp(lses.double(), dim=0)
    w = torch.exp(lses.double() - merged)  # [D, T, H]
    latent = torch.einsum("dth,dthc->thc", w.nan_to_num(0), outputs.double())
    want = torch.einsum("thc,hcv->thv", latent, w_uv.double())
    for rank in range(D):
        group = _Group(outputs, lses, w_uv, rank)
        monkeypatch.setattr(cp_mla, "get_dcp_group", lambda g=group: g)
        assert cp_mla.vspace_fused_applies(T, True)
        assert not cp_mla.vspace_fused_applies(T, False)
        got = cp_mla.merge_lse_partials_scatter_heads_vspace(
            outputs[rank], lses[rank], w_uv, None
        )
        assert got.shape == (T, LOCAL, V)
        torch.testing.assert_close(
            got.double(),
            want[:, rank * LOCAL : (rank + 1) * LOCAL],
            rtol=1e-4,
            atol=1e-4,
        )


@pytest.mark.tpu_test
@pytest.mark.parametrize("scaled", [False, True])
def test_vspace_operand_kernel_matches_reference(scaled):
    import jax
    import jax.numpy as jnp

    from vllm_torchtpu.layers.adapter import dcp_vspace

    if jax.default_backend() != "tpu":
        pytest.skip("requires a TPU")
    t, h, c, v = 512, 32, 512, 256
    k = jax.random.split(jax.random.key(0), 4)
    o = jax.random.normal(k[0], (t, h, c)).astype(jnp.bfloat16)
    w = jax.nn.softmax(jax.random.normal(k[1], (t, h)) * 3, axis=-1)
    w = w.at[5].set(0.0)
    o = o.at[5].set(jnp.nan)  # zero-weight partial may be NaN
    w_uv = (jax.random.normal(k[2], (h, c, v)) * 0.05).astype(jnp.bfloat16)
    scale = (jax.random.uniform(k[3], (h, 1, v)) + 0.5) if scaled else None
    got = dcp_vspace.vspace_weighted_heads_major_jax(
        o, w, w_uv, scale, out_dtype=jnp.float32
    )
    of = jnp.nan_to_num(o.astype(jnp.float32))
    ref = jnp.einsum("thc,hcv->htv", of, w_uv.astype(jnp.float32))
    if scaled:
        ref = ref * scale
    ref = ref * jnp.transpose(w, (1, 0))[:, :, None]
    got, ref = np.asarray(got), np.asarray(ref)
    assert np.isfinite(got).all()
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    assert rel < 1e-2, rel
    bf = np.asarray(
        dcp_vspace.vspace_weighted_heads_major_jax(
            o, w, w_uv, scale, out_dtype=jnp.bfloat16
        )
    ).astype(np.float32)
    np.testing.assert_allclose(bf, got, rtol=1e-2, atol=1e-3)
