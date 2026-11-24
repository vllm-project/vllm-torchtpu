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
"""CPU checks of the residual add + RMSNorm epilogue of qar_all_reduce
(TPU_TP_QAR_FUSE_NORM): the Pallas body (interpret mode) and its XLA form
against the torch rounding points of the TPU fused_add_rms_norm provider."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.experimental import pallas as pl

from vllm_torchtpu.kernels.collectives import qar_all_reduce as qar

H = 6144
EPS = 1e-6


def _inputs(rows, seed=0):
    rng = np.random.default_rng(seed)
    ar = (rng.standard_normal((rows, H)) * 3).astype(np.float32)
    res = (rng.standard_normal((rows, H)) * 20).astype(np.float32)
    w = (1.0 + rng.standard_normal(H) * 0.2).astype(np.float32)
    bf = lambda a: jnp.asarray(a).astype(jnp.bfloat16)
    return bf(ar), bf(res), jnp.asarray(w)


def _torch_ref(ar, res, w):
    """layers/adapter/ir_ops.py fused_add_rms_norm (TPU provider) math."""
    t = lambda a: torch.from_numpy(np.array(a.astype(jnp.float32))).to(torch.bfloat16)
    x, r = t(ar), t(res)
    nr = (x.to(torch.float32) + r.to(torch.float32)).to(torch.bfloat16)
    xf = nr.to(torch.float32)
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    out = xf * torch.rsqrt(var + EPS) * torch.from_numpy(np.asarray(w))
    return out.to(torch.bfloat16).to(torch.float32).numpy(), nr.to(torch.float32).numpy()


def _ulps(got, ref):
    got, ref = np.asarray(got, np.float64), np.asarray(ref, np.float64)
    return float(np.max(np.abs(got - ref) / (np.maximum(np.abs(ref), 1e-30) * 2.0**-8)))


def _pallas_epilogue(ar, res, w):
    rows = ar.shape[0]
    r = qar._NORM_ROWS
    spec = pl.BlockSpec((r, H), lambda i: (i, 0))
    return pl.pallas_call(
        qar._add_norm_body(EPS, H),
        grid=(rows // r,),
        in_specs=[spec, spec, pl.BlockSpec((1, H), lambda i: (0, 0))],
        out_specs=[spec, spec],
        out_shape=[jax.ShapeDtypeStruct((rows, H), jnp.bfloat16)] * 2,
        interpret=True,
    )(ar, res, w.reshape(1, H))


@pytest.mark.parametrize("rows", [128, 256])
def test_xla_form_matches_torch_provider(rows):
    ar, res, w = _inputs(rows)
    normed, nr = qar.add_norm_reference(ar, res, w, EPS)
    t_norm, t_nr = _torch_ref(ar, res, w)
    assert np.array_equal(np.asarray(nr.astype(jnp.float32)), t_nr)
    assert _ulps(normed.astype(jnp.float32), t_norm) <= 2.0


@pytest.mark.parametrize("rows", [128, 192])
def test_pallas_epilogue_matches_xla_form(rows):
    ar, res, w = _inputs(rows, seed=rows)
    o_res, o_norm = _pallas_epilogue(ar, res, w)
    normed, nr = qar.add_norm_reference(ar, res, w, EPS)
    assert np.array_equal(np.asarray(o_res), np.asarray(nr))
    got, want = np.asarray(o_norm.astype(jnp.float32)), np.asarray(normed.astype(jnp.float32))
    assert _ulps(got, want) <= 2.0
    # Almost every element is bit-identical; the rest are 1-ulp ties.
    assert np.mean(got != want) < 1e-3


def test_padded_rows_stay_zero():
    ar, res, w = _inputs(64)
    z = jnp.zeros_like(ar)
    o_res, o_norm = _pallas_epilogue(z, z, w)
    assert not np.any(np.asarray(o_res.astype(jnp.float32)))
    assert not np.any(np.asarray(o_norm.astype(jnp.float32)))


def test_fault_is_caught():
    """Positive control: a wrong eps or weight must fail the 2-ulp gate."""
    ar, res, w = _inputs(128)
    normed, _ = qar.add_norm_reference(ar, res, w * 1.01, EPS)
    t_norm, _ = _torch_ref(ar, res, w)
    assert _ulps(normed.astype(jnp.float32), t_norm) > 2.0
