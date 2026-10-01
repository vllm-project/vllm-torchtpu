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
"""State buffer layout and decode tile the V3 wrapper traces per config."""

import jax
import jax.numpy as jnp
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.gdn.v3 import config, wrapper

pytestmark = pytest.mark.cpu_test

_D_K, _D_V, _KERNEL_SIZE = 128, 128, 4


@pytest.fixture(autouse=True)
def _v7x_info(monkeypatch):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)


def _pallas_eqns(closed_jaxpr):
    eqns = []

    def walk(jaxpr):
        for eqn in jaxpr.eqns:
            if eqn.primitive.name == "pallas_call":
                eqns.append(eqn)
            for param in eqn.params.values():
                inner = getattr(param, "jaxpr", None)
                if inner is not None:
                    walk(getattr(inner, "jaxpr", inner))

    walk(closed_jaxpr.jaxpr)
    return eqns


def _traced_configs(n_kq, n_v, num_seqs, num_spec_tokens, attention_mode):
    """{pass name: (tile_size, split_state)} of the traced kernels."""
    is_kda = attention_mode == config.AttentionMode.KDA
    dim = n_kq * _D_K * 2 + n_v * _D_V
    gate_dim = n_v * _D_K if is_kda else n_v
    rows = num_seqs * (num_spec_tokens + 1)

    def run():
        return wrapper.fused_conv1d_gdn(
            jnp.zeros((rows, dim)),
            jnp.zeros((rows, n_v)),
            jnp.zeros((rows, gate_dim)),
            jnp.zeros((num_seqs + 1, _KERNEL_SIZE - 1, dim)),
            jnp.zeros((num_seqs + 1, n_v, _D_K, _D_V)),
            jnp.zeros((dim, 1, _KERNEL_SIZE)),
            None,
            jnp.zeros((n_v,)),
            jnp.zeros((n_v, _D_K) if is_kda else (n_v,)),
            jnp.zeros((num_seqs + 1,), jnp.int32),
            jnp.zeros((num_seqs,), jnp.int32),
            jnp.zeros((3,), jnp.int32),
            jnp.zeros((num_seqs,), jnp.int32),
            jnp.zeros((num_seqs,), jnp.int32),
            n_kq=n_kq,
            n_v=n_v,
            d_k=_D_K,
            d_v=_D_V,
            kernel_size=_KERNEL_SIZE,
            attention_mode=attention_mode,
            num_spec_tokens=num_spec_tokens,
        )

    out = {}
    for eqn in _pallas_eqns(jax.make_jaxpr(run)()):
        meta = eqn.params["metadata"]
        kind = "batched" if "batched" in eqn.params["name"] else "per_seq"
        out[kind] = (meta["tile_size"], meta["split_state"])
    return out


GDN, KDA = config.AttentionMode.GDN, config.AttentionMode.KDA


@pytest.mark.parametrize(
    ("n_kq", "n_v", "num_seqs", "num_spec_tokens", "mode", "batched"),
    [
        # 64 v-heads: tile 4 does not fit split buffers, tile 2 does.
        (16, 64, 32, 0, GDN, (2, True)),
        (2, 8, 64, 0, GDN, (4, True)),
        (2, 8, 16, 3, GDN, (4, True)),
        # KDA keeps its tile (one sequence per verify window).
        (8, 8, 16, 0, KDA, (4, True)),
        (8, 8, 16, 3, KDA, (1, True)),
        (64, 64, 32, 0, KDA, (4, False)),
    ],
)
def test_state_buffer_selection(n_kq, n_v, num_seqs, num_spec_tokens, mode, batched):
    got = _traced_configs(n_kq, n_v, num_seqs, num_spec_tokens, mode)
    assert got["batched"] == batched
    assert got["per_seq"][1]
