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
"""`case_compose` merges per-request-count variants
into one module that computes only the step's live requests, keeps the donated
state aliased without a copy, and matches the full variant."""

import re

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.extend.backend  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax._src.lib import xla_client as xc  # noqa: E402

from vllm_torchtpu.experimental.export import case_compose  # noqa: E402

SLOTS, KVD, SMAX, BLOCK, HID, V = 4096, 64, 16, 8, 64, 128
NS = (1, 2, 4, 8, 16)


def _draft(n, req_dist, slots, x, w, wout, kv):
    rows = n * BLOCK
    h = jax.nn.relu(x[:rows] @ w)
    tok = jnp.argmax(h @ wout, -1).astype(jnp.int32).reshape(n, BLOCK)[:, 1:]
    tok = jnp.where((jnp.arange(n) < req_dist[2])[:, None], tok, 0)
    return kv.at[slots[:rows]].set(h), jnp.pad(tok, ((0, SMAX - n), (0, 0)))


@pytest.fixture(scope="module")
def inputs():
    r = np.random.default_rng(0)
    return dict(
        slots=jnp.array(r.permutation(SLOTS)[: SMAX * BLOCK], jnp.int32),
        x=jnp.array(r.standard_normal((SMAX * BLOCK, HID)), jnp.float32),
        w=jnp.array(r.standard_normal((HID, KVD)) / 8, jnp.float32),
        wout=jnp.array(r.standard_normal((KVD, V)) / 8, jnp.float32),
    )


def _variant(n, inputs):
    kv = jnp.zeros((SLOTS, KVD), jnp.float32)
    # JAX marks the donated state as aliased to output 0, as the exporter's
    # `set_aliasing_outputs` does for TorchTPU modules.
    return (
        jax.jit(lambda *a: _draft(n, *a), donate_argnums=(5,))
        .lower(
            jnp.zeros(3, jnp.int32),
            inputs["slots"],
            inputs["x"],
            inputs["w"],
            inputs["wout"],
            kv,
        )
        .as_text()
    )


def test_composed_draft_matches_the_full_one_without_copying_the_state(inputs) -> None:
    merged = case_compose.compose_by_request_count(
        {n: _variant(n, inputs) for n in NS}, req_dist_arg=0
    )
    backend = jax.extend.backend.get_backend("cpu")
    exe = backend.compile_and_load(
        merged, xc.DeviceList(tuple(jax.devices("cpu")[:1])), xc.CompileOptions()
    )
    hlo = exe.hlo_modules()[0].to_string()
    assert hlo.count(" conditional(") == 1
    assert "input_output_alias={ {0}" in hlo
    state = f"[{SLOTS},{KVD}]"
    assert not [ln for ln in hlo.splitlines() if " copy(" in ln and state in ln]

    full = jax.jit(lambda *a: _draft(SMAX, *a))
    dev = jax.devices("cpu")[0]
    for s in (1, 2, 3, 5, 8, 9, 16):
        rd = jnp.array([0, 0, s], jnp.int32)
        args = [rd, inputs["slots"], inputs["x"], inputs["w"], inputs["wout"]]
        out = exe.execute_sharded(
            [
                jax.device_put(a, dev)
                for a in [*args, jnp.zeros((SLOTS, KVD), jnp.float32)]
            ]
        ).disassemble_into_single_device_arrays()
        kv_ref, tok_ref = full(*args, jnp.zeros((SLOTS, KVD), jnp.float32))
        live = np.asarray(inputs["slots"][: s * BLOCK])
        assert np.array_equal(np.asarray(out[1][0]), np.asarray(tok_ref)), s
        np.testing.assert_allclose(
            np.asarray(out[0][0])[live], np.asarray(kv_ref)[live], atol=1e-5
        )


def test_variants_must_share_a_signature(inputs) -> None:
    other = re.sub(r"tensor<3xi32>", "tensor<4xi32>", _variant(1, inputs))
    with pytest.raises(ValueError, match="signature"):
        case_compose.compose_by_request_count(
            {1: other, 16: _variant(16, inputs)}, req_dist_arg=0
        )
