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
"""CPU checks for exact routing dispatch eligibility."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.experimental import decode_preparation as prep
from vllm_torchtpu.kernels.experimental.push_fused_moe import routing


def supported(tokens=288, **changes):
    i32 = jnp.zeros((1,), jnp.int32)
    gathered = SimpleNamespace(
        topk=6, n_tokens=tokens * 16, local_blocks=i32, block_hist=i32
    )
    values = dict(
        fuse_metadata=True,
        t_local=tokens,
        gathered=gathered,
        e_total=384,
        ep=16,
        tile_m=128,
        block=routing.host.routing_block(tokens, 6),
        shard_stride=routing.host.ragged_stride_bound(tokens * 16, 6, 384, 128),
        runs=(i32,) * 4,
        me=i32,
    )
    for key in ("topk", "n_tokens", "local_blocks", "block_hist"):
        if key in changes:
            setattr(gathered, key, changes.pop(key))
    values.update(changes)
    try:
        routing.prepare_routing(**values)
    except DispatchReached as exc:
        return str(exc) == "fused"
    raise AssertionError("routing did not reach either dispatch path")


class DispatchReached(Exception):
    pass


def stop(path):
    def reached(*args, **kwargs):
        raise DispatchReached(path)

    return reached


@pytest.fixture(autouse=True)
def tpu_v7(monkeypatch):
    monkeypatch.setattr(routing, "build_routing_tables_sharded", lambda *a, **k: None)
    monkeypatch.setattr(routing.host, "local_slab_rows", lambda *a, **k: None)
    monkeypatch.setattr(
        routing.host, "shard_transport_tables_in_blocks", stop("fallback")
    )
    monkeypatch.setattr(routing, "fuse_function", stop("fused"))
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    monkeypatch.setattr(prep, "get_tpu_version", lambda: 7)


@pytest.mark.parametrize("tokens", [96, 192, 256, 288, 384])
def test_exact_approved_tokens(tokens):
    assert supported(tokens)
    stride = routing.host.ragged_stride_bound(tokens * 16, 6, 384, 128)
    assert not supported(tokens, shard_stride=stride + 128)
    assert not supported(tokens, block=routing.host.routing_block(tokens, 6) + 128)


@pytest.mark.parametrize(
    "tokens", [1, 95, 97, 128, 191, 193, 255, 257, 287, 289, 383, 385, 512]
)
def test_unapproved_tokens(tokens):
    assert not supported(tokens)


@pytest.mark.parametrize(
    "changes",
    [
        {"fuse_metadata": False},
        {"topk": 4},
        {"e_total": 383},
        {"ep": 8},
        {"tile_m": 256},
        {"n_tokens": 288 * 16 + 1},
    ],
)
def test_contract_guards(changes):
    assert not supported(**changes)


@pytest.mark.parametrize("field", ["me", "local_blocks", "block_hist", "runs"])
def test_dtype_guards(field):
    bad = jax.ShapeDtypeStruct((1,), jnp.int16)
    assert not supported(**{field: (bad,) * 4 if field == "runs" else bad})


@pytest.mark.parametrize("backend,version", [("cpu", 7), ("gpu", 7), ("tpu", 6)])
def test_backend_guards(monkeypatch, backend, version):
    monkeypatch.setattr(jax, "default_backend", lambda: backend)
    monkeypatch.setattr(prep, "get_tpu_version", lambda: version)
    assert not supported()
