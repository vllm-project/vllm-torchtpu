# SPDX-License-Identifier: Apache-2.0
"""CPU dispatch-envelope checks; device numerics use routing_integration."""

import ast
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.experimental.push_fused_moe import routing

ROOT = Path(__file__).resolve().parents[4]


def supported(tokens=288, **changes):
    # Evaluate the actual production predicate without running TPU kernels.
    tree = ast.parse(Path(routing.__file__).read_text())
    predicate = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "supported" for t in node.targets)
    )
    i32 = jax.ShapeDtypeStruct((1,), jnp.int32)
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
    return eval(
        compile(ast.Expression(predicate), "routing-gate", "eval"),
        vars(routing),
        values,
    )


@pytest.fixture(autouse=True)
def tpu_v7(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    monkeypatch.setattr(routing, "get_tpu_version", lambda: 7)


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
    monkeypatch.setattr(routing, "get_tpu_version", lambda: version)
    assert not supported()


def test_frozen_module_identity_and_t288(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-c",
            r"""
import ast
import sys
from pathlib import Path
sys.path.insert(0, 'scripts/dsv41_bench')
import routing_integration as h
baseline = h.load_production_baseline()
candidate = h.load_routing(h.ROUTING)
assert baseline is not candidate
assert baseline.prepare_routing.__globals__ is vars(baseline)
assert candidate.prepare_routing.__globals__ is vars(candidate)
assert baseline._position_metadata.__globals__ is vars(baseline)
assert baseline.prepare_routing.__kwdefaults__['fuse_metadata'] is True
h.configure_shape(288)
out = Path(sys.argv[1])
for name, module in [('baseline', baseline), ('candidate', candidate)]:
    fn, _ = h.extract(h.LAYER, h.bench.STRIDE, out, routing=module, tag=name)
    bound = fn.__globals__['prepare_routing']
    assert bound.func is module.prepare_routing
    assert bound.keywords['fuse_metadata'] is True
# Aside from the supported predicate, all production statements are unchanged.
def normalized(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == 'supported' for t in node.targets
        ):
            node.value = ast.Constant(True)
    return ast.dump(tree)
assert normalized(h.ROUTING) == normalized(h.BASELINE_ROUTING)
bad = out / 'bad.py'
bad.write_bytes(h.BASELINE_ROUTING.read_bytes() + b'\n')
try:
    h.load_production_baseline(bad)
except ValueError:
    pass
else:
    raise AssertionError('unvalidated baseline accepted')
""",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=True,
    )
