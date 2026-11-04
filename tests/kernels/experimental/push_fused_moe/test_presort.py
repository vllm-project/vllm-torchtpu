# SPDX-License-Identifier: Apache-2.0
"""CPU support-envelope and dispatch regressions; TPU numerics use the harness."""

import ast
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.experimental.push_fused_moe import presort


@pytest.fixture
def tpu_v7(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    monkeypatch.setattr(presort, "get_tpu_version", lambda: 7)


def inputs(tokens=288):
    return [
        jax.ShapeDtypeStruct((tokens * 6,), jnp.int32),
        jax.ShapeDtypeStruct((384,), jnp.int32),
        jax.ShapeDtypeStruct((tokens, 40, 128), jnp.float8_e4m3fn),
    ]


@pytest.mark.parametrize("tokens", [96, 192, 256, 288, 384])
def test_approved_tokens(tpu_v7, tokens):
    assert presort.supports_local_sort(*inputs(tokens), e_total=384, topk=6)


@pytest.mark.parametrize(
    "index,shape,dtype",
    [
        (0, (1727,), jnp.int32),
        (0, (288, 6), jnp.int32),
        (0, (1728,), jnp.int16),
        (1, (384,), jnp.int16),
        (1, (383,), jnp.int32),
        (2, (288, 40, 128), jnp.bfloat16),
        (2, (288, 40, 128), jnp.float8_e5m2),
        (2, (288, 20, 128), jnp.float8_e4m3fn),
        (2, (288, 40, 64), jnp.float8_e4m3fn),
        (2, (), jnp.float8_e4m3fn),
    ],
)
def test_rejects_other_contracts(tpu_v7, index, shape, dtype):
    xs = inputs()
    xs[index] = jax.ShapeDtypeStruct(shape, dtype)
    assert not presort.supports_local_sort(*xs, e_total=384, topk=6)


@pytest.mark.parametrize("tokens", [0, 95, 128, 193, 287, 385, 512])
def test_unapproved_tokens_fall_back(tpu_v7, monkeypatch, tokens):
    xs = inputs(tokens)
    assert not presort.supports_local_sort(*xs, e_total=384, topk=6)
    calls = []

    def fallback(*args, **kwargs):
        calls.append((args, kwargs))
        return "native"

    monkeypatch.setattr(presort.host, "local_sort_activations", fallback)
    assert presort.local_sort_activations(*xs, e_total=384, topk=6) == "native"
    assert calls == [(tuple(xs), {"e_total": 384, "topk": 6})]


@pytest.mark.parametrize("backend,version", [("cpu", 7), ("gpu", 7), ("tpu", 6)])
def test_backend_guard(monkeypatch, backend, version):
    monkeypatch.setattr(jax, "default_backend", lambda: backend)
    monkeypatch.setattr(presort, "get_tpu_version", lambda: version)
    assert not presort.supports_local_sort(*inputs(), e_total=384, topk=6)


@pytest.mark.parametrize("experts,topk", [(128, 6), (384, 4)])
def test_routing_guard(tpu_v7, experts, topk):
    assert not presort.supports_local_sort(*inputs(), e_total=experts, topk=topk)


def test_original_math_and_baseline_are_frozen(tpu_v7):
    # Importing the standalone harness installs minimal package stubs, so keep
    # its namespace changes out of the test runner's global module registry.
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[4]
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
from scripts.dsv41_bench import presort_metadata as bench
from vllm_torchtpu.kernels.experimental.push_fused_moe import presort
baseline = bench.load_production_baseline()
assert baseline is not presort
assert baseline.local_sort_activations.__globals__ is not vars(presort)
xs = bench.fixture('uniform', tokens=128)
actual = bench.jax.jit(lambda *a: presort.local_sort_activations(
    *a, e_total=384, topk=6))(*xs)
bench.assert_equal(actual, bench.reference(xs))
bench.jax.default_backend = lambda: 'tpu'
baseline.get_tpu_version = lambda: 7
for tokens in (96, 192, 256, 288, 384):
    xs = bench.fixture('uniform', tokens=tokens)
    assert baseline.supports_local_sort(*xs, e_total=384, topk=6) == (tokens == 288)
""",
        ],
        cwd=root,
        check=True,
    )
    frozen = root / "scripts/dsv41_bench/_presort_baseline_09b82cc91.py"

    def math_ast(path):
        return [
            ast.dump(node)
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.FunctionDef)
            and node.name in ("_metadata", "local_sort_activations")
        ]

    assert math_ast(Path(presort.__file__)) == math_ast(frozen)
