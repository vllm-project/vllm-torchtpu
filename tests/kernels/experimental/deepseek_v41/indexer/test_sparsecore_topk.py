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
"""Executable specification tests for deepseek_v41.indexer.sparsecore_topk."""

import ast
import contextlib
import dataclasses
import inspect
from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.deepseek_v41.indexer import (
    BITS_NEG_INF,
    MAX_N,
    SC_LANES,
    sparsecore_topk,
)
from vllm_torchtpu.kernels.experimental.deepseek_v41.indexer import (
    sparsecore_topk as sparsecore_topk_mod,
)


@contextlib.contextmanager
def _isolated_jit_patch(
    monkeypatch: pytest.MonkeyPatch,
    target: object,
    attr: str,
    value: Any,
) -> Iterator[None]:
    sparsecore_topk.clear_cache()
    with monkeypatch.context() as mp:
        mp.setattr(target, attr, value)
        try:
            yield
        finally:
            sparsecore_topk.clear_cache()


@pytest.fixture(autouse=True)
def _setup_tpu7x_abstract_mesh() -> Iterator[None]:
    if jax.default_backend() == "tpu":
        yield
        return
    abstract_device = jax.sharding.AbstractDevice("TPU7x", 1, "tpu")
    abstract_mesh = jax.sharding.AbstractMesh(
        (), (), (), abstract_device=abstract_device
    )
    with jax.sharding.use_abstract_mesh(abstract_mesh):
        yield


def _collect_imported_modules(module: object) -> set[str]:
    source = inspect.getsource(module)  # type: ignore[arg-type] -- module object
    tree = ast.parse(source)
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def test_should_keep_sparsecore_topk_pure_jax_and_self_contained() -> None:
    # Act
    imports = _collect_imported_modules(sparsecore_topk_mod)
    roots = {mod.split(".")[0] for mod in imports}

    # Assert
    assert SC_LANES == 16
    assert MAX_N == 1_048_576
    assert roots.isdisjoint({"torch", "torch_tpu", "vllm", "os"})
    assert all("deepseek_v4." not in mod for mod in imports)


@pytest.mark.parametrize(
    ("scores_shape", "scores_dtype", "k", "match"),
    [
        ((32,), jnp.float32, 16, "scores must be 2D"),
        ((2, 4, 32), jnp.float32, 16, "scores must be 2D"),
        ((4, 32), jnp.bfloat16, 16, "scores must be f32 or i32"),
        ((4, 0), jnp.float32, 16, "positive multiple of 16"),
        ((4, 40), jnp.float32, 16, "positive multiple of 16"),
        ((4, MAX_N + SC_LANES), jnp.float32, 16, f"must be <= {MAX_N}"),
        ((4, 64), jnp.float32, 0, "positive multiple of 16"),
        ((4, 64), jnp.float32, 24, "positive multiple of 16"),
        ((4, 8192), jnp.float32, 4112, "at most 4096"),
    ],
)
def test_should_reject_invalid_inputs_in_sparsecore_topk(
    scores_shape: tuple[int, ...],
    scores_dtype: Any,
    k: int,
    match: str,
) -> None:
    # Arrange
    scores = jax.ShapeDtypeStruct(scores_shape, scores_dtype)

    # Act & Assert
    with pytest.raises(ValueError, match=match):
        sparsecore_topk.trace(scores, k)


def test_should_reject_missing_tpu7x_or_sparsecore_in_sparsecore_topk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    scores = jax.ShapeDtypeStruct((4, 2048), jnp.float32)

    # Act & Assert
    if jax.default_backend() != "tpu":
        empty_mesh = jax.sharding.AbstractMesh((), (), ())
        with (
            jax.sharding.use_abstract_mesh(empty_mesh),
            pytest.raises(
                RuntimeError, match="sparsecore_topk requires TPU7x, got backend="
            ),
        ):
            sparsecore_topk.trace(scores, 16)

    v6e_device = jax.sharding.AbstractDevice("TPU v6e", 1, "tpu")
    v6e_mesh = jax.sharding.AbstractMesh((), (), (), abstract_device=v6e_device)
    with (
        jax.sharding.use_abstract_mesh(v6e_mesh),
        pytest.raises(
            RuntimeError,
            match=r"sparsecore_topk requires TPU7x SparseCore \(2x16 subcores\)",
        ),
    ):
        sparsecore_topk.trace(scores, 16)

    orig_get_tpu_info = pltpu.get_tpu_info
    base_info = orig_get_tpu_info()
    assert base_info.sparse_core is not None
    bad_infos = (
        dataclasses.replace(base_info, sparse_core=None),
        dataclasses.replace(
            base_info,
            sparse_core=dataclasses.replace(base_info.sparse_core, num_cores=1),
        ),
        dataclasses.replace(
            base_info,
            sparse_core=dataclasses.replace(base_info.sparse_core, num_subcores=8),
        ),
    )
    for bad_info in bad_infos:
        with (
            _isolated_jit_patch(
                monkeypatch, pltpu, "get_tpu_info", lambda info=bad_info: info
            ),
            pytest.raises(
                RuntimeError,
                match=r"sparsecore_topk requires TPU7x SparseCore \(2x16 subcores\)",
            ),
        ):
            sparsecore_topk.trace(scores, 16)


def _jaxpr_kernel_names(jaxpr: Any) -> list[str]:
    names: list[str] = []
    stack = [jaxpr.jaxpr if hasattr(jaxpr, "jaxpr") else jaxpr]
    while stack:
        cur = stack.pop()
        for eqn in cur.eqns:
            name = eqn.params.get("name")
            if isinstance(name, str):
                names.append(name)
            for val in eqn.params.values():
                if hasattr(val, "eqns"):
                    stack.append(val)
                elif hasattr(val, "jaxpr"):
                    stack.append(val.jaxpr)
    return names


@pytest.mark.parametrize(
    (
        "b",
        "n",
        "k",
        "dtype",
        "use_row_lengths",
        "write_empty",
        "expected_p",
        "lower_hlo",
    ),
    [
        (32, 4096, 16, jnp.int32, False, True, 1, False),
        (33, 64, 16, jnp.float32, True, True, 1, False),
        (32, 32784, 16, jnp.float32, True, True, 2, False),
        (2, 64, 32, jnp.float32, True, True, 2, False),
        (3, 2064, 64, jnp.int32, True, False, 8, True),
        (3, 1048576, 16, jnp.float32, False, True, 16, False),
    ],
)
def test_should_select_expected_partition_and_lower_to_tpu_hlo(
    b: int,
    n: int,
    k: int,
    dtype: Any,
    use_row_lengths: bool,
    write_empty: bool,
    expected_p: int,
    lower_hlo: bool,
) -> None:
    # Arrange
    scores_sds = jax.ShapeDtypeStruct((b, n), dtype)
    row_lengths_sds = jax.ShapeDtypeStruct((b,), jnp.int32) if use_row_lengths else None

    # Act
    traced = sparsecore_topk.trace(
        scores_sds,
        k,
        row_lengths_sds,
        write_empty_rows=write_empty,
    )
    expected_name = f"sc_topk_direct_b{b}_n{n}_k{k}_p{expected_p}"
    if lower_hlo:
        assert expected_name in traced.lower(lowering_platforms=("tpu",)).as_text()
    else:
        assert expected_name in _jaxpr_kernel_names(traced.jaxpr)


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    ("b", "n", "k", "as_bits", "write_empty"),
    [
        (36, 2048, 64, False, True),
        (4, 8192, 256, True, True),
        (2, 65536, 512, False, False),
        (40, 65536, 64, False, True),
    ],
)
def test_should_select_exact_topk_indices_on_tpu7x_hardware(
    b: int,
    n: int,
    k: int,
    as_bits: bool,
    write_empty: bool,
) -> None:
    if jax.default_backend() != "tpu":
        pytest.skip("TPU7x hardware required")
    # Arrange
    rng = np.random.default_rng(404 + b + n + k)
    scores_np = rng.standard_normal((b, n), dtype=np.float32)
    scores_np[:, :32] = 7.5
    scores_np[:, 32:48] = -np.inf
    row_lens_np = np.full((b,), n, dtype=np.int32)
    if b == 2:
        row_lens_np[1] = 0
    else:
        row_lens_np[0] = min(n, k // 2)
        row_lens_np[1] = 0

    scores_jax = jnp.asarray(scores_np)
    if as_bits:
        scores_jax = jax.lax.bitcast_convert_type(scores_jax, jnp.int32)

    # Act
    actual = np.asarray(
        sparsecore_topk(
            scores_jax,
            k,
            row_lengths=jnp.asarray(row_lens_np),
            write_empty_rows=write_empty,
        )
    )

    # Assert
    assert actual.shape == (b, k)
    assert actual.dtype == np.int32
    neg_inf_f32 = np.int32(BITS_NEG_INF).view(np.float32)
    assert np.isneginf(neg_inf_f32)
    for row_idx in range(b):
        eff_len = int(row_lens_np[row_idx])
        if eff_len == 0 and not write_empty:
            continue
        row_scores = scores_np[row_idx, :eff_len]
        valid_scores = row_scores[np.isfinite(row_scores)]
        expected_count = min(k, int(valid_scores.size))
        prefix = actual[row_idx, :expected_count]
        suffix = actual[row_idx, expected_count:]
        assert np.all(prefix >= 0)
        assert np.all(suffix == -1)
        assert np.unique(prefix).size == expected_count
        if expected_count > 0:
            assert np.all(prefix < eff_len)
            chosen_scores = row_scores[prefix]
            assert np.all(np.isfinite(chosen_scores))
            kth_idx = valid_scores.size - expected_count
            kth_score = float(np.partition(valid_scores, kth_idx)[kth_idx])
            assert float(np.min(chosen_scores)) >= kth_score
