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

from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import AbstractMesh, Mesh
from jax.sharding import PartitionSpec as P
from torch._dynamo.testing import CompileCounterWithBackend
from torch._subclasses.fake_tensor import FakeTensorMode
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.experimental.deepseek_v41.mhc import mhc_post
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41 import (
    get_mhc_post_op,
    mhc_op,
)
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)


def _torch_dtype_for(jax_dtype: jnp.dtype) -> torch.dtype:
    mapping = {
        jnp.dtype(jnp.bfloat16): torch.bfloat16,
        jnp.dtype(jnp.float32): torch.float32,
    }
    return mapping[jnp.dtype(jax_dtype)]


class _JaxBackedTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, elem: jax.Array, env: object = None) -> "_JaxBackedTensor":
        del env
        shape = tuple(int(dim) for dim in elem.shape)
        base = torch.empty(shape, dtype=_torch_dtype_for(elem.dtype))
        return base.as_subclass(cls)

    def __init__(self, elem: jax.Array, env: object = None) -> None:
        self._elem = elem
        self._env = env

    def to_numpy(self) -> np.ndarray:
        return np.asarray(self._elem)


def _ref_mhc_post_jax(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
) -> jax.Array:
    mixed = jnp.einsum(
        "...ij,...ih->...jh",
        comb_res_mix.astype(jnp.float32),
        residual.astype(jnp.float32),
    )
    out = mixed + post_layer_mix.astype(jnp.float32) * x[..., None, :].astype(
        jnp.float32
    )
    return out.astype(residual.dtype)


def _clear_mhc_op_caches() -> None:
    cache: dict[str, Any] = mhc_op._mhc_post_op_cache  # noqa: SLF001 -- clear cached ops
    for op in cache.values():
        init_fn = op._init_fn  # noqa: SLF001 -- JaxCustomOpDef holds JaxCallable
        init_fn.jit_fn.clear_cache()
        if init_fn.output_shapes:
            init_fn.output_shapes.clear()
        if init_fn.kernel_key_to_mlir_fingerprint:
            init_fn.kernel_key_to_mlir_fingerprint.clear()
    cache.clear()


@pytest.fixture(autouse=True)
def _reset_mhc_op_caches() -> Iterator[None]:
    _clear_mhc_op_caches()
    yield
    _clear_mhc_op_caches()


def _capture_jax_op_registrations(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, object]]:
    recorded: list[dict[str, object]] = []
    real_jax_op = pallas.jax_op

    def _spy_jax_op(*args: object, **kwargs: object):
        recorded.append(dict(kwargs))
        return real_jax_op(*args, **kwargs)

    monkeypatch.setattr(pallas, "jax_op", _spy_jax_op)
    return recorded


def test_should_cache_and_trace_mhc_post_custom_op_across_mesh_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded_registrations = _capture_jax_op_registrations(monkeypatch)

    single_mesh = Mesh(np.array([jax.devices()[0]]), ("data",))
    with set_vllm_model_wrapper_context(mesh=single_mesh):
        op_single = get_mhc_post_op()
        assert get_mhc_post_op() is op_single
        assert len(recorded_registrations) == 1
        assert recorded_registrations[-1]["mesh"] is None
        assert recorded_registrations[-1]["input_partition_specs"] is None

        counter = CompileCounterWithBackend("aot_eager")

        def _run_post(
            x: torch.Tensor,
            residual: torch.Tensor,
            post_mix: torch.Tensor,
            comb_mix: torch.Tensor,
        ) -> torch.Tensor:
            return op_single(x, residual, post_mix, comb_mix)

        compiled = torch.compile(_run_post, backend=counter, fullgraph=True)
        with FakeTensorMode():
            for _ in range(2):
                x = torch.empty((6, 128), dtype=torch.bfloat16)
                residual = torch.empty((4, 6, 128), dtype=torch.bfloat16).transpose(
                    0, 1
                )
                post_mix = torch.empty((6, 4, 1), dtype=torch.float32)
                comb_mix = torch.empty((6, 4, 4), dtype=torch.float32)
                out = compiled(x, residual, post_mix, comb_mix)
                assert out.shape == (6, 4, 128) and out.dtype == torch.bfloat16
                assert out.is_contiguous()
                assert out is not residual
        assert counter.frame_count == 1

    multi_mesh = AbstractMesh((2,), ("data",))
    with set_vllm_model_wrapper_context(mesh=multi_mesh):
        op_multi = get_mhc_post_op()
        assert get_mhc_post_op() is op_multi
        assert op_multi is not op_single
        assert recorded_registrations[-1]["mesh"] is multi_mesh
        assert recorded_registrations[-1]["input_partition_specs"] == (
            P(),
            P(),
            P(),
            P(),
        )
        with FakeTensorMode():
            out_multi = op_multi(
                torch.empty((4, 128), dtype=torch.bfloat16),
                torch.empty((4, 4, 128), dtype=torch.bfloat16),
                torch.empty((4, 4, 1), dtype=torch.float32),
                torch.empty((4, 4, 4), dtype=torch.float32),
            )
            assert out_multi.shape == (4, 4, 128)
            assert out_multi.dtype == torch.bfloat16


def test_should_forward_mhc_post_arguments_to_jax_kernel_without_pallas_cpu_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded_registrations = _capture_jax_op_registrations(monkeypatch)
    monkeypatch.setattr(mhc_op, "mhc_post", _ref_mhc_post_jax)
    rng = np.random.default_rng(505)
    x_np = rng.standard_normal((3, 16)).astype(np.float32)
    res_np = rng.standard_normal((3, 2, 16)).astype(np.float32)
    post_np = rng.uniform(0.2, 1.7, size=(3, 2, 1)).astype(np.float32)
    comb_np = rng.uniform(0.1, 0.9, size=(3, 2, 2)).astype(np.float32)

    single_mesh = Mesh(np.array([jax.devices()[0]]), ("data",))
    with set_vllm_model_wrapper_context(mesh=single_mesh):
        op = get_mhc_post_op()
        assert len(recorded_registrations) == 1
        assert recorded_registrations[-1]["mesh"] is None
        assert recorded_registrations[-1]["input_partition_specs"] is None
        out = op(
            _JaxBackedTensor(jnp.asarray(x_np, dtype=jnp.bfloat16)),
            _JaxBackedTensor(jnp.asarray(res_np, dtype=jnp.bfloat16)),
            _JaxBackedTensor(jnp.asarray(post_np, dtype=jnp.float32)),
            _JaxBackedTensor(jnp.asarray(comb_np, dtype=jnp.float32)),
        )
    assert isinstance(out, _JaxBackedTensor)
    expected = _ref_mhc_post_jax(
        jnp.asarray(x_np, dtype=jnp.bfloat16),
        jnp.asarray(res_np, dtype=jnp.bfloat16),
        jnp.asarray(post_np, dtype=jnp.float32),
        jnp.asarray(comb_np, dtype=jnp.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(out.to_numpy(), dtype=np.float32),
        np.asarray(expected, dtype=np.float32),
    )


@pytest.mark.tpu_test
def test_tpu_should_execute_mhc_post_custom_op_under_tpu_compile() -> None:
    if jax.default_backend() != "tpu":
        pytest.skip("Requires TPU hardware backend")

    rng = np.random.default_rng(606)
    x_np = rng.standard_normal((48, 2560)).astype(np.float32)
    res_np = rng.standard_normal((48, 4, 2560)).astype(np.float32)
    post_np = rng.uniform(0.2, 1.5, size=(48, 4, 1)).astype(np.float32)
    comb_np = rng.uniform(0.1, 0.8, size=(48, 4, 4)).astype(np.float32)

    device = torch.device("tpu")
    single_mesh = Mesh(np.array([jax.devices()[0]]), ("data",))
    with set_vllm_model_wrapper_context(mesh=single_mesh):
        op = get_mhc_post_op()
        compiled = torch.compile(
            lambda a, b, c, d: op(a, b, c, d), backend="tpu", fullgraph=True
        )
        out_tpu = compiled(
            torch.from_numpy(x_np).to(device=device, dtype=torch.bfloat16),
            torch.from_numpy(res_np).to(device=device, dtype=torch.bfloat16),
            torch.from_numpy(post_np).to(device=device, dtype=torch.float32),
            torch.from_numpy(comb_np).to(device=device, dtype=torch.float32),
        )
        actual = out_tpu.cpu().float().numpy()

    expected = np.asarray(
        mhc_post(
            jnp.asarray(x_np, dtype=jnp.bfloat16),
            jnp.asarray(res_np, dtype=jnp.bfloat16),
            jnp.asarray(post_np, dtype=jnp.float32),
            jnp.asarray(comb_np, dtype=jnp.float32),
        ),
        dtype=np.float32,
    )
    np.testing.assert_array_equal(actual, expected)
