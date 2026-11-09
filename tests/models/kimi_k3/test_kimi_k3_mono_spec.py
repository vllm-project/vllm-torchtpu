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
"""CPU closure-execution unit tests for KimiK3MonoProgram custom ops.

Exercises make_tpu_finish_body, _get_or_register_tpu_finish_op, and
make_tpu_decode_body closures on CPU.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Generator
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import torch

from vllm_torchtpu.megakernel.runtime.contract import Bucket
from vllm_torchtpu.models.kimi_k3.megakernel import adapter
from vllm_torchtpu.models.kimi_k3.megakernel.kernels import paged


def _mock_mesh_32() -> MagicMock:
    """Create a mock mesh with 32 device slots for flat k3_sp."""
    mesh = MagicMock()
    mesh.size = 32
    mesh.axis_names = ("k3_sp",)
    mesh.devices = np.array([MagicMock() for _ in range(32)])
    return mesh


def _noop_jit(
    target_function: Callable[..., object], **unused_kwargs: object
) -> Callable[..., object]:
    """No-op jit replacement returning target function directly."""
    return target_function


def _noop_shard_map(
    target_function: Callable[..., object],
    *unused_args: object,
    **unused_kwargs: object,
) -> Callable[..., object]:
    """No-op shard_map replacement returning target function directly."""
    return target_function


def _noop_commit_kda_states(
    conv_state: object,
    recurrent_state: object,
    *unused_args: object,
    **unused_kwargs: object,
) -> tuple[object, object]:
    """No-op commit_kda_states returning states unchanged."""
    return (conv_state, recurrent_state)


def _mock_axis_index(unused_axis_name: object) -> jnp.ndarray:
    """Mock axis_index returning rank 0 index."""
    return jnp.int32(0)


def _mock_identity_transform(
    operand: object, *unused_args: object, **unused_kwargs: object
) -> object:
    """Mock collective transform returning operand unchanged."""
    return operand


def _mock_dynamic_update_slice(
    operand: object,
    unused_update: object,
    *unused_start_indices: object,
) -> object:
    """Mock dynamic_update_slice returning operand unchanged."""
    return operand


def _mock_dynamic_index_in_dim(
    operand: jnp.ndarray,
    unused_index: object,
    axis: int = 0,
    keepdims: bool = False,
) -> jnp.ndarray:
    """Mock dynamic_index_in_dim returning first slice along target axis."""
    return jnp.take(operand, 0, axis=axis)


@contextlib.contextmanager
def _mock_mono_program_environment(
    mesh: MagicMock,
) -> Generator[None, None, None]:
    """Provide scoped CPU mocks for mono program closures without global leakage."""
    original_to = torch.Tensor.to

    def _safe_to(
        tensor_self: torch.Tensor, *args: object, **kwargs: object
    ) -> torch.Tensor:
        """Route torch.Tensor.to('tpu') calls to CPU during test."""
        if args and args[0] == "tpu":
            return original_to(tensor_self, "cpu", *args[1:], **kwargs)
        if kwargs.get("device") == "tpu":
            kwargs["device"] = "cpu"
        return original_to(tensor_self, *args, **kwargs)

    patches = [
        patch.object(torch.Tensor, "to", _safe_to),
        patch.object(jax, "jit", _noop_jit),
        patch.object(jax, "shard_map", _noop_shard_map),
        patch.object(adapter, "commit_kda_states", _noop_commit_kda_states),
        patch.object(paged, "_commit_kda_states", _noop_commit_kda_states),
        patch.object(paged, "commit_kda_states", _noop_commit_kda_states),
        patch.object(jax.lax, "axis_index", _mock_axis_index),
        patch.object(jax.lax, "psum", _mock_identity_transform),
        patch.object(jax.lax, "all_gather", _mock_identity_transform),
        patch.object(jax.lax, "dynamic_update_slice", _mock_dynamic_update_slice),
        patch.object(jax.lax, "dynamic_index_in_dim", _mock_dynamic_index_in_dim),
    ]

    for patch_item in patches:
        patch_item.start()
    try:
        yield
    finally:
        for patch_item in reversed(patches):
            patch_item.stop()


def _build_test_mono_program(mesh: MagicMock) -> adapter.KimiK3MonoProgram:
    """Build a test instance of KimiK3MonoProgram with allocated slot shapes."""
    mono_program = object.__new__(adapter.KimiK3MonoProgram)
    mono_program.conv_slots = torch.zeros(
        (32, 10, 4, 16, 16, 128), dtype=torch.bfloat16
    )
    mono_program.recurrent_slots = torch.zeros(
        (32, 10, 16, 128, 128), dtype=torch.float32
    )
    mono_program.decoder_weights = {}
    mono_program.raw_pools = []
    mono_program.mla_raw = []
    mono_program.attention_dp = 1
    mono_program.dcp_owner_table = None
    mono_program._tpu_finish_ops = {}
    mono_program._tpu_ops = {}
    mono_program.mesh = mesh
    return mono_program


def test_mono_program_finish_body_closure() -> None:
    """Verify make_tpu_finish_body returns executable closure updating slot states."""
    mesh = _mock_mesh_32()
    with _mock_mono_program_environment(mesh):
        mono_program = _build_test_mono_program(mesh)
        finish_body_tuple = mono_program.make_tpu_finish_body(requests=2, bank_slots=10)
        finish_body = finish_body_tuple[0]

        finish_args = [
            jnp.zeros((32, 10, 4, 16, 16, 128), jnp.bfloat16),
            jnp.zeros((32, 10, 16, 128, 128), jnp.float32),
            jnp.zeros((32, 2, 4, 16, 16, 128), jnp.bfloat16),
            jnp.zeros((32, 2, 16, 128, 128), jnp.float32),
            jnp.zeros((2,), jnp.int32),
            jnp.zeros((2,), jnp.int32),
        ]
        finish_results = finish_body(*finish_args)
        assert len(finish_results) == 2
        assert finish_results[0].shape == (1, 10, 4, 16, 16, 128)
        assert finish_results[1].shape == (1, 10, 16, 128, 128)


def test_mono_program_get_or_register_tpu_finish_op() -> None:
    """Verify _get_or_register_tpu_finish_op caches and returns executable callable."""
    mesh = _mock_mesh_32()
    with _mock_mono_program_environment(mesh):
        mono_program = _build_test_mono_program(mesh)

        def _mock_register_op(
            unused_name: str,
            unused_mesh: object,
            body: Callable[..., object],
            unused_spec: object,
        ) -> Callable[..., object]:
            """Mock registration of custom TPU ops."""
            return body

        def _mock_require_tpu_register_op(
            unused_caller_name: str,
        ) -> Callable[..., Callable[..., object]]:
            """Mock loader returning registration helper."""
            return _mock_register_op

        mono_program._require_tpu_register_op = _mock_require_tpu_register_op

        finish_op = mono_program._get_or_register_tpu_finish_op(requests=2)
        assert callable(finish_op)

        # Assert caching returns the same callable on repeated request
        cached_op = mono_program._get_or_register_tpu_finish_op(requests=2)
        assert cached_op is finish_op


def test_mono_program_make_tpu_decode_body() -> None:
    """Verify make_tpu_decode_body constructs decode closure and contract specs."""
    mesh = _mock_mesh_32()
    with _mock_mono_program_environment(mesh):
        mono_program = _build_test_mono_program(mesh)
        test_bucket = Bucket(1, 4, 4)
        (
            decode_body,
            input_specs,
            output_specs,
            donated_specs,
            expert_keys,
        ) = mono_program.make_tpu_decode_body(test_bucket, layers=2)

        assert callable(decode_body)
        assert isinstance(input_specs, tuple)
        assert isinstance(output_specs, tuple)
        assert isinstance(donated_specs, tuple)
        assert isinstance(expert_keys, (tuple, str))
        assert len(input_specs) > 0
        assert len(output_specs) > 0
