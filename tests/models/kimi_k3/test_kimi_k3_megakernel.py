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
"""CPU closure-execution unit tests for Kimi-K3 megakernel custom ops.

Exercises custom-op closures across PrefillProgram, PoolWriter, DecodeOps,
KimiDecoder, DsparkDraft, and MlaDsparkDraft on CPU.
"""

from __future__ import annotations

import contextlib
import functools
import gc
from collections.abc import Callable, Generator, Sequence
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.kernels import pool_adapters
from vllm_torchtpu.models.kimi_k3.megakernel import (
    decode,
    decode_ops,
    draft,
    draft_mla,
    prefill,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kernels import (
    dspark,
    paged,
    prefill_megakernel,
)
from vllm_torchtpu.models.kimi_k3.megakernel.runtime_shim import bridge


@pytest.fixture(autouse=True)
def _cleanup_memory_after_test() -> Generator[None, None, None]:
    yield
    jax.clear_caches()
    gc.collect()


def _torch_to_jax(tensor_or_val: object) -> object:
    """Convert torch.Tensor to jax.Array safely without float32 copy."""
    if not isinstance(tensor_or_val, torch.Tensor):
        return tensor_or_val
    if tensor_or_val.dtype == torch.bfloat16:
        return jnp.asarray(tensor_or_val.view(torch.uint16).numpy()).view(jnp.bfloat16)
    return jnp.asarray(tensor_or_val.numpy())


def _jax_to_torch(array_or_val: object) -> object:
    """Convert jax.Array or tuple to torch.Tensor."""
    if isinstance(array_or_val, tuple):
        return tuple(_jax_to_torch(x) for x in array_or_val)
    if isinstance(array_or_val, list):
        return [_jax_to_torch(x) for x in array_or_val]
    if isinstance(array_or_val, jax.Array):
        if array_or_val.dtype == jnp.bfloat16:
            return torch.from_numpy(np.asarray(array_or_val, dtype=np.float32)).to(
                torch.bfloat16
            )
        return torch.from_numpy(np.asarray(array_or_val))
    return array_or_val


def _mock_mesh_32() -> MagicMock:
    """Create a mock mesh with 32 device slots for flat k3_sp."""
    mesh = MagicMock()
    mesh.size = 32
    mesh.axis_names = ("k3_sp",)
    mesh.devices = np.array([MagicMock() for _ in range(32)])
    return mesh


def _mock_kda(
    captured_calls: list[dict[str, object]],
) -> Callable[..., Callable[..., list[object]]]:
    """Mock leaf KDA SPMD kernel builder with argument capture and type assertions."""

    def kda_builder(
        *unused_args: object, **unused_kwargs: object
    ) -> Callable[..., list[object]]:
        def kda_fn(
            prefix: jax.Array,
            blocks: jax.Array,
            attn_res: jax.Array,
            attn_norm: jax.Array,
            families: object,
            recurrent: jax.Array,
            conv: jax.Array,
            active: jax.Array,
            length: jax.Array,
            slot: object,
            segment_desc: jax.Array,
            arena: object = None,
        ) -> list[object]:
            assert hasattr(segment_desc, "ndim") and segment_desc.ndim == 1, (
                "segment_desc must be a 1D array"
            )
            assert segment_desc.dtype == jnp.int32, (
                f"segment_desc must be int32, got {segment_desc.dtype}"
            )
            assert segment_desc.shape == (10,), (
                f"segment_desc shape must be (10,), got {segment_desc.shape}"
            )
            if arena is not None:
                assert hasattr(arena, "active") or hasattr(arena, "split"), (
                    "arena must be an arena object"
                )
            captured_calls.append(
                {"op": "kda", "segment_desc": segment_desc, "arena": arena}
            )
            return [
                prefix,
                prefix,
                prefix,
                recurrent[:, 0],
                conv[:, 0],
                prefix,
                recurrent[:, 0],
                conv[:, 0],
            ]

        return kda_fn

    return kda_builder


def _mock_mla(
    captured_calls: list[dict[str, object]],
) -> Callable[..., Callable[..., tuple[jax.Array, jax.Array]]]:
    """Mock leaf MLA SPMD kernel builder with argument capture and type assertions."""

    def mla_builder(
        *unused_args: object, **unused_kwargs: object
    ) -> Callable[..., tuple[jax.Array, jax.Array]]:
        def mla_fn(
            gathered: jax.Array,
            prefix: jax.Array,
            families: object,
            segment_desc: jax.Array,
            length: jax.Array,
            slot: object,
            arena: object,
        ) -> tuple[jax.Array, jax.Array]:
            assert hasattr(segment_desc, "ndim") and segment_desc.ndim == 1, (
                "segment_desc must be a 1D array"
            )
            assert segment_desc.dtype == jnp.int32, (
                f"segment_desc must be int32, got {segment_desc.dtype}"
            )
            assert segment_desc.shape == (10,), (
                f"segment_desc shape must be (10,), got {segment_desc.shape}"
            )
            assert hasattr(arena, "active") or hasattr(arena, "split"), (
                "arena must be an arena object"
            )
            captured_calls.append(
                {"op": "mla", "segment_desc": segment_desc, "arena": arena}
            )
            return (prefix, jnp.zeros_like(prefix))

        return mla_fn

    return mla_builder


def _mock_logits_sp(
    *unused_args: object, **unused_kwargs: object
) -> Callable[..., jax.Array]:
    """Mock prefill logits SPMD kernel builder with segment_desc assertion."""

    def logits_op(
        _prefix: jax.Array,
        _blocks: jax.Array,
        _output_res: jax.Array,
        _final_norm: jax.Array,
        _lm_head: jax.Array,
        segment_desc: jax.Array,
    ) -> jax.Array:
        assert hasattr(segment_desc, "ndim") and segment_desc.ndim == 1, (
            "segment_desc must be a 1D array"
        )
        assert segment_desc.dtype == jnp.int32, (
            f"segment_desc must be int32, got {segment_desc.dtype}"
        )
        return jnp.zeros((32, 10, 16), dtype=jnp.float32)

    return logits_op


@contextlib.contextmanager
def _mock_execution_environment(
    mesh: MagicMock,
    captured_calls: list[dict[str, object]] | None = None,
) -> Generator[list[dict[str, object]], None, None]:
    """Provide scoped CPU mocks for megakernel execution without global leakage."""
    if captured_calls is None:
        captured_calls = []

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

    def _intercept_register_op(
        _name: str,
        _mesh: object,
        fn: Callable[..., object],
        _spec: object,
    ) -> Callable[..., object]:
        """Intercept register_op and wrap for CPU execution."""

        def op_wrapper(*args: object) -> object:
            """Execute intercepted custom op on CPU."""
            jax_args = [_torch_to_jax(a) for a in args]
            return _jax_to_torch(fn(*jax_args))

        return op_wrapper

    def _mock_commit_kda(
        conv_state: jax.Array,
        recurrent_state: jax.Array,
        *unused_args: object,
        **unused_kwargs: object,
    ) -> tuple[jax.Array, jax.Array]:
        return (conv_state, recurrent_state)

    def _mock_decode_paged(*args: object, **kwargs: object) -> tuple[object, ...]:
        return (jnp.zeros((1, 16)), args[1], args[2], ())

    def _mock_embedding_sp(
        *unused_args: object, **unused_kwargs: object
    ) -> Callable[..., jax.Array]:
        def embed_fn(tokens: object, embedding: object, length: object) -> jax.Array:
            del tokens, embedding, length
            return jnp.zeros((32, 64, 7168), jnp.bfloat16)

        return embed_fn

    def _mock_mixture_norm_sp(
        *unused_args: object, **unused_kwargs: object
    ) -> Callable[..., jax.Array]:
        def mixture_fn(
            after_attention: jax.Array,
            blocks: object,
            mlp_residual: object,
            ffn_norm: object,
            active_mask: object,
        ) -> jax.Array:
            del blocks, mlp_residual, ffn_norm, active_mask
            return after_attention

        return mixture_fn

    def _mock_dense_ffn_sp(
        *unused_args: object, **unused_kwargs: object
    ) -> Callable[..., jax.Array]:
        def dense_fn(
            families: object,
            after_norm: jax.Array,
            dense_gu: object,
            dense_down: object,
            length: object,
        ) -> jax.Array:
            del families, dense_gu, dense_down, length
            return after_norm

        return dense_fn

    def _mock_norm_gather_sp(
        *unused_args: object, **unused_kwargs: object
    ) -> Callable[..., tuple[jax.Array, jax.Array]]:
        def norm_gather_fn(
            prefix: jax.Array,
            blocks: object,
            attn_res: object,
            attn_norm: object,
            active_mask: object,
        ) -> tuple[jax.Array, jax.Array]:
            del blocks, attn_res, attn_norm, active_mask
            return (prefix, prefix)

        return norm_gather_fn

    def _mock_indexed_moe_sp_ring(
        *unused_args: object, **unused_kwargs: object
    ) -> Callable[..., jax.Array]:
        def moe_fn(
            families: object,
            after_norm: jax.Array,
            moe_weights: object,
            length: object,
            active_mask: object,
        ) -> jax.Array:
            del families, moe_weights, length, active_mask
            return after_norm

        return moe_fn

    def _noop_pass_through(
        target_fn: Callable[..., object], *unused_args: object, **unused_kwargs: object
    ) -> Callable[..., object]:
        return target_fn

    def _noop_pool_pass(
        pool: object, *unused_args: object, **unused_kwargs: object
    ) -> object:
        return pool

    def _mock_axis_idx(_axis_name: object) -> jnp.ndarray:
        return jnp.int32(0)

    def _mock_identity_op(
        operand: object, *unused_args: object, **unused_kwargs: object
    ) -> object:
        return operand

    def _mock_dynamic_update_slice_fn(
        operand: object, update: object, *indices: object
    ) -> object:
        del update, indices
        return operand

    def _mock_dynamic_index_in_dim_fn(
        operand: jnp.ndarray,
        index: object,
        axis: int = 0,
        keepdims: bool = False,
    ) -> jnp.ndarray:
        del index, keepdims
        return jnp.take(operand, 0, axis=axis)

    def _mock_fori_loop(
        lower: int,
        upper: int,
        body: Callable[[jax.Array, object], object],
        init: object,
        **unused_kwargs: object,
    ) -> object:
        del upper, unused_kwargs
        idx = jnp.int32(lower)
        jax.make_jaxpr(body)(idx, init)
        return body(idx, init)

    def _mock_kernel_mesh(
        _axis: object, size: int | None = None
    ) -> tuple[MagicMock, int]:
        del size
        return (mesh, 0)

    def _mock_named_sharding(_mesh: object, _spec: object) -> MagicMock:
        return MagicMock()

    def _mock_zeros_sharded(
        _self: object, shape: Sequence[int], dtype: object
    ) -> jnp.ndarray:
        return jnp.zeros(shape, dtype)

    patches = [
        patch.object(torch.Tensor, "to", _safe_to),
        patch.object(jax, "jit", _noop_pass_through),
        patch.object(jax, "shard_map", _noop_pass_through),
        patch.object(pool_adapters, "scatter_region", _noop_pool_pass),
        patch.object(paged, "_commit_kda_states", _mock_commit_kda),
        patch.object(paged, "decode_paged", _mock_decode_paged),
        patch.object(jax.lax, "axis_index", _mock_axis_idx),
        patch.object(jax.lax, "psum", _mock_identity_op),
        patch.object(jax.lax, "all_gather", _mock_identity_op),
        patch.object(jax.lax, "dynamic_update_slice", _mock_dynamic_update_slice_fn),
        patch.object(jax.lax, "dynamic_index_in_dim", _mock_dynamic_index_in_dim_fn),
        patch.object(jax.lax, "fori_loop", _mock_fori_loop),
        patch.object(bridge, "kernel_mesh", _mock_kernel_mesh),
        patch.object(prefill, "kernel_mesh", _mock_kernel_mesh),
        patch.object(bridge, "register_global_op", _intercept_register_op),
        patch.object(bridge, "register_op", _intercept_register_op),
        patch.object(prefill, "register_global_op", _intercept_register_op),
        patch.object(prefill, "register_op", _intercept_register_op),
        patch.object(decode_ops, "register_op", _intercept_register_op),
        patch.object(decode, "register_global_op", _intercept_register_op),
        patch.object(draft, "register_global_op", _intercept_register_op),
        patch.object(prefill_megakernel, "NamedSharding", _mock_named_sharding),
        patch.object(prefill.PrefillProgram, "_zeros_sharded", _mock_zeros_sharded),
        patch.object(prefill_megakernel, "make_embedding_sp", _mock_embedding_sp),
        patch.object(
            prefill_megakernel, "make_norm_gather_kda_sp", _mock_kda(captured_calls)
        ),
        patch.object(prefill_megakernel, "make_mixture_norm_sp", _mock_mixture_norm_sp),
        patch.object(prefill_megakernel, "make_dense_ffn_sp", _mock_dense_ffn_sp),
        patch.object(prefill_megakernel, "make_norm_gather_sp", _mock_norm_gather_sp),
        patch.object(
            prefill_megakernel,
            "make_indexed_mla_layer_sp_ring",
            _mock_mla(captured_calls),
        ),
        patch.object(
            prefill_megakernel,
            "make_indexed_moe_sp_ring",
            _mock_indexed_moe_sp_ring,
        ),
        patch.object(prefill_megakernel, "make_prefill_logits_sp", _mock_logits_sp),
    ]

    for patch_item in patches:
        patch_item.start()
    try:
        yield captured_calls
    finally:
        for patch_item in reversed(patches):
            patch_item.stop()


def test_prefill_program_window_ops_and_state_closures() -> None:
    """Verify PrefillProgram first, group, final, stack, and split closures."""
    mesh = _mock_mesh_32()
    captured_calls: list[dict[str, object]] = []
    with _mock_execution_environment(mesh, captured_calls=captured_calls):
        program = prefill.PrefillProgram(windows=(2048,), capacity=0, slots=1)
        window_ops = program.build_window_ops(2048)

        for group_name, weight_names in program.names.items():
            program.weights[group_name] = {
                name: torch.zeros((32, 1, 1), dtype=torch.bfloat16)
                for name in weight_names
            }
        program.weights["io"].update(
            {
                "embedding": torch.zeros((128, 7168), dtype=torch.bfloat16),
                "output_res": torch.zeros((32, 7168), dtype=torch.bfloat16),
                "final_norm": torch.zeros((7168,), dtype=torch.bfloat16),
                "lm_head": torch.zeros((32, 1, 1), dtype=torch.bfloat16),
            }
        )
        program.weights["common"].update(
            {
                "attn_res": torch.zeros((32, 93, 1, 1), dtype=torch.bfloat16),
                "attn_norm": torch.zeros((32, 93, 1), dtype=torch.bfloat16),
                "mlp_res": torch.zeros((32, 93, 1, 1), dtype=torch.bfloat16),
                "ffn_norm": torch.zeros((32, 93, 1), dtype=torch.bfloat16),
                "dense_gu": torch.zeros((32, 1, 1), dtype=torch.bfloat16),
                "dense_down": torch.zeros((32, 1, 1), dtype=torch.bfloat16),
            }
        )

        hidden = torch.zeros((32, 64, 7168), dtype=torch.bfloat16)
        extent = torch.tensor([64], dtype=torch.int32)
        segments = torch.zeros((10,), dtype=torch.int32)
        recurrent = torch.zeros((1, 69, 16, 128, 128), dtype=torch.float32)
        conv = torch.zeros((1, 69, 4, 16, 16, 128), dtype=torch.bfloat16)
        pool = torch.zeros((1, 1), dtype=torch.bfloat16)
        ids = torch.zeros((64,), dtype=torch.int32)
        local_ids = torch.zeros((64,), dtype=torch.int32)
        blocks_tensor = torch.zeros((32, 8, 64, 7168), dtype=torch.bfloat16)

        first_args = (
            hidden,
            extent,
            segments,
            recurrent,
            conv,
            pool,
            ids,
            local_ids,
            *program.flat_weights("io", "common", "kda", "mla", "moe"),
        )
        first_outputs = window_ops["first"](*first_args)
        assert len(first_outputs) == 8

        group_args = (
            hidden,
            blocks_tensor,
            recurrent,
            conv,
            recurrent,
            conv,
            hidden,
            extent,
            torch.tensor([0], dtype=torch.int32),
            segments,
            pool,
            pool,
            ids,
            local_ids,
            *program.flat_weights("common", "kda", "mla", "moe", expert_bank=1),
        )
        group_outputs = window_ops["group"](*group_args)
        assert len(group_outputs) == 9

        final_args = (
            hidden,
            blocks_tensor,
            hidden,
            extent,
            segments,
            pool,
            ids,
            local_ids,
            *program.flat_weights("io", "common", "mla", "moe", expert_bank=1),
        )
        final_outputs = window_ops["final"](*final_args)
        assert len(final_outputs) == 2

        assert len(captured_calls) > 0
        for call_record in captured_calls:
            assert call_record["segment_desc"].shape == (10,), (
                f"Expected segment_desc shape (10,), got "
                f"{call_record['segment_desc'].shape}"
            )
            assert call_record["segment_desc"].dtype == jnp.int32, (
                f"Expected segment_desc int32, got {call_record['segment_desc'].dtype}"
            )
            arena_obj = call_record["arena"]
            if arena_obj is not None:
                assert hasattr(arena_obj, "active") or hasattr(arena_obj, "split"), (
                    "arena object must not be swapped with segment_desc"
                )

        stack_op = program.state_op("stack", count=2)
        states_in = [jnp.zeros((69, 2, 4, 4), jnp.float32) for _ in range(4)]
        stacked_states = stack_op(*states_in)
        assert len(stacked_states) == 2

        split_op = program.state_op("split", count=2)
        split_states = split_op(stacked_states[0], stacked_states[1])
        assert len(split_states) == 4


def test_prefill_stack_and_factory_closures() -> None:
    """Verify prefill megakernel stack and factories support arguments and tapping."""
    mesh = _mock_mesh_32()
    mock_arena = MagicMock()
    mock_arena.split.return_value = (mock_arena, mock_arena)
    mock_arena.active = jnp.zeros((1,), jnp.int32)
    mock_arena.arenas = (mock_arena, mock_arena)

    prefix = jnp.zeros((32, 64, 7168), jnp.bfloat16)
    blocks = jnp.zeros((32, 8, 64, 7168), jnp.bfloat16)
    common = {
        "attn_res": jnp.zeros((32, 93, 1, 1), jnp.bfloat16),
        "attn_norm": jnp.zeros((32, 93, 1), jnp.bfloat16),
        "mlp_res": jnp.zeros((32, 93, 1, 1), jnp.bfloat16),
        "ffn_norm": jnp.zeros((32, 93, 1), jnp.bfloat16),
        "dense_gu": jnp.zeros((32, 1, 1), jnp.bfloat16),
        "dense_down": jnp.zeros((32, 1, 1), jnp.bfloat16),
    }
    families = MagicMock()
    kda_state = (
        jnp.zeros((32, 69, 2, 4, 4), jnp.float32),
        jnp.zeros((32, 69, 4, 2, 4, 4), jnp.bfloat16),
        jnp.zeros((32, 69, 2, 4, 4), jnp.float32),
        jnp.zeros((32, 69, 4, 2, 4, 4), jnp.bfloat16),
    )
    mla_state = (jnp.zeros((32, 24, 64, 576), jnp.bfloat16),)
    length = jnp.asarray([64], jnp.int32)
    segment_desc = jnp.zeros((10,), jnp.int32)
    pair = jnp.asarray([0], jnp.int32)
    group = jnp.asarray([0], jnp.int32)

    captured_calls: list[dict[str, object]] = []
    with _mock_execution_environment(mesh, captured_calls=captured_calls):
        # 1. Test make_prefill_stack_sp_indexed with tap_layer=None
        stack_fn_none = prefill_megakernel.make_prefill_stack_sp_indexed(
            mesh, segments=10, seq_len=2048, prefix_capacity=2048, tap_layer=None
        )
        stack_none_pos = stack_fn_none(
            prefix,
            common,
            families,
            families,
            families,
            kda_state,
            mla_state,
            length,
            segment_desc,
            mock_arena,
        )
        assert len(stack_none_pos) == 4

        stack_none_kw = stack_fn_none(
            prefix,
            common,
            families,
            families,
            families,
            kda_state,
            mla_state,
            length,
            segment_desc=segment_desc,
            arena=mock_arena,
        )
        assert len(stack_none_kw) == 4

        # Single keyword segment_desc, arena positional (tests arena fallback)
        stack_none_single_kw = stack_fn_none(
            prefix,
            common,
            families,
            families,
            families,
            kda_state,
            mla_state,
            length,
            mock_arena,
            segment_desc=segment_desc,
        )
        assert len(stack_none_single_kw) == 4
        del stack_fn_none, stack_none_pos, stack_none_kw, stack_none_single_kw
        gc.collect()

        # 2. Test make_prefill_stack_sp_indexed with tap_layer=18
        stack_fn_18 = prefill_megakernel.make_prefill_stack_sp_indexed(
            mesh, segments=10, seq_len=2048, prefix_capacity=2048, tap_layer=18
        )
        # Negative probe: missing arena raises IndexError (kills _args[-1] mutant)
        with pytest.raises(IndexError):
            stack_fn_18(
                prefix,
                common,
                families,
                families,
                families,
                kda_state,
                mla_state,
                length,
                segment_desc=segment_desc,
            )

        stack_18_outputs = stack_fn_18(
            prefix,
            common,
            families,
            families,
            families,
            kda_state,
            mla_state,
            length,
            segment_desc=segment_desc,
            arena=mock_arena,
        )
        assert len(stack_18_outputs) == 5
        tap = stack_18_outputs[4]
        assert isinstance(tap, jax.Array)
        assert tap.shape == (32, 64, 7168)
        assert not isinstance(tap, tuple)
        del stack_fn_18, stack_18_outputs, tap
        gc.collect()

        # 3. Test make_regular_group8_sp_indexed
        group8_fn = prefill_megakernel.make_regular_group8_sp_indexed(
            mesh, segments=10, seq_len=2048, prefix_capacity=2048
        )
        group8_pos = group8_fn(
            prefix,
            blocks,
            common,
            families,
            families,
            families,
            kda_state,
            mla_state,
            length,
            pair,
            segment_desc,
            mock_arena,
        )
        assert len(group8_pos) == 6

        group8_kw = group8_fn(
            prefix,
            blocks,
            common,
            families,
            families,
            families,
            kda_state,
            mla_state,
            length,
            pair,
            segment_desc=segment_desc,
            arena=mock_arena,
        )
        assert len(group8_kw) == 6

        # Single keyword segment_desc, arena positional
        group8_single_kw = group8_fn(
            prefix,
            blocks,
            common,
            families,
            families,
            families,
            kda_state,
            mla_state,
            length,
            pair,
            mock_arena,
            segment_desc=segment_desc,
        )
        assert len(group8_single_kw) == 6
        del group8_fn, group8_pos, group8_kw, group8_single_kw
        gc.collect()

        # 4. Test make_regular_group4_sp_indexed directly (unpatched)
        group4_fn = prefill_megakernel.make_regular_group4_sp_indexed(
            mesh, segments=10, seq_len=2048, prefix_capacity=2048
        )
        group4_pos = group4_fn(
            prefix,
            blocks,
            common,
            families,
            families,
            families,
            kda_state,
            mla_state,
            length,
            group,
            segment_desc,
            mock_arena,
        )
        assert len(group4_pos) == 5

        group4_kw = group4_fn(
            prefix,
            blocks,
            common,
            families,
            families,
            families,
            kda_state,
            mla_state,
            length,
            group,
            segment_desc=segment_desc,
            arena=mock_arena,
        )
        assert len(group4_kw) == 5

        # Single keyword segment_desc, arena positional
        group4_single_kw = group4_fn(
            prefix,
            blocks,
            common,
            families,
            families,
            families,
            kda_state,
            mla_state,
            length,
            group,
            mock_arena,
            segment_desc=segment_desc,
        )
        assert len(group4_single_kw) == 5
        del group4_fn, group4_pos, group4_kw, group4_single_kw
        gc.collect()

        # 5. Test make_final_mla_sp_indexed
        final_fn = prefill_megakernel.make_final_mla_sp_indexed(
            mesh, segments=10, seq_len=2048, prefix_capacity=2048
        )
        final_pos = final_fn(
            prefix,
            blocks,
            common,
            families,
            families,
            mla_state,
            length,
            segment_desc,
            mock_arena,
        )
        assert len(final_pos) == 3

        final_kw = final_fn(
            prefix,
            blocks,
            common,
            families,
            families,
            mla_state,
            length,
            segment_desc=segment_desc,
            arena=mock_arena,
        )
        assert len(final_kw) == 3

        # Single keyword segment_desc, arena positional
        final_single_kw = final_fn(
            prefix,
            blocks,
            common,
            families,
            families,
            mla_state,
            length,
            mock_arena,
            segment_desc=segment_desc,
        )
        assert len(final_single_kw) == 3
        del final_fn, final_pos, final_kw, final_single_kw
        gc.collect()

        assert len(captured_calls) > 0
        for call_record in captured_calls:
            assert call_record["segment_desc"].shape == (10,), (
                f"Expected segment_desc shape (10,), got "
                f"{call_record['segment_desc'].shape}"
            )
            assert call_record["segment_desc"].dtype == jnp.int32, (
                f"Expected segment_desc int32, got {call_record['segment_desc'].dtype}"
            )
            arena_obj = call_record["arena"]
            if arena_obj is not None:
                assert hasattr(arena_obj, "active") or hasattr(arena_obj, "split"), (
                    "arena object must not be swapped with segment_desc"
                )

    del (
        prefix,
        blocks,
        common,
        families,
        kda_state,
        mla_state,
        length,
        segment_desc,
        pair,
        group,
        captured_calls,
    )
    gc.collect()


def test_pool_writer_closures() -> None:
    """Verify PoolWriter copy, write, rings, and state-reading closures."""
    mesh = _mock_mesh_32()
    raw_pools = [torch.zeros((16, 128, 2, 640), dtype=torch.bfloat16)]
    kda_raw = [0] * prefill.KDA_LAYERS
    mla_raw = [0] * prefill.MLA_LAYERS

    with _mock_execution_environment(mesh):
        writer = prefill.PoolWriter(
            mesh=mesh,
            rank=0,
            raw_pools=raw_pools,
            indices=prefill._BoundPoolIndices(
                mla_raw=mla_raw,
                kda_raw=kda_raw,
                state_perm=None,
            ),
        )

        copy_fn = writer.copy_op()
        copy_results = copy_fn(raw_pools[0], torch.tensor(0), torch.tensor(1))
        assert len(copy_results) == 1

        write_fn = writer.write_op(window=2048, slots=1, count=1)
        latents_tensor = torch.zeros((1, 24, 2048, 576), dtype=torch.bfloat16)
        recurrent_tensor = torch.zeros((1, 69, 1, 3, 128, 128), dtype=torch.float32)
        conv_tensor = torch.zeros((1, 69, 1, 3, 3, 3, 128), dtype=torch.bfloat16)
        table_tensor = torch.zeros((1, 16), dtype=torch.int32)
        kda_blocks_tensor = torch.zeros((69,), dtype=torch.int32)
        write_results = write_fn(
            latents_tensor,
            recurrent_tensor,
            conv_tensor,
            recurrent_tensor,
            conv_tensor,
            raw_pools[0],
            table_tensor,
            kda_blocks_tensor,
            kda_blocks_tensor,
        )
        assert len(write_results) == 1

        def _mock_pack_ring(*unused_args: object, **unused_kwargs: object) -> jax.Array:
            return jnp.zeros((1, 69, 30, 640))

        rings_fn = writer.rings_op(count=1)
        with patch.object(dspark, "pack_ring", _mock_pack_ring):
            rings_results = rings_fn(
                raw_pools[0],
                torch.zeros((69,), dtype=torch.int32),
                torch.zeros((1, 16), dtype=torch.bfloat16),
                torch.zeros((1,), dtype=torch.int32),
            )
        assert len(rings_results) == 1

        def _mock_read_draft_rings(
            *unused_args: object, **unused_kwargs: object
        ) -> list[jax.Array]:
            return [jnp.zeros((69, 16))]

        def _mock_unpack_ring(
            *unused_args: object, **unused_kwargs: object
        ) -> tuple[jax.Array, jax.Array]:
            return (
                jnp.zeros((1, 16)),
                jnp.zeros((1,)),
            )

        read_rings_fn = writer.read_rings_op(draft_config=MagicMock(), sequences=1)
        with (
            patch.object(prefill, "read_draft_rings", _mock_read_draft_rings),
            patch.object(dspark, "unpack_ring", _mock_unpack_ring),
        ):
            read_rings_results = read_rings_fn(
                raw_pools[0],
                torch.zeros((69,), dtype=torch.int32),
                torch.tensor([10], dtype=torch.int32),
            )
        assert len(read_rings_results) == 2

        read_states_fn = writer.read_states_op(sequences=1)

        def _mock_gather_region(
            _raw_pool: object,
            token_ids: jax.Array,
            tok0: int = 0,
            **unused_kwargs: object,
        ) -> jax.Array:
            if tok0 == 0:
                return jnp.zeros((token_ids.shape[0], 385, 128), dtype=jnp.float32)
            return jnp.zeros((token_ids.shape[0], 8, 640), dtype=jnp.bfloat16)

        with patch.object(prefill.pool_adapters, "gather_region", _mock_gather_region):
            read_states_results = read_states_fn(
                raw_pools[0], torch.zeros((69,), dtype=torch.int32)
            )
        assert len(read_states_results) == 2


def test_decode_ops_and_decoder_closures() -> None:
    """Verify DecodeOps plain, mask_pack, verify, and KimiDecoder state closures."""
    mesh = _mock_mesh_32()
    control_layout = decode_ops.Control(slots=16, rows=2, pages=16)

    decode_instance = object.__new__(decode_ops.DecodeOps)
    decode_instance.mesh = mesh
    decode_instance.rank = 0
    decode_instance.batch_sizes = (1, 2)
    decode_instance.names = []
    decode_instance.weights = {}
    decode_instance.slots = [torch.zeros((1, 16)), torch.zeros((1, 16))]
    decode_instance.unified_pools = [torch.zeros((1, 16))]
    decode_instance._ops = {}
    decode_instance._shared = {}
    decode_instance.pool_map = None
    decode_instance.owner_table = None
    decode_instance.attention_dp = 1
    decode_instance.VERIFY_ROWS = 8
    decode_instance.paged = MagicMock()

    def _mock_paged_decode(*args: object, **kwargs: object) -> tuple[object, ...]:
        batch_size = int(kwargs.get("batch_size", 2))
        if "aux_hidden_layers" in kwargs:
            return (
                jnp.zeros((batch_size, 16), jnp.float32),
                [jnp.zeros((batch_size, 16))],
                args[1],
                args[2],
                args[3],
            )
        return (
            jnp.zeros((batch_size, 16), jnp.float32),
            args[1],
            args[2],
            args[3],
        )

    def _mock_unpack_verify(
        _packet: jax.Array, _ctrl: object, requests_count: int
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return (
            jnp.zeros((8 * requests_count,), dtype=jnp.int32),
            jnp.zeros((requests_count, 8), dtype=jnp.int32),
            jnp.zeros((requests_count,), dtype=jnp.int32),
        )

    def _mock_finish_verify(
        _ctrl: object,
        states: tuple[object, ...],
        _results: object,
        _configs: object,
    ) -> tuple[jax.Array, object, object]:
        return (
            jnp.zeros((2,), dtype=jnp.int32),
            states[2],
            states[3],
        )

    decode_instance.paged.decode_paged = _mock_paged_decode
    decode_instance._unpack_verify_metadata = _mock_unpack_verify
    decode_instance._finish_verify = _mock_finish_verify

    with _mock_execution_environment(mesh):
        plain_op = decode_instance._build_plain(
            batch=2,
            control=control_layout,
            sub_base=0,
            sampled=False,
            masked=False,
        )
        state_array = jnp.zeros((control_layout.size,), dtype=jnp.int32)
        plain_outputs = plain_op(
            jnp.zeros((1, 16)),
            jnp.zeros((1, 16)),
            jnp.zeros((1, 16)),
            state_array,
        )
        assert len(plain_outputs) == 5

        mask_pack_op = decode_instance._build_mask_pack(control_layout, batch=2)
        mask_pack_outputs = mask_pack_op(
            state_array, jnp.zeros((2, 100), dtype=jnp.int32)
        )
        assert len(mask_pack_outputs) == 1

        verify_op = decode_instance._build_verify(
            aux_layers=(3, 88), control=control_layout
        )
        packet = jnp.zeros((100,), dtype=jnp.int32)
        verify_outputs = verify_op(
            jnp.zeros((1, 16)), jnp.zeros((1, 16)), jnp.zeros((1, 16)), packet
        )
        assert len(verify_outputs) == 6

        verify_batched_op = decode_instance._build_verify_batched(
            aux_layers=(3, 88), requests=2, rows_per=4, control=control_layout
        )
        verify_batched_outputs = verify_batched_op(
            jnp.zeros((1, 16)), jnp.zeros((1, 16)), jnp.zeros((1, 16)), packet
        )
        assert len(verify_batched_outputs) == 6

        decoder = object.__new__(decode.KimiDecoder)
        decoder.ops = decode_instance
        decoder.layout = control_layout
        decoder.sub_base = 0
        decoder.WIDTH = 8

        splice_fn = decoder._splice()
        splice_outputs = splice_fn(
            jnp.zeros((control_layout.size,), dtype=jnp.int32),
            jnp.zeros(
                (control_layout.size - control_layout.plan_start,),
                dtype=jnp.int32,
            ),
        )
        assert len(splice_outputs) == 1

        enter_fn = decoder._enter()
        enter_outputs = enter_fn(
            jnp.zeros((control_layout.size,), dtype=jnp.int32),
            jnp.zeros((1, 100), dtype=jnp.int32),
            jnp.zeros((1,), dtype=jnp.int32),
        )
        assert len(enter_outputs) == 1

        enter_batched_fn = decoder._enter_batched(requests=2, rows_per=4)
        enter_batched_outputs = enter_batched_fn(
            jnp.zeros((control_layout.size,), dtype=jnp.int32),
            jnp.zeros((1, 100), dtype=jnp.int32),
            jnp.zeros((1, 100), dtype=jnp.int32),
            jnp.zeros((2,), dtype=jnp.int32),
        )
        assert len(enter_batched_outputs) == 1


def test_dspark_draft_closures() -> None:
    """Verify DsparkDraft and MlaDsparkDraft single-step closures."""
    mesh = _mock_mesh_32()
    control_layout = decode_ops.Control(slots=16, rows=2, pages=16)
    draft_config = dspark.DraftConfig()

    draft_instance = object.__new__(draft.DsparkDraft)
    draft_instance.config = draft_config
    draft_instance.mesh = mesh
    draft_instance.max_position = 4096
    draft_instance.synthetic_rates = None
    draft_instance.synthetic_rates_by_width = None
    draft_instance.names = []
    draft_instance._ops = {}
    draft_instance.dspark = MagicMock()
    draft_instance.dspark.ROWS = 8
    draft_instance.dspark.OUT_BONUS = 7
    draft_instance.dspark.OUT_COUNT = 0
    draft_instance.dspark.OUT_DRAFTS = 1

    def _mock_make_fused_draft(
        *unused_builder_args: object, **unused_builder_kwargs: object
    ) -> Callable[..., tuple[jax.Array, object, object]]:
        def fused_fn(*call_args: object) -> tuple[jax.Array, object, object]:
            return (
                jnp.zeros((1, 16), dtype=jnp.int32),
                call_args[1],
                call_args[2],
            )

        return fused_fn

    draft_instance.dspark.make_fused_draft = _mock_make_fused_draft

    mla_draft_instance = object.__new__(draft_mla.MlaDsparkDraft)
    mla_draft_instance.config = draft_config
    mla_draft_instance.mesh = mesh
    mla_draft_instance.max_position = 4096
    mla_draft_instance.synthetic_rates = None
    mla_draft_instance.synthetic_rates_by_width = None
    mla_draft_instance.names = []
    mla_draft_instance._ops = {}
    mla_draft_instance.attention_dp = 1
    mla_draft_instance.owner_table = None
    mla_draft_instance.kernel = MagicMock()

    def _mock_fused_draft_local(
        *unused_builder_args: object, **unused_builder_kwargs: object
    ) -> Callable[..., tuple[jax.Array, jax.Array]]:
        def draft_local_fn(*unused_call_args: object) -> tuple[jax.Array, jax.Array]:
            return (
                jnp.zeros((1, 32), dtype=jnp.int32),
                jnp.zeros((1, 8, 16), dtype=jnp.bfloat16),
            )

        return draft_local_fn

    def _mock_commit_latent_rows(
        pools: object, *unused_args: object, **unused_kwargs: object
    ) -> object:
        del unused_args, unused_kwargs
        return pools

    def _mock_register_op(
        _name: str, _mesh: object, body: Callable[..., object], _spec: object
    ) -> Callable[..., object]:
        del _name, _mesh, _spec
        return body

    mla_draft_instance.kernel.fused_draft_local = _mock_fused_draft_local
    mla_draft_instance.paged = MagicMock()
    mla_draft_instance.paged.commit_latent_rows = _mock_commit_latent_rows
    mla_draft_instance._bridge = MagicMock()
    mla_draft_instance._bridge.register_op = _mock_register_op
    mla_draft_instance._bridge.OpRegistrationSpec = MagicMock()
    mla_draft_instance.dspark = MagicMock()
    mla_draft_instance.dspark.ROWS = 8
    mla_draft_instance.dspark.OUT_BONUS = 7
    mla_draft_instance.dspark.OUT_COUNT = 0
    mla_draft_instance.dspark.OUT_DRAFTS = 1
    mla_draft_instance.dspark.make_fused_draft = _mock_make_fused_draft

    with _mock_execution_environment(mesh):
        single_op = draft_instance._build_single(control_layout)
        rows = 8
        ids = jnp.zeros((4 * rows + 10,), dtype=jnp.int32)
        state = jnp.zeros((control_layout.size,), dtype=jnp.int32)
        decisions = jnp.zeros((1, 2), dtype=jnp.int32)
        single_outputs = single_op(
            jnp.zeros((1, 16)),
            jnp.zeros((1, 16)),
            jnp.zeros((1, 16)),
            jnp.zeros((1, 16)),
            jnp.zeros((1, 16)),
            ids,
            state,
            decisions,
        )
        assert len(single_outputs) == 6

        mla_single_op = mla_draft_instance._build_single(control_layout)
        mla_pools = [jnp.zeros((1, 16)) for _ in range(draft_config.layers)]
        mla_single_outputs = mla_single_op(
            *mla_pools,
            jnp.zeros((1, 16)),
            jnp.zeros((1, 16)),
            jnp.zeros((1, 16)),
            ids,
            state,
            decisions,
        )
        assert len(mla_single_outputs) == 9


def test_mla_flash_prefill_packed_unmocked_pallas_trace() -> None:
    """Unmocked jax.eval_shape trace for mla_flash_prefill_packed."""
    tokens = 256
    query = jnp.zeros((3, tokens, 256), dtype=jnp.bfloat16)
    key = jnp.zeros((3, tokens, 256), dtype=jnp.bfloat16)
    value = jnp.zeros((3, tokens, 128), dtype=jnp.bfloat16)
    length = jnp.array([tokens], dtype=jnp.int32)

    # 1. Unmocked trace with arena=None (segments shape (3,))
    segments_no_arena = jnp.zeros((3,), dtype=jnp.int32)
    fn_no_arena = functools.partial(
        prefill_megakernel.mla_flash_prefill_packed, count=1, arena=None
    )
    out_no_arena = jax.eval_shape(
        fn_no_arena,
        query,
        key,
        value,
        length,
        segments_no_arena,
    )
    assert out_no_arena.shape == (3, 256, 128)
    assert out_no_arena.dtype == jnp.bfloat16

    # 2. Unmocked trace with arena=(arena_keys, arena_values) (segments shape (5,))
    arena_keys = jnp.zeros((1, 3, 128, 256), dtype=jnp.bfloat16)
    arena_values = jnp.zeros((1, 3, 128, 128), dtype=jnp.bfloat16)
    segments_arena = jnp.zeros((5,), dtype=jnp.int32)
    fn_arena = functools.partial(
        prefill_megakernel.mla_flash_prefill_packed,
        count=1,
        arena=(arena_keys, arena_values),
    )
    out_arena = jax.eval_shape(
        fn_arena,
        query,
        key,
        value,
        length,
        segments_arena,
    )
    assert out_arena.shape == (3, 256, 128)
    assert out_arena.dtype == jnp.bfloat16

    # 3. Direct unit check on _load_kv_block with partial keyword args
    class _MockCopy:
        def wait(self) -> None:
            pass

        def start(self) -> None:
            pass

    def _mock_copies(
        _key_block: int | jax.Array, _slot: int | jax.Array
    ) -> tuple[_MockCopy, _MockCopy]:
        return _MockCopy(), _MockCopy()

    k_buf = jnp.zeros((2, 128, 256), dtype=jnp.bfloat16)
    v_buf = jnp.zeros((2, 128, 128), dtype=jnp.bfloat16)

    load_fn = functools.partial(
        prefill_megakernel._load_kv_block,
        hi=4,
        copies_fn=_mock_copies,
        k_buf=k_buf,
        v_buf=v_buf,
        tile=128,
        value_mask=None,
    )
    k_res, v_res = load_fn(0)
    assert k_res.shape == (128, 256)
    assert v_res.shape == (136, 128)

    del (
        query,
        key,
        value,
        length,
        segments_no_arena,
        out_no_arena,
        arena_keys,
        arena_values,
        segments_arena,
        out_arena,
        k_buf,
        v_buf,
        k_res,
        v_res,
    )
    gc.collect()
