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
from collections.abc import Callable, Generator
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
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


def _torch_to_jax(tensor_or_val: object) -> object:
    """Convert torch.Tensor to jax.Array safely on CPU."""
    if not isinstance(tensor_or_val, torch.Tensor):
        return tensor_or_val
    if tensor_or_val.dtype == torch.bfloat16:
        return jnp.asarray(tensor_or_val.float().numpy(), dtype=jnp.bfloat16)
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


def _mock_kda(*_args: object, **_kwargs: object) -> Callable[..., object]:
    """Mock leaf KDA SPMD kernel builder."""

    def kda_fn(
        prefix: jax.Array,
        _blocks: jax.Array,
        _attn_res: jax.Array,
        _attn_norm: jax.Array,
        _families: object,
        rec: jax.Array,
        hist: jax.Array,
        _active: jax.Array,
        _length: jax.Array,
        _slot: object,
        _seg: object,
    ) -> list[object]:
        """Mock leaf KDA kernel implementation."""
        return [
            prefix,
            prefix,
            prefix,
            rec[:, 0],
            hist[:, 0],
            prefix,
            rec[:, 0],
            hist[:, 0],
        ]

    return kda_fn


def _mock_mla(*_args: object, **_kwargs: object) -> Callable[..., object]:
    """Mock leaf MLA SPMD kernel builder."""
    return lambda _gathered, prefix, _families, _seg, _length, _slot, _arena: (
        prefix,
        jnp.zeros((32, 64, 576), jnp.bfloat16),
    )


def _mock_group4(*_args: object, **_kwargs: object) -> Callable[..., object]:
    """Mock group4 SPMD kernel builder."""

    def _group4_fn(
        prefix: jax.Array,
        blocks: jax.Array,
        _common: object,
        _kda_f: object,
        _mla_f: object,
        _moe_f: object,
        kda_s: object,
        mla_s: object,
        _length: object,
        _group: object,
        _seg: object,
        _arena: object,
    ) -> tuple[object, ...]:
        """Mock group4 SPMD kernel implementation."""
        return prefix, blocks, kda_s, mla_s, prefix

    return _group4_fn


@contextlib.contextmanager
def _mock_execution_environment(
    mesh: MagicMock,
) -> Generator[None, None, None]:
    """Provide scoped CPU mocks for megakernel execution without global leakage."""
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

    patches = [
        patch.object(torch.Tensor, "to", _safe_to),
        patch.object(jax, "jit", lambda fn, **_kw: fn),
        patch.object(jax, "shard_map", lambda fn, *_a, **_kw: fn),
        patch.object(pool_adapters, "scatter_region", lambda pool, *_a, **_kw: pool),
        patch.object(
            paged,
            "_commit_kda_states",
            lambda conv, rec, *_a, **_kw: (conv, rec),
        ),
        patch.object(
            paged,
            "decode_paged",
            lambda *_a, **_kw: (jnp.zeros((1, 16)), _a[1], _a[2], ()),
        ),
        patch.object(jax.lax, "axis_index", lambda _axis: jnp.int32(0)),
        patch.object(jax.lax, "psum", lambda x, *_a, **_kw: x),
        patch.object(jax.lax, "all_gather", lambda x, *_a, **_kw: x),
        patch.object(jax.lax, "dynamic_update_slice", lambda op, _up, *_idx: op),
        patch.object(
            jax.lax,
            "dynamic_index_in_dim",
            lambda op, _idx, axis=0, keepdims=False: jnp.take(op, 0, axis=axis),
        ),
        patch.object(jax.lax, "fori_loop", lambda _l, _u, _body, init, **_kw: init),
        patch.object(bridge, "kernel_mesh", lambda _axis, size=None: (mesh, 0)),
        patch.object(prefill, "kernel_mesh", lambda _axis, size=None: (mesh, 0)),
        patch.object(bridge, "register_global_op", _intercept_register_op),
        patch.object(bridge, "register_op", _intercept_register_op),
        patch.object(prefill, "register_global_op", _intercept_register_op),
        patch.object(prefill, "register_op", _intercept_register_op),
        patch.object(decode_ops, "register_op", _intercept_register_op),
        patch.object(decode, "register_global_op", _intercept_register_op),
        patch.object(draft, "register_global_op", _intercept_register_op),
        patch.object(prefill_megakernel, "NamedSharding", lambda _m, _p: MagicMock()),
        patch.object(
            prefill.PrefillProgram,
            "_zeros_sharded",
            lambda _self, shape, dtype: jnp.zeros(shape, dtype),
        ),
        patch.object(
            prefill_megakernel,
            "make_embedding_sp",
            lambda _m, **_kw: lambda _tokens, _emb, _length: jnp.zeros(
                (32, 64, 7168), jnp.bfloat16
            ),
        ),
        patch.object(prefill_megakernel, "make_norm_gather_kda_sp", _mock_kda),
        patch.object(
            prefill_megakernel,
            "make_mixture_norm_sp",
            lambda *_a, **_kw: lambda after, _b, _m, _f, _act: after,
        ),
        patch.object(
            prefill_megakernel,
            "make_dense_ffn_sp",
            lambda *_a, **_kw: lambda _f, after, _gu, _d, _l: after,
        ),
        patch.object(
            prefill_megakernel,
            "make_norm_gather_sp",
            lambda *_a, **_kw: lambda prefix, _b, _r, _n, _act: (
                prefix,
                prefix,
            ),
        ),
        patch.object(prefill_megakernel, "make_indexed_mla_layer_sp_ring", _mock_mla),
        patch.object(
            prefill_megakernel,
            "make_indexed_moe_sp_ring",
            lambda *_a, **_kw: lambda _f, after, _fam, _l, _act: after,
        ),
        patch.object(
            prefill_megakernel, "make_regular_group4_sp_indexed", _mock_group4
        ),
    ]

    for patch_item in patches:
        patch_item.start()
    try:
        yield
    finally:
        for patch_item in reversed(patches):
            patch_item.stop()


def test_prefill_program_window_ops_and_state_closures() -> None:
    """Verify PrefillProgram first, group, final, stack, and split closures."""
    mesh = _mock_mesh_32()
    with _mock_execution_environment(mesh):
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
                "lm_head": torch.zeros((32, 7168, 4096), dtype=torch.bfloat16),
            }
        )
        program.weights["common"].update(
            {
                "attn_res": torch.zeros((32, 93, 64, 7168), dtype=torch.bfloat16),
                "attn_norm": torch.zeros((32, 93, 7168), dtype=torch.bfloat16),
                "mlp_res": torch.zeros((32, 93, 64, 7168), dtype=torch.bfloat16),
                "ffn_norm": torch.zeros((32, 93, 7168), dtype=torch.bfloat16),
                "dense_gu": torch.zeros((32, 7168, 2048), dtype=torch.bfloat16),
                "dense_down": torch.zeros((32, 2048, 7168), dtype=torch.bfloat16),
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
            hidden,
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

        blocks_tensor = torch.zeros((32, 8, 64, 7168), dtype=torch.bfloat16)
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

        stack_op = program.state_op("stack", count=2)
        states_in = [jnp.zeros((69, 16, 128, 128), jnp.float32) for _ in range(4)]
        stacked_states = stack_op(*states_in)
        assert len(stacked_states) == 2

        split_op = program.state_op("split", count=2)
        split_states = split_op(stacked_states[0], stacked_states[1])
        assert len(split_states) == 4


def test_prefill_megakernel_closures_support_positional_and_keyword_args() -> None:
    """Verify prefill megakernel closures accept segment_desc and arena both ways."""
    mesh = _mock_mesh_32()
    mock_arena = MagicMock()
    mock_arena.split.return_value = (mock_arena, mock_arena)
    mock_arena.active = jnp.zeros((1,), jnp.int32)
    mock_arena.arenas = (mock_arena, mock_arena)

    prefix = jnp.zeros((32, 64, 7168), jnp.bfloat16)
    blocks = jnp.zeros((32, 8, 64, 7168), jnp.bfloat16)
    common = {
        "attn_res": jnp.zeros((32, 93, 64, 7168), jnp.bfloat16),
        "attn_norm": jnp.zeros((32, 93, 7168), jnp.bfloat16),
        "mlp_res": jnp.zeros((32, 93, 64, 7168), jnp.bfloat16),
        "ffn_norm": jnp.zeros((32, 93, 7168), jnp.bfloat16),
        "dense_gu": jnp.zeros((32, 7168, 2048), jnp.bfloat16),
        "dense_down": jnp.zeros((32, 2048, 7168), jnp.bfloat16),
    }
    families = MagicMock()
    kda_state = (
        jnp.zeros((32, 16, 128, 128), jnp.float32),
        jnp.zeros((32, 4, 16, 16, 128), jnp.bfloat16),
        jnp.zeros((32, 16, 128, 128), jnp.float32),
        jnp.zeros((32, 4, 16, 16, 128), jnp.bfloat16),
    )
    mla_state = (jnp.zeros((32, 24, 64, 576), jnp.bfloat16),)
    length = jnp.asarray([64], jnp.int32)
    segment_desc = jnp.zeros((10,), jnp.int32)
    pair = jnp.asarray([0], jnp.int32)

    with _mock_execution_environment(mesh):
        stack_fn = prefill_megakernel.make_prefill_stack_sp_indexed(
            mesh, segments=10, seq_len=2048, prefix_capacity=2048
        )
        out_pos = stack_fn(
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
        assert len(out_pos) == 4

        out_kw = stack_fn(
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
        assert len(out_kw) == 4

        group8_fn = prefill_megakernel.make_regular_group8_sp_indexed(
            mesh, segments=10, seq_len=2048, prefix_capacity=2048
        )
        out_grp_pos = group8_fn(
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
        assert len(out_grp_pos) == 6

        out_grp_kw = group8_fn(
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
        assert len(out_grp_kw) == 6

        final_fn = prefill_megakernel.make_final_mla_sp_indexed(
            mesh, segments=10, seq_len=2048, prefix_capacity=2048
        )
        out_fin_pos = final_fn(
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
        assert len(out_fin_pos) == 3

        out_fin_kw = final_fn(
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
        assert len(out_fin_kw) == 3


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
        copy_res = copy_fn(raw_pools[0], torch.tensor(0), torch.tensor(1))
        assert len(copy_res) == 1

        write_fn = writer.write_op(window=2048, slots=1, count=1)
        latents_tensor = torch.zeros((1, 24, 2048, 576), dtype=torch.bfloat16)
        recurrent_tensor = torch.zeros((1, 69, 1, 3, 128, 128), dtype=torch.float32)
        conv_tensor = torch.zeros((1, 69, 1, 3, 3, 3, 128), dtype=torch.bfloat16)
        table_tensor = torch.zeros((1, 16), dtype=torch.int32)
        kda_blocks_tensor = torch.zeros((69,), dtype=torch.int32)
        write_res = write_fn(
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
        assert len(write_res) == 1

        rings_fn = writer.rings_op(count=1)
        with patch.object(
            dspark,
            "pack_ring",
            lambda _caches, _metas, _layers: jnp.zeros((1, 69, 30, 640)),
        ):
            rings_res = rings_fn(
                raw_pools[0],
                torch.zeros((69,), dtype=torch.int32),
                torch.zeros((1, 16), dtype=torch.bfloat16),
                torch.zeros((1,), dtype=torch.int32),
            )
        assert len(rings_res) == 1

        read_rings_fn = writer.read_rings_op(draft_config=MagicMock(), sequences=1)
        with (
            patch.object(
                prefill,
                "read_draft_rings",
                lambda _raws, _kda, _blk, _lanes: [jnp.zeros((69, 16))],
            ),
            patch.object(
                dspark,
                "unpack_ring",
                lambda _cfg, _packed, _length: (
                    jnp.zeros((1, 16)),
                    jnp.zeros((1,)),
                ),
            ),
        ):
            read_rings_res = read_rings_fn(
                raw_pools[0],
                torch.zeros((69,), dtype=torch.int32),
                torch.tensor([10], dtype=torch.int32),
            )
        assert len(read_rings_res) == 2

        read_states_fn = writer.read_states_op(sequences=1)
        with patch.object(
            prefill.pool_adapters,
            "gather_region",
            lambda _raw, ids, tok0=0, **_kw: (
                jnp.zeros((ids.shape[0], 385, 128), dtype=jnp.float32)
                if tok0 == 0
                else jnp.zeros((ids.shape[0], 8, 640), dtype=jnp.bfloat16)
            ),
        ):
            read_states_res = read_states_fn(
                raw_pools[0], torch.zeros((69,), dtype=torch.int32)
            )
        assert len(read_states_res) == 2


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
    decode_instance.paged.decode_paged = lambda *_a, **kw: (
        (
            jnp.zeros((kw.get("batch_size", 2), 4096), jnp.float32),
            [jnp.zeros((kw.get("batch_size", 2), 7168))],
            _a[1],
            _a[2],
            _a[3],
        )
        if "aux_hidden_layers" in kw
        else (
            jnp.zeros((kw.get("batch_size", 2), 4096), jnp.float32),
            _a[1],
            _a[2],
            _a[3],
        )
    )
    decode_instance._unpack_verify_metadata = lambda _packet, _ctrl, reqs: (
        jnp.zeros((8 * reqs,), dtype=jnp.int32),
        jnp.zeros((reqs, 8), dtype=jnp.int32),
        jnp.zeros((reqs,), dtype=jnp.int32),
    )
    decode_instance._finish_verify = lambda _ctrl, states, _results, _configs: (
        jnp.zeros((2,), dtype=jnp.int32),
        states[2],
        states[3],
    )

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
    draft_instance.dspark.make_fused_draft = lambda *_a, **_kw: lambda *args: (
        jnp.zeros((1, 16), dtype=jnp.int32),
        args[1],
        args[2],
    )

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
    mla_draft_instance.kernel.fused_draft_local = lambda *_a, **_kw: lambda *_args: (
        jnp.zeros((1, 32), dtype=jnp.int32),
        jnp.zeros((1, 8, 128), dtype=jnp.bfloat16),
    )
    mla_draft_instance.paged = MagicMock()
    mla_draft_instance.paged.commit_latent_rows = lambda pools, *_a, **_kw: pools
    mla_draft_instance._bridge = MagicMock()
    mla_draft_instance._bridge.register_op = lambda _name, _mesh, body, _spec: body
    mla_draft_instance._bridge.OpRegistrationSpec = MagicMock()
    mla_draft_instance.dspark = MagicMock()
    mla_draft_instance.dspark.ROWS = 8
    mla_draft_instance.dspark.OUT_BONUS = 7
    mla_draft_instance.dspark.OUT_COUNT = 0
    mla_draft_instance.dspark.OUT_DRAFTS = 1
    mla_draft_instance.dspark.make_fused_draft = lambda *_a, **_kw: lambda *args: (
        jnp.zeros((1, 16), dtype=jnp.int32),
        args[1],
        args[2],
    )

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
