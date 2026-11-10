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
"""CPU lowering tests for overlapping `transport.finish_call` with row scatter."""

import dataclasses
import re
from unittest import mock

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import (
    AbstractDevice,
    AbstractMesh,
    NamedSharding,
    use_abstract_mesh,
)
from jax.sharding import (
    PartitionSpec as P,
)

from vllm_torchtpu.kernels.fused_moe import v3 as fused_ep
from vllm_torchtpu.kernels.fused_moe.v3 import (
    Config,
    layout,
    transport,
)
from vllm_torchtpu.kernels.fused_moe.v3 import (
    ffn_kernel as ffn_kernel_mod,
)
from vllm_torchtpu.kernels.fused_moe.v3 import (
    layer as layer_mod,
)


def _assert_finish_overlap_dag(hlo: str, *, bridge_mode: bool) -> None:
    """Assert SSA data-flow ordering of `finish_call` relative to tables and gather."""
    computations = re.split(r"\n(?=[A-Za-z_%].*\{)", hlo)
    target_block = next(
        b
        for b in computations
        if 'custom_call_target="tpu_custom_call"' in b and "s32[1,128]" in b
    )
    defs: dict[str, tuple[str, list[str]]] = {}
    for raw in target_block.splitlines():
        if (
            not raw.startswith("  ")
            or "=" not in raw
            or raw.lstrip().startswith(("ENTRY", "}", "//", '"'))
        ):
            continue
        line = raw.strip().removeprefix("ROOT ").strip()
        lhs, rhs = line.split("=", 1)
        name = lhs.strip().lstrip("%")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", name):
            continue
        head = rhs.split(", custom_call_target=")[0] if "custom-call(" in rhs else rhs
        m = re.search(r"[a-z][a-z0-9_-]*\(([^()]*)\)", head)
        ops = (
            [o.strip().lstrip("%") for o in m.group(1).split(",") if o.strip()]
            if m
            else []
        )
        defs[name] = (rhs.strip(), ops)

    def ancestors(root: str) -> set[str]:
        seen: set[str] = set()
        stack = [root]
        while stack:
            cur = stack.pop()
            if cur in seen or cur not in defs:
                continue
            seen.add(cur)
            stack.extend(defs[cur][1])
        return seen

    scatter_calls = [
        n
        for n, (rhs, _) in defs.items()
        if " call(" in f" {rhs}" and re.search(r"\(s32\[\d+\].*,\s*s32\[\d+\]", rhs)
    ]
    assert len(scatter_calls) == 1, scatter_calls
    scatter_call = scatter_calls[0]

    alias_op = 2 if bridge_mode else 1
    shard_tables_calls = [
        n
        for n, (rhs, _) in defs.items()
        if f"output_to_operand_aliasing={{{{14}}: ({alias_op}, {{}})}}" in rhs
    ]
    assert len(shard_tables_calls) == 1, shard_tables_calls
    shard_tables_call = shard_tables_calls[0]

    finish_calls = [
        n
        for n, (rhs, _) in defs.items()
        if 'custom_call_target="tpu_custom_call"' in rhs
        and rhs.startswith("s32[1,128]")
    ]
    assert len(finish_calls) == 1, finish_calls
    finish_call = finish_calls[0]
    finish_rhs, finish_ops = defs[finish_call]

    assert len(finish_ops) == 3 + int(bridge_mode), finish_ops
    assert ("custom_call_has_side_effect=true" in finish_rhs) == (not bridge_mode)

    landing_op = finish_ops[1 if bridge_mode else 0]
    landing_rhs, landing_parents = defs[landing_op]
    assert "get-tuple-element(" in landing_rhs and "index=14" in landing_rhs
    assert landing_parents == [shard_tables_call], landing_parents

    finish_anc = ancestors(finish_call)
    assert shard_tables_call in finish_anc
    assert scatter_call not in finish_anc

    opt_barriers = [n for n, (rhs, _) in defs.items() if "opt-barrier(" in rhs]
    assert len(opt_barriers) == 1, opt_barriers
    opt_anc = ancestors(opt_barriers[0])
    assert finish_call in opt_anc and landing_op in opt_anc

    sort_calls = [
        n
        for n, (rhs, _) in defs.items()
        if " call(" in f" {rhs}" and re.match(r"(f8e4m3fn|bf16)\[\d+,\d+,128\]", rhs)
    ]
    assert len(sort_calls) == 1, sort_calls
    sort_anc = ancestors(sort_calls[0])
    assert opt_barriers[0] in sort_anc and scatter_call in sort_anc


class OverlapFinishTest(parameterized.TestCase):
    @parameterized.parameters(False, True)
    def test_finish_call_lowering_arity(self, pass_shard_index: bool):
        info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
        mesh = AbstractMesh(
            (8,), (layout.MESH_AXIS,), abstract_device=AbstractDevice("TPU7x", 1, "tpu")
        )
        rows = jax.ShapeDtypeStruct(
            (8 * 384, 4096),
            jnp.bfloat16,
            sharding=NamedSharding(mesh, P("d", None)),
        )
        msg = jax.ShapeDtypeStruct(
            (8 * 40, 128), jnp.int32, sharding=NamedSharding(mesh, P("d", None))
        )
        shard_idx = jax.ShapeDtypeStruct(
            (8,), jnp.int32, sharding=NamedSharding(mesh, P("d"))
        )

        def shard_fn(rows, msg, shard_idx):
            gathered, send_sem, recv_sems, _ = transport.start_transport(
                rows,
                msg,
                scale_base=32,
                num_shards=8,
                shard_index=shard_idx if pass_shard_index else None,
            )
            return transport.finish_call(
                gathered,
                send_sem,
                recv_sems,
                num_shards=8,
                rows_per_shard=384,
                shard_index=shard_idx if pass_shard_index else None,
            )

        fn = jax.shard_map(
            shard_fn,
            mesh=mesh,
            in_specs=(
                P(layout.MESH_AXIS, None),
                P(layout.MESH_AXIS, None),
                P(layout.MESH_AXIS),
            ),
            out_specs=P(layout.MESH_AXIS, None),
            check_vma=False,
        )

        with (
            use_abstract_mesh(mesh),
            mock.patch.object(pltpu, "get_tpu_info", return_value=info),
            mock.patch.object(layout, "SHARD_INDEX_OPERAND", pass_shard_index),
        ):
            hlo = (
                jax.jit(fn)
                .trace(rows, msg, shard_idx)
                .lower(lowering_platforms=("tpu",))
                .as_text("hlo")
            )
        finish_lines = [
            line
            for line in hlo.splitlines()
            if 'custom_call_target="tpu_custom_call"' in line and "s32[1,128]" in line
        ]
        self.assertLen(finish_lines, 1)
        expected_arity = 3 + int(pass_shard_index)
        m = re.search(r"custom-call\(([^)]*)\)", finish_lines[0])
        self.assertIsNotNone(m)
        assert m is not None
        operands = [op.strip() for op in m.group(1).split(",") if op.strip()]
        self.assertLen(operands, expected_arity)

    @parameterized.parameters(
        (8, 1024, 512, 10, 128, False),
        (8, 1024, 512, 10, 128, True),
        (8, 384, 512, 10, 128, False),
        (8, 384, 512, 10, 128, True),
        (8, 48, 512, 10, 16, False),
        (8, 48, 512, 10, 16, True),
        (4, 384, 512, 10, 128, True),
        (2, 384, 512, 10, 128, True),
    )
    def test_sorted_rows_finish_call_decoupled_from_scatter_output(
        self,
        num_shards: int,
        local_tokens: int,
        num_experts: int,
        top_k: int,
        tile_rows: int,
        bridge_mode: bool,
    ):
        info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
        mesh = AbstractMesh(
            (num_shards,),
            (layout.MESH_AXIS,),
            abstract_device=AbstractDevice("TPU7x", 1, "tpu"),
        )
        experts_per_shard = num_experts // num_shards
        d_sharding = NamedSharding(mesh, P(layout.MESH_AXIS))
        x = jax.ShapeDtypeStruct(
            (local_tokens * num_shards, 4096),
            jnp.bfloat16,
            sharding=d_sharding if bridge_mode else None,
        )
        w1 = jax.ShapeDtypeStruct(
            (num_experts, 4096, 2048),
            jnp.float8_e4m3fn,
            sharding=d_sharding if bridge_mode else None,
        )
        w2 = jax.ShapeDtypeStruct(
            (num_experts, 1024, 4096),
            jnp.float8_e4m3fn,
            sharding=d_sharding if bridge_mode else None,
        )
        w1_s = jax.ShapeDtypeStruct(
            (num_experts, 2048),
            jnp.float32,
            sharding=d_sharding if bridge_mode else None,
        )
        w2_s = jax.ShapeDtypeStruct(
            (num_experts, 4096),
            jnp.float32,
            sharding=d_sharding if bridge_mode else None,
        )
        logits = jax.ShapeDtypeStruct(
            (local_tokens * num_shards, num_experts),
            jnp.float32,
            sharding=d_sharding if bridge_mode else None,
        )
        rank = jax.ShapeDtypeStruct((num_shards, 1), jnp.int32, sharding=d_sharding)
        cfg = dataclasses.replace(
            Config.for_shape(
                hidden=4096,
                inter=1024,
                experts_per_shard=experts_per_shard,
                top_k=top_k,
            ),
            sorted_rows=True,
        )
        with (
            use_abstract_mesh(mesh),
            mock.patch.object(pltpu, "get_tpu_info", return_value=info),
            mock.patch.object(layer_mod, "WRAP_IN_JIT", not bridge_mode),
            mock.patch.object(layout, "SIDE_EFFECTS", not bridge_mode),
            mock.patch.object(layout, "SHARD_INDEX_OPERAND", bridge_mode),
            mock.patch.dict(ffn_kernel_mod._BUILD_CACHE, {}, clear=True),
            mock.patch.dict(layer_mod._LAYER_CACHE, {}, clear=True),
        ):
            if bridge_mode:
                hlo = (
                    jax.jit(
                        lambda x, w1, w2, w1_s, w2_s, logits, rank: (
                            fused_ep.fused_ep_moe(
                                x,
                                w1,
                                w2,
                                w1_s,
                                w2_s,
                                logits,
                                top_k=top_k,
                                renormalize=True,
                                mesh=mesh,
                                tile_rows=tile_rows,
                                config=cfg,
                                shard_index=rank,
                            )
                        )
                    )
                    .trace(x, w1, w2, w1_s, w2_s, logits, rank)
                    .lower(lowering_platforms=("tpu",))
                    .as_text("hlo")
                )
            else:
                hlo = (
                    jax.jit(
                        lambda x, w1, w2, w1_s, w2_s, logits: fused_ep.fused_ep_moe(
                            x,
                            w1,
                            w2,
                            w1_s,
                            w2_s,
                            logits,
                            top_k=top_k,
                            renormalize=True,
                            mesh=mesh,
                            tile_rows=tile_rows,
                            config=cfg,
                        )
                    )
                    .trace(x, w1, w2, w1_s, w2_s, logits)
                    .lower(lowering_platforms=("tpu",))
                    .as_text("hlo")
                )
        _assert_finish_overlap_dag(hlo, bridge_mode=bridge_mode)


if __name__ == "__main__":
    absltest.main()
