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
"""Tests for `megakernel_runtime.exported_bridge`: HLO donation and pointer identity."""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any

_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_force_host_platform_device_count" not in _xla_flags:
    os.environ["XLA_FLAGS"] = (
        f"{_xla_flags} --xla_force_host_platform_device_count=8".strip()
    )

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

import vllm_torchtpu.megakernel as mk  # noqa: E402
from vllm_torchtpu.megakernel import planner  # noqa: E402
from vllm_torchtpu.megakernel.exec_spec import ExecSpec  # noqa: E402
from vllm_torchtpu.megakernel.models import qwen35  # noqa: E402
from vllm_torchtpu.megakernel.models.qwen35_profile import TINY_TILES  # noqa: E402
from vllm_torchtpu.megakernel_runtime.exported_bridge import (  # noqa: E402
    build_pooled_step_op,
    execute_pooled_step_inplace,
    export_pooled_step,
    verify_pooled_step_hlo,
    verify_torchtpu_compiled_step_ir,
)


def _cpu_mesh(r: int) -> Mesh:
    devices = np.array(jax.devices("cpu")[:r])
    return Mesh(devices, ("tp",))


def _tiny_geom(L: int = 4, att_every: int = 4) -> qwen35.Qwen35Geometry:
    return qwen35.Qwen35Geometry(
        L=L,
        att_every=att_every,
        H=256,
        NH=4,
        NKV=1,
        HD=256,
        ROT=64,
        KH=2,
        VH=4,
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=256,
        IS=256,
        moe_tp=False,
        e_slots=0,
    )


def _tiny_exec(
    R: int = 1,
    B: int = 2,
    T: int = 1,
    ctx: int = 256,
    num_pages: int = 64,
    stack_e: bool = False,
) -> ExecSpec:
    tiles = replace(TINY_TILES, bank=512 << 10).with_blocks(
        {"gqa": {"kv_blk": min(TINY_TILES.block("gqa").kv_blk, ctx)}}
    )
    return ExecSpec(
        R=R,
        B=B,
        T=T,
        ctx=ctx,
        stack_e=stack_e,
        tiles=tiles,
        interpret=True,
        cache=mk.CacheExec(paged_kv=True, num_pages=num_pages, max_pages_per_seq=2),
    )


def _jax_to_torch(a: jax.Array) -> torch.Tensor:
    arr = np.asarray(a)
    if a.dtype == jnp.float8_e4m3fn:
        return torch.from_numpy(arr.view(np.uint8).copy()).view(torch.float8_e4m3fn)
    if a.dtype == jnp.bfloat16:
        return torch.from_numpy(arr.view(np.uint16).copy()).view(torch.bfloat16)
    return torch.from_numpy(arr.copy())


def _flatten_weights_to_torch(
    w_mk: dict[str, Any], plan: planner.Plan
) -> list[torch.Tensor]:
    w_flat: list[torch.Tensor] = []
    for k in plan.w_names:
        v = w_mk[k]
        if isinstance(v, (tuple, list)):
            w_flat.extend(_jax_to_torch(x) for x in v)
        else:
            w_flat.append(_jax_to_torch(v))
    return w_flat


@pytest.mark.parametrize("T,B", [(1, 2), (2, 4)])
def test_exported_bridge_raiden_pointer_identity_and_bit_exact(T: int, B: int) -> None:
    """Verify `execute_pooled_step_inplace` preserves pool `data_ptr()` and parity."""
    mesh = _cpu_mesh(1)
    tp = NamedSharding(mesh, P("tp"))
    g = _tiny_geom(L=4, att_every=4)
    ex = _tiny_exec(R=1, B=B, T=T, ctx=256, num_pages=96, stack_e=False)
    spec = qwen35.qwen35_spec(g)
    plan = planner.plan(spec, ex)

    pos_j, hid_l, w_l, st_l = qwen35.random_logical_inputs(
        jax.random.key(500 + T), g, ex
    )
    _, hid_mk, w_mk, _ = qwen35.pack_inputs(
        pos_j, hid_l, w_l, st_l, g, ex, mesh=mesh, build_default_pool=True
    )
    pt_j, ckpt_j, roff_j, _ = mk.default_pool_layout(g, ex)
    raw_pools_j = tuple(
        jax.device_put(r, tp)
        for r in mk.states_to_pool(g, ex, st_l, pt_j, ckpt_j, roff_j)
    )
    unsh = NamedSharding(mesh, P())
    ckpt_step_j = jax.device_put(ckpt_j, unsh)
    roff_step_j = jax.device_put(roff_j, unsh)
    pt_step_j = jax.device_put(pt_j, unsh)
    pos_step_j = jax.device_put(pos_j, unsh)

    # 1. Pure-JAX reference pooled step
    step_jax, _ = qwen35.make_pooled_step(mesh, g, ex, donate_pools=False)
    hid_ref_j, raw_ref_j = step_jax(
        pos_step_j, hid_mk, w_mk, raw_pools_j, pt_step_j, ckpt_step_j, roff_step_j
    )

    # 2. Convert inputs to PyTorch base buffers (with Raiden-style pinned storage)
    w_flat_t = _flatten_weights_to_torch(w_mk, plan)
    pools_t = [_jax_to_torch(r).clone() for r in raw_pools_j]
    orig_data_ptrs = [p.data_ptr() for p in pools_t]
    orig_storage_ptrs = [p.untyped_storage().data_ptr() for p in pools_t]
    orig_pool_snap = [p.view(torch.uint8).clone() for p in pools_t]

    pos_t = _jax_to_torch(pos_step_j)
    hid_t = _jax_to_torch(hid_mk)
    pt_t = _jax_to_torch(pt_step_j)
    ckpt_t = _jax_to_torch(ckpt_step_j)
    roff_t = _jax_to_torch(roff_step_j)

    # 3. Build and execute the bridge op in-place
    op, in_specs, out_specs, donate_argnums = build_pooled_step_op(
        mesh, spec, ex, donate_pools=True
    )
    assert donate_argnums == tuple(
        range(2 + len(w_flat_t), 2 + len(w_flat_t) + len(pools_t))
    )
    assert len(in_specs) == 2 + len(w_flat_t) + len(pools_t) + 3
    assert len(out_specs) == 1 + len(pools_t)

    hid_out_t = execute_pooled_step_inplace(
        op, pos_t, hid_t, pt_t, ckpt_t, roff_t, w_flat_t, pools_t
    )

    # 4. Assert Raiden pointer identity: neither data_ptr() nor storage ptr changed
    for idx, p in enumerate(pools_t):
        assert p.data_ptr() == orig_data_ptrs[idx]
        assert p.untyped_storage().data_ptr() == orig_storage_ptrs[idx]
        assert p.storage_offset() == 0
        # Pool contents were updated in-place
        assert not torch.equal(p.view(torch.uint8), orig_pool_snap[idx])

    # 5. Assert uint8 bit-exactness against JAX `make_pooled_step`
    np.testing.assert_array_equal(
        hid_out_t.view(torch.uint8).numpy(),
        np.asarray(hid_ref_j).view(np.uint8),
    )
    for p_t, p_ref in zip(pools_t, raw_ref_j, strict=True):
        np.testing.assert_array_equal(
            p_t.view(torch.uint8).numpy(),
            np.asarray(p_ref).view(np.uint8),
        )


def test_verify_and_export_pooled_step_hlo() -> None:
    """Verify `verify_pooled_step_hlo` and `export_pooled_step` on a sharded step."""
    mesh = _cpu_mesh(2)
    g = replace(_tiny_geom(L=4, att_every=4), NKV=2)
    ex = _tiny_exec(R=2, B=2, T=1, ctx=256, num_pages=64, stack_e=False)
    spec = qwen35.qwen35_spec(g)
    plan = planner.plan(spec, ex)
    w_shapes = planner.weight_shapes(spec, ex, plan)

    # Per-rank (unfolded) shapes for `verify_pooled_step_hlo` / `export_pooled_step`
    w_flat_local: list[torch.Tensor] = []
    for k in plan.w_names:
        w_spec = w_shapes[k]
        if k in plan.dynamic_kinds and not ex.stack_e:
            dt = w_spec[0].dtype
            t_dt = (
                torch.float8_e4m3fn
                if dt == jnp.float8_e4m3fn
                else torch.bfloat16
                if dt == jnp.bfloat16
                else torch.float32
            )
            ew = g.EW(ex.R)
            k_dim = g.H if k == "e13" else g.IR(ex.R)
            n_dim = 2 * g.IR(ex.R) if k == "e13" else g.H
            for _ in range(g.L):
                w_flat_local.append(torch.zeros((ew, k_dim, n_dim), dtype=t_dt))
        else:
            sh, dt = w_spec
            t_dt = (
                torch.float8_e4m3fn
                if dt == jnp.float8_e4m3fn
                else torch.bfloat16
                if dt == jnp.bfloat16
                else torch.float32
            )
            w_flat_local.append(torch.zeros(sh, dtype=t_dt))

    pools_local = [
        torch.zeros(
            (mk.cache_exec(ex).num_pages, 2, 64, 4, 128), dtype=torch.float8_e4m3fn
        )
        for _ in range(g.NATT)
    ]

    ok, msg = verify_pooled_step_hlo(
        mesh,
        spec,
        ex,
        w_flat_local,
        pools_local,
        max_pages_per_seq=mk.cache_exec(ex).max_pages_per_seq,
        compile_hlo=True,
    )
    assert ok, f"verify_pooled_step_hlo failed: {msg}"

    ok_2col, msg_2col = verify_pooled_step_hlo(
        mesh,
        spec,
        ex,
        w_flat_local,
        pools_local,
        max_pages_per_seq=mk.cache_exec(ex).max_pages_per_seq,
        compile_hlo=True,
        ckpt_cols=2,
    )
    assert ok_2col, f"verify_pooled_step_hlo (ckpt_cols=2) failed: {msg_2col}"

    exported = export_pooled_step(
        mesh,
        spec,
        ex,
        w_flat_local,
        pools_local,
        max_pages_per_seq=mk.cache_exec(ex).max_pages_per_seq,
        platforms=("cpu",),
        ckpt_cols=2,
    )
    assert len(exported.mlir_module_serialized) > 0
    assert exported.input_output_aliases == {2 + len(w_flat_local): 1}


def test_verify_torchtpu_compiled_step_ir() -> None:
    """Verify `verify_torchtpu_compiled_step_ir` on `@main` `jax.buffer_donor`."""
    pool_ty = "tensor<64x2x64x4x128xf8E4M3FN>"
    pool_args = ", ".join(
        f"%arg{i}: {pool_ty} {{jax.buffer_donor = true}}" for i in range(15)
    )
    valid_mlir = (
        f"module @TorchTpuStep {{\n"
        f"  func.func public @main({pool_args}) -> ({', '.join([pool_ty] * 15)}) {{\n"
        f"    %0 = stablehlo.custom_call @tpu_custom_call(%arg0) : "
        f"({pool_ty}) -> {pool_ty}\n"
        f"    return {', '.join(f'%arg{i}' for i in range(15))} : "
        f"{', '.join([pool_ty] * 15)}\n"
        f"  }}\n"
        f"}}\n"
    )
    ok, msg = verify_torchtpu_compiled_step_ir(
        valid_mlir, num_pages=64, page_size=128, kvh=1, hd=256, num_pools=15
    )
    assert ok, f"Expected valid StableHLO MLIR to pass: {msg}"

    # Missing one jax.buffer_donor on @main must fail even if an inlined submodule
    # carries 15 tf.aliasing_output attributes (cannot pass vacuously)
    missing_donor_args = ", ".join(
        f"%arg{i}: {pool_ty}" + (" {jax.buffer_donor = true}" if i < 14 else "")
        for i in range(15)
    )
    inner_alias_args = ", ".join(
        f"%arg{i}: {pool_ty} {{tf.aliasing_output = {i}}}" for i in range(15)
    )
    bad_donor_mlir = (
        valid_mlir.replace(pool_args, missing_donor_args)
        + f"\n  func.func private @inlined_jax({inner_alias_args}) -> () {{ return }}\n"
    )
    ok_bad, msg_bad = verify_torchtpu_compiled_step_ir(
        bad_donor_mlir, num_pages=64, page_size=128, kvh=1, hd=256, num_pools=15
    )
    assert not ok_bad
    assert "expected >= 15 donated pool arguments" in msg_bad

    # Full-pool dynamic_update_slice / as_strided_copy_inverse must fail
    bad_copy_mlir = valid_mlir.replace(
        "return",
        f"%c = stablehlo.dynamic_update_slice %arg0, %arg1, %c0 : "
        f"({pool_ty}, {pool_ty}, tensor<i32>) -> {pool_ty}\n    return",
    )
    ok_copy, msg_copy = verify_torchtpu_compiled_step_ir(
        bad_copy_mlir, num_pages=64, page_size=128, kvh=1, hd=256, num_pools=15
    )
    assert not ok_copy
    assert "unexpected full-pool copy in torch_tpu IR" in msg_copy

    # Valid HLO text vs empty input_output_alias={} vs HLO text with full-pool copy
    alias_entries = ", ".join(f"{{{{{i}}}, {i + 2}, {{}}}}" for i in range(15))
    valid_hlo = (
        f"HloModule TorchTpuStep, input_output_alias={{ {alias_entries} }}\n"
        f"ENTRY %main {{\n"
        f'  %k = () custom-call(), custom_call_target="tpu_custom_call"\n'
        f"  ROOT %t = () tuple()\n"
        f"}}\n"
    )
    ok_hlo, msg_hlo = verify_torchtpu_compiled_step_ir(
        valid_hlo, num_pages=64, page_size=128, kvh=1, hd=256, num_pools=15
    )
    assert ok_hlo, f"Expected valid HLO to pass: {msg_hlo}"

    ok_both, msg_both = verify_torchtpu_compiled_step_ir(
        valid_mlir,
        num_pages=64,
        page_size=128,
        kvh=1,
        hd=256,
        num_pools=15,
        hlo_text=valid_hlo,
    )
    assert ok_both, f"Expected combined MLIR + post-XLA HLO to pass: {msg_both}"

    empty_alias_hlo = valid_hlo.replace(f"{{ {alias_entries} }}", "{}")
    ok_empty_alias, msg_empty_alias = verify_torchtpu_compiled_step_ir(
        empty_alias_hlo, num_pages=64, page_size=128, kvh=1, hd=256, num_pools=15
    )
    assert not ok_empty_alias
    assert "expected >= 15 entries in HloModule input_output_alias" in msg_empty_alias

    bad_hlo = valid_hlo.replace(
        "ROOT %t = () tuple()",
        "  %cp = f8e4m3fn[64,2,64,4,128] copy(%p0)\n  ROOT %t = () tuple()",
    )
    ok_bad_hlo, msg_bad_hlo = verify_torchtpu_compiled_step_ir(
        valid_mlir,
        num_pages=64,
        page_size=128,
        kvh=1,
        hd=256,
        num_pools=15,
        hlo_text=bad_hlo,
    )
    assert not ok_bad_hlo
    assert "unexpected full-pool copy in torch_tpu HLO" in msg_bad_hlo


def test_verify_pooled_step_hlo_compile_on_mock_mesh_is_config_error():
    """`compile_hlo=True` on a non-`jax.Device` mesh is a config error, not layout.

    `MegakernelConfigError` must not be a `ValueError`, so runtime code that
    disables the megakernel on `ValueError` (incompatible KV layout) cannot
    swallow it.
    """
    from vllm_torchtpu.megakernel_runtime.exported_bridge import (
        MegakernelConfigError,
    )

    class _MockDevice:
        id = 0
        process_index = 0
        device_kind = "mock"
        platform = "mock"

    mesh = Mesh(np.array([_MockDevice()], dtype=object), ("tp",))
    assert not issubclass(MegakernelConfigError, ValueError)
    with pytest.raises(MegakernelConfigError, match="compile_hlo=True requires"):
        verify_pooled_step_hlo(
            mesh, None, None, [], [], max_pages_per_seq=1, compile_hlo=True
        )


def test_build_pooled_step_op_without_torch_tpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing `torch_tpu` defers to TPU execution; other import errors propagate."""
    from vllm_torchtpu.megakernel_runtime import exported_bridge

    mesh = _cpu_mesh(1)
    g = _tiny_geom(L=4, att_every=4)
    ex = _tiny_exec(R=1, B=2, T=1, ctx=256, num_pages=96)
    spec = qwen35.qwen35_spec(g)

    def _missing(name: str) -> Any:
        def _raise(*_args: Any, **_kwargs: Any) -> Any:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)

        return _raise

    monkeypatch.setattr(
        exported_bridge, "sharded_jax_callable", _missing("torch_tpu._internal")
    )
    op, _, _, _ = exported_bridge.build_pooled_step_op(
        mesh, spec, ex, op_name="megakernel_runtime::no_torch_tpu_probe"
    )
    assert op is not None

    monkeypatch.setattr(
        exported_bridge, "sharded_jax_callable", _missing("some_other_dep")
    )
    with pytest.raises(ModuleNotFoundError, match="some_other_dep"):
        exported_bridge.build_pooled_step_op(
            mesh, spec, ex, op_name="megakernel_runtime::other_dep_probe"
        )
