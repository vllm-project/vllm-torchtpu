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
"""Unit and integration tests for module capability introspection, `StateDecl`
output roles and comparison modes, `ModelDef`-free verification input builders,
`verify_module`, and `verify_model` custom-module / oracle paths.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import pytest
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh

from vllm_torchtpu import megakernel as mk
from vllm_torchtpu.megakernel.ops import BF16, F32, rmsnorm, rnd
from vllm_torchtpu.models.qwen3_5.megakernel import model as qwen35


@pytest.fixture(autouse=True)
def _use_cpu_default_device():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


class _CustomStatefulModule(mk.Module):
    """Minimal custom `Module` implementing only kernel execution hooks
    (`weights`, `smalls`, `states`, `outputs`, `scratch`, `emit`) without
    `reference` or `hf_params`.

    Exercises all four `OutputRole` values (`"state"`, `"tap"`, `"token"`,
    `"residual"`) plus `compare="none"`.
    """

    name = "custom_stateful"
    reduction = "allreduce"
    input_norm = "in_norm"
    GEOMETRY_FIELDS = {"scale_factor": ("custom_scale", 1.0)}
    OPTIONS = {"include_extra_roles": True}

    def weights(self, g: Any, ex: mk.ExecSpec) -> list[mk.WeightDecl]:
        return [
            mk.WeightDecl(
                name="custom_w",
                K=1,
                N=g.H,
                dtype=F32,
                stream_ring="hbm",
            )
        ]

    def smalls(self, g: Any, ex: mk.ExecSpec) -> list[mk.SmallField]:
        return [
            mk.SmallField("in_norm", g.H),
            mk.SmallField("custom_gain", g.H),
        ]

    def states(self, g: Any, ex: mk.ExecSpec) -> list[mk.StateDecl]:
        return [
            mk.StateDecl(
                name="custom_st",
                shape=(ex.MB, g.H),
                dtype=F32,
                access="rw",
                per_layer=True,
                memory_space="vmem",
                role="state",
            )
        ]

    def outputs(self, g: Any, ex: mk.ExecSpec) -> list[mk.StateDecl]:
        decls = [
            mk.StateDecl(
                name="custom_st",
                shape=(ex.MB, g.H),
                dtype=F32,
                access="w",
                per_layer=True,
                memory_space="vmem",
                role="state",
            )
        ]
        if self.options.get("include_extra_roles", True):
            decls.extend(
                [
                    mk.StateDecl(
                        name="custom_tap",
                        shape=(ex.MB, g.H),
                        dtype=F32,
                        access="w",
                        per_layer=True,
                        memory_space="vmem",
                        role="tap",
                        compare="close",
                        atol=1e-4,
                        rtol=1e-4,
                    ),
                    mk.StateDecl(
                        name="custom_tok",
                        shape=(ex.MB, 8),
                        dtype=jnp.int32,
                        access="w",
                        per_layer=True,
                        memory_space="vmem",
                        role="token",
                        compare="exact",
                    ),
                    mk.StateDecl(
                        name="custom_res",
                        shape=(ex.MB, g.H),
                        dtype=F32,
                        access="w",
                        per_layer=True,
                        memory_space="vmem",
                        role="residual",
                        compare="cos",
                        min_cos=0.9999,
                    ),
                    mk.StateDecl(
                        name="custom_unmonitored",
                        shape=(ex.MB, 8),
                        dtype=F32,
                        access="w",
                        per_layer=True,
                        memory_space="vmem",
                        role="tap",
                        compare="none",
                    ),
                ]
            )
        return decls

    def scratch(self, g: Any, ex: mk.ExecSpec, plan: Any) -> dict[str, Any]:
        return {
            "cw_buf": pltpu.VMEM((1, g.H), F32),
            "cw_sem": pltpu.SemaphoreType.DMA(()),
        }

    def emit(self, ctx: Any, l: Any, slot: Any) -> Any:  # noqa: E741
        ord_l = ctx.stream.sp.module_ord(l, self.name)
        cp = pltpu.make_async_copy(
            ctx.W["custom_w"].at[ord_l], ctx.scr["cw_buf"], ctx.scr["cw_sem"]
        )
        cp.start()
        cp.wait()

        x = ctx.scr["xb"][...].astype(F32)
        gain = ctx.sm(slot, "custom_gain")  # [1, H]
        w_row = ctx.scr["cw_buf"][...]  # [1, H]
        prev_st = ctx.ST["custom_st"][ord_l]  # [MB, H]

        new_st = jnp.where(ctx.live, 0.5 * prev_st + x * gain + w_row, prev_st)
        ctx.O["custom_st"][ord_l] = new_st

        if self.options.get("include_extra_roles", True):
            tap_val = jnp.where(ctx.live, x * w_row, 0.0)
            ctx.O["custom_tap"][ord_l] = tap_val
            tok_val = jnp.where(
                ctx.live[:, :1],
                (jnp.abs(x[:, :8]) * 10.0).astype(jnp.int32) + ord_l,
                jnp.int32(0),
            )
            ctx.O["custom_tok"][ord_l] = tok_val
            ctx.O["custom_unmonitored"][ord_l] = tap_val[:, :8]
            ctx.O["custom_res"][ord_l] = jnp.where(ctx.live, x + new_st, 0.0)

        return jnp.where(ctx.live, new_st / float(ctx.ex.R), 0.0)


def _custom_module_oracle(
    params: dict[str, Any],
    x: jax.Array,
    state: dict[str, Any],
    *,
    g: Any,
    ex: mk.ExecSpec,
    layer_idx: int = 0,
    pos: jax.Array | None = None,
    include_extra_roles: bool = True,
) -> tuple[jax.Array, dict[str, Any]]:
    del pos
    R, MB, B, H = ex.R, ex.MB, ex.B, int(g.H)
    row_m = jnp.arange(MB, dtype=jnp.int32)[:, None]
    live = row_m < B

    x_mb = jnp.zeros((MB, H), dtype=F32).at[:B].set(x[:B].astype(BF16).astype(F32))
    gain = params["custom_gain"][:, layer_idx].astype(F32)[:, None, :]  # [R, 1, H]
    w_row = params["custom_w"][:, layer_idx].astype(F32)  # [R, 1, H]
    prev_st = state["custom_st"][:, layer_idx].astype(F32)  # [R, MB, H]

    new_st = jnp.where(
        live[None, :, :], 0.5 * prev_st + x_mb[None, :, :] * gain + w_row, prev_st
    )
    st_updates: dict[str, Any] = dict(state)
    st_updates["custom_st"] = state["custom_st"].at[:, layer_idx].set(new_st)

    if include_extra_roles:
        tap_val = jnp.where(live[None, :, :], x_mb[None, :, :] * w_row, 0.0)
        tok_val = jnp.broadcast_to(
            jnp.where(
                live[None, :, :1],
                (jnp.abs(x_mb[None, :, :8]) * 10.0).astype(jnp.int32)
                + jnp.int32(layer_idx),
                jnp.int32(0),
            ),
            (R, MB, 8),
        )
        res_val = jnp.where(live[None, :, :], x_mb[None, :, :] + new_st, 0.0)
        st_updates["custom_tap"] = tap_val[:, None, :, :]
        st_updates["custom_tok"] = tok_val[:, None, :, :]
        st_updates["custom_res"] = res_val[:, None, :, :]
        st_updates["custom_unmonitored"] = tap_val[:, None, :, :8]

    delta_mb = jnp.sum(new_st / float(R), axis=0)  # [MB, H]
    return delta_mb[:B], st_updates


class _CustomOnlyReferenceModule(_CustomStatefulModule):
    """Custom module that overrides `reference` without overriding `hf_params`."""

    name = "custom_with_ref"

    def reference(
        self,
        params: Any,
        x: Any,
        state: Any = None,
        *,
        g: Any = None,
        ex: mk.ExecSpec,
        layer_idx: int = 0,
        **kw: Any,
    ) -> Any:
        return _custom_module_oracle(
            params,
            x,
            state or {},
            g=g,
            ex=ex,
            layer_idx=layer_idx,
            pos=kw.get("pos"),
            include_extra_roles=bool(self.options.get("include_extra_roles", True)),
        )


class _CustomNoRefModelDef(mk.ModelDef):
    """`ModelDef` whose `to_model_spec` uses `_CustomStatefulModule` (no
    `hf_params`, no `reference`).
    """

    name = "custom_no_ref_test"
    supported_tp_sizes = (2,)

    @classmethod
    def shrink(
        cls,
        hf_config_or_geom: Any = None,
        *,
        periods: int = 2,
        R: int = 2,
        ex: mk.ExecSpec | None = None,
    ) -> mk.ModelGeometry:
        del hf_config_or_geom, periods, ex
        g = dataclasses.replace(
            qwen35.Qwen35.shrink(periods=1, R=R),
            L=2,
            att_every=1,
        )
        object.__setattr__(g, "_model_def", cls)
        return g

    @classmethod
    def to_model_spec(
        cls,
        hf_config_or_geom: Any = None,
        *,
        gdn_module: mk.Module | None = None,
        moe_tp: bool | None = None,
        e_slots: int | None = None,
        R: int | None = 2,
    ) -> mk.ModelSpec:
        del gdn_module, moe_tp, e_slots
        g = (
            hf_config_or_geom
            if isinstance(hf_config_or_geom, mk.ModelGeometry)
            else cls.shrink(R=R or 2)
        )
        mod = _CustomStatefulModule(include_extra_roles=False)
        return mk.ModelSpec(
            name=cls.name,
            config=g,
            layer_type=mk.LayerType(name="custom_layer", slots=(mod,)),
        )


class _CustomWithRefNoHFModelDef(_CustomNoRefModelDef):
    """`ModelDef` whose `to_model_spec` uses `_CustomOnlyReferenceModule`
    (`has_reference=True`, `has_hf_params=False`).
    """

    name = "custom_ref_no_hf_test"

    @classmethod
    def to_model_spec(
        cls,
        hf_config_or_geom: Any = None,
        *,
        gdn_module: mk.Module | None = None,
        moe_tp: bool | None = None,
        e_slots: int | None = None,
        R: int | None = 2,
    ) -> mk.ModelSpec:
        del gdn_module, moe_tp, e_slots
        g = (
            hf_config_or_geom
            if isinstance(hf_config_or_geom, mk.ModelGeometry)
            else cls.shrink(R=R or 2)
        )
        mod = _CustomOnlyReferenceModule(include_extra_roles=False)
        return mk.ModelSpec(
            name=cls.name,
            config=g,
            layer_type=mk.LayerType(name="custom_ref_layer", slots=(mod,)),
        )


def _tiny_geom(L: int = 1) -> mk.ModelGeometry:
    return dataclasses.replace(qwen35.Qwen35.shrink(periods=1, R=2), L=L, att_every=1)


# ---------------------------------------------------------------------------
# 1. Capability introspection (`has_reference`, `has_hf_params`, MoE `**kw`)
# ---------------------------------------------------------------------------


def test_builtin_and_custom_module_capability_properties():
    gqa = mk.GQAModule()
    gdn = mk.GDNModule()
    moe = mk.MoEModule()
    for mod in (gqa, gdn, moe):
        assert mod.has_reference is True
        assert mod.has_hf_params is True

    custom = _CustomStatefulModule()
    assert custom.has_reference is False
    assert custom.has_hf_params is False
    with pytest.raises(NotImplementedError):
        custom.reference(None, None, None)

    custom_ref = _CustomOnlyReferenceModule()
    assert custom_ref.has_reference is True
    assert custom_ref.has_hf_params is False


def test_moe_reference_accepts_extra_keyword_arguments():
    g = _tiny_geom(L=1)
    ex = mk.ExecSpec(
        R=2,
        B=2,
        ctx=64,
        interpret=True,
        tiles=mk.Tiles(),
        cache=mk.DirectHBM(),
    )
    _, hid, w_log, _ = mk.random_logical_inputs(
        jax.random.key(0), g, ex, model_def=qwen35.Qwen35
    )
    moe = mk.MoEModule(geometry=g)
    out, updates = moe.reference(
        w_log,
        hid,
        {},
        g=g,
        ex=ex,
        layer_idx=0,
        pos=jnp.zeros((ex.B,), dtype=jnp.int32),
        unused_future_kwarg="ignored",
    )
    assert out.shape == (ex.B, g.H)
    assert updates == {}


# ---------------------------------------------------------------------------
# 2. `StateDecl` output roles and comparison mode validation
# ---------------------------------------------------------------------------


def test_state_decl_output_roles_and_compare_validation():
    s_default = mk.StateDecl("kc", (4, 8), F32)
    assert s_default.role == "state"
    assert s_default.compare == "close"
    assert s_default.atol is None
    assert s_default.rtol is None
    assert s_default.min_cos is None

    s_tok = mk.StateDecl("router_ids", (4, 2), jnp.int32, role="token")
    assert s_tok.role == "token"
    assert s_tok.compare == "exact"

    s_tok_none = mk.StateDecl(
        "router_ids_unmonitored", (4, 2), jnp.int32, role="token", compare="none"
    )
    assert s_tok_none.compare == "none"

    with pytest.raises(ValueError, match="role='token' requires compare"):
        mk.StateDecl("bad_tok", (4, 2), jnp.int32, role="token", compare="close")

    with pytest.raises(ValueError, match="invalid role"):
        mk.StateDecl("bad_role", (4, 2), F32, role="invalid_role")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="invalid compare"):
        mk.StateDecl("bad_cmp", (4, 2), F32, compare="invalid_cmp")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 3. `ModelDef`-free setup (`random_spec_inputs`, `pack_spec_inputs`)
# ---------------------------------------------------------------------------


def test_random_spec_inputs_and_pack_spec_inputs_without_model_def():
    g = _tiny_geom(L=2)
    ex = mk.ExecSpec(
        R=2,
        B=4,
        ctx=64,
        interpret=True,
        tiles=mk.Tiles(),
        cache=mk.DirectHBM(),
    )
    mod = _CustomStatefulModule(include_extra_roles=True)
    spec = mk.ModelSpec(
        name="custom",
        config=g,
        layer_type=mk.LayerType(name="custom_layer", slots=(mod,)),
    )
    devices = jax.devices("cpu")[: ex.R]
    mesh = Mesh(devices, ("tp",))

    pos, hid, w_log, st_log = mk.random_spec_inputs(jax.random.key(7), spec, ex)
    assert "custom_w" in w_log
    assert "custom_gain" in w_log
    assert "in_norm" in w_log
    assert "final_norm" in w_log
    assert w_log["custom_w"].shape == (ex.R, g.L, 1, g.H)
    assert w_log["custom_gain"].shape == (ex.R, g.L, g.H)

    # Persistent `role="state"` is in `st_log`; step-local `tap`/`token`/`residual`
    # are kernel outputs only, not initial input states.
    assert "custom_st" in st_log
    assert st_log["custom_st"].shape == (ex.R, g.L, ex.MB, g.H)
    for step_local_key in (
        "custom_tap",
        "custom_tok",
        "custom_res",
        "custom_unmonitored",
    ):
        assert step_local_key not in st_log

    pos_m, hid_m, w_m, st_m = mk.pack_spec_inputs(
        pos, hid, w_log, st_log, spec, ex, mesh=mesh
    )
    assert pos_m.shape == (ex.B,)
    assert hid_m.shape == (ex.R * ex.M, g.H)
    assert "custom_w" in w_m
    assert "smalls" in w_m
    assert "custom_st" in st_m


# ---------------------------------------------------------------------------
# 4. `verify_module` single-module harness (built-in + custom with oracle)
# ---------------------------------------------------------------------------


def test_verify_module_builtin_moe_passes():
    g = _tiny_geom(L=1)
    res = mk.verify_module(mk.MoEModule(geometry=g), g, R=2, B_per_T=2, T_values=(1,))
    assert res.passed is True, res
    by_name = {c.name: c for c in res.checks}
    assert by_name["setup"].status == "passed"
    assert by_name["derived_artifacts"].status == "passed"
    assert by_name["kernel_vs_reference"].status == "passed"


def test_verify_module_builtin_gqa_and_gdn_pass():
    g = _tiny_geom(L=1)
    res_gqa = mk.verify_module(
        mk.GQAModule(geometry=g), g, R=2, B_per_T=2, T_values=(1, 2)
    )
    assert res_gqa.passed is True, res_gqa
    assert {c.name: c.status for c in res_gqa.checks}["kernel_vs_reference"] == "passed"

    res_gdn = mk.verify_module(
        mk.GDNModule(geometry=g), g, R=2, B_per_T=2, T_values=(1, 2)
    )
    assert res_gdn.passed is True, res_gdn
    assert {c.name: c.status for c in res_gdn.checks}["kernel_vs_reference"] == "passed"


def test_verify_module_custom_fails_closed_without_oracle():
    g = _tiny_geom(L=1)
    mod = _CustomStatefulModule(include_extra_roles=False)
    res = mk.verify_module(mod, g, R=2, B_per_T=2, T_values=(1,))
    assert res.passed is False
    by_name = {c.name: c for c in res.checks}
    assert by_name["kernel_vs_reference"].status == "failed"
    assert "implements no reference()" in by_name["kernel_vs_reference"].reason


def test_verify_module_custom_passes_with_module_oracle_and_all_roles():
    g = _tiny_geom(L=1)
    mod = _CustomStatefulModule(include_extra_roles=True)

    def oracle(params, x, state, *, g, ex, layer_idx=0, pos=None):
        delta, updates = _custom_module_oracle(
            params,
            x,
            state,
            g=g,
            ex=ex,
            layer_idx=layer_idx,
            pos=pos,
            include_extra_roles=True,
        )
        # Perturb the `compare="none"` output arbitrarily; verification must ignore it.
        updates["custom_unmonitored"] = updates["custom_unmonitored"] + 999.0
        return delta, updates

    res = mk.verify_module(
        mod, g, module_oracle=oracle, R=2, B_per_T=2, T_values=(1, 2)
    )
    assert res.passed is True, res
    by_name = {c.name: c for c in res.checks}
    assert by_name["kernel_vs_reference"].status == "skipped"
    assert by_name["kernel_vs_oracle"].status == "passed"


@pytest.mark.parametrize(
    "corrupt_key,corrupt_val",
    [
        ("custom_st", 5.0),
        ("custom_tap", 5.0),
        ("custom_tok", 1),
        ("custom_res", -5.0),
    ],
)
def test_verify_module_detects_corrupted_output_for_each_role(
    corrupt_key: str, corrupt_val: Any
):
    g = _tiny_geom(L=1)
    mod = _CustomStatefulModule(include_extra_roles=True)

    def bad_oracle(params, x, state, *, g, ex, layer_idx=0, pos=None):
        delta, updates = _custom_module_oracle(
            params,
            x,
            state,
            g=g,
            ex=ex,
            layer_idx=layer_idx,
            pos=pos,
            include_extra_roles=True,
        )
        if corrupt_key == "custom_res":
            # Flip direction so cosine similarity drops below min_cos=0.9999.
            updates["custom_res"] = -updates["custom_res"]
        else:
            updates[corrupt_key] = updates[corrupt_key] + corrupt_val
        return delta, updates

    res = mk.verify_module(
        mod, g, module_oracle=bad_oracle, R=2, B_per_T=2, T_values=(1,)
    )
    assert res.passed is False
    by_name = {c.name: c for c in res.checks}
    assert by_name["kernel_vs_oracle"].status == "failed"
    assert corrupt_key in by_name["kernel_vs_oracle"].reason


# ---------------------------------------------------------------------------
# 5. `verify_model` with custom modules (`has_hf_params == False`, `oracle`)
# ---------------------------------------------------------------------------


def test_verify_model_custom_module_skips_loader_and_hf_and_uses_oracle():
    # 1. Without `oracle`, `loader_parity` and `reference_vs_hf` are `"skipped"`,
    #    and `kernel_vs_reference` fails closed because `_CustomStatefulModule`
    #    does not implement `Module.reference`.
    res_no_oracle = mk.verify_model(
        _CustomNoRefModelDef,
        hf=True,
        approx_profiles=False,
        R=2,
        B_per_T=2,
        T_values=(1,),
        seed=11,
    )
    assert res_no_oracle.passed is False
    by_name = {c.name: c for c in res_no_oracle.checks}
    assert by_name["derived_artifacts"].status == "passed"
    assert by_name["loader_parity"].status == "skipped"
    assert by_name["kernel_vs_reference"].status == "failed"
    assert by_name["reference_vs_hf"].status == "skipped"

    # 2. With `oracle=...`, `kernel_vs_oracle` runs against `inputs.w_logical`
    #    (while `weights == {}`) and passes.
    def full_step_oracle(weights, inputs):
        assert weights == {}
        assert inputs.w_logical, "expected non-empty w_logical in OracleInputs"
        g_arg = inputs.g
        ex_arg = inputs.ex
        w_l = inputs.w_logical
        B = ex_arg.B
        res = inputs.hidden[:B].astype(F32)
        cur_state = dict(inputs.states)
        for l_idx in range(g_arg.L):
            norm_w = w_l["in_norm"][l_idx]
            x_norm = rmsnorm(
                res,
                norm_w,
                eps=float(g_arg.eps),
                zero_centered=bool(g_arg.zero_centered_norm),
            ).astype(BF16)
            delta, cur_state = _custom_module_oracle(
                w_l,
                x_norm,
                cur_state,
                g=g_arg,
                ex=ex_arg,
                layer_idx=l_idx,
                pos=inputs.pos,
                include_extra_roles=False,
            )
            res = rnd(res + delta)
        out_hid = rmsnorm(
            res,
            w_l["final_norm"],
            eps=float(g_arg.eps),
            zero_centered=bool(g_arg.zero_centered_norm),
        ).astype(BF16)
        return out_hid, cur_state

    res_with_oracle = mk.verify_model(
        _CustomNoRefModelDef,
        hf=True,
        oracle=full_step_oracle,
        approx_profiles=False,
        R=2,
        B_per_T=2,
        T_values=(1, 2),
        seed=11,
    )
    assert res_with_oracle.passed is True, res_with_oracle
    by_name_ok = {c.name: c for c in res_with_oracle.checks}
    assert by_name_ok["derived_artifacts"].status == "passed"
    assert by_name_ok["loader_parity"].status == "skipped"
    assert by_name_ok["kernel_vs_reference"].status == "skipped"
    assert by_name_ok["kernel_vs_oracle"].status == "passed"
    assert by_name_ok["reference_vs_hf"].status == "skipped"


def test_verify_model_custom_module_with_reference_and_no_hf_params():
    res = mk.verify_model(
        _CustomWithRefNoHFModelDef,
        hf=True,
        approx_profiles=False,
        R=2,
        B_per_T=2,
        T_values=(1, 2),
        seed=13,
    )
    assert res.passed is True, res
    by_name = {c.name: c for c in res.checks}
    assert by_name["derived_artifacts"].status == "passed"
    assert by_name["loader_parity"].status == "skipped"
    assert by_name["kernel_vs_reference"].status == "passed"
    assert by_name["reference_vs_hf"].status == "skipped"
