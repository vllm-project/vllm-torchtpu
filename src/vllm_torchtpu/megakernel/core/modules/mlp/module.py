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
"""Dense SwiGLU FFN module (`DenseMLPModule`)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp

from ...decls import (
    REQUIRED,
    FrozenDict,
    HFParamDecl,
    ResolvedModuleGeometry,
    apply_decl_overrides,
    normalize_weight_dtypes,
)
from ...errors import MegakernelConfigError
from ...exec_spec import ExecSpec
from ...module import Module
from ...ops import BF16, F32
from ...spec import SmallField, WeightDecl
from ..swiglu import SwiGLU, swiglu_reference

MLP_WEIGHT_NAMES: tuple[str, ...] = ("gate_up", "down")


class DenseMLPModule(Module):
    """Dense SwiGLU FFN, TP-sharded on the intermediate dimension.

    It streams `gate_up` `[H, 2*FF/R]` and `down` `[FF/R, H]` on the static
    dense ring with the MoE shared expert's numerics, so its weights prefetch
    across layers like every other static weight.
    """

    name: str = "mlp"
    reduction = "allreduce"
    input_norm: str | None = "post_norm"
    PARAM_NAMES = frozenset(
        {"gate_proj", "gate_proj_s", "up_proj", "up_proj_s", "down", "down_s"}
    )
    GEOMETRY_FIELDS = {
        "hidden": ("hidden_size", REQUIRED),
        "inter": ("intermediate_size", REQUIRED),
    }
    FIELD_ALIASES = {
        "H": "hidden",
        "hidden_size": "hidden",
        "FF": "inter",
        "intermediate_size": "inter",
    }
    OPTIONS = {"weight_dtypes": None}
    GEOMETRY_METHODS = {"FFR": lambda g, R: int(g.FF) // R}

    def __init__(
        self,
        *,
        geometry: Any = None,
        prefix: str = "mlp",
        weight_dtypes: Mapping[str, str] | None = None,
    ) -> None:
        self.geometry = geometry
        self.prefix = prefix
        self.weight_dtypes = normalize_weight_dtypes(
            weight_dtypes, MLP_WEIGHT_NAMES, "DenseMLPModule", MegakernelConfigError
        )
        opts = {} if weight_dtypes is None else {"weight_dtypes": self.weight_dtypes}
        self.options = FrozenDict(opts)
        w_dt = self.weight_dtypes
        self._swiglu = SwiGLU(
            "gate_up", "down", w_dt["gate_up"] == "fp8", w_dt["down"] == "fp8"
        )

    @classmethod
    def _validate_options(cls, options: Mapping[str, Any]) -> None:
        normalize_weight_dtypes(
            options.get("weight_dtypes"),
            MLP_WEIGHT_NAMES,
            "DenseMLPModule",
            MegakernelConfigError,
        )

    def weights(self, g: Any, ex: ExecSpec) -> list[WeightDecl]:
        return self._swiglu.weights(g.H, g.FF // ex.R)

    def smalls(self, g: Any, ex: ExecSpec) -> list[SmallField]:
        return [SmallField("post_norm", g.H), *self._swiglu.smalls(g.H, g.FF // ex.R)]

    def shared_widths(self, g: Any, ex: ExecSpec) -> dict[str, int]:
        ir = g.FF // ex.R
        return {"acc": max(g.H, 2 * ir), "xb2": ir}

    def plan_params(self, g: Any, ex: ExecSpec, plan: Any) -> bool:
        """Whether the module's tiles refill lazily (all fit the dense ring)."""
        tiles = ex.resolve_tiles(g)
        n_t = plan.nt[self._swiglu.gu] + plan.nt[self._swiglu.down]
        return bool(tiles.lazy and n_t <= tiles.nbd)

    def emit(self, ctx: Any, l: Any, slot: Any) -> Any:  # noqa: E741
        del l
        g, ex = ctx.g, ctx.ex
        with ctx.stream.segment(lazy=ctx.module_params(self.name)):
            return self._swiglu.emit(ctx, slot, g.FF // ex.R, g.H)

    def hf_params(self, prefix: str | None = None, g: Any = None) -> list[HFParamDecl]:
        geom = self._require_geometry(g)
        p = self.prefix if prefix is None else prefix
        rename = geom.rename if isinstance(geom, ResolvedModuleGeometry) else {}
        decls = self._swiglu.hf_params(
            lambda c: rename.get(c, c),
            f"{p}." if p else "",
            int(geom.H),
            int(geom.FF),
        )
        return apply_decl_overrides(decls, geom)

    def reference(
        self,
        params: Any,
        x: Any,
        state: Any = None,
        *,
        g: Any = None,
        ex: ExecSpec,
        layer_idx: int = 0,
        **kw: Any,
    ) -> tuple[jax.Array, Any]:
        """Sum over ranks of each rank's bf16 partial (the bf16 all-reduce)."""
        del kw
        geom = self._require_geometry(g)
        sw = self._swiglu
        xb = x[: ex.B].astype(F32).astype(BF16).astype(F32)

        def _w(name: str, on: bool = True) -> Any:
            return params[name][:, layer_idx] if on else None

        w_gu, w_dn = _w(sw.gu), _w(sw.down)
        s_gu, s_dn = _w(f"{sw.gu}_s", sw.gu_fp8), _w(f"{sw.down}_s", sw.down_fp8)
        parts = [
            swiglu_reference(
                xb,
                w_gu[r],
                w_dn[r],
                int(geom.FF) // ex.R,
                None if s_gu is None else s_gu[r],
                None if s_dn is None else s_dn[r],
            )
            .astype(BF16)
            .astype(F32)
            for r in range(ex.R)
        ]
        return jnp.sum(jnp.stack(parts, axis=0), axis=0), state
