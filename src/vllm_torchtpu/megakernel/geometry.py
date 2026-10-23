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
"""Model-level `FromHF`, `LMHead`, and `ModelGeometry` for hybrid GQA/GDN + MoE."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .decls import (
    REQUIRED,
    FrozenDict,
    _resolve_field_spec,
)
from .errors import MegakernelConfigError
from .exec_spec import ExecSpec
from .modules.gdn import GDNModule
from .pool.config import cache_exec, resolved_max_pages_per_seq
from .spec import ModelSpec

if TYPE_CHECKING:
    from .model_def import ModelDef

_KNOWN_FROM_HF_KEYS = frozenset(
    {
        "hidden",
        "layers",
        "vocab",
        "eps",
        "gdn_l2_eps",
        "rope_theta",
        "rope_partial",
    }
)


@dataclass(frozen=True)
class LMHead:
    """Declarative model-level LM head specification."""

    prefix: str = "lm_head"
    tied_to: str | None = None


class _BoundFromHF:
    """`FromHF` bound to a `ModelDef` class; callable to derive geometry."""

    __slots__ = ("_from_hf", "_model_cls")

    def __init__(self, from_hf: FromHF, model_cls: type[ModelDef]) -> None:
        self._from_hf = from_hf
        self._model_cls = model_cls

    @property
    def fields(self) -> Mapping[str, Any]:
        return self._from_hf.fields

    def resolve(self, hf_config: Any) -> FrozenDict:
        return self._from_hf.resolve(hf_config)

    def __call__(
        self,
        hf_config: Any = None,
        *,
        moe_tp: bool | None = None,
        e_slots: int | None = None,
    ) -> ModelGeometry:
        return self._model_cls.derive_geometry(
            hf_config, moe_tp=moe_tp, e_slots=e_slots
        )

    def __eq__(self, other: object) -> Any:
        if isinstance(other, _BoundFromHF):
            return self._from_hf == other._from_hf
        if isinstance(other, FromHF):
            return self._from_hf == other
        return NotImplemented

    def __repr__(self) -> str:
        return repr(self._from_hf)


@dataclass(frozen=True)
class FromHF:
    """Strict model-level geometry declaration from HuggingFace config fields."""

    fields: Mapping[str, Any] = field(default_factory=FrozenDict)

    def __init__(self, **fields: Any) -> None:
        if not fields:
            raise MegakernelConfigError("FromHF requires at least one field mapping")
        unknown = sorted(set(fields) - _KNOWN_FROM_HF_KEYS)
        if unknown:
            raise MegakernelConfigError(
                f"FromHF got unknown key(s) {unknown}; "
                f"supported: {sorted(_KNOWN_FROM_HF_KEYS)}"
            )
        for k, v in fields.items():
            if not (
                isinstance(v, (str, int, float, bool))
                or (isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], str))
            ):
                raise MegakernelConfigError(
                    f"FromHF field {k!r} must be an HF field name, (field, default), "
                    f"or scalar literal, got {v!r}"
                )
        object.__setattr__(self, "fields", FrozenDict(fields))

    def __get__(self, obj: Any, owner: type[Any] | None = None) -> Any:
        if (
            owner is not None
            and getattr(owner, "_IS_MODEL_DEF", False)
            and owner.__name__ != "ModelDef"
        ):
            return _BoundFromHF(self, owner)
        return self

    def resolve(self, hf_config: Any) -> FrozenDict:
        """Resolve all declared fields strictly from `hf_config`."""
        return FrozenDict(
            {
                name: _resolve_field_spec(
                    hf_config, name, spec_val, REQUIRED, (), explicit=True
                )
                for name, spec_val in self.fields.items()
            }
        )


_GEOM_FIELD_NAMES = (
    "L",
    "att_every",
    "H",
    "NH",
    "NKV",
    "HD",
    "ROT",
    "KH",
    "VH",
    "KD",
    "VD",
    "E",
    "TOPK",
    "I",
    "IS",
    "moe_tp",
    "e_slots",
    "eps",
    "gdn_l2_eps",
    "theta",
    "zero_centered_norm",
)


@dataclass(frozen=True, eq=False)
class ModelGeometry:
    """Resolved hybrid attention / linear-attention (`GQA` + `GDN`) + `MoE` geometry."""

    L: int
    att_every: int
    H: int
    NH: int
    NKV: int
    HD: int
    ROT: int
    KH: int
    VH: int
    KD: int
    VD: int
    E: int
    TOPK: int
    I: int  # noqa: E741
    IS: int
    moe_tp: bool = True
    e_slots: int = 0
    eps: float = 1e-6
    gdn_l2_eps: float = 1e-6
    theta: float = 10_000.0
    zero_centered_norm: bool = False

    def _tuple_key(self) -> tuple[Any, ...]:
        return tuple(getattr(self, f) for f in _GEOM_FIELD_NAMES)

    def __eq__(self, other: object) -> Any:
        if isinstance(other, ModelGeometry):
            return self._tuple_key() == other._tuple_key()
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._tuple_key())

    def nh(self, R: int) -> int:
        return self.NH // R

    def G(self, R: int) -> int:
        return R // self.NKV

    def kh(self, R: int) -> int:
        return GDNModule.lin_in_layout(self, R).kh

    def vh(self, R: int) -> int:
        return GDNModule.lin_in_layout(self, R).vh

    def QKV(self, R: int) -> int:
        return GDNModule.lin_in_layout(self, R).QKV

    def Z(self, R: int) -> int:
        return GDNModule.lin_in_layout(self, R).Z

    def BA(self, R: int) -> int:
        return GDNModule.lin_in_layout(self, R).BA

    def NLI(self, R: int) -> int:
        return GDNModule.lin_in_layout(self, R).NLI

    def NAI(self, R: int) -> int:
        return (2 * self.nh(R) + 2) * self.HD

    def ER(self, R: int) -> int:
        return self.E // R

    def ISR(self, R: int) -> int:
        return self.IS // R

    def IR(self, R: int) -> int:
        return self.I // R if self.moe_tp else self.I

    def EW(self, R: int) -> int:
        return (
            self.e_slots
            if self.e_slots > 0
            else (self.E if self.moe_tp else self.ER(R))
        )

    @property
    def NGDN(self) -> int:
        return self.L - self.L // self.att_every

    @property
    def NATT(self) -> int:
        return self.L // self.att_every

    def check(self, ex: ExecSpec) -> None:
        R, B, T, M = ex.R, ex.B, ex.T, ex.M
        pc = min(1024, self.H)
        min_slots = self.E if self.moe_tp else self.ER(R)
        for cond, msg in (
            (
                self.L % self.att_every == 0,
                f"L={self.L} must be divisible by att_every={self.att_every}",
            ),
            (
                self.NH % R == 0 and R % self.NKV == 0,
                f"NH={self.NH} and NKV={self.NKV} incompatible with R={R}",
            ),
            (
                # Routing tables are sized from B and MR; the hard walls are the
                # 128-lane rows-on-lanes top-k and hits table (TOPK <= 128) and
                # the paged scatter's `page_size >= MR` (asserted there).
                self.E % R == 0 and self.TOPK <= 128 and 1 <= B <= M <= 128,
                f"Invalid E={self.E}, R={R}, B={B}, M={M}, TOPK={self.TOPK}",
            ),
            (
                self.E % 128 == 0
                and self.E <= 1024
                and self.H % pc == 0
                and self.nh(R) <= 8,
                f"Invalid E={self.E}, H={self.H}, nh={self.nh(R)}",
            ),
            (
                self.ROT % 2 == 0 and (R & (R - 1) == 0),
                f"ROT={self.ROT} must be even and R={R} a power of 2",
            ),
            (T >= 1 and B % T == 0, f"Invalid B={B}, T={T}"),
            (
                self.e_slots == 0 or self.e_slots >= min_slots,
                f"e_slots={self.e_slots} must be 0 or >= {min_slots}",
            ),
        ):
            if not cond:
                raise MegakernelConfigError(msg)
        cx = cache_exec(ex)
        if cx.paged_kv and not (
            cx.page_size == 128
            and ex.ctx % cx.page_size == 0
            and cx.num_pages >= 1
            and resolved_max_pages_per_seq(ex) >= ex.ctx // cx.page_size
        ):
            raise MegakernelConfigError(
                f"Invalid paged_kv config: page_size={cx.page_size}, ctx={ex.ctx}, "
                f"num_pages={cx.num_pages}, max_pages={resolved_max_pages_per_seq(ex)}"
            )
        for n in (self.H, self.HD, self.IR(R), self.ISR(R)):
            if n % 128 != 0:
                raise MegakernelConfigError(f"Dimension {n} must be a multiple of 128")
        if self.NGDN > 0:
            if not (self.KH % R == 0 and self.VH % R == 0 and self.VH % self.KH == 0):
                raise MegakernelConfigError(
                    f"Invalid GDN heads KH={self.KH}, VH={self.VH}, R={R}"
                )
            lay = GDNModule.lin_in_layout(self, R)
            if lay.vh > 64:
                raise MegakernelConfigError(f"GDN vh={lay.vh} exceeds 64")
            for n in (lay.KD, lay.VD, lay.QKV, lay.Z):
                if n % 128 != 0:
                    raise MegakernelConfigError(
                        f"GDN dimension {n} must be a multiple of 128"
                    )

    def shrink(
        self, *, periods: int = 2, R: int = 8, ex: ExecSpec | None = None
    ) -> ModelGeometry:
        """Return a tiny geometry (`periods` cycles, legal heads/experts for `R`)."""
        if ex is not None:
            R = ex.R
        if not isinstance(periods, int) or isinstance(periods, bool) or periods < 1:
            raise MegakernelConfigError(f"shrink periods must be >= 1, got {periods!r}")
        if (
            not isinstance(R, int)
            or isinstance(R, bool)
            or R < 1
            or (R & (R - 1)) != 0
            or R > 8
        ):
            raise MegakernelConfigError(
                f"shrink R must be a power of 2 in 1..8, got {R!r}"
            )
        nkv = min(self.NKV, R, 2)
        while R % nkv != 0 and nkv > 1:
            nkv -= 1
        nh = min(self.NH, max(R, nkv, 8))
        if nh % R != 0 or nh % nkv != 0:
            nh = max(R, nkv)
        if nh // R > 8:
            nh = 8 * R
        if self.NGDN > 0:
            kh = min(self.KH, max(R, 8))
            if kh % R != 0:
                kh = R
            vh = kh * min(max(1, self.VH // max(1, self.KH)), 2)
        else:
            kh = vh = 0
        e_tiny = min(self.E, 128)
        if self.e_slots == 0:
            e_slots = 0
        elif self.moe_tp:
            e_slots = e_tiny + max(0, self.e_slots - self.E)
        else:
            base_er = self.E // R if self.e_slots >= self.E // R else self.E // 8
            e_slots = e_tiny // R + max(0, self.e_slots - base_er)
        i_min = 128 * R if self.moe_tp else 128
        shrunk = dataclasses.replace(
            self,
            L=periods * self.att_every,
            H=min(self.H, 1024),
            NH=nh,
            NKV=nkv,
            KH=kh,
            VH=vh,
            E=e_tiny,
            TOPK=min(self.TOPK, 4),
            I=max(i_min, min(self.I, 1024)),
            IS=max(128 * R, min(self.IS, 1024)),
            e_slots=e_slots,
        )
        mdef = getattr(self, "_model_def", None)
        if mdef is not None:
            object.__setattr__(shrunk, "_model_def", mdef)
        return shrunk

    def to_model_spec(self) -> ModelSpec:
        """Convert this geometry into its declarative `ModelSpec`."""
        mdef = getattr(self, "_model_def", None)
        if mdef is not None:
            return mdef.to_model_spec(self)
        raise MegakernelConfigError(
            "ModelGeometry is not bound to a ModelDef; call ModelDef.to_model_spec(g)"
        )


__all__ = [
    "FromHF",
    "LMHead",
    "ModelGeometry",
    "_BoundFromHF",
]
