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
"""Declarative `Norm` and `Embedding` blocks (`bind`, `hf_params`, `reference`)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import jax
import jax.numpy as jnp

from ..decls import (
    REQUIRED,
    BlockBinding,
    FrozenDict,
    HFParamDecl,
    ParamTransform,
    ResolvedBlockGeometry,
    VLLMAlias,
    apply_decl_overrides,
    validate_rename_and_transforms,
)
from ..ops import BF16, F32, I32, rmsnorm
from ..spec import Block


def _canonical_norm_name(prefix: str, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if prefix == "input_layernorm" or prefix.endswith(".input_layernorm"):
        return "in_norm"
    if prefix == "post_attention_layernorm" or prefix.endswith(
        ".post_attention_layernorm"
    ):
        return "post_norm"
    if prefix in ("model.norm", "norm") or prefix.endswith(".norm"):
        return "final_norm"
    return prefix.rsplit(".", 1)[-1] if prefix else "norm"


class Norm(Block):
    """RMSNorm block declaration (`bind`, `hf_params`, and pure-JAX `reference`)."""

    name: str = "norm"
    reduction = "none"
    GEOMETRY_FIELDS: ClassVar[dict[str, tuple[str, Any]]] = {
        "hidden": ("hidden_size", REQUIRED),
        "eps": ("rms_norm_eps", 1e-6),
    }
    FIELD_ALIASES: ClassVar[dict[str, str]] = {
        "H": "hidden",
        "hidden_size": "hidden",
        "rms_norm_eps": "eps",
    }
    OPTIONS: ClassVar[dict[str, Any]] = {
        "zero_centered": False,
        "param_name": None,
    }

    def __init__(
        self,
        prefix: str = "",
        *,
        geometry: Any = None,
        zero_centered: bool | None = None,
        eps: float | None = None,
        param_name: str | None = None,
    ) -> None:
        self.prefix = prefix
        self.geometry = geometry
        self.zero_centered = zero_centered
        self.eps = eps
        self.param_name = param_name
        self.options = FrozenDict(
            {
                "zero_centered": False
                if zero_centered is None
                else bool(zero_centered),
                "param_name": param_name,
            }
        )

    @classmethod
    def bind(cls, prefix: str = "", **hf_fields_or_options: Any) -> BlockBinding:
        canon = _canonical_norm_name(prefix, hf_fields_or_options.get("param_name"))
        rename = dict(hf_fields_or_options.get("rename", {}) or {})
        transforms = dict(hf_fields_or_options.get("transforms", {}) or {})
        if set(rename) - {canon}:
            raise ValueError(
                f"Norm.bind got unknown rename key(s) {sorted(set(rename) - {canon})}; "
                f"expected {[canon]}"
            )
        valid_tf = {canon} | set(rename.values())
        if set(transforms) - valid_tf:
            raise ValueError(
                f"Norm.bind got unknown transforms key(s) "
                f"{sorted(set(transforms) - valid_tf)}; expected {sorted(valid_tf)}"
            )
        validate_rename_and_transforms(
            cls, rename, transforms, param_name_override=canon
        )
        return super().bind(prefix, **hf_fields_or_options)

    @classmethod
    def _from_resolved(cls, resolved: ResolvedBlockGeometry) -> Norm:
        return cls(
            prefix=resolved.prefix,
            geometry=resolved,
            zero_centered=resolved.options.get("zero_centered", False),
            eps=resolved.values.get("eps"),
            param_name=resolved.options.get("param_name"),
        )

    def canonical_param_name(self, prefix: str | None = None) -> str:
        p = self.prefix if prefix is None else prefix
        return _canonical_norm_name(p, self.param_name)

    def renamed_param_name(self, prefix: str | None = None, g: Any = None) -> str:
        geom = g if g is not None else self.geometry
        canon = self.canonical_param_name(prefix)
        if isinstance(geom, ResolvedBlockGeometry):
            return geom.rename.get(canon, canon)
        return canon

    def hf_params(self, prefix: str | None = None, g: Any = None) -> list[HFParamDecl]:
        geom = self._require_geometry(g)
        p = self.prefix if prefix is None else prefix
        renamed = self.renamed_param_name(p, geom)
        hf_name = f"{p}.weight" if p else "weight"
        H = int(getattr(geom, "H", getattr(geom, "hidden", 0)))
        decls = [
            HFParamDecl(
                name=renamed,
                hf_name=hf_name,
                shape=(H,),
                dtype=F32,
                transform=ParamTransform(shard_axis=None),
                vllm_aliases=(
                    VLLMAlias(
                        source=hf_name,
                        targets=(renamed,),
                        already_sharded=False,
                        already_fp8=False,
                        flipped=False,
                    ),
                ),
            )
        ]
        return apply_decl_overrides(decls, geom)

    def reference(
        self,
        params: Any,
        x: Any,
        state: Any = None,
        *,
        g: Any = None,
        eps: float | None = None,
        zero_centered: bool | None = None,
        out_dtype: Any = F32,
        **_: Any,
    ) -> tuple[jax.Array, Any]:
        geom = g if g is not None else self.geometry
        if isinstance(params, Mapping):
            key = self.renamed_param_name(self.prefix, geom)
            if key not in params:
                raise KeyError(
                    f"Norm.reference could not find weight {key!r} in params"
                )
            w = params[key]
        else:
            w = params

        if eps is not None:
            eps_val = float(eps)
        elif self.eps is not None:
            eps_val = float(self.eps)
        elif geom is not None and hasattr(geom, "eps"):
            eps_val = float(geom.eps)
        else:
            eps_val = 1e-6

        if zero_centered is not None:
            zc_val = bool(zero_centered)
        elif self.zero_centered is not None:
            zc_val = bool(self.zero_centered)
        elif (
            isinstance(geom, ResolvedBlockGeometry) and "zero_centered" in geom.options
        ):
            zc_val = bool(geom.options["zero_centered"])
        else:
            zc_val = False

        out = rmsnorm(x.astype(F32), w.astype(F32), eps_val, zero_centered=zc_val)
        if out_dtype != F32:
            out = out.astype(out_dtype)
        return out, state


class Embedding(Block):
    """Token embedding block declaration (`bind`, `hf_params`, `reference`)."""

    name: str = "embedding"
    reduction = "none"
    GEOMETRY_FIELDS: ClassVar[dict[str, tuple[str, Any]]] = {
        "vocab": ("vocab_size", REQUIRED),
        "hidden": ("hidden_size", REQUIRED),
    }
    FIELD_ALIASES: ClassVar[dict[str, str]] = {
        "vocab_size": "vocab",
        "H": "hidden",
        "hidden_size": "hidden",
    }
    OPTIONS: ClassVar[dict[str, Any]] = {
        "param_name": "embed_tokens",
        "dtype": BF16,
    }

    def __init__(
        self,
        prefix: str = "model.embed_tokens",
        *,
        geometry: Any = None,
        param_name: str = "embed_tokens",
        dtype: Any = BF16,
    ) -> None:
        self.prefix = prefix
        self.geometry = geometry
        self.param_name = param_name
        self.dtype = dtype
        self.options = FrozenDict({"param_name": param_name, "dtype": dtype})

    @classmethod
    def bind(cls, prefix: str = "", **hf_fields_or_options: Any) -> BlockBinding:
        canon = str(hf_fields_or_options.get("param_name") or "embed_tokens")
        rename = dict(hf_fields_or_options.get("rename", {}) or {})
        transforms = dict(hf_fields_or_options.get("transforms", {}) or {})
        if set(rename) - {canon}:
            raise ValueError(
                f"Embedding.bind got unknown rename key(s) "
                f"{sorted(set(rename) - {canon})}; expected {[canon]}"
            )
        valid_tf = {canon} | set(rename.values())
        if set(transforms) - valid_tf:
            raise ValueError(
                f"Embedding.bind got unknown transforms key(s) "
                f"{sorted(set(transforms) - valid_tf)}; expected {sorted(valid_tf)}"
            )
        validate_rename_and_transforms(
            cls, rename, transforms, param_name_override=canon
        )
        return super().bind(prefix or "model.embed_tokens", **hf_fields_or_options)

    @classmethod
    def _from_resolved(cls, resolved: ResolvedBlockGeometry) -> Embedding:
        return cls(
            prefix=resolved.prefix or "model.embed_tokens",
            geometry=resolved,
            param_name=resolved.options.get("param_name", "embed_tokens"),
            dtype=resolved.options.get("dtype", BF16),
        )

    def renamed_param_name(self, g: Any = None) -> str:
        geom = g if g is not None else self.geometry
        canon = self.param_name or "embed_tokens"
        if isinstance(geom, ResolvedBlockGeometry):
            return geom.rename.get(canon, canon)
        return canon

    def hf_params(self, prefix: str | None = None, g: Any = None) -> list[HFParamDecl]:
        geom = self._require_geometry(g)
        p = (self.prefix if prefix is None else prefix) or "model.embed_tokens"
        canon = self.param_name or "embed_tokens"
        renamed = self.renamed_param_name(geom)
        hf_name = f"{p}.weight" if p else f"{canon}.weight"
        vocab = int(getattr(geom, "vocab_size", getattr(geom, "vocab", 0)))
        H = int(getattr(geom, "H", getattr(geom, "hidden", 0)))
        decls = [
            HFParamDecl(
                name=renamed,
                hf_name=hf_name,
                shape=(vocab, H),
                dtype=self.dtype,
                transform=ParamTransform(shard_axis=0),
                vllm_aliases=(
                    VLLMAlias(
                        source=hf_name,
                        targets=(renamed,),
                        shard_axis=0,
                        already_sharded=True,
                        already_fp8=False,
                        flipped=False,
                    ),
                ),
            )
        ]
        return apply_decl_overrides(decls, geom)

    def reference(
        self,
        params: Any,
        x: Any,
        state: Any = None,
        *,
        g: Any = None,
        out_dtype: Any = None,
        **_: Any,
    ) -> tuple[jax.Array, Any]:
        geom = g if g is not None else self.geometry
        if isinstance(params, Mapping):
            key = self.renamed_param_name(geom)
            if key not in params:
                raise KeyError(
                    f"Embedding.reference could not find weight {key!r} in params"
                )
            w = params[key]
        else:
            w = params
        out = w[jnp.asarray(x, dtype=I32)]
        if out_dtype is not None:
            out = out.astype(out_dtype)
        return out, state


NormBlock = Norm
EmbeddingBlock = Embedding

__all__ = [
    "Embedding",
    "EmbeddingBlock",
    "Norm",
    "NormBlock",
]
