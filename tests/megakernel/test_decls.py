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
"""Unit tests for `decls` and `Module` binding/resolution primitives."""

from __future__ import annotations

import copy
import pickle
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import pytest

from vllm_torchtpu.megakernel.decls import (
    REQUIRED,
    FrozenDict,
    HFParamDecl,
    KVCacheDecl,
    ModuleBinding,
    ParamTransform,
    RecurrentCacheDecl,
    ResolvedModuleGeometry,
    VLLMAlias,
    apply_decl_overrides,
    combined_hf_param_shapes,
)
from vllm_torchtpu.megakernel.ops import BF16, F32, FP8
from vllm_torchtpu.megakernel.spec import Module


class _ToyAttn(Module):
    """Minimal self-contained module for testing `bind` and `resolve`."""

    name = "toy_attn"
    GEOMETRY_FIELDS = {
        "H": ("hidden_size", REQUIRED),
        "NH": ("num_attention_heads", REQUIRED),
        "NKV": ("num_key_value_heads", REQUIRED),
        "HD": ("head_dim", 64),
        "theta": ("rope_theta", REQUIRED),
        "rot_factor": ("partial_rotary_factor", 1.0),
    }
    FIELD_ALIASES = {
        "hidden": "H",
        "heads": "NH",
        "kv_heads": "NKV",
    }
    GEOMETRY_METHODS = {
        "q_width": lambda g: g.NH * g.HD,
        "rot_dim": lambda g: int(g.HD * g.rot_factor),
    }
    OPTIONS = {
        "qk_norm": True,
        "param_name": None,
    }
    PARAM_NAMES = frozenset({"q_proj", "k_proj", "o_proj"})

    def __init__(
        self,
        prefix: str = "self_attn",
        *,
        geometry: Any = None,
        qk_norm: bool = True,
        param_name: str | None = None,
    ) -> None:
        opts = {"qk_norm": qk_norm, "param_name": param_name}
        self._validate_options(opts)
        self.prefix = prefix
        self.geometry = geometry
        self.options = FrozenDict(opts)

    @classmethod
    def _validate_options(cls, options: Mapping[str, Any]) -> None:
        if not options.get("qk_norm", True):
            raise ValueError("_ToyAttn requires qk_norm=True")

    def hf_params(self, prefix: str | None = None, g: Any = None) -> list[HFParamDecl]:
        geom = self._require_geometry(g)
        pfx = self.prefix if prefix is None else prefix
        ren = getattr(geom, "rename", {}) or {}
        q_name = ren.get("q_proj", "q_proj")
        k_name = ren.get("k_proj", "k_proj")
        o_name = ren.get("o_proj", "o_proj")
        decls = [
            HFParamDecl(
                name=q_name,
                hf_name=f"{pfx}.q_proj.weight",
                shape=(geom.NH, geom.H, geom.HD),
                dtype=FP8,
                transform=ParamTransform(transpose=(0, 2, 1), shard_axis=0),
                vllm_aliases=(VLLMAlias(source=f"{pfx}.qkv_proj", targets=(q_name,)),),
            ),
            HFParamDecl(
                name=k_name,
                hf_name=f"{pfx}.k_proj.weight",
                shape=(geom.NKV, geom.H, geom.HD),
                dtype=FP8,
                transform=ParamTransform(transpose=(0, 2, 1), shard_axis=0),
                vllm_aliases=(VLLMAlias(source=f"{pfx}.qkv_proj", targets=(k_name,)),),
            ),
            HFParamDecl(
                name=o_name,
                hf_name=f"{pfx}.o_proj.weight",
                shape=(geom.NH * geom.HD, geom.H),
                dtype=BF16,
                transform=ParamTransform(transpose=(1, 0), shard_axis=0),
                vllm_aliases=(VLLMAlias(source=f"{pfx}.o_proj", targets=(o_name,)),),
            ),
        ]
        return apply_decl_overrides(decls, geom)

    def cache_decl(self, g: Any = None) -> KVCacheDecl:
        geom = self._require_geometry(g)
        return KVCacheDecl(hd=geom.HD, kvh=geom.NKV, page_size=128, page_dtype=BF16)

    def reference(self, params: Any, x: Any, state: Any = None, **kwargs: Any) -> Any:
        geom = self._require_geometry(kwargs.get("g"))
        ren = getattr(geom, "rename", {}) or {}
        key = self.options.get("param_name") or ren.get("q_proj", "q_proj")
        w = params[key] if isinstance(params, Mapping) else params
        return (x + w, state)


def test_required_and_frozen_dict_invariants():
    assert repr(REQUIRED) == "REQUIRED"
    assert copy.deepcopy(REQUIRED) is REQUIRED
    assert pickle.loads(pickle.dumps(REQUIRED)) is REQUIRED

    fd1 = FrozenDict({"a": [1, {"x": [2, 3]}], "b": 4})
    fd2 = FrozenDict({"b": 4, "a": (1, {"x": (2, 3)})})
    assert fd1 == fd2
    assert fd1 == {"a": (1, {"x": (2, 3)}), "b": 4}
    assert fd1 != 42
    assert hash(fd1) == hash(fd2)
    assert len(fd1) == 2 and set(fd1) == {"a", "b"}
    assert isinstance(fd1["a"], tuple)
    assert isinstance(fd1["a"][1], FrozenDict)
    assert "FrozenDict(" in repr(fd1)
    assert pickle.loads(pickle.dumps(fd1)) == fd1
    assert copy.deepcopy(fd1) == fd1


def test_cache_and_hf_param_declarations():
    kv = KVCacheDecl(hd=256, kvh=2, page_size=64, page_dtype=BF16)
    assert kv.kv_page_shape == (4, 64, 4, 64)

    rec = RecurrentCacheDecl(kh=4, vh=8)
    assert (rec.KD, rec.VD, rec.conv_taps) == (128, 128, 3)
    assert rec.rec_pages_per_slot is None and rec.conv_pages_per_slot is None

    # Transpose permutation derives inverse-permuted hf_shape when omitted.
    decl = HFParamDecl(
        name="k_proj",
        hf_name="self_attn.k_proj.weight",
        shape=(4, 512, 64),
        dtype=FP8,
        transform=ParamTransform(transpose=(0, 2, 1)),
    )
    assert decl.hf_shape == (4, 64, 512)

    with pytest.raises(ValueError, match="not a valid rank-3 permutation"):
        HFParamDecl(
            name="bad",
            hf_name="bad.weight",
            shape=(4, 512, 64),
            dtype=FP8,
            transform=ParamTransform(transpose=(0, 0, 1)),
        )

    with pytest.raises(TypeError, match="permutation tuple or None"):
        ParamTransform(transpose=True)  # type: ignore[arg-type]

    # combined_hf_param_shapes fuses uniform groups in fuse_order along fuse_axis.
    w1 = HFParamDecl(
        name="w1",
        hf_name="mlp.w1.weight",
        shape=(8, 256, 128),
        dtype=FP8,
        transform=ParamTransform(fuse_into="w13", fuse_order=0, fuse_axis=2),
    )
    w3 = HFParamDecl(
        name="w3",
        hf_name="mlp.w3.weight",
        shape=(8, 256, 64),
        dtype=FP8,
        transform=ParamTransform(fuse_into="w13", fuse_order=1, fuse_axis=2),
    )
    assert combined_hf_param_shapes([w3, w1]) == {"w13": ((8, 256, 192), FP8)}


def test_module_bind_resolve_and_strict_key_lookup():
    # Base Module defaults reject unimplemented bind/hf_params/reference/geometry.
    base = Module()
    with pytest.raises(NotImplementedError, match="Module.bind"):
        Module.bind()
    with pytest.raises(ValueError, match="requires a resolved geometry"):
        base._require_geometry()
    with pytest.raises(NotImplementedError, match="hf_params"):
        base.hf_params()
    with pytest.raises(NotImplementedError, match="reference"):
        base.reference()
    assert base.cache_decl() is None

    # Nested text_config and rope_parameters resolution + defaults + geometry methods.
    cfg = SimpleNamespace(
        text_config=SimpleNamespace(
            hidden_size=512,
            num_attention_heads=8,
            num_key_value_heads=2,
            rope_parameters={"rope_theta": 500000.0, "partial_rotary_factor": 0.5},
        )
    )
    binding = _ToyAttn.bind("self_attn")
    assert isinstance(binding, ModuleBinding)
    resolved = binding.resolve(cfg)
    assert isinstance(resolved, ResolvedModuleGeometry)
    assert resolved.geometry is resolved
    assert (resolved.H, resolved.hidden) == (512, 512)
    assert (resolved.NH, resolved.heads) == (8, 8)
    assert (resolved.NKV, resolved.kv_heads) == (2, 2)
    assert resolved.HD == 64
    assert resolved.theta == 500000.0
    assert resolved.q_width() == 512
    assert resolved.rot_dim() == 32
    with pytest.raises(AttributeError, match="no_such_attr"):
        _ = resolved.no_such_attr

    assert binding.cache_decl(cfg) == KVCacheDecl(
        hd=64, kvh=2, page_size=128, page_dtype=BF16
    )
    assert len(binding.hf_params(cfg)) == 3
    assert isinstance(binding.to_module(cfg), _ToyAttn)
    assert pickle.loads(pickle.dumps(binding)) == binding
    assert hash(binding) == hash(copy.deepcopy(binding))
    assert pickle.loads(pickle.dumps(resolved)) == resolved

    # Literal and (hf_key, default) field specs in bind.
    lit_resolved = _ToyAttn.bind(
        "self_attn", H=256, HD=("custom_hd", 32), kv_heads=4
    ).resolve(cfg)
    assert (lit_resolved.H, lit_resolved.HD, lit_resolved.NKV) == (256, 32, 4)

    # Missing REQUIRED field raises ValueError.
    with pytest.raises(ValueError, match="Cannot resolve field 'theta'"):
        _ToyAttn.bind("self_attn").resolve(
            SimpleNamespace(
                hidden_size=512, num_attention_heads=8, num_key_value_heads=2
            )
        )

    # Explicit HF config key must exist (no silent fallback to canonical/alias keys).
    with pytest.raises(ValueError, match="num_attn_heads_typo"):
        _ToyAttn.bind("self_attn", heads="num_attn_heads_typo").resolve(cfg)

    # Unknown field/option and fail-closed option validation.
    with pytest.raises(ValueError, match="unknown field or option 'bad_opt'"):
        _ToyAttn.bind("self_attn", bad_opt=1)
    with pytest.raises(ValueError, match="requires qk_norm=True"):
        _ToyAttn.bind("self_attn", qk_norm=False)


def test_rename_transforms_validation_and_reference_lookup():
    cfg = {
        "hidden_size": 256,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rope_theta": 10000.0,
    }

    with pytest.raises(ValueError, match="unknown rename key"):
        _ToyAttn.bind("self_attn", rename={"nonexistent": "foo"})
    with pytest.raises(ValueError, match="invalid rename"):
        _ToyAttn.bind("self_attn", rename={"q_proj": ""})
    with pytest.raises(ValueError, match="rename target collision"):
        _ToyAttn.bind("self_attn", rename={"q_proj": "k_proj"})
    with pytest.raises(ValueError, match="rename target collision"):
        _ToyAttn.bind("self_attn", rename={"q_proj": "dup_w", "k_proj": "dup_w"})
    with pytest.raises(ValueError, match="unknown transforms key"):
        _ToyAttn.bind("self_attn", transforms={"nonexistent": {"shard_axis": 0}})
    with pytest.raises(ValueError, match="unknown fields"):
        _ToyAttn.bind("self_attn", transforms={"q_proj": {"bad_field": 1}})
    with pytest.raises(TypeError, match="invalid type"):
        _ToyAttn.bind("self_attn", transforms={"q_proj": 123})

    # Valid rename + transforms override (both dict and ParamTransform).
    custom = _ToyAttn.bind(
        "self_attn",
        rename={"q_proj": "custom_q"},
        transforms={
            "custom_q": {"shard_axis": None},
            "o_proj": ParamTransform(transpose=None, shard_axis=1),
        },
    ).resolve(cfg)
    by_name = {d.name: d for d in custom.hf_params()}
    assert "custom_q" in by_name and "q_proj" not in by_name
    assert by_name["custom_q"].transform == ParamTransform(
        transpose=(0, 2, 1), shard_axis=None
    )
    assert by_name["o_proj"].transform == ParamTransform(transpose=None, shard_axis=1)

    with pytest.raises(TypeError, match="Invalid transforms"):
        apply_decl_overrides(
            custom.hf_params(),
            SimpleNamespace(transforms={"custom_q": 123}, rename={}),
        )

    # Strict key lookup in reference: renamed key succeeds, wrong key raises KeyError.
    out, st = custom.reference({"custom_q": 3}, 10, state="ok")
    assert (out, st) == (13, "ok")
    with pytest.raises(KeyError, match="custom_q"):
        custom.reference({"q_proj": 3}, 10)
    with pytest.raises(KeyError, match="custom_q"):
        custom.reference({"weight": 3}, 10)

    # param_name option override is allowed in rename/transforms and used by reference.
    named = _ToyAttn.bind(
        "self_attn",
        param_name="alt_q",
        transforms={"alt_q": {"shard_axis": None}},
    ).resolve(cfg)
    assert named.reference({"alt_q": 5}, 2) == (7, None)
    assert named.reference(5, 2, state=F32) == (7, F32)

    # A Module subclass without a custom __init__ works with bind(...).resolve().
    class _PlainModule(Module):
        name = "plain_module"
        GEOMETRY_FIELDS = {"H": ("hidden_size", REQUIRED)}
        OPTIONS = {"flag": False}

    plain = _PlainModule.bind("plain", flag=True).resolve(cfg).to_module()
    assert plain.prefix == "plain"
    assert plain.geometry.H == 256
    assert plain.options == FrozenDict({"flag": True})
    with pytest.raises(TypeError, match="unexpected option"):
        _PlainModule(bad_opt=1)
