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
"""Named execution profiles: per-bucket tuning bundles selected by name.

A `Profile` groups everything a performance engineer tunes for a deployment
(tiles per bucket, opt-in perf flags, bucket lists, pool-I/O mode) so model
files and library defaults stay general. The `default` profile is the
model-agnostic library default (`profiles/default.py`, split pool I/O, no
GPR). A model may declare one profile as its default (`model_default=True`,
`model_default_profile`); adapters (vLLM, SGLang, plain JAX) use it when the
user names no profile, select others by name, and apply a profile per bucket
via `Profile.exec_for`. Nothing in the core changes behaviour based on profiles.

Numerics classes (enforced, not just labels), relative to the reference
profile: the default profile of the profile's model (`Profile.model`), or the
`default` profile when the model declares none (or `model` is None):
- `exact`: must reproduce the reference profile bit-exactly.
- `approx`: changes numerics (e.g. `dense_acc_pair`, `dmm_unroll`,
  hierarchical all-reduce variants, fused pool I/O); must pass the accuracy
  ladder.

Every tile knob (core `Tiles` fields and each block's `Tunables`) and pool-I/O
mode declares its class (`exec_spec.tile_numerics()`, `pool.POOL_IO_NUMERICS`).
`approx_features(profile)` lists the approx features a profile turns on
relative to its reference; a profile declared `exact` that turns any of them on
is rejected at construction, and `Profile.with_overrides` re-labels
(`<name>+overrides`, class derived) instead of silently staying `exact`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from .exec_spec import ExecSpec, Tiles, tile_numerics

# Profiles own the pool-I/O mode, so they depend on the pool package's config
# (its numerics classes and the `CacheExec.pool_gather` a profile resolves).
from .pool.config import POOL_IO_NUMERICS, cache_exec, with_cache
from .profiles.default import default_tiles

# Name of the model-agnostic library default profile.
DEFAULT_PROFILE_NAME = "default"
# Tile rule: `(B, *, moe_tp, R) -> Tiles` for a bucket of `B` rows.
TileRule = Callable[..., Tiles]


@dataclass(frozen=True)
class Baseline:
    """Provenance of the reference a profile is tuned against."""

    ref: str
    commit: str = ""
    job: str = ""
    config: str = ""
    bench: str = ""


@dataclass(frozen=True)
class Profile:
    """Named bundle of execution tuning (tiles, flags, buckets, pool I/O)."""

    name: str
    model: str | None = None  # Model name this profile targets (None: any)
    numerics: Literal["exact", "approx"] = "exact"
    # Tile rule. None: the rule of `model`'s default profile (`tile_rule`), or
    # with no model (or one without a default profile) the library default
    # (`profiles/default.py`), left unresolved by `exec_for` so ring depths may
    # be refined at build time to fit the VMEM budget.
    tiles: TileRule | None = None
    buckets: tuple[int, ...] | None = None  # T=1 decode buckets (None: adapter default)
    verify_buckets: tuple[int, ...] | None = None  # T>1 buckets (None: adapter default)
    # Single-launch fused pool I/O: False | True | "auto" (bucket-dependent
    # policy using `fused_pool_io_max_b` and `fused_pool_io_min_b_hi`). Requires
    # pool donation.
    fused_pool_io: bool | str = False
    fused_pool_io_max_b: int = 24
    fused_pool_io_min_b_hi: int = 56
    # In-kernel recurrent gather (GPR): False (split) | None (auto: WIDE verify
    # buckets where eligible; exact) | True (every bucket where eligible, never
    # a T=1 bucket; approx, see `pool.POOL_IO_NUMERICS`).
    pool_gather: bool | None = False
    baseline: Baseline | None = None
    description: str = ""
    # The default profile of `model` (registered by the model; the reference
    # of every profile targeting that model, see `model_default_profile`).
    model_default: bool = False

    def __post_init__(self) -> None:
        if not self.name or self.name.strip() != self.name:
            raise ValueError(f"invalid profile name {self.name!r}")
        if self.numerics not in ("exact", "approx"):
            raise ValueError(
                f"profile {self.name!r}: numerics must be 'exact' or 'approx', "
                f"got {self.numerics!r}"
            )
        if isinstance(self.fused_pool_io, str) and self.fused_pool_io != "auto":
            raise ValueError(
                f"profile {self.name!r}: fused_pool_io must be a bool or 'auto', "
                f"got {self.fused_pool_io!r}"
            )
        if self.pool_gather not in (False, None, True):
            raise ValueError(
                f"profile {self.name!r}: pool_gather must be False, None or True, "
                f"got {self.pool_gather!r}"
            )
        if self.model_default and (self.model is None or self.numerics != "exact"):
            raise ValueError(
                f"profile {self.name!r}: a model default profile needs a model "
                "and numerics='exact'"
            )
        if self.numerics == "exact":
            feats = approx_features(self)
            if feats:
                raise ValueError(
                    f"profile {self.name!r} is declared numerics='exact' but turns on "
                    f"approx features {list(feats)}; declare numerics='approx'"
                )

    def with_overrides(self, **overrides: Any) -> Profile:
        """Return this profile with `overrides` applied, re-labelled if changed.

        Overrides equal to the current values are ignored. Any real change
        renames the profile to `<name>+overrides` and derives its numerics
        class: it stays `exact` only if this profile is `exact` and the result
        turns on no approx feature relative to the reference profile (it never
        silently stays `exact`). The result is never a model default.
        """
        changed = {k: v for k, v in overrides.items() if getattr(self, k) != v}
        if not changed:
            return self
        name = (
            self.name if self.name.endswith("+overrides") else f"{self.name}+overrides"
        )
        cand = dataclasses.replace(
            self, name=name, numerics="approx", model_default=False, **changed
        )
        exact = self.numerics == "exact" and not approx_features(cand)
        return dataclasses.replace(cand, numerics="exact" if exact else "approx")

    def tile_rule(self) -> TileRule | None:
        """This profile's tile rule (None: the library default rule).

        A profile targeting a model with `tiles=None` inherits the rule of that
        model's default profile, the reference its numerics are measured
        against.
        """
        if self.tiles is not None or self.model is None or self.model_default:
            return self.tiles
        ref = model_default_profile(self.model)
        return None if ref is self else ref.tiles

    def tiles_for(self, B: int, *, R: int, moe_tp: bool = False) -> Tiles:
        """Return this profile's tiles for a bucket of `B` rows on `R` ranks.

        For the library default rule (`tile_rule() is None`) these are the
        tiles before build-time VMEM refinement.
        """
        rule = self.tile_rule()
        return (default_tiles if rule is None else rule)(B, moe_tp=moe_tp, R=R)

    def exec_for(self, base: ExecSpec, *, B: int, T: int, g: Any = None) -> ExecSpec:
        """Specialize `base` for bucket `(T, B)` under this profile.

        Explicit settings on `base` win (expert / test overrides), the profile
        fills the rest:
        - `tiles`: `base.tiles` if set, else the profile's tile rule
          (`tile_rule`; left `None` for the library default rule, which may be
          refined at build time to fit the VMEM budget);
        - `pool_gather`: the `CacheExec.pool_gather` of `base` unless it is the
          library default
          `False` ("no explicit choice"), else the profile's: `None` -> auto,
          `True` -> `"eligible"` (gather wherever the pool backend can, e.g.
          never on a `T=1` bucket). Pass `pool_gather="split"` to force split
          under a GPR profile.
        Fused pool I/O is resolved by the caller because it also depends on the
        pool backend.
        """
        moe_tp = bool(getattr(g, "moe_tp", False)) if g is not None else False
        tiles = base.tiles
        if tiles is None and self.tile_rule() is not None:
            tiles = self.tiles_for(B, R=base.R, moe_tp=moe_tp)
        base_gather = cache_exec(base).pool_gather
        if base_gather is not False:
            pool_gather = base_gather
        else:
            pool_gather = "eligible" if self.pool_gather is True else self.pool_gather
        ex = dataclasses.replace(base, B=B, T=T, tiles=tiles)
        return (
            ex
            if pool_gather == base_gather
            else with_cache(ex, pool_gather=pool_gather)
        )


_REGISTRY: dict[str, Profile] = {}
# Model name -> name of the model's default profile.
_MODEL_DEFAULTS: dict[str, str] = {}
_BUILTINS_LOADED = False

# Bucket grid on which a profile's tile rule is compared to its reference.
_NUMERICS_GRID_B = range(1, 65)
_NUMERICS_GRID_R = (1, 2, 4, 8, 16)


def _knob(t: Tiles, key: str) -> Any:
    """Value of tile knob `key` (`field` or `<block>.<field>`) in `t`."""
    if "." not in key:
        return getattr(t, key)
    block, field = key.split(".", 1)
    return getattr(t.block(block), field)


def approx_features(profile: Profile) -> tuple[str, ...]:
    """Approx-class features `profile` turns on relative to its reference.

    The reference is `model_default_profile(profile.model)`; a model default
    profile is its own reference (no features). A tile field counts when the
    two tile rules give it different values on any bucket of the grid (`B` in
    1..64, `R` in {1, 2, 4, 8, 16}, `moe_tp` in {False, True}) and the knob is
    `approx` in `exec_spec.tile_numerics()` (block knobs as
    `tiles.<block>.<field>`); a pool-I/O mode the profile turns on and the
    reference does not counts per `POOL_IO_NUMERICS`.
    """
    if profile.model_default:
        return ()
    ref_name = _MODEL_DEFAULTS.get(profile.model or "", DEFAULT_PROFILE_NAME)
    ref = _REGISTRY.get(ref_name)
    if ref is profile:
        return ()
    # While the `default` profile itself is being built, its rule is the
    # library default with split pool I/O.
    ref_tiles = default_tiles if ref is None else ref.tiles_for
    ref_fused = False if ref is None else ref.fused_pool_io
    ref_gather = False if ref is None else ref.pool_gather
    feats: set[str] = set()
    approx_knobs = [k for k, v in tile_numerics().items() if v == "approx"]
    for R in _NUMERICS_GRID_R:
        for moe_tp in (False, True):
            for B in _NUMERICS_GRID_B:
                t = profile.tiles_for(B, moe_tp=moe_tp, R=R)
                d = ref_tiles(B, moe_tp=moe_tp, R=R)
                feats.update(
                    f"tiles.{k}" for k in approx_knobs if _knob(t, k) != _knob(d, k)
                )
    if (
        profile.fused_pool_io is not False
        and profile.fused_pool_io != ref_fused
        and POOL_IO_NUMERICS["fused_pool_io"] == "approx"
    ):
        feats.add("fused_pool_io")
    if profile.pool_gather is not False and profile.pool_gather != ref_gather:
        mode = "pool_gather_eligible" if profile.pool_gather is True else "pool_gather"
        if POOL_IO_NUMERICS[mode] == "approx":
            feats.add(mode)
    return tuple(sorted(feats))


DEFAULT_PROFILE = Profile(
    name=DEFAULT_PROFILE_NAME,
    numerics="exact",
    description=(
        "Model-agnostic library default: one tile set for every bucket "
        "(profiles/default.py) fitted to the model's VMEM budget, split pool "
        "I/O, no GPR, all opt-in perf flags off."
    ),
)

_REGISTRY[DEFAULT_PROFILE.name] = DEFAULT_PROFILE


def register_profile(profile: Profile, *, replace: bool = False) -> Profile:
    """Register `profile` by name (`replace=True` to overwrite an existing one).

    A `model_default` profile also becomes its model's default profile (one
    per model).
    """
    if profile.name in _REGISTRY and not replace:
        if _REGISTRY[profile.name] == profile:
            return _REGISTRY[profile.name]  # idempotent (e.g. re-import)
        raise ValueError(f"profile {profile.name!r} is already registered")
    if profile.model_default:
        assert profile.model is not None  # checked in `Profile.__post_init__`
        prev = _MODEL_DEFAULTS.get(profile.model)
        if prev is not None and prev != profile.name and not replace:
            raise ValueError(
                f"model {profile.model!r} already has default profile {prev!r}"
            )
        _MODEL_DEFAULTS[profile.model] = profile.name
    _REGISTRY[profile.name] = profile
    return profile


def model_default_profile(model: str | None) -> Profile:
    """The default profile of `model`, else `default`."""
    name = _MODEL_DEFAULTS.get(model) if model is not None else None
    return _REGISTRY[name] if name is not None else DEFAULT_PROFILE


def _load_builtins() -> None:
    global _BUILTINS_LOADED
    if not _BUILTINS_LOADED:
        from . import profiles

        profiles.load_builtin_profiles()  # loads model-agnostic profile hook
        _BUILTINS_LOADED = True  # only after a successful import


def list_profiles() -> tuple[str, ...]:
    """Return the names of all registered profiles (built-ins included)."""
    _load_builtins()
    return tuple(sorted(_REGISTRY))


def get_profile(name: str | Profile | None) -> Profile:
    """Resolve a profile by name (`None` / `""` -> `default`)."""
    if isinstance(name, Profile):
        return name
    key = (name or "default").strip()
    _load_builtins()
    if key not in _REGISTRY:
        raise KeyError(
            f"unknown megakernel profile {key!r}; available: {list(list_profiles())}"
        )
    return _REGISTRY[key]


__all__ = [
    "DEFAULT_PROFILE",
    "Baseline",
    "Profile",
    "TileRule",
    "approx_features",
    "get_profile",
    "list_profiles",
    "model_default_profile",
    "register_profile",
]
