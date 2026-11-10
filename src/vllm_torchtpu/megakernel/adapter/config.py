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
"""Megakernel adapter config and plugin loading (stdlib only).

Sources, in order: `VLLM_TPU_MEGAKERNEL`, then `additional_config["megakernel"]`.
Each is unset (`None` or an empty string), a bool, a string or a mapping. A
string is a false word (off), a true word (on, plugin from the checkpoint's HF
arch), `ep` / `tp` (on, with `options.mode`), a shorthand template name
(`plugin.SHORTHANDS`; on, with that plugin), a JSON object, or `k=v,k=v`
(`options` needs JSON).

Precedence is per key, the later source winning; `parse_config` states the
rule, including how `enabled` is decided.

Runtime keys: `plugin`, `options`, `enabled`, `ab`, `stats_period_s`. Any
other top-level key (e.g. `profile`) is a plugin setting: `resolve_plugin_ref`
forwards it into `options` when the resolved plugin's `SPEC.top_level_options`
lists it, and raises a `MegakernelConfigError` naming it otherwise. The
adapter reads `SPEC` by importing the plugin's module (stdlib-only at import,
so none of the plugin's heavy code), and only when such keys are set.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import json
import math
from collections.abc import Callable, Mapping
from types import ModuleType
from typing import Any

from vllm_torchtpu.megakernel.core.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.core.runtime.contract import API_VERSION, TpuMonoSpec

from .plugin import SHORTHANDS

RUNTIME_KEYS = frozenset({"plugin", "options", "enabled", "ab", "stats_period_s"})
_TRUE_STRS = ("1", "true", "yes", "on")
_FALSE_STRS = ("0", "false", "no", "off", "none", "")
DEFAULT_STATS_PERIOD_S = 60.0


def parse_bool(val: Any, *, key: str = "enabled") -> bool:
    """Parse a boolean config value (`1/true/yes/on`, `0/false/no/off/none/""`)."""
    if isinstance(val, (bool, int, float)):
        return bool(val)
    s = str(val).strip().lower()
    if s in _TRUE_STRS or s in _FALSE_STRS:
        return s in _TRUE_STRS
    raise MegakernelConfigError(f"megakernel config: invalid boolean {key}={val!r}")


@dataclasses.dataclass(frozen=True)
class MegakernelConfig:
    """The parsed runtime keys, plus top-level template keys not yet forwarded."""

    enabled: bool = False
    plugin: str | None = None
    options: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    ab: bool = False
    stats_period_s: float = DEFAULT_STATS_PERIOD_S
    legacy: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    raw_env: str = ""

    def runtime_keys(self) -> dict[str, Any]:
        """The resolved runtime keys (as hashed into the plugin fingerprint)."""
        return {
            "enabled": self.enabled,
            "plugin": self.plugin,
            "options": dict(self.options),
            "ab": self.ab,
            "stats_period_s": self.stats_period_s,
        }


def _parse_options(val: Any) -> dict[str, Any]:
    if isinstance(val, str):
        try:
            val = json.loads(val)
        except json.JSONDecodeError as exc:
            raise MegakernelConfigError(
                f"megakernel config: options must be a JSON object, got {val!r}"
            ) from exc
    if not isinstance(val, Mapping):
        raise MegakernelConfigError(
            f"megakernel config: options must be a mapping, got {val!r}"
        )
    return {str(k): v for k, v in val.items()}


def _parse_plugin_ref(val: Any) -> str:
    s = str(val).strip()
    mod, _, attr = s.partition(":")
    if not mod or (":" in s and not attr.isidentifier()):
        raise MegakernelConfigError(
            f"megakernel config: invalid plugin={val!r}; expected "
            "'<module>[:<attr>]' or a shorthand name"
        )
    return s


def _parse_stats_period_s(val: Any) -> float:
    """Parse `stats_period_s`: a finite number of seconds, `>= 0` (`0` logs
    every step). Bools, negatives, NaN and inf are errors."""
    if isinstance(val, bool):
        raise MegakernelConfigError(
            f"megakernel config: stats_period_s={val!r} must be a number of "
            "seconds, not a bool"
        )
    try:
        period = float(val)
    except (TypeError, ValueError) as exc:
        raise MegakernelConfigError(
            f"megakernel config: invalid stats_period_s={val!r}"
        ) from exc
    if not math.isfinite(period) or period < 0:
        raise MegakernelConfigError(
            f"megakernel config: stats_period_s={val!r} must be finite and >= 0"
        )
    return period


@dataclasses.dataclass
class _ParseState:
    """What `parse_config` has gathered from the sources so far."""

    runtime: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "plugin": None,
            "options": {},
            "ab": False,
            "stats_period_s": DEFAULT_STATS_PERIOD_S,
        }
    )
    template_keys: dict[str, Any] = dataclasses.field(default_factory=dict)
    # The last explicit on / off, and whether any source configured anything.
    explicit_enabled: bool | None = None
    configured: bool = False


def _merge_template_settings(
    state: _ParseState, options: dict[str, Any], template_keys: dict[str, Any]
) -> None:
    """Merge one source's template settings: a key it sets at top level or in
    `options` replaces that key from every earlier source."""
    if clash := sorted(set(template_keys) & set(options)):
        raise MegakernelConfigError(
            f"megakernel config: {clash} set both at top level and in 'options'"
        )
    merged = state.runtime["options"]
    for k in options:
        state.template_keys.pop(k, None)
    for k in template_keys:
        merged.pop(k, None)
    state.runtime["options"] = {**merged, **options}
    state.template_keys.update(template_keys)


def _apply_mapping_source(state: _ParseState, source: Mapping[str, Any]) -> None:
    """Apply one mapping source (a dict, a JSON object or parsed `k=v` pairs)."""
    options: dict[str, Any] = {}
    template_keys: dict[str, Any] = {}
    for k, v in source.items():
        k = str(k)
        if k not in RUNTIME_KEYS and not k.isidentifier():
            raise MegakernelConfigError(
                f"megakernel config: invalid key {k!r}; runtime keys are "
                f"{sorted(RUNTIME_KEYS)} and plugin settings go in 'options'"
            )
        if v is None:
            continue
        state.configured = True
        if k == "enabled":
            state.explicit_enabled = parse_bool(v)
        elif k == "plugin":
            state.runtime["plugin"] = _parse_plugin_ref(v)
            if source.get("enabled") is None:
                state.explicit_enabled = True
        elif k == "options":
            options.update(_parse_options(v))
        elif k == "ab":
            state.runtime["ab"] = parse_bool(v, key="ab")
        elif k == "stats_period_s":
            state.runtime["stats_period_s"] = _parse_stats_period_s(v)
        else:
            template_keys[k] = v
    _merge_template_settings(state, options, template_keys)


def _apply_string_source(state: _ParseState, raw: str) -> None:
    """Apply one string source: an on / off word, `ep` / `tp`, a shorthand
    name, a JSON object or `k=v,k=v`."""
    s, sl = raw.strip(), raw.strip().lower()
    if not sl:
        return  # an empty string (an unset variable) is no source
    if sl in _FALSE_STRS:
        state.explicit_enabled = False
    elif sl in _TRUE_STRS:
        state.explicit_enabled = True
    elif sl in ("ep", "tp"):
        state.explicit_enabled = True
        _merge_template_settings(state, {"mode": sl}, {})
    elif sl in SHORTHANDS:
        state.explicit_enabled = True
        state.runtime["plugin"] = sl
    elif s.startswith("{"):
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError as exc:
            raise MegakernelConfigError(
                f"megakernel config: invalid JSON {s!r}: {exc}"
            ) from exc
        if not isinstance(parsed, dict):
            raise MegakernelConfigError(
                f"megakernel config: expected a JSON object in {s!r}"
            )
        _apply_mapping_source(state, parsed)
    else:
        pairs: dict[str, Any] = {}
        for part in (p.strip() for p in s.split(",") if p.strip()):
            if "=" not in part:
                raise MegakernelConfigError(
                    f"Invalid megakernel config entry {part!r} in {s!r}"
                )
            k, v = part.split("=", 1)
            pairs[k.strip()] = v.strip()
        _apply_mapping_source(state, pairs)


def parse_config(*sources: Any, raw_env: Any = None) -> MegakernelConfig:
    """Parse config sources (str, bool or mapping; `None` skipped) in order.

    Precedence is per key, the later source winning: `options` merges per key,
    and a top-level template key (e.g. `mode`) and the same key in `options`
    override each other across sources (a source setting both is an error).

    `enabled` is decided by the last source that sets it explicitly: a bool, a
    false / true word, `ep` / `tp`, a shorthand name, an `enabled` key, or a
    `plugin` key without `enabled` (which counts as true). If no source sets it
    explicitly, the megakernel is on when any source configures it (any runtime
    or template key, e.g. `ab=1` or `{"options": {...}}`) and off when no
    source configures anything. So `VLLM_TPU_MEGAKERNEL=0` stays off whatever
    settings a later source adds, unless that source turns it on explicitly,
    and `ab=1` alone turns it on.
    """
    state = _ParseState()
    for src in sources:
        if src is None:
            continue
        if isinstance(src, bool):
            state.explicit_enabled = src
        elif isinstance(src, Mapping):
            _apply_mapping_source(state, src)
        else:
            _apply_string_source(state, str(src))
    enabled = (
        state.explicit_enabled
        if state.explicit_enabled is not None
        else state.configured
    )
    raw = "" if raw_env is None else str(raw_env)
    return MegakernelConfig(
        enabled=enabled,
        **state.runtime,
        legacy=state.template_keys,
        raw_env=raw,
    )


def collect_config(env_raw: Any, additional_config: Any) -> MegakernelConfig:
    """Parse `VLLM_TPU_MEGAKERNEL`, then `additional_config["megakernel"]`."""
    sources: list[Any] = [env_raw]
    if isinstance(additional_config, dict) and "megakernel" in additional_config:
        sources.append(additional_config["megakernel"])
    return parse_config(*sources, raw_env=env_raw)


@dataclasses.dataclass(frozen=True)
class PluginRef:
    """Where a plugin comes from: `module:attr`, and its shorthand name if any."""

    module: str
    attr: str = "SPEC"
    shorthand: str | None = None


def _hf_arch_candidates(hf_config: Any) -> list[str]:
    def sub(cfg: Any, key: str) -> Any:
        if isinstance(cfg, Mapping):
            return cfg.get(key)
        return getattr(cfg, key, None)

    cands: list[str] = []
    if hf_config is not None:
        archs = sub(hf_config, "architectures") or ()
        if isinstance(archs, (list, tuple)):
            cands.extend(str(a) for a in archs if a)
        for root in (hf_config, sub(hf_config, "text_config")):
            if root is not None and (mt := sub(root, "model_type")):
                cands.append(str(mt))
    return cands


def shorthand_for_arch(hf_config: Any) -> str | None:
    """The shorthand template that serves this checkpoint's arch, if any (from
    the HF config's `architectures` / `model_type`, also of `text_config`)."""
    for cand in _hf_arch_candidates(hf_config):
        for name, sh in SHORTHANDS.items():
            if cand in sh.architectures or cand in sh.model_types:
                return name
    return None


def resolve_plugin_ref(
    cfg: MegakernelConfig, hf_config: Any = None
) -> tuple[PluginRef, dict[str, Any]] | None:
    """Resolve the plugin and its `options` (`None` if no template matches the
    arch): a shorthand name (`SHORTHANDS`), a `module[:attr]` reference, or
    the shorthand claiming the checkpoint. Top-level plugin keys are forwarded
    into `options` when the plugin's `SPEC.top_level_options` lists them (see
    `_top_level_options`)."""
    if cfg.plugin is not None and cfg.plugin.strip().lower() in SHORTHANDS:
        name = cfg.plugin.strip().lower()
        ref = PluginRef(SHORTHANDS[name].module, shorthand=name)
    elif cfg.plugin is not None:
        mod, _, attr = cfg.plugin.partition(":")
        ref = PluginRef(mod.strip(), attr.strip() or "SPEC")
    else:
        name = shorthand_for_arch(hf_config)
        if name is None:
            return None
        ref = PluginRef(SHORTHANDS[name].module, shorthand=name)
    options = dict(cfg.options)
    if cfg.legacy:
        allowed = _top_level_options(ref)
        if allowed is not None and (
            bad := sorted(k for k in cfg.legacy if k not in allowed)
        ):
            raise MegakernelConfigError(
                f"megakernel config: unknown key {bad[0]!r} for plugin "
                f"{ref.module!r}; runtime keys are {sorted(RUNTIME_KEYS)}, its "
                f"SPEC.top_level_options are {sorted(allowed)}, and other plugin "
                "settings go in 'options'"
            )
        options = {**cfg.legacy, **options}
    return ref, options


def _top_level_options(ref: PluginRef) -> frozenset[str] | None:
    """The plugin's `SPEC.top_level_options`, read by importing the module
    that defines `SPEC` (stdlib-only at import; the plugin's heavy code is
    imported later, from `create`). `None` for a shorthand template that
    isn't installed: then `missing_template` turns the megakernel off."""
    if ref.shorthand is not None and missing_template(ref) is not None:
        return None
    spec, _ = import_plugin(ref)
    return spec.top_level_options


def select_plugin(
    env_raw: Any, additional_config: Any, hf_config: Any
) -> tuple[MegakernelConfig, tuple[PluginRef, dict[str, Any]] | None]:
    """The config and the resolved plugin (`None` when the megakernel is off
    or no shorthand claims the checkpoint)."""
    cfg = collect_config(env_raw, additional_config)
    return cfg, (resolve_plugin_ref(cfg, hf_config) if cfg.enabled else None)


def _is_package_or_parent(name: str | None, module: str) -> bool:
    """Whether `name` is `module` itself or one of its parent packages."""
    return name is not None and (module == name or module.startswith(name + "."))


def missing_template(ref: PluginRef, hf_config: Any = None) -> str | None:
    """The log line when `ref` is a shorthand template whose module isn't
    installed (`None` otherwise). Shorthands are imported by string, so the
    adapter works without the model folders; then the megakernel stays off.
    An explicit plugin reference that doesn't import is an error instead.

    `find_spec` imports the parent packages, so a `ModuleNotFoundError` from
    a broken import inside one of them (a module other than the template
    package or its parents) is re-raised rather than read as "not installed".
    """
    if ref.shorthand is None:
        return None
    try:
        if importlib.util.find_spec(ref.module) is not None:
            return None
    except ModuleNotFoundError as exc:
        if not _is_package_or_parent(exc.name, ref.module):
            raise
    archs = ", ".join(_hf_arch_candidates(hf_config)[:1]) or ref.shorthand
    return (
        f"Megakernel: no megakernel template for arch {archs} "
        f"({ref.module} is not installed); skipping."
    )


def import_plugin(ref: PluginRef) -> tuple[TpuMonoSpec, ModuleType]:
    """Import `ref` and check it is a `TpuMonoSpec` of a supported API."""
    try:
        module = importlib.import_module(ref.module)
    except Exception as exc:
        raise MegakernelConfigError(
            f"megakernel plugin module {ref.module!r} failed to import: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    plugin = getattr(module, ref.attr, None)
    if not isinstance(plugin, TpuMonoSpec):
        raise MegakernelConfigError(
            f"megakernel plugin {ref.module}:{ref.attr} is not a TpuMonoSpec "
            f"(got {type(plugin).__name__})"
        )
    if plugin.api_version != API_VERSION:
        raise MegakernelConfigError(
            f"megakernel plugin {ref.module}:{ref.attr} targets api_version "
            f"{plugin.api_version}; this adapter supports {API_VERSION}"
        )
    return plugin, module


def resolve_callable(
    target: str | Callable[..., Any], module: ModuleType, *, what: str
) -> Callable[..., Any]:
    """Resolve a plugin `"module:attr"` string (`.x:y` relative to the plugin
    package) or return the callable itself."""
    if callable(target):
        return target
    if not isinstance(target, str) or ":" not in target:
        raise MegakernelConfigError(
            f"megakernel plugin {module.__name__}: {what}={target!r} must be a "
            "callable or a 'module:attr' string"
        )
    mod_name, _, attr = target.partition(":")
    package = module.__name__ if hasattr(module, "__path__") else module.__package__
    try:
        mod = importlib.import_module(mod_name, package=package or None)
        fn = getattr(mod, attr)
    except Exception as exc:
        raise MegakernelConfigError(
            f"megakernel plugin {module.__name__}: can't resolve {what}={target!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not callable(fn):
        raise MegakernelConfigError(
            f"megakernel plugin {module.__name__}: {what}={target!r} is not callable"
        )
    return fn


__all__ = [
    "RUNTIME_KEYS",
    "MegakernelConfig",
    "MegakernelConfigError",
    "PluginRef",
    "collect_config",
    "import_plugin",
    "missing_template",
    "parse_bool",
    "parse_config",
    "resolve_callable",
    "resolve_plugin_ref",
    "select_plugin",
    "shorthand_for_arch",
]
