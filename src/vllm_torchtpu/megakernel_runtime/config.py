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
"""Megakernel runtime config and plugin loading (stdlib only).

Sources, in order (per key the later one wins; `options` merges per key):
`VLLM_TPU_MEGAKERNEL`, then `additional_config["megakernel"]`. Each is unset /
a false string (off), a true string (on, plugin from the checkpoint's HF arch),
`ep` / `tp` (on, `options.mode`), a shorthand template name (`plugin.SHORTHANDS`),
a JSON object, or `k=v,k=v` (`:` separates list items; `options` needs JSON).

Runtime keys: `plugin`, `options`, `enabled`, `ab`, `stats_period_s`. A
top-level template key (e.g. `profile`) is forwarded into `options` only when
the resolved plugin is a shorthand template that lists it; any other key is a
`MegakernelConfigError` naming it.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import json
from collections.abc import Callable, Mapping
from types import ModuleType
from typing import Any

from vllm_torchtpu.megakernel.errors import MegakernelConfigError

from .plugin import API_VERSION, SHORTHANDS, TpuMonoSpec

RUNTIME_KEYS = frozenset({"plugin", "options", "enabled", "ab", "stats_period_s"})
_LEGACY_KEYS = frozenset().union(*(s.legacy_keys for s in SHORTHANDS.values()))
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


def parse_config(*sources: Any, raw_env: Any = None) -> MegakernelConfig:
    """Parse config sources (str, bool or mapping; `None` skipped) in order."""
    st: dict[str, Any] = {
        "enabled": False,
        "plugin": None,
        "options": {},
        "ab": False,
        "stats_period_s": DEFAULT_STATS_PERIOD_S,
    }
    legacy: dict[str, Any] = {}

    def apply_dict(d: Mapping[str, Any]) -> None:
        st["enabled"] = parse_bool(d.get("enabled", True))
        for k, v in d.items():
            k = str(k)
            if k not in RUNTIME_KEYS and k not in _LEGACY_KEYS:
                raise MegakernelConfigError(
                    f"megakernel config: unknown key {k!r}; runtime keys are "
                    f"{sorted(RUNTIME_KEYS)} and template settings go in 'options'"
                )
            if v is None or k == "enabled":
                continue
            if k == "plugin":
                st["plugin"] = _parse_plugin_ref(v)
            elif k == "options":
                st["options"] = {**st["options"], **_parse_options(v)}
            elif k == "ab":
                st["ab"] = parse_bool(v, key="ab")
            elif k == "stats_period_s":
                try:
                    st["stats_period_s"] = float(v)
                except (TypeError, ValueError) as exc:
                    raise MegakernelConfigError(
                        f"megakernel config: invalid stats_period_s={v!r}"
                    ) from exc
            else:
                legacy[k] = v

    for src in sources:
        if src is None:
            continue
        if isinstance(src, bool):
            st["enabled"] = src
        elif isinstance(src, Mapping):
            apply_dict(src)
        else:
            s, sl = str(src).strip(), str(src).strip().lower()
            if sl in _FALSE_STRS:
                st["enabled"] = False
            elif sl in _TRUE_STRS:
                st["enabled"] = True
            elif sl in ("ep", "tp"):
                st["enabled"], legacy["mode"] = True, sl
            elif sl in SHORTHANDS:
                st["enabled"], st["plugin"] = True, sl
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
                apply_dict(parsed)
            else:
                kv: dict[str, Any] = {}
                for part in (p.strip() for p in s.split(",") if p.strip()):
                    if "=" not in part:
                        raise MegakernelConfigError(
                            f"Invalid megakernel config entry {part!r} in {s!r}"
                        )
                    k, v = part.split("=", 1)
                    kv[k.strip()] = v.strip()
                apply_dict(kv)

    if clash := sorted(set(legacy) & set(st["options"])):
        raise MegakernelConfigError(
            f"megakernel config: {clash} set both at top level and in 'options'"
        )
    raw = "" if raw_env is None else str(raw_env)
    return MegakernelConfig(**st, legacy=legacy, raw_env=raw)


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
    the HF config's `architectures` / `model_type` only)."""
    for cand in _hf_arch_candidates(hf_config):
        for name, sh in SHORTHANDS.items():
            if cand in sh.architectures or cand in sh.model_types:
                return name
    return None


def resolve_plugin_ref(
    cfg: MegakernelConfig, hf_config: Any = None
) -> tuple[PluginRef, dict[str, Any]] | None:
    """Resolve the plugin and its `options` (`None` if no template matches the
    arch). Top-level template keys are forwarded into `options` only for a
    shorthand template that lists them."""
    if cfg.plugin is not None and cfg.plugin.lower() in SHORTHANDS:
        name = cfg.plugin.lower()
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
        allowed = SHORTHANDS[ref.shorthand].legacy_keys if ref.shorthand else ()
        if bad := sorted(k for k in cfg.legacy if k not in allowed):
            raise MegakernelConfigError(
                f"megakernel config: unknown key {bad[0]!r} for plugin "
                f"{ref.module!r}; put plugin settings in 'options'"
            )
        options = {**cfg.legacy, **options}
    return ref, options


def select_plugin(
    env_raw: Any, additional_config: Any, hf_config: Any
) -> tuple[MegakernelConfig, tuple[PluginRef, dict[str, Any]] | None]:
    """The config and the resolved plugin (`None` when the megakernel is off
    or no shorthand claims the checkpoint)."""
    cfg = collect_config(env_raw, additional_config)
    return cfg, (resolve_plugin_ref(cfg, hf_config) if cfg.enabled else None)


def missing_template(ref: PluginRef, hf_config: Any = None) -> str | None:
    """The log line when `ref` is a shorthand template whose module isn't
    installed (`None` otherwise). Shorthands are imported by string, so the
    runtime works without the model folders; then the megakernel stays off.
    An explicit plugin reference that doesn't import is an error instead."""
    if ref.shorthand is None:
        return None
    try:
        if importlib.util.find_spec(ref.module) is not None:
            return None
    except ModuleNotFoundError:
        pass
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
    if int(plugin.api_version) != API_VERSION:
        raise MegakernelConfigError(
            f"megakernel plugin {ref.module}:{ref.attr} targets api_version "
            f"{plugin.api_version}; this runtime supports {API_VERSION}"
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
