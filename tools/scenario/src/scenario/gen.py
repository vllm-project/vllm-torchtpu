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
"""The generated files: JSON schemas for editors and the reference.

The pydantic models are the only definition of the schema; these files are
derived from them, the rule table and the plumbing registry, and committed.
`scenario dev regen` writes them and `--check` fails when one is stale.
"""

import json
import pathlib
import types
from collections.abc import Callable, Iterator, Sequence
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel, JsonValue
from pydantic.fields import FieldInfo

from scenario import PLAN_FORMAT
from scenario.accel import ACCELERATORS
from scenario.model import baseline, common, scenario, tasks
from scenario.model.rules import rule_table
from scenario.plan import argv, codec, plumbing
from scenario.plan import types as plan_types
from scenario.reserved import (
    AGENTX_CLIENT_ENV,
    CHIPLESS_DRIVER_ENV,
    INFERENCEX_PERF_ENV,
    LM_EVAL_MODEL_ARGS,
    PERF_CLIENT_ARGS,
    RAY_ENV,
    ROUTER_ARGS,
    SERVE_ARGS,
    TOOL_ENV,
    TOOL_ENV_PREFIXES,
)

SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
REFERENCE = "REFERENCE.md"
SCENARIO_SCHEMA = "schema/scenario.schema.json"
DEFAULTS_SCHEMA = "schema/defaults.schema.json"
REGEN = "uv run --project tools/scenario scenario dev regen"


# JSON schemas -----------------------------------------------------------------


def _layer_schema(node: Any) -> Any:
    """A schema any single layer satisfies: nothing required, null allowed.

    A file with only `extends` and a few keys is valid; null is how a layer
    deletes a key. Unknown keys still fail, and enums stay.
    """
    if isinstance(node, list):
        return [_layer_schema(item) for item in node]
    if not isinstance(node, dict):
        return node
    out = {k: _layer_schema(v) for k, v in node.items() if k != "required"}
    if isinstance(node.get("properties"), dict):
        out["properties"] = {
            name: _nullable(_layer_schema(prop))
            for name, prop in node["properties"].items()
        }
    return out


def _nullable(prop: dict[str, Any]) -> dict[str, Any]:
    prop = {k: v for k, v in prop.items() if k != "title"}
    if {"type": "null"} in prop.get("anyOf", []):
        return prop
    keep = {k: prop.pop(k) for k in ("description", "default") if k in prop}
    return {**keep, "anyOf": [prop, {"type": "null"}]}


def json_schema(model: type[BaseModel], title: str) -> str:
    schema = _layer_schema(model.model_json_schema(by_alias=True, mode="validation"))
    schema = {"$schema": SCHEMA_DIALECT, **schema, "title": title}
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


# Types in words ---------------------------------------------------------------

# Field types with a name of their own; matched by the validators they carry.
_NAMED: list[tuple[Any, str]] = [
    (common.Duration, "duration: 2h, 90m, 4h10m, 30s or seconds"),
    (common.ByteSize, "size, e.g. 64Gi"),
    (common.Slice, "slice: accelerator/topology"),
    (common.ImageRef, "`ci` or an image URI"),
    (common.Name, "name"),
    (common.RoleName, "role name"),
    (common.PointName, "point name"),
    (common.TaskName, "task name"),
    (common.Label, "label"),
    (common.ServedModelName, "string"),
    (common.RefId, "scenario reference"),
    (common.RefOrPath, "scenario reference or relative path"),
    (common.EnvName, "env name"),
    (common.EnvValue, "string or integer"),
    (common.ArgKey, "flag name"),
    (common.KeyPathText, "key path"),
    (common.PosInt, "positive integer"),
    (common.NonNegInt, "non-negative integer"),
    (common.Tolerance, "number in (0, 1)"),
    (common.UnitInterval, "number in [0, 1]"),
    (tasks.IslOsl, "[input length, output length]"),
    (tasks.LmEvalTaskName, "lm_eval task name"),
    (tasks.ModelArgKey, "model_args key"),
    (tasks.ModelArgValue, "string, number or boolean"),
    (baseline.CellKey, "cell key"),
    (baseline.RunKey, "run key"),
    (baseline.EvalMetricKey, "metric,filter"),
    (baseline.MetricName, "metric name"),
    (baseline.NonNegFloat, "number >= 0"),
    (scenario.SchemaVersion, "`1`"),
]
_BY_METADATA = {id(meta): text for alias, text in _NAMED for meta in alias.__metadata__}
# Patterns declared in place, by their kind.
_MATCHES = {"driver": "role name or `client`"}
_MAPS = {
    common.ArgMap: "arg map: flag name to value",
    common.RoleArgMap: "arg map; null removes an inherited flag",
    common.EnvMap: "env map: name to string",
    common.RoleEnvMap: "env map; null removes an inherited entry",
    common.ArgValue: "boolean, number, string, list or mapping",
}


# annotated-types constraint classes, by name, and the attribute holding the bound.
_BOUNDS = {
    "Gt": "gt",
    "Ge": "ge",
    "Lt": "lt",
    "Le": "le",
    "MinLen": "min_length",
    "MaxLen": "max_length",
}


def _constrained(base: Any, meta: Sequence[Any]) -> str:
    bounds = {}
    for item in meta:
        for bound in item.metadata if isinstance(item, FieldInfo) else [item]:
            kind = type(bound).__name__
            if kind in _BOUNDS:
                bounds[kind] = getattr(bound, _BOUNDS[kind])
    text = type_text(base)
    if base is int and bounds.get("Gt") == 0:
        return "positive integer"
    if base is int and bounds.get("Ge") == 0:
        return "non-negative integer"
    if "MinLen" in bounds and bounds.get("MinLen") == bounds.get("MaxLen"):
        return f"{text}, {bounds['MinLen']} items"
    if bounds.get("MinLen") == 1:
        return f"non-empty {text}"
    words = {"Gt": ">", "Ge": ">=", "Lt": "<", "Le": "<="}
    limits = [f"{words[k]} {v}" for k, v in bounds.items() if k in words]
    return f"{text} {' and '.join(limits)}" if limits else text


def type_text(annotation: Any) -> str:
    """A field type in words, models as links to their tables."""
    if annotation in _MAPS:
        return _MAPS[annotation]
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Annotated:
        for meta in args[1:]:
            if id(meta) in _BY_METADATA:
                return _BY_METADATA[id(meta)]
            if isinstance(meta, common.Matches):
                return _MATCHES.get(meta.kind, "string")
        return _constrained(args[0], args[1:])
    if origin in (Union, types.UnionType):
        parts = [type_text(arg) for arg in args if arg is not type(None)]
        text = " or ".join(dict.fromkeys(parts))
        return text + (" or null" if type(None) in args else "")
    if origin is Literal:
        return " or ".join(f"`{value}`" for value in args)
    if origin is list:
        return f"list of {type_text(args[0])}"
    if origin is dict:
        return f"mapping of {type_text(args[0])} to {type_text(args[1])}"
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return f"[{annotation.__name__}](#{annotation.__name__.lower()})"
    if annotation is JsonValue:
        return "any JSON value"
    names = {int: "integer", str: "string", float: "number", bool: "boolean"}
    return names.get(annotation, getattr(annotation, "__name__", str(annotation)))


# Defaults models that can supply a field the scenario model requires.
_DEFAULTED: list[tuple[type[BaseModel], type[BaseModel]]] = [
    (scenario.Server, scenario.DefaultsServer),
    (tasks.PerfTask, scenario.DefaultsPerf),
    (tasks.AgentxTask, scenario.DefaultsAgentx),
    (tasks.AgentxLabels, scenario.DefaultsAgentxLabels),
    (tasks.LmEval, scenario.DefaultsLmEval),
]


def _value(value: Any) -> str:
    if value is None:
        return "unset"
    if isinstance(value, BaseModel):
        return "`{}`"
    return f"`{json.dumps(value)}`" if not isinstance(value, str) else f"`{value}`"


def default_text(model: type[BaseModel], name: str, field: FieldInfo) -> str:
    if field.is_required():
        defaulted = any(
            model is owner and name in defaults.model_fields
            for owner, defaults in _DEFAULTED
        )
        return "required; defaults.yaml can set it" if defaulted else "required"
    if field.default_factory is not None:
        return _value(field.default_factory())  # type: ignore[call-arg]
    return _value(field.default)


def models_of(root: type[BaseModel]) -> list[type[BaseModel]]:
    """`root` and every model its fields reach, in field order."""
    found: list[type[BaseModel]] = []

    def visit(annotation: Any) -> None:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            if annotation not in found:
                found.append(annotation)
                for field in annotation.model_fields.values():
                    visit(field.rebuild_annotation())
                extra = _extra_type(annotation)
                if extra is not None:
                    visit(extra)
            return
        for arg in get_args(annotation):
            visit(arg)

    visit(root)
    return found


def _extra_type(model: type[BaseModel]) -> Any:
    for klass in model.__mro__:
        annotation = klass.__dict__.get("__annotations__", {}).get("__pydantic_extra__")
        if annotation is not None:
            return get_args(annotation)[1]
    return None


# Markdown ---------------------------------------------------------------------


def _cell(text: str) -> str:
    """Table cell text; `<` outside code spans is escaped, or it reads as HTML."""
    parts = text.replace("|", "\\|").replace("\n", " ").split("`")
    return "`".join(
        part if index % 2 else part.replace("<", "\\<")
        for index, part in enumerate(parts)
    )


def table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return [*lines, ""]


def _model_section(model: type[BaseModel]) -> list[str]:
    lines = [
        f"### {model.__name__}",
        "",
        (model.__doc__ or "").strip().split("\n\n")[0],
    ]
    lines[-1] = " ".join(lines[-1].split())
    lines.append("")
    rows = [
        (
            f"`{field.alias or name}`",
            type_text(field.rebuild_annotation()),
            default_text(model, name, field),
            field.description or "",
        )
        for name, field in model.model_fields.items()
    ]
    extra = _extra_type(model)
    if extra is not None:
        rows.append(("other keys", type_text(extra), "", "Entries of the mapping."))
    if rows:
        lines += table(("Key", "Type", "Default", "Description"), rows)
    return lines


def _code(names: Sequence[str]) -> str:
    return ", ".join(f"`{name}`" for name in names) or "none"


def _reserved_rows() -> list[tuple[str, str, str]]:
    rows = [(f"`{name}`", "every env block", owner) for name, owner in TOOL_ENV.items()]
    rows += [
        (f"`{p}*`", "every env block", "reserved prefix") for p in TOOL_ENV_PREFIXES
    ]
    rows += [(f"`{n}`", "engines of a ray role", o) for n, o in RAY_ENV.items()]
    rows += [
        (f"`{n}`", "task env with a client driver", o)
        for n, o in CHIPLESS_DRIVER_ENV.items()
    ]
    rows += [(f"`{n}`", "`tasks.agentx.env`", o) for n, o in AGENTX_CLIENT_ENV.items()]
    rows += [
        (f"`{n}`", "`tasks.perf.env` with client inferencex", o)
        for n, o in INFERENCEX_PERF_ENV.items()
    ]
    return rows


def reference() -> str:
    lines = [
        "# Scenario reference",
        "",
        f"Generated from the models by `{REGEN}`; do not edit by hand.",
        "",
        "## Layers and merging",
        "",
        "Values come from layers, later wins: model defaults (tool) <",
        "`scenarios/defaults.yaml` < the `extends` chain, root parent first < the",
        "scenario < its sweep point < `--set`, in order. Task defaults apply only to",
        "the tasks a later layer declares. `name`, `extends` and baselines are never",
        "inherited.",
        "",
        "Mappings merge key by key, lists and scalars replace, and null deletes the",
        "key. In `server.roles.<role>.args` and `.env` a null removes the inherited",
        "server-level key for that role. `tasks.eval.lm_eval.gen_kwargs` replaces as",
        "a whole, since lm-eval reads it as one value.",
        "",
        "## Fields",
        "",
    ]
    for model in models_of(scenario.Scenario):
        lines += _model_section(model)
    lines += ["## defaults.yaml fields", ""]
    documented = set(models_of(scenario.Scenario))
    for model in models_of(scenario.Defaults):
        if model not in documented:
            lines += _model_section(model)
    lines += ["## Arg rendering", ""]
    lines += table(
        ("Value", "argv"), [(f"`{v}`" if v == "null" else v, r) for v, r in argv.RULES]
    )
    lines += ["## Validation rules", ""]
    lines += table(
        ("Rule", "Checked at", "Summary"),
        [(rule.id, rule.stage, rule.summary) for rule in rule_table()],
    )
    lines += [
        "## Derived plumbing",
        "",
        "The only values the tool derives. It never adds an engine flag or env var",
        "because of another value; choices belong in `scenarios/defaults.yaml`.",
        "",
    ]
    lines += table(
        ("Item", "What", "Rule", "Env it sets"),
        [
            (item.id, item.title, item.rule, _code(item.env))
            for item in sorted(plumbing.ITEMS.values(), key=lambda item: item.id)
        ],
    )
    lines += [
        "## Reserved names",
        "",
        "A scenario may not set these env names, since the tool sets them.",
        "",
    ]
    lines += table(("Env", "Reserved in", "Set from"), _reserved_rows())
    lines += ["Arg keys a scenario may not set:", ""]
    lines += table(
        ("Where", "Keys"),
        [
            ("`server.args`, `server.roles.<role>.args`", _code(list(SERVE_ARGS))),
            ("`server.router.args`", _code(list(ROUTER_ARGS))),
            ("`tasks.perf.client_args`", _code(list(PERF_CLIENT_ARGS))),
            ("`tasks.eval.lm_eval.model_args`", _code(list(LM_EVAL_MODEL_ARGS))),
        ],
    )
    lines += ["## Accelerators", "", "Counts are chips, never TensorCores.", ""]
    lines += table(
        ("Slice", "Hosts", "Chips per host", "Devices per host"),
        [
            (
                f"`{accel.name}/{topology}`",
                str(hosts),
                str(chips),
                str(chips * accel.devices_per_chip),
            )
            for accel in ACCELERATORS.values()
            for topology, (hosts, chips) in accel.topologies.items()
        ],
    )
    lines += [
        "## Plan format",
        "",
        f"This is Plan format {PLAN_FORMAT}; a pod refuses a Plan of another format.",
        "A Plan is one sweep point, as canonical JSON (sorted keys, compact, ASCII).",
        "Pods get it in `SCENARIO_PLAN` as base64 of gzip, at most",
        f"{codec.MAX_ENCODED_BYTES} bytes. Values only a host knows at run time are",
        "Refs:",
        "",
    ]
    lines += table(("Ref", "Filled with"), list(_ref_rows()))
    lines += table(
        ("Port", "Use"),
        [
            (str(plan_types.API_PORT), "every server's API"),
            (str(plan_types.ROUTER_PORT), "the router, on the driver host"),
            (
                str(plan_types.STATUS_PORT),
                "the driver's status, when other hosts wait on it",
            ),
            (str(plan_types.DP_RPC_PORT), "mp DP coordination"),
            (str(plan_types.RAY_PORT), "the Ray head"),
            (str(plan_types.KV_PORT), "the KV connector"),
            (str(plan_types.KV_TRANSFER_PORT), "KV transfer"),
            (str(plan_types.SIDE_CHANNEL_PORT), "the KV side channel"),
        ],
    )
    while lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"


def _ref_rows() -> Iterator[tuple[str, str]]:
    meanings = {
        "self_ip": "this host's advertise address",
        "head_ip": "the role's first host, as IPv4",
        "host_index": "this host's index in its role",
        "role_head": "the first host of a role by name; 127.0.0.1 on that host",
        "role_ips": "every host of a role as IPv4, joined by commas",
        "results_dir": "the run directory",
        "pulled_model_dir": "an object-storage model's pulled directory, or a fallback",
        "server_log": "the driver host's own server log",
    }
    for kind in get_args(plan_types.RefKind):
        yield f"`{kind}`", meanings[kind]


# Files ------------------------------------------------------------------------

GENERATORS: dict[str, Callable[[], str]] = {
    SCENARIO_SCHEMA: lambda: json_schema(scenario.Scenario, "scenario"),
    DEFAULTS_SCHEMA: lambda: json_schema(scenario.Defaults, "scenarios/defaults.yaml"),
    REFERENCE: reference,
}


def tool_dir() -> pathlib.Path:
    """tools/scenario of the checkout this package runs from.

    Raises:
        FileNotFoundError: the package is installed, not run from a checkout.
    """
    found = pathlib.Path(__file__).resolve().parents[2]
    if (
        not (found / "pyproject.toml").is_file()
        or not (found / "src" / "scenario").is_dir()
    ):
        raise FileNotFoundError(
            "scenario dev regen runs from a checkout: uv run --project tools/scenario"
        )
    return found


def stale(root: pathlib.Path) -> list[str]:
    """The generated files under `root` that differ from what the code gives."""
    return [
        name
        for name, generate in GENERATORS.items()
        if not (root / name).is_file() or (root / name).read_text() != generate()
    ]


def write(root: pathlib.Path) -> list[str]:
    """Write every generated file that changed; their names."""
    changed = stale(root)
    for name in changed:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(GENERATORS[name]())
    return changed
