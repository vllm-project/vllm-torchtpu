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
"""Unknown keys in one layer, before layers are merged.

Each file (and each --set) is checked on its own, so a typo is reported in
the file that has it, even when a later layer deletes that subtree. The walk
follows the model tree and stops at free-form mappings (args, env and the
like), whose keys the models check after merging.
"""

import inspect
import types
from collections.abc import Iterator
from typing import Annotated, Any, Union, get_args, get_origin

from pydantic import BaseModel

from scenario.errors import Issue, suggest
from scenario.model.server import Server
from scenario.model.sweep import SweepPoint
from scenario.model.tasks import Tasks
from scenario.paths import KeyPath, format_path

# Fields typed as free-form data that are overlays on a part of the tree.
_OVERLAYS: dict[tuple[type[BaseModel], str], type[BaseModel]] = {
    (SweepPoint, "server"): Server,
    (SweepPoint, "tasks"): Tasks,
}
_LISTED_KEYS = 12


def check_keys(data: Any, model: type[BaseModel], base: KeyPath = ()) -> list[Issue]:
    """Every unknown key in `data` read as `model`, with the nearest valid key."""
    return list(_walk_model(model, data, base)) if isinstance(data, dict) else []


def _leaves(annotation: Any) -> list[Any]:
    """The member types of an annotation, through unions and Annotated."""
    origin = get_origin(annotation)
    if origin is Annotated:
        return _leaves(get_args(annotation)[0])
    if origin in (Union, types.UnionType):
        return [leaf for arg in get_args(annotation) for leaf in _leaves(arg)]
    return [annotation]


def _is_model(leaf: Any) -> bool:
    return inspect.isclass(leaf) and issubclass(leaf, BaseModel)


def _aliases(model: type[BaseModel]) -> dict[str, str]:
    return {field.alias or name: name for name, field in model.model_fields.items()}


def _extra_type(model: type[BaseModel]) -> Any | None:
    if model.model_config.get("extra") != "allow":
        return None
    for klass in model.__mro__:
        annotation = inspect.get_annotations(klass).get("__pydantic_extra__")
        if annotation is not None:
            return get_args(annotation)[1]
    return Any


def _pick_model(models: list[type[BaseModel]], data: dict) -> type[BaseModel]:
    """The member model that knows the most of `data`'s keys."""
    return max(models, key=lambda model: len(set(_aliases(model)) & set(data)))


def _walk_value(annotation: Any, value: Any, path: KeyPath) -> Iterator[Issue]:
    if value is None:
        return
    leaves = _leaves(annotation)
    if isinstance(value, dict):
        models = [leaf for leaf in leaves if _is_model(leaf)]
        if models:
            yield from _walk_model(_pick_model(models, value), value, path)
            return
        for leaf in leaves:
            if get_origin(leaf) is dict:
                item_type = get_args(leaf)[1]
                for key, item in value.items():
                    yield from _walk_value(item_type, item, (*path, key))
                return
    elif isinstance(value, list):
        for leaf in leaves:
            if get_origin(leaf) is list:
                (item_type,) = get_args(leaf)
                for index, item in enumerate(value):
                    yield from _walk_value(item_type, item, (*path, index))
                return


def _walk_model(model: type[BaseModel], data: dict, path: KeyPath) -> Iterator[Issue]:
    aliases = _aliases(model)
    extra = _extra_type(model)
    for key, value in data.items():
        where = (*path, key)
        name = aliases.get(key)
        if name is None:
            if extra is not None:
                yield from _walk_value(extra, value, where)
            else:
                yield _unknown(where, list(aliases))
            continue
        overlay = _OVERLAYS.get((model, name))
        if overlay is not None:
            yield from _walk_value(overlay, value, where)
        else:
            yield from _walk_value(model.model_fields[name].annotation, value, where)


def _unknown(path: KeyPath, known: list[str]) -> Issue:
    key = path[-1]
    near = suggest(str(key), known)
    if near is not None:
        return Issue(
            path, f"unknown key; did you mean {format_path((*path[:-1], near))}?"
        )
    listed = ", ".join(known[:_LISTED_KEYS])
    more = ", ..." if len(known) > _LISTED_KEYS else ""
    return Issue(path, f"unknown key; known keys here: {listed}{more}")
