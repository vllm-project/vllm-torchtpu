# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Jinja namespace aliases in chat-template content-format detection.

Backport from https://github.com/vllm-project/vllm/pull/62622 of
``_is_var_access``, ``_attr_dotted_name`` and
``_iter_nodes_assign_var_or_elems`` in ``vllm/renderers/hf.py``. The pinned
detection follows ``messages`` only through plain ``{% set x = ... %}``
assignments, so a template that iterates messages held in a Jinja namespace is
detected as the string content format. Remove this module and its ``PATCHES``
entry once the vLLM pin includes that PR.
"""

from collections import deque

import jinja2
import jinja2.nodes

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def _is_var_access(node: jinja2.nodes.Node, varname: str) -> bool:
    if isinstance(node, jinja2.nodes.Name):
        return node.ctx == "load" and node.name == varname

    # `ns.rest` matches the dotted alias "ns.rest".
    if isinstance(node, jinja2.nodes.Getattr) and "." in varname:
        base, _, attr = varname.rpartition(".")
        return node.attr == attr and _is_var_access(node.node, base)

    return False


def _attr_dotted_name(node: jinja2.nodes.Node) -> str | None:
    """Render an attribute-assignment target like ``ns.rest`` as a dotted name."""
    if isinstance(node, jinja2.nodes.NSRef):
        return f"{node.name}.{node.attr}"
    parts = []
    while isinstance(node, jinja2.nodes.Getattr):
        parts.append(node.attr)
        node = node.node
    if isinstance(node, jinja2.nodes.Name):
        parts.append(node.name)
        return ".".join(reversed(parts))
    return None


def _iter_nodes_assign_var_or_elems(root: jinja2.nodes.Node, varname: str):
    from vllm.renderers.hf import _is_var_or_elems_access

    # Global variable that is implicitly defined at the root
    yield root, varname

    # Iterative BFS
    seen = {varname}
    related_varnames = deque([varname])
    while related_varnames:
        related_varname = related_varnames.popleft()

        for assign_ast in root.find_all(jinja2.nodes.Assign):
            lhs = assign_ast.target
            rhs = assign_ast.node

            if isinstance(lhs, jinja2.nodes.Name) and _is_var_or_elems_access(
                rhs, related_varname
            ):
                yield assign_ast, lhs.name

                # Avoid infinite looping for self-assignment
                if lhs.name != related_varname and lhs.name not in seen:
                    seen.add(lhs.name)
                    related_varnames.append(lhs.name)

            # Namespace aliases: `namespace(rest=messages)` or `ns.rest = messages`.
            aliases: list[str] = []
            if (
                isinstance(lhs, jinja2.nodes.Name)
                and isinstance(rhs, jinja2.nodes.Call)
                and isinstance(rhs.node, jinja2.nodes.Name)
                and rhs.node.name == "namespace"
            ):
                aliases = [
                    f"{lhs.name}.{kwarg.key}"
                    for kwarg in rhs.kwargs
                    if _is_var_or_elems_access(kwarg.value, related_varname)
                ]
            elif isinstance(
                lhs, (jinja2.nodes.Getattr, jinja2.nodes.NSRef)
            ) and _is_var_or_elems_access(rhs, related_varname):
                dotted = _attr_dotted_name(lhs)
                aliases = [dotted] if dotted is not None else []

            for alias in aliases:
                yield assign_ast, alias
                if alias != related_varname and alias not in seen:
                    seen.add(alias)
                    related_varnames.append(alias)


def patch_chat_template_namespace_aliases() -> None:
    """Follow Jinja namespace aliases of ``messages`` in content-format detection."""
    from vllm.renderers import hf

    hf._is_var_access = _is_var_access
    hf._iter_nodes_assign_var_or_elems = _iter_nodes_assign_var_or_elems
    logger.info("Applied TPU patch: chat-template namespace aliases (vLLM #62622).")
