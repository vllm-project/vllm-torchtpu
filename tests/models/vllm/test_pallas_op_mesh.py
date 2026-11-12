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
import ast
import pathlib
from types import SimpleNamespace

import jax
import numpy as np
from jax.sharding import Mesh

import vllm_torchtpu
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    get_pallas_op_mesh,
    set_vllm_model_wrapper_context,
)

_KERNEL_BUILDERS = ("jax_op", "custom_jax_kernel")


def test_single_device_mesh_is_dropped():
    """The per-worker mesh is one device; torch_tpu rejects it on any
    multi-chip run ("mesh size (1) must match world size (N)")."""
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape((1,)), axis_names=("model",))
    with set_vllm_model_wrapper_context(mesh=mesh):
        assert get_pallas_op_mesh() is None


def test_multi_device_mesh_is_passed_through():
    mesh = SimpleNamespace(size=4)
    with set_vllm_model_wrapper_context(mesh=mesh):
        assert get_pallas_op_mesh() is mesh


def test_no_mesh_stays_none():
    with set_vllm_model_wrapper_context(mesh=None):
        assert get_pallas_op_mesh() is None


def _call_name(call: ast.Call) -> str:
    func = call.func
    return func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")


def _is_wrapper_mesh(node: ast.expr, context_names: set[str]) -> bool:
    """`get_vllm_model_wrapper_context().mesh` or `<context var>.mesh`."""
    if not (isinstance(node, ast.Attribute) and node.attr == "mesh"):
        return False
    base = node.value
    if isinstance(base, ast.Call):
        return _call_name(base) == "get_vllm_model_wrapper_context"
    return isinstance(base, ast.Name) and base.id in context_names


def _wrapper_mesh_kernel_meshes(tree: ast.AST) -> list[int]:
    """Lines where the wrapper-context mesh is a kernel builder's `mesh=`."""
    hits = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        context_names, mesh_names = set(), set()
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if not isinstance(target, ast.Name):
                    continue
                value = node.value
                if (
                    isinstance(value, ast.Call)
                    and _call_name(value) == "get_vllm_model_wrapper_context"
                ):
                    context_names.add(target.id)
                elif _is_wrapper_mesh(value, context_names):
                    mesh_names.add(target.id)
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            if _call_name(node) not in _KERNEL_BUILDERS:
                continue
            for kw in node.keywords:
                if kw.arg != "mesh":
                    continue
                value = kw.value
                if _is_wrapper_mesh(value, context_names) or (
                    isinstance(value, ast.Name) and value.id in mesh_names
                ):
                    hits.append(kw.value.lineno)
    return sorted(set(hits))


def test_kernels_do_not_take_the_wrapper_mesh_directly():
    """Pallas ops must get the wrapper-context mesh via get_pallas_op_mesh.

    Handing it to `jax_op(mesh=...)` as is fails every multi-chip run at op
    build time, which only a multi-chip nightly notices. Passing the mesh into
    the kernel body (as GDN and MLA attention do) is not checked by torch_tpu
    and is fine.
    """
    src = pathlib.Path(vllm_torchtpu.__file__).parent
    offenders = [
        f"{path.relative_to(src)}:{line}"
        for path in sorted(src.rglob("*.py"))
        for line in _wrapper_mesh_kernel_meshes(ast.parse(path.read_text()))
    ]
    assert not offenders, offenders
