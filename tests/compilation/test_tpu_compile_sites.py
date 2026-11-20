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
"""The package's compiled TPU functions go through ``tpu_compile``.

A function compiled with a plain ``torch.compile(backend="tpu",
fullgraph=True, dynamic=False)`` silently loses the ``tpu_compile`` options,
so these tests read the package's source: no such call is left outside the
places that may not use the decorator, and every function the decorator is
applied to at import is the fast-dispatch wrapper once its module is imported.
"""

import ast
import importlib
import pathlib

import vllm_torchtpu
from vllm_torchtpu.layers.adapter.custom_ops import mamba_state_copy_op

PACKAGE = pathlib.Path(vllm_torchtpu.__file__).parent

# Package directories whose compiled functions stay on plain torch.compile.
PLAIN_TORCH_COMPILE = {
    # The coding guidance (R-2.1) keeps kernels/ free of the package's other
    # layers, compilation/ included.
    "kernels",
}


def _sources():
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(PACKAGE)
        yield rel, ast.parse(path.read_text(), filename=str(path))


def _is_tpu_compile_form(call: ast.Call) -> bool:
    """``torch.compile(..., backend="tpu", fullgraph=True, dynamic=False)``
    with no other keyword: what ``tpu_compile`` builds."""
    func = call.func
    if not (
        isinstance(func, ast.Attribute)
        and func.attr == "compile"
        and isinstance(func.value, ast.Name)
        and func.value.id == "torch"
    ):
        return False
    kwargs = {
        k.arg: k.value.value if isinstance(k.value, ast.Constant) else k.value
        for k in call.keywords
    }
    return kwargs == {"backend": "tpu", "fullgraph": True, "dynamic": False}


def _compiles_a_bound_method(call: ast.Call) -> bool:
    """``torch.compile(self.<...>, ...)``: tpu_compile takes plain functions."""
    target = call.args[0] if call.args else None
    if not isinstance(target, ast.Attribute):
        return False
    while isinstance(target, ast.Attribute):
        target = target.value
    return isinstance(target, ast.Name) and target.id == "self"


def test_no_plain_torch_compile_of_a_tpu_function():
    found = []
    for rel, tree in _sources():
        if rel.parts[0] in PLAIN_TORCH_COMPILE:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and _is_tpu_compile_form(node)
                and not _compiles_a_bound_method(node)
            ):
                found.append(f"{rel}:{node.lineno}")
    assert not found, (
        "compile these with compilation.fast_dispatch.tpu_compile, which the "
        f"tpu_compile options act on: {found}"
    )


def _uses_tpu_compile(node: ast.expr) -> bool:
    if isinstance(node, ast.Call):
        node = node.func
    return isinstance(node, ast.Name) and node.id == "tpu_compile"


def _decorated_at_import(body, prefix=()):
    """Attribute paths of the functions decorated with, or assigned the result
    of, tpu_compile in a module or class body (not inside a function)."""
    for node in body:
        if isinstance(node, ast.ClassDef):
            yield from _decorated_at_import(node.body, (*prefix, node.name))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if any(_uses_tpu_compile(d) for d in node.decorator_list):
                yield (*prefix, node.name)
        elif (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and _uses_tpu_compile(node.value)
        ):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    yield (*prefix, target.id)


def _import_time_sites():
    sites = []
    for rel, tree in _sources():
        parts = rel.with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        module = ".".join(("vllm_torchtpu", *parts))
        sites += [(module, path) for path in _decorated_at_import(tree.body)]
    return sites


def test_every_import_time_tpu_compile_site_is_wrapped():
    sites = _import_time_sites()
    assert sites, "no tpu_compile site found"
    unwrapped = []
    for module, path in sites:
        obj = importlib.import_module(module)
        for name in path:
            obj = getattr(obj, name)
        if not hasattr(obj, "_fast_dispatch_state"):
            unwrapped.append(f"{module}:{'.'.join(path)}")
    assert not unwrapped, unwrapped


def test_the_row_copy_program_is_wrapped():
    """The one site decorated when called: the per-geometry row copy."""
    assert hasattr(mamba_state_copy_op._row_copy_fn(2, 1, 1), "_fast_dispatch_state")


# The functions the Qwen3.5-397B serving configuration dispatches directly with
# both tpu_compile options on: (module under vllm_torchtpu, attribute path). A
# decorator deleted from one of them loses the options for it without leaving a
# plain torch.compile behind, which the source-reading tests above do not see.
# Changes that add such functions extend this list.
DISPATCHED_DIRECTLY = [
    ("layers.adapter.custom_ops.mamba_state_copy_op", "copy_mamba_state_blocks"),
    (
        "layers.adapter.sample.rejection_sampler",
        "_random_rejection_sample_with_segment",
    ),
    ("layers.core.attention_metadata", "_staged_walk_products_compiled"),
    ("runner.min_tokens", "mask_stop_tokens"),
    ("runner.min_tokens", "restore_all_masked_rows"),
    ("runner.spec_boundary_state", "_spec_boundary_pairs_compiled"),
    ("runner.tpu_runner", "_reset_read_offsets_compiled"),
    ("runner.tpu_runner", "_rollback_offsets_seed_compiled"),
    ("runner.tpu_runner", "_rollback_offsets_migrate_compiled"),
    ("runner.tpu_runner", "TPUModelRunner.compute_selected_logits"),
    ("runner.tpu_runner", "TPUModelRunner.compute_selected_top_tokens"),
    ("runner.tpu_runner", "TPUModelRunner.gather_logprobs"),
    ("runner.tpu_runner", "TPUModelRunner.sample_from_logits"),
    ("runner.tpu_runner", "TPUModelRunner.spec_gather_bonus_and_target_logits"),
    ("spec_decode.eagle3", "Eagle3Proposer._draft_gather_carries"),
    ("spec_decode.eagle3", "Eagle3Proposer._draft_propose_token"),
    ("spec_decode.eagle3", "Eagle3Proposer._draft_seed_input_ids"),
    (
        "structured_decoding.structured_decoding_manager",
        "StructuredDecodingManager.structured_decode",
    ),
]


def test_the_functions_serving_dispatches_directly_stay_wrapped():
    unwrapped = []
    for module, path in DISPATCHED_DIRECTLY:
        obj = importlib.import_module(f"vllm_torchtpu.{module}")
        for name in path.split("."):
            obj = getattr(obj, name)
        if not hasattr(obj, "_fast_dispatch_state"):
            unwrapped.append(f"{module}:{path}")
    assert not unwrapped, unwrapped
