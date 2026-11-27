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
applied to at import is the fast-dispatch wrapper, with direct dispatch
available, once its module is imported.
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


def _torch_compile_names(tree: ast.Module) -> tuple[set[str], set[str]]:
    """The names a module binds to the torch module (``import torch``,
    ``import torch as t``) and to ``torch.compile`` itself (``from torch import
    compile``, with or without ``as``)."""
    modules, functions = {"torch"}, set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(a.asname or a.name for a in node.names if a.name == "torch")
        elif isinstance(node, ast.ImportFrom) and node.module == "torch":
            functions.update(
                a.asname or a.name for a in node.names if a.name == "compile"
            )
    return modules, functions


def _is_tpu_compile_form(
    call: ast.Call, modules: set[str], functions: set[str]
) -> bool:
    """``torch.compile(..., backend="tpu", fullgraph=True, dynamic=False)``
    with no other keyword: what ``tpu_compile`` builds. ``torch.compile`` is
    recognized as ``<torch module name>.compile`` or as a name imported from
    torch (``modules`` and ``functions``, from :func:`_torch_compile_names`).
    A call that reaches it any other way (an alias assigned in the module,
    ``getattr``, keywords passed as ``**kwargs``) is not recognized."""
    func = call.func
    if isinstance(func, ast.Attribute):
        named = (
            func.attr == "compile"
            and isinstance(func.value, ast.Name)
            and func.value.id in modules
        )
    else:
        named = isinstance(func, ast.Name) and func.id in functions
    if not named:
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
        names = _torch_compile_names(tree)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and _is_tpu_compile_form(node, *names)
                and not _compiles_a_bound_method(node)
            ):
                found.append(f"{rel}:{node.lineno}")
    assert not found, (
        "compile these with compilation.decorators.tpu_compile, which the "
        f"tpu_compile options act on: {found}"
    )


# The modules that may import compilation.fast_dispatch, which is removed once
# PyTorch has a public guarded call path (its TODO): the decorator that wraps
# with it and the options that arm it. Compile sites import the decorator.
FAST_DISPATCH_IMPORTERS = {
    pathlib.Path("compilation/decorators.py"),
    pathlib.Path("compilation/compile_options.py"),
}


def _imports_fast_dispatch(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                a.name == "vllm_torchtpu.compilation.fast_dispatch" for a in node.names
            ):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.module == "vllm_torchtpu.compilation.fast_dispatch":
                return True
            if node.module == "vllm_torchtpu.compilation" and any(
                a.name == "fast_dispatch" for a in node.names
            ):
                return True
    return False


def test_only_the_decorator_and_the_options_import_fast_dispatch():
    importers = {rel for rel, tree in _sources() if _imports_fast_dispatch(tree)}
    assert importers == FAST_DISPATCH_IMPORTERS, importers ^ FAST_DISPATCH_IMPORTERS


def _uses_tpu_compile(node: ast.expr) -> bool:
    """``tpu_compile`` or ``<module>.tpu_compile``, applied or called."""
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute):
        return node.attr == "tpu_compile"
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
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and isinstance(node.value, ast.Call)
            and _uses_tpu_compile(node.value)
        ):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
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


_SPELLINGS = """
import torch as t
from torch import compile as torch_compile

t.compile(f, backend="tpu", fullgraph=True, dynamic=False)
torch_compile(f, backend="tpu", fullgraph=True, dynamic=False)
torch.compile(f, backend="tpu", fullgraph=True, dynamic=True)
compile(source, "<string>", "exec")

@tpu_compile
def decorated(x): ...

@decorators.tpu_compile
def through_the_module(x): ...

assigned = tpu_compile(f)
annotated: Callable = decorators.tpu_compile(f)

class Holder:
    @tpu_compile
    def method(self, x): ...
"""


def test_the_source_checks_recognize_each_spelling():
    tree = ast.parse(_SPELLINGS)
    names = _torch_compile_names(tree)
    plain = [
        ast.unparse(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _is_tpu_compile_form(node, *names)
    ]
    assert sorted(plain) == ["t.compile", "torch_compile"]
    assert list(_decorated_at_import(tree.body)) == [
        ("decorated",),
        ("through_the_module",),
        ("assigned",),
        ("annotated",),
        ("Holder", "method"),
    ]


# Sites whose signature takes *args or **kwargs, (module under vllm_torchtpu,
# attribute path). The wrapper cannot build the frame the guards check from
# such a call, so it turns direct dispatch off and every call goes through
# Dynamo. A change that adds such a site lists it here, so that losing direct
# dispatch is a decision rather than an accident.
THROUGH_DYNAMO: list[tuple[str, str]] = [
    ("runner.tpu_runner", "_stack_mamba_groups_compiled"),
]

DISABLED = "direct dispatch disabled"


def _why_not_dispatched_directly(obj) -> str | None:
    """None when ``obj`` is the fast-dispatch wrapper with direct dispatch
    available, else what is wrong."""
    state = getattr(obj, "_fast_dispatch_state", None)
    if state is None:
        return "not the fast-dispatch wrapper"
    if state["disabled"]:
        return DISABLED
    return None


def test_every_import_time_tpu_compile_site_is_wrapped():
    sites = _import_time_sites()
    assert sites, "no tpu_compile site found"
    wrong = []
    for module, path in sites:
        obj = importlib.import_module(module)
        for name in path:
            obj = getattr(obj, name)
        why = _why_not_dispatched_directly(obj)
        if (module.removeprefix("vllm_torchtpu."), ".".join(path)) in THROUGH_DYNAMO:
            if why == DISABLED:
                why = None
            elif why is None:
                why = "listed in THROUGH_DYNAMO, but dispatches directly"
        if why:
            wrong.append(f"{module}:{'.'.join(path)}: {why}")
    assert not wrong, wrong
    found = {(m.removeprefix("vllm_torchtpu."), ".".join(p)) for m, p in sites}
    assert set(THROUGH_DYNAMO) <= found, set(THROUGH_DYNAMO) - found


def test_the_row_copy_program_is_wrapped():
    """The one site decorated when called: the per-geometry row copy."""
    why = _why_not_dispatched_directly(mamba_state_copy_op._row_copy_fn(2, 1, 1))
    assert why is None, why


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
    ("runner.tpu_runner", "_clear_read_offsets_compiled"),
    ("runner.tpu_runner", "_update_read_offsets_verify_compiled"),
    ("runner.tpu_runner_async_output", "_compiled_extract_draft_token_ids"),
]


def test_the_functions_serving_dispatches_directly_stay_wrapped():
    assert not set(DISPATCHED_DIRECTLY) & set(THROUGH_DYNAMO)
    wrong = []
    for module, path in DISPATCHED_DIRECTLY:
        obj = importlib.import_module(f"vllm_torchtpu.{module}")
        for name in path.split("."):
            obj = getattr(obj, name)
        why = _why_not_dispatched_directly(obj)
        if why:
            wrong.append(f"{module}:{path}: {why}")
    assert not wrong, wrong
