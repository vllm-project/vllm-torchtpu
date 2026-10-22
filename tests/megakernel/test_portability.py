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
"""Portability AST guard for `vllm_torchtpu.megakernel`."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

from vllm_torchtpu import megakernel


def _derive_required_modules(
    root: Path, py_files: list[Path], rel_paths: set[str]
) -> set[str]:
    """Derive required modules from package directories and relative imports."""
    required = {
        "__init__.py"
        if p.parent == root
        else f"{p.parent.relative_to(root).as_posix()}/__init__.py"
        for p in py_files
    }
    for path in py_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level >= 1:
                base = path.parent
                for _ in range(node.level - 1):
                    base = base.parent
                if not base.is_relative_to(root):
                    continue
                targets = (
                    [node.module.replace(".", "/")]
                    if node.module
                    else (
                        [alias.name for alias in node.names]
                        if path.name == "__init__.py" and node.level == 1
                        else []
                    )
                )
                for target in targets:
                    rel_t = (base / target).relative_to(root).as_posix()
                    pkg_init = f"{rel_t}/__init__.py"
                    required.add(pkg_init if pkg_init in rel_paths else f"{rel_t}.py")
    return required


def test_megakernel_portability_ast_guard():
    """Enforce relative internal imports, 0 vllm/torch deps, and jax._src allowlist.

    Checks all `.py` files under `src/vllm_torchtpu/megakernel/`:
    - Zero imports from `vllm`, `vllm_torchtpu`, or `torch`.
    - All absolute (`level == 0`) imports come strictly from Python stdlib, `jax`,
      or `numpy` (enforcing relative imports for all internal modules).
    - Private `jax._src.*` imports are restricted to `{"pool_alias.py"}`.
    - Dynamic import calls (`__import__`, `import_module`) are forbidden.
    """
    pkg_root = Path(megakernel.__file__).resolve().parent
    jax_src_allowlist = {"pool_alias.py"}
    forbidden_roots = {"vllm", "vllm_torchtpu", "torch"}
    allowed_external_roots = sys.stdlib_module_names | {"jax", "numpy"}

    py_files = sorted(pkg_root.rglob("*.py"))
    rel_paths = {p.relative_to(pkg_root).as_posix() for p in py_files}
    required_modules = _derive_required_modules(pkg_root, py_files, rel_paths)
    assert len(required_modules) > 1
    assert required_modules <= rel_paths, required_modules - rel_paths

    for path in py_files:
        rel = path.relative_to(pkg_root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                func_name = (
                    func.id
                    if isinstance(func, ast.Name)
                    else (func.attr if isinstance(func, ast.Attribute) else None)
                )
                assert func_name not in {"__import__", "import_module"}, (
                    f"{rel}:{node.lineno} calls dynamic import '{func_name}'; "
                    "use static relative imports within megakernel/."
                )
                continue

            if isinstance(node, ast.Import):
                level = 0
                modules = [alias.name for alias in node.names]
                imported_names: list[str] = []
            elif isinstance(node, ast.ImportFrom):
                level = node.level
                modules = [node.module] if node.module else []
                imported_names = [alias.name for alias in node.names]
            else:
                continue

            # `__init__.py` is the public package facade and may re-export
            # model helpers; all other core modules outside `models/` must not.
            facade_allowlist = {"__init__.py"}
            if rel not in facade_allowlist and not rel.startswith("models/"):
                for mod in modules:
                    assert mod != "models" and not mod.startswith("models."), (
                        f"{rel}:{node.lineno} imports '{mod}'; core megakernel "
                        "modules must be model-agnostic and never import models/."
                    )
                if not modules:
                    assert "models" not in imported_names, (
                        f"{rel}:{node.lineno} imports 'models'; core megakernel "
                        "modules must be model-agnostic and never import models/."
                    )

            for mod in modules:
                root = mod.split(".")[0]
                assert root not in forbidden_roots, (
                    f"{rel}:{node.lineno} imports forbidden module '{mod}'; "
                    "use relative imports and keep 0 vllm/torch dependencies."
                )
                if level == 0:
                    assert root in allowed_external_roots, (
                        f"{rel}:{node.lineno} uses non-relative or disallowed import "
                        f"'{mod}'; internal megakernel imports must be relative and "
                        "external imports must be Python stdlib, jax, or numpy."
                    )
                if mod == "jax._src" or mod.startswith("jax._src."):
                    assert rel in jax_src_allowlist, (
                        f"{rel}:{node.lineno} imports private JAX API '{mod}', "
                        f"which is restricted to allowlist {sorted(jax_src_allowlist)}."
                    )


def test_blocks_do_not_import_experimental_code():
    """Blocks reach fused pool I/O only via the core `fused_io_hook` module.

    No module under `blocks/` may import `experimental/` (or the fused helpers)
    at any level, so GQA / GDN can land without the experimental code.
    """
    blocks_root = Path(megakernel.__file__).resolve().parent / "blocks"
    py_files = sorted(blocks_root.rglob("*.py"))
    rel_paths = {p.relative_to(blocks_root).as_posix() for p in py_files}
    expected_block_files = {
        "__init__.py",
        "norm.py",
        "gqa/__init__.py",
        "gqa/block.py",
        "gqa/decls.py",
        "gqa/kv.py",
        "gqa/reference.py",
        "gdn/__init__.py",
        "gdn/block.py",
        "gdn/decls.py",
        "gdn/states.py",
        "gdn/reference.py",
        "moe/__init__.py",
        "moe/block.py",
        "moe/decls.py",
        "moe/reference.py",
    }
    assert expected_block_files <= rel_paths, expected_block_files - rel_paths
    for path in py_files:
        rel = path.relative_to(blocks_root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            # Only the library's own modules matter (relative imports or
            # `vllm_torchtpu.*`); third-party `jax.experimental` is fine.
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [
                    a.name for a in node.names if a.name.startswith("vllm_torchtpu")
                ]
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if node.level > 0 or mod.startswith("vllm_torchtpu"):
                    names = [mod] + [a.name for a in node.names]
            for name in names:
                assert "experimental" not in name and "fused_pool_io" not in name, (
                    f"blocks/{rel}:{node.lineno} imports '{name}'; blocks "
                    "must reach fused pool I/O through fused_io_hook"
                )


def test_pool_package_is_framework_free():
    """`megakernel/pool/` imports only stdlib, jax, numpy and core spec modules.

    The generic pool package must not depend on `models/` (no per-model code),
    nor on blocks, the kernel builder or the cache pipes; its only internal
    dependencies are `ops`, `errors`, `spec`, `exec_spec` and its own modules.
    """
    pool_root = Path(megakernel.__file__).resolve().parent / "pool"
    py_files = sorted(pool_root.rglob("*.py"))
    rel_paths = {p.relative_to(pool_root).as_posix() for p in py_files}
    required_pool_modules = _derive_required_modules(pool_root, py_files, rel_paths)
    assert len(required_pool_modules) > 1
    assert required_pool_modules <= rel_paths, required_pool_modules - rel_paths
    allowed_external_roots = sys.stdlib_module_names | {"jax", "numpy"}
    allowed_core = {"ops", "errors", "spec", "decls", "exec_spec"}
    for path in py_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    assert root in allowed_external_roots, (
                        f"pool/{path.name}:{node.lineno} imports '{alias.name}'"
                    )
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if node.level == 0:
                    assert mod.split(".")[0] in allowed_external_roots, (
                        f"pool/{path.name}:{node.lineno} imports '{mod}'"
                    )
                elif node.level == 2:
                    names = [mod] if mod else [a.name for a in node.names]
                    for name in names:
                        assert name.split(".")[0] in allowed_core, (
                            f"pool/{path.name}:{node.lineno} imports '..{name}'; "
                            f"pool/ may only use {sorted(allowed_core)} from core "
                            "(never models/, blocks/, kernel or cache)"
                        )
                else:
                    assert node.level == 1, (
                        f"pool/{path.name}:{node.lineno} relative import level "
                        f"{node.level} leaves the megakernel package"
                    )
