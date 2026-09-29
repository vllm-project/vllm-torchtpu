# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Unit tests verifying strict zero-vLLM dependency isolation for llm_tpu_core."""

import ast
import subprocess
import sys
from pathlib import Path


def test_zero_vllm_imports_in_llm_tpu_core_ast():
    """Automated AST audit: verifies zero files under llm_tpu_core contain
    forbidden vllm imports.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    core_root = repo_root / "src" / "vllm_torchtpu" / "llm_tpu_core"
    assert core_root.is_dir(), f"Core directory not found: {core_root}"

    violations = []
    for py_path in core_root.rglob("*.py"):
        with open(py_path, encoding="utf-8") as f:
            try:
                tree = ast.parse(f.read(), filename=str(py_path))
            except SyntaxError as e:
                violations.append(f"SyntaxError in {py_path}: {e}")
                continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "vllm" or alias.name.startswith("vllm."):
                        violations.append(
                            f"{py_path}:{node.lineno} -> {ast.unparse(node)}"
                        )
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module
                and (node.module == "vllm" or node.module.startswith("vllm."))
            ):
                violations.append(f"{py_path}:{node.lineno} -> {ast.unparse(node)}")

    assert not violations, (
        f"Found {len(violations)} forbidden vLLM imports in llm_tpu_core:\n"
        + "\n".join(violations)
    )


def test_isolated_clean_process_import():
    """Verify importing vllm_torchtpu.llm_tpu_core leaves sys.modules free of vllm."""
    script = (
        "import sys; "
        "import vllm_torchtpu.llm_tpu_core; "
        "vllm_mods = [m for m in sys.modules if m == 'vllm' or m.startswith('vllm.')]; "
        "assert not vllm_mods, f'vllm loaded in sys.modules: {vllm_mods}'; "
        "print('PASS')"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"Clean process import test failed:\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    assert "PASS" in result.stdout
