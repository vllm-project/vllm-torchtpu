# SPDX-License-Identifier: Apache-2.0
"""Static analysis test to catch init-order bugs in TPUModelRunner.__init__."""

import ast
from pathlib import Path
from typing import Set, Dict, List, Tuple

import pytest


def get_runner_source(commit: str = "HEAD") -> str:
    """Get the tpu_runner.py source from a specific commit or HEAD."""
    import subprocess

    repo_root = Path(__file__).parent.parent.parent
    if commit == "HEAD":
        runner_path = repo_root / "src" / "vllm_torchtpu" / "runner" / "tpu_runner.py"
        return runner_path.read_text()
    else:
        # Get from git history
        result = subprocess.run(
            ["git", "show", f"{commit}:src/vllm_torchtpu/runner/tpu_runner.py"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout


class InitOrderChecker:
    """Analyzes __init__ method for attribute access before assignment bugs."""

    # Known safe cases: these attributes are legitimately accessed before
    # assignment in __init__ via safe guards. Add entries here only after
    # verifying the access is properly guarded.
    ALLOWLIST: List[Tuple[str, str, int]] = [
        # (method_name, attribute_name, approx_line): Comment explaining why it's safe
    ]

    def __init__(self, source: str):
        self.tree = ast.parse(source)
        self.source_lines = source.splitlines()

    def find_class_init(self, class_name: str) -> ast.FunctionDef:
        """Find the __init__ method of a class."""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                        return item
        raise ValueError(f"Could not find {class_name}.__init__")

    def get_self_assignments(self, init_method: ast.FunctionDef) -> Dict[str, int]:
        """Get all self.X = ... assignments in __init__ with their line numbers."""
        assignments = {}
        for node in ast.walk(init_method):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute):
                        if isinstance(target.value, ast.Name) and target.value.id == "self":
                            assignments[target.attr] = node.lineno
        return assignments

    def get_self_method_calls(self, init_method: ast.FunctionDef) -> List[Tuple[str, int]]:
        """Get all self.method(...) calls in __init__ with their line numbers."""
        calls = []
        for stmt in init_method.body:
            for node in ast.walk(stmt):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        if isinstance(node.func.value, ast.Name) and node.func.value.id == "self":
                            calls.append((node.func.attr, node.lineno))
        return calls

    def find_method(self, class_name: str, method_name: str) -> ast.FunctionDef:
        """Find a method in a class."""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == method_name:
                        return item
        return None

    def get_self_attr_reads(
        self, method: ast.FunctionDef, depth: int = 1
    ) -> Set[str]:
        """Get all self.X attribute reads in a method.

        Excludes reads that are guarded by getattr, hasattr, or self.__dict__.get.
        Recursively checks self.method() calls up to depth levels.
        """
        if method is None or depth < 0:
            return set()

        reads = set()

        # Walk the method body
        for node in ast.walk(method):
            # Check for self.attr reads
            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name) and node.value.id == "self":
                    # Check if this is inside a safe guard
                    if not self._is_guarded_access(node, method):
                        reads.add(node.attr)

            # Recursively check self.method() calls
            if depth > 0 and isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name) and node.func.value.id == "self":
                        # Find and analyze the called method
                        called_method = self.find_method("TPUModelRunner", node.func.attr)
                        if called_method:
                            reads.update(self.get_self_attr_reads(called_method, depth - 1))

        return reads

    def _is_guarded_access(self, attr_node: ast.Attribute, method: ast.FunctionDef) -> bool:
        """Check if an attribute access is guarded by getattr/hasattr/__dict__.get."""
        # Walk up the tree to find the enclosing Call node
        for node in ast.walk(method):
            if isinstance(node, ast.Call):
                # Check for getattr(self, "attr", default)
                if isinstance(node.func, ast.Name) and node.func.id == "getattr":
                    if len(node.args) >= 2:
                        if node.args[0] is attr_node.value:
                            return True

                # Check for hasattr(self, "attr")
                if isinstance(node.func, ast.Name) and node.func.id == "hasattr":
                    if len(node.args) >= 2:
                        if node.args[0] is attr_node.value:
                            return True

                # Check for self.__dict__.get("attr", ...)
                if isinstance(node.func, ast.Attribute) and node.func.attr == "get":
                    if isinstance(node.func.value, ast.Attribute):
                        if (isinstance(node.func.value.value, ast.Name) and
                            node.func.value.value.id == "self" and
                            node.func.value.attr == "__dict__"):
                            return True

        return False

    def check_init_order(self, class_name: str = "TPUModelRunner") -> List[Tuple[str, str, int]]:
        """Check for init-order violations.

        Returns a list of (method_name, attribute_name, line_number) tuples
        for attributes that are accessed before being assigned.
        """
        init_method = self.find_class_init(class_name)
        assignments = self.get_self_assignments(init_method)
        method_calls = self.get_self_method_calls(init_method)

        violations = []

        for method_name, call_line in method_calls:
            method = self.find_method(class_name, method_name)
            if method is None:
                continue

            # Get all self.attr reads in this method (and one level of called methods)
            attr_reads = self.get_self_attr_reads(method, depth=1)

            for attr in attr_reads:
                # Skip if not assigned in __init__ (might be from parent class or dynamic)
                if attr not in assignments:
                    continue

                # Check if the attribute is assigned before this method call
                if assignments[attr] > call_line:
                    violation = (method_name, attr, call_line)
                    # Check if this is in the allowlist
                    if not any(
                        v[0] == method_name and v[1] == attr
                        for v in self.ALLOWLIST
                    ):
                        violations.append(violation)

        return violations


@pytest.mark.parametrize(
    "commit,expected_violations",
    [
        ("HEAD", 0),  # Current code should pass
        ("d9d29adc29", 1),  # Bug commit should fail
    ],
)
def test_tpu_runner_init_order(commit, expected_violations):
    """Test that TPUModelRunner.__init__ doesn't use attributes before assigning them.

    Regression test for the bug in d9d29adc29 where _refresh_smem_request_caps()
    was called before self._steady_decode_cache was assigned, causing AttributeError.
    """
    source = get_runner_source(commit)
    checker = InitOrderChecker(source)
    violations = checker.check_init_order()

    if expected_violations == 0:
        if violations:
            msg = "Found init-order violations:\n"
            for method, attr, line in violations:
                msg += f"  - {method}() at line {line} accesses self.{attr} before assignment\n"
            pytest.fail(msg)
    else:
        assert len(violations) >= expected_violations, (
            f"Expected at least {expected_violations} violations, "
            f"found {len(violations)}: {violations}"
        )


def test_init_order_checker_on_head():
    """Standalone test on HEAD - reports any findings for review."""
    source = get_runner_source("HEAD")
    checker = InitOrderChecker(source)
    violations = checker.check_init_order()

    # This test always passes but reports findings
    if violations:
        print("\n=== Init-order findings on HEAD ===")
        for method, attr, line in violations:
            print(f"  {method}() at ~line {line} reads self.{attr} before assignment")
        print(f"Total: {len(violations)} findings")
        print("If these are false positives, add to ALLOWLIST with explanation.")
