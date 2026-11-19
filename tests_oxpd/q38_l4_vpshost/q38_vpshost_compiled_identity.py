#!/usr/bin/env python3
"""q38 L4 VPSHOST compile-identity census (CDK CPU pod). Usage: q38_vpshost_compiled_identity.py <tree>

The VPSHOST levers are host-only. A program a lever could change is either a function compiled with torch.compile
(decorated, or wrapped by a module-level ``torch.compile(fn)`` / ``_compiled(fn)`` style assignment) or a module-level
``*_impl`` body those wrappers compile. This prints one line per such function in the runner modules the levers touch,
with a digest of its ast dump (source identity, comments excluded), and an overall digest:

  COMPILED <module>:<name> <sha16>
  COMPILED_ALL <sha16> n=<N>

The proof requires the head lines to equal the base lines (b551e9b84), so no compiled program body differs, and that
no compiled function reads an l6_ab lever (``LEVERREAD`` lines must be absent).
"""

import ast
import hashlib
import os
import sys

TREE = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
FILES = ["src/vllm_torchtpu/runner/tpu_runner.py", "src/vllm_torchtpu/runner/vps_wiring.py",
         "src/vllm_torchtpu/layers/adapter/sample/vp_sampler.py",
         "src/vllm_torchtpu/layers/adapter/sample/rejection_sampler.py",
         "src/vllm_torchtpu/layers/adapter/sample/top_k_top_p.py"]


def compiled_names(tree):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if any("compile" in ast.unparse(d) for d in node.decorator_list):
                names.add(node.name)
            if node.name.endswith("_impl"):
                names.add(node.name)
        if isinstance(node, ast.Call) and "compile" in ast.unparse(node.func):
            for a in node.args[:1]:
                if isinstance(a, ast.Name):
                    names.add(a.id)
    return names


def main():
    rows, bad = [], 0
    for rel in FILES:
        tree = ast.parse(open(os.path.join(TREE, rel)).read())
        names = compiled_names(tree)
        mod = os.path.basename(rel)[:-3]
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
                digest = hashlib.sha1(ast.dump(node, include_attributes=False).encode()).hexdigest()[:16]
                rows.append((f"{mod}:{node.name}", digest))
                body = ast.unparse(node)
                for lever in ("_l6_ab.", "l6_ab.on(", "_l4_vpshost", "l4_vpshost"):
                    if lever in body:
                        print(f"LEVERREAD {mod}:{node.name} reads {lever}")
                        bad += 1
    rows.sort()
    for name, digest in rows:
        print(f"COMPILED {name} {digest}")
    allh = hashlib.sha1(repr(rows).encode()).hexdigest()[:16]
    print(f"COMPILED_ALL {allh} n={len(rows)}")
    return 1 if bad or not rows else 0


if __name__ == "__main__":
    sys.exit(main())
