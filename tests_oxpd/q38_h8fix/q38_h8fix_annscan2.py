"""H8-FIX extended string-annotation scan (extends L1's q38_l1_annscan.py; same idea, three blind spots closed).
A function's annotation is evaluated lazily from the module globals (torch_tpu jax_op: inspect.signature(eval_str=True);
typing.get_type_hints) when it is a string: every annotation in a `from __future__ import annotations` module, and quoted
annotations in any module. Flag every function (any nesting) whose string annotations use a root name that is NOT a
runtime module global:
  - L1 rule: not bound at module level at all (e.g. imported only inside the enclosing function);
  - new: bound only under `if TYPE_CHECKING:` (absent at run time);
  - new: quoted annotations in modules WITHOUT the future import (L1 skipped those modules);
  - reported separately (WARN): bound only inside a module-level `try:` (absent if the import failed).
PEP 695 type parameters of the function itself are treated as bound (L1's known false positive online_fp8.py:138).
Usage: python3 -I q38_h8fix_annscan2.py <src dir>   -> HIT / WARN lines + SUMMARY hits=N warns=M"""
import ast, builtins, os, sys

def is_type_checking(test):
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")

def names_bound(stmts, out):
    for node in stmts:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                out.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            for t in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                for x in ast.walk(t):
                    if isinstance(x, ast.Name):
                        out.add(x.id)
        elif isinstance(node, getattr(ast, "TypeAlias", ())):
            out.add(node.name.id)
        elif isinstance(node, (ast.For, ast.With)):
            names_bound(node.body, out)

def module_scopes(tree):
    runtime, tc_only, try_only = set(), set(), set()
    for node in tree.body:
        if isinstance(node, ast.If):
            if is_type_checking(node.test):
                names_bound(node.body, tc_only)
                names_bound(node.orelse, runtime)
            else:
                names_bound(node.body, runtime); names_bound(node.orelse, runtime)
        elif isinstance(node, ast.Try):
            names_bound(node.body, try_only)
            for h in node.handlers:
                names_bound(h.body, try_only)
            names_bound(node.orelse, try_only); names_bound(node.finalbody, runtime)
        else:
            names_bound([node], runtime)
    return runtime, tc_only - runtime, try_only - runtime

def roots(expr):
    out = set()
    for x in ast.walk(expr):
        if isinstance(x, ast.Name):
            out.add(x.id)
        elif isinstance(x, ast.Constant) and isinstance(x.value, str):
            try:
                out |= roots(ast.parse(x.value, mode="eval"))
            except SyntaxError:
                pass
    return out

def string_parts(expr):
    """Sub-expressions that are string constants (quoted annotations), for non-future modules."""
    return [x for x in ast.walk(expr) if isinstance(x, ast.Constant) and isinstance(x.value, str)]

B = set(dir(builtins))
root = sys.argv[1]
hits = warns = 0
for dp, ds, fs in os.walk(root):
    ds.sort()
    for f in sorted(fs):
        if not f.endswith(".py"):
            continue
        p = os.path.join(dp, f)
        src = open(p, encoding="utf-8", errors="replace").read()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        future = any(isinstance(n, ast.ImportFrom) and n.module == "__future__" and any(a.name == "annotations" for a in n.names)
                     for n in tree.body)
        runtime, tc_only, try_only = module_scopes(tree)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            tparams = {t.name for t in getattr(node, "type_params", [])}
            anns = [a.annotation for a in node.args.args + node.args.kwonlyargs + node.args.posonlyargs if a.annotation]
            for a in (node.args.vararg, node.args.kwarg):
                if a is not None and a.annotation is not None:
                    anns.append(a.annotation)
            if node.returns is not None:
                anns.append(node.returns)
            used = set()
            for a in anns:
                if future:
                    used |= roots(a)
                else:
                    for s in string_parts(a):
                        used |= roots(s)
            used -= tparams | B
            bad = {r for r in used if r not in runtime and r not in try_only}
            weak = {r for r in used if r in try_only}
            rel = os.path.relpath(p, root)
            if bad:
                hits += 1
                kind = "future" if future else "quoted"
                tc = sorted(bad & tc_only)
                print(f"HIT {rel}:{node.lineno} def {node.name} ({kind}) not-a-runtime-global {sorted(bad)}" + (f" TYPE_CHECKING-only {tc}" if tc else ""))
            if weak:
                warns += 1
                print(f"WARN {rel}:{node.lineno} def {node.name} bound-only-in-try {sorted(weak)}")
print(f"SUMMARY hits={hits} warns={warns}")
