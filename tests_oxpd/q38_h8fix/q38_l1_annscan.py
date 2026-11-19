"""L1 string-annotation scan: in modules with `from __future__ import annotations`, flag every function (any nesting)
whose parameter / return annotations use a root name that is not bound at module level (so typing.get_type_hints on it,
as torch_tpu jax_op registration may do, raises NameError). Usage: python3 -I q38_l1_annscan.py <src dir>"""
import ast, builtins, os, sys

def module_names(tree):
    n = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                n.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            n.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            for t in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                for x in ast.walk(t):
                    if isinstance(x, ast.Name):
                        n.add(x.id)
        elif isinstance(node, (ast.If, ast.Try)):
            for x in ast.walk(node):
                if isinstance(x, (ast.Import, ast.ImportFrom)):
                    for a in x.names:
                        n.add((a.asname or a.name).split(".")[0])
                elif isinstance(x, (ast.FunctionDef, ast.ClassDef)):
                    n.add(x.name)
                elif isinstance(x, ast.Assign):
                    for t in x.targets:
                        for y in ast.walk(t):
                            if isinstance(y, ast.Name):
                                n.add(y.id)
    return n

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

B = set(dir(builtins))
root = sys.argv[1]
hits = 0
for dp, _, fs in os.walk(root):
    for f in fs:
        if not f.endswith(".py"):
            continue
        p = os.path.join(dp, f)
        src = open(p, encoding="utf-8", errors="replace").read()
        if "from __future__ import annotations" not in src:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        mod = module_names(tree)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) or isinstance(node, ast.Lambda):
                continue
            anns = [a.annotation for a in node.args.args + node.args.kwonlyargs + node.args.posonlyargs if a.annotation]
            for a in (node.args.vararg, node.args.kwarg):
                if a is not None and a.annotation is not None:
                    anns.append(a.annotation)
            if node.returns is not None:
                anns.append(node.returns)
            bad = set()
            for a in anns:
                bad |= {r for r in roots(a) if r not in mod and r not in B}
            if bad:
                hits += 1
                print(f"HIT {os.path.relpath(p, root)}:{node.lineno} def {node.name} unbound-at-module {sorted(bad)}")
print(f"SUMMARY hits={hits}")
