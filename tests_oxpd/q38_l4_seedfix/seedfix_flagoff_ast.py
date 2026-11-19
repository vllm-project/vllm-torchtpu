"""Flag-off structural identity of q38 L4 SEEDFIX (VLLM_TPU_SEEDED_SAMPLING) in tpu_runner.py (torch-free).
SEEDFIX-BUILD, after SAMP2-BUILD's tests/q38_l4_samp2v/samp2_flagoff_ast.py and VFY-BUILD's vfy_flagoff_ast.py (credited).

With the flag off, seedfix.build_plan returns None, so every SEEDFIX statement in the runner is inert: the per-step plan
`self._seed_plan` is None, every `if seed_plan is not None:` / `if self._seed_plan is not None:` overlay block is
skipped, `_seedfix.warm_bucket` / `warm_rows` return at once, and the rewind's `and not _seedfix.enabled()` is True.
`off_tree(tree)` resolves that code to its flag-off arm:
  * an `if` whose test reads `seed_plan` / `_seed_plan` / `_seedfix` becomes its `else` body; in an `and` test only the
    operands that read them are removed (`generator is not None and not _seedfix.enabled()` -> `generator is not None`);
  * an assignment to `seed_plan`, `self._seed_plan` or `seed_fb_rows` is dropped;
  * an expression statement calling `_seedfix.*` (the warm-up calls) is dropped;
  * the `from vllm_torchtpu.runner import seedfix as _seedfix` import is dropped.
`check(new_src, ref_src, pre=None)`: the resolved module against the reference, top-level statement by statement and
TPUModelRunner method by method; returns the names that differ (empty = identical). `pre` is an optional source -> tree
resolver applied first (to chain other levers' resolvers)."""
import ast
import copy

NAMES = {"seed_plan", "_seed_plan", "_seedfix", "seed_fb_rows"}


def _reads(node):
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and n.attr in NAMES:
            return True
        if isinstance(n, ast.Name) and n.id in NAMES:
            return True
    return False


def _ours(t):
    if isinstance(t, ast.Name):
        return t.id in NAMES
    if isinstance(t, ast.Attribute):
        return t.attr in NAMES
    return False


class _Off(ast.NodeTransformer):
    def visit_If(self, node):
        self.generic_visit(node)
        t = node.test
        if isinstance(t, ast.BoolOp) and isinstance(t.op, ast.And) and any(_reads(v) for v in t.values):
            keep = [v for v in t.values if not _reads(v)]
            if keep:
                node.test = keep[0] if len(keep) == 1 else ast.BoolOp(op=ast.And(), values=keep)
                return node
            return node.orelse or None
        if _reads(t):
            return node.orelse or None
        return node

    def visit_Assign(self, node):
        if any(_ours(t) for t in node.targets):
            return None
        return self.generic_visit(node)

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and _reads(node.value.func):
            return None
        return self.generic_visit(node)

    def visit_ImportFrom(self, node):
        node.names = [a for a in node.names if a.asname != "_seedfix"]
        return node if node.names else None


def off_tree(tree):
    tree = _Off().visit(tree)
    ast.fix_missing_locations(tree)
    return tree


def off(src):
    return off_tree(copy.deepcopy(ast.parse(src)))


def _units(tree):
    out = {}
    for i, n in enumerate(tree.body):
        if isinstance(n, ast.ClassDef) and n.name == "TPUModelRunner":
            for m in n.body:
                key = f"TPUModelRunner.{getattr(m, 'name', type(m).__name__)}"
                out[key] = out.get(key, "") + ast.dump(m)
            out["TPUModelRunner<class>"] = ast.dump(ast.ClassDef(
                name=n.name, bases=n.bases, keywords=n.keywords, body=[], decorator_list=n.decorator_list))
        else:
            out[f"<top {i}:{getattr(n, 'name', type(n).__name__)}>"] = ast.dump(n)
    return out


def check(new_src, ref_src, pre=None):
    tree = pre(new_src) if pre is not None else copy.deepcopy(ast.parse(new_src))
    a, b = _units(off_tree(tree)), _units(ast.parse(ref_src))
    names = sorted(set(a) | set(b))
    return [f"differs: {k}" for k in names if a.get(k) != b.get(k)]
