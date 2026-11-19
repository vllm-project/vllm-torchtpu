"""Flag-off structural identity of q38 L4 AGREE (runner/l4_agree.py; torch-free). AGREE-BUILD (helper of L4-E), after
VFY-BUILD's tests/l4vfy/vfy_flagoff_ast.py and DRAFT-ASM's tests/q38_draftasm/draftasm_flagoff_ast.py (credited).

L4 AGREE adds exactly three statements to tpu_runner.py, all behind the runner's agreement, which exists only when a
lever registered (VLLM_TPU_DRAFT_ASM_FOLD or VLLM_TPU_VPS_VFY_ONE on):
  * the module import ``from vllm_torchtpu.runner import l4_agree as _l4agree``;
  * in _init_speculative_decoding, ``if self._vfy is not None: _l4agree.register(self, self._vfy)``;
  * in execute_model, ``_agree = self.__dict__.get("_l4_agree")`` and ``if _agree is not None: _agree.tick(self)``.
`off(src, extra=None)` resolves them to their flag-off arm:
  * an `if` whose test reads ``_l4_agree`` (attribute, name or string key) or the local ``_agree`` becomes its
    `else` body (none if absent);
  * an assignment to ``_agree`` / ``_l4_agree`` is dropped;
  * the import of l4_agree is dropped;
  * an `if` whose body is exactly one ``_l4agree.register(...)`` call (the guarded registration) is dropped.
An unguarded tick or an unguarded registration is NOT resolved, so the identity check bites on it. `extra` is a resolver
applied first (it returns a tree). `check(new_src, ref_src, extra=None)` returns the differences (empty = identical)."""
import ast
import copy

NAMES = {"_l4_agree", "_agree"}
ALIAS = "_l4agree"


def _reads(node):
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and n.attr in NAMES:
            return True
        if isinstance(n, ast.Name) and n.id in NAMES:
            return True
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value in NAMES:
            return True
    return False


def _ours(t):
    return (isinstance(t, ast.Name) and t.id in NAMES) or (isinstance(t, ast.Attribute) and t.attr in NAMES)


def _is_registration(stmt):
    if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
        return False
    f = stmt.value.func
    return isinstance(f, ast.Attribute) and f.attr == "register" and isinstance(f.value, ast.Name) and f.value.id == ALIAS


class _Off(ast.NodeTransformer):
    def visit_If(self, node):
        self.generic_visit(node)
        if _reads(node.test):
            return node.orelse or None
        if len(node.body) == 1 and _is_registration(node.body[0]) and not node.orelse:
            return None
        return node

    def visit_Assign(self, node):
        if any(_ours(t) for t in node.targets):
            return None
        return self.generic_visit(node)

    def visit_ImportFrom(self, node):
        node.names = [a for a in node.names if a.asname != ALIAS and a.name != "l4_agree"]
        return node if node.names else None


def off(src, extra=None):
    tree = copy.deepcopy(ast.parse(src)) if extra is None else extra(src)
    tree = _Off().visit(tree)
    ast.fix_missing_locations(tree)
    return tree


def _dump(x):
    return ast.dump(x, annotate_fields=False)


def _defs(body, prefix=""):
    out = {}
    for n in body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[prefix + n.name] = n
            if isinstance(n, ast.ClassDef):
                out.update(_defs(n.body, prefix + n.name + "."))
    return out


def check(new_src, ref_src, extra=None):
    new, ref = off(new_src, extra), ast.parse(ref_src)
    if _dump(new) == _dump(ref):
        return []
    d_new, d_ref = _defs(new.body), _defs(ref.body)
    diffs = [f"only in new: {k}" for k in sorted(set(d_new) - set(d_ref))]
    diffs += [f"only in ref: {k}" for k in sorted(set(d_ref) - set(d_new))]
    diffs += [f"differs: {k}" for k in sorted(set(d_new) & set(d_ref))
              if not isinstance(d_new[k], ast.ClassDef) and _dump(d_new[k]) != _dump(d_ref[k])]
    top_new = [_dump(n) for n in new.body if not isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    top_ref = [_dump(n) for n in ref.body if not isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    if top_new != top_ref:
        diffs.append("module-level statements differ")
    return diffs or ["differs (unlocated)"]
