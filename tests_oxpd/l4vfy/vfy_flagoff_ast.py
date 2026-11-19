"""Flag-off structural identity of q38 L4 VFY-ONE (VLLM_TPU_VPS_VFY_ONE; torch-free). VFY-BUILD, after DRAFT-ASM's
tests/q38_draftasm/draftasm_flagoff_ast.py (credited).

VFY-ONE only adds code behind its controller: tpu_runner's `self._vfy` (None unless the flag is on), read through
`self._vfy` / `self.__dict__.get("_vfy")` / a local `_vfy`, the VfyChunk gate `vfy_defer`, the DH-C stash
`_vfy_seed`, the `_vfy_one` module alias, and new functions / methods named `_vfy*` / `vfy_*`. `off(src)` resolves
that code to its flag-off arm:
  * an `if` whose test reads one of those names (an attribute, a name or the string key) becomes its `else` body;
  * an assignment to one of them (or to any `_vfy*` name / attribute) is dropped;
  * a function definition named `_vfy*` / `vfy_*` is dropped; `from ... import vfy_one as _vfy_one` is dropped.
`check(new_src, ref_src, extra=None)` returns the differences against the reference module (empty = identical with
the flag off); `extra` is another resolver applied first (DRAFT-ASM's `off` on the composed tree)."""
import ast
import copy

FLAG = "VLLM_TPU_VPS_VFY_ONE"
NAMES = {"_vfy", "_vfy_one", "vfy_defer", "_vfy_seed"}


def _reads(node):
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and (n.attr in NAMES or n.attr == FLAG):
            return True
        if isinstance(n, ast.Name) and n.id in NAMES:
            return True
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value in NAMES:
            return True
    return False


def _ours(t):
    if isinstance(t, ast.Name):
        return t.id in NAMES or t.id.startswith("_vfy")
    if isinstance(t, ast.Attribute):
        return t.attr in NAMES or t.attr.startswith("_vfy")
    return False


class _Off(ast.NodeTransformer):
    def visit_If(self, node):
        self.generic_visit(node)
        if _reads(node.test):
            return node.orelse or None
        return node

    def visit_Assign(self, node):
        if any(_ours(t) for t in node.targets):
            return None
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if node.name.startswith("_vfy") or node.name.startswith("vfy_"):
            return None
        return self.generic_visit(node)

    def visit_ImportFrom(self, node):
        node.names = [a for a in node.names if a.asname != "_vfy_one" and a.name != "vfy_one"]
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
    cls_new = {k: [_dump(n) for n in v.body if not isinstance(n, ast.FunctionDef)]
               for k, v in d_new.items() if isinstance(v, ast.ClassDef)}
    cls_ref = {k: [_dump(n) for n in v.body if not isinstance(n, ast.FunctionDef)]
               for k, v in d_ref.items() if isinstance(v, ast.ClassDef)}
    diffs += [f"class-level statements differ: {k}" for k in sorted(set(cls_new) & set(cls_ref))
              if cls_new[k] != cls_ref[k]]
    return diffs or ["differs (unlocated)"]
