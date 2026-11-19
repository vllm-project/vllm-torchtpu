"""Flag-off structural identity of q38 L4 DRAFT1-B (VLLM_TPU_DRAFT_ASM_CARRY_LOOP / _ONE; torch-free).

DRAFT1-B adds to eagle3.py only (draft_asm.py and tpu_runner.py are unchanged; spec_decode/draft_one.py
is new): DraftOne's construction under ``if envs.VLLM_TPU_DRAFT_ASM_CARRY_LOOP or ...ONE:`` (else: DRAFT-
ASM's DraftAsm), and two hooks whose tests read the controller's ``forward0`` / ``keep_rows`` (absent on
DraftAsm). `off(src)` resolves that code to its flag-off arm, DRAFT-ASM's resolver pattern:
  * an `if` whose test reads ``VLLM_TPU_DRAFT_ASM_CARRY_LOOP``, ``VLLM_TPU_DRAFT_ASM_ONE``, the attribute
    ``forward0`` / ``keep_rows`` or the string constant "forward0" / "keep_rows" becomes its `else` body.
`check(new_src, ref_src)` returns the differences against the reference (9b8d39a74's eagle3.py; empty =
identical with the stage flags off)."""
import ast
import copy

FLAGS = ("VLLM_TPU_DRAFT_ASM_CARRY_LOOP", "VLLM_TPU_DRAFT_ASM_ONE")
HOOKS = ("forward0", "keep_rows")
META = "meta_i32"


def _reads(node):
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and n.attr in FLAGS + HOOKS:
            return True
        if isinstance(n, ast.Constant) and n.value in HOOKS:
            return True
    return False


class _FlagOff(ast.NodeTransformer):
    def visit_If(self, node):
        self.generic_visit(node)
        if _reads(node.test):
            return node.orelse or None
        return node

    # mtp.py (X-D14-DRAFT-3 (a)): `a if self.meta_i32 else b` -> b; `self.meta_i32 = ...` dropped.
    def visit_IfExp(self, node):
        self.generic_visit(node)
        if any(isinstance(n, ast.Attribute) and n.attr == META for n in ast.walk(node.test)):
            return node.orelse
        return node

    def visit_Assign(self, node):
        if any(isinstance(t, ast.Attribute) and t.attr == META for t in node.targets):
            return None
        return self.generic_visit(node)


def off(src):
    tree = _FlagOff().visit(copy.deepcopy(ast.parse(src)))
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


def check(new_src, ref_src):
    new, ref = off(new_src), ast.parse(ref_src)
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
