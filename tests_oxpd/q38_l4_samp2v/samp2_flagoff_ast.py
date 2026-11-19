"""Flag-off structural identity of q38 L4 VRNG-VPS (VLLM_TPU_VPS_INGRAPH_RNG) on the composed v5 tree (torch-free).
SAMP2-BUILD, after VFY-BUILD's tests/l4vfy/vfy_flagoff_ast.py and DRAFT-ASM's draftasm_flagoff_ast.py (credited).

VRNG only adds code behind its controller: tpu_runner's `self._vrng` (None unless the flag is on), read through
`self._vrng` / `getattr(self, "_vrng", None)` / the locals `vrng`, `use_vrng`, `rng`, `spec_early`, `vrng_counter`,
`vrng_programs`, the `_l4_samp2` module alias, the parameters `vrng` / `force_full`, and new functions / methods named
`_vps_verify_rng*` / `_vrng*` / `_precompile_vps_vrng` / `_vps_verify_vrng` (the `_vfy*` ones are VFY-ONE's resolver's).
`off_tree(tree)` resolves that code to its flag-off arm:
  * an `if` whose test reads one of those names becomes its `else` body; in an `and` test, only the operands that read
    them are removed (`x and not use_vrng` -> `x`);
  * an IfExp whose test reads them becomes its `orelse`;
  * an assignment to one of them, or to a name with one of the prefixes, is dropped;
  * a function definition with one of the prefixes is dropped; the parameters `vrng` / `force_full` and their
    defaults are removed; call arguments / keywords naming them are removed; the `_l4_samp2` import is dropped.
`compose(*resolvers)` chains source -> tree resolvers for vfy_flagoff_ast.check's `extra`."""
import ast
import copy

FLAG = "VLLM_TPU_VPS_INGRAPH_RNG"
NAMES = {"_vrng", "use_vrng", "vrng", "rng", "spec_early", "vrng_counter", "vrng_programs", "_l4_samp2", "force_full"}
PREFIXES = ("_vps_verify_rng", "_vrng", "_precompile_vps_vrng", "_vps_verify_vrng")
PARAMS = {"vrng", "force_full"}


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
        return t.id in NAMES or t.id.startswith(PREFIXES)
    if isinstance(t, ast.Attribute):
        return t.attr in NAMES or t.attr.startswith(PREFIXES)
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

    def visit_IfExp(self, node):
        self.generic_visit(node)
        return node.orelse if _reads(node.test) else node

    def visit_Assign(self, node):
        if any(_ours(t) for t in node.targets):
            return None
        return self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if _ours(node.target):
            return None
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if node.name.startswith(PREFIXES):
            return None
        a = node.args
        nd = len(a.defaults)
        first_default = len(a.args) - nd
        args, defaults = [], []
        for i, arg in enumerate(a.args):
            if arg.arg in PARAMS:
                continue
            args.append(arg)
            if i >= first_default:
                defaults.append(a.defaults[i - first_default])
        a.args, a.defaults = args, defaults
        return self.generic_visit(node)

    def visit_Call(self, node):
        self.generic_visit(node)
        node.args = [x for x in node.args if not (isinstance(x, ast.Name) and x.id in NAMES)]
        node.keywords = [k for k in node.keywords if k.arg not in PARAMS]
        return node

    def visit_ImportFrom(self, node):
        node.names = [a for a in node.names if a.asname != "_l4_samp2" and a.name != "l4_samp2"]
        return node if node.names else None


def off_tree(tree):
    tree = _Off().visit(tree)
    ast.fix_missing_locations(tree)
    return tree


def off(src):
    return off_tree(copy.deepcopy(ast.parse(src)))


def compose(*resolvers):
    """A source -> tree resolver: the first resolver of `resolvers` takes the source, each next one the tree."""
    def run(src):
        tree = resolvers[0](src)
        for r in resolvers[1:]:
            tree = r(tree)
        return tree
    return run
