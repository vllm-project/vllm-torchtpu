"""Chained flag-off structural identity of q38-l4-v5c against q38-L1-v4 ecf4cc34a (torch-free). V5COMP-BUILD.

Each v5 piece's own flag-off resolver, applied in a chain, resolves the branch's src with every L4 v5 flag off; the
result must be ecf4cc34a's AST. The pieces' resolvers are their own (imported from their test folders); VPSHOST
(sampsnap, statsasync) and DRAFT-2 (c) / EMB-REP-TGT had none, so their resolvers are written here, each rule
naming exactly one construct the piece added:

  tpu_runner.py   P1 (VPSHOST, below) -> AGREE (l4agree_flagoff_ast) -> VRNG-VPS (samp2_flagoff_ast) -> DRAFT-ASM
                  (draftasm_flagoff_ast) -> VFY-ONE; v6 (V6COMP-BUILD): K1FUSE (k1fuse_flagoff_ast) and H3 (H3Off, below)
                  first
  vp_sampler.py   VRNG-VPS + S2 (VpOff, below)
  eagle3.py       K1FUSE -> DRAFT1-B (draft1b_flagoff_ast) -> DRAFT-ASM -> VFY-ONE
  mtp.py          H3 -> EMB (DRAFT-2 (c) / EMB-REP-TGT, below) -> DRAFT1-B (its meta_i32 switch)
  __init__.py     EMB
  l6_ab.py        the v5 / v6 LEVERS entries dropped (an unset lever parses to off: no state, no behaviour)
  envs.py         the v5 / v6 flags' annotations and getters dropped, and VFY-ONE's vfy_one_value()
  tpu_compiler.py the v5 / v6 flags dropped from _TPU_COMPILE_ENV_IGNORED
v6 (q38-l4-v6c2): spec_decode/k1fuse.py, megakernel/wholestep/qsa_draft.py and megakernel_runtime/qwen38_draft_attn.py
are new files, imported only under their flags (K1FUSE: eagle3 under VLLM_TPU_K1FUSE; H3: tpu_runner under
VLLM_TPU_MK_DRAFT_ATTN); with the flags off they never run.

Allowances (stated, each counted exactly; everything else must be identical):
  A1 mtp.py `self.bind_replicated_embedding()` in bind_shared_selection: with the flags off it still runs DRAFT-2
     (c)'s load-time agreement, one CPU-group all_gather of two int64 through l4_agree.gather on a TP > 1 draft
     (none on TP 1), and binds nothing (L4-E: the only flag-off change of that piece).
  A2 mtp.py `from vllm_torchtpu.logger import init_logger` + `logger = init_logger(__name__)` (import time only).
  A3 five docstrings gained one sentence each (P1: _vps_fallback_rows; EMB: embed, bind_shared_selection; VRNG:
     vp_verify, vp_verify_fb).
"""

import ast
import copy
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for _d in ("l4vfy", "q38_draftasm", "q38_draft1b", "q38_l4agree", "q38_l4_samp2v", "q38_lmhs0", "q38_l4_seedfix",
           "q38_k1fuse"):
    sys.path.insert(0, os.path.join(HERE, "..", _d))
import draft1b_flagoff_ast as D1  # noqa: E402
import draftasm_flagoff_ast as DA  # noqa: E402
import k1fuse_flagoff_ast as K1  # noqa: E402
import l4agree_flagoff_ast as AG  # noqa: E402
import lmhs0_flagoff_ast as LM  # noqa: E402
import samp2_flagoff_ast as S2  # noqa: E402
import seedfix_flagoff_ast as SF  # noqa: E402
import vfy_flagoff_ast as VFY  # noqa: E402

P1_LEVERS = ("sampsnap", "statsasync")
P1_DOC = {"_vps_fallback_rows": "\n        q38 L4 T-2 stage 1 SAMP-SNAP (l6_ab lever sampsnap): kept in the batch snapshot."}
EMB_DOC = {
    "embed": "        With VLLM_TPU_DRAFT_EMBED_REPL (q38 L4 X-D14-DRAFT-2 (c)) and TP > 1, the rows come from\n"
             "        the replicated table bound at load (``replicated_lookup``), with no all-reduce at all.\n",
    "bind_shared_selection": "\n\n        Also the load hook of q38 X-D14-DRAFT-2 (c): the proposer calls this once per rank after the\n"
                             "        draft borrowed the target's embedding and before anything traces the draft (eagle3\n"
                             "        ``load_model``), whatever the flags, so the replicated-embedding bind and its all-rank\n"
                             "        agreement run first (``bind_replicated_embedding``; a no-op for a TP-1 embedding).",
}
V5_LEVERS = ("sampsnap", "statsasync", "vfyone", "draftasm", "draftasmcl", "draftasmone", "k1fuse")
V5_FLAGS = (
    "VLLM_TPU_L4_SAMP_SNAP", "VLLM_TPU_L4_SAMP_SNAP_CHECK", "VLLM_TPU_L4_VPS_STATS_ASYNC",
    "VLLM_TPU_DRAFT_ASM_FOLD", "VLLM_TPU_DRAFT_ASM_CHECK_STEPS", "VLLM_TPU_DRAFT_ASM_AB",
    "VLLM_TPU_VPS_VFY_ONE", "VLLM_TPU_VPS_VFY_ONE_CHECK", "VLLM_TPU_VPS_VFY_ONE_AB", "VLLM_TPU_VPS_VFY_ONE_KEYLITE",
    "VLLM_TPU_DRAFT_ASM_CARRY_LOOP", "VLLM_TPU_DRAFT_ASM_ONE", "VLLM_TPU_DRAFT_ASM_CARRY_LOOP_AB",
    "VLLM_TPU_DRAFT_ASM_ONE_AB", "VLLM_TPU_DRAFT_EMBED_REPL", "VLLM_TPU_TARGET_EMBED_REPL",
    "VLLM_TPU_DRAFT_ASM_CAP", "VLLM_TPU_LMH_HIDDEN_MAJOR_DRAFT", "VLLM_TPU_LMH_HIDDEN_MAJOR_VERIFY",
    "VLLM_TPU_SEEDED_SAMPLING",
    "VLLM_TPU_VPS_TOPK_CUTOFF", "VLLM_TPU_VPS_INGRAPH_RNG", "VLLM_TPU_VPS_INGRAPH_RNG_AB",
    # v6 (V6COMP-BUILD): T1-K1FUSE and L1-H3 / ATTNWIRE
    "VLLM_TPU_K1FUSE", "VLLM_TPU_K1FUSE_AB", "VLLM_TPU_K1FUSE_CHECK", "VLLM_TPU_MK_DRAFT_ATTN",
)
VRNG_DOC = {
    "vp_verify": "; with `draws`\n    (q38 L4 VRNG-VPS) they are computed in-graph at those ids instead (bonus_u / accept_u / "
                 "recover_u unused)",
    "vp_verify_fb": "\n    draws: q38 L4 VRNG-VPS, (counter, full vocab, counter_uniform_at): every uniform is computed "
                    "in-graph with the\n    VPBX (full-vocabulary) semantics, whatever cand_draws says; bonus_u / accept_u / "
                    "recover_u / fb_u are unused.",
}


def _names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _on_call(node, levers):
    """node calls `_l6_ab.on('<lever>')` for one of `levers`."""
    for n in ast.walk(node):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "on"
                and isinstance(n.func.value, ast.Name) and n.func.value.id == "_l6_ab" and n.args
                and isinstance(n.args[0], ast.Constant) and n.args[0].value in levers):
            return True
    return False


def _strip_doc(fn, allowed, counts, tag):
    if fn.name not in allowed or not fn.body:
        return
    d = fn.body[0]
    if isinstance(d, ast.Expr) and isinstance(d.value, ast.Constant) and isinstance(d.value.value, str):
        if allowed[fn.name] in d.value.value:
            d.value.value = d.value.value.replace(allowed[fn.name], "", 1)
            counts[tag] = counts.get(tag, 0) + 1


class _Subst(ast.NodeTransformer):
    def __init__(self, mapping):
        self.mapping = mapping

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id in self.mapping:
            return copy.deepcopy(self.mapping[node.id])
        return node


class P1Off(ast.NodeTransformer):
    """VPSHOST stage 1 SAMP-SNAP + T-1(i) STATS-ASYNC (l6_ab levers sampsnap, statsasync) resolved off:
      * an `if` whose test calls _l6_ab.on('sampsnap' | 'statsasync') becomes its `else` body;
      * the snapshot / async-copy locals: `snap = self._samp_snap if _l6_ab.on('sampsnap') else None` and
        `pend = self._vps_fb_async` dropped (exactly these values), an `if` reading `snap` / `pend` dropped, a `for`
        over `pend.ready(...)` dropped (with statsasync off nothing is ever issued, so both are empty);
      * `self._samp_snap = _l4_vpshost.SampSnap()`, `self._vps_fb_async = _l4_vpshost.AsyncCounts()` dropped; the
        `_l4_vpshost` import dropped;
      * `_log_vps_fallback` (T-1(i)'s factored VPS_FALLBACK line) dropped and each `self._log_vps_fallback(a, b)`
        statement inlined as its body with chunks -> a, counts -> b (the base's inline statements);
      * in _vps_fallback_rows, `rows = E` directly followed by `return rows` becomes `return E`;
      * _vps_fallback_rows' added docstring sentence removed (allowance A3)."""

    def __init__(self, counts):
        self.counts = counts
        self.log_fn = None

    def resolve(self, tree):
        for n in ast.walk(tree):
            if isinstance(n, ast.FunctionDef) and n.name == "_log_vps_fallback":
                self.log_fn = n
        tree = self.visit(tree)
        tree = _Inline(self.log_fn, self.counts).visit(tree) if self.log_fn is not None else tree
        for n in ast.walk(tree):
            if isinstance(n, ast.FunctionDef) and n.name == "_vps_fallback_rows":
                b = n.body
                if (len(b) >= 2 and isinstance(b[-1], ast.Return) and isinstance(b[-1].value, ast.Name)
                        and b[-1].value.id == "rows" and isinstance(b[-2], ast.Assign)
                        and [ast.unparse(t) for t in b[-2].targets] == ["rows"]):
                    n.body = b[:-2] + [ast.Return(value=b[-2].value)]
                    self.counts["p1_return_rows"] = self.counts.get("p1_return_rows", 0) + 1
        return tree

    def visit_FunctionDef(self, node):
        if node.name == "_log_vps_fallback":
            self.counts["p1_log_fn"] = self.counts.get("p1_log_fn", 0) + 1
            return None
        _strip_doc(node, P1_DOC, self.counts, "doc_p1")
        self.generic_visit(node)
        return node

    def visit_If(self, node):
        self.generic_visit(node)
        if _on_call(node.test, P1_LEVERS) or _names(node.test) & {"snap", "pend"}:
            return node.orelse or None
        return node

    def visit_For(self, node):
        self.generic_visit(node)
        if "pend" in _names(node.iter):
            return None
        return node

    # each dropped assignment must have exactly the piece's value: an unguarded snapshot read (`snap =
    # self._samp_snap`) is a behaviour change with the lever off and must stay visible
    DROP = {
        "snap": "self._samp_snap if _l6_ab.on('sampsnap') else None",
        "pend": "self._vps_fb_async",
        "self._samp_snap": "_l4_vpshost.SampSnap()",
        "self._vps_fb_async": "_l4_vpshost.AsyncCounts()",
    }

    def visit_Assign(self, node):
        if len(node.targets) == 1 and self.DROP.get(ast.unparse(node.targets[0])) == ast.unparse(node.value):
            return None
        return self.generic_visit(node)

    def visit_ImportFrom(self, node):
        node.names = [a for a in node.names if a.asname != "_l4_vpshost"]
        return node if node.names else None


class _Inline(ast.NodeTransformer):
    def __init__(self, fn, counts):
        self.params = [a.arg for a in fn.args.args][1:]  # (self,) chunks, counts
        body = fn.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            body = body[1:]
        self.body, self.counts = body, counts

    def visit_Expr(self, node):
        c = node.value
        if (isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == "_log_vps_fallback"
                and isinstance(c.func.value, ast.Name) and c.func.value.id == "self"):
            sub = _Subst(dict(zip(self.params, c.args)))
            self.counts["p1_inlined"] = self.counts.get("p1_inlined", 0) + 1
            return [sub.visit(copy.deepcopy(s)) for s in self.body]
        return node


class EmbOff(ast.NodeTransformer):
    """DRAFT-2 (c) + EMB-REP-TGT (VLLM_TPU_DRAFT_EMBED_REPL / VLLM_TPU_TARGET_EMBED_REPL) resolved off:
      * replicated_lookup / agree_draft_embed_repl / bind_replicated_embedding dropped;
      * `self.embed_repl = ...` / `self.embed_repl_tgt = ...` dropped; an `if` reading embed_repl (attribute or
        getattr string) becomes its `else` body;
      * __init__.py patched_forward: `table = getattr(self, "_q38_repl_table", None)` dropped and the `if` reading
        `table` whose body calls replicated_lookup dropped;
      * allowances A1 (the load-time agreement call), A2 (mtp's logger), A3 (two docstrings), each counted."""

    FNS = ("replicated_lookup", "agree_draft_embed_repl", "bind_replicated_embedding")
    ATTRS = ("embed_repl", "embed_repl_tgt")

    def __init__(self, counts, logger_allowance=True):
        self.counts = counts
        self.a2 = logger_allowance  # A2 is mtp.py's only (__init__.py's base already has its own logger)

    def _bump(self, k):
        self.counts[k] = self.counts.get(k, 0) + 1

    def visit_FunctionDef(self, node):
        if node.name in self.FNS:
            return None
        _strip_doc(node, EMB_DOC, self.counts, "doc_emb")
        self.generic_visit(node)
        return node

    def visit_If(self, node):
        self.generic_visit(node)
        t = node.test
        reads = any((isinstance(n, ast.Attribute) and n.attr in self.ATTRS)
                    or (isinstance(n, ast.Constant) and n.value in self.ATTRS) for n in ast.walk(t))
        table = "table" in _names(t) and any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                                             and n.func.id == "replicated_lookup"
                                             for s in node.body for n in ast.walk(s))
        if reads or table:
            return node.orelse or None
        return node

    def visit_Assign(self, node):
        for t in node.targets:
            if isinstance(t, ast.Attribute) and t.attr in self.ATTRS:
                return None
            if isinstance(t, ast.Name) and t.id == "table" and any(
                    isinstance(n, ast.Constant) and n.value == "_q38_repl_table" for n in ast.walk(node.value)):
                return None
            if self.a2 and isinstance(t, ast.Name) and t.id == "logger" and ast.unparse(node.value) == "init_logger(__name__)":
                self._bump("A2_logger")
                return None
        return self.generic_visit(node)

    def visit_Expr(self, node):
        if ast.unparse(node.value) == "self.bind_replicated_embedding()":
            self._bump("A1_load_agreement")
            return None
        return self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if self.a2 and node.module == "vllm_torchtpu.logger" and [a.name for a in node.names] == ["init_logger"]:
            self._bump("A2_logger_import")
            return None
        return node


class VpOff(ast.NodeTransformer):
    """VRNG-VPS (VLLM_TPU_VPS_INGRAPH_RNG) + S2 (VLLM_TPU_VPS_TOPK_CUTOFF) in layers/adapter/sample/vp_sampler.py,
    resolved off (q38-l4-samp2-vfy had no AST resolver for this file; its FX-graph identity covered it):
      * S2: `_TOPK_CUTOFF` / `_F32_MAX` assignments, `_topk_mask_cutoff` / `_apply_top_k_top_p_cutoff` and the
        `topp_mask` import dropped; an `if` reading `_TOPK_CUTOFF` becomes its `else` body;
      * VRNG: `_counter_draws` / `_counter_rows` dropped; `if draws is None:` becomes its body, `if draws is not
        None:` its `else` body; the parameter `draws` (default None) and `draws=` call keywords removed;
      * the two docstring sentences VRNG added (vp_verify, vp_verify_fb) removed (allowance A3, counted)."""

    FNS = ("_topk_mask_cutoff", "_apply_top_k_top_p_cutoff", "_counter_draws", "_counter_rows")

    def __init__(self, counts):
        self.counts = counts

    def visit_FunctionDef(self, node):
        if node.name in self.FNS:
            return None
        _strip_doc(node, VRNG_DOC, self.counts, "doc_vrng")
        names = [a.arg for a in node.args.args]
        if "draws" in names:
            i = names.index("draws")
            nd = len(node.args.defaults)
            j = i - (len(names) - nd)
            if j >= 0:
                del node.args.defaults[j]
            del node.args.args[i]
        self.generic_visit(node)
        return node

    def visit_If(self, node):
        self.generic_visit(node)
        t = node.test
        if isinstance(t, ast.Compare) and isinstance(t.left, ast.Name) and t.left.id == "draws" and len(t.ops) == 1 \
                and isinstance(t.comparators[0], ast.Constant) and t.comparators[0].value is None:
            if isinstance(t.ops[0], ast.Is):
                return node.body
            if isinstance(t.ops[0], ast.IsNot):
                return node.orelse or None
        if "_TOPK_CUTOFF" in _names(t):
            return node.orelse or None
        return node

    def visit_Assign(self, node):
        if any(isinstance(x, ast.Name) and x.id in ("_TOPK_CUTOFF", "_F32_MAX") for x in node.targets):
            return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        self.generic_visit(node)
        node.keywords = [k for k in node.keywords if k.arg != "draws"]
        return node

    def visit_ImportFrom(self, node):
        if node.module and node.module.endswith("top_k_top_p"):
            node.names = [a for a in node.names if a.name != "topp_mask"]
        return node if node.names else None


class SfPlatOff(ast.NodeTransformer):
    """SEEDFIX (VLLM_TPU_SEEDED_SAMPLING, default ON) in platforms/tpu_platform.py validate_request, resolved to the flag
    set off (=0): `if not envs.VLLM_TPU_SEEDED_SAMPLING: raise ...` becomes its body (the raise), and the RANDOM_SEED
    `if` whose body then always raises turns its `elif` back into the base's following `if` (same control flow)."""

    def __init__(self, counts):
        self.counts = counts

    def visit_If(self, node):
        self.generic_visit(node)
        t = node.test
        if isinstance(t, ast.UnaryOp) and isinstance(t.op, ast.Not) and "VLLM_TPU_SEEDED_SAMPLING" in ast.unparse(t):
            self.counts["sfplat_gate"] = self.counts.get("sfplat_gate", 0) + 1
            return node.body
        if ("SamplingType.RANDOM_SEED" in ast.unparse(t) and node.body and isinstance(node.body[-1], ast.Raise)
                and len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If)):
            self.counts["sfplat_elif"] = self.counts.get("sfplat_elif", 0) + 1
            return [ast.If(test=node.test, body=node.body, orelse=[]), node.orelse[0]]
        return node


class H3Off(ast.NodeTransformer):
    """L1-H3 + ATTNWIRE (VLLM_TPU_MK_DRAFT_ATTN) resolved off (V6COMP-BUILD; neither had an AST resolver for these
    files): an `if` whose test reads VLLM_TPU_MK_DRAFT_ATTN or `mk_attn` (tpu_runner's attach after bind_and_warmup;
    mtp.py's `elif self.mk_attn is not None:` pass) becomes its `else` body; the `self.mk_attn = None` (annotated)
    assignment in the predictor's __init__ is dropped."""

    NAMES = ("VLLM_TPU_MK_DRAFT_ATTN", "mk_attn")

    def _reads(self, node):
        return any((isinstance(n, ast.Attribute) and n.attr in self.NAMES)
                   or (isinstance(n, ast.Name) and n.id in self.NAMES) for n in ast.walk(node))

    def visit_If(self, node):
        self.generic_visit(node)
        if self._reads(node.test):
            return node.orelse or None
        return node

    def visit_AnnAssign(self, node):
        if isinstance(node.target, ast.Attribute) and node.target.attr == "mk_attn":
            return None
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if any(isinstance(t, ast.Attribute) and t.attr == "mk_attn" for t in node.targets):
            return None
        return self.generic_visit(node)


class _V5Tables(ast.NodeTransformer):
    """l6_ab.py / envs.py / tpu_compiler.py: the v5 entries dropped."""

    def visit_Dict(self, node):
        self.generic_visit(node)
        keep = [(k, v) for k, v in zip(node.keys, node.values)
                if not (isinstance(k, ast.Constant) and (k.value in V5_FLAGS or k.value in V5_LEVERS))]
        node.keys, node.values = [k for k, _ in keep], [v for _, v in keep]
        return node

    def visit_Set(self, node):
        self.generic_visit(node)
        node.elts = [e for e in node.elts if not (isinstance(e, ast.Constant) and e.value in V5_FLAGS)]
        return node

    def visit_AnnAssign(self, node):
        if isinstance(node.target, ast.Name) and node.target.id in V5_FLAGS:
            return None
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if node.name == "vfy_one_value":
            return None
        self.generic_visit(node)
        return node


CHAINS = {
    "runner/tpu_runner.py": ("k1", "h3", "p1", "agree", "samp2", "sf", "lmh", "da", "vfy"),
    "platforms/tpu_platform.py": ("sfplat",),
    "runner/utils.py": ("lmh",),
    "layers/adapter/sample/vp_sampler.py": ("vp",),
    "spec_decode/eagle3.py": ("k1", "d1", "da", "vfy"),
    "models/vllm/qwen4_exp/mtp.py": ("h3", "emb", "d1"),
    "__init__.py": ("emb_init",),
    "core/l6_ab.py": ("tables",),
    "envs.py": ("tables",),
    "compilation/tpu_compiler.py": ("tables",),
}


def resolve(src, chain, counts=None):
    counts = {} if counts is None else counts
    tree = ast.parse(src)
    for step in chain:
        if step == "p1":
            tree = P1Off(counts).resolve(tree)
        elif step == "emb":
            tree = EmbOff(counts).visit(tree)
        elif step == "emb_init":
            tree = EmbOff(counts, logger_allowance=False).visit(tree)
        elif step == "agree":
            tree = AG._Off().visit(tree)
        elif step == "samp2":
            tree = S2.off_tree(tree)
        elif step == "sf":
            tree = SF.off_tree(tree)
        elif step == "sfplat":
            tree = SfPlatOff(counts).visit(tree)
        elif step == "lmh":
            defines = any(isinstance(n, ast.FunctionDef) and n.name == "arm_lm_head_hidden_major" for n in tree.body)
            tree = LM._Off(defines).visit(tree)
        elif step == "vp":
            tree = VpOff(counts).visit(tree)
        elif step == "da":
            tree = DA._FlagOff().visit(tree)
        elif step == "vfy":
            tree = VFY._Off().visit(tree)
        elif step == "k1":
            tree = K1._Off().visit(tree)
        elif step == "h3":
            tree = H3Off().visit(tree)
        elif step == "d1":
            tree = D1._FlagOff().visit(tree)
        elif step == "tables":
            tree = _V5Tables().visit(tree)
        else:
            raise ValueError(step)
        ast.fix_missing_locations(tree)
    return tree, counts


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


def check(src, ref_src, chain, counts=None):
    """[] when `src` resolved through `chain` is `ref_src`'s AST; else the differing definitions (located on the
    resolved tree itself, as the pieces' check() functions do)."""
    new, _ = resolve(src, chain, counts)
    ref = ast.parse(ref_src)
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
