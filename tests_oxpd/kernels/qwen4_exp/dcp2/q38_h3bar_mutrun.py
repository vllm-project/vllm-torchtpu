#!/usr/bin/env python3
"""q38 L4 v6 H3-bar mutants (CDK CPU pod only). V6COMP-BUILD. Usage: q38_h3bar_mutrun.py <head tree> <plugins dir>

The prereg's mutants (reports/prereg/q38_l4_h3_v6_bar_DRAFT.md sections 3 and 6.3) against the bar's own node
(test_v6c_h3bar.py::test_cp_bar[2-1-8], p = 8, R = 2, D = 1, 8 steps, meta on and off):
  M-elt / M-agg / M-elt-hi: test-side (Q38_H3BAR_MUT=elt | agg | elthi on the recorded kernel output, h3bar_harness);
  A1 / A2 / A4: ATTNWIRE's product mutants (q38_attnwire_mutants.py's exact texts) on a copy of the tree.
Each runs the node in ATTNWIRE's environment (VLLM_TARGET_DEVICE=cpu, L1's q38_cpu_device plugin); KILLED = pytest exit 1
AND the pinned reason in the output; a clean copy first runs the node and must pass (control). 4 at a time.
Lines: CONTROL rc=..., MUT <name> KILLED|SURVIVED|BAD rc=... reason=<found|missing>, last
Q38_H3BAR_MUT PASS|FAIL killed K/N control_ok=<0|1> (argv[3]: an optional comma-separated subset of names).
"""

import concurrent.futures
import os
import re
import shutil
import subprocess
import sys
import tempfile

TARGET = "tests/kernels/qwen4_exp/dcp2/test_v6c_h3bar.py::test_cp_bar[2-1-8]"
DA = "src/vllm_torchtpu/megakernel_runtime/qwen38_draft_attn.py"
MTP = "src/vllm_torchtpu/models/vllm/qwen4_exp/mtp.py"
MUTS = [
    # addendum A1: the gated form of leg B is binding (the harness default); M-elt-hi perturbs by 2 x the gated bound
    ("M-elt", None, "elt", r"LEG B FAIL \(form tight\)"),
    ("M-agg", None, "agg", r"LEG C FAIL"),
    ("M-elt-hi", None, "elthitight", r"LEG B FAIL \(form tight\)"),
    # the literal (u S) record: M-elt-hi with the literal perturbation asserted under the literal form
    ("M-elt-hi-literal", None, "elthi:literal", r"LEG B FAIL \(form literal\)"),
    ("M-elt-literal", None, "elt:literal", r"LEG B FAIL \(form literal\)"),
    # the lead's mutant: beyond the gated bound, inside the literal one -> the locked (gated) leg B fails, literal 0
    ("M-gated-only", None, "gatedonly", r"LEG B FAIL \(form tight\) .*\(literal 0, gated 1;"),
    ("A1", (MTP, "            if rows is not None:\n                rows = rows.to(torch.int64)\n",
            "            if rows is not None and self.mk_attn is None:\n                rows = rows.to(torch.int64)\n"),
     None, r"asked for|mk_attn calls|LEG D"),
    ("A2", (DA, "            fn, in_specs, outs, donate = build_draft_attn_call(self.mesh, self.g, B, **self.opts)\n",
            "            fn, in_specs, outs, donate = build_draft_attn_call(self.mesh, self.g, max(self.buckets or (B,)), "
            "**self.opts)\n"), None, r"kernel rows|cannot reshape"),
    ("A4", (DA, "                                             metadata.context_lens.to(torch.int32), attn.rope_cache, rank,\n",
            "                                             metadata.context_lens, attn.rope_cache, rank,\n"),
     None, r"op operand dtype"),
    # the D > 1 oracle fix: the old routing (the |v| / 1 transform never reaches dcp2_harness.reference_attend) must fail
    # the independent sum p|v| check (an env switch of h3bar_harness.install)
    ("M-dcp-old-routing", None, "env:Q38_H3BAR_NO_DCP_MODAL=1", r"DCP ORACLE TERM MISMATCH",
     "tests/kernels/qwen4_exp/dcp2/test_v6c_h3bar.py::test_dcp_oracle_term_matches_the_reference_sum"),
]


def _copy(tree, dst):
    for d in ("src", "tests"):
        shutil.copytree(os.path.join(tree, d), os.path.join(dst, d), ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    if os.path.exists(os.path.join(tree, "pyproject.toml")):
        shutil.copy(os.path.join(tree, "pyproject.toml"), dst)


def _run(tree, plugins, mut_env, target=TARGET):
    env = dict(os.environ, JAX_PLATFORMS="cpu", VLLM_TARGET_DEVICE="cpu", TORCH_DEVICE_BACKEND_AUTOLOAD="0",
               PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=f"{plugins}:{tree}/src:{tree}")
    env.pop("XLA_FLAGS", None)
    env.pop("Q38_H3BAR_MUT", None)
    env.pop("Q38_H3BAR_BFORM", None)
    env.pop("Q38_H3BAR_NO_DCP_MODAL", None)
    if mut_env and mut_env.startswith("env:"):
        k, _, v = mut_env[4:].partition("=")
        env[k] = v
    elif mut_env:
        mut, _, form = mut_env.partition(":")
        env["Q38_H3BAR_MUT"] = mut
        if form:
            env["Q38_H3BAR_BFORM"] = form
    cmd = [sys.executable, "-B", "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", "-W", "ignore", "-s",
           "-p", "q38_cpu_device", target]
    p = subprocess.run(cmd, cwd=tree, env=env, capture_output=True, text=True, timeout=10800)
    return p.returncode, p.stdout + p.stderr


def _one(args):
    tree, plugins, work, m = args
    name, edit, mut_env, reason = m[:4]
    target = m[4] if len(m) > 4 else TARGET
    dst = os.path.join(work, name)
    _copy(tree, dst)
    if edit is not None:
        rel, old, new = edit
        path = os.path.join(dst, rel)
        src = open(path).read()
        if src.count(old) != 1:
            return name, "BAD", -1, "anchor count %d" % src.count(old), ""
        open(path, "w").write(src.replace(old, new))
    rc, out = _run(dst, plugins, mut_env, target)
    found = re.search(reason, out) is not None
    lines = [ln for ln in out.splitlines() if ln.startswith(("E ", "FAILED", "H3BAR ")) or "LEG " in ln][:6]
    verdict = "KILLED" if rc == 1 and found else "SURVIVED" if rc == 0 else "BAD"
    shutil.rmtree(dst, ignore_errors=True)
    return name, verdict, rc, "found" if found else "missing", " | ".join(x[:300] for x in lines)


def main():
    tree, plugins = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    only = set(sys.argv[3].split(",")) if len(sys.argv) > 3 and sys.argv[3] else None
    muts = [m for m in MUTS if only is None or m[0] in only]
    work = tempfile.mkdtemp(prefix="h3barmut_")
    ctl = os.path.join(work, "control")
    _copy(tree, ctl)
    control_ok = True
    for tgt in sorted({(m[4] if len(m) > 4 else TARGET) for m in muts}):
        rc, out = _run(ctl, plugins, "", tgt)
        control_ok &= rc == 0
        print(f"CONTROL rc={rc} {tgt} :: " + " | ".join(ln[:300] for ln in out.splitlines() if ln.startswith(
            ("H3BAR_VERDICT", "H3BAR_RAW", "H3BAR_DCPTERM", "FAILED", "E ")))[:2000], flush=True)
    killed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        for name, verdict, rc, reason, tail in ex.map(_one, [(tree, plugins, work, m) for m in muts]):
            killed += verdict == "KILLED"
            print(f"MUT {name} {verdict} rc={rc} reason={reason} :: {tail}", flush=True)
    ok = control_ok and killed == len(muts)
    print(f"Q38_H3BAR_MUT {'PASS' if ok else 'FAIL'} killed {killed}/{len(muts)} control_ok={int(control_ok)}", flush=True)
    shutil.rmtree(work, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
