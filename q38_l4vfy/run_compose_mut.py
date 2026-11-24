"""q38 L4 VFY-ONE x DRAFT-ASM composition mutants (VFY-BUILD), run from the q38-l4-vfy-v5 tree root:
    python3 -u q38_l4vfy/run_compose_mut.py
Each mutant edits a private copy of src/ (inside one named function, or once in the file); tests/l4vfy/
test_vfy_dha_compose.py must FAIL under it: rc 1, >= 1 failed, no collection error, the mutated copy imported
(Q38_COMPOSE_ORIGIN inside the copy), no failing test that also fails clean. Clean first: rc 0, exactly 17 passed.
Final line Q38_L4VFY_COMPOSE_MUT PASS|FAIL killed=N/N; exit 0 only on PASS."""
import ast, os, re, shutil, subprocess, sys, tempfile

ROOT = os.getcwd()
DA = "src/vllm_torchtpu/spec_decode/draft_asm.py"
EG = "src/vllm_torchtpu/spec_decode/eagle3.py"
VF = "src/vllm_torchtpu/runner/vfy_one.py"
T = "tests/l4vfy/test_vfy_dha_compose.py"
TF = "tests/l4vfy/test_vfy_v5_flagoff.py"
RN = "src/vllm_torchtpu/runner/tpu_runner.py"
EXPECT = 17
MUTANTS = {
    "cm1_draftasm_runs_its_seed_after_a_take": (DA, "seed", "        if out is None:\n            out = _async_seed_inputs_asm_compiled(*args)\n",
                                                "        if True:\n            out = _async_seed_inputs_asm_compiled(*args)\n"),
    "cm2_take_compares_shapes_not_identity": (VF, "same_args", "            if x is not y:\n",
                                              "            if tuple(x.shape) != tuple(y.shape):\n"),
    "cm3_hook_ignores_static": (DA, "vfy_seed_spec", "        if self._static() or chunk.draft_lengths is None:\n",
                                "        if chunk.draft_lengths is None:\n"),
    "cm4_warm_spec_seq_lens_wrong_shape": (DA, "vfy_seed_warm_spec",
                                           "            torch.zeros(capacity, dtype=torch.int32).to(dev),\n            capacity,\n",
                                           "            torch.zeros(1, dtype=torch.int32).to(dev),\n            capacity,\n"),
    "cm5_superset_args_reordered": (DA, "_vfy_seed_args", "return (*base[:4], seq_lens, base[4], base[5], self.max_reqs), base",
                                    "return (*base[:4], base[4], seq_lens, base[5], self.max_reqs), base"),
    "cm6_base_block_take_dropped": (EG, "_prepare_draft_inputs", 'if self.runner.__dict__.get("_vfy_seed") is not None:',
                                    "if False:"),
    "cm7_draftasm_takes_with_the_base_body": (DA, "seed", "p._vfy_seed_take(chunk, nt, _async_seed_inputs_asm_impl, args[1:])",
                                              "p._vfy_seed_take(chunk, nt, eagle3._async_seed_inputs_impl, args[1:])"),
    # flag-off identity on the composed tree (target: test_vfy_v5_flagoff.py)
    "cm8_vfy_hook_unguarded_composed": (RN, "capture_model",
                                        '                if self.__dict__.get("_vfy") is not None:\n'
                                        "                    # q38 L4 VFY-ONE: its verify programs beside the D83 chain's.\n"
                                        "                    self._vfy_precompile()\n",
                                        "                self._vfy_precompile()\n"),
    "cm9_draftasm_hook_unguarded_composed": (EG, "propose",
                                             "        if self._draft_asm is not None:\n            self._draft_asm.begin(\n",
                                             "        if True:\n            self._draft_asm.begin(\n"),
}
TARGETS = {"cm8_vfy_hook_unguarded_composed": TF, "cm9_draftasm_hook_unguarded_composed": TF}
ORIGIN = ("import os, vllm_torchtpu.spec_decode.draft_asm as d, vllm_torchtpu.runner.vfy_one as v; "
          "print('Q38_COMPOSE_ORIGIN', os.path.realpath(d.__file__), os.path.realpath(v.__file__))")
# (the flag-off targets read the source files through Q38MM_SRC_ROOT, set to the mutated copy below)


def apply_mutant(rel, fn, old, new):
    text = open(rel).read()
    nodes = [n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef) and n.name == fn]
    if len(nodes) != 1:
        return None
    lines = text.splitlines(keepends=True)
    a = sum(len(ln) for ln in lines[: nodes[0].lineno - 1])
    b = sum(len(ln) for ln in lines[: nodes[0].end_lineno])
    seg = text[a:b]
    return None if seg.count(old) != 1 else text[:a] + seg.replace(old, new) + text[b:]


def pytest(src, args=(), test=T):
    env = dict(os.environ, PYTHONPATH=src, JAX_PLATFORMS="cpu", OMP_NUM_THREADS="2", PYTHONUNBUFFERED="1",
               Q38MM_SRC_ROOT=src)
    o = subprocess.run([sys.executable, "-c", ORIGIN], env=env, capture_output=True, text=True, cwd=ROOT).stdout
    p = subprocess.run([sys.executable, "-u", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:logging",
                        "-rfEs", *args, test], env=env, capture_output=True, text=True, cwd=ROOT, timeout=3000)
    tail = next((ln for ln in reversed(p.stdout.splitlines()) if re.search(r"\d+ (passed|failed|error)", ln)), "<none>")
    num = lambda pat: int((re.findall(pat, tail) or ["0"])[-1])
    want = os.path.realpath(src) + os.sep
    m = re.search(r"Q38_COMPOSE_ORIGIN (\S+) (\S+)", o)
    origin_ok = bool(m) and all(x.startswith(want) for x in m.groups())
    return dict(rc=p.returncode, tail=tail, failed=re.findall(r"^FAILED (\S+)", p.stdout, re.M), npass=num(r"(\d+) passed"),
                nfail=num(r"(\d+) failed"), nerr=num(r"(\d+) errors?"), origin_ok=origin_ok, out=p.stdout + p.stderr,
                origin=m.group(0) if m else "<origin missing>")


clean = pytest("src")
print(f"CLEAN rc={clean['rc']} origin={'OK' if clean['origin_ok'] else 'INVALID'} :: {clean['tail']}", flush=True)
ok = clean["rc"] == 0 and clean["origin_ok"] and clean["npass"] == EXPECT and not clean["nfail"] and not clean["nerr"]
if not ok:
    print(clean["out"][-8000:], flush=True)
killed = 0
for name, (rel, fn, old, new) in MUTANTS.items():
    text = apply_mutant(rel, fn, old, new)
    if text is None:
        print(f"MUTANT {name}: ABORT anchor not found once in {fn}", flush=True)
        ok = False
        continue
    d = tempfile.mkdtemp(prefix=f"q38_cmp_{name}_")
    try:
        shutil.copytree("src", os.path.join(d, "src"))
        open(os.path.join(d, rel), "w").write(text)
        r = pytest(os.path.join(d, "src"), ("-x",), TARGETS.get(name, T))
    finally:
        shutil.rmtree(d, ignore_errors=True)
    v = ("INVALID" if not r["origin_ok"] or r["nerr"] else
         "KILLED" if r["rc"] == 1 and r["nfail"] >= 1 and not set(r["failed"]) & set(clean["failed"]) else "SURVIVED")
    killed += v == "KILLED"
    ok &= v == "KILLED"
    print(f"MUTANT {name} -> {v} :: {r['tail']} (failed: {', '.join(f.split('::')[-1] for f in r['failed'][:3])}) {r['origin']}",
          flush=True)
    if v != "KILLED":
        print(r["out"][-3000:], flush=True)
print(f"Q38_L4VFY_COMPOSE_MUT {'PASS' if ok else 'FAIL'} killed={killed}/{len(MUTANTS)}", flush=True)
sys.exit(0 if ok else 1)
