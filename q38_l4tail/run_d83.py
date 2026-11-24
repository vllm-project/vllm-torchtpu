"""q38 L4 L1-D83-SERVED CPU proof (D83S-BUILD): VLLM_TPU_VPS_FUSED_READ_OFFSETS on q38-l4-d83s (D83 rebased onto
q38-l4-stack 11b0c44ef + the warm-draws rebase fix b5b49a8d7 + the flag-first gate fix). Ported from q38-l4-tail's run_d83.py (j-9011c0d1 PASS on
ee289379f). The per-file regression leg moved to tests/q38_d83s/run_cpu.sh (L3's regress_perfile helper).
Run from the head tree root:  python3 -u q38_l4tail/run_d83.py clean|mutants
Legs (every PASS condition is stated here, before the run):
  SRC-HASH  the src hash (LC_ALL=C sorted sha1 of src/**/*.py) is printed; run_cpu.sh pins it.
  IMPORT    under this script's cwd + PYTHONPATH (src:q38_l4tail) vllm_torchtpu, .envs, .compilation.tpu_compiler and
            .runner.tpu_runner import and each __file__ is INSIDE <tree>/src (asserted). Two mutants must bite:
            FALLBACK (PYTHONPATH without src: vllm_torchtpu must resolve OUTSIDE <tree>/src, else the check proves
            nothing) and RESOLUTION (a scratch copy of src with `raise` appended to runner/tpu_runner.py: the import
            must fail, so the module really executes from the staged src).
  CLEAN     tests/l4tail/test_vps_ro_fold.py: rc 0, exactly EXPECT_PASSED passed, 0 failed / skipped / errors, the
            origin plugin's every non-None path INSIDE <tree>/src, and the counts JSON equal to EXPECT_COUNTS (exact
            keys; the two operand split keys >= 100 and the two fx tracer keys summing to 8).
  MUTANTS   (mutants mode) each mutant edits a private copy of src/ (inside one named function, or once in the file);
            test_vps_ro_fold.py must FAIL under the strict rule: origin OK, rc 1, >= 1 failed, 0 errors / no collection
            error, and no failing test that also fails clean. 4 workers; each copy deleted after its run; the tree's
            watched md5s re-checked after every mutant.
  SRC-AT-END md5 of every watched file is unchanged.
Final line Q38_L4D83S_SUMMARY ... VERDICT=PASS|FAIL; exit 0 only on PASS."""
import ast, concurrent.futures, hashlib, json, os, re, shutil, subprocess, sys, tempfile

MODE = sys.argv[1] if len(sys.argv) > 1 else "clean"
assert MODE in ("clean", "mutants"), MODE
ROOT = os.getcwd()
HERE = os.path.join(ROOT, "q38_l4tail")
SRC_REAL = os.path.realpath(os.path.join(ROOT, "src")) + os.sep
RUNNER = "src/vllm_torchtpu/runner/tpu_runner.py"
ENVS = "src/vllm_torchtpu/envs.py"
TC = "src/vllm_torchtpu/compilation/tpu_compiler.py"
T_FOLD = "tests/l4tail/test_vps_ro_fold.py"
WATCH = [RUNNER, ENVS, TC, T_FOLD, "tests/l4tail/ref_11b0c44ef_vps.py.txt", "tests/q38mm_dstock/test_vps_wiring.py",
         "q38_l4tail/q38_l4tail_origin.py"]
FLAGS = ("VLLM_TPU_VPS_FUSED_READ_OFFSETS", "VLLM_TPU_VPS_FUSED_READ_OFFSETS_CHECK",
         "VLLM_TPU_VPS_FUSED_READ_OFFSETS_CHECK_FAULT", "VLLM_TPU_VPS_WARM_DRAWS")
WORKERS = 4
EXPECT_PASSED = 86  # 78 ported (45 identity, 33 others) + 4 warm-draw + 2 census + 2 flags-off state
EXPECT_COUNTS = {  # exact, from the test's own COUNTS (the 78 ported ones match j-9011c0d1's)
    "eager_ops_added_fb": 1, "eager_ops_added_plain": 1, "fx_identical_programs": 4, "identity_cases": 45,
    "identity_row_cases": 9, "operand_cases": 300, "warm_fold_programs": 21, "warm_serving_keys_covered": 21,
    "warm_draw_calls": 8, "warm_draws_issued": 104, "census_steps_fb": 8, "census_steps_plain": 8,
    "census_separate_per_step_off": 1, "census_separate_per_step_on": 0}
EXPECT_MIN = {"operand_pad": 100, "operand_truncate": 100}
FX_KEYS, FX_TOTAL = ("fx_tracer_make_fx", "fx_tracer_dynamo"), 8

# name: (file, function or None (whole file), exact original text (once inside the function / file), replacement)
F_OPS = "_read_offset_scatter_operands"
MUTANTS = {
    # ---- the 28 of q38-l4-tail (none moot after the rebase: every anchor is found exactly once on the stack head)
    "m01_pad_on_the_wrong_side": (RUNNER, F_OPS, "next_tokens, (0, 0, 0, width - rows), value=INVALID_TOKEN_ID",
                                  "next_tokens, (0, 0, width - rows, 0), value=INVALID_TOKEN_ID"),
    "m02_truncate_keeps_the_tail": (RUNNER, F_OPS, "next_tokens = next_tokens[:width]",
                                    "next_tokens = next_tokens[-width:]"),
    "m03_cast_dropped": (RUNNER, F_OPS, "(num_valid - 1).clamp(min=0), 0).to(dtype)",
                         "(num_valid - 1).clamp(min=0), 0)"),
    "m04_offset_off_by_one": (RUNNER, F_OPS, "(num_valid - 1).clamp(min=0)", "num_valid.clamp(min=0)"),
    "m05_valid_mask_le": (RUNNER, F_OPS, "device=next_tokens.device) < num_reqs", "device=next_tokens.device) <= num_reqs"),
    "m06_no_per_group_repeat": (RUNNER, F_OPS, "return indices, offsets.repeat(len(groups))", "return indices, offsets"),
    "m07_first_group_only": (RUNNER, F_OPS,
                             "indices = torch.cat(groups, dim=0).long() if len(groups) > 1 else groups[0].long()",
                             "indices = groups[0].long().repeat(len(groups))"),
    "m08_fb_program_drops_ro": (RUNNER, "_vps_verify_fb_impl", "    if ro_groups is None:\n        return out\n",
                                "    if True:\n        return out\n"),
    "m09_scatter_skipped": (RUNNER, "_apply_folded_read_offsets", "        buf.index_put_((indices,), offsets)\n",
                            "        pass\n"),
    "m10_applied_not_marked": (RUNNER, "_apply_folded_read_offsets",
                               "        self._fused_read_offsets_applied = True\n", ""),
    # (anchor moved by the flag-first gate of the full-v2 fix; same mutation: the fold runs whatever the flag says)
    "m11_gate_ignores_flag": (RUNNER, "_sample_spec_verify_vps", 'if getattr(self, "_vps_ro_fold", False):',
                              "if True:"),
    "m12_init_ignores_flag": (RUNNER, "_init_speculative_decoding",
                              "self._vps_ro_fold = envs.VLLM_TPU_VPS_FUSED_READ_OFFSETS",
                              "self._vps_ro_fold = False"),
    "m13_check_skipped": (RUNNER, "_apply_folded_read_offsets",
                          "ref = buf.clone() if self._vps_ro_checks_left > 0 else None", "ref = None"),
    "m14_check_reference_not_run": (RUNNER, "_check_folded_read_offsets",
                                    "        _update_read_offsets_impl(ref, groups, tokens, num_reqs)\n", ""),
    "m15_check_fault_not_applied": (RUNNER, "_check_folded_read_offsets", "            ref[:1] += 1\n",
                                    "            pass\n"),
    "m16_warmup_ignores_flag_off": (RUNNER, "_precompile_vps_verify", 'if getattr(self, "_vps_ro_fold", False) else []',
                                    "if False else []"),
    "m17_warmup_ignores_flag_on": (RUNNER, "_precompile_vps_verify", 'if getattr(self, "_vps_ro_fold", False) else []',
                                   "if True else []"),
    "m18_warmup_misses_most_width": (RUNNER, "_read_offset_fold_geometry",
                                     "            widths.add(self.num_reqs_most_model_len)\n", "            pass\n"),
    "m19_warmup_aliased_groups": (RUNNER, "_vps_ro_fold_warmup_args",
                                  "            groups = [\n                torch.zeros(width, dtype=torch.int32).to(self.device)\n"
                                  "                for _ in range(num_groups)\n            ]\n",
                                  "            groups = [torch.zeros(width, dtype=torch.int32).to(self.device)] * num_groups\n"),
    "m20_warm_scatter_writes_values": (RUNNER, "_warm_folded_scatter", "torch.zeros_like(offsets)", "offsets"),
    "m21_geometry_counts_every_builder": (RUNNER, "_read_offset_fold_geometry",
                                          "                if group.metadata_builders[0].is_mamba_group\n", ""),
    "m22_plain_path_not_applied": (RUNNER, "_sample_spec_verify_vps",
                                   "            self._apply_folded_read_offsets(out, ro, ro_idx, ro_vals)\n",
                                   "            pass\n"),
    "m23_extra_eager_op": (RUNNER, "_apply_folded_read_offsets", "buf.index_put_((indices,), offsets)",
                           "buf.index_put_((indices,), offsets.clone())"),
    "m24_flags_off_fb_graph_changed": (RUNNER, "_vps_verify_fb_impl", "    if ro_groups is None:\n        return out\n",
                                       "    if ro_groups is None:\n        return out[0], out[1] * 1\n"),
    "m25_flags_off_plain_graph_changed": (RUNNER, "_vps_verify_impl", "    if ro_groups is None:\n        return out\n",
                                          "    if ro_groups is None:\n        return out + 0\n"),
    "m26_flag_unregistered": (ENVS, None,
                              '    "VLLM_TPU_VPS_FUSED_READ_OFFSETS": env_bool("VLLM_TPU_VPS_FUSED_READ_OFFSETS"),\n', ""),
    "m27_check_flag_hashed": (TC, None, '    "VLLM_TPU_VPS_FUSED_READ_OFFSETS_CHECK",\n', ""),
    "m28_fb_path_ro_not_passed": (RUNNER, "_sample_spec_verify_vps",
                                  "fallback_rows, sampling_generator, ro\n", "fallback_rows, sampling_generator, None\n"),
    # ---- new on the stack
    # the resolved precompile conflict: f0ba1ccad's `if fold: continue` ahead of EAGER-FIX's warm draws (fallback path)
    "m29_fold_skips_fb_warm_draws": (RUNNER, "_precompile_vps_verify",
                                     '            if getattr(envs, "VLLM_TPU_VPS_WARM_DRAWS", 0):\n'
                                     "                self._warm_vps_draws(\n                    num_reqs,\n",
                                     "            if fold:\n                continue\n"
                                     '            if getattr(envs, "VLLM_TPU_VPS_WARM_DRAWS", 0):\n'
                                     "                self._warm_vps_draws(\n                    num_reqs,\n"),
    # the same conflict on the plain VPS path
    "m30_fold_skips_plain_warm_draws": (RUNNER, "_precompile_vps_verify",
                                        '            if getattr(envs, "VLLM_TPU_VPS_WARM_DRAWS", 0):\n'
                                        "                self._warm_vps_draws(\n                    num_reqs, draft_token_ids",
                                        '            if getattr(envs, "VLLM_TPU_VPS_WARM_DRAWS", 0) and not fold:\n'
                                        "                self._warm_vps_draws(\n                    num_reqs, draft_token_ids"),
    # the fix's other half: with the fold on, the flag-off fallback program must not be warmed as well
    "m31_fold_also_warms_flag_off_fb": (RUNNER, "_precompile_vps_verify",
                                        "            if not fold:\n                out, stats = _vps_verify_fb_compiled(\n",
                                        "            if True:\n                out, stats = _vps_verify_fb_compiled(\n"),
    # the fold flag compile-ignored (it changes the verify program: it must stay hashed)
    "m32_fold_flag_compile_ignored": (TC, None, '    "VLLM_TPU_VPS_FUSED_READ_OFFSETS_CHECK",\n',
                                      '    "VLLM_TPU_VPS_FUSED_READ_OFFSETS",\n    "VLLM_TPU_VPS_FUSED_READ_OFFSETS_CHECK",\n'),
    # sample_tokens runs the separate program even when the verify applied the fold
    "m33_sample_tokens_gate_dropped": (RUNNER, "sample_tokens", "                    if not self._fused_read_offsets_applied:\n",
                                       "                    if True:\n"),
    # the full-v2 regression (EAGER-FIX's fake): flag off, the VPS path reads D83's state again (f0ba1ccad's order)
    "m34_flag_off_reads_fold_state": (RUNNER, "_sample_spec_verify_vps",
                                      '        ro = None\n        if getattr(self, "_vps_ro_fold", False):\n',
                                      '        ro = None\n        _unused = self._fused_read_offsets\n'
                                      '        if getattr(self, "_vps_ro_fold", False):\n'),
}
DROPPED = {}  # mutants of q38-l4-tail made moot by the rebase: none

md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
START = {p: md5(p) for p in WATCH}
print("SRC-AT-START " + " ".join("%s=%s" % (p, START[p][:8]) for p in WATCH), flush=True)
srch = subprocess.run("find src -type f -name '*.py' -print0 | LC_ALL=C sort -z | xargs -0 sha1sum | sha1sum | cut -c1-10",
                      shell=True, capture_output=True, text=True).stdout.strip()
print(f"SRC-HASH {srch}", flush=True)


def run(cmd, src, extra=None, timeout=3000, pythonpath=None):
    env = {k: v for k, v in os.environ.items() if k not in FLAGS}
    env.update(PYTHONPATH=pythonpath if pythonpath is not None else f"{src}:{HERE}", PYTHONUNBUFFERED="1",
               Q38MM_SRC_ROOT=src, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", JAX_PLATFORMS="cpu", **(extra or {}))
    try:
        return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout, cwd=ROOT)
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        return subprocess.CompletedProcess(cmd, 124, out, "TIMEOUT")


def pytest(src, test, args=(), extra=None):
    p = run([sys.executable, "-u", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "q38_l4tail_origin",
             "-rfEs", *args, test], src, extra)
    lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
    tail = next((ln for ln in reversed(lines) if re.search(r"\d+ (passed|failed|error|deselected|skipped)", ln)),
                lines[-1] if lines else "<no output>")
    m = re.search(r"^Q38_L4TAIL_ORIGIN (.*)$", p.stdout, re.M)
    want = os.path.realpath(os.path.join(ROOT, src)) + os.sep
    vals = [kv.split("=", 1)[1] for kv in m.group(1).split()] if m else []
    origin_ok = bool(m) and any(v != "None" for v in vals) and all(
        v == "None" or os.path.realpath(os.path.join(ROOT, v)).startswith(want) for v in vals)
    failed = re.findall(r"^FAILED (\S+)", p.stdout, re.M)
    num = lambda pat: int((re.findall(pat, tail) or ["0"])[-1])
    return dict(rc=p.returncode, tail=tail, out=p.stdout + p.stderr, origin_ok=origin_ok,
                origin=m.group(0) if m else "Q38_L4TAIL_ORIGIN <missing>", failed=failed, npass=num(r"(\d+) passed"),
                nfail=num(r"(\d+) failed"), nerr=num(r"(\d+) errors?"), nskip=num(r"(\d+) skipped"))


def show(tag, r):
    print(f"{tag} rc={r['rc']} origin={'OK' if r['origin_ok'] else 'INVALID'} :: {r['tail']}", flush=True)
    print(f"  {r['origin']}", flush=True)


def apply_mutant(rel, fn, old, new):
    text = open(rel).read()
    if fn is None:
        return text.replace(old, new) if text.count(old) == 1 else None
    nodes = [n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef) and n.name == fn]
    if len(nodes) != 1:
        return None
    lines = text.splitlines(keepends=True)
    a = sum(len(ln) for ln in lines[: nodes[0].lineno - 1])
    b = sum(len(ln) for ln in lines[: nodes[0].end_lineno])
    seg = text[a:b]
    if seg.count(old) != 1:
        return None
    return text[:a] + seg.replace(old, new) + text[b:]


ok = True
# ---- IMPORT: origin asserted under this script's cwd + PYTHONPATH, with FALLBACK and RESOLUTION mutants
PROBE = ("import importlib, os\nfor n in ('vllm_torchtpu', 'vllm_torchtpu.envs', 'vllm_torchtpu.compilation.tpu_compiler',"
         " 'vllm_torchtpu.runner.tpu_runner'):\n    print('Q38_ORIGIN', n, os.path.realpath(importlib.import_module(n).__file__))\n")
p = run([sys.executable, "-u", "-c", PROBE], "src")
got = re.findall(r"^Q38_ORIGIN (\S+) (\S+)$", p.stdout, re.M)
inside = [(n, f, f.startswith(SRC_REAL)) for n, f in got]
for n, f, i in inside:
    print(f"IMPORT {n} {f} {'INSIDE' if i else 'OUTSIDE'}", flush=True)
imp = p.returncode == 0 and len(inside) == 4 and all(i for _, _, i in inside)
if not imp:
    print(f"IMPORT rc={p.returncode}\n{p.stderr[-4000:]}", flush=True)
empty = tempfile.mkdtemp(prefix="q38_l4d83s_empty_")
p = run([sys.executable, "-u", "-c", "import vllm_torchtpu, os; print('Q38_ORIGIN', os.path.realpath(vllm_torchtpu.__file__))"],
        "src", pythonpath=empty)
fb_file = (re.findall(r"^Q38_ORIGIN (\S+)$", p.stdout, re.M) or ["none"])[-1]
fallback_killed = not fb_file.startswith(SRC_REAL)
print(f"IMPORT MUTANT FALLBACK (PYTHONPATH without src) -> {fb_file} -> {'KILLED' if fallback_killed else 'SURVIVED'}",
      flush=True)
shutil.rmtree(empty, ignore_errors=True)
rdir = tempfile.mkdtemp(prefix="q38_l4d83s_resolve_")
shutil.copytree("src", os.path.join(rdir, "src"))
with open(os.path.join(rdir, RUNNER), "a") as f:
    f.write("\nraise RuntimeError('q38 D83S resolution mutant')\n")
p = run([sys.executable, "-u", "-c", PROBE], os.path.join(rdir, "src"),
        pythonpath=f"{os.path.join(rdir, 'src')}:{HERE}")
resolution_killed = p.returncode != 0 and "q38 D83S resolution mutant" in p.stderr
print(f"IMPORT MUTANT RESOLUTION (raise appended to the staged tpu_runner.py) -> rc={p.returncode} -> "
      f"{'KILLED' if resolution_killed else 'SURVIVED'}", flush=True)
shutil.rmtree(rdir, ignore_errors=True)
imp_ok = imp and fallback_killed and resolution_killed
print(f"IMPORT -> {'PASS' if imp_ok else 'FAIL'}", flush=True)
ok &= imp_ok

# ---- CLEAN
counts_path = os.path.join(tempfile.gettempdir(), "q38_l4d83s_counts.json")
if os.path.exists(counts_path):
    os.remove(counts_path)
r = pytest("src", T_FOLD, ("--noconftest",), {"Q38_L4TAIL_COUNTS": counts_path})
clean_failed = list(r["failed"])  # mutant legs deselect these: a kill must come from a test that passes clean
try:
    counts = json.load(open(counts_path))
except (OSError, ValueError):
    counts = None
cnt_bad = []
if counts is None:
    cnt_bad.append("<counts missing>")
else:
    for k, v in EXPECT_COUNTS.items():
        if counts.get(k) != v:
            cnt_bad.append(f"{k}={counts.get(k)}!={v}")
    for k, v in EXPECT_MIN.items():
        if (counts.get(k) or 0) < v:
            cnt_bad.append(f"{k}={counts.get(k)}<{v}")
    if sum(counts.get(k, 0) for k in FX_KEYS) != FX_TOTAL:
        cnt_bad.append(f"fx tracers {[counts.get(k) for k in FX_KEYS]} != {FX_TOTAL}")
    extra = set(counts) - set(EXPECT_COUNTS) - set(EXPECT_MIN) - set(FX_KEYS)
    if extra:
        cnt_bad.append(f"unexpected keys {sorted(extra)}")
clean = (r["rc"] == 0 and r["origin_ok"] and r["npass"] == EXPECT_PASSED and r["nfail"] == 0 and r["nerr"] == 0
         and r["nskip"] == 0 and not cnt_bad)
show(f"CLEAN -> {'PASS' if clean else 'FAIL'} (expect exactly {EXPECT_PASSED} passed)", r)
print(f"  Q38_L4D83S_COUNTS {json.dumps(counts, sort_keys=True)} {'OK' if not cnt_bad else 'BAD ' + '; '.join(cnt_bad)}",
      flush=True)
if not clean:
    print(r["out"][-12000:], flush=True)
ok &= clean

# ---- MUTANTS
killed = invalid = 0
if MODE == "mutants":
    def one(item):
        name, (rel, fn, old, new) = item
        text = apply_mutant(rel, fn, old, new)
        if text is None:
            return name, "ABORT", f"the anchor was not found exactly once in {fn or rel}", None
        try:
            ast.parse(text)
        except SyntaxError as e:
            return name, "ABORT", f"the mutated file does not parse: {e}", None
        d = tempfile.mkdtemp(prefix=f"q38_l4d83s_{name}_")
        try:
            shutil.copytree("src", os.path.join(d, "src"))
            with open(os.path.join(d, rel), "w") as f:
                f.write(text)
            r = pytest(os.path.join(d, "src"), T_FOLD,
                       ("--noconftest", "-x", *[a for f in clean_failed for a in ("--deselect", f)]))
        finally:
            shutil.rmtree(d, ignore_errors=True)
        if not r["origin_ok"] or r["nerr"] or "error" in r["tail"]:
            verdict = "INVALID"
        elif r["rc"] == 1 and r["nfail"] >= 1 and not set(r["failed"]) & set(clean_failed):
            verdict = "KILLED"
        else:
            verdict = "SURVIVED"
        return name, verdict, "", r

    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        results = list(ex.map(one, MUTANTS.items()))
    for name, verdict, why, r in results:
        if verdict == "ABORT":
            print(f"MUTANT {name}: ABORT, {why}", flush=True)
            ok = False
            continue
        killed += verdict == "KILLED"
        invalid += verdict == "INVALID"
        ok &= verdict == "KILLED"
        show(f"MUTANT {name} -> {verdict} (failed: {', '.join(f.split('::')[-1] for f in r['failed'][:4])}"
             f"{' ...' if len(r['failed']) > 4 else ''})", r)
        if verdict != "KILLED":
            print(r["out"][-4000:], flush=True)
    for name in DROPPED:
        print(f"MUTANT {name}: DROPPED ({DROPPED[name]})", flush=True)
    mid = {p: md5(p) for p in WATCH}
    print(f"  md5 after the mutants (copies only): {'== start' if mid == START else '!= START'}", flush=True)
    ok &= mid == START
    ok &= killed == len(MUTANTS)

END = {p: md5(p) for p in WATCH}
unchanged = END == START
print("SRC-AT-END %s" % ("UNCHANGED" if unchanged else "CHANGED: " + " ".join(p for p in WATCH if END[p] != START[p])),
      flush=True)
ok &= unchanged
mut = f" mutants_killed={killed}/{len(MUTANTS)} invalid={invalid} dropped={len(DROPPED)}" if MODE == "mutants" else ""
print(f"Q38_L4D83S_SUMMARY mode={MODE} src={srch} import={'PASS' if imp_ok else 'FAIL'} "
      f"clean={'PASS' if clean else 'FAIL'}{mut} src_at_end={'UNCHANGED' if unchanged else 'CHANGED'} "
      f"VERDICT={'PASS' if ok else 'FAIL'}", flush=True)
sys.exit(0 if ok else 1)
