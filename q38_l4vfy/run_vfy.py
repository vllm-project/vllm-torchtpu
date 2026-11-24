"""q38 L4 VFY-ONE CPU proof (VFY-BUILD, helper of L4-E): VLLM_TPU_VPS_VFY_ONE on q38-l4-vfy (q38-L1-v3c b551e9b84 + D83S
cherry-picks + VFY-ONE). Structure after D83S-BUILD's q38_l4tail/run_d83.py (credited). Run from the head tree root:
    python3 -u q38_l4vfy/run_vfy.py clean
    python3 -u q38_l4vfy/run_vfy.py mutants LO HI      (mutants number LO..HI of the sorted list, 1-based, inclusive)
Legs (every PASS condition is stated here, before the run):
  SRC-HASH  the src hash (LC_ALL=C sorted sha1 of src/**/*.py) is printed; run_cpu.sh pins it.
  IMPORT    under this script's cwd + PYTHONPATH (src:q38_l4vfy) vllm_torchtpu, .envs, .compilation.tpu_compiler,
            .runner.vfy_one and .runner.tpu_runner import and each __file__ is INSIDE <tree>/src (asserted). Two
            mutants must bite: FALLBACK (PYTHONPATH without src: vllm_torchtpu resolves OUTSIDE <tree>/src) and
            RESOLUTION (a scratch copy of src with `raise` appended to runner/vfy_one.py: the import must fail).
  CLEAN     (both modes) tests/l4vfy/test_vfy_one.py, test_vfy_agree.py and test_vfy_flagoff.py (3: flag-off AST identity vs f24edcc94): rc 0, exactly EXPECT_ONE /
            EXPECT_AGREE passed, 0 failed / skipped / errors, the origin plugin's every non-None path INSIDE <tree>/src,
            the counts JSONs equal to EXPECT_COUNTS / EXPECT_AGREE_COUNTS (exact keys); D83's
            tests/l4tail/test_vps_ro_fold.py on this head: rc 0, exactly 86 passed (the D83 behaviour VFY-ONE keeps).
  MUTANTS   (mutants mode) each mutant edits a private copy of src/ (inside one named function, or once in the file);
            its target test files must FAIL under the strict rule: origin OK, rc 1, >= 1 failed, 0 errors / no
            collection error, no failing test that also fails clean. 4 workers; copies deleted; watched md5s re-checked.
  SRC-AT-END md5 of every watched file is unchanged.
Final line Q38_L4VFY_SUMMARY ... VERDICT=PASS|FAIL; exit 0 only on PASS."""
import ast, concurrent.futures, hashlib, json, os, re, shutil, subprocess, sys, tempfile

MODE = sys.argv[1] if len(sys.argv) > 1 else "clean"
assert MODE in ("clean", "mutants"), MODE
LO, HI = (int(sys.argv[2]), int(sys.argv[3])) if MODE == "mutants" else (0, 0)
ROOT = os.getcwd()
HERE = os.path.join(ROOT, "q38_l4vfy")
SRC_REAL = os.path.realpath(os.path.join(ROOT, "src")) + os.sep
RUNNER = "src/vllm_torchtpu/runner/tpu_runner.py"
VFY = "src/vllm_torchtpu/runner/vfy_one.py"
EAGLE = "src/vllm_torchtpu/spec_decode/eagle3.py"
ENVS = "src/vllm_torchtpu/envs.py"
TC = "src/vllm_torchtpu/compilation/tpu_compiler.py"
L6AB = "src/vllm_torchtpu/core/l6_ab.py"
T_ONE = "tests/l4vfy/test_vfy_one.py"
T_AGREE = "tests/l4vfy/test_vfy_agree.py"
T_D83 = "tests/l4tail/test_vps_ro_fold.py"
T_FLAG = "tests/l4vfy/test_vfy_flagoff.py"
T_REG = "tests/compilation/test_tpu_compile_key_registration.py"
WATCH = [RUNNER, VFY, EAGLE, ENVS, TC, L6AB, T_ONE, T_AGREE, T_FLAG, "tests/l4vfy/vfy_flagoff_ast.py",
         "tests/l4vfy/ref_f24edcc94_eagle3.py.txt", "tests/l4vfy/ref_f24edcc94_tpu_runner.py.txt",
         "tests/l4vfy/vfy_harness.py", T_D83, T_REG,
         "tests/q38mm_dstock/test_vps_wiring.py", "q38_l4vfy/q38_l4vfy_origin.py"]
FLAGS = ("VLLM_TPU_VPS_VFY_ONE", "VLLM_TPU_VPS_VFY_ONE_CHECK", "VLLM_TPU_VPS_VFY_ONE_AB", "VLLM_TPU_VPS_FUSED_READ_OFFSETS",
         "VLLM_TPU_VPS_FUSED_READ_OFFSETS_CHECK", "VLLM_TPU_VPS_FUSED_READ_OFFSETS_CHECK_FAULT", "VLLM_TPU_VPS_WARM_DRAWS",
         "VLLM_TPU_L6_AB_STEPS", "VLLM_TPU_L6_AB_PERIOD_MS", "VLLM_TPU_L6_AB_CLASSES", "VLLM_TPU_L6_AB_CONFIRM",
         "VLLM_TPU_L6_AB_DIR", "VLLM_TPU_SERVED_AB", "VLLM_TPU_HOST_AB_PERIOD_MS")
WORKERS = 4
EXPECT_ONE = 78
EXPECT_AGREE = 15
EXPECT_D83 = 86
EXPECT_FLAG = 3
EXPECT_COUNTS = {  # exact, from test_vfy_one.py's own COUNTS
    "identity_cases": 42, "identity_seed_cases": 20, "accumulate_steps": 8, "census_calls_d83": 2,
    "census_calls_vfy": 1, "census_eager_removed": 2, "fallback_cases": 3, "warm_programs_m1_w1": 24,
    "warm_programs_m2_w1": 24, "warm_programs_m2_w2": 42, "warm_serving_steps": 228, "check_cases": 4,
    "seed_taken": 1, "seed_miss_cases": 4, "dha_adopted": 1, "seed_written_cases": 4, "dha_seq_lens_caught": 1,
    "warm_cap_programs": 12}
EXPECT_AGREE_COUNTS = {"lockstep_checks": 10, "windowed_on_steps": 30, "gloo_ranks": 2}

ONE, AGREE, REG, FLAG = ("one",), ("agree",), ("reg",), ("flag",)
# name: (file, function or None (whole file), exact original text (once inside the function / file), replacement,
#        target test files)
MUTANTS = {
    # ---- the program and its runner path (T-3 VFY-ONE, DH-C)
    "v01_stats_pair_dropped": (RUNNER, "_vfy_one_impl", "acc = stats_acc + stats.to(torch.int64)",
                               "acc = stats.to(torch.int64)", ONE),
    "v02_read_offsets_wrong_row": (RUNNER, "_vfy_one_impl", "        return tokens, acc, ro_idx, ro_vals\n",
                                   "        return tokens, acc, ro_idx, ro_vals.roll(1, 0)\n", ONE),
    "v03_uniform_draw_moved_into_program": (RUNNER, "_vfy_one_impl",
                                            "    if logits_indices is None:\n        selected = hidden_states\n",
                                            "    recover_u = torch.rand_like(recover_u)\n"
                                            "    if logits_indices is None:\n        selected = hidden_states\n", ONE),
    "v04_seed_outputs_off_by_one": (RUNNER, "_vfy_one_impl", "*seed_fn(tokens, *seed_args)",
                                    "*(o + 1 for o in seed_fn(tokens, *seed_args))", ONE),
    "v05_flag_ignored": (VFY, "from_env", "    mode = envs.vfy_one_value()\n",
                         "    mode = envs.vfy_one_value() or 1\n", ONE),
    "v06_value_2_ignored": (RUNNER, "_vfy_seed_spec", "if vfy.mode != 2 or not vfy.seed_on(self):", "if True:", ONE),
    "v07_missing_seed_variant": (RUNNER, "_vfy_precompile", "if drafter is not None:\n", "if False:\n", ONE),
    "v08_missing_fallback_variant": (RUNNER, "_vfy_precompile", "for rows in variants:\n",
                                     "for rows in variants[:1]:\n", ONE),
    "v09_seed_variant_only_at_value_2": (RUNNER, "_vfy_precompile",
                                         "drafter = self._async_drafter if seed_on else None",
                                         "drafter = self._async_drafter if seed_on and vfy.mode == 2 else None", ONE),
    "v10_cold_check_bypassed": (RUNNER, "_vfy_verify", "if _vfy_one.key(*call[1:]) not in vfy.warm:", "if False:", ONE),
    "v11_cold_path_drops_the_drawn_fb_u": (RUNNER, "_vfy_verify",
                                           "return self._vfy_chain(local_cold, args, num_active, fallback_rows, fb_u, ro)",
                                           "return self._vfy_chain(local_cold, args, num_active, fallback_rows, None, ro)",
                                           ONE),
    "v12_defer_ignores_single_chunk": (RUNNER, "execute_model", "and self._vfy.decide(self, target_num_chunks == 1)",
                                       "and self._vfy.decide(self, True)", ONE),
    "v13_seed_take_ignores_identity": (EAGLE, "_vfy_seed_take", "ok = tokens is nt and (", "ok = (", ONE),
    "v14_seed_block_also_runs_its_program": (EAGLE, "_vfy_seed_or_run",
                                             "        return out if out is not None else _async_seed_inputs_compiled(nt, *args)\n",
                                             "        ran = _async_seed_inputs_compiled(nt, *args)\n"
                                             "        return out if out is not None else ran\n", ONE),
    "v15_check_counts_nothing": (RUNNER, "_vfy_check", "n = int(mism.item())", "n = 0", ONE),
    "v16_check_serves_the_vfy_result": (RUNNER, "_vfy_verify",
                                        "            self._vfy_check(_vfy_one_compiled(*call), out, pre, spec)\n",
                                        "            res_v = _vfy_one_compiled(*call)\n"
                                        "            self._vfy_check(res_v, out, pre, spec)\n"
                                        "            out = res_v[0]\n", ONE),
    "v17_unused_seed_not_dropped": (RUNNER, "sample_tokens", '            self.__dict__.pop("_vfy_seed")\n',
                                    "            pass\n", ONE),
    "v18_step_hook_removed": (RUNNER, "execute_model", "            _vfy.begin_step(self)\n", "            pass\n", ONE),
    "v19_logits_not_deferred": (RUNNER, "_vfy_defer_logits",
                                "        return _vfy_one.VfyChunk(hidden_states, logits_indices), hidden_states\n",
                                "        return self.compute_selected_local_logits(hidden_states, logits_indices), hidden_states\n",
                                ONE),
    "v20_accumulator_not_stored": (RUNNER, "_vfy_verify", "        self._vps_fb_stats = acc\n", "", ONE),
    "v21_fold_not_applied": (RUNNER, "_vfy_verify", "        self._apply_folded_read_offsets(out, ro, ro_idx, ro_vals)\n",
                             "", ONE),
    "v22_flag_compile_ignored": (TC, None, '    "VLLM_TPU_VPS_VFY_ONE_CHECK",\n',
                                 '    "VLLM_TPU_VPS_VFY_ONE",\n    "VLLM_TPU_VPS_VFY_ONE_CHECK",\n', REG),
    "v23_check_flag_hashed": (TC, None, '    "VLLM_TPU_VPS_VFY_ONE_CHECK",\n', "", REG),
    "v24_values_1_and_2_hash_apart": (ENVS, None, '    "VLLM_TPU_VPS_VFY_ONE": lambda: vfy_one_value() > 0,\n',
                                      '    "VLLM_TPU_VPS_VFY_ONE": lambda: vfy_one_value(),\n', REG),
    "v26_inplace_version_guard_dropped": (EAGLE, "_vfy_seed_take",
                                          "        if vers is None or vfy_one.versions((tokens, *spec.args, *spec.base_args)) != vers:\n",
                                          "        if False:\n", ONE),
    "v27_versions_not_recorded": (RUNNER, "_vfy_verify", "_vfy_one.versions((out, *spec.args, *spec.base_args))", "None",
                                  ONE),
    "v28_chain_drops_the_stats_accumulate": (RUNNER, "_vfy_chain",
                                             "        self._vps_fb_stats = stats if self._vps_fb_stats is None else "
                                             "self._vps_fb_stats + stats\n", "        pass\n", ONE),
    "v29_precompile_call_unguarded": (RUNNER, "capture_model",
                                      '                if self.__dict__.get("_vfy") is not None:\n'
                                      "                    # q38 L4 VFY-ONE: its verify programs beside the D83 chain's.\n"
                                      "                    self._vfy_precompile()\n",
                                      "                self._vfy_precompile()\n", FLAG),
    "v30_route_unguarded": (RUNNER, "_sample_spec_verify_vps",
                            "if not isinstance(local_logits, torch.Tensor) and self._vfy is not None:",
                            "if not isinstance(local_logits, torch.Tensor):", FLAG),
    "v25_l6_lever_unregistered": (L6AB, None, '    "vfyone": "VLLM_TPU_VPS_VFY_ONE_AB",\n', "", AGREE),
    # ---- the all-rank agreement (lead safety rule): bypass mutants
    "s01_own_bit_instead_of_min": (VFY, "apply", "engaged = not self.failed and min(en) == 1",
                                   "engaged = not self.failed and en[self._rank()] == 1", AGREE),
    "s02_local_disable_takes_effect_at_once": (VFY, "disable_locally", "            self.local_disabled.append(why)\n",
                                               "            self.local_disabled.append(why)\n"
                                               "            self.engaged = False\n", AGREE),
    "s03_failed_gather_swallowed": (VFY, "gather", "    except Exception as exc:\n        raise RuntimeError(",
                                    "    except Exception as exc:\n        return [list(vec)]\n        raise RuntimeError(",
                                    AGREE),
    "s04_decide_reads_the_rank_wish_per_step": (VFY, "decide",
                                                "if not self.engaged or self.failed or not single_chunk:",
                                                "if not self.wish() or self.failed or not single_chunk:", AGREE),
    "s05_no_agreement_at_window_rolls": (VFY, "begin_step", "            if cls != self.last_cls:\n",
                                         "            if False:\n", AGREE),
    "s06_check_mismatch_ignored": (VFY, "apply", "            if sum(mism):\n", "            if False:\n", AGREE),
    "s07_agreement_every_step": (VFY, "begin_step", "        boundary = self.serving == 1\n",
                                 "        boundary = True\n", AGREE),
    "s08_unequal_warm_keys_ignored": (VFY, "apply", "        if len(set(fps)) != 1:\n", "        if False:\n", AGREE),
    "s09_failure_not_sticky": (VFY, "apply", "        if why and not self.failed:\n            self.failed = True\n",
                               "        if why and not self.failed:\n            pass\n", AGREE),
}
NAMES = sorted(MUTANTS)
SELECTED = NAMES[LO - 1:HI] if MODE == "mutants" else []

md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
START = {p: md5(p) for p in WATCH}
print("SRC-AT-START " + " ".join("%s=%s" % (p, START[p][:8]) for p in WATCH), flush=True)
srch = subprocess.run("find src -type f -name '*.py' -print0 | LC_ALL=C sort -z | xargs -0 sha1sum | sha1sum | cut -c1-10",
                      shell=True, capture_output=True, text=True).stdout.strip()
print(f"SRC-HASH {srch}", flush=True)
print(f"MUTANT-LIST {len(NAMES)}: " + " ".join(f"{i + 1}={n}" for i, n in enumerate(NAMES)), flush=True)


def run(cmd, src, extra=None, timeout=3000, pythonpath=None):
    env = {k: v for k, v in os.environ.items() if k not in FLAGS}
    env.update(PYTHONPATH=pythonpath if pythonpath is not None else f"{src}:{HERE}", PYTHONUNBUFFERED="1",
               Q38MM_SRC_ROOT=src, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", JAX_PLATFORMS="cpu", **(extra or {}))
    try:
        return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout, cwd=ROOT)
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        return subprocess.CompletedProcess(cmd, 124, out, "TIMEOUT")


def pytest(src, tests, args=(), extra=None):
    p = run([sys.executable, "-u", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "q38_l4vfy_origin",
             "--noconftest", "-rfEs", *args, *tests], src, extra)
    lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
    tail = next((ln for ln in reversed(lines) if re.search(r"\d+ (passed|failed|error|deselected|skipped)", ln)),
                lines[-1] if lines else "<no output>")
    m = re.search(r"^Q38_L4VFY_ORIGIN (.*)$", p.stdout, re.M)
    want = os.path.realpath(os.path.join(ROOT, src)) + os.sep
    vals = [kv.split("=", 1)[1] for kv in m.group(1).split()] if m else []
    origin_ok = bool(m) and any(v != "None" for v in vals) and all(
        v == "None" or os.path.realpath(os.path.join(ROOT, v)).startswith(want) for v in vals)
    failed = re.findall(r"^FAILED (\S+)", p.stdout, re.M)
    num = lambda pat: int((re.findall(pat, tail) or ["0"])[-1])
    return dict(rc=p.returncode, tail=tail, out=p.stdout + p.stderr, origin_ok=origin_ok,
                origin=m.group(0) if m else "Q38_L4VFY_ORIGIN <missing>", failed=failed, npass=num(r"(\d+) passed"),
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


def counts_ok(path, expect):
    try:
        got = json.load(open(path))
    except (OSError, ValueError):
        return None, ["<counts missing>"]
    bad = [f"{k}={got.get(k)}!={v}" for k, v in expect.items() if got.get(k) != v]
    extra = set(got) - set(expect)
    if extra:
        bad.append(f"unexpected keys {sorted(extra)}")
    return got, bad


ok = True
# ---- IMPORT
PROBE = ("import importlib, os\nfor n in ('vllm_torchtpu', 'vllm_torchtpu.envs', 'vllm_torchtpu.compilation.tpu_compiler',"
         " 'vllm_torchtpu.runner.vfy_one', 'vllm_torchtpu.runner.tpu_runner'):\n"
         "    print('Q38_ORIGIN', n, os.path.realpath(importlib.import_module(n).__file__))\n")
p = run([sys.executable, "-u", "-c", PROBE], "src")
got = re.findall(r"^Q38_ORIGIN (\S+) (\S+)$", p.stdout, re.M)
inside = [(n, f, f.startswith(SRC_REAL)) for n, f in got]
for n, f, i in inside:
    print(f"IMPORT {n} {f} {'INSIDE' if i else 'OUTSIDE'}", flush=True)
imp = p.returncode == 0 and len(inside) == 5 and all(i for _, _, i in inside)
if not imp:
    print(f"IMPORT rc={p.returncode}\n{p.stderr[-4000:]}", flush=True)
empty = tempfile.mkdtemp(prefix="q38_l4vfy_empty_")
p = run([sys.executable, "-u", "-c", "import vllm_torchtpu, os; print('Q38_ORIGIN', os.path.realpath(vllm_torchtpu.__file__))"],
        "src", pythonpath=empty)
fb_file = (re.findall(r"^Q38_ORIGIN (\S+)$", p.stdout, re.M) or ["none"])[-1]
fallback_killed = not fb_file.startswith(SRC_REAL)
print(f"IMPORT MUTANT FALLBACK (PYTHONPATH without src) -> {fb_file} -> {'KILLED' if fallback_killed else 'SURVIVED'}",
      flush=True)
shutil.rmtree(empty, ignore_errors=True)
rdir = tempfile.mkdtemp(prefix="q38_l4vfy_resolve_")
shutil.copytree("src", os.path.join(rdir, "src"))
with open(os.path.join(rdir, VFY), "a") as f:
    f.write("\nraise RuntimeError('q38 VFY resolution mutant')\n")
p = run([sys.executable, "-u", "-c", PROBE], os.path.join(rdir, "src"), pythonpath=f"{os.path.join(rdir, 'src')}:{HERE}")
resolution_killed = p.returncode != 0 and "q38 VFY resolution mutant" in p.stderr
print(f"IMPORT MUTANT RESOLUTION (raise appended to the staged vfy_one.py) -> rc={p.returncode} -> "
      f"{'KILLED' if resolution_killed else 'SURVIVED'}", flush=True)
shutil.rmtree(rdir, ignore_errors=True)
imp_ok = imp and fallback_killed and resolution_killed
print(f"IMPORT -> {'PASS' if imp_ok else 'FAIL'}", flush=True)
ok &= imp_ok

# ---- CLEAN
tmpd = tempfile.mkdtemp(prefix="q38_l4vfy_counts_")
c1, c2 = os.path.join(tmpd, "one.json"), os.path.join(tmpd, "agree.json")
r1 = pytest("src", [T_ONE], (), {"Q38_L4VFY_COUNTS": c1})
r2 = pytest("src", [T_AGREE], (), {"Q38_L4VFY_AGREE_COUNTS": c2})
r3 = pytest("src", [T_D83])
r4 = pytest("src", [T_FLAG])
k1, bad1 = counts_ok(c1, EXPECT_COUNTS)
k2, bad2 = counts_ok(c2, EXPECT_AGREE_COUNTS)
clean_failed = {"one": list(r1["failed"]), "agree": list(r2["failed"]), "reg": [], "flag": list(r4["failed"])}
cl = lambda r, n: r["rc"] == 0 and r["origin_ok"] and r["npass"] == n and not (r["nfail"] or r["nerr"] or r["nskip"])
clean1, clean2, clean3 = cl(r1, EXPECT_ONE) and not bad1, cl(r2, EXPECT_AGREE) and not bad2, cl(r3, EXPECT_D83)
clean4 = cl(r4, EXPECT_FLAG)
show(f"CLEAN one -> {'PASS' if clean1 else 'FAIL'} (expect exactly {EXPECT_ONE} passed)", r1)
print(f"  Q38_L4VFY_COUNTS {json.dumps(k1, sort_keys=True)} {'OK' if not bad1 else 'BAD ' + '; '.join(bad1)}", flush=True)
show(f"CLEAN agree -> {'PASS' if clean2 else 'FAIL'} (expect exactly {EXPECT_AGREE} passed)", r2)
print(f"  Q38_L4VFY_AGREE_COUNTS {json.dumps(k2, sort_keys=True)} {'OK' if not bad2 else 'BAD ' + '; '.join(bad2)}",
      flush=True)
r3o = r3["origin"]
print(f"CLEAN d83 -> {'PASS' if clean3 else 'FAIL'} (expect exactly {EXPECT_D83} passed) rc={r3['rc']} :: {r3['tail']}",
      flush=True)
show(f"CLEAN flagoff -> {'PASS' if clean4 else 'FAIL'} (expect exactly {EXPECT_FLAG} passed)", r4)
for tag, r, c in (("one", r1, clean1), ("agree", r2, clean2), ("d83", r3, clean3), ("flagoff", r4, clean4)):
    if not c:
        print(f"---- CLEAN {tag} output tail\n" + r["out"][-15000:], flush=True)
clean = clean1 and clean2 and clean3 and clean4
shutil.rmtree(tmpd, ignore_errors=True)
ok &= clean

# ---- MUTANTS
killed = invalid = 0
TARGET = {"one": [T_ONE], "agree": [T_AGREE], "reg": [T_REG, "-k", "VFY_ONE or vfy_one"], "flag": [T_FLAG]}
if MODE == "mutants":
    def one(name):
        rel, fn, old, new, targets = MUTANTS[name]
        text = apply_mutant(rel, fn, old, new)
        if text is None:
            return name, "ABORT", f"the anchor was not found exactly once in {fn or rel}", None
        try:
            ast.parse(text)
        except SyntaxError as e:
            return name, "ABORT", f"the mutated file does not parse: {e}", None
        d = tempfile.mkdtemp(prefix=f"q38_l4vfy_{name}_")
        try:
            shutil.copytree("src", os.path.join(d, "src"))
            with open(os.path.join(d, rel), "w") as f:
                f.write(text)
            tests = [x for t in targets for x in TARGET[t]]
            desel = [a for t in targets for f in clean_failed[t] for a in ("--deselect", f)]
            r = pytest(os.path.join(d, "src"), tests, ("-x", *desel))
        finally:
            shutil.rmtree(d, ignore_errors=True)
        allc = {f for t in targets for f in clean_failed[t]}
        if not r["origin_ok"] or r["nerr"] or "error" in r["tail"]:
            verdict = "INVALID"
        elif r["rc"] == 1 and r["nfail"] >= 1 and not set(r["failed"]) & allc:
            verdict = "KILLED"
        else:
            verdict = "SURVIVED"
        return name, verdict, "", r

    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        results = list(ex.map(one, SELECTED))
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
    mid = {p: md5(p) for p in WATCH}
    print(f"  md5 after the mutants (copies only): {'== start' if mid == START else '!= START'}", flush=True)
    ok &= mid == START
    ok &= killed == len(SELECTED) and len(SELECTED) > 0

END = {p: md5(p) for p in WATCH}
unchanged = END == START
print("SRC-AT-END %s" % ("UNCHANGED" if unchanged else "CHANGED: " + " ".join(p for p in WATCH if END[p] != START[p])),
      flush=True)
ok &= unchanged
mut = (f" range={LO}-{HI} mutants_killed={killed}/{len(SELECTED)} invalid={invalid} of_total={len(NAMES)}"
       if MODE == "mutants" else "")
print(f"Q38_L4VFY_SUMMARY mode={MODE} src={srch} import={'PASS' if imp_ok else 'FAIL'} "
      f"clean={'PASS' if clean else 'FAIL'}{mut} src_at_end={'UNCHANGED' if unchanged else 'CHANGED'} "
      f"VERDICT={'PASS' if ok else 'FAIL'}", flush=True)
sys.exit(0 if ok else 1)
