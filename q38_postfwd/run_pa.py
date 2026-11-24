"""q38 DEC-FB-3 POSTFWD P-a re-proof (DECODE-LEAD item 1 GO, 08:1xZ 11-12): q38-decfb3-postfwd-pa = fb1f4ef44 + G2 + the
warm-set fix + owned_args.py (port of dongruizi's 1e044e7d7) + the owned-alias stamp strip on the fused entry. Run from the
tree root:  python3 -u q38_postfwd/run_pa.py
Legs (every pytest leg prints Q38_ORIGIN from inside pytest; a leg that imported a module under test from outside its own
src/ reads INVALID and fails the verdict):
  IMPORT      tpu_runner imports on this image.
  CLEAN       G2 bitwise, runner wiring and the static-args probe tests PASS.
  S2C-OFF/ON  the candidate's s2c suites (P15 L1a Q38_P15_EXPECT_ADMP=1: rc 0, >= 7 passed, 0 failed; KVB1W: rc 0; L1b
              descriptive) with VLLM_TPU_POSTFWD_FUSED unset, then =1.
  M7          with the flag on, s2c's M7 must still FAIL test_adm_p_prompt_tails_ship_untouched with a "PTA-ADMP:" line.
  REGRESSION  CPU (not tpu_test) runner / warmup / lifecycle / spec-manager tests; each failure re-run on a src copy with
              fb1f4ef44's envs.py, tpu_runner.py and tpu_runner_async_output.py; counts against P-a only if it passes there.
  MUTANTS     G2 m1-m6 (run_g2.py), wiring w1-w3 (run_wiring.py), static-args s1-s4 (below); each on a private copy of
              src/, each must FAIL its suite; after each, the main tree's mutated file is re-hashed and must equal its
              start md5 (the copies never touch it).
  SRC-AT-END  md5 of every file under test and every test file, at start and end; must be unchanged.
Final line Q38_POSTFWD_PA_SUMMARY; exit 0 only if every leg behaves as required."""
import hashlib, json, os, re, shutil, subprocess, sys, tempfile

ROOT = os.getcwd()
HERE = os.path.join(ROOT, "q38_postfwd")
RUNNER = "src/vllm_torchtpu/runner/tpu_runner.py"
ASYNC = "src/vllm_torchtpu/runner/tpu_runner_async_output.py"
OWNED = "src/vllm_torchtpu/runner/owned_args.py"
ENVS = "src/vllm_torchtpu/envs.py"
T_G2, T_WIRE, T_STATIC = ("tests/runner/test_postfwd_fused.py", "tests/runner/test_postfwd_runner_wiring.py",
                          "tests/runner/test_postfwd_static_args.py")
REGRESSION = ["tests/runner/test_tpu_runner.py", "tests/runner/test_tpu_runner_kv_connector_lifecycle.py",
              "tests/spec_decode/test_speculative_decoding_manager.py"]
BASE = {ENVS: "envs.py", RUNNER: "runner_tpu_runner.py", ASYNC: "runner_tpu_runner_async_output.py"}
P15ENV = {"VLLM_TARGET_DEVICE": "cpu", "TORCH_DEVICE_BACKEND_AUTOLOAD": "0", "Q38_P15_EXPECT_ADMP": "1"}
FLAG_ON = {"VLLM_TPU_POSTFWD_FUSED": "1"}
WATCH = [RUNNER, ASYNC, OWNED, ENVS, T_G2, T_WIRE, T_STATIC, "tests/test_prompt_ids_ndarray_pd.py",
         "src/vllm_torchtpu/core/prompt_ids_ndarray.py"]
STATIC_MUTANTS = {  # name: [(file, exact original text, replacement)]
    "s1_wrapper_dropped": [(RUNNER, "    return static_shape_args_owned(_postfwd_subst_compiled)(*args)\n",
                            "    return _postfwd_subst_compiled(*args)\n")],
    "s2_callsite_bypass": [(RUNNER, "                _postfwd_subst_fused(*args)\n", "                _postfwd_subst_compiled(*args)\n")],
    "s3_alias_keeps_stamp": [(OWNED, "                alias = a.detach()\n", "                alias = a\n")],
    "s4_view_not_owned": [(OWNED, "            return a.clone(memory_format=torch.contiguous_format)\n\n        new_args",
                           "            return a.detach()\n\n        new_args")],
}
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
START = {p: md5(p) for p in WATCH}
print("SRC-AT-START " + " ".join("%s=%s" % (p, START[p]) for p in WATCH), flush=True)


def header(path, stop):
    ns = {}
    exec(open(os.path.join(HERE, path)).read().split(f"\ndef {stop}(")[0], ns)
    return ns


g2 = header("run_g2.py", "pytest")
wire = header("run_wiring.py", "run")


def run(cmd, src, extra=None, cwd=ROOT, timeout=2400):
    env = {k: v for k, v in os.environ.items() if k not in FLAG_ON and k not in ("VLLM_TPU_PROMPT_IDS_NDARRAY", "Q38MM_DM_TOKHASH")}
    env.update(PYTHONPATH=f"{src}:{HERE}", PYTHONUNBUFFERED="1", **(extra or {}))
    try:
        return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout, cwd=cwd)
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        return subprocess.CompletedProcess(cmd, 124, out, "TIMEOUT")


def pytest(src, tests, extra=None, args=(), cwd=ROOT):
    p = run([sys.executable, "-u", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "q38_origin", "-rf",
             *args, *tests], src, extra, cwd)
    lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
    tail = next((ln for ln in reversed(lines) if re.search(r"\d+ (passed|failed|error|deselected)", ln)),
                lines[-1] if lines else "<no output>")
    m = re.search(r"^Q38_ORIGIN (.*)$", p.stdout, re.M)
    origin = dict(kv.split("=", 1) for kv in m.group(1).split()) if m else {}
    want = os.path.realpath(src) + os.sep
    origin_ok = origin.get("vllm_torchtpu", "None") != "None" and all(
        v not in (None, "None") and os.path.realpath(v).startswith(want) for v in origin.values())
    failed = re.findall(r"^FAILED (\S+)", p.stdout, re.M)
    npass = int((re.findall(r"(\d+) passed", tail) or ["0"])[-1])
    nfail = int((re.findall(r"(\d+) failed", tail) or ["0"])[-1])
    return dict(rc=p.returncode, tail=tail, out=p.stdout + p.stderr, origin_ok=origin_ok,
                origin=m.group(0) if m else "Q38_ORIGIN <missing>", failed=failed, npass=npass, nfail=nfail)


def show(tag, r):
    print(f"{tag} rc={r['rc']} origin={'OK' if r['origin_ok'] else 'INVALID'} :: {r['tail']}", flush=True)
    print(f"  {r['origin']}", flush=True)


def copy_tree(tag, edits=(), base=False, full=False):
    d = tempfile.mkdtemp(prefix=f"q38_pa_{tag}_")
    shutil.copytree("src", os.path.join(d, "src"))
    if full:
        shutil.copytree("tests", os.path.join(d, "tests"))
        shutil.copyfile("pyproject.toml", os.path.join(d, "pyproject.toml"))
    if base:
        for rel, name in BASE.items():
            shutil.copyfile(os.path.join(HERE, "base_fb1f4ef44", name), os.path.join(d, rel))
    for rel, text in edits:
        with open(os.path.join(d, rel), "w") as f:
            f.write(text)
    return d


ok = True
p = run([sys.executable, "-u", "-c", "import vllm_torchtpu.runner.tpu_runner as m, torch; "
         "print('Q38_IMPORT tpu_runner OK', m.__file__, 'torch', torch.__version__)"], "src")
imp = p.returncode == 0
print((p.stdout.strip().splitlines() or ["<no stdout>"])[-1] if imp else f"Q38_IMPORT tpu_runner FAIL rc={p.returncode}", flush=True)
if not imp:
    print(p.stderr[-4000:], flush=True)
ok &= imp

r = pytest("src", [T_G2, T_WIRE, T_STATIC])
clean = r["rc"] == 0 and r["origin_ok"]
show(f"CLEAN -> {'PASS' if clean else 'FAIL'}", r)
if not clean:
    print(r["out"][-8000:], flush=True)
ok &= clean

s2c = {}
for tag, extra in (("S2C-OFF", {}), ("S2C-ON", FLAG_ON)):
    l1a = pytest("src", ["tests/test_prompt_ids_ndarray_pd.py"], {**P15ENV, **extra}, args=("-o", "addopts="))
    kv = pytest("src", ["tests/kernels/test_qwen4_exp_kv_block_one_window.py"], extra)
    l1b = pytest("src", ["tests/q38mm_dm/test_prompt_ids_ndarray.py"], {**P15ENV, **extra}, args=("-o", "addopts="))
    l1a_ok = l1a["rc"] == 0 and l1a["npass"] >= 7 and l1a["nfail"] == 0 and l1a["origin_ok"]
    kv_ok = kv["rc"] == 0 and kv["origin_ok"]
    show(f"{tag} P15 L1a passed={l1a['npass']} failed={l1a['nfail']} -> {'PASS' if l1a_ok else 'FAIL'}", l1a)
    show(f"{tag} KVB1W -> {'PASS' if kv_ok else 'FAIL'}", kv)
    show(f"{tag} P15 L1b (descriptive)", l1b)
    for rr, good in ((l1a, l1a_ok), (kv, kv_ok)):
        if not good:
            print(rr["out"][-4000:], flush=True)
    s2c[tag] = l1a_ok and kv_ok
    ok &= s2c[tag]

PG = "src/vllm_torchtpu/core/prompt_ids_ndarray.py"
pg = open(PG).read()
old, new = "        if not isinstance(ids, list):", "        if ids is None or isinstance(ids, np.ndarray):"
if pg.count(old) != 1:
    print(f"M7: ABORT, anchor found {pg.count(old)} times", flush=True)
    m7 = False
else:
    d = copy_tree("m7", [(PG, pg.replace(old, new))], full=True)
    r = pytest(os.path.join(d, "src"), ["tests/test_prompt_ids_ndarray_pd.py"], {**P15ENV, **FLAG_ON},
               args=("-o", "addopts=", "-k", "test_adm_p_prompt_tails_ship_untouched"), cwd=d)
    lines = r["out"].count("PTA-ADMP:")
    m7 = r["origin_ok"] and r["rc"] != 0 and lines >= 1
    show(f"M7 (flag on) PTA-ADMP lines={lines} -> {'BITES' if m7 else 'DOES NOT BITE'}", r)
    shutil.rmtree(d, ignore_errors=True)
    print(f"  md5 after M7 (copy): {PG}={md5(PG)} {'== start' if md5(PG) == START[PG] else '!= START'}", flush=True)
    m7 &= md5(PG) == START[PG]
ok &= m7

r = pytest("src", REGRESSION, args=("-m", "not tpu_test"))
show("REGRESSION new tree", r)
reg_ok = r["origin_ok"] and (r["rc"] == 0 or bool(r["failed"]))
ours = []
if r["failed"]:
    d = copy_tree("base", base=True)
    b = pytest(os.path.join(d, "src"), r["failed"])
    show(f"REGRESSION base copy (fb1f4ef44 envs / tpu_runner / async_output) on the {len(r['failed'])} failures", b)
    ours = [f for f in r["failed"] if f not in set(b["failed"])]
    for f in r["failed"]:
        print(f"  {'OURS ' if f in ours else 'BASE '} {f}", flush=True)
    reg_ok &= b["origin_ok"] and not ours
    shutil.rmtree(d, ignore_errors=True)
elif r["rc"] != 0:
    print(r["out"][-6000:], flush=True)
print(f"REGRESSION -> {'PASS' if reg_ok else 'FAIL'} (failures caused by P-a: {len(ours)})", flush=True)
ok &= reg_ok

texts = {RUNNER: open(RUNNER).read(), ASYNC: open(ASYNC).read(), OWNED: open(OWNED).read()}
plan = []
fn_start = texts[ASYNC].index("def subtract_substitute_extract(")
fn_end = texts[ASYNC].index("\ndef ", fn_start + 10)
for name, (o, n) in g2["MUTANTS"].items():
    fn = texts[ASYNC][fn_start:fn_end]
    edits = [(o, n)] + ([g2["EXTRA"][name]] if name in g2["EXTRA"] else [])
    good = all(fn.count(eo) == 1 for eo, _ in edits)
    for eo, en in edits:
        fn = fn.replace(eo, en)
    plan.append(("g2_" + name, [(ASYNC, texts[ASYNC][:fn_start] + fn + texts[ASYNC][fn_end:])] if good else None, [T_G2]))
for name, edits in wire["MUTANTS"].items():
    t = texts[RUNNER]
    good = all(t.count(o) == 1 for o, _ in edits)
    for o, n in edits:
        t = t.replace(o, n)
    plan.append(("wire_" + name, [(RUNNER, t)] if good else None, [T_WIRE]))
for name, edits in STATIC_MUTANTS.items():
    new = dict(texts)
    good = True
    for rel, o, n in edits:
        if new[rel].count(o) != 1:
            good = False
            break
        new[rel] = new[rel].replace(o, n)
    plan.append(("static_" + name, [(rel, new[rel]) for rel in {e[0] for e in edits}] if good else None, [T_WIRE, T_STATIC]))
killed = invalid = 0
for name, edits, tests in plan:
    if edits is None:
        print(f"MUTANT {name}: ABORT, an anchor was not found exactly once", flush=True)
        ok = False
        continue
    d = copy_tree(name, edits)
    r = pytest(os.path.join(d, "src"), tests)
    verdict = "INVALID" if not r["origin_ok"] else ("KILLED" if r["rc"] != 0 else "SURVIVED")
    killed += verdict == "KILLED"
    invalid += verdict == "INVALID"
    ok &= verdict == "KILLED"
    show(f"MUTANT {name} -> {verdict}", r)
    shutil.rmtree(d, ignore_errors=True)
    for rel, _ in edits:
        same = md5(rel) == START[rel]
        print(f"  md5 after {name} (copy): {rel}={md5(rel)} {'== start' if same else '!= START'}", flush=True)
        ok &= same
END = {p: md5(p) for p in WATCH}
print("SRC-AT-END " + " ".join("%s=%s" % (p, END[p]) for p in WATCH), flush=True)
unchanged = END == START
print("SRC-AT-END %s" % ("UNCHANGED" if unchanged else "CHANGED: " + " ".join(p for p in WATCH if END[p] != START[p])), flush=True)
ok &= unchanged
print(f"Q38_POSTFWD_PA_SUMMARY import={'OK' if imp else 'FAIL'} clean={'PASS' if clean else 'FAIL'} "
      f"s2c_off={'PASS' if s2c.get('S2C-OFF') else 'FAIL'} s2c_on={'PASS' if s2c.get('S2C-ON') else 'FAIL'} "
      f"m7={'BITES' if m7 else 'FAIL'} regression={'PASS' if reg_ok else 'FAIL'} "
      f"mutants_killed={killed}/{len(plan)} invalid={invalid} src_at_end={'UNCHANGED' if unchanged else 'CHANGED'} "
      f"VERDICT={'PASS' if ok else 'FAIL'}", flush=True)
sys.exit(0 if ok else 1)
