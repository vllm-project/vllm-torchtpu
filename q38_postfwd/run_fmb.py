"""q38 DEC-FB-3 FAST_METADATA_BUILD proof: MM's VLLM_TPU_FAST_METADATA_BUILD (dongruizi's c2ae6a9a5, cherry-picked -x)
on the lever-2 head 648f9cdb1, registered compile-ignored. Run from the tree root (CPU job).
Legs (Q38_ORIGIN in every pytest leg; a leg that imported a module under test from outside its own src/ reads INVALID):
  IMPORT      tpu_runner imports, torch version.
  CLEAN       tests/runner/test_fast_metadata_build_port.py (gate, fallback, plan rebuild, switch-off AST identity with
              648f9cdb1's tpu_runner, default), tests/layers/core/test_attention_metadata.py (MM's equality test and the
              bitwise bucket sweep), the compile-key registration test (the flag must not move the hash).
  ORDER16     the 16 drafter / KV-connector ordering tests.
  MUTANTS     19, each on a private copy of src/, run before REGRESSION / COV; killed when the CLEAN invocation has
              rc 1 with >= 1 failed test and no error; md5 of the tree's files after each must equal SRC-AT-START.
  REGRESSION  per file (-v, --durations, timeouts; a hang prints HUNG <node> and reads INCOMPLETE), the runner /
              spec-decode / metadata suites on the tree and on a base copy (648f9cdb1's three files over src, which is
              648f9cdb1's src exactly): a test failing or erroring on the tree must fail or error on the base too.
  FLAGON      the same suites on the tree with VLLM_TPU_FAST_METADATA_BUILD=1 against the tree with it unset: a test
              passing with the flag off but not with it on is a finding.
  COV         (coverage leg only) the pinned q38_cpu_device plugin + VLLM_TARGET_DEVICE=cpu on the three files whose
              `device` tests error on CPU pods, tree vs base: 0 tests passing on the base but not on the tree.
Last line: Q38_FMB_SUMMARY ... VERDICT=PASS|INCOMPLETE|FAIL."""
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.getcwd()
HERE = os.path.join(ROOT, "q38_postfwd")
COV = os.path.join(ROOT, "q38_cov51")
PKG = "src/vllm_torchtpu"
RUNNER = PKG + "/runner/tpu_runner.py"
ENVS = PKG + "/envs.py"
COMPILER = PKG + "/compilation/tpu_compiler.py"
T_PORT = "tests/runner/test_fast_metadata_build_port.py"
T_AM = "tests/layers/core/test_attention_metadata.py"
T_KEY = "tests/compilation/test_tpu_compile_key_registration.py"
T_LC = "tests/runner/test_tpu_runner_kv_connector_lifecycle.py"
K16 = "test_device_drafter_runs_before_kv_connector_save or test_spec_kv_fence_runs_after_drafter_before_connector_save"
REGRESSION = ["tests/runner/test_tpu_runner.py", T_LC, T_AM, "tests/runner/test_draft_carries_port.py",
              "tests/runner/test_carry_rows_fresh_port.py", "tests/spec_decode/test_speculative_decoding_manager.py",
              "tests/spec_decode/test_eagle3.py", "tests/spec_decode/test_draft_top_tokens_port.py"]
COVFILES = ["tests/spec_decode/test_eagle3.py", "tests/runner/test_tpu_runner.py",
            "tests/spec_decode/test_speculative_decoding_manager.py"]
BASE = {ENVS: "envs.py", RUNNER: "runner_tpu_runner.py", COMPILER: "compilation_tpu_compiler.py"}
FLAGS = ("VLLM_TPU_FAST_METADATA_BUILD", "VLLM_TPU_DRAFT_TOP_TOKENS", "VLLM_TPU_DRAFT_TOP_TOKENS_CHECK",
         "VLLM_TPU_DRAFT_ROWS_FRESH", "VLLM_TPU_DRAFT_CARRY_ROWS_FRESH", "VLLM_TPU_DRAFT_CARRY_FUSED",
         "VLLM_TPU_DRAFT_SEED_EXTRAS", "VLLM_TPU_VERIFY_FUSED_READ_OFFSETS", "VLLM_TPU_POSTFWD_FUSED",
         "VLLM_TPU_SPEC_TAIL_COMPILED", "VLLM_TPU_PROMPT_IDS_NDARRAY", "Q38MM_DM_TOKHASH", "VLLM_TPU_VPS_VERIFY",
         "Q38MM_SRC_ROOT")
MUTANTS = {  # name: (file, exact original text, replacement)
    "f1_plan_cache_never_invalidated": (RUNNER, "        if plan is None or plan[0] is not self.attn_groups:\n",
                                        "        if plan is None:\n"),
    "f2_staged_check_dropped": (RUNNER, "            if gid not in staged or (\n", "            if False or (\n"),
    "f3_products_check_dropped": (RUNNER, "                is_mamba and unified and (products is None or gid not in products)\n",
                                  "                False\n"),
    "f4_compact_state_dropped": (RUNNER, "                if compact is not None:\n                    state = compact\n",
                                 "                if compact is not None:\n                    state = None\n"),
    "f5_ckpt_dropped": (RUNNER, "                    state, ckpt = products[gid]\n",
                        "                    state, ckpt = products[gid][0], None\n"),
    "f6_unified_list_not_appended": (RUNNER, "                    ctx.unified_mamba_state_indices.append(state)\n",
                                     "                    pass\n"),
    "f7_shared_fields_swapped": (RUNNER, "            seq_lens=ctx.seq_lens,\n            query_start_loc=ctx.query_start_loc,\n",
                                 "            seq_lens=ctx.query_start_loc,\n            query_start_loc=ctx.seq_lens,\n"),
    "f8_positions_wrong": (RUNNER, "            input_positions=self.position_ids,\n            seq_lens=ctx.seq_lens,\n",
                           "            input_positions=ctx.seq_lens,\n            seq_lens=ctx.seq_lens,\n"),
    "f9_layout_version_off": (RUNNER, "            sequence_layout_version=descriptor.version,\n",
                              "            sequence_layout_version=descriptor.version + 1,\n"),
    "f10_gate_ignores_override": (RUNNER, "            and ctx.staged_block_tables is not None\n            and ctx.position_ids_override is None\n",
                                  "            and ctx.staged_block_tables is not None\n"),
    "f11_gate_ignores_flag": (RUNNER, "            envs.VLLM_TPU_FAST_METADATA_BUILD\n            and ctx.staged_block_tables is not None\n",
                              "            ctx.staged_block_tables is not None\n"),
    "f12_flag_hashed": (COMPILER, "    \"VLLM_TPU_FAST_METADATA_BUILD\",\n", ""),
    "f13_fallback_empty_dict": (RUNNER, "            ):\n                return None\n        descriptor = ctx.sequence_layout_descriptor\n",
                                "            ):\n                return {}\n        descriptor = ctx.sequence_layout_descriptor\n"),
    "f14_read_offsets_dropped": (RUNNER, "                    mamba_slot_read_offsets=ctx.mamba_slot_read_offsets,\n                    mamba_ckpt_indices=ckpt,\n",
                                 "                    mamba_slot_read_offsets=None,\n                    mamba_ckpt_indices=ckpt,\n"),
    "f15_mamba_distribution_wrong": (RUNNER, "                    mamba_request_distribution=ctx.mamba_request_distribution,\n                    **shared,\n",
                                     "                    mamba_request_distribution=ctx.request_distribution,\n                    **shared,\n"),
    "f16_attn_group_gets_mamba_state": (RUNNER, "                built = AttentionMetadata(block_tables=staged[gid], **shared)\n",
                                        "                built = AttentionMetadata(block_tables=staged[gid], mamba_slot_read_offsets=ctx.mamba_slot_read_offsets, **shared)\n"),
    # engagement lines (log only, DECODE-LEAD 11-12): each must be caught by a log-capture test
    "f17_plan_line_dropped": (RUNNER, "            logger.info(\n                \"FAST_METADATA_BUILD: inline metadata walk over %d groups (%d mamba)\",\n",
                              "            (\n                \"FAST_METADATA_BUILD: inline metadata walk over %d groups (%d mamba)\",\n"),
    "f18_counter_never_incremented": (RUNNER, "    c[0 if inline else 1] += 1\n    c[2] += 1\n", "    pass\n"),
    "f19_fallback_counted_as_inline": (RUNNER, "    c[0 if inline else 1] += 1\n", "    c[0] += 1\n"),
}
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()  # noqa: E731


def run(cmd, src, extra=None, timeout=2400, cov=False):
    env = {k: v for k, v in os.environ.items() if k not in FLAGS}
    pp = [src, HERE, os.path.join(ROOT, "tests", "runner")] + ([COV] if cov else [])
    env.update(PYTHONPATH=":".join(pp), PYTHONUNBUFFERED="1", **(extra or {}))
    try:
        return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout, cwd=ROOT)
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        return subprocess.CompletedProcess(cmd, 124, out, "TIMEOUT")


def pytest(src, args, extra=None, noconftest=False, cov=False, timeout=2400):
    pre = ["--noconftest"] if noconftest else []
    plug = ["-p", "q38_cpu_device"] if cov else []
    if cov:
        extra = dict(extra or {}, VLLM_TARGET_DEVICE="cpu")
    p = run([sys.executable, "-u", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "q38_origin", *plug,
             *pre, "-rfE", "--tb=short", *args], src, extra, cov=cov, timeout=timeout)
    lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
    tail = next((ln for ln in reversed(lines) if re.search(r"\d+ (passed|failed|error|errors|deselected|skipped)", ln)),
                lines[-1] if lines else "<no output>")
    m = re.search(r"^Q38_ORIGIN (.*)$", p.stdout, re.M)
    origin = dict(kv.split("=", 1) for kv in m.group(1).split()) if m else {}
    want = os.path.realpath(src) + os.sep
    origin_ok = origin.get("vllm_torchtpu", "None") != "None" and all(
        v not in (None, "None") and os.path.realpath(v).startswith(want) for v in origin.values())
    num = lambda w: int((re.findall(r"(\d+) " + w, tail) or ["0"])[-1])  # noqa: E731
    plugline = (re.findall(r"^Q38_CPU_DEVICE .*$", p.stdout, re.M) or ["Q38_CPU_DEVICE <missing>"])[-1]
    return dict(rc=p.returncode, tail=tail, out=p.stdout + p.stderr, origin_ok=origin_ok,
                origin=m.group(0) if m else "Q38_ORIGIN <missing>", npass=num("passed"), nfail=num("failed"),
                errors=bool(re.search(r"\d+ errors?\b", tail)), failed=re.findall(r"^FAILED (\S+)", p.stdout, re.M),
                errored=re.findall(r"^ERROR (\S+)", p.stdout, re.M), plug=plugline,
                passed=set(re.findall(r"^PASSED (\S+)", p.stdout, re.M)))


FILE_TIMEOUT = 900   # per file (DECODE-LEAD 11-12, lever 4 i3's machinery)
TEST_TIMEOUT = 300   # per test, when pytest --help lists --timeout


def leg(src, label, files, timeout_mode, cov=False, extra=None):
    """One file at a time, -v (node ids printed as tests start), --durations=20, a file timeout and, where available,
    --timeout per test. A timeout prints HUNG <node id>; the caller reads the leg INCOMPLETE (never PASS)."""
    agg = dict(failed=[], errored=[], passed=set(), hung=[], origin_ok=True, plugs=[])
    for f in files:
        args = ["-v", "--durations=20"] + ([f"--timeout={TEST_TIMEOUT}"] if timeout_mode else []) + (["-rA"] if cov else [])
        r = pytest(src, [*args, f], extra=extra, cov=cov, timeout=FILE_TIMEOUT)
        show(f"{label} {f}", r)
        if cov:
            print("  " + r["plug"], flush=True)
            agg["plugs"].append(r["plug"])
        if r["rc"] == 124:
            started = re.findall(r"^(tests/\S+::\S+)(?: (PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS))?", r["out"], re.M)
            last = next((n for n, o in reversed(started) if not o), started[-1][0] if started else "<none started>")
            agg["hung"].append(last)
            print(f"  HUNG {last} (file timeout {FILE_TIMEOUT} s)", flush=True)
        else:
            agg["origin_ok"] &= r["origin_ok"]
        for n in re.findall(r"^FAILED (\S+) - Failed: Timeout", r["out"], re.M):
            agg["hung"].append(n)
            print(f"  HUNG {n} (per-test timeout {TEST_TIMEOUT} s)", flush=True)
        agg["failed"] += r["failed"]
        agg["errored"] += r["errored"]
        agg["passed"] |= r["passed"]
        for ln in [x for x in r["out"].splitlines() if re.match(r"^\d+\.\d+s (call|setup|teardown) ", x)][:3]:
            print("  DURATION " + ln, flush=True)
    return agg


def show(tag, r, origin=True):
    o = ("OK" if r["origin_ok"] else "INVALID") if origin else "n/a (stubbed packages)"
    print(f"{tag} rc={r['rc']} passed={r['npass']} failed={r['nfail']} errors={r['errors']} origin={o} :: {r['tail']}",
          flush=True)
    if origin:
        print("  " + r["origin"], flush=True)


def copy_src(tag, edits=(), base=False):
    d = tempfile.mkdtemp(prefix=f"q38_fmb_{tag}_")
    shutil.copytree(os.path.join(ROOT, "src"), os.path.join(d, "src"))
    if base:
        for rel, name in BASE.items():
            shutil.copyfile(os.path.join(HERE, "base_648f9cdb1", name), os.path.join(d, rel))
    for rel, old, new in edits:
        p = os.path.join(d, rel)
        s = open(p).read()
        assert s.count(old) == 1, (tag, rel, s.count(old))
        open(p, "w").write(s.replace(old, new))
    return os.path.join(d, "src")




def main():
    files = [os.path.join(ROOT, f) for f in (RUNNER, ENVS, COMPILER, T_PORT, T_AM, T_KEY, T_LC,
                                             "tests/runner/tpu_runner_ref_648f9cdb1.py.txt")]
    files.append(os.path.abspath(__file__))
    start = {p: md5(p) for p in files}
    print("SRC-AT-START " + " ".join("%s=%s" % (os.path.relpath(p, ROOT), start[p]) for p in files), flush=True)
    same = lambda: all(md5(p) == start[p] for p in files)  # noqa: E731
    ok = True
    src = os.path.join(ROOT, "src")
    p = run([sys.executable, "-u", "-c", "import vllm_torchtpu.runner.tpu_runner as m, torch; "
             "print('Q38_IMPORT tpu_runner OK', m.__file__, 'torch', torch.__version__)"], src)
    imp = p.returncode == 0
    print((p.stdout.strip().splitlines() or ["<no stdout>"])[-1] if imp else f"Q38_IMPORT FAIL rc={p.returncode}", flush=True)
    if not imp:
        print(p.stderr[-4000:], flush=True)
    ok &= imp
    # ---- CLEAN
    a = pytest(src, [T_PORT, T_AM, T_KEY])
    clean = a["rc"] == 0 and a["origin_ok"]
    show(f"CLEAN (port + metadata + compile key) -> {'PASS' if clean else 'FAIL'}", a)
    if not clean:
        print(a["out"][-8000:], flush=True)
    r16 = pytest(src, [T_LC, "-k", K16])
    lc16 = r16["rc"] == 0 and r16["npass"] == 16 and r16["origin_ok"]
    show(f"ORDER16 -> {'PASS' if lc16 else 'FAIL'}", r16)
    ok &= clean and lc16
    # ---- MUTANTS (before REGRESSION / COV)
    killed = invalid = 0
    for name, (rel, old, new) in MUTANTS.items():
        try:
            msrc = copy_src(name, ((rel, old, new),))
        except AssertionError as e:
            invalid += 1
            print(f"MUTANT {name} INVALID anchor {e}", flush=True)
            continue
        r = pytest(msrc, [T_PORT, T_AM, T_KEY])
        # i2 (after i1 j-a53d72e1): judged only after a clean CLEAN, and a kill needs a test that passed in CLEAN
        new_fail = sorted(set(r["failed"]) - set(a["failed"]))
        k = clean and r["rc"] == 1 and bool(new_fail) and not r["errors"] and r["origin_ok"]
        bad = clean and (not k) and r["rc"] not in (0, 1)
        killed += k
        invalid += bad
        word = ("KILLED" if k else ("INVALID" if bad else "SURVIVED")) if clean else "NOT-JUDGED (CLEAN failed)"
        print(f"MUTANT {name} -> {word} rc={r['rc']} new failures {len(new_fail)} :: {r['tail']} "
              f":: tree {'== start' if same() else '!= START'}", flush=True)
        for f in new_fail[:4]:
            print("  FAILED " + f, flush=True)
        shutil.rmtree(os.path.dirname(msrc), ignore_errors=True)
    # ---- REGRESSION A/B
    pt = run([sys.executable, "-m", "pytest", "--help", "-p", "no:cacheprovider"], src, timeout=300)
    timeout_mode = "--timeout=" in pt.stdout
    print(f"TIMEOUT MODE: {'pytest-timeout --timeout=%d per test' % TEST_TIMEOUT if timeout_mode else 'no pytest-timeout; file-level only'}"
          f" + file timeout {FILE_TIMEOUT} s", flush=True)
    rt = leg(src, "REGRESSION tree", REGRESSION, timeout_mode)
    bsrc = copy_src("base", base=True)
    rb = leg(bsrc, "REGRESSION base copy (648f9cdb1's three files)", REGRESSION, timeout_mode)
    bad_tree = set(rt["failed"]) | set(rt["errored"])
    bad_base = set(rb["failed"]) | set(rb["errored"])
    caused = sorted(bad_tree - bad_base - set(rt["hung"]))
    print(f"REGRESSION tree failed {len(rt['failed'])} errored {len(rt['errored'])} hung {len(rt['hung'])}; base failed "
          f"{len(rb['failed'])} errored {len(rb['errored'])} hung {len(rb['hung'])}; failing or erroring on the tree but on "
          f"neither list on the base: {len(caused)}", flush=True)
    for f in caused[:20]:
        print("  CAUSED " + f, flush=True)
    cls = {"errors_on_base": 0, "fails_on_base": 0, "neither": 0}
    for f in sorted(set(rt["errored"])):
        kk = "errors_on_base" if f in rb["errored"] else ("fails_on_base" if f in rb["failed"] else "neither")
        cls[kk] += 1
        if kk != "errors_on_base":
            print("  tree ERROR %-14s %s" % (kk, f), flush=True)
    print("REGRESSION tree errors classified %s" % cls, flush=True)
    reg_inc = bool(rt["hung"] or rb["hung"])
    regression = rt["origin_ok"] and rb["origin_ok"] and not caused
    reg_word = "INCOMPLETE" if reg_inc else ("PASS" if regression else "FAIL")
    print(f"REGRESSION -> {reg_word}", flush=True)
    ok &= regression and not reg_inc
    # ---- FLAGON (tree, flag on vs off, the same suites)
    ron = leg(src, "FLAGON tree (VLLM_TPU_FAST_METADATA_BUILD=1)", REGRESSION, timeout_mode,
              extra={"VLLM_TPU_FAST_METADATA_BUILD": "1"})
    bad_on = set(ron["failed"]) | set(ron["errored"])
    on_new = sorted(bad_on - bad_tree - set(ron["hung"]))
    print(f"FLAGON failing or erroring with the flag on but not off: {len(on_new)}", flush=True)
    for f in on_new[:20]:
        print("  FLAGON " + f, flush=True)
    on_inc = bool(ron["hung"])
    flagon = ron["origin_ok"] and not on_new
    on_word = "INCOMPLETE" if on_inc else ("PASS" if flagon else "FAIL")
    print(f"FLAGON -> {on_word}", flush=True)
    ok &= flagon and not on_inc
    # ---- COV (plugin; coverage leg only)
    cov_t = leg(src, "COV tree (plugin)", COVFILES, timeout_mode, cov=True)
    cov_b = leg(bsrc, "COV base copy (plugin)", COVFILES, timeout_mode, cov=True)
    newbad = sorted(n for n in cov_b["passed"] if n not in cov_t["passed"])
    print(f"COV passing on the base but not on the tree: {len(newbad)}", flush=True)
    for n in newbad[:20]:
        print("  COV REGRESSION " + n, flush=True)
    cov_inc = bool(cov_t["hung"] or cov_b["hung"])
    engaged = all("-> False" in x for x in cov_t["plugs"] + cov_b["plugs"])
    cov_ok = cov_t["origin_ok"] and cov_b["origin_ok"] and not newbad and engaged
    cov_word = "INCOMPLETE" if cov_inc else ("PASS" if cov_ok else "FAIL")
    print(f"COV -> {cov_word} (plugin engaged on every file: {engaged})", flush=True)
    shutil.rmtree(os.path.dirname(bsrc), ignore_errors=True)
    ok &= cov_ok and not cov_inc
    end = {p: md5(p) for p in files}
    print("SRC-AT-END " + " ".join("%s=%s" % (os.path.relpath(p, ROOT), end[p]) for p in files), flush=True)
    unchanged = end == start
    print("SRC-AT-END %s" % ("UNCHANGED" if unchanged else "CHANGED"), flush=True)
    ok &= killed == len(MUTANTS) and invalid == 0 and unchanged
    inc = reg_inc or on_inc or cov_inc
    core = imp and clean and lc16 and not caused and not on_new and not newbad and unchanged and killed == len(MUTANTS) and invalid == 0
    print("Q38_FMB_SUMMARY import=%s clean=%s order16=%s regression=%s flagon=%s cov=%s mutants_killed=%d/%d invalid=%d "
          "src_at_end=%s VERDICT=%s" % (
              "OK" if imp else "FAIL", "PASS" if clean else "FAIL", "PASS" if lc16 else "FAIL", reg_word, on_word, cov_word,
              killed, len(MUTANTS), invalid, "UNCHANGED" if unchanged else "CHANGED",
              "PASS" if ok else ("INCOMPLETE" if (inc and core) else "FAIL")), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
