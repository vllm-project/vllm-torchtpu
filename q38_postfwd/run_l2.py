"""q38 DEC-FB-3 lever 2 proof: MM DSTOCK's VLLM_TPU_DRAFT_TOP_TOKENS draft sites (6dcff75dc) plus q38 D-MK's inline
check (VLLM_TPU_DRAFT_TOP_TOKENS_CHECK), on lever 3's head 066323583. Run from the tree root (CPU job).
Legs (Q38_ORIGIN in every conftest pytest leg; a leg that imported a module under test from outside its own src/
reads INVALID):
  IMPORT      eagle3 imports, torch version.
  CLEAN       A: tests/spec_decode/test_draft_top_tokens_port.py, lever 3's test_carry_rows_fresh_port.py (its
              flags-off identity now runs on lever 2's switch-off source), P-c's test_draft_carries_port.py and the
              compile-key registration test. B: tests/q38mm_dstock/test_vps_wiring.py (--noconftest; it stubs vllm
              and reads the source under Q38MM_SRC_ROOT), with 6dcff75dc's two draft-site tests un-skipped.
  ORDER16     the 16 drafter / KV-connector ordering tests.
  REGRESSION  runner / spec-decode suites on the tree and on a base copy (066323583's six files laid over src, which
              is 066323583's src exactly); a test failing or erroring (-rfE) on the tree must fail or error on the
              base too; every tree error is classified.
  COV         (DECODE-LEAD / lead 11-12, coverage leg only) the three files whose `device` tests error on CPU pods,
              with the pinned plugin q38_cpu_device (use_tpu=False) and VLLM_TARGET_DEVICE=cpu, on the tree and on
              the base copy: per-file counts, and any test passing on the base but not on the tree (a regression).
              Plus a reach probe (an inserted raise) in `_propose_token_body`'s switch branch, judged on the tests
              that pass there, with a control probe on the stock path.
  MUTANTS     20, each on a private copy of src/; killed when either CLEAN invocation has rc 1 with >= 1 failed test
              and no error, and neither is INVALID; md5 of the tree's files after each must equal SRC-AT-START.
REGRESSION and COV run per file with -v / --durations / timeouts; a hang prints HUNG <node> and reads INCOMPLETE,
never PASS (DECODE-LEAD 11-12).
Last line: Q38_L2_SUMMARY ... VERDICT=PASS|INCOMPLETE|FAIL."""
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
EAGLE3 = PKG + "/spec_decode/eagle3.py"
LOOP = PKG + "/spec_decode/mtp_fused_loop.py"
MTP = PKG + "/models/vllm/qwen4_exp/mtp.py"
VW = PKG + "/runner/vps_wiring.py"
ENVS = PKG + "/envs.py"
COMPILER = PKG + "/compilation/tpu_compiler.py"
RUNNER = PKG + "/runner/tpu_runner.py"
T_DTT = "tests/spec_decode/test_draft_top_tokens_port.py"
T_VPS = "tests/q38mm_dstock/test_vps_wiring.py"
T_CRF = "tests/runner/test_carry_rows_fresh_port.py"
T_DC = "tests/runner/test_draft_carries_port.py"
T_KEY = "tests/compilation/test_tpu_compile_key_registration.py"
T_LC = "tests/runner/test_tpu_runner_kv_connector_lifecycle.py"
K16 = "test_device_drafter_runs_before_kv_connector_save or test_spec_kv_fence_runs_after_drafter_before_connector_save"
REGRESSION = ["tests/runner/test_tpu_runner.py", T_LC, "tests/spec_decode/test_speculative_decoding_manager.py",
              "tests/spec_decode/test_eagle3.py", "tests/spec_decode/test_mtp_fused_loop_dims.py"]
COVFILES = ["tests/spec_decode/test_eagle3.py", "tests/runner/test_tpu_runner.py",
            "tests/spec_decode/test_speculative_decoding_manager.py"]
BASE = {ENVS: "envs.py", EAGLE3: "spec_decode_eagle3.py", LOOP: "spec_decode_mtp_fused_loop.py",
        MTP: "models_vllm_qwen4_exp_mtp.py", VW: "runner_vps_wiring.py", COMPILER: "compilation_tpu_compiler.py"}
FLAGS = ("VLLM_TPU_DRAFT_TOP_TOKENS", "VLLM_TPU_DRAFT_TOP_TOKENS_CHECK", "VLLM_TPU_DRAFT_ROWS_FRESH",
         "VLLM_TPU_DRAFT_CARRY_ROWS_FRESH", "VLLM_TPU_DRAFT_CARRY_FUSED", "VLLM_TPU_DRAFT_SEED_EXTRAS",
         "VLLM_TPU_VERIFY_FUSED_READ_OFFSETS", "VLLM_TPU_POSTFWD_FUSED", "VLLM_TPU_SPEC_TAIL_COMPILED",
         "VLLM_TPU_PROMPT_IDS_NDARRAY", "Q38MM_DM_TOKHASH", "VLLM_TPU_VPS_VERIFY", "Q38MM_SRC_ROOT")
CARRY_HEAD = ("        # carry and the first draft token, the same ops in one dispatch.\n"
              "        assert self.draft_model is not None\n        indices = indices.clamp_min(0)\n"
              "        positions_carry = torch.index_select(positions, positions.ndim - 1, indices)\n"
              "        d2t = getattr(self.draft_model, \"draft_id_to_target_id\", None)\n")
BODY_HEAD = ("    def _carry_and_token_body(\n"
             "        self, positions: torch.Tensor, indices: torch.Tensor, last_hidden: torch.Tensor\n"
             "    ) -> tuple[torch.Tensor, torch.Tensor]:\n        assert self.draft_model is not None\n"
             "        indices = indices.clamp_min(0)\n"
             "        positions_carry = torch.index_select(positions, positions.ndim - 1, indices)\n"
             "        d2t = getattr(self.draft_model, \"draft_id_to_target_id\", None)\n")
SW = "        if d2t is None and getattr(self, \"_dvp\", False):\n"
MUTANTS = {  # name: (file, exact original text, replacement)
    "n1_body_switch_ignored": (EAGLE3, "            if getattr(self, \"_dvp\", False):\n"
                               "                return vps_wiring.draft_top_tokens(self.draft_model, hidden)\n",
                               "            if False:\n"
                               "                return vps_wiring.draft_top_tokens(self.draft_model, hidden)\n"),
    "n2_carry_switch_ignored": (EAGLE3, CARRY_HEAD + SW, CARRY_HEAD + "        if d2t is None and False:\n"),
    "n3_body_copy_switch_ignored": (EAGLE3, BODY_HEAD + SW, BODY_HEAD + "        if d2t is None and False:\n"),
    "n4_loop_switch_ignored": (LOOP, "            if self.dvp:\n", "            if False:\n"),
    "n5_loop_not_given_switch": (EAGLE3, "                    dvp=self._dvp,\n", ""),
    "n6_enabled_false": (VW, "    return bool(_envs().VLLM_TPU_DRAFT_TOP_TOKENS)\n", "    return False\n"),
    "n7_flag_not_in_compile_key": (COMPILER, "_TPU_COMPILE_ENV_IGNORED = {\n",
                                   "_TPU_COMPILE_ENV_IGNORED = {\n    \"VLLM_TPU_DRAFT_TOP_TOKENS\",\n"),
    "n8_check_compares_itself": (EAGLE3, "            ref = self._draft_propose_token_full(hidden)\n",
                                 "            ref = tokens\n"),
    "n9_check_never_runs": (EAGLE3, "            if getattr(self, \"_draft_top_checks_left\", 0) > 0:\n",
                            "            if getattr(self, \"_draft_top_checks_left\", 0) > 10**9:\n"),
    "n10_switch_skips_eligibility": (EAGLE3, "            self._dvp = vps_wiring.draft_top_tokens_ok(self.draft_model)\n",
                                     "            self._dvp = True\n"),
    "n11_mtp_not_opted_in": (MTP, "    plain_compute_logits = True\n\n    def compute_logits(",
                             "    plain_compute_logits = False\n\n    def compute_logits("),
    "n12_check_armed_always": (EAGLE3, "                self._draft_top_checks_left = int(envs.VLLM_TPU_DRAFT_TOP_TOKENS_CHECK)\n"
                               "            else:\n",
                               "            else:\n"
                               "                self._draft_top_checks_left = int(envs.VLLM_TPU_DRAFT_TOP_TOKENS_CHECK)\n"),
    "n13_switchoff_arm_changed": (EAGLE3, CARRY_HEAD + SW + "            token = vps_wiring.draft_top_tokens(self.draft_model, last_hidden)\n"
                                  "        elif d2t is None:\n            token = (\n"
                                  "                self.draft_model.compute_logits(last_hidden).argmax(dim=-1).to(torch.int32)\n",
                                  CARRY_HEAD + SW + "            token = vps_wiring.draft_top_tokens(self.draft_model, last_hidden)\n"
                                  "        elif d2t is None:\n            token = (\n"
                                  "                self.draft_model.compute_logits(last_hidden * 1).argmax(dim=-1).to(torch.int32)\n"),
    "n14_check_off_by_one_count": (EAGLE3, "        self._draft_top_checks_left = left - 1\n",
                                   "        self._draft_top_checks_left = left\n"),
    # Closes COV51's U1 (DECODE-LEAD 11-12): lever 3's _propose_token_body stock token path (48745d9b0's refactor),
    # byte-identical to 066323583's apart from lever 2's added switch guard; ML3 survived every test in j-6f95dd75.
    "n15_stock_body_value": (EAGLE3, "                self.draft_model.compute_logits(hidden).argmax(dim=-1).to(torch.int32)\n",
                             "                self.draft_model.compute_logits(hidden).argmax(dim=-1).to(torch.int32) + 1\n"),
    # COV51 U2 (lever 3's ROWS_FRESH propose branch) and U4 (P-c's fold precompile variant), DECODE-LEAD 11-12.
    "u2a_rows_fresh_layout_guard_dropped": (EAGLE3, "                if untrimmed is not None and not has_non_default_layout:\n",
                                            "                if untrimmed is not None:\n"),
    "u2b_rows_fresh_args_reordered": (EAGLE3, "                        fused_token0, hidden_carry = self._draft_token_and_rows(\n"
                                      "                            *untrimmed\n",
                                      "                        fused_token0, hidden_carry = self._draft_token_and_rows(\n"
                                      "                            untrimmed[1], untrimmed[0], untrimmed[2]\n"),
    "u4a_fold_precompile_no_num_reqs": (RUNNER, "                        topk_candidates,\n                        self._num_reqs_tensor(1),\n",
                                        "                        topk_candidates,\n"),
    "u4b_fold_precompile_no_continue": (RUNNER, "                    )\n                )\n                continue\n",
                                        "                    )\n                )\n"),
    "u4c_fold_precompile_flag_ignored": (RUNNER, "        fold_ro = bool(\n            envs.VLLM_TPU_VERIFY_FUSED_READ_OFFSETS\n",
                                         "        fold_ro = bool(\n            True\n"),
}
PROBE = "q38 lever 2 probe"
PROBES = {  # inserted raises for the COV leg (reach, judged on the COV tests that pass on the tree)
    "PR0_control_stock_path": (EAGLE3, "            # No draft->target mapping table. Upstream compute_logits then asserts\n",
                               f"            raise AssertionError(\"{PROBE} stock\")\n"
                               "            # No draft->target mapping table. Upstream compute_logits then asserts\n"),
    "PRL2_switch_branch": (EAGLE3, "                return vps_wiring.draft_top_tokens(self.draft_model, hidden)\n",
                           f"                raise AssertionError(\"{PROBE} switch\")\n"),
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


def leg(src, label, files, timeout_mode, cov=False):
    """One file at a time, -v (node ids printed as tests start), --durations=20, a file timeout and, where available,
    --timeout per test. A timeout prints HUNG <node id>; the caller reads the leg INCOMPLETE (never PASS)."""
    agg = dict(failed=[], errored=[], passed=set(), hung=[], origin_ok=True, plugs=[])
    for f in files:
        args = ["-v", "--durations=20"] + ([f"--timeout={TEST_TIMEOUT}"] if timeout_mode else []) + (["-rA"] if cov else [])
        r = pytest(src, [*args, f], cov=cov, timeout=FILE_TIMEOUT)
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
    d = tempfile.mkdtemp(prefix=f"q38_l2_{tag}_")
    shutil.copytree(os.path.join(ROOT, "src"), os.path.join(d, "src"))
    if base:
        for rel, name in BASE.items():
            shutil.copyfile(os.path.join(HERE, "base_066323583", name), os.path.join(d, rel))
    for rel, old, new in edits:
        p = os.path.join(d, rel)
        s = open(p).read()
        assert s.count(old) == 1, (tag, rel, s.count(old))
        open(p, "w").write(s.replace(old, new))
    return os.path.join(d, "src")


def clean_legs(src):
    a = pytest(src, [T_DTT, T_CRF, T_DC, T_KEY])
    b = pytest(src, [T_VPS], {"Q38MM_SRC_ROOT": src}, noconftest=True)
    return a, b


def main():
    files = [os.path.join(ROOT, f) for f in (EAGLE3, LOOP, MTP, VW, ENVS, COMPILER, RUNNER, T_DTT, T_VPS, T_CRF, T_DC, T_KEY,
                                             T_LC, "tests/spec_decode/dtt_flagoff_ast.py",
                                             "tests/spec_decode/eagle3_ref_066323583.py.txt")]
    files.append(os.path.abspath(__file__))
    start = {p: md5(p) for p in files}
    print("SRC-AT-START " + " ".join("%s=%s" % (os.path.relpath(p, ROOT), start[p]) for p in files), flush=True)
    same = lambda: all(md5(p) == start[p] for p in files)  # noqa: E731
    ok = True
    src = os.path.join(ROOT, "src")
    p = run([sys.executable, "-u", "-c", "import vllm_torchtpu.spec_decode.eagle3 as m, torch; "
             "print('Q38_IMPORT eagle3 OK', m.__file__, 'torch', torch.__version__)"], src)
    imp = p.returncode == 0
    print((p.stdout.strip().splitlines() or ["<no stdout>"])[-1] if imp else f"Q38_IMPORT FAIL rc={p.returncode}", flush=True)
    if not imp:
        print(p.stderr[-4000:], flush=True)
    ok &= imp
    # ---- CLEAN
    a, b = clean_legs(src)
    clean_a = a["rc"] == 0 and a["origin_ok"]
    clean_b = b["rc"] == 0
    show(f"CLEAN A (port + lever-3 + P-c tests + compile key) -> {'PASS' if clean_a else 'FAIL'}", a)
    show(f"CLEAN B (test_vps_wiring.py, --noconftest, Q38MM_SRC_ROOT=tree) -> {'PASS' if clean_b else 'FAIL'}", b,
         origin=False)
    for r, good in ((a, clean_a), (b, clean_b)):
        if not good:
            print(r["out"][-8000:], flush=True)
    clean = clean_a and clean_b
    r16 = pytest(src, [T_LC, "-k", K16])
    lc16 = r16["rc"] == 0 and r16["npass"] == 16 and r16["origin_ok"]
    show(f"ORDER16 -> {'PASS' if lc16 else 'FAIL'}", r16)
    ok &= clean and lc16
    # ---- MUTANTS (before REGRESSION / COV, so a slow or hung suite cannot starve them)
    killed = invalid = 0
    for name, (rel, old, new) in MUTANTS.items():
        try:
            msrc = copy_src(name, ((rel, old, new),))
        except AssertionError as e:
            invalid += 1
            print(f"MUTANT {name} INVALID anchor {e}", flush=True)
            continue
        ra, rb2 = clean_legs(msrc)
        bad = (ra["rc"] not in (0, 1)) or (rb2["rc"] not in (0, 1)) or not ra["origin_ok"]
        kill = lambda r: r["rc"] == 1 and r["nfail"] >= 1 and not r["errors"]  # noqa: E731
        k = (not bad) and (kill(ra) or kill(rb2))
        killed += k
        invalid += bad
        print(f"MUTANT {name} -> {'KILLED' if k else ('INVALID' if bad else 'SURVIVED')} A rc={ra['rc']} :: {ra['tail']} "
              f"|| B rc={rb2['rc']} :: {rb2['tail']} :: tree {'== start' if same() else '!= START'}", flush=True)
        for f in (ra["failed"] + rb2["failed"])[:4]:
            print("  FAILED " + f, flush=True)
        shutil.rmtree(os.path.dirname(msrc), ignore_errors=True)
    # ---- REGRESSION A/B (per file, -v, durations, timeouts; a hang reads INCOMPLETE)
    pt = run([sys.executable, "-m", "pytest", "--help", "-p", "no:cacheprovider"], src, timeout=300)
    timeout_mode = "--timeout=" in pt.stdout
    print(f"TIMEOUT MODE: {'pytest-timeout --timeout=%d per test' % TEST_TIMEOUT if timeout_mode else 'no pytest-timeout; file-level only'}"
          f" + file timeout {FILE_TIMEOUT} s", flush=True)
    rt = leg(src, "REGRESSION tree", REGRESSION, timeout_mode)
    bsrc = copy_src("base", base=True)
    rb = leg(bsrc, "REGRESSION base copy (066323583's six files)", REGRESSION, timeout_mode)
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
        print("  tree ERROR %-14s %s" % (kk, f), flush=True)
    print("REGRESSION tree errors classified %s" % cls, flush=True)
    reg_inc = bool(rt["hung"] or rb["hung"])
    regression = rt["origin_ok"] and rb["origin_ok"] and not caused
    reg_word = "INCOMPLETE" if reg_inc else ("PASS" if regression else "FAIL")
    print(f"REGRESSION -> {reg_word}", flush=True)
    ok &= regression and not reg_inc
    # ---- COV (plugin; coverage leg only; per file, same machinery)
    cov_t = leg(src, "COV tree (plugin)", COVFILES, timeout_mode, cov=True)
    cov_b = leg(bsrc, "COV base copy (plugin)", COVFILES, timeout_mode, cov=True)
    newbad = sorted(n for n in cov_b["passed"] if n not in cov_t["passed"])
    print(f"COV per file (tree): " + " | ".join(
        f"{f.split('/')[-1]}: passed {sum(1 for n in cov_t['passed'] if n.startswith(f))}, "
        f"failed {sum(1 for n in cov_t['failed'] if n.startswith(f))}, "
        f"error {sum(1 for n in cov_t['errored'] if n.startswith(f))}" for f in COVFILES), flush=True)
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
    covpass = cov_t["passed"]
    probe_v = {}
    for name, (rel, old, new) in PROBES.items():
        try:
            msrc = copy_src(name, ((rel, old, new),))
        except AssertionError as e:
            probe_v[name] = "INVALID(anchor)"
            print(f"PROBE {name} INVALID anchor {e}", flush=True)
            continue
        r = pytest(msrc, ["-rA", *COVFILES], cov=True, timeout=3 * FILE_TIMEOUT)
        lost = sorted(n for n in covpass if n not in r["passed"])
        bad = r["rc"] not in (0, 1) or not r["origin_ok"]
        probe_v[name] = "INVALID" if bad else ("REACHED" if lost else "NOT-REACHED")
        print(f"PROBE {name} -> {probe_v[name]} rc={r['rc']} :: lost {len(lost)} of {len(covpass)} passing :: "
              f"{r['tail']} :: tree {'== start' if same() else '!= START'}", flush=True)
        for n in lost[:3]:
            print("  LOST " + n, flush=True)
        shutil.rmtree(os.path.dirname(msrc), ignore_errors=True)
    if probe_v.get("PR0_control_stock_path") != "REACHED":
        print("PROBE control not REACHED: the switch-branch probe reads NOT-JUDGED", flush=True)
        probe_v["PRL2_switch_branch"] = "NOT-JUDGED"
    end = {p: md5(p) for p in files}
    print("SRC-AT-END " + " ".join("%s=%s" % (os.path.relpath(p, ROOT), end[p]) for p in files), flush=True)
    unchanged = end == start
    print("SRC-AT-END %s" % ("UNCHANGED" if unchanged else "CHANGED"), flush=True)
    ok &= killed == len(MUTANTS) and invalid == 0 and unchanged
    print("Q38_L2_SUMMARY import=%s clean=%s order16=%s regression=%s cov=%s probes=%s mutants_killed=%d/%d "
          "invalid=%d src_at_end=%s VERDICT=%s" % (
              "OK" if imp else "FAIL", "PASS" if clean else "FAIL", "PASS" if lc16 else "FAIL",
              reg_word, cov_word,
              ",".join(f"{k}={v}" for k, v in probe_v.items()), killed, len(MUTANTS), invalid,
              "UNCHANGED" if unchanged else "CHANGED",
              "PASS" if ok else ("INCOMPLETE" if ((reg_inc or cov_inc) and imp and clean and lc16 and not caused
                                                  and not newbad and unchanged and killed == len(MUTANTS)
                                                  and invalid == 0) else "FAIL")), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
