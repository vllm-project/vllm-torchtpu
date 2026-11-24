"""q38 DEC-FB-3 lever 4 proof: D-MK's VLLM_TPU_QSA_DCP_ONE_GATHER (99df0bbfe, from 80b612e33's 36624958c) on the
lever-3 branch aa83ece72, plus its envs registration (D-MK's cdb903755 lines). Run from the tree root (CPU job).
Legs (Q38_ORIGIN in every pytest leg):
  IMPORT      qsa_op imports.
  CLEAN       D-MK's test_qsa_dcp_one_gather.py (on == off bit for bit through _dcp_finish, 2 -> 1 collectives),
              test_qsa_dcp_one_gather_port.py (flag-off branch = the old two gathers, AST; the flag is read; mixed-dtype
              fallback) and the compile-key registration test (the flag hashed).
  MUTANTS run before REGRESSION (i2).
  REGRESSION per file with -v / --durations / per-test timeouts; a hang prints HUNG <node> and reads INCOMPLETE (i3).
  REGRESSION  the qsa_op-importing suites (minus test_o1s_request_shard.py, i2) on the tree and on a base copy (aa83ece72's qsa_op.py + envs.py over src, which
              is aa83ece72's src exactly); every tree failure must also fail on the base.
  MUTANTS     6, each on a private copy of src/; strict kill: rc 1, >= 1 failed test, no error; md5 after each.
Last line: Q38_L4_SUMMARY ... VERDICT=PASS|FAIL."""
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.getcwd()
HERE = os.path.join(ROOT, "q38_postfwd")
QSAOP = "src/vllm_torchtpu/layers/adapter/custom_ops/qsa_op.py"
ENVS = "src/vllm_torchtpu/envs.py"
COMPILER = "src/vllm_torchtpu/compilation/tpu_compiler.py"
T_DMK = "tests/layers/adapter/test_qsa_dcp_one_gather.py"
T_PORT = "tests/layers/adapter/test_qsa_dcp_one_gather_port.py"
T_KEY = "tests/compilation/test_tpu_compile_key_registration.py"
# i2: test_o1s_request_shard.py dropped. i1 (j-9f7cea25) ran 5 of its tests in 40 min on 2 CPUs (Pallas interpret)
# and hit the 2400 s call timeout before any mutant ran; it never reaches _dcp_gather_partials / _dcp_finish (C, grep).
# The other suites that import qsa_op are kept.
REGRESSION = ["tests/kernels/test_qwen4_exp_main_kv_dcp_ops.py", "tests/runner/test_qwen4_exp_pool.py"]
REG_TIMEOUT = 900   # i3: per file
TEST_TIMEOUT = 300  # i3: per test, when pytest-timeout is importable
BASE = {ENVS: "envs.py", QSAOP: "qsa_op.py"}
FLAGS = ("VLLM_TPU_QSA_DCP_ONE_GATHER", "VLLM_TPU_DRAFT_ROWS_FRESH", "VLLM_TPU_DRAFT_CARRY_ROWS_FRESH",
         "VLLM_TPU_PROMPT_IDS_NDARRAY", "Q38MM_DM_TOKHASH")
MUTANTS = {  # name: (file, exact original text, replacement)
    "m1_out_slice_offset": (QSAOP, "    out_pair = pair[:, :, :heads]\n", "    out_pair = pair[:, :, 1 : heads + 1]\n"),
    "m2_packing_order": (QSAOP, "    packed = torch.cat([out, lse_rows.reshape(tokens, rows, head_dim)], dim=1)\n",
                         "    packed = torch.cat([lse_rows.reshape(tokens, rows, head_dim), out], dim=1)\n"),
    "m3_flagoff_branch_swapped": (QSAOP, "        out_pair = pair_group.all_gather(out.unsqueeze(0).contiguous(), dim=0)\n"
                                  "        lse_pair = pair_group.all_gather(lse.unsqueeze(0).contiguous(), dim=0)\n",
                                  "        lse_pair = pair_group.all_gather(lse.unsqueeze(0).contiguous(), dim=0)\n"
                                  "        out_pair = pair_group.all_gather(out.unsqueeze(0).contiguous(), dim=0)\n"),
    "m4_flag_ignored": (QSAOP, "        out, lse, pair_group, bool(envs.VLLM_TPU_QSA_DCP_ONE_GATHER)\n",
                        "        out, lse, pair_group, True\n"),
    "m5_dtype_guard_dropped": (QSAOP, "    if not one_gather or out.dtype != lse.dtype:\n", "    if not one_gather:\n"),
    "m6_flag_not_in_compile_key": (COMPILER, "_TPU_COMPILE_ENV_IGNORED = {\n",
                                   "_TPU_COMPILE_ENV_IGNORED = {\n    \"VLLM_TPU_QSA_DCP_ONE_GATHER\",\n"),
}
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()  # noqa: E731


def run(cmd, src, extra=None, timeout=2400):
    env = {k: v for k, v in os.environ.items() if k not in FLAGS}
    env.update(PYTHONPATH=f"{src}:{HERE}:{os.path.join(ROOT, 'tests', 'runner')}", PYTHONUNBUFFERED="1", **(extra or {}))
    try:
        return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout, cwd=ROOT)
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        return subprocess.CompletedProcess(cmd, 124, out, "TIMEOUT")


def pytest(src, args, extra=None, timeout=2400):
    p = run([sys.executable, "-u", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "q38_origin", "-rf",
             "-rfE", "--tb=short", *args], src, extra, timeout=timeout)
    lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
    tail = next((ln for ln in reversed(lines) if re.search(r"\d+ (passed|failed|error|errors|deselected)", ln)),
                lines[-1] if lines else "<no output>")
    m = re.search(r"^Q38_ORIGIN (.*)$", p.stdout, re.M)
    origin = dict(kv.split("=", 1) for kv in m.group(1).split()) if m else {}
    want = os.path.realpath(src) + os.sep
    origin_ok = origin.get("vllm_torchtpu", "None") != "None" and all(
        v not in (None, "None") and os.path.realpath(v).startswith(want) for v in origin.values())
    num = lambda w: int((re.findall(r"(\d+) " + w, tail) or ["0"])[-1])  # noqa: E731
    return dict(rc=p.returncode, tail=tail, out=p.stdout + p.stderr, origin_ok=origin_ok,
                origin=m.group(0) if m else "Q38_ORIGIN <missing>", npass=num("passed"), nfail=num("failed"),
                errors=bool(re.search(r"\d+ errors?\b", tail)), failed=re.findall(r"^FAILED (\S+)", p.stdout, re.M),
                errored=re.findall(r"^ERROR (\S+)", p.stdout, re.M))


def reg_leg(src, label, timeout_mode):
    """i3 (DECODE-LEAD 11-12): REGRESSION one file at a time, -v (each node id is printed when it starts),
    --durations=20, and --timeout=TEST_TIMEOUT where pytest-timeout exists. A file-level timeout or a per-test timeout
    prints HUNG <node id> and makes the leg INCOMPLETE."""
    failed, errored, hung, ok_origin = [], [], [], True
    for f in REGRESSION:
        extra_args = ["-v", "--durations=20"] + ([f"--timeout={TEST_TIMEOUT}"] if timeout_mode else [])
        r = pytest(src, [*extra_args, f], timeout=REG_TIMEOUT)
        show(f"REGRESSION {label} {f}", r)
        if r["rc"] == 124:
            started = re.findall(r"^(tests/\S+::\S+)(?: (PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS))?", r["out"], re.M)
            last = next((n for n, o in reversed(started) if not o), started[-1][0] if started else "<none started>")
            hung.append(last)
            print(f"  HUNG {last} (file timeout {REG_TIMEOUT} s)", flush=True)
        else:
            ok_origin &= r["origin_ok"]
        for n in re.findall(r"^FAILED (\S+) - Failed: Timeout", r["out"], re.M):
            hung.append(n)
            print(f"  HUNG {n} (per-test timeout {TEST_TIMEOUT} s)", flush=True)
        failed += r["failed"]
        errored += r["errored"]
        for ln in [x for x in r["out"].splitlines() if re.match(r"^\d+\.\d+s (call|setup|teardown) ", x)][:5]:
            print("  DURATION " + ln, flush=True)
    return dict(failed=failed, errored=errored, hung=hung, origin_ok=ok_origin)


def show(tag, r):
    print(f"{tag} rc={r['rc']} passed={r['npass']} failed={r['nfail']} errors={r['errors']} "
          f"origin={'OK' if r['origin_ok'] else 'INVALID'} :: {r['tail']}", flush=True)
    print("  " + r["origin"], flush=True)


def copy_src(tag, edits=(), base=False):
    d = tempfile.mkdtemp(prefix=f"q38_crf_{tag}_")
    shutil.copytree(os.path.join(ROOT, "src"), os.path.join(d, "src"))
    if base:
        for rel, name in BASE.items():
            shutil.copyfile(os.path.join(HERE, "base_aa83ece72", name), os.path.join(d, rel))
    for rel, old, new in edits:
        p = os.path.join(d, rel)
        s = open(p).read()
        assert s.count(old) == 1, (tag, rel, s.count(old))
        open(p, "w").write(s.replace(old, new))
    return os.path.join(d, "src")


def main():
    files = [os.path.join(ROOT, f) for f in (QSAOP, ENVS, COMPILER, T_DMK, T_PORT, T_KEY)]
    files.append(os.path.abspath(__file__))
    start = {p: md5(p) for p in files}
    print("SRC-AT-START " + " ".join("%s=%s" % (os.path.relpath(p, ROOT), start[p]) for p in files), flush=True)
    same = lambda: all(md5(p) == start[p] for p in files)  # noqa: E731
    ok = True
    p = run([sys.executable, "-u", "-c", "import vllm_torchtpu.layers.adapter.custom_ops.qsa_op as m, torch; "
             "print('Q38_IMPORT qsa_op OK', m.__file__, 'torch', torch.__version__)"], os.path.join(ROOT, "src"))
    imp = p.returncode == 0
    print((p.stdout.strip().splitlines() or ["<no stdout>"])[-1] if imp else f"Q38_IMPORT FAIL rc={p.returncode}", flush=True)
    if not imp:
        print(p.stderr[-4000:], flush=True)
    ok &= imp
    # ---- CLEAN
    src = os.path.join(ROOT, "src")
    r = pytest(src, [T_DMK, T_PORT, T_KEY])
    clean = r["rc"] == 0 and r["origin_ok"]
    show(f"CLEAN -> {'PASS' if clean else 'FAIL'}", r)
    if not clean:
        print(r["out"][-8000:], flush=True)
    lc16 = True
    ok &= clean
    # ---- MUTANTS (i2: before REGRESSION, so a slow regression suite cannot starve them)
    killed = invalid = 0
    for name, (rel, old, new) in MUTANTS.items():
        try:
            msrc = copy_src(name, ((rel, old, new),))
        except AssertionError as e:
            invalid += 1
            print(f"MUTANT {name} INVALID anchor {e}", flush=True)
            continue
        r = pytest(msrc, [T_DMK, T_PORT, T_KEY])
        k = r["rc"] == 1 and r["nfail"] >= 1 and not r["errors"] and r["origin_ok"]
        bad = (not k) and r["rc"] != 0
        killed += k
        invalid += bad
        print(f"MUTANT {name} -> {'KILLED' if k else ('INVALID' if bad else 'SURVIVED')} rc={r['rc']} "
              f"origin={'OK' if r['origin_ok'] else 'INVALID'} :: {r['tail']} :: tree {'== start' if same() else '!= START'}",
              flush=True)
        for f in r["failed"][:4]:
            print("  FAILED " + f, flush=True)
        shutil.rmtree(os.path.dirname(msrc), ignore_errors=True)
    # ---- REGRESSION A/B (i3: per file, -v, durations, per-test timeout where available; a hang is INCOMPLETE)
    # --timeout exists only if pytest-timeout is installed and auto-loaded: ask pytest itself.
    pt = run([sys.executable, "-m", "pytest", "--help", "-p", "no:cacheprovider"], src, timeout=300)
    timeout_mode = "--timeout=" in pt.stdout
    print(f"REGRESSION timeout mode: {'pytest-timeout --timeout=%d per test' % TEST_TIMEOUT if timeout_mode else 'no pytest-timeout; file-level only'}"
          f" + file timeout {REG_TIMEOUT} s", flush=True)
    rt = reg_leg(src, "tree", timeout_mode)
    bsrc = copy_src("base", base=True)
    rb = reg_leg(bsrc, "base copy (aa83ece72 envs / qsa_op)", timeout_mode)
    bad_tree = set(rt["failed"]) | set(rt["errored"])
    bad_base = set(rb["failed"]) | set(rb["errored"])
    caused = sorted(bad_tree - bad_base - set(rt["hung"]))
    print(f"REGRESSION tree failed {len(rt['failed'])} errored {len(rt['errored'])} hung {len(rt['hung'])}; base failed "
          f"{len(rb['failed'])} errored {len(rb['errored'])} hung {len(rb['hung'])}; failing or erroring on the tree but on "
          f"neither list on the base: {len(caused)}", flush=True)
    for f in caused[:20]:
        print("  CAUSED " + f, flush=True)
    incomplete = bool(rt["hung"] or rb["hung"])
    regression = rt["origin_ok"] and rb["origin_ok"] and not caused
    reg_word = "INCOMPLETE" if incomplete else ("PASS" if regression else "FAIL")
    print(f"REGRESSION -> {reg_word}", flush=True)
    shutil.rmtree(os.path.dirname(bsrc), ignore_errors=True)
    ok &= regression and not incomplete
    end = {p: md5(p) for p in files}
    print("SRC-AT-END " + " ".join("%s=%s" % (os.path.relpath(p, ROOT), end[p]) for p in files), flush=True)
    unchanged = end == start
    print("SRC-AT-END %s" % ("UNCHANGED" if unchanged else "CHANGED"), flush=True)
    ok &= killed == len(MUTANTS) and invalid == 0 and unchanged
    print("Q38_L4_SUMMARY import=%s clean=%s order16=%s regression=%s mutants_killed=%d/%d invalid=%d src_at_end=%s "
          "VERDICT=%s" % ("OK" if imp else "FAIL", "PASS" if clean else "FAIL", "N/A" if lc16 else "FAIL",
                          reg_word, killed, len(MUTANTS), invalid,
                          "UNCHANGED" if unchanged else "CHANGED",
                          "PASS" if ok else ("INCOMPLETE" if (incomplete and imp and clean and not caused and unchanged
                                                              and killed == len(MUTANTS) and invalid == 0) else "FAIL")),
          flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
