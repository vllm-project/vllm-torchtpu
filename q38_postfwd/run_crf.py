"""q38 DEC-FB-3 lever 3 proof: dongruizi's 48745d9b0 (VLLM_TPU_DRAFT_ROWS_FRESH) + 7a0a3958e
(VLLM_TPU_DRAFT_CARRY_ROWS_FRESH) cherry-picked onto the P-c chain 68e6e31c9. Run from the tree root (CPU job).
Legs (Q38_ORIGIN in every pytest leg; a leg that imported a module under test from outside its own src/ reads INVALID):
  IMPORT      eagle3 imports, torch version.
  CLEAN       tests/runner/test_carry_rows_fresh_port.py (fresh equality, own-buffer, flags-off structural identity
              against 68e6e31c9, gates, precompile grids -> Q38_CRF_GRIDS) and the compile-key registration test.
  PCTESTS     P-c's test_draft_carries_port.py on the port (its stub binds _propose_token_body, 48745d9b0's refactor).
  ORDER16     the 16 drafter / KV-connector ordering tests.
  REGRESSION  runner / spec-decode suites on the tree and on a base copy (68e6e31c9's envs.py + eagle3.py laid over src,
              which is 68e6e31c9's src exactly); a test failing or erroring (-rfE) on the tree must fail or error on the
              base too (i1 compared failures only and missed 51 errors).
  MUTANTS     12, each on a private copy of src/; strict kill: rc 1, at least one failed test, no error; md5 of the
              tree's files after each must equal SRC-AT-START.
Last line: Q38_CRF_SUMMARY ... VERDICT=PASS|FAIL."""
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.getcwd()
HERE = os.path.join(ROOT, "q38_postfwd")
EAGLE3 = "src/vllm_torchtpu/spec_decode/eagle3.py"
ENVS = "src/vllm_torchtpu/envs.py"
COMPILER = "src/vllm_torchtpu/compilation/tpu_compiler.py"
T_CRF = "tests/runner/test_carry_rows_fresh_port.py"
T_DC = "tests/runner/test_draft_carries_port.py"
T_KEY = "tests/compilation/test_tpu_compile_key_registration.py"
T_LC = "tests/runner/test_tpu_runner_kv_connector_lifecycle.py"
K16 = "test_device_drafter_runs_before_kv_connector_save or test_spec_kv_fence_runs_after_drafter_before_connector_save"
REGRESSION = ["tests/runner/test_tpu_runner.py", T_LC, "tests/spec_decode/test_speculative_decoding_manager.py",
              "tests/spec_decode/test_eagle3.py", "tests/spec_decode/test_mtp_fused_loop_dims.py"]
BASE = {ENVS: "envs.py", EAGLE3: "spec_decode_eagle3.py"}
FLAGS = ("VLLM_TPU_DRAFT_ROWS_FRESH", "VLLM_TPU_DRAFT_CARRY_ROWS_FRESH", "VLLM_TPU_DRAFT_CARRY_FUSED",
         "VLLM_TPU_DRAFT_SEED_EXTRAS", "VLLM_TPU_VERIFY_FUSED_READ_OFFSETS", "VLLM_TPU_POSTFWD_FUSED",
         "VLLM_TPU_SPEC_TAIL_COMPILED", "VLLM_TPU_PROMPT_IDS_NDARRAY", "Q38MM_DM_TOKHASH")
MUTANTS = {  # name: (file, exact original text, replacement)
    "c1_fresh_trims_wrong_rows": (EAGLE3, "        positions_carry, token = self._carry_and_token_body(\n"
                                  "            positions, indices, last_hidden[:rows]\n",
                                  "        positions_carry, token = self._carry_and_token_body(\n"
                                  "            positions, indices, last_hidden[1 : rows + 1]\n"),
    "c2_fresh_hidden_is_a_view": (EAGLE3, "        return positions_carry, token, hidden[:rows].clone()\n",
                                  "        return positions_carry, token, hidden[:rows]\n"),
    "c3_fresh_body_no_clamp": (EAGLE3, "        self, positions: torch.Tensor, indices: torch.Tensor, last_hidden: torch.Tensor\n"
                               "    ) -> tuple[torch.Tensor, torch.Tensor]:\n        assert self.draft_model is not None\n"
                               "        indices = indices.clamp_min(0)\n",
                               "        self, positions: torch.Tensor, indices: torch.Tensor, last_hidden: torch.Tensor\n"
                               "    ) -> tuple[torch.Tensor, torch.Tensor]:\n        assert self.draft_model is not None\n"),
    "c4_wiring_swapped": (EAGLE3, "                                untrimmed[0],\n                                untrimmed[1],\n",
                          "                                untrimmed[1],\n                                untrimmed[0],\n"),
    "c5_carry_flag_ignored": (EAGLE3, "                    envs.VLLM_TPU_DRAFT_CARRY_ROWS_FRESH\n                    and untrimmed is not None\n",
                              "                    True\n                    and untrimmed is not None\n"),
    "c6_flagoff_branch_changed": (EAGLE3, "                            self._draft_rows_carry_and_token(\n"
                                  "                                positions, last_token_indices, last_hidden\n",
                                  "                            self._draft_rows_carry_and_token(\n"
                                  "                                positions, last_token_indices, last_hidden_carry\n"),
    "c7_fresh_precompile_skips": (EAGLE3, "                            if p > T or p >= cap:\n",
                                  "                            if p > T or p >= cap or p > 16:\n"),
    "c8_rows_fresh_hidden_is_a_view": (EAGLE3, "        token = self._propose_token_body(last_hidden[:rows])\n"
                                       "        return token, hidden[:rows].clone()\n",
                                       "        token = self._propose_token_body(last_hidden[:rows])\n"
                                       "        return token, hidden[:rows]\n"),
    "c9_flag_not_in_compile_key": (COMPILER, "_TPU_COMPILE_ENV_IGNORED = {\n",
                                   "_TPU_COMPILE_ENV_IGNORED = {\n    \"VLLM_TPU_DRAFT_CARRY_ROWS_FRESH\",\n"),
    "c10_rows_match_dropped": (EAGLE3, "                    and untrimmed[2] == last_token_indices.shape[0]\n", ""),
    "c11_rows_fresh_precompile_skips": (EAGLE3, "                        if rows < cap:\n",
                                        "                        if rows < cap and rows > 16:\n"),
    "c12_capture_flag_ignored": (EAGLE3, "                        envs.VLLM_TPU_DRAFT_ROWS_FRESH\n"
                                 "                        or envs.VLLM_TPU_DRAFT_CARRY_ROWS_FRESH\n",
                                 "                        True\n                        or envs.VLLM_TPU_DRAFT_CARRY_ROWS_FRESH\n"),
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


def pytest(src, args, extra=None):
    p = run([sys.executable, "-u", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "q38_origin", "-rf",
             "-rfE", "--tb=short", *args], src, extra)
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


def show(tag, r):
    print(f"{tag} rc={r['rc']} passed={r['npass']} failed={r['nfail']} errors={r['errors']} "
          f"origin={'OK' if r['origin_ok'] else 'INVALID'} :: {r['tail']}", flush=True)
    print("  " + r["origin"], flush=True)


def copy_src(tag, edits=(), base=False):
    d = tempfile.mkdtemp(prefix=f"q38_crf_{tag}_")
    shutil.copytree(os.path.join(ROOT, "src"), os.path.join(d, "src"))
    if base:
        for rel, name in BASE.items():
            shutil.copyfile(os.path.join(HERE, "base_68e6e31c9", name), os.path.join(d, rel))
    for rel, old, new in edits:
        p = os.path.join(d, rel)
        s = open(p).read()
        assert s.count(old) == 1, (tag, rel, s.count(old))
        open(p, "w").write(s.replace(old, new))
    return os.path.join(d, "src")


def main():
    files = [os.path.join(ROOT, f) for f in (EAGLE3, ENVS, COMPILER, T_CRF, T_DC, T_KEY, T_LC,
                                             "tests/runner/crf_flagoff_ast.py", "tests/runner/eagle3_ref_68e6e31c9.py.txt")]
    files.append(os.path.abspath(__file__))
    start = {p: md5(p) for p in files}
    print("SRC-AT-START " + " ".join("%s=%s" % (os.path.relpath(p, ROOT), start[p]) for p in files), flush=True)
    same = lambda: all(md5(p) == start[p] for p in files)  # noqa: E731
    ok = True
    p = run([sys.executable, "-u", "-c", "import vllm_torchtpu.spec_decode.eagle3 as m, torch; "
             "print('Q38_IMPORT eagle3 OK', m.__file__, 'torch', torch.__version__)"], os.path.join(ROOT, "src"))
    imp = p.returncode == 0
    print((p.stdout.strip().splitlines() or ["<no stdout>"])[-1] if imp else f"Q38_IMPORT FAIL rc={p.returncode}", flush=True)
    if not imp:
        print(p.stderr[-4000:], flush=True)
    ok &= imp
    # ---- CLEAN
    grid = "/tmp/q38_crf_grid.json"
    src = os.path.join(ROOT, "src")
    r = pytest(src, [T_CRF, T_KEY], {"Q38_CRF_GRID_OUT": grid})
    clean = r["rc"] == 0 and r["origin_ok"]
    show(f"CLEAN (port tests + compile key) -> {'PASS' if clean else 'FAIL'}", r)
    try:
        print(f"  Q38_CRF_GRIDS {open(grid).read()}", flush=True)
    except OSError:
        print("  Q38_CRF_GRIDS <missing>", flush=True)
    if not clean:
        print(r["out"][-8000:], flush=True)
    # P-c's own suite against the port (lead 11:4xZ): the lever sits on P-c, so P-c's tests must still pass on it.
    rpc = pytest(src, [T_DC])
    pctests = rpc["rc"] == 0 and rpc["origin_ok"] and rpc["nfail"] == 0 and not rpc["errors"]
    show(f"PCTESTS (test_draft_carries_port.py on the port) -> {'PASS' if pctests else 'FAIL'}", rpc)
    if not pctests:
        print(rpc["out"][-6000:], flush=True)
    r16 = pytest(src, [T_LC, "-k", K16])
    lc16 = r16["rc"] == 0 and r16["npass"] == 16 and r16["origin_ok"]
    show(f"ORDER16 -> {'PASS' if lc16 else 'FAIL'}", r16)
    ok &= clean and lc16 and pctests
    # ---- REGRESSION A/B
    rt = pytest(src, REGRESSION)
    show("REGRESSION tree", rt)
    bsrc = copy_src("base", base=True)
    rb = pytest(bsrc, REGRESSION)
    show("REGRESSION base copy (68e6e31c9 envs / eagle3), the same suites", rb)
    bad_tree = set(rt["failed"]) | set(rt["errored"])
    bad_base = set(rb["failed"]) | set(rb["errored"])
    caused = sorted(bad_tree - bad_base)
    print(f"REGRESSION tree failed {len(rt['failed'])} errored {len(rt['errored'])}; base failed {len(rb['failed'])} "
          f"errored {len(rb['errored'])}; failing or erroring on the tree but on neither list on the base: {len(caused)}",
          flush=True)
    for f in caused[:20]:
        print("  CAUSED " + f, flush=True)
    # every tree error, classified against the base (DECODE-LEAD 11:4xZ): errors on base / fails on base / neither
    cls = {"errors_on_base": 0, "fails_on_base": 0, "neither": 0}
    for f in sorted(set(rt["errored"])):
        k = "errors_on_base" if f in rb["errored"] else ("fails_on_base" if f in rb["failed"] else "neither")
        cls[k] += 1
        print("  tree ERROR %-14s %s" % (k, f), flush=True)
        if k != "errors_on_base":
            for ln in rt["out"].splitlines():
                if ln.startswith("ERROR " + f) or (f.split("::")[0] in ln and ln.startswith("E   ")):
                    print("    " + ln[:300], flush=True)
                    break
    print("REGRESSION tree errors classified %s" % cls, flush=True)
    regression = rt["origin_ok"] and rb["origin_ok"] and not caused
    print(f"REGRESSION -> {'PASS' if regression else 'FAIL'}", flush=True)
    shutil.rmtree(os.path.dirname(bsrc), ignore_errors=True)
    ok &= regression
    # ---- MUTANTS
    killed = invalid = 0
    for name, (rel, old, new) in MUTANTS.items():
        try:
            msrc = copy_src(name, ((rel, old, new),))
        except AssertionError as e:
            invalid += 1
            print(f"MUTANT {name} INVALID anchor {e}", flush=True)
            continue
        r = pytest(msrc, [T_CRF, T_KEY])
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
    end = {p: md5(p) for p in files}
    print("SRC-AT-END " + " ".join("%s=%s" % (os.path.relpath(p, ROOT), end[p]) for p in files), flush=True)
    unchanged = end == start
    print("SRC-AT-END %s" % ("UNCHANGED" if unchanged else "CHANGED"), flush=True)
    ok &= killed == len(MUTANTS) and invalid == 0 and unchanged
    print("Q38_CRF_SUMMARY import=%s clean=%s pctests=%s order16=%s regression=%s mutants_killed=%d/%d invalid=%d "
          "src_at_end=%s VERDICT=%s" % ("OK" if imp else "FAIL", "PASS" if clean else "FAIL", "PASS" if pctests else "FAIL",
                                       "PASS" if lc16 else "FAIL",
                          "PASS" if regression else "FAIL", killed, len(MUTANTS), invalid,
                          "UNCHANGED" if unchanged else "CHANGED", "PASS" if ok else "FAIL"), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
