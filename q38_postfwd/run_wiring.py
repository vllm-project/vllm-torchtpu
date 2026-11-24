"""q38 DEC-FB-3 POSTFWD CPU verdict for the warm-set fix (DECODE-LEAD ruling 06:10Z 11-12). Run from the tree root:
    python3 -u q38_postfwd/run_wiring.py
Steps: (0) tpu_runner must import on this image (the runner-level test relies on it); (1) the G2 bitwise test and
the runner wiring test must PASS on the clean tree; (2) each mutant of tpu_runner.py (applied to a private copy of
src/) must make the wiring test FAIL. Prints one line per step and a final Q38_POSTFWD_WIRING_SUMMARY line; exit 0
only if every step behaves as required."""
import os, shutil, subprocess, sys, tempfile

TESTS = ["tests/runner/test_postfwd_fused.py", "tests/runner/test_postfwd_runner_wiring.py"]
MUT_TEST = ["tests/runner/test_postfwd_runner_wiring.py"]
TARGET = "src/vllm_torchtpu/runner/tpu_runner.py"
GATE = ('            if key not in warm and getattr(self, "warmup_done", False):\n'
        '                args = None  # serving: never compile a new shape here\n')
MUTANTS = {  # name: [(exact original text in tpu_runner.py, replacement)]
    # The lead's mutant: the key joins the warm set before the fused call (the 9683ae840 order).
    "w1_warm_early": [(GATE, GATE + "            elif key not in warm:\n                warm.add(key)\n")],
    # Serving compiles any shape (no warm-set gate).
    "w2_no_gate": [(GATE, '            if False:\n                args = None\n')],
    # A successful warmup call never warms its shape (serving always takes the separate calls).
    "w3_no_add": [("            warm.add(key)\n            stats[0] += 1\n", "            stats[0] += 1\n")],
}


def run(cmd, pythonpath):
    env = dict(os.environ, PYTHONPATH=pythonpath, PYTHONUNBUFFERED="1")
    return subprocess.run(cmd, env=env, capture_output=True, text=True)


def pytest(pythonpath, tests):
    p = run([sys.executable, "-u", "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests], pythonpath)
    tail = [ln for ln in p.stdout.splitlines() if ln.strip()][-1:] or ["<no output>"]
    return p.returncode, tail[0], p.stdout + p.stderr


ok = True
p = run([sys.executable, "-u", "-c", "import vllm_torchtpu.runner.tpu_runner as m, torch; "
         "print('Q38_IMPORT tpu_runner OK', m.__file__, 'torch', torch.__version__)"], "src")
imp = p.returncode == 0
print((p.stdout.strip().splitlines() or ["<no stdout>"])[-1] if imp else f"Q38_IMPORT tpu_runner FAIL rc={p.returncode}", flush=True)
if not imp:
    print(p.stderr[-4000:], flush=True)
ok &= imp
rc, tail, out = pytest("src", TESTS)
clean = rc == 0
print(f"CLEAN rc={rc} -> {'PASS' if clean else 'FAIL'} :: {tail}", flush=True)
if not clean:
    print(out[-6000:], flush=True)
ok &= clean
src_text = open(TARGET).read()
killed = 0
for name, edits in MUTANTS.items():
    text = src_text
    for o, n in edits:
        if text.count(o) != 1:
            print(f"MUTANT {name}: ABORT, anchor found {text.count(o)} times", flush=True)
            ok = False
            break
        text = text.replace(o, n)
    else:
        d = tempfile.mkdtemp(prefix=f"q38_pfw_{name}_")
        shutil.copytree("src", os.path.join(d, "src"))
        with open(os.path.join(d, TARGET), "w") as f:
            f.write(text)
        rc, tail, _ = pytest(os.path.join(d, "src"), MUT_TEST)
        k = rc != 0
        killed += k
        ok &= k
        print(f"MUTANT {name} rc={rc} -> {'KILLED' if k else 'SURVIVED'} :: {tail}", flush=True)
        shutil.rmtree(d, ignore_errors=True)
print(f"Q38_POSTFWD_WIRING_SUMMARY import={'OK' if imp else 'FAIL'} clean={'PASS' if clean else 'FAIL'} "
      f"mutants_killed={killed}/{len(MUTANTS)} VERDICT={'PASS' if ok else 'FAIL'}", flush=True)
sys.exit(0 if ok else 1)
