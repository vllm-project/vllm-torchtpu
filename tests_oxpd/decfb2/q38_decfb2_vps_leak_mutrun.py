#!/usr/bin/env python3
"""Mutants of the VPS harnesses' sys.modules isolation; each must make tests/decfb2/test_vps_harness_no_leak.py fail.

The four harness files are copied to a scratch tests/ tree (Q38_VPS_HARNESS_TESTS), the source stays the tree's
(Q38MM_SRC_ROOT). "orig" is the harness as committed before the fix (PREFIX_REV, default aba96afb5): the defect that
killed INT-4's union job i5; Q38_VPS_LEAK_ORIG_DIR (q38mm_dstock/, decfb2/) supplies those files where there is no
.git. "identity" (no mutation) must pass.
Run: python tests/decfb2/q38_decfb2_vps_leak_mutrun.py [PREFIX_REV]"""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SP, SW = "q38mm_dstock/test_vp_sampler.py", "q38mm_dstock/test_vps_wiring.py"
FF, FW = "decfb2/test_vps_fallback.py", "decfb2/test_vps_fb_wiring.py"
FILES = [SP, SW, FF, FW]
LOAD_RESTORE = "            self.mods = {n: m for n, m in self.snapshot().items() if saved.get(n) is not m}\n        finally:\n            self.restore(saved)\n"
LOAD_NORESTORE = "            self.mods = {n: m for n, m in self.snapshot().items() if saved.get(n) is not m}\n        finally:\n            pass\n"
ACT_RESTORE = "        try:\n            yield\n        finally:\n            self.restore(saved)\n"
ACT_NORESTORE = "        try:\n            yield\n        finally:\n            pass\n"
KEEP_REAL = "                continue                    # keep the imported real vllm package; only its leaves are stubbed\n"
M = [
    ("identity", None, None, None),
    ("SW_load_no_restore", SW, LOAD_RESTORE, LOAD_NORESTORE),
    ("SP_load_no_restore", SP, LOAD_RESTORE, LOAD_NORESTORE),
    ("SW_tests_no_restore", SW, ACT_RESTORE, ACT_NORESTORE),
    ("SP_tests_no_restore", SP, ACT_RESTORE, ACT_NORESTORE),
    ("SW_stub_shadows_real_vllm", SW, KEEP_REAL, "                pass\n"),
    ("FW_distributed_setdefault", FW, "\n\n@pytest.fixture(autouse=True, scope=\"module\")",
     "\nsys.modules.setdefault(\"vllm.distributed\", types.ModuleType(\"vllm.distributed\"))\n\n"
     "@pytest.fixture(autouse=True, scope=\"module\")"),
    ("orig", "ORIG", None, None),
]


def main():
    rev = sys.argv[1] if len(sys.argv) > 1 else "aba96afb5"
    check = os.path.join(HERE, "test_vps_harness_no_leak.py")
    bad = []
    for name, path, old, new in M:
        d = tempfile.mkdtemp(prefix="q38_vps_leak_")
        try:
            for f in FILES:
                os.makedirs(os.path.join(d, os.path.dirname(f)), exist_ok=True)
                if path == "ORIG" and os.environ.get("Q38_VPS_LEAK_ORIG_DIR"):     # a tree without .git (CPU job)
                    src = open(os.path.join(os.environ["Q38_VPS_LEAK_ORIG_DIR"], f)).read()
                elif path == "ORIG":
                    src = subprocess.run(["git", "-C", REPO, "show", f"{rev}:tests/{f}"], check=True,
                                         capture_output=True, text=True).stdout
                else:
                    src = open(os.path.join(REPO, "tests", f)).read()
                    if f == path:
                        assert src.count(old) == 1, (name, src.count(old))
                        src = src.replace(old, new)
                open(os.path.join(d, f), "w").write(src)
            env = dict(os.environ, Q38_VPS_HARNESS_TESTS=d, Q38MM_SRC_ROOT=os.path.join(REPO, "src"),
                       PYTHONDONTWRITEBYTECODE="1")
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--noconftest", check],
                               capture_output=True, text=True, env=env, cwd=d, timeout=1800)
            summary = (r.stdout.strip().splitlines() or ["?"])[-1]
            ok = (r.returncode == 0) if name == "identity" else (r.returncode != 0)
            print(f"{'OK ' if ok else 'BAD'} {name:28s} rc={r.returncode}  {summary}", flush=True)
            if not ok:
                bad.append(name)
        finally:
            shutil.rmtree(d, ignore_errors=True)
    print("ALL MUTANTS KILLED, IDENTITY PASSES" if not bad else f"FAILED: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
