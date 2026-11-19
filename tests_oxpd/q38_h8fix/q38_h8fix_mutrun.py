# SPDX-License-Identifier: Apache-2.0
"""q38 H8-FIX mutant runner (CPU). Each mutant edits tp8_ar_decode_fn.py in a fresh COPY of the tree (the run tree is
never touched) and runs tests/arfix/test_tp_ar_decode_jaxop_signature.py there. A clean copy must pass first (control);
every mutant must make that test file fail (KILLED).

  M1 revert:      5d368f166's exact bytes (the TPU bug: local-only jax import + string annotations). Must show
                  "NameError: name 'jax' is not defined", the J0b j-db896a2a error. L1's scanner must flag :69.
  M2 no-annot:    module import kept, inner annotations dropped (``def fn(x):``). Not a valid fix: jax_op needs a
                  jax.Array return annotation for its schema (torch_tpu _map_jax_return_to_schema_str raises on empty).
  M3 tc-import:   the module import moved under ``if TYPE_CHECKING:`` (absent at run time; L1's scanner counts it as
                  bound, H8-FIX's extended scan does not).
Exit 0 only if the control passes and all mutants are killed. Last line: Q38_H8FIX_MUT PASS|FAIL."""
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REL = "src/vllm_torchtpu/kernels/collectives/tp8_ar_decode_fn.py"
TEST = "tests/arfix/test_tp_ar_decode_jaxop_signature.py"
FIX = open(os.path.join(ROOT, "tests/q38_h8fix/fixtures/tp8_ar_decode_fn.5d368f166.py.txt"), encoding="utf-8").read()
SRC = open(os.path.join(ROOT, REL), encoding="utf-8").read()
IMPORT = "import jax\n\nN_MEMBERS = 8\n"
ANNOT = "    def fn(x: jax.Array) -> jax.Array:\n"


def one(old, new, s):
    assert s.count(old) == 1, f"mutation anchor not unique / missing: {old!r}"
    return s.replace(old, new)


MUTANTS = {
    "M1_revert_to_5d368f166": lambda s: FIX,
    "M2_drop_inner_annotations": lambda s: one(ANNOT, "    def fn(x):\n", s),
    "M3_import_under_TYPE_CHECKING": lambda s: one(
        IMPORT, "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    import jax\n\nN_MEMBERS = 8\n", s
    ),
}


def run(name, text):
    d = tempfile.mkdtemp(prefix=f"h8mut_{name}_")
    try:
        for sub in ("src", "tests"):
            shutil.copytree(os.path.join(ROOT, sub), os.path.join(d, sub), ignore=shutil.ignore_patterns("__pycache__"))
        p = os.path.join(d, REL)
        if text is not None:
            if text == SRC:
                return name, "BAD", "mutation produced identical source"
            with open(p, "w", encoding="utf-8") as f:
                f.write(text)
            compile(text, p, "exec")
        env = dict(os.environ, PYTHONPATH=os.path.join(d, "src"), PYTHONDONTWRITEBYTECODE="1")
        r = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--noconftest", "-rs", TEST],
            cwd=d, env=env, capture_output=True, text=True, timeout=900,
        )
        out = r.stdout + r.stderr
        tail = [ln for ln in out.splitlines() if ln.strip()][-1:] or [""]
        extra = ""
        if name.startswith("M1"):
            has = "NameError: name 'jax' is not defined" in out
            scan = subprocess.run(
                [sys.executable, "-I", os.path.join(ROOT, "tests/q38_h8fix/q38_l1_annscan.py"), os.path.join(d, "src")],
                capture_output=True, text=True,
            ).stdout
            flagged = "HIT vllm_torchtpu/kernels/collectives/tp8_ar_decode_fn.py:69 def fn" in scan
            extra = f" NameError={has} L1-scan-flags-:69={flagged}"
            if not (has and flagged):
                return name, "BAD", f"rc={r.returncode}{extra} | {tail[0]}"
        if name.startswith("M3"):
            scan2 = subprocess.run(
                [sys.executable, "-I", os.path.join(ROOT, "tests/q38_h8fix/q38_h8fix_annscan2.py"), os.path.join(d, "src")],
                capture_output=True, text=True,
            ).stdout
            flagged = any("tp8_ar_decode_fn.py" in ln and "TYPE_CHECKING-only ['jax']" in ln for ln in scan2.splitlines())
            extra = f" scan2-flags-TYPE_CHECKING={flagged}"
            if not flagged:
                return name, "BAD", f"rc={r.returncode}{extra} | {tail[0]}"
        if text is None:
            return name, ("PASS" if r.returncode == 0 else "FAIL"), f"rc={r.returncode} | {tail[0]}"
        # KILLED only by a test failure (pytest "N failed"), never by an environment / collection error.
        killed = r.returncode != 0 and re.search(r"\b\d+ failed\b", out) is not None
        return name, ("KILLED" if killed else "SURVIVED"), f"rc={r.returncode}{extra} | {tail[0]}"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    ok = True
    name, verdict, info = run("CLEAN_control", None)
    print(f"{name}: {verdict} {info}")
    ok &= verdict == "PASS"
    for m, f in MUTANTS.items():
        name, verdict, info = run(m, f(SRC))
        print(f"{name}: {verdict} {info}")
        ok &= verdict == "KILLED"
    print(f"Q38_H8FIX_MUT {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
