#!/usr/bin/env python3
"""Source mutants of the D68 change (VLLM_TPU_DECODE_TP_AR_PAD16). Each must make test_tp_ar_pad16.py fail.
Runs each mutant on a copy of src/ (PYTHONPATH points at the copy); a clean copy must pass first.
Usage: python q38_l3_d68_mutrun.py   (env: JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8)"""
import os, shutil, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
F = "vllm_torchtpu/kernels/collectives/tp8_ar_decode_fn.py"
M = [
    ("no_pad_branch", "    if run:\n        xp = jnp.pad", "    if False:\n        xp = jnp.pad"),
    ("pad_default_on", 'return value == "1"', "return True"),
    ("pad_ignored_in_global_fn", "interpret=interpret, pad16_=pad16_\n", "interpret=interpret, pad16_=False\n"),
    ("round_to_32", "up = -(-rows // 16) * 16", "up = -(-rows // 32) * 32"),
    ("no_slice_back", "        return out[:rows]\n", "        return out\n"),
    ("pad_front", "jnp.pad(x, ((0, run - rows), (0, 0)))", "jnp.pad(x, ((run - rows, 0), (0, 0)))"),
    ("no_cap_check", "        if h8.supports(up, hidden, dtype, max_rows_):\n            return up\n", "        return up\n"),
    ("factor_never", "    if pad16():\n        factors[PAD16_ENV]", "    if False:\n        factors[PAD16_ENV]"),
    ("factor_always", "    if pad16():\n        factors[PAD16_ENV]", "    if True:\n        factors[PAD16_ENV]"),
    ("parse_accepts_any", '    if value not in ("", "1"):\n        raise', '    if False:\n        raise'),
    ("pad_multiples_too", "    if h8.supports(rows, hidden, dtype, max_rows_):\n        return rows\n    if pad16_",
     "    if h8.supports(rows, hidden, dtype, max_rows_) and not pad16_:\n        return rows\n    if pad16_"),
    ("psum_when_padded", "        return out[:rows]\n    return lax.psum(x, axis)", "        return lax.psum(x, axis)\n    return lax.psum(x, axis)"),
]
test = os.path.join(HERE, "test_tp_ar_pad16.py")
def run(srcdir):
    env = dict(os.environ, PYTHONPATH=srcdir)
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "--noconftest", test],
                       env=env, capture_output=True, text=True, cwd=ROOT)
    return r.returncode, (r.stdout.strip().splitlines() or ["?"])[-1][:100]
with tempfile.TemporaryDirectory() as d:
    shutil.copytree(os.path.join(ROOT, "src"), os.path.join(d, "src"))
    rc, tail = run(os.path.join(d, "src")); print("CLEAN", "PASS" if rc == 0 else "FAIL", tail)
    if rc != 0: sys.exit(2)
    orig = open(os.path.join(d, "src", F)).read(); k = 0
    for name, a, b in M:
        if orig.count(a) != 1: print("BAD-ANCHOR", name, orig.count(a)); continue
        open(os.path.join(d, "src", F), "w").write(orig.replace(a, b))
        rc, tail = run(os.path.join(d, "src")); kk = rc != 0; k += kk
        print("KILLED  " if kk else "SURVIVED", name, tail)
    open(os.path.join(d, "src", F), "w").write(orig)
print(f"D68 MUTANTS {k}/{len(M)} KILLED")
sys.exit(0 if k == len(M) else 1)
