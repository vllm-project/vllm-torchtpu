#!/usr/bin/env python3
"""Source mutants of the HCGATE fused decode HC row gate (q38 L3-HCGATE). Each must make the HCGATE tests fail.

    python3 tests/q38_l3_hcgate/q38_l3_hcgate_mutrun.py --mode mac   [--tests test_hcgate_rows.py]
    python3 tests/q38_l3_hcgate/q38_l3_hcgate_mutrun.py --mode image [--tests test_hcgate_rows.py]

Copies src/ once into a temp dir, applies one mutant at a time there (the tree itself is never edited), runs
pytest with Q38_L3_HCX_SRC_ROOT (and, in image mode, PYTHONPATH) on the copy, then restores the file. Controls:
the unmutated copy passes before and after; the tree's src hash is unchanged at the end. Exit 0 only if every
mutant is killed and both controls pass. Last line: Q38_L3_HCGATE_MUTANTS <killed>/<total> <PASS|FAIL>.
The four brief mutants are first: gate_off_by_one, gate_inverted, threshold_ignored, compile_key_missing;
then the band (N:M), the T95u guard-neutrality (band order, prefill full rows, norm / up tiles, HCX) and the
floor (cycle-4 ruling) mutants.
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(TREE, "src")
F = {
    "hcd": "vllm_torchtpu/kernels/qwen4_exp/hc_decode.py",
    "hc": "vllm_torchtpu/models/vllm/qwen4_exp/hyperconnection.py",
    "eng": "vllm_torchtpu/models/vllm/qwen4_exp/hc_engagement.py",
    "tc": "vllm_torchtpu/compilation/tpu_compiler.py",
    "envs": "vllm_torchtpu/envs.py",
}
BASE_HC = os.path.join(HERE, "..", "q38_l3_hcx", "data", "hyperconnection_5d368f166.py.txt")
REG = '    "VLLM_TPU_HC_DECODE_COUNT": env_nonnegative_int("VLLM_TPU_HC_DECODE_COUNT", 0),\n'
ALLOWS = "    return lo <= 0 or tokens <= lo or (hi > 0 and tokens % _SINGLE_TILE_MAX == 0)\n"
GUARD = "    if lo != _GUARD_NEUTRAL_LO or hi not in _GUARD_NEUTRAL_HI:\n"
GATE_FUSED = "            and decode_fused_allows(hyper_input.shape[0], self._hcf_max_rows)\n"
GATE_ROWS = "            and decode_fused_allows(tokens, self._hcf_max_rows)\n"
FLOOR = "    if lo < DECODE_FUSED_FLOOR_ROWS and decode_xla_max_rows() < DECODE_FUSED_FLOOR_ROWS:\n"
GAP = "    if len(parts) == 2 and not (lo + _BAND_MIN_GAP <= hi <= MAX_TOKENS):\n"
HCX = "    return limit > 0 and tokens % (2 * limit) != 0\n"
FULL_ROWS = "            return tokens % self._prefill_tile_step == 0\n"
NORM = "            and hyper_input.shape[0] % self.prefill_norm_tile == 0\n"
UP = "            and gate.shape[0] % self.prefill_up_tile == 0   # divisibility alone (see _normalize)\n"
M = [
    # the four brief mutants
    ("gate_off_by_one", "hcd", ALLOWS, "    return lo <= 0 or tokens < lo or (hi > 0 and tokens % _SINGLE_TILE_MAX == 0)\n"),
    ("gate_inverted", "hcd", ALLOWS, "    return lo <= 0 or not (tokens <= lo or (hi > 0 and tokens % _SINGLE_TILE_MAX == 0))\n"),
    ("threshold_ignored", "hcd", ALLOWS, "    return True\n"),
    ("compile_key_missing", "tc", "    if hcgate:\n", "    if False:\n"),
    # the band N:M (the 256 route parameter)
    ("band_ignored", "hcd", ALLOWS, "    return lo <= 0 or tokens <= lo\n"),
    # T95u: the old order puts a value bound on the traced token count (killed by test_hcgate_symint.py only)
    ("band_old_order_tokens_ge_hi", "hcd", ALLOWS, "    return lo <= 0 or tokens <= lo or (hi > 0 and tokens >= hi)\n"),
    ("band_wrong_tile_64", "hcd", ALLOWS, "    return lo <= 0 or tokens <= lo or (hi > 0 and tokens % 64 == 0)\n"),
    # T95u generalization (L2-LADDER j-4b5f2e6d, 128 slots): restore each old `>=` / `<=` test; killed by
    # test_hcgate_symint.py's bound inventory (a value bound other than FINAL's 128 / 512 or the stand-in's)
    ("prefill_rows_restore_ge_512", "hc", FULL_ROWS,
     "            return tokens >= 2 * self.prefill_full_tile and tokens % self._prefill_tile_step == 0\n"),
    ("norm_tile_restore_ge", "hc", NORM, "            and hyper_input.shape[0] >= self.prefill_norm_tile\n" + NORM),
    ("up_tile_restore_ge", "hc", UP, "            and gate.shape[0] >= self.prefill_up_tile\n" + UP),
    ("hcx_restore_le", "hcd", HCX, "    return limit > 0 and 0 < tokens <= limit\n"),
    ("guard_neutral_check_removed", "hcd", GUARD, "    if False:\n"),
    ("guard_neutral_accepts_any_hi", "hcd", GUARD, "    if lo != _GUARD_NEUTRAL_LO:\n"),
    ("band_parsed_as_off", "hcd", "    hi = _rows_int(value, parts[1]) if len(parts) == 2 else 0\n", "    hi = 0\n"),
    ("band_gap_unchecked", "hcd", GAP, "    if False:\n"),
    # the floor (cycle-4 ruling: N < 128 needs HCX >= 128)
    ("floor_removed", "hcd", FLOOR, "    if False:\n"),
    ("floor_ignores_hcx", "hcd", FLOOR, "    if lo < DECODE_FUSED_FLOOR_ROWS:\n"),
    ("floor_off_by_one", "hcd", "DECODE_FUSED_FLOOR_ROWS = 128 ", "DECODE_FUSED_FLOOR_ROWS = 127 "),
    # hc_decode.py helpers
    ("default_on", "hcd", "    if not value:\n        return 0, 0\n", "    if not value:\n        return 128, 0\n"),
    ("zero_accepted", "hcd", "    if lo < 1:\n", "    if lo < 0:\n"),
    ("off_compares_rows", "hcd", ALLOWS, "    return tokens <= lo or lo <= 0 or (hi > 0 and tokens % _SINGLE_TILE_MAX == 0)\n"),
    ("env_name_typo", "hcd", 'DECODE_FUSED_MAX_ROWS_ENV = "VLLM_TPU_HC_DECODE_FUSED_MAX_ROWS"',
     'DECODE_FUSED_MAX_ROWS_ENV = "VLLM_TPU_HC_DECODE_FUSED_MAXROWS"'),
    ("int_limit_read_as_off", "hcd", "    lo, hi = (limit, 0) if isinstance(limit, int) else limit\n",
     "    lo, hi = limit if isinstance(limit, tuple) else (0, 0)\n"),
    # hyperconnection.py routing
    ("ignored_in_decode_fused", "hc", GATE_FUSED, ""),
    ("ignored_in_decode_rows", "hc", GATE_ROWS, ""),
    ("init_reads_zero", "hc", "        self._hcf_max_rows = decode_fused_max_rows()", "        self._hcf_max_rows = 0"),
    ("unarmed_reads_rows", "hc",
     "            self._decode_mix_op is not None\n            and hyper_input.dim() == 2\n" + GATE_FUSED,
     "            decode_fused_allows(hyper_input.shape[0], self._hcf_max_rows)\n            and self._decode_mix_op is not None\n            and hyper_input.dim() == 2\n"),
    ("gate_reads_hcx_limit", "hc", GATE_FUSED, GATE_FUSED.replace("_hcf_max_rows", "_hcx_max_rows")),
    ("revert_hyperconnection", "hc", None, "BASE"),
    # hc_engagement.py host counter
    ("counter_ignores_gate", "eng", "        and decode_fused_allows(int(tokens), decode_fused_max_rows())\n", ""),
    # tpu_compiler.py compile key
    ("hash_always", "tc", "    if hcgate:\n", "    if True:\n"),
    ("hash_wrong_name", "tc", '        factors["VLLM_TPU_HC_DECODE_FUSED_MAX_ROWS"] = hcgate', '        factors["VLLM_TPU_HC_XLA_MAX_ROWS"] = hcgate'),
    # envs.py: registering would hash the variable whether set or not
    ("registered_in_envs", "envs", REG, REG + '    "VLLM_TPU_HC_DECODE_FUSED_MAX_ROWS": lambda: os.getenv("VLLM_TPU_HC_DECODE_FUSED_MAX_ROWS"),\n'),
]


def src_hash(root):
    h = hashlib.sha1()
    for dirpath, dirnames, filenames in os.walk(os.path.join(root, "vllm_torchtpu")):
        dirnames.sort()
        for name in sorted(filenames):
            if name.endswith(".py"):
                path = os.path.join(dirpath, name)
                h.update(os.path.relpath(path, root).encode())
                with open(path, "rb") as f:
                    h.update(hashlib.sha1(f.read()).digest())
    return h.hexdigest()[:12]


def run_tests(root, mode, tests, first_failure=False):
    env = dict(os.environ, Q38_L3_HCX_SRC_ROOT=root, Q38_L3_HCX_MODE=mode)
    cmd = [sys.executable]
    if mode == "mac":
        cmd += ["-I", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--noconftest"]
    else:
        env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        cmd += ["-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts="]
        if os.environ.get("Q38_L3_HCX_PYTEST_PLUGINS") == "1":  # the CDK CPU-job harness plugins (q38_plugins/)
            cmd += ["-p", "q38_cpu_device", "-p", "q38_tpu_device_gate_v2"]
    if first_failure:  # a mutant only has to fail once; stop there
        cmd += ["-x"]
    cmd += [os.path.join(HERE, t) for t in tests]
    r = subprocess.run(cmd, cwd=TREE, env=env, capture_output=True, text=True)
    lines = (r.stdout.strip().splitlines() or ["?"])
    return r.returncode, lines[-1][:100]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mac", "image"], required=True)
    ap.add_argument("--tests", nargs="+", default=None)
    a = ap.parse_args()
    tests = a.tests or ["test_hcgate_rows.py", "test_hcgate_symint.py"]
    before = src_hash(SRC)
    ok = True
    killed = 0
    with tempfile.TemporaryDirectory() as d:
        root = os.path.join(d, "src")
        shutil.copytree(os.path.join(SRC, "vllm_torchtpu"), os.path.join(root, "vllm_torchtpu"),
                        ignore=shutil.ignore_patterns("__pycache__"))
        copy_hash = src_hash(root)
        rc, last = run_tests(root, a.mode, tests)
        print(f"control-before: {'PASS' if rc == 0 else 'FAIL'} {last}", flush=True)
        ok &= rc == 0
        for name, key, old, new in M:
            path = os.path.join(root, F[key])
            src = open(path).read()
            if old is None:
                mutated = open(BASE_HC).read()
            else:
                if src.count(old) != 1:
                    print(f"BAD-ANCHOR {name} count={src.count(old)}", flush=True)
                    ok = False
                    continue
                mutated = src.replace(old, new)
            open(path, "w").write(mutated)
            try:
                rc, last = run_tests(root, a.mode, tests, first_failure=True)
            finally:
                open(path, "w").write(src)
            killed += rc != 0
            print(f"{'KILLED  ' if rc != 0 else 'SURVIVED'} {name:28s} {last}", flush=True)
        rc, last = run_tests(root, a.mode, tests)
        print(f"control-after: {'PASS' if rc == 0 else 'FAIL'} {last}", flush=True)
        ok &= rc == 0
        restored = src_hash(root) == copy_hash
        print(f"copy restored: {restored}", flush=True)
        ok &= restored
    after = src_hash(SRC)
    print(f"tree src unchanged: {before == after} ({before})", flush=True)
    ok &= before == after and killed == len(M)
    print(f"Q38_L3_HCGATE_MUTANTS {killed}/{len(M)} {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
