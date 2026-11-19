#!/usr/bin/env python3
"""Source mutants of the HCX port (q38 L3-HCX; port of q38mm 20e6fcf23). Each must make the HCX tests fail.

    python3 tests/q38_l3_hcx/q38_l3_hcx_mutrun.py --mode mac   [--tests test_hcx_rows.py]
    python3 tests/q38_l3_hcx/q38_l3_hcx_mutrun.py --mode image [--tests test_hcx_rows.py test_hcx_ladder.py]

Copies src/ once into a temp dir, applies one mutant at a time there (the tree itself is never edited), runs
pytest with Q38_L3_HCX_SRC_ROOT (and, in image mode, PYTHONPATH) on the copy, then restores the file. Controls:
the unmutated copy passes before and after; the tree's src hash is unchanged at the end. Exit 0 only if every
mutant is killed and both controls pass. Last line: Q38_L3_HCX_MUTANTS <killed>/<total> <PASS|FAIL>.
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
    "tc": "vllm_torchtpu/compilation/tpu_compiler.py",
    "envs": "vllm_torchtpu/envs.py",
}
BASE_HC = os.path.join(HERE, "data", "hyperconnection_5d368f166.py.txt")
REG = '    "VLLM_TPU_HC_DECODE_COUNT": env_nonnegative_int("VLLM_TPU_HC_DECODE_COUNT", 0),\n'
M = [
    # hc_decode.py helpers
    # q38 L3-HCGATE (T95u generalization): decided as tokens % (2 x limit) != 0; limits 128 / 256 / 512 only
    ("mod_by_limit", "hcd", "return limit > 0 and tokens % (2 * limit) != 0", "return limit > 0 and tokens % limit != 0"),
    ("limits_unchecked", "hcd", "    if rows not in _HCX_LIMITS:\n", "    if False:\n"),
    ("default_on", "hcd", "    if not value:\n        return 0\n", "    if not value:\n        return 256\n"),
    ("zero_accepted", "hcd", "    if rows < 1:\n", "    if rows < 0:\n"),
    ("float_accepted", "hcd", "        rows = int(value)\n", "        rows = int(float(value))\n"),
    ("mod_inverted", "hcd", "return limit > 0 and tokens % (2 * limit) != 0", "return limit > 0 and tokens % (2 * limit) == 0"),
    ("limit_arg_ignored", "hcd", "    if limit is None:\n        limit = decode_xla_max_rows()\n",
     "    limit = decode_xla_max_rows()\n"),
    # hyperconnection.py routing
    ("never_routes", "hc", "        xla = self._hcx_xla(hyper_input)\n", "        xla = False\n"),
    ("always_routes", "hc", "        xla = self._hcx_xla(hyper_input)\n", "        xla = True\n"),
    ("hcx_before_fused", "hc", "        if self._decode_fused(hyper_input):\n            out = self._decode_mix_op",
     "        if not self._hcx_xla(hyper_input) and self._decode_fused(hyper_input):\n            out = self._decode_mix_op"),
    ("norm_tile_kept", "hc", "            not xla\n            and self._prefill_norm_op is not None",
     "            self._prefill_norm_op is not None"),
    ("normalize_flag_dropped", "hc", "norm = self._normalize(hyper_input, xla=xla)", "norm = self._normalize(hyper_input)"),
    ("up_tile_kept", "hc", "            not xla\n            and self._prefill_up_op is not None",
     "            self._prefill_up_op is not None"),
    ("full_mix_kept", "hc", "            not xla\n            and self._prefill_full_op is not None",
     "            self._prefill_full_op is not None"),
    ("fused_prefill_kept", "hc", "        if not xla and self._prefill_fused_mix(hyper_input):",
     "        if self._prefill_fused_mix(hyper_input):"),
    ("combine_uncovered", "hc",
     "            and not self._hcx_xla(hyper_input)\n            and self._prefill_full_rows(hyper_input.shape[0])",
     "            and self._prefill_full_rows(hyper_input.shape[0])"),
    ("combine_mix_uncovered", "hc", "            and not next_hc._hcx_xla(hyper_input)\n", ""),
    ("check_without_tiles", "hc", "            and has_tiles\n", ""),
    ("no_dim_check", "hc", "            and hyper_input.dim() == 2\n            and decode_xla_applies",
     "            and decode_xla_applies"),
    ("combine_op_not_a_tile", "hc", "            or self._prefill_combine_op is not None\n        )", "        )"),
    ("init_reads_zero", "hc", "        self._hcx_max_rows = decode_xla_max_rows()", "        self._hcx_max_rows = 0"),
    ("off_reads_rows", "hc", "            self._hcx_max_rows > 0\n            and has_tiles", "            has_tiles"),
    ("injection_left_to_combine", "hc", "        return mixed.to(hyper_input.dtype), self._residual(hyper_input, norm)",
     "        return mixed.to(hyper_input.dtype), (hyper_input, norm)"),
    ("revert_hyperconnection", "hc", None, "BASE"),
    # tpu_compiler.py compile key
    ("hash_never", "tc", '        "VLLM_TPU_HC_XLA_MAX_ROWS",\n    ):', "    ):"),
    ("hash_always", "tc", '        value = os.environ.get(switch, "").strip()\n        if value:\n',
     '        value = os.environ.get(switch, "").strip()\n        if True:\n'),
    # envs.py: registering would hash the variable whether set or not
    ("registered_in_envs", "envs", REG, REG + '    "VLLM_TPU_HC_XLA_MAX_ROWS": lambda: os.getenv("VLLM_TPU_HC_XLA_MAX_ROWS"),\n'),
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
    tests = a.tests or (["test_hcx_rows.py"] if a.mode == "mac" else ["test_hcx_rows.py", "test_hcx_ladder.py"])
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
    print(f"Q38_L3_HCX_MUTANTS {killed}/{len(M)} {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
