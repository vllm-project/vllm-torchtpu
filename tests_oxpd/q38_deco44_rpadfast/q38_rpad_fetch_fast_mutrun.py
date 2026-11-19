# SPDX-License-Identifier: Apache-2.0
"""Source mutants of the PD port of MM's RPAd `fast` fetch (DEC-O4-4; from MM's q38mm_rpad_fetch_mutrun.py at 92c1f3fbb, the
7 mutants that apply to `fast`, plus PD-port mutants). Each mutant edits a COPY of src/ (never the worktree) and must make its
test file fail. Run: python tests/q38_deco44_rpadfast/q38_rpad_fetch_fast_mutrun.py [extra PYTHONPATH]"""
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "src")
EXTRA = sys.argv[1] if len(sys.argv) > 1 else ""
B = "vllm_torchtpu/kernels/experimental/batched_rpa/"
BO = B + "bref_override.py"
FAST_T = os.path.join(HERE, "test_rpad_fetch_fast.py")
KEY_T = os.path.join(ROOT, "tests", "compilation", "test_tpu_compile_key_registration.py")
M = [
    # MM's fast mutants (anchors as in 92c1f3fbb)
    ("fast_src_next_page", BO, "pltpu.make_async_copy(\n                            kv_cache_hbm_flat.at[pl.ds(row[i] * page_size, page_size)]",
     "pltpu.make_async_copy(\n                            kv_cache_hbm_flat.at[pl.ds(row[i] * page_size + page_size, page_size)]", FAST_T),
    ("fast_full_any_tile", BO, "full = tokens == pages * page_size", "full = tokens > 0", FAST_T),
    ("fast_ignored", BO, "pl.when(full)(_full_tile)", "pass", FAST_T),
    ("part_prefix_off_by_one", BO, "@pl.when(i < n_full)", "@pl.when(i <= n_full)", FAST_T),
    ("part_tail_full_size", BO, "kv_cache_hbm_flat.at[pl.ds(row[n_full] * page_size, rem)]",
     "kv_cache_hbm_flat.at[pl.ds(row[n_full] * page_size, page_size)]", FAST_T),
    ("part_range_unchecked", BO, "partial = jnp.logical_not(full) & (tokens > 0) & in_range",
     "partial = jnp.logical_not(full) & (tokens > 0)", FAST_T),
    ("part_empty_tiles_issue", BO, "jnp.logical_not(full) & (tokens > 0) & jnp.logical_not(in_range)",
     "jnp.logical_not(full) & jnp.logical_not(partial)", FAST_T),
    # PD-port mutants
    ("pd_in_range_off_by_one", BO, "in_range = (p0 + (pages - 1)) <= last_index", "in_range = (p0 + (pages - 1)) < last_index", FAST_T),
    ("pd_kernel_suffix_dropped", B + "kernel.py", "    name += configs.KV_FETCH_MODES[serve.kv_fetch_mode]\n", "", FAST_T),
    ("pd_passthrough_dropped", "vllm_torchtpu/kernels/qwen4_exp/qsa_rpa.py",
     'kv_fetch_mode=envs.VLLM_TPU_QSA_DECODE_RPA_FETCH if compact else "",', 'kv_fetch_mode="",', FAST_T),
    ("pd_passthrough_without_compact", "vllm_torchtpu/kernels/qwen4_exp/qsa_rpa.py",
     'kv_fetch_mode=envs.VLLM_TPU_QSA_DECODE_RPA_FETCH if compact else "",', "kv_fetch_mode=envs.VLLM_TPU_QSA_DECODE_RPA_FETCH,", FAST_T),
    ("pd_extra_mode_accepted", B + "configs.py", '    "fast": "-f",\n}', '    "fast": "-f",\n    "runs": "-r",\n}', FAST_T),
    ("pd_compile_key_ignored", "vllm_torchtpu/compilation/tpu_compiler.py", "_TPU_COMPILE_ENV_IGNORED = {",
     "_TPU_COMPILE_ENV_IGNORED = {\n    \"VLLM_TPU_QSA_DECODE_RPA_FETCH\",", KEY_T),
]


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


killed = 0
for name, f, old, new, test in M:
    tmp = tempfile.mkdtemp(prefix="q38_rpadfast_mut_")
    root = os.path.join(tmp, "src")
    shutil.copytree(SRC, root)
    path = os.path.join(root, f)
    s = open(path).read()
    assert s.count(old) == 1, f"{name}: anchor found {s.count(old)} times"
    open(path, "w").write(s.replace(old, new))
    env = dict(os.environ, PYTHONPATH=root + (os.pathsep + EXTRA if EXTRA else ""))
    args = [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "-W", "ignore", test]
    if test == FAST_T:
        args.insert(-1, "--noconftest")
    r = subprocess.run(args, capture_output=True, text=True, env=env, cwd=ROOT)
    ok = r.returncode != 0
    killed += ok
    last = [ln for ln in r.stdout.splitlines() if ln.strip()][-1:]
    print(f"{'KILLED ' if ok else 'SURVIVED'} {name}: {last[0] if last else ''}", flush=True)
    shutil.rmtree(tmp)
print(f"RPAD FAST PORT MUTANTS: {killed}/{len(M)} killed")
sys.exit(0 if killed == len(M) else 1)
