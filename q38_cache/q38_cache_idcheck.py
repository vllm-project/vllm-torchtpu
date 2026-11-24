"""q38 CACHE lane: in-image check of core/prefix_cache_policy.py before any serving run (BENCH=script).

1. Reports the image vLLM's KVCacheBlock hash API and runs prefix_cache_policy.queue_contract_ok().
2. Runs, with the vendored pytest (q38_cache/vendor), the CPU suite, the real-vLLM integration suite and the
   existing retention suite.
3. Mutation check on the image: re-runs the real-vLLM suite against copies of the module with
   (a) the srv1 bug (hash set through a property setter) and (b) demotion placed at the MRU end; each must fail.
Prints one line `Q38_CACHE_IDCHECK ok=... contract=... pytest_rc=... mutants_bite=...` and exits 0 only if all pass.
"""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(ROOT, "q38_cache", "vendor")
SUITES = ["tests/core/test_prefix_cache_policy.py", "tests/core/test_prefix_cache_policy_vllm.py",
          "tests/platforms/test_mamba_checkpoint_retention.py"]
MUTANTS = {
    "srv1_property_setter": ("    if callable(setter):\n        setter(block_hash)\n    else:\n        block.block_hash = block_hash",
                             "    block.block_hash = block_hash"),
    "demote_to_mru": ("            _link_after(queue, anchor, block)\n            anchor = block",
                      "            _link_after(queue, queue.fake_free_list_tail.prev_free_block, block)\n            anchor = block"),
}


def pytest(paths, src):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([VENDOR, src, env.get("PYTHONPATH", "")])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--noconftest", *paths],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr)[-2500:]


def main():
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

    prop = getattr(KVCacheBlock, "block_hash", None)
    print("Q38_CACHE_API set_block_hash=%s block_hash_setter=%s" % (
        hasattr(KVCacheBlock, "set_block_hash"), isinstance(prop, property) and prop.fset is not None), flush=True)
    from vllm_torchtpu.core import prefix_cache_policy as pcp

    contract = pcp.queue_contract_ok()
    print("Q38_CACHE_CONTRACT %s" % contract, flush=True)
    rc, tail = pytest(SUITES, os.path.join(ROOT, "src"))
    print(tail, flush=True)
    print("Q38_CACHE_PYTEST_RC=%d" % rc, flush=True)
    bite = {}
    for name, (old, new) in MUTANTS.items():
        tmp = tempfile.mkdtemp(prefix="q38mut_")
        shutil.copytree(os.path.join(ROOT, "src"), os.path.join(tmp, "src"))
        f = os.path.join(tmp, "src", "vllm_torchtpu", "core", "prefix_cache_policy.py")
        s = open(f).read()
        assert old in s, name
        open(f, "w").write(s.replace(old, new, 1))
        mrc, mtail = pytest(["tests/core/test_prefix_cache_policy_vllm.py"], os.path.join(tmp, "src"))
        bite[name] = mrc != 0
        print("Q38_CACHE_MUTANT %s rc=%d bites=%s :: %s" % (name, mrc, bite[name], mtail.strip().splitlines()[-1] if mtail.strip() else ""), flush=True)
        shutil.rmtree(tmp, ignore_errors=True)
    ok = contract and rc == 0 and all(bite.values())
    print("Q38_CACHE_IDCHECK ok=%s contract=%s pytest_rc=%d mutants_bite=%s" % (ok, contract, rc, all(bite.values())), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
