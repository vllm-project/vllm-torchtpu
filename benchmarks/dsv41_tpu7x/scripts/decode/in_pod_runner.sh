#!/bin/bash
# In-pod decode runner. Adapted from the handoff's clean_nvme_runner.sh (paste 6755399435850901).
# Differences vs the original (all non-functional for the benchmark itself):
#   * stack tar comes from scripts/lib/dsv41.py build_stack (same /workspace layout)
#   * no portpicker.py shim copy (file was not in the handoff; portpicker==1.6.0 is pip-installed)
#   * checkpoint download uses the shared downloader (same range-GET algorithm, all 48 shards)
#   * the buggy bash shm-unlinker is dropped: run_dp8_scale_driver.py already deletes the
#     safetensors once all 8 ranks print "Memory statistics" (handoff 4.1 "Things that bite").
set -euo pipefail
MODE="$1"
rm -rf /dev/shm/DeepSeek-V4.1-Flash /dev/shm/vllm_cache_27arr /dev/shm/xla_cache_27arr
# hostPath /dev/shm persists across pods on a node: drop regenerable caches left by earlier runs
# (86bca37-era mxfp4.py writes ~293 GB to /dev/shm/dsv41_fp4_cache -> OOM for the next pod).
rm -rf /dev/shm/dsv41_fp4_cache /dev/shm/result_*.json /dev/shm/accuracy_*.json
mount -o remount,size=850G /dev/shm
mkdir -p /dev/shm/DeepSeek-V4.1-Flash /dev/shm/vllm_cache_27arr /dev/shm/xla_cache_27arr /dev/shm/tmp

for sc_dst in /usr/local/lib/python3.12/sitecustomize.py /usr/local/lib/python3.12/site-packages/sitecustomize.py /usr/lib/python3.12/sitecustomize.py; do
  mkdir -p "$(dirname "$sc_dst")" && cp -f /workspace/sitecustomize_bench/sitecustomize.py "$sc_dst" || true
done
git config --global --add safe.directory '*'
export TMPDIR=/dev/shm/tmp

(
  pip uninstall -y tpu-inference torchax >/dev/null 2>&1 || true
  # libtpu pinned to 0.0.44 (handoff dependency table). The image default (0.0.48 as of 2026-10-20)
  # hangs at TPU init with this torch_tpu (INJECT_SLICE_BUILDER_FAULT metadata retries, no progress).
  pip install --no-cache-dir --no-deps libtpu==${DSV41_LIBTPU:-0.0.44} portpicker==1.6.0 compressed-tensors==0.15.0 sympy >/dev/null
  pip install --no-cache-dir --no-index --find-links=/workspace/whls --no-deps --force-reinstall /workspace/whls/*.whl >/dev/null
  SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.29.0 SETUPTOOLS_SCM_PRETEND_VERSION=0.29.0 VLLM_TARGET_DEVICE=empty pip install --no-cache-dir --no-index --no-deps --no-build-isolation -e /workspace/vllm >/dev/null
  SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 pip install --no-cache-dir --no-index --no-deps --no-build-isolation -e /workspace/vllm-torchtpu >/dev/null
  for d in /usr/local/lib/python3.12/*-packages/vllm_torchtpu; do
    [ -d "$d" ] && [ "$(realpath "$d")" != "$(realpath /workspace/vllm-torchtpu/src/vllm_torchtpu)" ] && cp -rf /workspace/vllm-torchtpu/src/vllm_torchtpu/* "$d/" || true
  done
  for d in /usr/local/lib/python3.12/*-packages/vllm; do
    [ -d "$d" ] && [ "$(realpath "$d")" != "$(realpath /workspace/vllm/vllm)" ] && cp -rf /workspace/vllm/vllm/* "$d/" || true
  done
  find /workspace -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
  python3 -c 'import importlib.metadata as m; print("[PIP] libtpu", m.version("libtpu"), "vllm", m.version("vllm"), "torch_tpu", m.version("torch_tpu"))'
  echo '[PIP] done'
) &
PIP_PID=$!

python3 -u /tmp/dsv41_dl.py
wait $PIP_PID

python3 - "$MODE" <<'DRVMOD'
import sys, ast
mode = sys.argv[1]
txt = open('/tmp/run_dp8_scale_driver.py').read()
if "elif mode == \"suite_" in txt:
    txt = txt[:txt.index("elif mode == \"suite_")]
# NOTE: the handoff runner used {"dp8_c64": "[48]", "dp8_c256": "[48, 192]"}; those crash in
# tpu_runner._get_padded_token_len (AssertionError) because prefill chunks reach
# --max-num-batched-tokens=384, which exceeds the largest bucket. 384 is therefore always included;
# steady-state decode still uses the 48 / 192 bucket (c/8 reqs per rank x 6 tokens).
bucket_map = {"dp8_c64": "[48, 60, 72, 96, 384]", "dp8_c256": "[48, 192, 216, 240, 384]", "dp8_c512": "[48, 384]",
              "dp8_all": "[48, 60, 72, 96, 192, 216, 240, 384]", "tp8_c64": "[384]"}
import os
# DSV41_BUCKETS (optional) overrides the per-mode compile_sizes, e.g. "[48, 60, 72, 96, 384]".
_bk = os.environ.get("DSV41_BUCKETS", "").strip() or bucket_map[mode]
for old_b in ["[48, 96, 192, 384]", "[48, 96, 192]", "[48, 192, 384]", "[48, 384]"]:
    txt = txt.replace(old_b, _bk)
print(f"[RUNNER] compile_sizes {_bk} dp_rr={os.environ.get('DSV41_DP_RR', '0')}", flush=True)
def block(m, topo, runs, acc):
    s = f'''elif mode == "{m}":
        boot_dp8_scale_server(topology="{topo}")
        gate_meta = run_sanity_and_usa_checks()
        gate_meta["topology"] = "{topo}"
        gate_meta["gpu_memory_utilization"] = 0.90
        atomic_write_json("{acc}", gate_meta)
        prefixes = warmup_decode_bench_templates()
'''
    for out, tag, c in runs:
        s += f'''        run_scaled_benchmark("{out}", "{tag}", {c}, {c}, 1024, prefixes=prefixes, extra_meta=gate_meta)
'''
    s += f'''        print("=== COMPLETED {m.upper()} ===", flush=True)
    '''
    return s
A8, A_T = "/dev/shm/accuracy_dp8_usa_states_verified.json", "/dev/shm/accuracy_tp8_usa_states_verified.json"
C64 = ("/dev/shm/result_dp8_hbm0p90_64x1024_verified.json", "1x_c64", 64)
C256 = ("/dev/shm/result_dp8_4x_c256.json", "4x_c256", 256)
C512 = ("/dev/shm/result_dp8_8x_c512.json", "8x_c512", 512)
import os
if os.environ.get("DSV41_NO_PREFIX_CACHING") == "1":
    # handoff 6.2: the reference rows ran with prefix caching OFF
    assert "--enable-prefix-caching" in txt
    txt = txt.replace("--enable-prefix-caching", "--no-enable-prefix-caching")
    print("[RUNNER] prefix caching disabled", flush=True)
# Optional, backward compatible (unset = byte-identical launch script):
#   DSV41_EXTRA_ENV="K1=V1 K2=V2"      -> extra `export` lines before `exec vllm serve`
#   DSV41_EXTRA_SERVE_ARGS="--flag v"  -> extra serve args before `--port 8000`
# Both are spliced into the driver's f-string, so braces are doubled here.
_xenv = os.environ.get("DSV41_EXTRA_ENV", "").split()
if _xenv:
    _anchor = "exec vllm serve /dev/shm/DeepSeek-V4.1-Flash"
    assert txt.count(_anchor) == 1
    _lines = "".join(f"export {kv}\n" for kv in _xenv).replace("{", "{{").replace("}", "}}")
    txt = txt.replace(_anchor, _lines + _anchor)
    print(f"[RUNNER] extra env: {_xenv}", flush=True)
_xargs = os.environ.get("DSV41_EXTRA_SERVE_ARGS", "").strip()
if _xargs:
    _anchor = "  --port 8000\n"
    assert txt.count(_anchor) == 1
    txt = txt.replace(_anchor, "  " + _xargs.replace("{", "{{").replace("}", "}}") + " \\\\\n" + _anchor)
    print(f"[RUNNER] extra serve args: {_xargs}", flush=True)
txt += block("dp8_c64", "dp8", [C64], A8)
txt += block("dp8_c256", "dp8", [C256], A8)
txt += block("dp8_c512", "dp8", [C512], A8)
txt += block("dp8_all", "dp8", [C64, C256, C512], A8)
txt += block("tp8_c64", "tp8", [("/dev/shm/result_tp8_hbm0p90_64x1024.json", "1x_c64", 64)], A_T)
ast.parse(txt)
open('/tmp/run_dp8_scale_driver.py', 'w').write(txt)
DRVMOD

exec python3 -u /tmp/run_dp8_scale_driver.py "$MODE"
