#!/bin/bash
# In-pod P/D setup (one pod, either role): install the staged stack, install the
# P/D sitecustomize, pull all 48 checkpoint shards into /dev/shm.
# Mirrors scripts/decode/in_pod_runner.sh (libtpu defaults to 0.0.48: mock_metadata.py
# serves all tpu-env attribute keys, and libtpu>=0.0.47 is required by tpu_sync for
# multi-process sub-chip DmaMap on tpu7x).
set -euo pipefail
[ -f /tmp/dsv41_env.sh ] && source /tmp/dsv41_env.sh
LIBTPU_VER="${DSV41_LIBTPU:-0.0.48}"
mount -o remount,size=850G /dev/shm || true
# hostPath /dev/shm outlives pods: drop regenerable caches other runs left on this node
# (perf's 86bca37 mxfp4 cache alone is ~293 GB and OOM-kills the model load). OK'd by
# the coordinator 2026-10-20.
rm -rf /dev/shm/dsv41_fp4_cache /dev/shm/vllm_cache_27arr /dev/shm/xla_cache_27arr /dev/shm/profiles /dev/shm/vllm_dp8_scale.log* /dev/shm/dsv41_runner.log*
mkdir -p /dev/shm/tmp
for sc_dst in /usr/local/lib/python3.12/sitecustomize.py /usr/local/lib/python3.12/site-packages/sitecustomize.py /usr/lib/python3.12/sitecustomize.py; do
  mkdir -p "$(dirname "$sc_dst")" && cp -f /workspace/sitecustomize_bench/sitecustomize.py "$sc_dst" || true
done
git config --global --add safe.directory '*'
export TMPDIR=/dev/shm/tmp

(
  pip uninstall -y tpu-inference torchax >/dev/null 2>&1 || true
  pip install --no-cache-dir --no-deps "libtpu==${LIBTPU_VER}" portpicker==1.6.0 compressed-tensors==0.15.0 sympy >/dev/null
  pip install --no-cache-dir --no-index --find-links=/workspace/whls --no-deps --force-reinstall /workspace/whls/*.whl >/dev/null
  if ls /workspace/whls_pd/*.whl >/dev/null 2>&1; then
    pip install --no-cache-dir --no-index --no-deps --force-reinstall /workspace/whls_pd/*.whl >/dev/null
    echo "[PIP] pd override wheels: $(ls /workspace/whls_pd)"
  fi
  SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.29.0 SETUPTOOLS_SCM_PRETEND_VERSION=0.29.0 VLLM_TARGET_DEVICE=empty pip install --no-cache-dir --no-index --no-deps --no-build-isolation -e /workspace/vllm >/dev/null
  SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 pip install --no-cache-dir --no-index --no-deps --no-build-isolation -e /workspace/vllm-torchtpu >/dev/null
  for d in /usr/local/lib/python3.12/*-packages/vllm_torchtpu; do
    [ -d "$d" ] && [ "$(realpath "$d")" != "$(realpath /workspace/vllm-torchtpu/src/vllm_torchtpu)" ] && cp -rf /workspace/vllm-torchtpu/src/vllm_torchtpu/* "$d/" || true
  done
  for d in /usr/local/lib/python3.12/*-packages/vllm; do
    [ -d "$d" ] && [ "$(realpath "$d")" != "$(realpath /workspace/vllm/vllm)" ] && cp -rf /workspace/vllm/vllm/* "$d/" || true
  done
  find /workspace -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
  python3 -c "import importlib.metadata as m; v=m.version('libtpu'); assert v=='${LIBTPU_VER}', v; print('[PIP] libtpu', v, 'vllm', m.version('vllm'), 'torch_tpu', m.version('torch_tpu'), 'tpu_sync_torch', m.version('tpu-sync-torch'))"
  echo '[PIP] done'
) &
PIP_PID=$!

# The node's /dev/shm is a hostPath shared with earlier pods: a stale `.done_all`
# marker can survive the safetensors being deleted after load, so the checkpoint
# is complete only if the marker AND all 48 shards are present.
n=$(ls /dev/shm/DeepSeek-V4.1-Flash/model-*-of-00048.safetensors 2>/dev/null | wc -l || true)
if [ "$n" != "48" ]; then
  rm -f /dev/shm/DeepSeek-V4.1-Flash/.done_*
fi
python3 -u /tmp/dsv41_dl.py
wait $PIP_PID
echo PD_SETUP_OK
