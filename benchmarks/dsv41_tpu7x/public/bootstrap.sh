#!/usr/bin/env bash
# In-pod bootstrap for the DeepSeek-V4.1-Flash 1P1D recipe, from public sources only:
#   GitHub (vllm-torchtpu branch, vLLM tag), PyPI + download.pytorch.org (runtime), Hugging Face (checkpoint).
# Run once in each pod. Re-runnable.
set -euo pipefail
VTT_REPO=${VTT_REPO:-https://github.com/vllm-project/vllm-torchtpu.git}
VTT_REF=${VTT_REF:-cpk/dsv41-mh-decode-main}
VLLM_REPO=${VLLM_REPO:-https://github.com/vllm-project/vllm.git}
VLLM_TAG=${VLLM_TAG:-v0.33.0}
HF_REPO=${HF_REPO:-deepseek-ai/DeepSeek-V4.1-Flash}
W=/workspace; B=$W/vllm-torchtpu/benchmarks/dsv41_tpu7x
t0=$SECONDS

mount -o remount,size=850G /dev/shm || true
mkdir -p $W /dev/shm/tmp /dev/shm/DeepSeek-V4.1-Flash; export TMPDIR=/dev/shm/tmp
if ! command -v git >/dev/null || ! command -v gcc >/dev/null; then
  apt-get update -qq >/dev/null && apt-get install -y -qq git gcc >/dev/null
fi
git config --global --add safe.directory '*'

# 1. Sources: the ported vllm-torchtpu branch, and vLLM v0.33.0 + the four patches shipped in that branch.
rm -rf $W/vllm-torchtpu $W/vllm
git clone -q "$VTT_REPO" $W/vllm-torchtpu && git -C $W/vllm-torchtpu checkout -q "$VTT_REF"
git clone -q --filter=blob:none --branch "$VLLM_TAG" "$VLLM_REPO" $W/vllm
git -C $W/vllm -c user.name=dsv41 -c user.email=dsv41@localhost am -q $B/patches/vllm-v0.33.0/*.patch
echo "[BOOTSTRAP] vllm-torchtpu $(git -C $W/vllm-torchtpu rev-parse --short HEAD) | vllm $VLLM_TAG + $(git -C $W/vllm rev-list --count $VLLM_TAG..HEAD) patches"

# 2. Support files the role launcher expects under /workspace.
cp $B/support/mock_metadata.py $B/support/metadata_hook.c $W/
gcc -shared -fPIC -O2 -o $W/metadata_hook.so $W/metadata_hook.c -ldl
mkdir -p $W/sitecustomize_bench
cp $B/scripts/pd/sitecustomize_pd.py $W/sitecustomize_bench/sitecustomize.py
cp $B/support/sitecustomize_bench/decode_bench_seeding.py $W/sitecustomize_bench/
for d in /usr/local/lib/python3.12 /usr/local/lib/python3.12/site-packages /usr/lib/python3.12; do
  mkdir -p "$d" && cp -f $W/sitecustomize_bench/sitecustomize.py "$d/sitecustomize.py" || true
done

# 3. Runtime: upstream main's pins, all public. Runs in the background while the checkpoint downloads.
(
  pip uninstall -y tpu-inference torchax >/dev/null 2>&1 || true
  pip install --no-cache-dir --no-deps --force-reinstall --index-url https://download.pytorch.org/whl/cpu \
    torch==2.13.0 torchvision==0.28.0 >/dev/null
  pip install --no-cache-dir --no-deps libtpu==0.0.50 torch_tpu==0.2.0 tpu-sync-torch==0.0.1.dev20261029081917 \
    jax==0.11.2 jaxlib==0.11.2 portpicker==1.6.0 sympy frozendict==2.4.7 immutabledict==4.3.1 >/dev/null
  python3 -c "import torch_tpu, frozendict" >/dev/null 2>&1 || { echo "[BOOTSTRAP] torch_tpu import FAILED"; exit 1; }
  SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.33.0 SETUPTOOLS_SCM_PRETEND_VERSION=0.33.0 VLLM_TARGET_DEVICE=empty \
    pip install --no-cache-dir --no-deps --no-build-isolation -e $W/vllm >/dev/null
  SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU_CORE=0.0.0 SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 \
    pip install --no-cache-dir --no-deps --no-build-isolation -e $W/vllm-torchtpu >/dev/null
  # The image ships its own copies of both packages; make them match the checkouts.
  for d in /usr/local/lib/python3.12/*-packages/vllm_torchtpu; do
    [ -d "$d" ] && [ "$(realpath "$d")" != "$(realpath $W/vllm-torchtpu/src/vllm_torchtpu)" ] && cp -rf $W/vllm-torchtpu/src/vllm_torchtpu/* "$d/" || true
  done
  for d in /usr/local/lib/python3.12/*-packages/vllm; do
    [ -d "$d" ] && [ "$(realpath "$d")" != "$(realpath $W/vllm/vllm)" ] && cp -rf $W/vllm/vllm/* "$d/" || true
  done
  find $W -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
  python3 -c "import importlib.metadata as m; print('[BOOTSTRAP] runtime:', ' '.join(f'{p}={m.version(p)}' for p in ('torch','torch_tpu','libtpu','jax','tpu-sync-torch','vllm')))" 2>/dev/null | tail -1
) &
PIP_PID=$!

# 4. Checkpoint from Hugging Face into tmpfs (parallel ranged downloads; the driver's own downloader).
python3 - <<PYEOF
import sys
sys.path.insert(0, "$B/scripts")
from lib import dsv41 as L
open("/tmp/dsv41_dl.py", "w").write(L.DOWNLOAD_PY)
PYEOF
DSV41_HF_REPO="$HF_REPO" DSV41_GCS_BUCKET=unused DSV41_GCS_PREFIX=unused DSV41_SHARDS=all python3 -u /tmp/dsv41_dl.py
wait $PIP_PID
n=$(ls /dev/shm/DeepSeek-V4.1-Flash/model-*-of-00048.safetensors 2>/dev/null | wc -l)
[ "$n" = 48 ] || { echo "[BOOTSTRAP] checkpoint incomplete: $n/48 shards"; exit 1; }
echo "BOOTSTRAP_OK in $((SECONDS - t0))s"
