#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Run the Host gather benchmark on one v7x-8 host with the measured versions.
#
#   bash run.sh <out dir> <native.so from build_native.py> <torch_tpu wheel>
set -euo pipefail

out=$(realpath -m "$1")
native=$(realpath "$2")
wheel=$(realpath "$3")
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out/profiling" "$out/compiler"

python3 -m venv "$out/env"
py=$out/env/bin/python
"$py" -m pip install -q torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
"$py" -m pip install -q jax==0.11.2 jaxlib==0.11.2 libtpu==0.0.51 absl-py \
  portpicker tensorboard==2.21.0 frozendict immutabledict pyzmq
"$py" -m pip install -q --no-deps "$wheel"
"$py" -m pip freeze > "$out/profiling/python-packages.txt"

# Scheme 2's Host FFI handler.
ffi_include=$("$py" -c 'import jax.ffi; print(jax.ffi.include_dir())')
g++ -std=c++17 -O3 -fPIC -shared -I "$ffi_include" \
  "$here/reference/host_gather_ffi.cc" -o "$out/ffi.so"

site=$("$py" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
export LD_LIBRARY_PATH="$site/torch/lib:$site/torch_tpu/csrc/common:$site/torch_tpu/csrc/common/glue_2_13_0${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$here/../../src/vllm_torchtpu/tools${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export LIBTPU_INIT_ARGS="--xla_enable_custom_call_region_trace=true --xla_xprof_register_llo_debug_info=true"
export XLA_FLAGS="--xla_dump_to=$out/compiler --xla_dump_hlo_as_text"
"$py" "$here/run.py" --out "$out" --native "$native" --ffi "$out/ffi.so" 2>&1 |
  tee "$out/output.log"
