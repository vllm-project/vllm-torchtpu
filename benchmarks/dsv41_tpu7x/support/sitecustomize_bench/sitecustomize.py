import os
os.environ.setdefault("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS", "false")
os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")
# _PODJ_NUMA_PIN
try:
    _cid = int(os.environ.get("POD_G_CHIP_ID", "-1"))
    if 0 <= _cid <= 3:
        os.sched_setaffinity(0, set(range(_cid * 48, (_cid + 1) * 48)))
except Exception:
    pass
# SPDX-License-Identifier: Apache-2.0
"""Post-import hook to install DecodeBenchConnector TPU seeding when enabled."""

import importlib.abc
import os
import sys

_TARGET = "vllm.distributed.kv_transfer.kv_connector.v1.decode_bench_connector"


class _DecodeBenchPatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if (
            fullname == _TARGET
            and os.environ.get("ENABLE_DECODE_BENCH_CONNECTOR", "1") == "1"
        ):
            sys.meta_path.remove(self)
            for finder in sys.meta_path:
                spec = finder.find_spec(fullname, path, target)
                if spec is not None and spec.loader is not None:
                    orig_exec = spec.loader.exec_module

                    def _exec_module(module):
                        orig_exec(module)
                        from decode_bench_seeding import patch_decode_bench_seeding

                        patch_decode_bench_seeding()

                    spec.loader.exec_module = _exec_module
                    return spec
        return None


sys.meta_path.insert(0, _DecodeBenchPatchFinder())
