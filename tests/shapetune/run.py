# SPDX-License-Identifier: Apache-2.0
"""Run per-shape tuning sweeps: python tests/shapetune/run.py rpad indexer ...

Each name maps to tests/shapetune/bench_<name>.py:main(emit). Results go to
stdout ("ST {json}") and $CDK_OUTPUT_DIR/shapetune_<name>.jsonl.
"""

import importlib
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402  (applies SERVED_ENV before vllm_torchtpu imports)


def main(names):
    import jax

    print("jax", jax.__version__, jax.devices()[0].device_kind, len(jax.devices()),
          file=sys.stderr, flush=True)
    for name in names:
        t0 = time.time()
        print(f"=== {name}", flush=True)
        try:
            mod = importlib.import_module(f"bench_{name}")
            mod.main(common.Emitter(name))
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        print(f"=== {name} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
