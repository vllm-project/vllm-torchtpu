# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch the DP8 Host gather benchmark with one TorchTPU process per core.

python run.py --out <artifact dir> --native <native.so> --ffi <ffi.so>
"""

import argparse
import os
import subprocess
import sys
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--ffi", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()

    if args.worker:
        import torch
        from torch import distributed as dist

        torch.set_num_threads(1)
        dist.init_process_group("tpu_dist")
        torch.empty(1, device="tpu").cpu()
        sys.path.insert(0, str(Path(__file__).resolve().parent / "reference"))
        try:
            import bench

            bench.run(args.out, args.native, args.ffi)
        except BaseException:
            path = args.out / f"rank-{os.environ['RANK']}/error.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(traceback.format_exc())
            raise
        dist.destroy_process_group()
        return

    import torch

    torch.tpu.distributed.set_tpu_launch_env(nproc_per_node=8)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=8",
            str(Path(__file__).resolve()),
            "--worker",
            "--out",
            str(args.out),
            "--native",
            str(args.native),
            "--ffi",
            str(args.ffi),
        ],
        env={**os.environ, "OMP_NUM_THREADS": "1"},
        check=True,
        timeout=3600,
    )


if __name__ == "__main__":
    main()
