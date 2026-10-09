# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tiling experiments: measures what the kernel needs, to fit tiling's models.

E1 (test_vmem): per model and tile size, bisects the smallest scoped VMEM
limit fused_qkv compiles with and compares it with tiling.vmem_bytes. The
search stays HEADROOM below capacity - weights, because limits near it can
hang XLA's weight placement.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import os
import re
import time
from collections.abc import Callable
from typing import Any

import jax
from absl.testing import absltest, parameterized
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.batched_rpa import configs as rpa_configs
from vllm_torchtpu.kernels.experimental.fused_qkv import models
from vllm_torchtpu.kernels.experimental.fused_qkv.kernel import rpa, tiling

TILES = (128, 256, 512)
KV_LAYOUT = rpa_configs.KVLayout.SEQ_ALONG_LANE
STEP = 128 * 2**10  # Bisection resolution.
HEADROOM = 12 * 2**20
_SIZE = r"([\d.]+)([KMG]?)B?"
_MOSAIC_OOM = re.compile(
    rf"Scoped allocation with size {_SIZE} and limit {_SIZE} exceeded scoped"
    rf" vmem limit by {_SIZE}"
)
_XLA_OOM = re.compile(rf"Program vmem requirement {_SIZE}")
_WEIGHTS_OOM = "Too many buffers are colored in the alternate memory"
_SPILL = re.compile(rf"Size: {_SIZE}\s+XLA label: register allocator spill")
_UNITS = {"": 1, "K": 2**10, "M": 2**20, "G": 2**30}


def _bytes(num: str, unit: str) -> float:
    return float(num) * _UNITS[unit]


def nbytes(tree: Any) -> int:
    """Total bytes of the arrays (or ShapeDtypeStructs) in a pytree."""
    return sum(a.size * a.dtype.itemsize for a in jax.tree.leaves(tree))


def compile_error(model: models.Model, tile_size: int, limit: int) -> str | None:
    """Compiles fused_qkv with 4 tiles; returns None or the VMEM OOM message."""
    s = model.config
    t = 4 * tile_size  # More than one tile, so the pipeline streams.
    x, residual, weights, cos, sin = jax.eval_shape(
        lambda: models.make_inputs(model, t)
    )
    weights = jax.eval_shape(lambda w: rpa.prepare_weights(s, w), weights)
    tiling_ = dataclasses.replace(
        rpa.choose_tiling(s, t, weights, kv_layout=KV_LAYOUT, tile_size=tile_size),
        vmem_limit_bytes=int(limit),
    )
    assert tiling_.tile_size == tile_size
    start = time.time()
    try:
        rpa.fused_qkv.lower(
            s, x, residual, weights, cos, sin, tiling=tiling_, kv_layout=KV_LAYOUT
        ).compile()
    except Exception as e:  # pylint: disable=broad-exception-caught
        msg = str(e)
        if not (_MOSAIC_OOM.search(msg) or _XLA_OOM.search(msg) or _WEIGHTS_OOM in msg):
            raise
    else:
        msg = None
    print(
        f"  compile {model.name} tile={tile_size} limit={limit / 2**20:.3f}M:"
        f" {'ok' if msg is None else parse(msg)} ({time.time() - start:.0f}s)",
        flush=True,
    )
    return msg


def parse(msg: str) -> dict[str, float]:
    """OOM byte counts: first_alloc, exceeded_by, program, spill, weights_oom."""
    out = {}
    if m := _MOSAIC_OOM.search(msg):
        out["first_alloc"] = _bytes(*m.group(1, 2))
        out["exceeded_by"] = _bytes(*m.group(5, 6))
    if m := _XLA_OOM.search(msg):
        out["program"] = _bytes(*m.group(1, 2))
    if m := _SPILL.search(msg):
        out["spill"] = _bytes(*m.group(1, 2))
    if _WEIGHTS_OOM in msg:  # XLA could not place the weights.
        out["weights_oom"] = 1.0
    return out


def bisect(fits: Callable[[int], bool], lo: int, hi: int) -> tuple[int, int]:
    """Narrows (lo, hi), where fits(lo) != fits(hi), to the edge within STEP."""
    while hi - lo > STEP:
        mid = (lo + hi) // 2 // STEP * STEP
        if fits(mid) == fits(lo):
            lo = mid
        else:
            hi = mid
    return lo, hi


class TilingExperiment(parameterized.TestCase):
    @parameterized.named_parameters(*((f"_{m.name}", m.name) for m in models.MODELS))
    def test_vmem(self, name: str) -> None:
        """E1: smallest VMEM limit that compiles vs. tiling.vmem_bytes."""
        model = models.BY_NAME[name]
        s = model.config
        weights = jax.eval_shape(
            lambda: rpa.prepare_weights(s, models.make_inputs(model, 128)[2])
        )
        hi = pltpu.get_tpu_info().vmem_capacity_bytes - nbytes(weights) - HEADROOM
        hi = hi // STEP * STEP
        rows = []
        for tile_size in TILES:
            terms = tiling.vmem_bytes(
                rpa.bytes_per_row(s, KV_LAYOUT, model.hidden_size), tile_size
            )
            row = dict(tile=tile_size, predicted=sum(terms.values()), terms=terms)

            @functools.cache
            def fits(limit: int, tile_size: int = tile_size) -> bool:
                return compile_error(model, tile_size, limit) is None

            if fits(hi):
                row["min_limit"] = bisect(fits, 0, hi)[1]
            else:
                row.update(parse(compile_error(model, tile_size, hi)))
                row["fails_at"] = hi
            rows.append(row)
            print(
                f"=== E1 {name} tile={tile_size:5d} "
                + " ".join(
                    f"{k}={v / 2**20:.2f}M"
                    for k, v in row.items()
                    if k not in ("tile", "terms")
                ),
                flush=True,
            )
            print(
                f"=== E1 {name} tile={tile_size:5d} terms "
                + " ".join(f"{k}={v / 2**20:.2f}M" for k, v in terms.items() if v),
                flush=True,
            )
        print(f"=== E1 {name} weights={nbytes(weights) / 2**20:.2f}M", flush=True)
        out_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "/tmp")
        with open(f"{out_dir}/e1_{name}.json", "w") as fh:
            json.dump(rows, fh)


if __name__ == "__main__":
    absltest.main()
