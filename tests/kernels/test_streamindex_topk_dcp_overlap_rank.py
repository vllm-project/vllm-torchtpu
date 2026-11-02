# SPDX-License-Identifier: Apache-2.0
"""streamindex_topk_dcp(overlap_rank=True): the overlap work is handed this
rank's DCP index and lane after its own arguments. The scoring, the SparseCore
merge and the candidate gather are stubbed (they need a TPU); the shard_map,
the rank and lane plumbing and the pinning are the real ones. 16 CPU devices in a
subprocess (the host device count must be set before JAX starts)."""

import json
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.cpu_test
N = 16


def _worker():
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh

    mod = sys.modules.get(
        "vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk"
    )
    if mod is None:
        import vllm_torchtpu.kernels.deepseek_v4.streamindex_topk  # noqa: F401

        mod = sys.modules[
            "vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk"
        ]
    k, t = 128, 4

    def local(q, *args, return_scores, final_sc_tie=(), **kw):
        idxs = jnp.zeros((t, k), jnp.int32)
        scores = jnp.zeros((t, k), jnp.int32)
        out = (idxs, scores) if return_scores else idxs
        return (*out, tuple(final_sc_tie)) if final_sc_tie else out

    mod.streamindex_topk = local
    mod.sparsecore_topk_dcp_merge = lambda scores, idxs, *a, **kw: idxs
    seen = []

    def record(rank, lane):
        seen.append((int(rank[0]), None if lane is None else int(lane[0])))

    def overlap(x, w, rank, lane):
        args = (rank,) if lane is None else (rank, lane)
        jax.debug.callback(lambda *v: record(v[0], v[1] if len(v) > 1 else None), *args)
        return x * 2

    result = {}
    for lanes in (False, True):
        devices = np.asarray(sorted(jax.devices("cpu"), key=lambda d: d.id))
        mesh = (
            Mesh(devices.reshape(1, N), ("lane", "dcp"))
            if lanes
            else Mesh(devices, ("dcp",))
        )
        seen.clear()
        x = jnp.ones((t, 8), jnp.float32)
        out = mod.streamindex_topk_dcp(
            jnp.zeros((t, 2, 8), jnp.float32),
            jnp.zeros((t, 2), jnp.float32),
            jnp.zeros((N * 4, 8), jnp.uint8),
            jnp.zeros((4,), jnp.int32),
            jnp.zeros((16,), jnp.int32),
            jnp.zeros((5,), jnp.int32),
            jnp.zeros((3,), jnp.int32),
            mesh=mesh,
            k=k,
            compression_ratio=1,
            dcp_size=N,
            interleave_size=4,
            num_kv_pages_per_block=(2, 2, 2),
            num_queries_per_block=(1, 128, 4),
            dcp_axis_name="dcp",
            dcp_rank_ids=jnp.arange(N, dtype=jnp.int32),
            overlap_fn=overlap,
            overlap_args=(x, jnp.zeros((8,), jnp.float32)),
            overlap_pin_max_tokens=16,
            overlap_rank=True,
            flat_merge=False,
            small_all_gather=False,
        )
        jax.block_until_ready(out)
        result[str(lanes)] = dict(
            ranks=sorted(r for r, _ in seen),
            lanes=sorted({lane for _, lane in seen}, key=str),
            overlap_out=float(np.asarray(out[1]).max()),
        )
    print("RESULT " + json.dumps(result))


@pytest.fixture(scope="module")
def on_16_devices():
    flags = os.environ.get("XLA_FLAGS", "")
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=f"{flags} --xla_force_host_platform_device_count={N}",
    )
    proc = subprocess.run(
        [sys.executable, __file__],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stdout[-3000:] + proc.stderr[-3000:]
    return json.loads(lines[-1][len("RESULT ") :])


@pytest.mark.parametrize("lanes", [False, True])
def test_overlap_work_gets_every_rank_and_the_lane(on_16_devices, lanes):
    """Each shard's work sees its own DCP index (0..15 once each) and, on a
    (lane, dcp) mesh with one lane, lane 0; without a lane axis, None."""
    got = on_16_devices[str(lanes)]
    assert got["ranks"] == list(range(N))
    assert got["lanes"] == ([0] if lanes else [None])
    assert got["overlap_out"] == 2.0


if __name__ == "__main__":
    _worker()
