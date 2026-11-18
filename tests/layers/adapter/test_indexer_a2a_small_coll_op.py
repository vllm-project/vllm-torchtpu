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
"""TPU_DCP_INDEXER_A2A_SMALL_COLL through the DCP indexer op, on CPU.

A worker on 8 CPU devices builds the op with
`VllmTPUSparseAttnIndexer._build_streamidx_dcp_op`, flag on and flag off,
and runs the JAX function each one registers on a (2 lanes, 4 DCP ranks)
mesh, with the Pallas kernels in TPU interpret mode and the race detector
on. The local scoring kernel and the SparseCore top-k and compaction have no
CPU path, so exact JAX stand-ins replace them; the indexer K write and both
small-collective kernels run as they are. Checked:

* flag on, the op takes the rank and lane operands; flag off (and
  TPU_DCP_RANK_OPERAND unset), neither;
* flag on, the traced op holds the score all-to-all and the slot all-gather
  kernels; flag off, neither;
* the new K cache and the top-k equal the flag-off op's on all 8 devices,
  on a bucket of one token per rank (straight into HBM) and one of four
  (staged in VMEM), with no races;
* rotated rank ids change the result, so the rank operand is read.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import vllm_torchtpu
from vllm_torchtpu import envs

DCP, LANES, K = 4, 2, 2048
BUCKETS = (4, 16)  # one and four tokens per rank
# Below the CPU lane's 600 s pytest timeout, so a hung worker fails here with
# its output. The worker takes about 15 s on a 64-core workstation.
WORKER_TIMEOUT_S = 480


def test_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("TPU_DCP_INDEXER_A2A_SMALL_COLL", raising=False)
    assert envs.TPU_DCP_INDEXER_A2A_SMALL_COLL is False
    monkeypatch.setenv("TPU_DCP_INDEXER_A2A_SMALL_COLL", "1")
    assert envs.TPU_DCP_INDEXER_A2A_SMALL_COLL is True


def _tail(out) -> str:
    if isinstance(out, bytes):
        out = out.decode(errors="replace")
    return (out or "")[-3000:]


@pytest.fixture(scope="module")
def results():
    flags = os.environ.get("XLA_FLAGS", "")
    src = str(Path(vllm_torchtpu.__file__).resolve().parents[1])
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=f"{flags} --xla_force_host_platform_device_count={DCP * LANES}",
        # The worker imports the package this process imported.
        PYTHONPATH=os.pathsep.join([src, os.environ.get("PYTHONPATH", "")]),
    )
    try:
        proc = subprocess.run(
            [sys.executable, __file__, "worker"],
            env=env,
            capture_output=True,
            text=True,
            timeout=WORKER_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as e:
        output = _tail(e.stdout) + _tail(e.stderr)
        pytest.fail(f"worker timed out after {e.timeout} s\n{output}")
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, _tail(proc.stdout) + _tail(proc.stderr)
    return json.loads(lines[-1][len("RESULT ") :])


def test_op_takes_rank_and_lane_operands_only_with_the_flag(results):
    # q_bytes, weights, cache_kv, k_packed, seq_lens, page_indices,
    # cu_q_lens, distribution; then the rank and lane ids.
    assert results["schema"] == {"on": 10, "off": 8}


@pytest.mark.parametrize("num_tokens", BUCKETS, ids=["direct_hbm", "vmem_staged"])
def test_flag_on_runs_the_kernels_and_matches_flag_off(results, num_tokens):
    got = results["buckets"][str(num_tokens)]
    assert got["kernels_on"] == {"score_a2a": True, "slot_all_gather": True}, got
    assert got["kernels_off"] == {"score_a2a": False, "slot_all_gather": False}, got
    assert got["cache_equal"] and got["topk_equal"], got
    assert got["topk_nonempty"], got
    assert not got["races_on"] and not got["races_off"], got


def test_rotated_rank_ids_are_caught(results):
    assert results["rotated_ranks_differ"], results


def _worker():
    import contextlib
    import io
    from unittest import mock

    import jax
    import jax.numpy as jnp
    import numpy as np
    import torch
    from jax import lax
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.distributed.dcp import DCP_AXIS_NAME, DCP_LANE_AXIS_NAME
    from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as sc_module
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import dcp_sc_compact
    from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op as mop

    heads, head_dim, pages, page_size, lkv_dim = 8, 128, 4, 128, 256
    rng = np.random.default_rng(5)
    # Distinct scores per (rank, token, slot), so the top-k has no ties.
    scores_by_t = {
        t: rng.permutation(DCP * t * K).astype(np.int32).reshape(DCP, t, K)
        for t in BUCKETS
    }
    idxs_by_t = {
        t: (np.arange(DCP * t * K, dtype=np.int32).reshape(DCP, t, K) * 7) % 4096
        for t in BUCKETS
    }

    def scoring(q, w, cache_kv, *args, cp_rank=None, final_sc_tie=(), **kwargs):
        # This rank's local candidates; the cache read keeps the K write live.
        rank = jnp.asarray(cp_rank, jnp.int32)
        live = cache_kv[0, 0, 0, 0].astype(jnp.int32) * 0
        scores = jnp.asarray(scores_by_t[q.shape[0]])[rank] + live
        idxs = jnp.asarray(idxs_by_t[q.shape[0]])[rank]
        if final_sc_tie:
            return (
                idxs,
                scores,
                tuple(lax.optimization_barrier(x) for x in final_sc_tie),
            )
        return idxs, scores

    def sc_topk(scores, k, row_lengths, **kwargs):
        vals, slots = lax.top_k(scores, k)
        slots = slots.astype(jnp.int32)
        return (vals, slots) if kwargs.get("return_scores", True) else slots

    def pack_owned_winners(slots, idxs, dcp_rank):
        width = slots.shape[1]
        lo = jnp.asarray(dcp_rank, jnp.int32) * width
        owned = (slots >= lo) & (slots < lo + width)
        local = jnp.take_along_axis(idxs, jnp.clip(slots - lo, 0, width - 1), axis=1)
        col = jnp.arange(width, dtype=jnp.int32)[None, :]
        order = jnp.argsort(jnp.where(owned, col, width + col), axis=1)
        return jnp.take_along_axis(jnp.where(owned, local, -1), order, axis=1)

    st_mod = sys.modules[
        "vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk"
    ]
    st_mod.streamindex_topk = scoring
    sc_module._sc_topk_direct = sc_topk
    dcp_sc_compact.pack_owned_winners = pack_owned_winners

    devices = sorted(jax.devices("cpu"), key=lambda d: d.id)
    mesh = Mesh(
        np.asarray(devices).reshape(LANES, DCP), (DCP_LANE_AXIS_NAME, DCP_AXIS_NAME)
    )
    replicated = NamedSharding(mesh, P())
    sharded = NamedSharding(mesh, P(DCP_AXIS_NAME))

    def build(flag):
        """The op's registered JAX function and its torch schema."""
        captured = {}
        register = mop.pcp_streaming_jax_op

        def capture(name, fn, **kwargs):
            captured.update(name=name, fn=fn)
            return register(name, fn, **kwargs)

        env = {
            "TPU_DCP_INDEXER_A2A_SMALL_COLL": "1" if flag else "0",
            # Every bucket here takes the all-to-all merge.
            "TPU_DCP_INDEXER_HIST_MERGE": "0",
            "TPU_DCP_INDEXER_A2A_MIN_TOKENS": "4",
            "TPU_DCP_RANK_OPERAND": "0",
            "VLLM_TPU_INDEXER_KV_CACHE_MODE": "xla",
        }
        stub = mock.Mock(topk_tokens=K, _shared_meta=False)
        stub.k_cache.prefix = f"model.layers.0.self_attn.indexer_a2a_{int(flag)}"
        zero = torch.zeros(1, dtype=torch.int32)
        with (
            mock.patch.dict(os.environ, env),
            mock.patch.object(mop, "get_or_create_dcp_mesh", return_value=mesh),
            mock.patch.object(mop, "dcp_rank_operand", return_value=zero),
            mock.patch.object(mop, "dcp_lane_operand", return_value=zero),
            mock.patch.object(mop, "pcp_streaming_jax_op", side_effect=capture),
        ):
            mop.VllmTPUSparseAttnIndexer._build_streamidx_dcp_op(stub, DCP, 1)
        op = getattr(torch.ops.pallas, captured["name"].split("::")[1])
        return captured["fn"], len(op.default._schema.arguments)

    def per_device(value):
        """i32[DCP] over the DCP axis whose shard on device (lane, rank) holds
        value(lane, rank), as each process's own operand does under torch."""
        shards = [
            jax.device_put(np.asarray([value(lane, rank)], np.int32), d)
            for (lane, rank), d in np.ndenumerate(mesh.devices)
        ]
        return jax.make_array_from_single_device_arrays((DCP,), sharded, shards)

    def inputs(t):
        q = rng.integers(0, 256, size=(t, heads, head_dim), dtype=np.uint8)
        weights = rng.standard_normal((t, heads)).astype(np.float32)
        cache = rng.integers(
            0, 256, size=(DCP * pages, page_size // 4, 4, lkv_dim), dtype=np.uint8
        )
        k_packed = rng.integers(0, 256, size=(t, head_dim + 1), dtype=np.uint8)
        # Two requests, long enough that every rank owns some new K rows.
        return (
            jax.device_put(q, replicated),
            jax.device_put(jnp.asarray(weights, jnp.bfloat16), replicated),
            jax.device_put(cache, sharded),
            jax.device_put(k_packed, replicated),
            jax.device_put(jnp.array([300, 1000], jnp.int32), replicated),
            jax.device_put(jnp.array([0, 1, 2, 3] * 2, jnp.int32), replicated),
            jax.device_put(jnp.array([0, t // 4, t], jnp.int32), replicated),
            jax.device_put(jnp.array([0, 0, 2], jnp.int32), replicated),
        )

    def kernels(fn, args):
        jaxpr = str(jax.make_jaxpr(fn)(*args))
        return {
            "score_a2a": "small_idx_score_a2a" in jaxpr,
            "slot_all_gather": "small_all_gather" in jaxpr,
        }

    def run(fn, args):
        """Every device's (cache, top-k) shards, and whether a race was seen."""
        params = pltpu.InterpretParams(detect_races=True, dma_execution_mode="eager")
        pltpu.reset_tpu_interpret_mode_state()
        with (
            pltpu.force_tpu_interpret_mode(params),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            out = jax.block_until_ready(jax.jit(fn)(*args))
        races = interpret_pallas_call.races
        shards = [
            [
                np.asarray(s.data)
                for s in sorted(o.addressable_shards, key=lambda s: s.device.id)
            ]
            for o in out
        ]
        return shards, bool(races is not None and races.races_found)

    def same(a, b):
        return all(np.array_equal(x, y) for x, y in zip(a, b))

    fn_on, schema_on = build(True)
    fn_off, schema_off = build(False)
    ranks = per_device(lambda lane, rank: rank)
    lanes = per_device(lambda lane, rank: lane)
    buckets = {}
    rotated_differ = None
    for t in BUCKETS:
        args = inputs(t)
        (cache_on, topk_on), races_on = run(fn_on, (*args, ranks, lanes))
        (cache_off, topk_off), races_off = run(fn_off, args)
        buckets[str(t)] = dict(
            kernels_on=kernels(fn_on, (*args, ranks, lanes)),
            kernels_off=kernels(fn_off, args),
            cache_equal=same(cache_on, cache_off),
            topk_equal=same(topk_on, topk_off),
            topk_nonempty=bool(all((s >= 0).any() for s in topk_on)),
            races_on=races_on,
            races_off=races_off,
        )
        if rotated_differ is None:
            rotated = per_device(lambda lane, rank: (rank + 1) % DCP)
            (_, topk_bad), _ = run(fn_on, (*args, rotated, lanes))
            rotated_differ = not same(topk_bad, topk_off)
    result = dict(
        schema={"on": schema_on, "off": schema_off},
        buckets=buckets,
        rotated_ranks_differ=rotated_differ,
    )
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__" and sys.argv[1:] == ["worker"]:
    _worker()
