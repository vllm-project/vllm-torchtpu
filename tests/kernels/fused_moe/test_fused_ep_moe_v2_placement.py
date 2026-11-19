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
"""A static expert placement (`distributed/expert_placement.py`) only changes
which core computes each (token, expert) row, so the sequence-parallel fused
EP op must give bit-identical outputs on a relabelled checkpoint; and the
expert-load counter must count exactly the routed ids."""

import json
import os
import subprocess
import sys
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.distributed.expert_placement import ExpertPlacement
from vllm_torchtpu.kernels.fused_moe.v2 import AXIS, fused_ep_moe_v2
from vllm_torchtpu.kernels.fused_moe.v2.router_ops import expert_load_counts

# The served TP-pair schedule with the bf16 eager wire, on eight host devices
# in Pallas TPU interpret mode (a worker process: the device count is fixed
# when the backend starts); one hot expert so that a visit runs many tiles.
CFG = dict(ep=8, ep_tp=2, g_local=2, capacity=32, tokens=256, hidden=256, inter=256)
TOPK, LAYER = 4, 3


def _checkpoint(rng, e_total, hidden, inter):
    """One MoE layer as checkpoint tensors, per expert, plus the router."""
    p = f"model.layers.{LAYER}.mlp"
    fp8 = jnp.float8_e4m3fn
    out = {
        f"{p}.gate.weight": rng.standard_normal((e_total, hidden)).astype(np.float32)
        / 16,
        f"{p}.gate.e_score_correction_bias": np.r_[
            2.0, rng.standard_normal(e_total - 1) / 8
        ].astype(np.float32),
    }
    for e in range(e_total):
        out[f"{p}.experts.{e}.w13"] = np.asarray(
            jnp.asarray(rng.standard_normal((hidden, 2 * inter)) * 48, fp8)
        )
        out[f"{p}.experts.{e}.w2"] = np.asarray(
            jnp.asarray(rng.standard_normal((inter, hidden)) * 48, fp8)
        )
        out[f"{p}.experts.{e}.s13"] = (
            rng.uniform(0.5, 1.5, 2 * inter).astype(np.float32) / 2400
        )
        out[f"{p}.experts.{e}.s2"] = (
            rng.uniform(0.5, 1.5, hidden).astype(np.float32) / 2400
        )
    return out


def _operands(loaded, e_total, ep, ep_tp):
    """The op's operands as the loader lays them out: expert p of the stream
    at physical index p, core g * ep_tp + m holding TP slice m of group g."""
    p = f"model.layers.{LAYER}.mlp"
    stack = lambda k: np.stack([loaded[f"{p}.experts.{e}.{k}"] for e in range(e_total)])  # noqa: E731
    w13, w2, s13, s2 = (stack(k) for k in ("w13", "w2", "s13", "s2"))
    groups, h = ep // ep_tp, w13.shape[1]
    n, si = e_total // groups, w2.shape[1] // ep_tp
    w1 = w13.reshape(groups, n, h, 2, ep_tp, si).transpose(0, 4, 1, 2, 3, 5)
    s1 = s13.reshape(groups, n, 2, ep_tp, si).transpose(0, 3, 1, 2, 4)
    w2p = w2.reshape(groups, n, ep_tp, si, h).transpose(0, 2, 1, 3, 4)
    s2p = np.broadcast_to(s2.reshape(groups, 1, n, h), (groups, ep_tp, n, h))
    return (
        w1.reshape(-1, h, 2 * si),
        w2p.reshape(-1, si, h),
        s1.reshape(-1, 2 * si),
        s2p.reshape(-1, h),
        loaded[f"{p}.gate.weight"],
        loaded[f"{p}.gate.e_score_correction_bias"],
    )


def _placed_outputs():
    from jax._src import mesh as mesh_lib
    from jax.experimental.pallas import tpu as pltpu

    ep, ep_tp, g_local, cap = (CFG[k] for k in ("ep", "ep_tp", "g_local", "capacity"))
    e_total = ep * g_local
    rng = np.random.default_rng(29)
    ckpt = _checkpoint(rng, e_total, CFG["hidden"], CFG["inter"])
    x = jnp.asarray(rng.standard_normal((CFG["tokens"], CFG["hidden"])), jnp.bfloat16)
    mesh = Mesh(np.asarray(jax.devices()[:ep]), (AXIS,))
    sh, rep = NamedSharding(mesh, P(AXIS)), NamedSharding(mesh, P())
    rank = jax.device_put(np.arange(ep, dtype=np.int32).reshape(ep, 1), sh)
    target = mesh_lib.AbstractMesh(
        (1,),
        ("x",),
        axis_types=(jax.sharding.AxisType.Auto,),
        abstract_device=mesh_lib.AbstractDevice(
            device_kind="TPU7x", num_cores=1, platform="tpu"
        ),
    )
    with mesh_lib.use_abstract_mesh(target):
        info = pltpu.get_tpu_info()
    placements = {
        "identity": ExpertPlacement({LAYER: np.arange(e_total)}),
        "random": ExpertPlacement({LAYER: rng.permutation(e_total)}),
    }
    outs, logits = {}, {}
    with (
        patch.object(pltpu, "get_tpu_info", lambda: info),
        pltpu.force_tpu_interpret_mode(),
    ):
        for name, place in [("none", None), *placements.items()]:
            loaded = dict(
                ckpt.items() if place is None else place.relabel(ckpt.items())
            )
            w1, w2, s1, s2, gate, bias = _operands(loaded, e_total, ep, ep_tp)
            # The router as served: logits from the (relabelled) gate rows.
            lg = jnp.dot(x.astype(jnp.float32), jnp.asarray(gate).T)
            logits[name] = np.asarray(lg)
            out = fused_ep_moe_v2(
                jax.device_put(x, sh),
                *(jax.device_put(jnp.asarray(a), sh) for a in (w1, w2, s1, s2)),
                jax.device_put(lg, sh),
                rank=rank,
                topk=TOPK,
                renormalize=True,
                mesh=mesh,
                capacity=cap,
                sharded_plan=True,
                scoring_fn="sigmoid",
                score_bias=jax.device_put(jnp.asarray(bias), rep),
                wire_dtype=jnp.bfloat16,
                ep_tp_size=ep_tp,
                eager_push=True,
            )
            outs[name] = np.asarray(jax.block_until_ready(out)).astype(np.float32)
    inv = np.argsort(placements["random"].slots[LAYER])
    return dict(
        differs={k: int(np.sum(outs[k] != outs["none"])) for k in placements},
        logits_permuted=bool(np.array_equal(logits["random"], logits["none"][:, inv])),
        finite=bool(np.isfinite(outs["none"]).all()),
        nonzero=bool(np.abs(outs["none"]).sum() > 0),
    )


def test_placed_checkpoint_is_bit_identical_to_identity():
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=f"--xla_force_host_platform_device_count={CFG['ep']}",
    )
    proc = subprocess.run(
        [sys.executable, __file__], env=env, capture_output=True, text=True, timeout=900
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stdout[-3000:] + proc.stderr[-3000:]
    result = json.loads(lines[-1][len("RESULT ") :])
    assert result["finite"] and result["nonzero"], result
    assert result["logits_permuted"], result
    assert result["differs"] == {"identity": 0, "random": 0}, result


def test_expert_load_counts_equal_a_histogram_of_the_routed_ids():
    rng = np.random.default_rng(5)
    t, e, k = 300, 64, 8
    logits = jnp.asarray(rng.standard_normal((t, e)), jnp.float32)
    bias = jnp.asarray(rng.standard_normal(e) / 4, jnp.float32)
    valid = rng.random(t) > 0.25
    start = rng.integers(0, 1000, e).astype(np.int32)
    got = jax.jit(expert_load_counts, static_argnames="topk")(
        jnp.asarray(start), logits, bias, jnp.asarray(valid), topk=k
    )
    _, ids = jax.lax.top_k(jax.nn.sigmoid(logits) + bias, k)
    ids = np.asarray(ids)
    assert np.array_equal(
        np.sort(ids, 1),
        np.sort(
            np.argsort(-(1 / (1 + np.exp(-np.asarray(logits))) + np.asarray(bias)), 1)[
                :, :k
            ],
            1,
        ),
    )
    want = start + np.bincount(ids[valid].ravel(), minlength=e)
    assert got.dtype == jnp.int32
    assert np.array_equal(np.asarray(got), want)


if __name__ == "__main__":
    print("RESULT " + json.dumps(_placed_outputs()))
