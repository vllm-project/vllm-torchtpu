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
"""moe_tp_decode under a static expert placement (expert_columns): routing
stays in the router's logical order, so a pick is the unplaced kernel's,
exact score ties included. Where every token's expert in an EP group is alone
there (each rank's partial is one term), the output is bit-identical to the
unplaced kernel; otherwise the ranks' partials regroup and the output is held
to the f32 reference. Interpret mode on 16 CPU devices (EP 4 x TP 4)."""

import importlib.util
import json
import os
import pathlib
import subprocess
import sys

import numpy as np

_BASE = pathlib.Path(__file__).with_name("test_moe_tp_decode_kernel.py")


def _base():
    spec = importlib.util.spec_from_file_location("_moe_tp_decode_base", _BASE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _worker():
    import jax
    import jax.numpy as jnp
    from jax._src.pallas.mosaic import tpu_info
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.kernels.experimental.moe_tp_decode import kernel

    b = _base()
    pltpu.get_tpu_info = b._tpu7x
    tpu_info.get_tpu_info = b._tpu7x
    c, n = b.SMALL_EP, b.N_RANKS
    tp = n // c["ep"]
    mesh = Mesh(np.asarray(sorted(jax.devices("cpu"), key=lambda d: d.id)), ("d",))
    local, starts = b.ep_layout(c, n)
    keys = jax.random.split(jax.random.key(1), n + 3)
    ranks = [b._rank_weights(keys[r], dict(c, e=local)) for r in range(n)]
    gate = jax.random.normal(keys[-1], (c["e"], c["h"])) / np.sqrt(c["h"])
    # Experts 2j and 2j + 1 share a bias value (both in one EP group).
    bias = np.repeat(np.round(np.linspace(-0.04, 0.04, c["e"] // 2), 3), 2)
    bias = jnp.asarray(bias, jnp.float32)
    rank_ids = jnp.arange(n, dtype=jnp.int32).reshape(n, 1)
    e0s = jnp.asarray(starts, jnp.int32).reshape(n, 1)

    def placed_weights(slots):
        """Rank (g, m)'s local slot j holds logical inv[g * local + j]'s TP
        slice m, taken from the rank the identity layout keeps it on."""
        inv = np.argsort(slots)
        per_rank = []
        for r in range(n):
            g, m = divmod(r, tp)
            src = [inv[g * local + j] for j in range(local)]
            per_rank.append(
                [
                    jnp.stack([ranks[(e // local) * tp + m][k][e % local] for e in src])
                    if k < 4
                    else ranks[r][k]
                    for k in range(len(ranks[r]))
                ]
            )
        stacked = [jnp.concatenate(ws, axis=0) for ws in zip(*per_rank, strict=True)]
        cols = jnp.asarray(
            [inv[starts[r] : starts[r] + local] for r in range(n)], jnp.int32
        )
        return stacked, cols

    def build(fast, placed):
        def body(x, scores, bb, n_valid, rank, e0, cols, *weights):
            return kernel.moe_tp_decode(
                x,
                scores,
                bb,
                n_valid[0],
                rank.reshape(()),
                *weights,
                e0.reshape(()),
                routed_scale=b.ROUTED_SCALE,
                topk=c["topk"],
                n_ranks=n,
                fast=fast,
                expert_columns=cols.reshape(-1) if placed else None,
            )

        specs = (P("d"), P("d"), P(), P(), P("d"), P("d"), P("d")) + (P("d"),) * 8
        sm = jax.shard_map(
            body, mesh=mesh, in_specs=specs, out_specs=P("d"), check_vma=False
        )
        return jax.jit(sm), specs

    def run(x, scores, bb, rows, slots, *, fast, placed):
        stacked, cols = placed_weights(slots)
        fn, specs = build(fast, placed)
        ops = (
            jnp.tile(x, (n, 1)),
            jnp.tile(scores, (n, 1)),
            bb,
            jnp.asarray([rows], jnp.int32),
            rank_ids,
            e0s,
            cols,
            *stacked,
        )
        ops = [jax.device_put(a, NamedSharding(mesh, s)) for a, s in zip(ops, specs)]
        out, races = b._interpret(fn, *ops)
        out = np.asarray(out).reshape(n, rows, c["h"]).astype(np.float32)
        return out, races

    rows = 16
    x = jax.random.normal(keys[-3], (rows, c["h"])).astype(jnp.bfloat16)
    identity = np.arange(c["e"])
    # Twins swap places inside their group: every expert stays on its ranks.
    twins = identity ^ 1
    # One pick per group per token; the last pick ties exactly with its twin.
    rng = np.random.default_rng(3)
    scores = np.full((rows, c["e"]), 0.01, np.float32)
    for t in range(rows):
        for g, s in enumerate((0.9, 0.7, 0.5, 0.3)):
            e = g * local + rng.integers(local)
            scores[t, e] = s
            if g == c["ep"] - 1:
                scores[t, e ^ 1] = s
    tie_rows = int(
        np.sum(
            np.sort(scores + np.asarray(bias), 1)[:, -4]
            == np.sort(scores + np.asarray(bias), 1)[:, -5]
        )
    )
    res = dict(tie_rows=tie_rows)
    sj = jnp.asarray(scores)
    want, _ = run(x, sj, bias, rows, identity, fast=False, placed=False)
    for fast in (False, True):
        got, races = run(x, sj, bias, rows, twins, fast=fast, placed=True)
        ident, _ = run(x, sj, bias, rows, identity, fast=fast, placed=True)
        res[f"twins_fast{int(fast)}"] = dict(
            bitwise=bool(np.array_equal(got.view(np.uint32), want.view(np.uint32))),
            identity_bitwise=bool(
                np.array_equal(ident.view(np.uint32), want.view(np.uint32))
            ),
            races_found=races,
        )
    # Control: the router permuted instead (ties go to the lower physical id).
    inv = np.argsort(twins)
    ctl, _ = run(x, sj[:, inv], bias[inv], rows, twins, fast=False, placed=False)
    res["router_permuted_differs"] = int(np.sum(ctl != want))
    # A random cross-group placement: partials regroup, the picks do not.
    perm = rng.permutation(c["e"])
    sr = jax.nn.sigmoid(x.astype(jnp.float32) @ gate.T)
    got, races = run(x, sr, bias, rows, perm, fast=True, placed=True)
    ref = b._reference(x, sr, bias, rows, ranks, c, starts).astype(np.float64)
    g64 = got[0].astype(np.float64)
    res["random"] = dict(
        ranks_identical=bool((got.view(np.uint32) == got.view(np.uint32)[0]).all()),
        rel_l2=float(np.linalg.norm(g64 - ref) / np.linalg.norm(ref)),
        races_found=races,
    )
    unplaced, _ = run(x, sr, bias, rows, identity, fast=True, placed=False)
    u64 = unplaced[0].astype(np.float64)
    res["random"]["unplaced_rel_l2"] = float(
        np.linalg.norm(u64 - ref) / np.linalg.norm(ref)
    )
    return res


def test_placed_decode_kernel_keeps_the_unplaced_picks():
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS="--xla_force_host_platform_device_count=16",
    )
    proc = subprocess.run(
        [sys.executable, __file__],
        env=env,
        capture_output=True,
        text=True,
        timeout=1500,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stdout[-3000:] + proc.stderr[-3000:]
    got = json.loads(lines[-1][len("RESULT ") :])
    assert got["tie_rows"] == 16, got
    for fast in (0, 1):
        case = got[f"twins_fast{fast}"]
        assert case["bitwise"] and case["identity_bitwise"], case
        assert not case["races_found"], case
    assert got["router_permuted_differs"] > 0, got
    tol = _base().TOLERANCE["rel_l2"]
    assert got["random"]["ranks_identical"] and not got["random"]["races_found"], got
    assert got["random"]["rel_l2"] <= tol, got


if __name__ == "__main__":
    from jax.experimental.pallas import tpu as pltpu

    # Around everything: pallas_call reads the mode when it is built.
    with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams(detect_races=True)):
        print("RESULT " + json.dumps(_worker()))
