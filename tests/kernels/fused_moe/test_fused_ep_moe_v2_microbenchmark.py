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
"""Run fused_ep_moe_v2 using the input shapes/dtypes captured in real
inference workload.
"""

import glob
import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.fused_moe.v2 import (
    AXIS,
    WeightFormat,
    fused_ep_moe_v2,
    host,
)

jax.config.parse_flags_with_absl()

# The Pallas call's name prefix; the full name adds the weight format, the
# activation and the g/c/nb geometry.
KERNEL_NAME = "fused_ep_moe_v2"

FP4 = jnp.float4_e2m1fn
# Contraction rows one weight scale covers, which is the middle axis of the
# scale tables the profiles carry.
QB4 = host.QB4
# The serving adapter's _TILE_M, which every build name repeats as c128.
CAPACITY = 128
# Kernel runs captured by the profiler after warmup.
PROFILE_STEPS = 10


def _operands(case, mesh):
    e_total = case["e_total"]
    hidden = case["hidden"]
    inter = case["inter"]
    tokens = case["tokens"]
    ep = case["ep"]
    keys = jax.random.split(jax.random.key(0), 7)

    sharded_ep = NamedSharding(mesh, P(AXIS))
    replicated = NamedSharding(mesh, P())

    # Generated under jit with out_shardings so each device materializes only
    # its own shard; the full float32 weights do not fit on one device.
    @partial(jax.jit, out_shardings=(sharded_ep,) * 6 + (replicated, sharded_ep))
    def make():
        w1 = (jax.random.normal(keys[0], (e_total, hidden, 2 * inter)) * 2).astype(FP4)
        w2 = (jax.random.normal(keys[1], (e_total, inter, hidden)) * 2).astype(FP4)
        w1_scale = jax.random.uniform(keys[2], (e_total, hidden // QB4, 2 * inter)) / 64
        w2_scale = jax.random.uniform(keys[3], (e_total, inter // QB4, hidden)) / 64
        x = (jax.random.normal(keys[4], (tokens, hidden)) / 10).astype(jnp.bfloat16)
        gating = jax.random.normal(keys[5], (tokens, e_total))
        score_bias = jax.random.normal(keys[6], (e_total,)) / 10
        # This rank's mesh index as an explicit device tensor, matching
        # serving's `_RANK_BUFFER`. Prevents XLA from generating a dynamic
        # slice on a constant partition-id.
        rank = jnp.arange(ep, dtype=jnp.int32).reshape(ep, 1)
        return w1, w2, w1_scale, w2_scale, x, gating, score_bias, rank

    return make()


def _kernel_times_ms(profile_dir):
    """Device durations of every KERNEL_NAME run in this process's profile."""
    times = []
    for path in glob.glob(f"{profile_dir}/**/*.xplane.pb", recursive=True):
        for plane in jax.profiler.ProfileData.from_file(path).planes:
            if not plane.name.startswith("/device:TPU:") or "SparseCore" in plane.name:
                continue
            for line in plane.lines:
                if line.name != "XLA Ops":
                    continue
                times += [
                    event.duration_ns / 1e6
                    for event in line.events
                    if KERNEL_NAME in event.name
                ]
    return times


class FusedEpMoeV2MicrobenchmarkTest(parameterized.TestCase):
    @parameterized.named_parameters(
        dict(
            testcase_name="ep16_h7168_e384_t16384",
            name="ep16-h7168-e384-t16384",
            ep=16,
            tokens=16384,
            hidden=7168,
            inter=3072,
            e_total=384,
            topk=6,
        ),
        dict(
            testcase_name="ep16_h7168_e384_t1024",
            name="ep16-h7168-e384-t1024",
            ep=16,
            tokens=1024,
            hidden=7168,
            inter=3072,
            e_total=384,
            topk=6,
        ),
        dict(
            testcase_name="ep8_h4096_e256_t8192",
            name="ep8-h4096-e256-t8192",
            ep=8,
            tokens=8192,
            hidden=4096,
            inter=2048,
            e_total=256,
            topk=6,
        ),
        dict(
            testcase_name="ep8_h4096_e256_t512",
            name="ep8-h4096-e256-t512",
            ep=8,
            tokens=512,
            hidden=4096,
            inter=2048,
            e_total=256,
            topk=6,
        ),
    )
    def test_runs_the_served_shape(self, **case):
        """The shape builds, runs and returns the rows it was given."""
        if jax.device_count() < case["ep"]:
            self.skipTest(
                f"{case['name']} is a {case['ep']}-way expert-parallel collective "
                f"and this host reports {jax.device_count()} JAX devices"
            )
        mesh = Mesh(np.asarray(jax.devices()[: case["ep"]]), axis_names=(AXIS,))
        w1, w2, w1_scale, w2_scale, x, gating, score_bias, rank = _operands(case, mesh)

        @jax.jit
        def run_kernel(x, w1, w2, w1_scale, w2_scale, gating, score_bias, rank):
            return fused_ep_moe_v2(
                x,
                w1,
                w2,
                w1_scale,
                w2_scale,
                gating,
                rank=rank,
                topk=case["topk"],
                renormalize=True,
                mesh=mesh,
                capacity=CAPACITY,
                weight_format=WeightFormat.FP4,
                act_fn="silu_and_mul_with_clamp",
                scoring_fn="sigmoid",
                score_bias=score_bias,
            )

        # Warmup
        out = run_kernel(x, w1, w2, w1_scale, w2_scale, gating, score_bias, rank)
        out.block_until_ready()

        profile_dir = os.path.join(
            os.environ.get("PROFILE_DIR") or self.create_tempdir().full_path,
            case["name"],
        )
        with jax.profiler.trace(profile_dir):
            for _ in range(PROFILE_STEPS):
                out = run_kernel(
                    x, w1, w2, w1_scale, w2_scale, gating, score_bias, rank
                )
            out.block_until_ready()

        kernel_ms = _kernel_times_ms(profile_dir)
        self.assertNotEmpty(kernel_ms, f"no {KERNEL_NAME} events in {profile_dir}")
        print(
            f"{case['name']}: process {jax.process_index()} {KERNEL_NAME} "
            f"device time over {len(kernel_ms)} kernel runs on "
            f"{jax.local_device_count()} local devices: "
            f"median {np.median(kernel_ms):.3f} ms, min {min(kernel_ms):.3f} ms, "
            f"max {max(kernel_ms):.3f} ms; profile: {profile_dir}",
            flush=True,
        )


if __name__ == "__main__":
    absltest.main()
