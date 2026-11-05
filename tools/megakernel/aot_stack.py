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
"""Checked AOT gate for the Kimi-K3.1 FP8 decode stack (kimi_decoder_stack) as DecodeOps
builds it.

``--aot`` compiles for TPU7x on all 32 devices of the tpu7x:2x2x4 topology (every rank
index stays a runtime value and peer device ids are valid, as on the slice); without it
the stack is traced on one CPU device (the VMEM plan only). The shard_map body returns
every decode_paged output (a body without outputs is dead code, and XLA would compile an
empty program), and the compiled program must contain the kimi_decoder_stack custom
call: the result line reports its tpu_custom_call count and HLO size. Calibrated on
2026-11-04 against five device compiles: it reproduced both VMEM failures (the B1x8
verify with the BF16 query all-gather, 64.34M of 63.94M as on the slice; B16x4 with four
query/partial-output buffers in flight, 64.11M against 64.35M) and the three passes.

Shapes follow the serving builders: SEQUENCE_ROWS == ROWS is the single-request verify
(_build_verify: token scatter at 8 rows, the state row), 1 < SEQUENCE_ROWS < ROWS the
batched verify (_build_verify_batched), SEQUENCE_ROWS 1 plain decode (decode_options
only); verifies defer the KDA commit and capture the DSpark aux layers. Weights come
from load.selected_local_shapes() (so the KernelConfig's expert-weight layout) with
K3.1's packed MLA families; PAGES is the block-table width (local pages a request: 64 at
1M context with DCP32); 16 KV blocks a rank. Attention DP (--set attention_dp=2 --set
dcp_group=16) holds each group's attention shards as DecodeOps does
(decode_megakernel.ATTENTION_DP_WEIGHTS) with attention_dp x (rows + 1) KV slots, and
its single-request verify commits the KDA state in place.

Usage, from the tree under test (its src and root on PYTHONPATH), on a compute host as
an "-aot" step under nice -n 19 with JAX_PLATFORMS=cpu (no device is opened):

    python -u tools/megakernel/aot_stack.py ROWS SEQUENCE_ROWS PAGES [--aot] [--owner]
        [--set FIELD=VALUE ...]

--aot reserves a 300 KiB margin (vmem_pad_kb): at 64 rows AOT reads about 0.24M under
the slice (D6 passed unpadded, then failed on the slice at 64.16M of 63.94M), so a bare
pass near the limit is not a pass; --set vmem_pad_kb=0 shows the unpadded result.
--owner: a non-identity page-owner table; --set: KernelConfig fields (Python literals),
one config a process. Exit status 0 when the stack compiled (or traced) with the kernel
present, 1 otherwise.
"""

import ast
import os
import sys
import types

DEVICE_MARGIN_KB = 300
aot = "--aot" in sys.argv
if not aot:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALLOW_MULTIPLE_LIBTPU_LOAD", "1")
rows, sequence_rows, pages = (int(a) for a in sys.argv[1:4])
sets = {}
for i, arg in enumerate(sys.argv):
    if arg == "--set":
        key, value = sys.argv[i + 1].split("=", 1)
        try:
            sets[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            sets[key] = value
from vllm_torchtpu.megakernel.kimi_k3 import config  # noqa: E402

if aot:
    sets.setdefault("vmem_pad_kb", DEVICE_MARGIN_KB)
config.install(config.KernelConfig(vmem_report=True, **sets))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

sys.path.insert(0, ".")
stub = types.ModuleType("pytest")


class _Marks:
    def __getattr__(self, name):
        return lambda *args, **kwargs: (lambda function: function)


stub.mark = _Marks()
stub.fixture = (
    lambda *args, **kwargs: args[0]
    if args and callable(args[0])
    else (lambda function: function)
)
stub.param = lambda *args, **kwargs: args
sys.modules["pytest"] = stub
RANKS = 32 if aot else 1
if aot:
    from jax.experimental import topologies
    from jax.sharding import NamedSharding

    devices = np.asarray(
        topologies.get_topology_desc(
            platform="tpu", topology_name="tpu7x:2x2x4"
        ).devices
    ).reshape(-1)
else:
    devices = np.asarray(jax.devices()[:1])
from tests.megakernel.kimi_k3 import test_dsa  # noqa: E402
from vllm_torchtpu.megakernel.kimi_k3 import layout  # noqa: E402
from vllm_torchtpu.megakernel.kimi_k3.decode_ops import decode_options  # noqa: E402
from vllm_torchtpu.megakernel.kimi_k3.kernels import (  # noqa: E402
    dsa_cache,
    load,
    paged,
)

shapes = load.selected_local_shapes()
shapes["m_ka"] = (*shapes["m_ka"][:-1], 640 + 128)
shapes["m_qb"] = (*shapes["m_qb"][:-1], 768 + 128)
shapes["m_knorm"] = (*shapes["m_knorm"][:-1], 512 + 256 + shapes["m_ka"][-2])
attention_dp = config.active().attention_dp
if attention_dp > 1:
    from vllm_torchtpu.megakernel.kimi_k3.kernels.decode_megakernel import (
        ATTENTION_DP_WEIGHTS,
    )

    for name, (_, axis) in ATTENTION_DP_WEIGHTS.items():
        held = list(shapes[name])
        if axis is None:
            held.insert(1, attention_dp)
        else:
            held[1 + axis] *= attention_dp
        shapes[name] = tuple(held)
    shapes["m_knorm"] = (
        *shapes["m_knorm"][:-1],
        768 + attention_dp * shapes["m_ka"][-2],
    )
names = tuple(sorted(shapes))
cache = layout.cache_layout("fp8")
slots = 2 * rows + 1 if attention_dp == 1 else attention_dp * (rows + 1)
count, blocks = len(names), 16
pools = len(layout.MLA)
sharded = (
    *[
        ((RANKS, *shapes[n]), test_dsa.FAMILY_DTYPES.get(n, jnp.bfloat16))
        for n in names
    ],
    ((RANKS, slots, len(layout.KDA), 3, 3, layout.CONV_HEAD_ROWS, 128), jnp.bfloat16),
    ((RANKS, slots, len(layout.KDA), 3, 128, 128), jnp.float32),
    *[
        ((RANKS * blocks, *cache.block_shape), jnp.dtype(cache.dtype))
        for _ in layout.MLA
    ],
)
replicated = (
    ((rows,), jnp.int32),
    ((rows,), jnp.int32),
    ((rows, pages), jnp.int32),
    ((rows,), jnp.int32),
    ((1,), jnp.int32),
)
options = decode_options(rows)
if sequence_rows > 1:
    options["defer_kda_commit"] = attention_dp == 1 or sequence_rows < rows
    options.update(sequence_rows=sequence_rows, aux_hidden_layers=(76, 80, 84, 88, 92))
    if sequence_rows == rows:
        options.update(
            moe_routed_reduce_scatter=True,
            moe_token_scatter_output=True,
            attention_token_scatter=True,
            sequence_parallel_pre_attention=True,
        )


def body(*args):
    weights = {n: args[i][0] for i, n in enumerate(names)}
    tail = args[count + 2 + pools :]
    outputs = paged.decode_paged(
        weights,
        args[count][0],
        args[count + 1][0],
        args[count + 2 : count + 2 + pools],
        *tail[:4],
        batch_size=rows,
        layers=93,
        dcp_pool_map=tuple(range(pools)),
        dcp_owner_table=tuple(5 * i % 32 for i in range(32))
        if "--owner" in sys.argv
        else None,
        state_row=tail[4][0] if sequence_rows == rows else None,
        index_cache=jnp.zeros(
            (blocks, 24, *dsa_cache.index_block_shape(cache.tokens_per_block)),
            jnp.uint8,
        ),
        **options,
    )
    return tuple(
        leaf[None] for leaf in jax.tree.leaves(outputs)
    )  # kept: a body without outputs is dead code


mesh = Mesh(devices, ("tp",))
specs = (P("tp"),) * len(sharded) + (P(),) * len(replicated)
step = jax.shard_map(
    body, mesh=mesh, in_specs=specs, out_specs=P("tp"), check_vma=False
)
owner = "--owner" in sys.argv
tag = (
    f"{rows} rows, {sequence_rows}-row sequences, {pages} pages, {RANKS} ranks, "
    f"owner table {owner}, sets {sets}"
)
try:
    if aot:
        compiled = (
            jax.jit(step)
            .lower(
                *[
                    jax.ShapeDtypeStruct(s, d, sharding=NamedSharding(mesh, spec))
                    for (s, d), spec in zip((*sharded, *replicated), specs)
                ]
            )
            .compile()
        )
        text = compiled.as_text()
        calls = text.count("tpu_custom_call")
        if "kimi_decoder_stack" not in text or not calls:
            raise RuntimeError(
                "the compiled program has no kimi_decoder_stack custom call "
                f"({calls} tpu_custom_call, "
                f"{len(text)} chars): the kernel was eliminated"
            )
        print(
            f"STACK {tag}: AOT compiled for TPU7x, margin {sets['vmem_pad_kb']} KiB, "
            "kimi_decoder_stack lowered "
            f"({calls} tpu_custom_call, "
            f"{len(text)} chars of HLO)",
            flush=True,
        )
    else:
        jax.eval_shape(
            step, *[jax.ShapeDtypeStruct(s, d) for s, d in (*sharded, *replicated)]
        )
        print(f"STACK {tag}: traced", flush=True)
except Exception as error:  # report
    print(f"STACK {tag}: FAILED {type(error).__name__}: {str(error)[:900]}", flush=True)
    sys.exit(1)
