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
"""Checked AOT gate for the fused MLA DSpark draft (dspark_mla_fused_draft) as the
decoder compiles it: REQUESTS requests of ROWS verify rows (draft block min(block,
ROWS - 1)), PAGES block-table entries a request, the serving FP8 pools (512-token
pages), at attention DP 1 (DCP32) or 2 (DP2 x DCP16: --set attention_dp=2 --set
dcp_group=16, held shards' heads), on the 32 devices of the tpu7x:2x2x4 topology. The
program returns the kernel's outputs and the compiled HLO must contain the
dspark_mla_fused_draft custom call (its tpu_custom_call count and HLO size are
printed). Weight shapes come from the presharded draft's index.json.

Usage, from the tree under test (its src and root on PYTHONPATH), on a compute host as
an "-aot" step under nice -n 19 with JAX_PLATFORMS=cpu (no device is opened):

    python -u tools/megakernel/aot_draft.py DRAFT SHARDS REQUESTS ROWS PAGES
        [--set FIELD=VALUE ...]

DRAFT is the draft checkpoint, SHARDS its presharded directory. Exit status 0 when the
draft compiled with the kernel present, 1 otherwise.
"""

import ast
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("ALLOW_MULTIPLE_LIBTPU_LOAD", "1")
draft, shards_dir = sys.argv[1], Path(sys.argv[2])
requests, rows, pages = (int(a) for a in sys.argv[3:6])
sets = {}
for i, arg in enumerate(sys.argv):
    if arg == "--set":
        key, value = sys.argv[i + 1].split("=", 1)
        try:
            sets[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            sets[key] = value
from vllm_torchtpu.megakernel.kimi_k3 import config  # noqa: E402

config.install(config.KernelConfig(**sets))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import ml_dtypes  # noqa: E402
import numpy as np  # noqa: E402
from jax.experimental import topologies  # noqa: E402
from jax.sharding import Mesh  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.megakernel.kimi_k3 import layout  # noqa: E402
from vllm_torchtpu.megakernel.kimi_k3.kernels import (  # noqa: E402
    dspark_mla,
    dspark_mla_kernel,
)

warnings.filterwarnings("ignore")
dp = config.active().attention_dp
topology = topologies.get_topology_desc(platform="tpu", topology_name="tpu7x:2x2x4")
c = dspark_mla.MlaDraftConfig.from_checkpoint(Path(draft))
index = json.loads((shards_dir / "index.json").read_text())
dtype = {"bfloat16": jnp.bfloat16, "float32": jnp.float32, "int32": jnp.int32}
shapes = {n: (list(s["shape"]), dtype[s["dtype"]]) for n, s in index["arrays"].items()}
for name, axis in getattr(dspark_mla, "HELD_AXES", {}).items():
    shapes[name][0][axis] *= dp
qkv = np.zeros(shapes.pop("qkv_a")[0], ml_dtypes.bfloat16)
qkv_tile, context_tile = dspark_mla.qkv_tiles(qkv, 0)
del qkv
shapes["qkv_tile"] = (list(qkv_tile.shape), jnp.bfloat16)
shapes["context_tile"] = (list(context_tile.shape), jnp.bfloat16)
shapes["rope"] = (list(dspark_mla.rope_table(c, 1 << 20).shape), jnp.float32)
shapes["mask_embed"] = ([1, c.hidden], jnp.bfloat16)
shapes["lm_head"] = ([c.hidden, c.vocab // 32], jnp.bfloat16)
shapes["embed"] = ([c.vocab // 32, c.hidden], jnp.bfloat16)
weights = {
    n: jax.ShapeDtypeStruct(tuple(shapes[n][0]), shapes[n][1])
    for n in dspark_mla_kernel.WEIGHT_NAMES
}
T = requests * rows
cache = layout.cache_layout("fp8")  # the serving FP8 pools: 512-token pages
pools = tuple(
    jax.ShapeDtypeStruct(
        (1 + requests * pages, *cache.block_shape), jnp.dtype(cache.dtype)
    )
    for _ in range(c.layers)
)
operands = (
    weights,
    pools,
    jax.ShapeDtypeStruct((len(c.aux_layers), T, c.hidden), jnp.bfloat16),
    jax.ShapeDtypeStruct((requests, rows, c.vocab // 32), jnp.float32),
    jax.ShapeDtypeStruct((requests, 128), jnp.int32),
    *(jax.ShapeDtypeStruct((requests,), jnp.int32) for _ in range(3)),
    jax.ShapeDtypeStruct((requests, pages), jnp.int32),
)
block = min(c.block, rows - 1)
fused = dspark_mla_kernel.fused_draft_local(
    c,
    requests=requests,
    pages=pages,
    block=block,
    rows=rows,
    **({"attention_dp": dp} if dp > 1 else {}),
)
mesh = Mesh(np.array(topology.devices[:32]), ("tp",))
call = jax.shard_map(fused, mesh=mesh, in_specs=P(), out_specs=P(), check_vma=False)
tag = (
    f"draft dp {dp}, {requests} requests x {rows} rows (block {block}), {pages} pages, "
    f"sets {sets}"
)
try:
    with jax.set_mesh(mesh):
        text = jax.jit(call).lower(*operands).compile().as_text()
    calls = text.count("tpu_custom_call")
    if "dspark_mla_fused_draft" not in text or not calls:
        raise RuntimeError(
            "the compiled program has no dspark_mla_fused_draft custom call "
            f"({calls} tpu_custom_call, {len(text)} chars): the kernel was eliminated"
        )
    print(
        f"DRAFT {tag}: AOT compiled for TPU7x, dspark_mla_fused_draft lowered "
        f"({calls} tpu_custom_call, {len(text)} chars of HLO)",
        flush=True,
    )
except Exception as error:  # report
    print(f"DRAFT {tag}: FAILED {type(error).__name__}: {str(error)[:900]}", flush=True)
    sys.exit(1)
