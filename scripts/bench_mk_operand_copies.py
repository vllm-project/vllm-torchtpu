"""Which torch inputs of a compiled program does torch_tpu re-materialize per call?

The served MK program runs ~23 eager `tt_jit_execute_as_strided` programs before
every call, copying specific small operands (f32 'smalls' / 'es', a bf16 [W, 128]
HC weight) but never the big bf16 / fp8 weights. Each arm here builds a tiny
module holding one kind of buffer, feeds it with an input `x` to a
`sharded_jax_callable` op inside `torch.compile(backend="tpu")`, and counts the
as_strided programs per call from the PJRT log of a fresh subprocess. Optional
`mutate` arms also write a second buffer in place (as the MK writes its GDN
pool), to test AOTAutograd's synthetic-base regeneration. Prints OPC lines.
"""

import json
import os
import subprocess
import sys

ARMS = {
    "none": dict(),
    "f32_64x3840": dict(dtype="float32", shape=(64, 3840)),
    "bf16_64x3840": dict(dtype="bfloat16", shape=(64, 3840)),
    "f32_1608x128": dict(dtype="float32", shape=(1608, 128)),
    "bf16_10240x128": dict(dtype="bfloat16", shape=(10240, 128)),
    "bf16_10240x384": dict(dtype="bfloat16", shape=(10240, 384)),
    "f32_slice_assigned": dict(dtype="float32", shape=(4, 1, 51456), build="slice"),
    "f32_as_i32": dict(dtype="float32", shape=(1608, 128), build="i32"),
    "f32_param": dict(dtype="float32", shape=(64, 3840), build="param"),
    "f32_64x3840_mutpool": dict(dtype="float32", shape=(64, 3840), mutate=True),
    "bf16_10240x384_mutpool": dict(dtype="bfloat16", shape=(10240, 384), mutate=True),
}

CHILD = r"""
import json, sys
import jax, jax.numpy as jnp, numpy as np, torch
from jax.sharding import Mesh, PartitionSpec as P
from vllm_torchtpu.megakernel_runtime.exported_bridge import (
    make_typed_step_wrapper, sharded_jax_callable)

cfg = json.loads(sys.argv[1])
dev = torch.device("tpu")
mesh = Mesh(np.asarray(jax.devices()[:1]), ("tp",))
dt = getattr(torch, cfg.get("dtype", "float32"))
has_w = "shape" in cfg
mut = cfg.get("mutate", False)

def step(x, *rest):
    out = x
    if has_w:
        w = rest[0]
        if cfg.get("build") == "i32":
            w = jax.lax.bitcast_convert_type(w, jnp.float32)
        out = out + jnp.sum(w.astype(jnp.float32)).astype(x.dtype)
    if mut:
        pool = rest[-1]
        return out, pool + 1
    return (out,)

n_in = 1 + int(has_w) + int(mut)
n_out = 1 + int(mut)
typed = make_typed_step_wrapper("opc_step", step, n_in, n_out)
call = sharded_jax_callable(
    "opc::step", typed, mesh=mesh, input_partition_specs=(P(),) * n_in,
    output_partition_specs=(P(),) * n_out, donate_argnums=((n_in - 1,) if mut else ()))

@torch.library.custom_op("opc::run", mutates_args=())
def run(x: torch.Tensor, ws: list[torch.Tensor]) -> list[torch.Tensor]:
    return list(call(x, *ws))

@run.register_fake
def _(x, ws):
    out = [torch.empty_like(x)]
    if mut:
        out.append(torch.empty_like(ws[-1]))
    return out

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        if has_w:
            shape = tuple(cfg["shape"])
            b = cfg.get("build")
            if b == "slice":
                t = torch.zeros(shape, dtype=dt, device=dev)
                t[:, 0, :1000] = 1.0
            elif b == "i32":
                t = torch.ones(shape, dtype=torch.float32, device=dev).view(torch.int32)
            else:
                t = torch.ones(shape, dtype=dt, device=dev)
            if b == "param":
                self.w = torch.nn.Parameter(t, requires_grad=False)
            else:
                self.register_buffer("w", t, persistent=False)
        if mut:
            self.register_buffer("pool", torch.zeros((256, 128), dtype=torch.float32, device=dev),
                                 persistent=False)

    def forward(self, x):
        ws = ([self.w] if has_w else []) + ([self.pool] if mut else [])
        out = run(x, ws)
        if mut:
            self.pool.copy_(out[1])
        return out[0]

m = M()
f = torch.compile(m, backend="tpu", fullgraph=True, dynamic=False)
x = torch.ones((8, 128), dtype=torch.bfloat16, device=dev)
for _ in range(3):
    y = f(x)
torch.tpu.synchronize() if hasattr(torch, "tpu") else None
print("MARK_BEGIN", flush=True)
sys.stderr.write("MARK_BEGIN\n"); sys.stderr.flush()
for _ in range(20):
    y = f(x)
_ = y.cpu()
sys.stderr.write("MARK_END\n"); sys.stderr.flush()
print("MARK_END", flush=True)
"""


def test_operand_copies():
    for name, cfg in ARMS.items():
        env = dict(os.environ)
        env["TPU_VMODULE"] = "tpu_pjrt_client=1"
        env["TPU_STDERR_LOG_LEVEL"] = "0"
        env["TPU_MIN_LOG_LEVEL"] = "0"
        r = subprocess.run([sys.executable, "-c", CHILD, json.dumps(cfg)], env=env,
                           capture_output=True, text=True, timeout=900)
        err = r.stderr
        seg = err.split("MARK_BEGIN", 1)[-1].split("MARK_END", 1)[0] if "MARK_BEGIN" in err else ""
        n_as = seg.count("Start scheduling program tt_jit_execute_as_strided")
        n_prog = seg.count("Start scheduling program")
        import collections
        import re

        progs = collections.Counter(re.findall(r"Start scheduling program (\S+)", seg))
        row = dict(arm=name, as_strided_per_call=n_as / 20, programs_per_call=n_prog / 20,
                   rc=r.returncode, top=progs.most_common(5))
        if r.returncode or not seg:
            row["error"] = " | ".join(l for l in err.splitlines() if "Error" in l)[-600:]
        print("OPC " + json.dumps(row), flush=True)
