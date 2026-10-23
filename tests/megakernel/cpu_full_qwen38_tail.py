"""Full-geometry Qwen3.8 tail segment in CPU interpret mode (needs ~100+ GB RAM)."""
import dataclasses, sys, time, jax, numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from vllm_torchtpu.megakernel import kernel
from vllm_torchtpu.megakernel.blocks.hc import RESIDUAL_OUT
from vllm_torchtpu.megakernel.exec_spec import ExecSpec, Tiles
from vllm_torchtpu.megakernel.models import qwen38
from vllm_torchtpu.megakernel.models.qwen38_inputs import pack_segment, random_segment_inputs
from vllm_torchtpu.megakernel.models.qwen38_pack import shard_step_inputs
from vllm_torchtpu.megakernel.models.qwen38_reference import segment_reference
g = qwen38.FULL; R, L, T, B = 8, 1, 4, 8
# Interpret mode allocates one typed ring per weight kind, hence the budget.
tiles = Tiles(ebank=2 << 20, nbe=12, nbd=10, vmem_mib=8192).with_blocks({"moe": {"pipe_ep": True, "moe_unroll": True}})
ex = ExecSpec(R=R, B=B, T=T, ctx=128, interpret=True, tiles=tiles)
spec = qwen38.segment_spec(g, R, L=L)
t0 = time.time()
inp = random_segment_inputs(jax.random.PRNGKey(B), g, ex, L, gain=0.25, router_gain=30.0)
ref, _ = segment_reference(g, ex, inp.x, inp.hc, inp.w_l, inp.st_l, inp.attn, inp.o_proj, L=L)
print("ref done", time.time() - t0, flush=True)
w, st, attn = pack_segment(spec, g, ex, inp)
mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
pos, hid, w, st = shard_step_inputs(mesh, ex, inp.pos, inp.x, w, st)
a = jax.device_put(kernel.fold(attn), NamedSharding(mesh, P("tp")))
step, _ = kernel.make_step(mesh, spec, ex)
out = step(pos, hid, w, st, a)[1][RESIDUAL_OUT]
res = np.asarray(out.reshape(R, ex.M, -1)[:, :B], np.float32)
refn = np.asarray(ref, np.float32)
d = np.abs(res[0] - refn).max(1) / np.abs(refn).mean()
print("kernel done", time.time() - t0, "row err", np.round(d, 3), flush=True)
import os
out = os.environ.get("CDK_OUTPUT_DIR", "/tmp")
np.save(os.path.join(out, "res.npy"), res); np.save(os.path.join(out, "ref.npy"), refn)
for k in ("x", "attn", "o_proj"):
    np.save(os.path.join(out, k + ".npy"), np.asarray(getattr(inp, k), np.float32))
