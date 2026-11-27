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
"""Selection drift of an indexer arm (default TPU_MINIMAX_M3_INDEXER_V2=1,
fp8 queries) against the bf16 baseline on synthetic served c48 shapes.

Per case and layer (each layer its own random queries and caches), every
DCP rank's `msa_write_and_candidates_dcp` runs in both arms and the ranks'
candidates are merged (`merge_candidates`) into the global top-k blocks.
Reported: the fraction of baseline-selected blocks also selected by the arm
(per layer, min, mean), and a proxy attention-output rel L2: random bf16
K/V and attention queries, softmax attention over each sampled token's
selected blocks (causal), arm vs baseline. Synthetic random data, not model
activations.

    PYTHONPATH=src python scripts/bench_indexer_v2_overlap.py --smoke
"""

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np

from bench_indexer_prefill_opt import PAGE, D, build_caches, build_step
from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", default="1x8192@30000,1x8192@230000,1x8192@400000")
    p.add_argument("--base-env", default="", help="ENV=VAL;... for both arms")
    p.add_argument("--arm-env", default="TPU_MINIMAX_M3_INDEXER_V2=1")
    p.add_argument("--dcp", type=int, default=4)
    p.add_argument("--max-seqs", type=int, default=48)
    p.add_argument("--max-len", type=int, default=1 << 20)
    p.add_argument("--topk", type=int, default=16)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--att-tokens", type=int, default=512)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def _set(spec):
    keys = []
    for item in filter(None, spec.split(";")):
        k, v = item.split("=", 1)
        os.environ[k] = v
        keys.append(k)
    return keys


def candidates_fn(args, interpret):
    def f(k, v, iq, ik, kv_c, idx, sc, seq_lens, tables, cu, dist, rank):
        return msa_dcp.msa_write_and_candidates_dcp(
            k, v, iq, ik, kv_c, idx, seq_lens, tables, cu, dist,
            topk=args.topk, init_blocks=0, local_blocks=1, rank=rank,
            dcp=args.dcp, scale_cache=sc, interpret=interpret,
        )[-1]  # fmt: skip

    return jax.jit(f)


def global_selection(cands, g, topk):
    """``[dcp, T, G * 2 * topk]`` -> ``[T, G, topk]`` global blocks (-1 pad)."""
    c = cands.reshape(cands.shape[0], cands.shape[1], g, 2 * topk)
    return jnp.stack(
        [msa_dcp.merge_candidates(c[:, :, h], topk) for h in range(g)], axis=1
    )


@jax.jit
def proxy_attention(sel, q, kk, vv, pos):
    """Softmax attention of ``q [N, G, D]`` over the selected DCP blocks
    ``sel [N, G, K]`` of ``kk, vv [L, G, D]`` (keys after `pos` masked)."""
    span = sel.shape[-1]

    def one(x):
        s_, q_, p_ = x  # [G, K], [G, D], []
        tok = s_[:, :, None] * BLOCK + jnp.arange(BLOCK)[None, None, :]
        tok = tok.reshape(s_.shape[0], span * BLOCK)  # [G, K * B]
        ok = (s_[:, :, None] >= 0).repeat(BLOCK, 2).reshape(tok.shape) & (tok <= p_)
        t = jnp.clip(tok, 0, kk.shape[0] - 1)
        g = jnp.arange(s_.shape[0])[:, None]
        kg, vg = kk[t, g], vv[t, g]  # [G, K * B, D]
        lg = jnp.einsum("gd,gnd->gn", q_.astype(jnp.float32), kg.astype(jnp.float32))
        lg = jnp.where(ok, lg / np.sqrt(D), -jnp.inf)
        w = jax.nn.softmax(lg, axis=-1)
        return jnp.einsum("gn,gnd->gd", w, vg.astype(jnp.float32))

    return jax.lax.map(one, (sel, q, pos))


BLOCK = 0


def main():
    global BLOCK
    args = _parse()
    interpret = args.smoke
    if args.smoke:
        args.layers, args.max_len, args.max_seqs, args.att_tokens = 1, 1 << 15, 4, 16
        args.cases = "1x1024@3000"
    BLOCK = args.dcp * PAGE
    g = max(1, args.dcp // 2)
    _set(args.base_env)
    print(f"devices {jax.devices()[:1]} base_env {args.base_env!r} arm {args.arm_env!r}",
          flush=True)  # fmt: skip
    for case in args.cases.split(","):
        overlaps, rels = [], []
        for layer in range(args.layers):
            rng = np.random.default_rng(1000 * layer + sum(map(ord, case)))
            meta, dist, num_pages, T = build_step(
                case, args.dcp, args.max_seqs, args.max_len, rng
            )

            def bf16(*shape):
                return jnp.asarray(rng.standard_normal(shape), jnp.bfloat16)

            k, v, iq, ik = bf16(T, g, D), bf16(T, g, D), bf16(T, g, D), bf16(T, D)
            caches = [build_caches(num_pages, g, 7 + 31 * layer + r) for r in range(args.dcp)]
            sels = {}
            for arm in ("base", "arm"):
                keys = _set(args.arm_env) if arm == "arm" else []
                fn = candidates_fn(args, interpret)
                per_rank = [
                    fn(k, v, iq, ik, *caches[r], *meta, dist, jnp.asarray([r], jnp.int32))
                    for r in range(args.dcp)
                ]
                sels[arm] = np.asarray(
                    global_selection(jnp.stack(per_rank), g, args.topk)
                )
                for key in keys:
                    os.environ.pop(key, None)
                _set(args.base_env)
            b, a = sels["base"], sels["arm"]
            real = b >= 0
            hit = (b[..., :, None] == a[..., None, :]).any(-1) & real
            overlaps.append(hit.sum() / max(real.sum(), 1))
            # Proxy attention on sampled tokens of the first request.
            seq_len = int(meta[0][0])
            q_len = int(meta[2][1] - meta[2][0])
            rows = np.linspace(0, q_len - 1, args.att_tokens).astype(np.int32)
            pos = jnp.asarray(seq_len - q_len + rows)
            kk, vv = bf16(seq_len, g, D), bf16(seq_len, g, D)
            qa = bf16(len(rows), g, D) * 4
            ob = proxy_attention(jnp.asarray(b[rows]), qa, kk, vv, pos)
            oa = proxy_attention(jnp.asarray(a[rows]), qa, kk, vv, pos)
            rels.append(float(jnp.linalg.norm(oa - ob) / jnp.linalg.norm(ob)))
            print(f"LAYER {case} {layer} overlap {overlaps[-1]:.5f} attn_rel_l2 {rels[-1]:.3e} "
                  f"real {int(real.sum())}", flush=True)  # fmt: skip
        print(f"OVERLAP {case} min {min(overlaps):.5f} mean {np.mean(overlaps):.5f} "
              f"per_layer {[round(float(x), 5) for x in overlaps]} attn_rel_l2 mean "
              f"{np.mean(rels):.3e} max {max(rels):.3e}", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
