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
"""Server-free 8-rank decode replay of CED-decoder SparseMLA + SWA.

Decode shape of one ``L20..L39`` attention call: ``C = --num-seqs`` requests,
one query token each (``query_start_loc = arange(C + 1)``), every request
attending its full 128-row sliding window plus ``index_topk = 512`` sorted
(pre-compacted) rows of a C1 FP8 cache of ``--kv-len`` rows behind a
``[C, kv_len / page_size]`` page table.

Modes:
  - ``head_parallel``: the serving path for ``tokens < 128 * tp`` today; each
    rank attends all ``C`` tokens with its ``64 / tp`` heads.
  - ``token_parallel``: ``all_to_all`` q to ``[C / tp, 64, 512]``, attend this
    rank's ``C / tp`` requests with all 64 heads, ``all_to_all`` back
    (`fp8_attention._token_parallel_body` without its ``tokens >= 1024`` gate).

Subops (``token_parallel`` subops other than ``all`` time this rank's local
``C / tp`` tokens with no collective; ``a2a_only`` times the two collectives):
  ``all``, ``swa_only``, ``sc_gather_only``, ``mla_tc_only``, ``a2a_only``.

Each timed call runs ``--reps`` dependent repetitions inside one compiled
``fori_loop`` so host dispatch is amortized; reported samples are per rep.
``--verify`` checks ``token_parallel`` vs ``head_parallel`` ``all`` outputs.
"""

import argparse
import functools
import pathlib
import statistics
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import common  # noqa: E402
from bench_sparse_mla_swa import (  # noqa: E402
    _mla_tc_only_jax,
    _pregather_all_chunks_jax,
    _sc_gather_only_jax,
)

SUBOPS = ("all", "swa_only", "sc_gather_only", "mla_tc_only", "a2a_only")
MODES = ("head_parallel", "token_parallel")
ROPE_ROW_BYTES = 128
NOPE_ROW_BYTES = 512
SWA_ROW_BYTES = 1024


def build_decode_inputs(
    torch,
    *,
    num_seqs: int,
    kv_len: int,
    page_size: int,
    window: int,
    topk: int,
    heads: int,
    head_dim: int,
    seed: int,
) -> dict[str, Any]:
    """Deterministic CPU inputs for one decode attention call.

    The last request has 16 trailing ``-1`` main records (a short request)
    so the ``kv_lens`` / ``-1`` tail path of SparseMLA is exercised.
    """
    import jax.numpy as jnp
    import numpy as np

    from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import encode_fp8_kv

    if kv_len < topk or kv_len % page_size or page_size % 4:
        raise ValueError(
            f"kv_len={kv_len} must be >= topk={topk}, a multiple of "
            f"page_size={page_size}, and page_size a multiple of 4"
        )
    rng = torch.Generator(device="cpu").manual_seed(seed + 2027)
    q = (
        torch.randn((num_seqs, heads, head_dim), generator=rng, dtype=torch.float32)
        * 0.25
    ).to(torch.bfloat16)
    sink = torch.randn((heads,), generator=rng, dtype=torch.float32) * 0.5
    query_start_loc = torch.arange(num_seqs + 1, dtype=torch.int32)

    swa_rows = num_seqs * window + 64
    swa_vals = (
        torch.randn((swa_rows, head_dim), generator=rng, dtype=torch.float32) * 0.25
    ).to(torch.bfloat16)
    swa_pool = swa_vals.view(torch.int32).reshape(swa_rows * 2, 128).contiguous()
    swa_records = (
        torch.arange(num_seqs, dtype=torch.int32)[:, None] * window
        + torch.arange(window, dtype=torch.int32)[None, :]
    ).contiguous()

    pages_per_seq = kv_len // page_size
    total_pages = pages_per_seq + num_seqs * 64
    total_rows = total_pages * page_size
    template_rows = 1024
    latent = (
        torch.randn((template_rows, head_dim), generator=rng, dtype=torch.float32) * 0.5
    ).to(torch.bfloat16)
    words_jax, rope_jax = encode_fp8_kv(
        jnp.asarray(latent.float().numpy(), jnp.bfloat16)
    )
    words = torch.from_numpy(np.asarray(words_jax, dtype=np.int32).copy())
    rope = torch.from_numpy(np.asarray(rope_jax, dtype=np.int32).copy())
    repeats = -(-total_rows // template_rows)
    main_pool = words.repeat((repeats, 1))[:total_rows].contiguous()
    rope_pool = (
        rope.repeat((repeats, 1))[:total_rows]
        .reshape(total_rows // 4, 128)
        .contiguous()
    )
    pages = (
        torch.arange(pages_per_seq, dtype=torch.int32)[None, :]
        + 61 * torch.arange(num_seqs, dtype=torch.int32)[:, None]
    ) % total_pages

    main_records = torch.empty((num_seqs, topk), dtype=torch.int32)
    for s in range(num_seqs):
        row = torch.randperm(kv_len, generator=rng, dtype=torch.int32)[:topk]
        main_records[s] = torch.sort(row).values
    main_records[-1, topk - 16 :] = -1
    return {
        "q": q.contiguous(),
        "sink": sink.contiguous(),
        "query_start_loc": query_start_loc,
        "swa_pool": swa_pool,
        "swa_records": swa_records,
        "main_pool": main_pool,
        "rope_pool": rope_pool,
        "pages": pages.contiguous(),
        "main_records": main_records,
    }


def decode_bytes_per_rank(tokens_local: int, topk: int, window: int) -> dict[str, int]:
    """HBM bytes one rank gathers for ``tokens_local`` decode tokens."""
    main = tokens_local * topk * (NOPE_ROW_BYTES + ROPE_ROW_BYTES)
    swa = tokens_local * window * SWA_ROW_BYTES
    return {"main_gather": main, "swa_gather": swa, "total": main + swa}


def worker(ctx: common.WorkerContext, args: argparse.Namespace) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import torch
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal import pallas
    from vllm.distributed import get_tp_group

    from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels import (
        _attend_window,
        _attention_state,
        slice_token_parallel_attention_inputs,
        sparse_fp8_attention_jax,
    )

    geo = common.Dsv41Geometry()
    tp = ctx.tp_size
    seqs = args.num_seqs
    heads_total = geo.num_attention_heads
    heads_local = geo.heads_per_rank(tp)
    head_dim = geo.head_dim
    window = geo.sliding_window
    topk = geo.index_topk
    scale = float(head_dim**-0.5)
    page_size = args.page_size
    max_requests = args.swa_max_requests
    reps = args.reps
    if args.mode == "token_parallel" and seqs % tp:
        raise ValueError(f"token_parallel needs --num-seqs % tp == 0, got {seqs}")
    tp_group = get_tp_group().device_group
    tp_rank = get_tp_group().rank_in_group
    tp_mesh = pallas.get_pallas_mesh(axis_names=("tp",), process_group=tp_group)

    cpu = build_decode_inputs(
        torch,
        num_seqs=seqs,
        kv_len=args.kv_len,
        page_size=page_size,
        window=window,
        topk=topk,
        heads=heads_total,
        head_dim=head_dim,
        seed=args.seed,
    )
    t_local = seqs // tp if seqs % tp == 0 else seqs
    t_start = tp_rank * t_local
    dev = {k: v.to("tpu") for k, v in cpu.items()}
    q_hp = (
        cpu["q"][:, tp_rank * heads_local : (tp_rank + 1) * heads_local]
        .contiguous()
        .to("tpu")
    )
    sink_hp = (
        cpu["sink"][tp_rank * heads_local : (tp_rank + 1) * heads_local]
        .contiguous()
        .to("tpu")
    )
    tp_rank_t = torch.tensor([tp_rank], dtype=torch.int32, device="tpu")
    # Rank-local token slices for token_parallel subops without collectives.
    loc = {
        "q": cpu["q"][t_start : t_start + t_local].contiguous().to("tpu"),
        "swa_records": cpu["swa_records"][t_start : t_start + t_local]
        .contiguous()
        .to("tpu"),
        "main_records": cpu["main_records"][t_start : t_start + t_local]
        .contiguous()
        .to("tpu"),
        "query_start_loc": (cpu["query_start_loc"] - t_start)
        .clamp(min=0, max=t_local)
        .to(torch.int32)
        .to("tpu"),
    }

    def _op(name, fn, n_in):
        # jax_op reads the annotated signature; wraps() exposes fn's.
        @functools.wraps(fn)
        def wrapped(*xs):
            return jax.shard_map(
                fn, mesh=tp_mesh, in_specs=(P(),) * n_in, out_specs=P(), check_vma=False
            )(*xs)

        return pallas.jax_op(
            f"pallas::dsv41_bench_decode_{name}",
            wrapped,
            mesh=tp_mesh,
            input_partition_specs=(P(),) * n_in,
        )

    attn_kw = dict(
        scale=scale, page_size=page_size, max_requests=max_requests, precompacted=True
    )

    def hp_once(q, s_p, m_p, r_p, s_r, m_r, pgs, q_loc, snk, _r_id):
        return sparse_fp8_attention_jax(
            q, s_p, m_p, r_p, s_r, m_r, pgs, q_loc, snk, **attn_kw
        )

    def tp_once(q, s_p, m_p, r_p, s_r, m_r, pgs, q_loc, snk, r_id):
        q_tok = jax.lax.all_to_all(q, "tp", split_axis=0, concat_axis=1, tiled=True)
        s_r_t, m_r_t, q_loc_t = slice_token_parallel_attention_inputs(
            s_r, m_r, q_loc, r_id, tp, seqs
        )
        out = sparse_fp8_attention_jax(
            q_tok, s_p, m_p, r_p, s_r_t, m_r_t, pgs, q_loc_t, snk, **attn_kw
        )
        return jax.lax.all_to_all(out, "tp", split_axis=1, concat_axis=0, tiled=True)

    def looped_all(once, n_reps):
        def fn(
            q: jax.Array,
            s_p: jax.Array,
            m_p: jax.Array,
            r_p: jax.Array,
            s_r: jax.Array,
            m_r: jax.Array,
            pgs: jax.Array,
            q_loc: jax.Array,
            snk: jax.Array,
            r_id: jax.Array,
        ) -> jax.Array:
            rest = (s_p, m_p, r_p, s_r, m_r, pgs, q_loc, snk, r_id)
            return jax.lax.fori_loop(0, n_reps, lambda _, x: once(x, *rest), q)

        return fn

    def swa_loop(q: jax.Array, s_p: jax.Array, s_r: jax.Array) -> jax.Array:
        def body(_, x):
            _, _, num = _attend_window(
                x, s_p, s_r, _attention_state(x), scale, max_requests=max_requests
            )
            return num.astype(x.dtype)

        return jax.lax.fori_loop(0, reps, body, q)

    def gather_loop(
        m_p: jax.Array, r_p: jax.Array, m_r: jax.Array, pgs: jax.Array, q_loc: jax.Array
    ) -> jax.Array:
        def body(_, carry):
            rec, acc = jax.lax.optimization_barrier((m_r, carry))
            n_s, r_s = _sc_gather_only_jax(
                m_p, r_p, rec, pgs, q_loc, page_size=page_size
            )
            return acc + n_s.sum() + r_s.sum()

        return jax.lax.fori_loop(0, reps, body, jnp.zeros((), jnp.int32))

    def mla_loop(
        q: jax.Array,
        n_c: jax.Array,
        r_c: jax.Array,
        kv_l: jax.Array,
        q_loc: jax.Array,
        snk: jax.Array,
        s_n: jax.Array,
        s_d: jax.Array,
        s_m: jax.Array,
    ) -> jax.Array:
        mla = functools.partial(
            _mla_tc_only_jax, scale=scale, max_num_seqs=int(q_loc.shape[0] - 1)
        )

        def body(_, x):
            return mla(x, n_c, r_c, kv_l, q_loc, snk, s_n, s_d, s_m)

        return jax.lax.fori_loop(0, reps, body, q)

    def a2a_loop(q: jax.Array) -> jax.Array:
        def body(_, x):
            t = jax.lax.all_to_all(x, "tp", split_axis=0, concat_axis=1, tiled=True)
            return jax.lax.all_to_all(t, "tp", split_axis=1, concat_axis=0, tiled=True)

        return jax.lax.fori_loop(0, reps, body, q)

    tp_mode = args.mode == "token_parallel"
    full_args = (
        q_hp,
        dev["swa_pool"],
        dev["main_pool"],
        dev["rope_pool"],
        dev["swa_records"],
        dev["main_records"],
        dev["pages"],
        dev["query_start_loc"],
        dev["sink"] if tp_mode else sink_hp,
        tp_rank_t,
    )
    q_l = loc["q"] if tp_mode else q_hp
    s_r_l = loc["swa_records"] if tp_mode else dev["swa_records"]
    m_r_l = loc["main_records"] if tp_mode else dev["main_records"]
    q_loc_l = loc["query_start_loc"] if tp_mode else dev["query_start_loc"]
    snk_l = dev["sink"] if tp_mode else sink_hp

    with common.model_wrapper_context():
        if args.subop == "all":
            once = tp_once if tp_mode else hp_once
            op = _op(f"all_{args.mode}", looped_all(once, reps), 10)

            def step():
                return {"out": op(*full_args)}

        elif args.subop == "swa_only":
            op = _op("swa", swa_loop, 3)

            def step():
                return {"num": op(q_l, dev["swa_pool"], s_r_l)}

        elif args.subop == "sc_gather_only":
            op = _op("gather", gather_loop, 5)

            def step():
                return {
                    "acc": op(
                        dev["main_pool"], dev["rope_pool"], m_r_l, dev["pages"], q_loc_l
                    )
                }

        elif args.subop == "mla_tc_only":

            def swa_state(
                q: jax.Array, s_p: jax.Array, s_r: jax.Array
            ) -> tuple[jax.Array, jax.Array, jax.Array]:
                return _attend_window(
                    q, s_p, s_r, _attention_state(q), scale, max_requests=max_requests
                )

            def pregather(
                m_p: jax.Array,
                r_p: jax.Array,
                m_r: jax.Array,
                pgs: jax.Array,
                q_loc: jax.Array,
            ) -> tuple[jax.Array, jax.Array, jax.Array]:
                return _pregather_all_chunks_jax(
                    m_p, r_p, m_r, pgs, q_loc, page_size=page_size
                )

            swa_once = _op("swa_state", swa_state, 3)
            pre = _op("pregather", pregather, 5)
            s_m, s_d, s_n = swa_once(q_l, dev["swa_pool"], s_r_l)
            n_c, r_c, kv_l = pre(
                dev["main_pool"], dev["rope_pool"], m_r_l, dev["pages"], q_loc_l
            )
            common.synchronize((s_m, s_d, s_n, n_c, r_c, kv_l))
            op = _op("mla", mla_loop, 9)

            def step():
                return {"out": op(q_l, n_c, r_c, kv_l, q_loc_l, snk_l, s_n, s_d, s_m)}

        else:  # a2a_only
            if seqs % tp:
                raise ValueError("a2a_only needs --num-seqs % tp == 0")
            op = _op("a2a", a2a_loop, 1)

            def step():
                return {"out": op(q_hp)}

        common.synchronize(full_args)
        hbm_before = common.hbm_stats()
        loop_samples, out = common.time_steps(
            ctx, step, warmup=args.warmup, iters=args.iters
        )
        profile_dir = common.profile_steps(
            ctx, args, step, label=f"decode_attn_{args.subop}_{args.mode}"
        )
        hbm_after = common.hbm_stats()

        verify_report: dict[str, Any] = {}
        if args.verify and args.subop == "all" and seqs % tp == 0:
            hp1 = _op("verify_hp", looped_all(hp_once, 1), 10)
            tp1 = _op("verify_tp", looped_all(tp_once, 1), 10)
            hp_args = full_args[:8] + (sink_hp, tp_rank_t)
            tp_args = full_args[:8] + (dev["sink"], tp_rank_t)
            out_hp = hp1(*hp_args)
            out_tp = tp1(*tp_args)
            common.synchronize((out_hp, out_tp))
            hp_cpu = out_hp.cpu()
            verify_report["finite_head_parallel"] = bool(
                torch.isfinite(hp_cpu.float()).all().item()
            )
            verify_report["token_parallel_vs_head_parallel"] = common.verify_outputs(
                {"out": out_tp.cpu()},
                {"out": hp_cpu},
                common.LOSSLESS,
                label="decode token_parallel vs head_parallel",
            )
            if not verify_report["finite_head_parallel"]:
                raise common.VerificationError("head_parallel output has non-finite")

    per_rep = [s / reps for s in loop_samples]
    bytes_hp = decode_bytes_per_rank(seqs, topk, window)
    bytes_tp = decode_bytes_per_rank(t_local, topk, window)
    return {
        "bench": "decode_attention",
        "subop": args.subop,
        "mode": args.mode,
        "num_seqs": seqs,
        "kv_len": args.kv_len,
        "page_size": page_size,
        "reps": reps,
        "swa_max_requests": max_requests,
        "wall_samples_ms": per_rep,
        "per_rep_median_ms": statistics.median(per_rep),
        "loop_median_ms": statistics.median(loop_samples),
        "gather_bytes_per_rank": bytes_tp if tp_mode else bytes_hp,
        "hbm_before": hbm_before,
        "hbm_after": hbm_after,
        "profile_dir": profile_dir,
        "verify": verify_report,
        "output_keys": sorted(out) if isinstance(out, dict) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(parser, SUBOPS)
    parser.add_argument("--mode", choices=list(MODES), default="head_parallel")
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument(
        "--swa-max-requests",
        type=int,
        default=None,
        help="serving uses max_num_seqs if <= 4 else None (attention.py:487)",
    )
    parser.set_defaults(num_seqs=16, kv_len=100000)
    common.run_main("decode_attention", parser, worker)


if __name__ == "__main__":
    main()
