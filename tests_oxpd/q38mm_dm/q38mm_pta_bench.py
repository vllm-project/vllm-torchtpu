# SPDX-License-Identifier: Apache-2.0
"""CPU microbench for VLLM_TPU_PROMPT_IDS_NDARRAY (q38mm DM lane, P15): the two prompt-size costs of a
decode->mixed boundary, list vs int32 array, on the image's vLLM 0.32 classes.

A real SchedulerOutput holds one new request (prompt N tokens, block_ids over the served 59 KV groups:
37 x 512-token, 9 x 1024-token, 13 Mamba groups) plus 31 running decoders (3 spec tokens each). Per N and mode:
  dumps   engine: the shm broadcast's pickler (HIGHEST_PROTOCOL, out-of-band buffers >= 1 MiB)
  loads   worker: pickle.loads of that message (the 5.3 ms measured in j-ed3c3078)
  add     worker: CachedRequestState(...) + InputBatch.add_request (the 4.9 ms measured)
  convert engine, once at arrival: the list -> int32 array conversion (input thread when served)
Prints one JSON line per (N, mode) and a summary table. Usage: python q38mm_pta_bench.py [--reps 25] [--n 94000 ...]
"""

import argparse
import copyreg
import io
import json
import pickle
import random
import statistics as st
import time

import numpy as np
import torch

from vllm_torchtpu.core import prompt_ids_ndarray as P

VOCAB = 151_936


def shm_dumps(obj):
    bufs = [b""]

    def oob(buf):
        raw = buf.raw()
        if len(raw) < 1024 * 1024:
            return True
        bufs.append(raw)
        return False

    with io.BytesIO() as bio:
        p = pickle.Pickler(bio, protocol=pickle.HIGHEST_PROTOCOL, buffer_callback=oob)
        p.dispatch_table = dict(copyreg.dispatch_table)
        p.dump(obj)
        bufs[0] = bio.getvalue()
    return bufs


def make_so(prompt, n, rnd):
    from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput

    groups = [512] * 37 + [1024] * 9 + [0] * 13
    blocks = tuple([rnd.randrange(0, 134_802) for _ in range((n + bs - 1) // bs if bs else 2)] for bs in groups)
    nrd = NewRequestData(req_id="new-0", prompt_token_ids=prompt, mm_features=[], sampling_params=None,
                         pooling_params=None, block_ids=blocks, num_computed_tokens=max(0, n - 5400),
                         lora_request=None)
    so = SchedulerOutput.make_empty()
    so.scheduled_new_reqs = [nrd]
    rids = [f"run-{i}" for i in range(31)]
    cr = CachedRequestData.make_empty()
    cr.req_ids = rids
    cr.new_block_ids = [None] * 31
    cr.num_computed_tokens = [90_000 + i for i in range(31)]
    cr.num_output_tokens = [500 + i for i in range(31)]
    so.scheduled_cached_reqs = cr
    so.num_scheduled_tokens = {"new-0": 2048, **{r: 4 for r in rids}}
    so.total_num_scheduled_tokens = 2048 + 4 * 31
    so.scheduled_spec_decode_tokens = {r: [rnd.randrange(VOCAB) for _ in range(3)] for r in rids}
    return so


def timeit(fn, reps):
    xs = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t0) * 1e3)
    xs.sort()
    return {"median": round(st.median(xs), 4), "p90": round(xs[int(0.9 * (len(xs) - 1))], 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--n", type=int, nargs="*", default=[30_000, 60_000, 94_000, 130_000])
    a = ap.parse_args()
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

    sp = SamplingParams(temperature=0.0, max_tokens=16)

    def new_batch():
        return InputBatch(max_num_reqs=2, max_model_len=262_144, max_num_batched_tokens=8192,
                          device=torch.device("cpu"), vocab_size=VOCAB, block_sizes=[1024],
                          max_num_blocks_per_req=[256], num_spec_tokens=3)

    rows = []
    for n in a.n:
        rnd = random.Random(n)
        lst = [rnd.randrange(0, VOCAB) for _ in range(n)]
        conv = timeit(lambda: P.to_array_chunked(lst), a.reps)
        conv_np = timeit(lambda: np.asarray(lst, dtype=np.int32), a.reps)
        for mode in ("list", "ndarray"):
            prompt = list(lst) if mode == "list" else P.to_array(lst)
            so = make_so(prompt, n, rnd)
            bufs = shm_dumps(so)
            dumps = timeit(lambda: shm_dumps(so), a.reps)
            loads = timeit(lambda: pickle.loads(bufs[0], buffers=[bytearray(b) for b in bufs[1:]]), a.reps)
            got = pickle.loads(bufs[0], buffers=[bytearray(b) for b in bufs[1:]]).scheduled_new_reqs[0]

            def add_once():
                # A fresh InputBatch per sample (untimed); timed: CachedRequestState(...) + add_request, as in
                # gpu_model_runner._update_states for a new request.
                batch = new_batch()
                t0 = time.perf_counter()
                st_ = CachedRequestState(req_id="new-0", prompt_token_ids=got.prompt_token_ids, mm_features=[],
                                         sampling_params=sp, generator=None, block_ids=([0, 1],),
                                         num_computed_tokens=got.num_computed_tokens, output_token_ids=[])
                i = batch.add_request(st_)
                dt = (time.perf_counter() - t0) * 1e3
                assert batch.token_ids_cpu[i, :n].tolist() == lst
                return dt

            xs = sorted(add_once() for _ in range(a.reps))
            addt = {"median": round(st.median(xs), 4), "p90": round(xs[int(0.9 * (len(xs) - 1))], 4)}
            ok = True
            r = {"n": n, "mode": mode, "msg_bytes": sum(len(b) for b in bufs), "dumps_ms": dumps, "loads_ms": loads,
                 "add_ms": addt, "convert_once_ms": conv if mode == "ndarray" else None,
                 "convert_np_asarray_ms": conv_np if mode == "ndarray" else None, "ok": ok}
            print("Q38MM_PTA_BENCH " + json.dumps(r), flush=True)
            rows.append(r)
    print(f"{'N':>7} {'mode':>8} {'bytes':>9} {'dumps':>8} {'loads':>8} {'add':>8} {'convert@arrival':>16}")
    for r in rows:
        print(f"{r['n']:>7} {r['mode']:>8} {r['msg_bytes']:>9} {r['dumps_ms']['median']:>8.3f} {r['loads_ms']['median']:>8.3f} "
              f"{r['add_ms']['median']:>8.3f} {(r['convert_once_ms'] or {}).get('median', float('nan')):>16.3f}")
    for n in a.n:
        l = next(r for r in rows if r["n"] == n and r["mode"] == "list")
        d = next(r for r in rows if r["n"] == n and r["mode"] == "ndarray")
        save = (l["loads_ms"]["median"] + l["add_ms"]["median"]) - (d["loads_ms"]["median"] + d["add_ms"]["median"])
        print(f"N={n}: worker critical-path saving (loads + add) {save:.3f} ms; engine dumps change "
              f"{d['dumps_ms']['median'] - l['dumps_ms']['median']:+.3f} ms; one-time conversion {d['convert_once_ms']['median']:.3f} ms")


if __name__ == "__main__":
    main()
