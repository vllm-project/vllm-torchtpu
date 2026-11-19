# SPDX-License-Identifier: Apache-2.0
"""q38 L6 H0-2 PREP-TRIMS: CPU microbench of ``TPUModelRunner._prepare_inputs``.

CDK CPU pod only (never the Mac or the VM). Usage, from the tree root:
  python3 tests/q38_l6/bench_l6_ptrims.py --config model_config.json [--n 4,9,11,25,52]
      [--steps 400] [--warm 40] [--reps 3] [--arms off,sc,sce[,pt]] [--fragments]

Drives the real ``_prepare_inputs`` (and the real ``_modify_prev_results`` /
``_update_placeholder``) through the CARRY helper's harness
(tests/q38_l6/l6_carry_harness.py) on a served-shaped plan: async scheduling,
MTP k = 3 (every request schedules 1 + 3 tokens), VLLM_TPU_POSTFWD_FUSED on,
compact mamba slots and the combined RPA / GDN request distribution, decode
max_num_seqs 64 (so the request bucket is 64), token buckets 16..512.

Arms (each a fresh runner and the same seeded stream):
  off  every L6 lever unset (the stock path);
  sc   STEADY + CARRY on (VLLM_TPU_L6_STEADY_STEP=1, VLLM_TPU_L6_DECODE_CARRY=1);
  sce  sc + envcache (VLLM_TPU_L6_ENVS_CACHE=1);
  pt   sce + VLLM_TPU_L6_PREP_TRIMS=1 (only when the tree registers the lever);
  ab   sce + VLLM_TPU_L6_PREP_TRIMS=ab:1 with step windows of 50 (2 classes): the
       paired in-process A/B (one runner, one stream; steps split by the lever's
       state), the robust per-item delta.

Streams: ``steady`` (no batch change) and ``churn`` (a request joins, then one
leaves, alternating every 25 steps: the carried share drops).

Timing: ``time.perf_counter_ns`` around each ``_prepare_inputs`` call, split at
the runner's own sub-marks (host_phases.sub / mark): pi_entry, pi_fence,
pi_tokens, pi_layout (pi_h2d on an L6-INSTR tree), pi_spec_qsl, pi_post (pi_spec_qsl -> the
prep_host_and_uploads mark), am (-> the prep_attention_metadata mark; the
metadata builders are the harness stub here) and tail (-> return). On CPU the
compiled programs run as the eager functions they wrap and an upload is a
clone; their time is measured and reported separately (``prog``, ``h2d``) and
excluded from the host columns, which hold the Python / numpy work only. GC is
off inside the timed loop. Last line: ``BENCH_DONE ok=<bool>``.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from types import SimpleNamespace as NS

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import l6_carry_harness as H  # noqa: E402

from vllm_torchtpu.core import l6_ab, l6_envcache  # noqa: E402
from vllm_torchtpu.runner import decode_carry, tpu_runner  # noqa: E402
from vllm_torchtpu.runner.owned_args import static_shape_args_owned  # noqa: E402
from vllm_torchtpu.runner.tpu_runner_async_output import subtract_substitute_extract  # noqa: E402

PERF = time.perf_counter_ns
REGIONS = ("pi_entry", "pi_fence", "pi_tokens", "pi_layout", "pi_spec_qsl", "pi_post", "am", "tail")
PREP = REGIONS[:6]  # the prep_host_and_uploads phase
# q38 L6-V5: on a tree with L6-INSTR the former pi_layout mark is split into pi_lhost / pi_carry / pi_h2d; its
# region ends at pi_h2d, so pi_h2d closes the same pi_layout column (pi_lhost / pi_carry fall inside it).
SUBIDX = {"pi_entry": 0, "pi_fence": 1, "pi_tokens": 2, "pi_layout": 3, "pi_h2d": 3, "pi_spec_qsl": 4}
MARKIDX = {"prep_host_and_uploads": 5, "prep_attention_metadata": 6}
PADS = (16, 32, 64, 128, 256, 512)
MAX_NUM_REQS = 64
MAXLEN = 4096
ARM_ENV = {
    "off": {},
    "sc": {"VLLM_TPU_L6_STEADY_STEP": "1", "VLLM_TPU_L6_DECODE_CARRY": "1"},
    "sce": {"VLLM_TPU_L6_STEADY_STEP": "1", "VLLM_TPU_L6_DECODE_CARRY": "1",
            "VLLM_TPU_L6_ENVS_CACHE": "1"},
    "pt": {"VLLM_TPU_L6_STEADY_STEP": "1", "VLLM_TPU_L6_DECODE_CARRY": "1",
           "VLLM_TPU_L6_ENVS_CACHE": "1", "VLLM_TPU_L6_PREP_TRIMS": "1"},
    "ab": {"VLLM_TPU_L6_STEADY_STEP": "1", "VLLM_TPU_L6_DECODE_CARRY": "1",
           "VLLM_TPU_L6_ENVS_CACHE": "1", "VLLM_TPU_L6_PREP_TRIMS": "ab:1",
           "VLLM_TPU_L6_AB_STEPS": "50", "VLLM_TPU_L6_AB_CLASSES": "2"},
}


class Rec:
    """Stands in for runner/host_phases.py inside tpu_runner: region clock."""

    def __init__(self):
        self.t = 0
        self.i = 0
        self.acc = [0] * len(REGIONS)
        self.exc = [0] * len(REGIONS)
        self.nh2d = 0
        self.mamba = 0

    def start(self):
        self.acc = [0] * len(REGIONS)
        self.exc = [0] * len(REGIONS)
        self.nh2d = 0
        self.mamba = 0
        self.i = 0
        self.t = PERF()

    def _close(self, idx):
        now = PERF()
        self.acc[idx] += now - self.t
        self.t = now
        self.i = idx + 1

    def sub(self, name):
        idx = SUBIDX.get(name)
        if idx is not None:
            self._close(idx)

    def mark(self, name):
        idx = MARKIDX.get(name)
        if idx is not None:
            self._close(idx)

    def end(self):
        self._close(len(REGIONS) - 1)

    def excl(self, ns):
        self.exc[min(self.i, len(REGIONS) - 1)] += ns

    def __getattr__(self, name):  # note / charge / slow_enabled / begin_step ...
        return lambda *a, **k: None


REC = Rec()


def timed(fn):
    def w(*a, **k):
        t0 = PERF()
        out = fn(*a, **k)
        REC.excl(PERF() - t0)
        return out

    return w


class TimedUploads(H.Uploads):
    """H.Uploads (an upload is a counted clone) with its time charged to ``h2d``."""

    def __enter__(self):
        self._had = "to" in torch.Tensor.__dict__
        self._prev = torch.Tensor.__dict__.get("to")
        orig = torch.Tensor.to
        me = self

        def to(t, *args, **kwargs):
            t0 = PERF()
            out = orig(t, *args, **kwargs)
            dev = args[0] if args else kwargs.get("device")
            if isinstance(dev, torch.device):
                me.n += 1
                REC.nh2d += 1
                if out is t:
                    out = t.clone()
            REC.excl(PERF() - t0)
            return out

        torch.Tensor.to = to
        return self


ENV_KEYS = sorted({k for v in ARM_ENV.values() for k in v} | {
    "VLLM_TPU_L6_PREP_TRIMS_CHECK", "VLLM_TPU_L6_DECODE_CARRY_CHECK",
    "VLLM_TPU_L6_STEADY_STEP_CHECK", "VLLM_TPU_L6_ENVS_CACHE_CHECK", "VLLM_TPU_L6_CHECK_STEPS",
    "VLLM_TPU_L6_AB_PERIOD_MS", "VLLM_TPU_L6_AB_STEPS", "VLLM_TPU_L6_AB_CLASSES"})


def set_arm(arm):
    for k in ENV_KEYS:
        os.environ.pop(k, None)
    for k, v in ARM_ENV[arm].items():
        os.environ[k] = v
    l6_envcache.restore()
    l6_ab._resolve()
    try:
        from vllm_torchtpu.runner import l6_steady

        l6_steady._CHECK[0] = None
    except ImportError:
        pass


def events_for(stream, step):
    if stream == "churn" and step > 0 and step % 25 == 0:
        return ("join",) if (step // 25) % 2 else ("leave",)
    return ()


def run_cfg(n, arm, stream, steps, warm, seed):
    set_arm(arm)
    r = H.make_runner(nreq=n, max_num_reqs=MAX_NUM_REQS, paddings=PADS, mamba=True, comb=True,
                      max_len=MAXLEN, seed=seed)
    sim = H.Sim(r, nreq=n, seed=seed)
    sim._digest = lambda out, nn: {}
    orig = r._prepare_inputs
    orig_m = r._build_mamba_state_indices
    rows = []

    def mamba(*a, **k):
        e0 = sum(REC.exc)
        t0 = PERF()
        out = orig_m(*a, **k)
        REC.mamba += PERF() - t0 - (sum(REC.exc) - e0)
        return out

    def prep(*a, **k):
        pt_on = l6_ab.on("ptrims")
        REC.start()
        out = orig(*a, **k)
        REC.end()
        rows.append((REC.acc, REC.exc, REC.nh2d, REC.mamba, pt_on))
        return out

    r._prepare_inputs = prep
    r._build_mamba_state_indices = mamba
    gc.collect()
    gc.disable()
    try:
        for s in range(warm + steps):
            sim.step(events_for(stream, s))
    finally:
        gc.enable()
    counts = dict(l6_ab._state.get("counts", {}))
    return rows[warm:], counts, r, sim


def summarize(rows):
    """Per region: mean and p50 host us (excluding prog/h2d stand-ins); plus totals."""
    k = len(rows)
    host = np.array([[a - e for a, e in zip(r[0], r[1])] for r in rows], dtype=np.float64) / 1e3
    exc = np.array([r[1] for r in rows], dtype=np.float64) / 1e3
    mam = np.array([r[3] for r in rows], dtype=np.float64) / 1e3
    prep_host = host[:, :6].sum(axis=1)
    call_host = host.sum(axis=1)
    return NS(
        n=k,
        region_mean=host.mean(axis=0),
        region_p50=np.median(host, axis=0),
        prep_mean=float(prep_host.mean()),
        prep_p50=float(np.median(prep_host)),
        call_mean=float(call_host.mean()),
        call_p50=float(np.median(call_host)),
        mamba_mean=float(mam.mean()),
        mamba_p50=float(np.median(mam)),
        excl_mean=float(exc.sum(axis=1).mean()),
        h2d=float(np.mean([r[2] for r in rows])),
    )


# ---------------------------------------------------------------- fragments


def tmin(fn, reps=7, loops=400):
    """Best-of-reps mean us per call."""
    best = None
    for _ in range(reps):
        t0 = PERF()
        for _ in range(loops):
            fn()
        dt = (PERF() - t0) / loops / 1e3
        best = dt if best is None or dt < best else best
    return best


def fragments(r, sim, n):
    """us per call of each candidate trim's stock code and its trimmed form, on
    the runner's state after a steady run (sizes for the decision; I-level:
    isolated, cache-warm)."""
    ib = r.input_batch
    req_ids = list(ib.req_ids[:n])
    sched = {rid: 1 + H.K for rid in req_ids}
    spec = {rid: [0] * H.K for rid in req_ids}
    so = NS(num_scheduled_tokens=sched, scheduled_spec_decode_tokens=spec,
            total_num_scheduled_tokens=n * (1 + H.K))
    out = {}

    def sched_stock():
        lst = []
        for i in range(0, n):
            rid = ib.req_ids[i]
            assert rid is not None
            t = so.num_scheduled_tokens[rid]
            if not True and r.most_model_len is not None and t > r.most_model_len:
                pass
            lst.append(t)
        mx = max(lst)
        a = np.array(lst, dtype=np.int32)
        tot = sum(a)
        return mx, a, tot

    def sched_trim():
        lst = list(map(so.num_scheduled_tokens.__getitem__, ib.req_ids[0:n]))
        mx = max(lst)
        a = np.array(lst, dtype=np.int32)
        tot = int(a.sum())
        return mx, a, tot

    out["F1a sched gather + array + sum"] = (tmin(sched_stock), tmin(sched_trim))
    nsched = np.full(n, 1 + H.K, dtype=np.int32)
    total = int(nsched.sum())
    stride = ib.token_ids_cpu.shape[1]

    def idx_stock():
        req_indices = np.repeat(r.arange_np[0:n], nsched)
        arange = np.concatenate([r.arange_np[:k] for k in nsched])
        positions_np = r.positions_np[:total]
        np.add(ib.num_computed_tokens_cpu[req_indices], arange, out=positions_np)
        token_indices = positions_np + req_indices * stride
        torch.index_select(ib.token_ids_cpu_tensor.flatten(), 0, torch.from_numpy(token_indices),
                           out=r.input_ids_cpu[:total])

    req_indices_c = np.repeat(r.arange_np[0:n], nsched)
    arange_c = np.concatenate([r.arange_np[:k] for k in nsched])
    off_c = (req_indices_c * stride).astype(np.int64)
    tok_c = np.empty(total, dtype=np.int64)
    flat_t = ib.token_ids_cpu_tensor.flatten()
    tok_t = torch.from_numpy(tok_c)

    def idx_trim():
        positions_np = r.positions_np[:total]
        np.add(ib.num_computed_tokens_cpu[req_indices_c], arange_c, out=positions_np)
        np.add(positions_np, off_c, out=tok_c)
        torch.index_select(flat_t, 0, tok_t, out=r.input_ids_cpu[:total])

    out["F1b token index build"] = (tmin(idx_stock), tmin(idx_trim))

    def qsl():
        r.query_start_loc_np[0] = 0
        np.cumsum(nsched, out=r.query_start_loc_np[1 : n + 1])
        r.query_start_loc_np[n + 1 :] = r.query_start_loc_np[n]

    out["F2a qsl cumsum + tail"] = (tmin(qsl), 0.0)
    out["F2b schedule check"] = (
        tmin(lambda: r._check_attention_schedule(n, nsched, n, True)), 0.0)

    def nd_stock():
        return np.array([len(so.scheduled_spec_decode_tokens.get(rid, ())) for rid in ib.req_ids[0:n]],
                        dtype=np.int32)

    import itertools

    def nd_trim():
        return np.fromiter(map(len, map(so.scheduled_spec_decode_tokens.get, ib.req_ids[0:n],
                                        itertools.repeat(()))), dtype=np.int32, count=n)

    assert np.array_equal(nd_stock(), nd_trim())
    out["F3a num_draft_tokens gather"] = (tmin(nd_stock), tmin(nd_trim))
    nd = nd_stock()
    qsl()
    mgr = r.spec_decode_manager
    pad_reqs = tpu_runner._get_padded_num_reqs_with_upper_limit(n, r.max_num_reqs)
    mgr.get_spec_decode_metadata(nd, r.query_start_loc_np[1 : n + 1], pad_reqs, defer_draft_token_ids=True)
    out["F3b spec metadata (cache hit)"] = (
        tmin(lambda: mgr.get_spec_decode_metadata(nd, r.query_start_loc_np[1 : n + 1], pad_reqs,
                                                  defer_draft_token_ids=True)), 0.0)
    args = tuple(torch.zeros(4 * n + 4, dtype=torch.int32) for _ in range(11))

    def ckey():
        key = (4,) + tuple((tuple(a.shape), a.dtype) for a in args)
        return key

    def cwrap():
        return static_shape_args_owned(lambda *a: None)

    out["F4a carry shape key"] = (tmin(ckey), 0.0)
    out["F4b carry wrapper build"] = (tmin(cwrap), 0.0)
    w = static_shape_args_owned(lambda *a: None)
    out["F4c carry wrapper call (stamp scan)"] = (tmin(lambda: w(4, *args)), 0.0)
    if r._pre_async_results is not None:
        out["F5 substitution index build"] = (
            tmin(lambda: r._prepare_async_token_substitution_indices(0, n, nsched)), 0.0)
    if r._has_mamba_state:
        target = r.num_reqs_max_model_len

        def mamba_stock():
            idx = r.mamba_state_indices_cpu[:target]
            idx.fill_(0)
            for i in range(n):
                idx[i] = r._mamba_slot_by_req_id[ib.req_ids[i]]

        def mamba_trim():
            idx = r.mamba_state_indices_cpu[:target]
            idx.fill_(0)
            get = r._mamba_slot_by_req_id.__getitem__
            idx.numpy()[:n] = list(map(get, ib.req_ids[0:n]))

        out["F6 mamba slot indices"] = (tmin(mamba_stock), tmin(mamba_trim))
    comb = r._combined_request_distribution_cpu

    def comb_stock():
        comb[0] = n
        comb[1] = n
        comb[2] = n
        comb[3] = n
        comb[4] = n
        comb[5] = n

    def comb_trim():
        comb.numpy()[:] = (n, n, n, n, n, n)

    out["F7 combined distribution writes"] = (tmin(comb_stock), tmin(comb_trim))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", default="4,9,11,25,52")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--warm", type=int, default=40)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--arms", default="off,sc,sce,pt,ab")
    ap.add_argument("--streams", default="steady,churn")
    ap.add_argument("--fragments", action="store_true")
    a = ap.parse_args()
    if a.config:
        cfg = json.load(open(a.config))
        t = cfg.get("text_config", cfg)
        print(f"BENCH config model_type={cfg.get('model_type')} ngram_size={t.get('ngram_size')} "
              f"mtp_layers={t.get('mtp_num_hidden_layers')} vocab={t.get('vocab_size')} "
              f"(harness NGRAM={H.NGRAM}, K={H.K})")
        assert int(t.get("ngram_size", H.NGRAM)) == H.NGRAM
    arms = [x for x in a.arms.split(",") if x]
    if "ptrims" not in l6_ab.LEVERS:
        for x in ("pt", "ab"):
            if x in arms:
                print(f"BENCH arm {x} skipped: this tree registers no ptrims lever")
                arms.remove(x)
    ns = [int(x) for x in a.n.split(",")]
    streams = [x for x in a.streams.split(",") if x]
    # Keep the wrappers (static_shape_args_owned) and replace only the compiled programs.
    tpu_runner.host_phases = REC
    tpu_runner._postfwd_subst_compiled = timed(subtract_substitute_extract)
    decode_carry._PROGRAM[:] = [timed(decode_carry._carry_body)]
    H.Uploads = TimedUploads
    os.environ["VLLM_TPU_POSTFWD_FUSED"] = "1"
    l6_ab.logger.info = lambda *x, **k: None
    print(f"BENCH start {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} nproc {os.cpu_count()} "
          f"python {sys.version.split()[0]} torch {torch.__version__} numpy {np.__version__} "
          f"arms={arms} n={ns} streams={streams} steps={a.steps} warm={a.warm} reps={a.reps}")
    res: dict = {}
    counts_seen: dict = {}
    last = {}
    for rep in range(a.reps):
        for stream in streams:
            for n in ns:
                for arm in arms:
                    rows, counts, r, sim = run_cfg(n, arm, stream, a.steps, a.warm, seed=100 + n)
                    if arm == "ab":
                        on_rows = [x for x in rows if x[4]]
                        off_rows = [x for x in rows if not x[4]]
                        res.setdefault((stream, n, "ab.on"), []).append(summarize(on_rows))
                        res.setdefault((stream, n, "ab.off"), []).append(summarize(off_rows))
                    else:
                        res.setdefault((stream, n, arm), []).append(summarize(rows))
                    counts_seen[(stream, n, arm)] = counts
                    if stream == "steady" and arm == "sce":
                        last[n] = (r, sim)
    ok = True
    tarms = [x for x in arms if x != "ab"] + (["ab.off", "ab.on"] if "ab" in arms else [])
    for stream in streams:
        print(f"BENCH table stream={stream}: host us per _prepare_inputs call (median over {a.reps} reps of the "
              f"per-run mean; prog/h2d stand-ins excluded)")
        print("BENCH   n arm    | prep_host mean p50 | call_host mean | region p50: " + " ".join(f"{x:>11}" for x in REGIONS)
              + " | mamba_p50 | h2d/call prog+h2d_us")
        for n in ns:
            for arm in tarms:
                s = res[(stream, n, arm)]
                med = lambda f: statistics.median(f(x) for x in s)  # noqa: E731
                reg = [statistics.median(float(x.region_p50[i]) for x in s) for i in range(len(REGIONS))]
                print(f"BENCH {n:3d} {arm:6s} | {med(lambda x: x.prep_mean):9.1f} {med(lambda x: x.prep_p50):6.1f} | "
                      f"{med(lambda x: x.call_mean):9.1f}      |             " + " ".join(f"{v:11.1f}" for v in reg)
                      + f" | {med(lambda x: x.mamba_p50):9.1f} | {med(lambda x: x.h2d):5.2f} {med(lambda x: x.excl_mean):8.1f}")
        for n in ns:
            for arm in arms:
                c = counts_seen.get((stream, n, arm)) or {}
                if arm == "ab":
                    c = {}  # window counts reset every 50 steps
                if c:
                    keys = ",".join(f"{k}={v}" for k, v in sorted(c.items()))
                    print(f"BENCH counts stream={stream} n={n} arm={arm} {keys}")
        if "off" in arms and "sce" in arms:
            for n in ns:
                off = statistics.median(x.prep_mean for x in res[(stream, n, "off")])
                sce = statistics.median(x.prep_mean for x in res[(stream, n, "sce")])
                line = f"BENCH delta stream={stream} n={n} prep_host off {off:.1f} sce {sce:.1f} (sce-off {sce - off:+.1f})"
                if "pt" in arms:
                    pt = statistics.median(x.prep_mean for x in res[(stream, n, "pt")])
                    ptc = statistics.median(x.call_mean for x in res[(stream, n, "pt")])
                    scec = statistics.median(x.call_mean for x in res[(stream, n, "sce")])
                    line += f" pt {pt:.1f} (pt-sce {pt - sce:+.1f}; whole call pt-sce {ptc - scec:+.1f})"
                print(line)
        if "ab" in arms:
            for n in ns:
                on_, off_ = res[(stream, n, "ab.on")], res[(stream, n, "ab.off")]
                d = lambda f: statistics.median(f(x) for x in on_) - statistics.median(f(x) for x in off_)  # noqa: E731
                reg = [statistics.median(float(x.region_p50[i]) for x in on_)
                       - statistics.median(float(x.region_p50[i]) for x in off_) for i in range(len(REGIONS))]
                print(f"BENCH paired stream={stream} n={n} ptrims on-off: prep_host mean {d(lambda x: x.prep_mean):+.1f} "
                      f"p50 {d(lambda x: x.prep_p50):+.1f}; whole call mean {d(lambda x: x.call_mean):+.1f} "
                      f"p50 {d(lambda x: x.call_p50):+.1f}; mamba p50 {d(lambda x: x.mamba_p50):+.1f}; region p50 "
                      + " ".join(f"{REGIONS[i]}={reg[i]:+.1f}" for i in range(len(REGIONS))))
    if a.fragments:
        print("BENCH fragments: us per call, stock / trimmed (best of 7 x 400, isolated, cache-warm)")
        for n in ns:
            if n not in last:
                continue
            set_arm("sce")
            r, sim = last[n]
            for name, (stock, trim) in fragments(r, sim, n).items():
                tr = f"{trim:7.2f}" if trim else "      -"
                print(f"BENCH frag n={n:3d} {name:40s} stock {stock:7.2f} trimmed {tr}")
    set_arm("off")
    print(f"BENCH_DONE ok={ok}")


if __name__ == "__main__":
    main()
