# SPDX-License-Identifier: Apache-2.0
"""q38 HXSMALL (DSTEP lane, 2026-11-22): shared switches and helpers of the five bitwise S2 smalls on the ox form.

Every member is behind its own registered, hashed, strict env key (envs.py: "0" default, "1" on, anything else raises):

  VLLM_TPU_HXS_GLUE_TRIM    P1d  forward glue outside the whole-step kernel
  VLLM_TPU_HXS_LEAN_DRAFT   1a   lean MTP draft program
  VLLM_TPU_HXS_VERIFY_FUSE  6    verify / sampler glue (XLA part)
  VLLM_TPU_HXS_AR0ST        B    AR0S-T: last expert's twin half before the all-reduce round 0 (in-kernel)
  VLLM_TPU_HXS_RUNNER_HOIST R    runner input programs (device part)

A key is read when the program that uses it is TRACED (JAX / torch.compile cache their traces: tests re-trace per
setting). When a key engages, `engage` prints one line per process and key through vLLM's handlers:
  'q38 HXSMALL <KEY> engaged: <detail>'
which the STEP car's reader requires on its x arms and forbids on its base arms.
"""

from __future__ import annotations

import logging
import math

KEYS = (
    "VLLM_TPU_HXS_GLUE_TRIM",
    "VLLM_TPU_HXS_LEAN_DRAFT",
    "VLLM_TPU_HXS_VERIFY_FUSE",
    "VLLM_TPU_HXS_AR0ST",
    "VLLM_TPU_HXS_RUNNER_HOIST",
)
ENGAGE_PREFIX = "q38 HXSMALL "
_ENGAGED: set = set()


def on(key: str) -> bool:
    """True iff the registered strict key `key` is "1" (its envs getter raises on any value but "0" / "1"). The registry's own
    getter is called, so the value is read from the environment at trace time (not through envs' optional __getattr__
    cache, which a process enables once the service is up: the environment does not change after that)."""
    assert key in KEYS, key
    from vllm_torchtpu import envs

    return envs.environment_variables[key]() == "1"


def engage(key: str, detail: str) -> None:
    """The engagement line of `key`, once per process (at trace time)."""
    assert key in KEYS, key
    if key in _ENGAGED:
        return
    _ENGAGED.add(key)
    logging.getLogger("vllm.vllm_torchtpu.hxsmall").info("%s%s engaged: %s", ENGAGE_PREFIX, key, detail)


# ---------------------------------------------------------------------------------------------------------- P1d
def ring_tag_batch_pad(count: int) -> int:
    """Rows to pad a ring-tag id vector of `count` ids to, so the batched ring read / write runs whole DMA batches:
    with VLLM_TPU_QSA_RING_DMA_BATCH=B the kernels use the largest power of two <= B dividing the row count
    (tuned_block_sizes.get_ring_tag_dma_batch), so 44 ids run 11 grid steps of 4 and 64 ids 2 grid steps of 32.
    Returns `count` (no padding) when the override is unset or already divides it."""
    from vllm_torchtpu import envs

    override = envs.VLLM_TPU_QSA_RING_DMA_BATCH
    if not override:
        return count
    b = 1 << (int(override).bit_length() - 1)
    return int(math.ceil(count / b) * b)


def read_ring_and_tags_padded(pool, ids, ring_elements: int, tag_elements: int):
    """`qsa_state.read_ring_and_tags(pool, ids, ...)`, bit for bit, with `ids` zero-padded to whole DMA batches and the
    padding rows cropped off. Each output row is a DMA copy of `pool[max(id, 0)]` alone (the kernel's lanes are
    independent), so padding rows (id 0: the null page) change no other row."""
    import jax.numpy as jnp

    from vllm_torchtpu.kernels.qwen4_exp.qsa_state import read_ring_and_tags

    n = int(ids.shape[0])
    m = ring_tag_batch_pad(n)
    if m != n:
        ids = jnp.pad(ids, (0, m - n))
    ring, tags = read_ring_and_tags(pool, ids, ring_elements, tag_elements)
    return ring[:n], tags[:n]


def write_ring_and_tags_padded(pool, ring, tags, ids):
    """`qsa_state.write_ring_and_tags(pool, ring, tags, ids)`, same pool bytes, with the rows zero-padded to whole DMA
    batches: the kernel issues a lane's DMA only for id > 0, so padding rows (id 0) write nothing."""
    import jax.numpy as jnp

    from vllm_torchtpu.kernels.qwen4_exp.qsa_state import write_ring_and_tags

    n = int(ids.shape[0])
    m = ring_tag_batch_pad(n)
    if m != n:
        ids = jnp.pad(ids, (0, m - n))
        ring = jnp.pad(ring, ((0, m - n), (0, 0)))
        tags = jnp.pad(tags, ((0, m - n), (0, 0)))
    return write_ring_and_tags(pool, ring, tags, ids)


# ----------------------------------------------------------------------------------------------------------- 1a
def zero_rows_routing(x, topk_weights):
    """q38 HXSMALL LEAN_DRAFT (the draft's armed MoE layers): top-k weights of all-zero input rows set to 0, so experts
    reached only by such rows drop out of the fused kernel's active set (no weight DMA, no matmul).

    Which rows: the drafter zeroes its pad rows before the MoE (models/vllm/qwen4_exp/model.py loop step
    `torch.where(valid_tokens[:, None], block, 0)`; step 0's zero-padded rows past the live width); a live row with a
    nonzero hidden state keeps its weights bit for bit. Live rows: their coefficients are unchanged and the kernel visits
    the remaining active experts in the same ascending order into the same accumulator, so a live row's output is the
    flag-off output, bit for bit, except in one theoretical case: an expert dropped here contributed exactly +0.0 to the
    live row (its coefficient for that row is 0), and +0.0 changes an accumulator only when it is exactly -0.0. An
    all-zero row itself gets the shared expert only (pad outputs are never consumed)."""
    import torch

    keep = (x != 0).any(dim=-1, keepdim=True)
    return torch.where(keep, topk_weights, torch.zeros_like(topk_weights))


def mark_lean_draft(draft, supported) -> int:
    """Mark the drafter's MoE layers whose quant method is armed for the fused decode kernel (`supported(qm)`); the
    fp8 apply reads the static attribute at trace time. Returns the count; logs the engagement line when > 0."""
    n = 0
    for m in draft.modules():
        qm = getattr(m, "quant_method", None)
        if qm is not None and supported(qm):
            m._hxs_lean_draft = True
            n += 1
    if n:
        engage("VLLM_TPU_HXS_LEAN_DRAFT", "draft MoE: %d armed layer(s) route all-zero (pad) rows nowhere" % n)
    return n


# ------------------------------------------------------------------------------------------- construction lines
# q38 HXSMALL construction lines (DSTEP lane, 2026-11-23; after T59o j-09663f5f read VOID-ENG of record): GLUE_TRIM's and
# AR0ST's `engage` lines print only when the program that uses them is TRACED, and a warm compile cache loads the compiled
# whole step without tracing it (T59o's xb arm: 0 lines, 'Loading compiled artifact'). `construct_lines` runs at
# Qwen38WholeStep construction (every engine start, warm or cold; called from one line of its __init__) and prints one
# 'q38 HXSMALL construct <KEY> (whole step <tag>): ...' line per key that is "1", with the decision the trace makes.
# No side effects: `_ENGAGED` is untouched (the trace-time 'engaged:' lines still print on a cold trace), the tile / plan
# work below runs with its own log records dropped and every module-level once-state of vllm_torchtpu restored
# (`wholestep_tiles` logs once-per-process lines of other members). Keys unset / "0": no line, returns [], nothing runs.
# Appended after every existing line of this module.
CONSTRUCT_PREFIX = "q38 HXSMALL construct "


def construct_lines(tag, buckets, spec, R, max_groups) -> list:
    """The construction lines of GLUE_TRIM and AR0ST (one INFO line per key that is "1"); returns the messages."""
    msgs = []
    if on("VLLM_TPU_HXS_GLUE_TRIM"):
        from vllm_torchtpu import envs

        msgs.append("%sVLLM_TPU_HXS_GLUE_TRIM (whole step %s, buckets %s): whole-step inputs read every QSA layer's ring ids "
                    "in one padded read + one rope gather; the QSA writer pads its ring ids to whole DMA batches "
                    "(VLLM_TPU_QSA_RING_DMA_BATCH=%s)" % (CONSTRUCT_PREFIX, tag, ",".join(str(int(b)) for b in buckets),
                                                          envs.VLLM_TPU_QSA_RING_DMA_BATCH or "unset"))
    if on("VLLM_TPU_HXS_AR0ST"):
        try:
            per = _quiet(lambda: [(int(b), ar0st_decide(spec, _whole_ex(R, int(b), max_groups))) for b in buckets])
            detail = " ".join("b%d:%s(%s)" % (b, "on" if ok else "off", why) for b, (ok, why) in per)
        except Exception as e:  # never stop an engine start; the line says why
            detail = "decision error %s: %s" % (type(e).__name__, e)
        msgs.append("%sVLLM_TPU_HXS_AR0ST (whole step %s): %s" % (CONSTRUCT_PREFIX, tag, detail))
    msgs.append(smalls_armed_line(tag))  # q38 S3A-ENG T3: the three smalls' armed state, always (end of module)
    for m in msgs:
        logging.getLogger("vllm.vllm_torchtpu.hxsmall").info(m)
    return msgs


def _whole_ex(R, B, max_groups):
    """The ExecSpec fields `build_wholestep_call` plans bucket B with (R, B, T_VERIFY, ctx 128, `wholestep_tiles`)."""
    from vllm_torchtpu.megakernel.exec_spec import ExecSpec
    from vllm_torchtpu.megakernel_runtime.qwen38_segment_call import T_VERIFY
    from vllm_torchtpu.megakernel_runtime.qwen38_wholestep_call import wholestep_tiles

    return ExecSpec(R=R, B=B, T=T_VERIFY, ctx=128, tiles=wholestep_tiles(B, max_groups))


def ar0st_decide(spec, ex) -> tuple:
    """(engages, why) of AR0S-T for one planned kernel: the trace-time rule, i.e. `hc/block.py _ar0s_hook` (an HC sublayer
    with an all-reduce, an inner MoE block, a communicating ring with HIER 2 and WAR) and `moe/block.py`'s expert-path
    choice (not PIPE, not PIPE_EP, `ar0s_supported(mp)`), for every such sublayer of `spec` (all must agree)."""
    from vllm_torchtpu.megakernel import planner
    from vllm_torchtpu.megakernel.blocks.hc.block import HCSublayer
    from vllm_torchtpu.megakernel.blocks.hc.comm import hc_comm_params
    from vllm_torchtpu.megakernel.blocks.moe.block import MoEBlock
    from vllm_torchtpu.megakernel.blocks.moe.experts import ar0s_supported

    blks = [b for b in spec.all_blocks() if isinstance(b, HCSublayer) and b.allreduce and isinstance(b.inner, MoEBlock)]
    if not blks:
        return False, "no HC MoE sublayer with an all-reduce"
    plan = planner.plan(spec, ex, fit_vmem=False)
    out = set()
    for b in blks:
        cp = hc_comm_params(ex, b.h)
        mp = plan.block_params[b.name]
        path = "T13 %d T2 %d CH %d" % (mp.T13, mp.T2, mp.CH)
        if not (cp.R > 1 and "comm" not in ex.skip):
            out.add((False, "no comm"))
        elif cp.HIER != 2 or not cp.WAR:
            out.add((False, "hier %d war %d" % (cp.HIER, int(bool(cp.WAR)))))
        elif not ar0s_supported(mp):
            out.add((False, "%s PIPE %d PIPE_EP %d unroll %d" % (path, int(bool(mp.PIPE)), int(bool(mp.PIPE_EP)),
                                                               int(bool(mp.unroll)))))
        else:
            out.add((True, "hier 2 " + path))
    if len(out) != 1:
        return False, "sublayers disagree: " + "; ".join(sorted(w for _, w in out))
    return out.pop()


class _MuteThread(logging.Filter):
    def __init__(self, tid) -> None:
        super().__init__()
        self.tid = tid

    def filter(self, record) -> bool:
        return record.thread != self.tid


def _quiet(fn):
    """fn() with the log records this thread emits meanwhile dropped (every handler) and every module-level set / dict /
    list / bool / int of the loaded vllm_torchtpu modules restored afterwards (in place), so once-per-process lines that
    fn's callees mark as logged still print when their own code runs."""
    import sys
    import threading

    snap = []
    for name, m in list(sys.modules.items()):
        if m is None or not (name == "vllm_torchtpu" or name.startswith("vllm_torchtpu.")):
            continue
        d = vars(m)
        keep = {}
        for k, v in list(d.items()):
            if k.startswith("__"):
                continue
            t = type(v)
            if t in (set, dict, list):
                keep[k] = (v, t(v))
            elif t in (bool, int):
                keep[k] = (v, None)
        snap.append((d, keep))
    hs = {id(h): h for h in logging.root.handlers}
    for lg in list(logging.root.manager.loggerDict.values()):
        for h in getattr(lg, "handlers", ()):
            hs[id(h)] = h
    if logging.lastResort is not None:
        hs[id(logging.lastResort)] = logging.lastResort
    mute = _MuteThread(threading.get_ident())
    for h in hs.values():
        h.addFilter(mute)
    try:
        return fn()
    finally:
        for h in hs.values():
            h.removeFilter(mute)
        for d, keep in snap:
            for k in [k for k in d if not k.startswith("__") and k not in keep and type(d[k]) in (set, dict, list, bool, int)]:
                del d[k]
            for k, (v, c) in keep.items():
                if d.get(k) is not v:
                    d[k] = v
                if c is None:
                    continue
                if type(v) is list:
                    v[:] = c
                else:
                    v.clear()
                    v.update(c)


# q38 S3A-ENG T3 begin (HXSMALL bind twin; HIX-LEAD-6 rulings 2026-11-24 items 6 and 9, Q2). `construct_lines` (above; called from one line
# of Qwen38WholeStep.__init__, so every engine start, every rank, through logger.info) also logs, ALWAYS (keys on or off, so absence is
# visible), one line with the ARMED state (the resolved strict env, not engagement) of the three smalls that run outside the whole step:
#   'q38 HXSMALL armed (whole step <tag>): VLLM_TPU_HXS_LEAN_DRAFT=<0|1> VLLM_TPU_HXS_VERIFY_FUSE=<0|1> VLLM_TPU_HXS_RUNNER_HOIST=<0|1>
#    runner_hoist_env_gate=<ok|refused(<needs>, ...)>'
# runner_hoist_env_gate is what the ENV-ONLY legs of RUNNER_HOIST's own gate decide (models/vllm/qwen4_exp/inputs.py
# Qwen4ExpAdapter._hxs_i32, whose conditions stay as they are; the adapter legs "speculative records" and "no host PLE ids" are not env
# and are not in it), printed whether or not the key is set; <needs> are _hxs_i32's own names, in its order. Their engagement evidence
# stays their own load / import / first-host-call lines. A key that does not parse prints error(<type>) instead of raising (never stop
# an engine start at bind). The prefix is not CONSTRUCT_PREFIX, and the line names no AR0ST / GLUE_TRIM key, so the readers' construct
# and NEVER rows are unchanged. Appended after every existing line.
ARMED_PREFIX = "q38 HXSMALL armed "
ARMED_KEYS = ("VLLM_TPU_HXS_LEAN_DRAFT", "VLLM_TPU_HXS_VERIFY_FUSE", "VLLM_TPU_HXS_RUNNER_HOIST")


def runner_hoist_env_needs() -> list:
    """The env-only legs of Qwen4ExpAdapter._hxs_i32's RUNNER_HOIST gate that fail, named and ordered as there. Pure: the registry getters
    are called directly (no envs attribute cache), and nothing is logged."""
    import os

    from vllm_torchtpu import envs

    ev = envs.environment_variables
    return [n for n, ok in (
        ("VLLM_TPU_QWEN4_FUSED_PREP", bool(ev["VLLM_TPU_QWEN4_FUSED_PREP"]())),
        ("VLLM_TPU_QWEN4_FUSED_PREP_FUNCTIONAL", bool(ev["VLLM_TPU_QWEN4_FUSED_PREP_FUNCTIONAL"]())),
        ("VLLM_TPU_QWEN4_NGRAM_WINDOW=1", ev["VLLM_TPU_QWEN4_NGRAM_WINDOW"]() == "1"),
        ("no VLLM_TPU_QWEN4_FUSED_PREP_CHECK", not ev["VLLM_TPU_QWEN4_FUSED_PREP_CHECK"]()),
        ("no VLLM_TPU_L6_ADAPT_CARRY", os.environ.get("VLLM_TPU_L6_ADAPT_CARRY", "").strip() in ("", "0")),
    ) if not ok]


def smalls_armed_line(tag) -> str:
    """The armed line (module comment above); returns the message, which `construct_lines` logs with its other lines."""
    vals = []
    for k in ARMED_KEYS:
        try:
            vals.append("%s=%s" % (k, "1" if on(k) else "0"))
        except Exception as e:  # noqa: BLE001  never stop an engine start at bind
            vals.append("%s=error(%s)" % (k, type(e).__name__))
    try:
        needs = runner_hoist_env_needs()
        gate = "refused(%s)" % ", ".join(needs) if needs else "ok"
    except Exception as e:  # noqa: BLE001
        gate = "error(%s)" % type(e).__name__
    return "%s(whole step %s): %s runner_hoist_env_gate=%s" % (ARMED_PREFIX, tag, " ".join(vals), gate)
# q38 S3A-ENG T3 end


# q38 S3A-ENG T5 begin (AR0S-T per-rank used line; HIX-LEAD-6 2026-11-24, plan 6 item 3a, option b). With VLLM_TPU_HXS_AR0ST "1",
# Qwen38WholeStep.__init__ (every engine start, every rank, logger.info) logs ONE line saying, per bucket, whether the AR0S-T hook engages
# in the kernel the served path builds, and if not, why:
#   'q38 HXSMALL AR0ST used rank=<r> (whole step <tag>): b8=<1|0>(<why>) b16=<1|0>(<why>)'
# The decision is the hook's own: hc/block.py `_ar0s_hook` (the key, an HC sublayer with an all-reduce and an inner MoE block,
# `ar0s_ring_ok` on hc_comm_params(ex, h) with the ring's comm rule, megakernel/comm.py:47) and moe/block.py's expert path (no PREGATE PG,
# then `ar0s_supported`: not PIPE / PIPE_EP, unrolled, T13 > 1, CH split), planned with exactly the ExecSpec the served build uses
# (qwen38_wholestep.served_exec: callable_for's tiles). S2's construct line and ar0st_decide are unchanged. A failed decision prints
# error(<type>: <msg>), never raises. No line with the key unset / "0". Appended after every existing line.
AR0ST_USED_PREFIX = "q38 HXSMALL AR0ST used rank="


def ar0st_used(spec, ex) -> tuple:
    """(used, why) of the AR0S-T hook in one planned kernel (spec, ex), all HC MoE all-reduce sublayers agreeing. Pure."""
    from vllm_torchtpu.megakernel import planner
    from vllm_torchtpu.megakernel.blocks.hc.block import HCSublayer, ar0s_ring_ok
    from vllm_torchtpu.megakernel.blocks.hc.comm import hc_comm_params
    from vllm_torchtpu.megakernel.blocks.moe.block import MoEBlock
    from vllm_torchtpu.megakernel.blocks.moe.experts import ar0s_supported

    if not on("VLLM_TPU_HXS_AR0ST"):
        return False, "key unset"
    blks = [b for b in spec.all_blocks() if isinstance(b, HCSublayer) and b.allreduce and isinstance(b.inner, MoEBlock)]
    if not blks:
        return False, "no HC MoE sublayer with an all-reduce"
    plan = planner.plan(spec, ex, fit_vmem=False)
    out = set()
    for b in blks:
        cp = hc_comm_params(ex, b.h)
        ok, why = ar0s_ring_ok(cp.R > 1 and "comm" not in ex.skip, cp)
        mp = plan.block_params[b.name]
        path = "T13 %d T2 %d CH %d" % (mp.T13, mp.T2, mp.CH)
        if ok and getattr(mp, "PG", 0):
            ok, why = False, "PREGATE expert path (PG %d)" % int(mp.PG)
        elif ok and not ar0s_supported(mp):
            ok, why = False, "expert path %s PIPE %d PIPE_EP %d unroll %d" % (path, int(bool(mp.PIPE)), int(bool(mp.PIPE_EP)),
                                                                             int(bool(mp.unroll)))
        elif ok:
            why = "%s %s" % (why, path)
        out.add((ok, why))
    if len(out) != 1:
        return False, "sublayers disagree: " + "; ".join(sorted(w for _, w in out))
    return out.pop()


def ar0st_used_line(rank, tag, buckets, spec, R, max_groups, exec_for) -> str:
    """Log (logger.info, every call) and return the T5 line; exec_for(R, B, max_groups) is the served ExecSpec of bucket B."""
    try:
        if not on("VLLM_TPU_HXS_AR0ST"):
            return ""
        per = _quiet(lambda: [(int(b), ar0st_used(spec, exec_for(R, int(b), max_groups))) for b in buckets])
        detail = " ".join("b%d=%d(%s)" % (b, int(ok), why) for b, (ok, why) in per)
    except Exception as e:  # noqa: BLE001  never stop an engine start at bind
        detail = "error(%s: %s)" % (type(e).__name__, e)
    msg = "%s%d (whole step %s): %s" % (AR0ST_USED_PREFIX, int(rank), tag, detail)
    logging.getLogger("vllm.vllm_torchtpu.hxsmall").info(msg)
    return msg
# q38 S3A-ENG T5 end
