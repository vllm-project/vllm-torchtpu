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
"""Post-scheduler HLO pass: overlap the DSv4 chip-pair q send with KV work.

Under TPU_DSV4_ATTN_INTRA_CHIP_TP the q swap is a Pallas start and wait
(``q_send_start`` / ``q_send_wait``, kernels/deepseek_v4/
q_send.py) inside one torch op: their DMA semaphores cannot be
passed between torch ops, so nothing at the torch level can put the layer's
indexer or compressor between them, and XLA's scheduler sees two ordinary
custom calls with no reason to separate them. It also runs each layer's KV
kernels before the q projection, so the transfer would have nothing to overlap.

This pass runs after XLA's scheduler and before memory-space assignment. For
each start it hoists the start, and the part of its operand chain (q latent
gather, wq_b, relayout) scheduled after the layer's KV kernels, to just before
them; the wait stays put. Per layer type:

  * CSA layers (they have an indexer): before the indexer's StreamIdx kernels,
    keeping the projection's lifetime short;
  * HCA layers: before the compressor's project + compress kernels;
  * layers with neither (SWA only) are left alone.

A move is made only if everything hoisted depends on nothing at or after the
insertion point other than what is hoisted with it; the schedule is then
updated and verified. On any failure the module is returned unchanged.
"""

import re

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

START = "q_send_start"
WAIT = "q_send_wait"
INDEXER = re.compile(r"^StreamIdx")
COMPRESSOR = re.compile(r"^(proj_and_save_state|compress_norm_rope_store)")
# Look back at most this many instructions for a layer's KV kernels.
MAX_SCAN = 4000
_OP_NAME = re.compile(r'op_name="([^"]*)"')


def _name(inst) -> str:
    """The instruction's (unique) HLO name."""
    n = getattr(inst, "name", None)
    if callable(n):
        n = n()
    if n is None:
        n = str(inst).split(" = ", 1)[0].strip().lstrip("%")
    return n


def _label(inst) -> str:
    """The name to match on. Before XLA renames them, Pallas custom calls are
    ``pallas_call.N``; the kernel name is in their op_name metadata
    (``.../q_send_start/pallas_call`` -> ``q_send_start``)."""
    n = _name(inst)
    if n.startswith((START, WAIT)) or INDEXER.match(n) or COMPRESSOR.match(n):
        return n
    m = _OP_NAME.search(inst.to_string())
    if not m:
        return n
    parts = [p for p in m.group(1).split("/") if p and p != "pallas_call"]
    return parts[-1] if parts else n


def _is_custom_call(inst) -> bool:
    # get-tuple-elements of a custom call carry its op_name too.
    return " custom-call(" in inst.to_string()[:2000]


def count_starts(seq) -> int:
    """The number of q-send starts in one scheduled sequence."""
    return sum(1 for x in seq if _label(x).startswith(START) and _is_custom_call(x))


def _report(found: int, moved: int, layers: int) -> None:
    """Log the outcome; finding starts but moving none means the overlap is
    lost, usually because the KV kernels this pass matches were renamed."""
    if moved:
        logger.info("q_send_reorder: moved %d instructions in %d layers", moved, layers)
    elif found:
        logger.warning(
            "q_send_reorder: found %d q-send starts but moved none; the q send "
            "runs without overlap (check the indexer / compressor kernel names "
            "this pass matches)",
            found,
        )


def reorder_sequence(seq):
    """Hoists each q-send start above its layer's KV kernels in one scheduled
    sequence. Returns ``(new_sequence, instructions_moved, layers_moved)``."""
    labels = [_label(x) for x in seq]
    # Identity by HLO name: the HLO API returns fresh wrapper objects per call.
    pos = {_name(x): i for i, x in enumerate(seq)}
    starts = [
        i
        for i, n in enumerate(labels)
        if n.startswith(START) and _is_custom_call(seq[i])
    ]
    waits = [
        i
        for i, n in enumerate(labels)
        if n.startswith(WAIT) and _is_custom_call(seq[i])
    ]
    order = list(range(len(seq)))
    moved, layers = 0, 0
    for s in starts:
        prev_wait = max([w for w in waits if w < s], default=-1)
        lo = max(prev_wait + 1, s - MAX_SCAN)
        kernels = [i for i in range(lo, s) if INDEXER.match(labels[i])] or [
            i for i in range(lo, s) if COMPRESSOR.match(labels[i])
        ]
        if not kernels:
            continue
        # The start's operand chain within [lo, s).
        chain, todo = set(), [s]
        while todo:
            i = todo.pop()
            if i in chain:
                continue
            chain.add(i)
            for o in seq[i].operands():
                j = pos.get(_name(o), -1)
                if lo <= j < s:
                    todo.append(j)
        kernels = [i for i in kernels if i not in chain]  # ones the start needs stay
        if not kernels:
            continue
        k0 = min(kernels)
        hoist = sorted(i for i in chain if i > k0)
        hoisted = set(hoist)
        legal = all(
            not (pos.get(_name(o), -1) >= k0 and pos.get(_name(o), -1) not in hoisted)
            for i in hoist
            for o in seq[i].operands()
        )
        if not hoist or not legal:
            continue
        keep = [i for i in order if i not in hoisted]
        at = keep.index(k0)
        order = keep[:at] + hoist + keep[at:]
        moved += len(hoist)
        layers += 1
    if not moved:
        return seq, 0, 0
    return [seq[i] for i in order], moved, layers


def transform(serialized: bytes) -> bytes | None:
    """The registered transformation: serialized HloModuleProto in, out."""
    from jax._src.lib import hlo as _hlo

    try:
        module = _hlo.HloModule.from_serialized_hlo_module_proto(serialized)
        sched = module.schedule()
        if sched is None:
            return None
        found, moved, layers = 0, 0, 0
        for comp in module.make_nonfusion_computations():
            seq = list(sched.sequence(comp))
            found += count_starts(seq)
            new, m, n = reorder_sequence(seq)
            if m:
                sched.set_sequence(comp, new)
                moved, layers = moved + m, layers + n
        if not moved:
            _report(found, 0, 0)
            return None
        sched.update()
        sched.verify()
        module.set_schedule(sched)
        _report(found, moved, layers)
        return module.as_serialized_hlo_module_proto()
    except Exception as e:  # noqa: BLE001  never break compilation
        logger.warning("q_send_reorder: left the module unchanged: %r", e)
        return None


_registered = False


def register_once() -> None:
    """Registers the pass for TPU compiles, once per process (each vLLM worker
    compiles its own programs)."""
    global _registered
    if _registered:
        return
    _registered = True  # try once per process; warn once on failure
    try:
        import jax.extend.xla as jex_xla

        jex_xla.register_hlo_module_transformation(
            transform,
            name="dsv4_q_send_reorder",
            stage=jex_xla.PipelineStage.POST_SCHEDULER,
            platforms="tpu",
        )
    except Exception as e:  # noqa: BLE001  the q send is correct without the pass
        logger.warning(
            "q_send_reorder: could not register the HLO pass (%r); the q send "
            "runs without overlap",
            e,
        )
