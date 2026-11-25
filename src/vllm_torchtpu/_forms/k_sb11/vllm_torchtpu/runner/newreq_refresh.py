# SPDX-License-Identifier: Apache-2.0
"""Skips the unread sampling-metadata rebuild on the TPU runner's batch (VLLM_TPU_NEWREQ_REFRESH_SKIP; q38 SV-NEWREQ2 S2).

On every step ``GPUModelRunner._update_states`` ends with ``InputBatch.refresh_metadata()`` (vLLM 0.31
gpu_input_batch.py:834-851). After an admission, finish or reorder it applies the batch update to the logits
processors and rebuilds ``self.sampling_metadata`` with ``_make_sampling_metadata()``: a ``SamplingMetadata`` object,
device copies of the temperature / top-p / top-k / penalty slices when requests are not greedy, and (with penalties or
token-id logits processors) the padded prompt-token tensor and its H2D copy.

Nothing on the TPU path reads that object or the device slices it fills. The TPU runner overrides ``sample_tokens``,
and reads only the host tensors ``temperature_cpu_tensor`` / ``top_k_cpu_tensor`` / ``top_p_cpu_tensor`` (written by
``add_request``), ``all_greedy``, ``max_num_logprobs`` and ``generators``; vLLM's readers of ``sampling_metadata``
(``GPUModelRunner._sample`` / ``sample_tokens`` / ``_copy_draft_token_ids_to_cpu`` and the InputBatch async helpers
they call) are reached only from vLLM's own ``sample_tokens``.

With the key on, ``refresh_metadata`` on a registered TPU runner's current ``input_batch`` runs vLLM's body without
the rebuild: ``batch_update_builder.get_and_reset``, the thinking-budget ``sync_batch`` and every logits processor's
``update_state`` are kept, in the same order. Any other batch (not a registered TPU runner's ``input_batch``, a
pooling model's batch) and every call in an A/B OFF window takes vLLM's function unchanged.

``VLLM_TPU_NEWREQ_REFRESH_SKIP=1`` turns it on (default off: nothing is patched).
``VLLM_TPU_NEWREQ_REFRESH_SKIP_AB_MS=<P>`` (0 = off): in-process A/B in wall-clock windows, the skip in even
``time_ns // P ms`` windows and vLLM's rebuild in odd ones; P must divide 86,400,000 with an even quotient. Both keys
are host-only and excluded from the compile key. Installed from ``TPUModelRunner.__init__`` (which registers the
runner); refused loudly (RuntimeError) if vLLM's ``refresh_metadata`` or ``_make_sampling_metadata`` differ from the
sources this copy was written against.

Runtime stats line, from the thread that refreshes (the model thread), every LOG_EVERY calls on a registered batch:
``NEWREQ_REFRESH_SKIP skipped=.. kept_lp_updates=.. idle=.. ab_off=.. stock=.. calls=.. rank=..``
(skipped: batch changes whose rebuild was skipped; kept_lp_updates: logits-processor ``update_state`` calls made in ON
windows; idle: ON-window calls without a batch change, where vLLM would not rebuild either; ab_off: calls passed to
vLLM's function in OFF windows; stock: calls on other batches). An A/B window line
``NEWREQ_REFRESH_SKIP AB window=.. on=..`` is logged at each window change.
"""

import hashlib
import inspect
import time
import weakref

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

DAY_MS = 86_400_000
# One call per step (about 40 a second at B1 decode), so about one line a minute per rank.
LOG_EVERY = 2500
_PATCH_FLAG = "_tpu_newreq_refresh_patch"
# sha256 of the normalized sources (blank lines dropped, lines right-stripped, "\0" after each function) of vLLM 0.31's
# InputBatch.refresh_metadata (patched) and InputBatch._make_sampling_metadata (the skipped rebuild).
REFRESH_SHA256 = "05e8cc28ca7e94d53806375c6bb209c2390752002d2251fb75706ccabc8f85f3"

ENABLED = False
AB_MS = 0
_state = {"ab_window": None, "runners": []}


def _new_stats() -> dict:
    return {"calls": 0, "skipped": 0, "kept_lp_updates": 0, "idle": 0, "ab_off": 0, "stock": 0}


stats = _new_stats()


def _load() -> None:
    global ENABLED, AB_MS
    ENABLED = envs.VLLM_TPU_NEWREQ_REFRESH_SKIP
    AB_MS = envs.VLLM_TPU_NEWREQ_REFRESH_SKIP_AB_MS
    if AB_MS and (DAY_MS % AB_MS or (DAY_MS // AB_MS) % 2):
        raise ValueError(
            f"Invalid value '{AB_MS}' for VLLM_TPU_NEWREQ_REFRESH_SKIP_AB_MS: must be 0 or divide {DAY_MS} with an "
            "even quotient."
        )


_load()


def enabled() -> bool:
    return bool(ENABLED)


def _now_ns() -> int:
    return time.time_ns()


def ab_on() -> bool:
    """Whether the skip is on in the current A/B window (always, without AB)."""
    if AB_MS <= 0:
        return True
    k = _now_ns() // (AB_MS * 1_000_000)
    if k != _state["ab_window"]:
        _state["ab_window"] = k
        logger.info(
            "NEWREQ_REFRESH_SKIP AB window=%d on=%d skipped=%d ab_off=%d calls=%d",
            k, int(k % 2 == 0), stats["skipped"], stats["ab_off"], stats["calls"],
        )
    return k % 2 == 0


def source_sha256(*fns) -> str:
    h = hashlib.sha256()
    for fn in fns:
        src = inspect.getsource(inspect.unwrap(fn))
        h.update("\n".join(line.rstrip() for line in src.splitlines() if line.strip()).encode())
        h.update(b"\0")
    return h.hexdigest()


def _register(runner) -> None:
    refs = [r for r in _state["runners"] if r() is not None]
    if not any(r() is runner for r in refs):
        try:
            refs.append(weakref.ref(runner))
        except TypeError:
            refs.append(lambda: runner)
    _state["runners"] = refs


def _owner(batch):
    """The registered TPU runner whose current input_batch is ``batch``, else None."""
    for ref in _state["runners"]:
        runner = ref()
        if runner is not None and getattr(runner, "input_batch", None) is batch:
            return runner
    return None


def _tick(runner) -> None:
    s = stats
    s["calls"] += 1
    if s["calls"] % LOG_EVERY == 0:
        # The runtime stats line (the model thread, which refreshes the batch).
        logger.info(
            "NEWREQ_REFRESH_SKIP skipped=%d kept_lp_updates=%d idle=%d ab_off=%d stock=%d calls=%d rank=%s",
            s["skipped"], s["kept_lp_updates"], s["idle"], s["ab_off"], s["stock"], s["calls"],
            getattr(runner, "_profiler_rank", None),
        )


def make_refresh_metadata(stock):
    """vLLM 0.31's ``InputBatch.refresh_metadata`` without the sampling-metadata rebuild on a TPU runner's batch."""

    def refresh_metadata(self):
        runner = None if self.is_pooling_model else _owner(self)
        if runner is None:
            stats["stock"] += 1
            return stock(self)
        if not ab_on():
            stats["ab_off"] += 1
            out = stock(self)
            _tick(runner)
            return out
        # gpu_input_batch.py:844-851 for a non-pooling batch, without `self.sampling_metadata =
        # self._make_sampling_metadata()`: nothing on the TPU path reads it.
        batch_update = self.batch_update_builder.get_and_reset(self.num_reqs)
        if self.thinking_budget_state_holder is not None and batch_update:
            self.thinking_budget_state_holder.sync_batch(batch_update)
        n = 0
        for logit_proc in self.logitsprocs.all:
            logit_proc.update_state(batch_update)
            n += 1
        stats["kept_lp_updates"] += n
        if batch_update:
            stats["skipped"] += 1
        else:
            stats["idle"] += 1
        _tick(runner)

    setattr(refresh_metadata, _PATCH_FLAG, True)
    refresh_metadata.__wrapped__ = stock
    return refresh_metadata


def patch_refresh_skip(runner=None) -> bool:
    """TPUModelRunner.__init__; a no-op (nothing imported, nothing patched) unless VLLM_TPU_NEWREQ_REFRESH_SKIP is set."""
    if not enabled():
        return False
    from vllm.v1.worker.gpu_input_batch import InputBatch

    cur = InputBatch.refresh_metadata
    if not getattr(cur, _PATCH_FLAG, False):
        sha = source_sha256(cur, InputBatch._make_sampling_metadata)
        if sha != REFRESH_SHA256:
            raise RuntimeError(
                "VLLM_TPU_NEWREQ_REFRESH_SKIP refused: vLLM's InputBatch.refresh_metadata / _make_sampling_metadata "
                "(%s, want %s) differ from the sources this copy was written against" % (sha[:12], REFRESH_SHA256[:12])
            )
        InputBatch.refresh_metadata = make_refresh_metadata(cur)
        logger.info(
            "VLLM_TPU_NEWREQ_REFRESH_SKIP=1: the TPU runner's batch refresh keeps the logits-processor updates and "
            "skips the unread sampling-metadata rebuild (AB %d ms; stats every %d refreshes)",
            AB_MS, LOG_EVERY,
        )
    if runner is not None:
        _register(runner)
    return True
