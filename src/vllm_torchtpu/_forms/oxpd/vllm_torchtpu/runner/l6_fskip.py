# SPDX-License-Identifier: Apache-2.0
"""q38 L6 lever fskip (FENCE-SKIP, host cycle H2): record no input-staging
fence on a carried chunk that issued no upload since the previous fence point.

``TPUModelRunner._record_input_staging_fence`` records a ``torch.tpu.Event``
before every chunk's forward, and the next chunk's ``_prepare_inputs`` waits on
it (``_wait_input_staging_fence``, a no-op on None) before rewriting the
reusable host staging tensors. The event orders every upload enqueued before it
against host writes after the next wait. If no upload was issued since the
previous fence point (recorded or skipped), a new event orders nothing the
previous one did not, so the record (``Event.record`` blocks the main thread
until the materialize thread takes the snapshot) buys nothing.

Rule (at the record site, once per chunk, ``skip()``): skip iff the lever is on,
the chunk was carried by lever carry (runner/decode_carry.py: same batch, order,
token counts and bucket as the recorded step), and ``S.n == 0``. ``S.n`` counts
upload statements since the previous record-site visit and is reset there,
whether the lever is on or off, so switching the lever on mid-run starts from a
correct baseline. The count is conservative: it also counts uploads from fresh
temporaries (seed-copy, KV-copy and substitution indices, spec metadata), so the
rule never relies on classifying a source; over-counting only adds fences.

Counted sites (one mark per issued branch; tests/q38_l6/test_l6_fskip.py scans
every possibly non-blocking host-to-device call in src -- ``non_blocking=`` not
the literal False, or ``copy_(src, flag)`` -- and requires a mark that runs
whenever the upload does: a mark statement before it in an enclosing block, or
right after it in its own block with nothing in between that can leave the
block; else an exemption below):

* runner/tpu_runner.py: ``_prepare_inputs`` (input ids / positions / seq_lens,
  PCP request-major ids, logits indices, request distribution, the combined
  distribution), ``_upload_keyed`` / ``_upload_unless_unchanged`` (a real upload,
  not a reuse hit; covers decode_carry.reuse_upload and the sampling-parameter
  reuse), ``_build_mamba_state_indices``, the mamba seed copies, the KV block
  copies of ``_update_states``, ``_apply_async_token_substitution``,
  ``_postfwd_subst_indices``, ``_reset_read_offsets_for_new_requests`` and
  ``_padded_sampling_params``;
* runner/decode_carry.py ``upload``; runner/speculative_decoding_manager.py (spec
  metadata uploads on a cache miss, the draft ids on a cache hit when not
  deferred, the deferred draft-id upload); layers/core/attention_metadata.py
  (both block-table uploads); layers/core/pcp_sequence_layout.py (PCP logits
  indices); models/vllm/qwen4_exp/inputs.py (the whole token store, non-window
  mode); runner/sampling_copy_full.py (the full-length copy and vLLM's own slice
  copy it falls back to: batch refreshes); models/vllm/qwen4_exp/ple_offload.py
  (the PLE host-table lookups, uploaded from the forward: with
  VLLM_TPU_PLE_HOST_FAST_LOOKUP they come from a 2-deep reused host ring, which
  the stock fence also orders; the served decode role keeps PLE on HBM, so this
  costs no engagement there).

Exempt (scan list, exact counts): device-to-host copies (async output, spec
drafts, the PLE ids); the Qwen3.8 adapter's per-step window / context uploads
from fresh arrays (inputs.py; ``_host_upload`` keeps its sources alive for 8
calls, and the stock fence is waited only at the next prepare, so it never
protected a temporary's lifetime; with a skipped fence the device lag that
bounds those lifetimes is the per-step wait on the previous step's token copy in
``_modify_prev_results``, which a carried chunk always has); pooling-model inputs
(runner/tpu_pooling.py: pooling models never take the async-spec carried path).

Host-only: the forward, every compiled program and every input value are
unchanged; the next chunk simply finds no fence to wait on. No CHECK mode: what
the fence protects (an in-flight DMA reading host memory) has no host-side
equivalent to compare.

Engagement on the L6AB window line: ``fskip.skip`` (fence skipped),
``fskip.rec_up`` (recorded: an upload was counted), ``fskip.rec_nocarry``
(recorded: the chunk was not carried).

History: written by a duplicate copy of the l6_h2_cheap builder (marks in
tpu_runner.py) and completed by the continuing copy (the other modules' marks,
the scan, the harness), per the lead's ruling in
scratch/overflow/l6_h2_cheap_COORD.md (2026-11-14 05:16Z).
"""

from __future__ import annotations

from vllm_torchtpu.core import l6_ab as _l6_ab

LEVER = "fskip"


class S:
    """Per-process counters (one model runner per worker process)."""

    # Host-to-device uploads counted since the previous record site.
    n = 0
    # The current chunk was carried (decode_carry); set by _prepare_inputs.
    carried = False


def skip() -> bool:
    """Called once per chunk at the record site: whether to skip the fence.
    Resets the counters for the next chunk either way."""
    n, carried = S.n, S.carried
    S.n = 0
    S.carried = False
    if not (_l6_ab.ACTIVE and _l6_ab.on(LEVER)):
        return False
    if carried and n == 0:
        _l6_ab.note(LEVER, "skip")
        return True
    _l6_ab.note(LEVER, "rec_up" if n else "rec_nocarry")
    return False
