# SPDX-License-Identifier: Apache-2.0
import os

from vllm.utils.math_utils import cdiv

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.runner.kv_connector_glue import _suspend_kv_transfer_group

logger = init_logger(__name__)


def _spec_warmup_all_token_ids(
        req_ids: list[str],
        num_computed_tokens: list[int]) -> dict[str, list[int]]:
    """Build complete synthetic token histories for async cached requests."""
    assert len(req_ids) == len(num_computed_tokens), (
        "spec-decode warmup request IDs and computed-token counts must align: "
        f"requests={len(req_ids)}, counts={len(num_computed_tokens)}")
    # The current sampled token is not computed yet, so the complete sequence
    # contains num_computed + 1 tokens at scheduler handoff.
    return {
        req_id: [0] * (num_computed + 1)
        for req_id, num_computed in zip(req_ids, num_computed_tokens)
    }


class SpecWarmupMixin:
    """Speculative-decoding graph precompilation."""

    def _warmup_pcp_mtp_prefill(self) -> None:
        """Warm the real PCP MTP K1 prefill proposal and nothing else.

        The real path may fail once while its draft forward is being cold
        compiled. Run one bounded retry during startup so no live request
        becomes the compiler probe.

        This lifecycle is intentionally separate from ``_warmup_spec_decode``:
        a prefill-only producer must never synthesize decode/verify work.
        """
        if os.environ.get("SPEC_WARMUP", "1") == "0":
            return
        if self.enforce_eager or not self._is_async_drafter:
            return
        P = min(int(self.num_tokens_paddings[0]), self.max_model_len)
        n0 = self.num_xla_graphs
        with _suspend_kv_transfer_group(), self._precompile_timed(
                "PCP MTP real prefill warmup"):
            if not self._warmup_one_pcp_mtp_prefill(P, quiet=True):
                if not self._warmup_one_pcp_mtp_prefill(P):
                    raise RuntimeError(
                        "PCP MTP prefill warmup failed after retry; refusing "
                        "to start a producer whose first live proposal would "
                        "hit the same failure")
        logger.info("PCP MTP prefill warmup compiled %d graphs",
                    self.num_xla_graphs - n0)

    def _warmup_one_pcp_mtp_prefill(self, P: int, quiet: bool = False) -> bool:
        """Run one synthetic PCP prefill through target sampling + K1 propose."""
        from vllm.sampling_params import SamplingParams
        from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput

        rid = "__pcp_mtp_warmup__"
        max_pos = min(self.max_model_len, P + 1)
        top = int(self.kv_cache_config.num_blocks)
        block_ids_per_group = []
        max_nblk = 0
        for group in self.kv_cache_config.kv_cache_groups:
            group_block_size = int(group.kv_cache_spec.block_size)
            group_num_blocks = min(cdiv(max_pos, group_block_size),
                                   cdiv(self.max_model_len, group_block_size))
            block_ids_per_group.append(list(range(top - group_num_blocks,
                                                  top)))
            max_nblk = max(max_nblk, group_num_blocks)
        if top <= max_nblk + 1:
            raise RuntimeError(
                "PCP MTP prefill warmup has not enough KV blocks for its "
                f"synthetic request: num_blocks={top}, required>{max_nblk + 1}"
            )

        attempt_succeeded = True
        try:
            request = NewRequestData(
                req_id=rid,
                prompt_token_ids=[0] * P,
                mm_features=[],
                sampling_params=SamplingParams(temperature=0.0),
                pooling_params=None,
                block_ids=tuple(block_ids_per_group),
                num_computed_tokens=0,
                lora_request=None,
            )
            scheduler_output = SchedulerOutput.make_empty()
            scheduler_output.scheduled_new_reqs = [request]
            scheduler_output.num_scheduled_tokens = {rid: P}
            scheduler_output.total_num_scheduled_tokens = P
            self.execute_model(scheduler_output)
            self.sample_tokens(None)
            self.take_draft_token_ids()
        except Exception:
            attempt_succeeded = False
            if quiet:
                logger.warning(
                    "PCP MTP prefill warmup P=%d first attempt failed; "
                    "retrying once.",
                    P,
                    exc_info=True)
            else:
                logger.exception("PCP MTP prefill warmup P=%d retry failed", P)
        finally:
            cleanup_succeeded = self._warmup_spec_decode_cleanup(rid)

        if not cleanup_succeeded:
            raise RuntimeError(
                "PCP MTP prefill warmup cleanup failed; refusing to retry or "
                "start with residual synthetic request state")
        return attempt_succeeded

    def _warmup_spec_decode(self) -> None:
        """Warm the real spec-decode dispatch so the first request doesn't recompile.

        The isolated AOT precompile (``drafter.precompile`` /
        ``_precompile_rejection_sampler``) emits *standalone* fused programs, but
        torch-tpu's DEFER_AND_FUSE fuses the per-step propose + verify + sampling
        ops into *combined* programs that depend on the live dispatch sequence —
        which an isolated precompile cannot reproduce. The only way to emit those
        exact fused programs is to run the real two-phase execute path once. We
        do that here on synthetic requests -- both greedy and non-greedy
        (do_sampling=True), since the two verify+sample paths fuse into distinct
        programs -- then fully tear each request down so serving starts from a
        clean batch. execute_model / sample_tokens
        branch internally on async_scheduling, so this same body warms whichever
        mode the process is configured for: in async it auto-routes through the
        real async dispatch (the unified device-seeded propose,
        _assemble_async_spec_substitution, cross-step subtract/extract), which
        an isolated precompile likewise can't
        reproduce. The contiguous prefill + >=2 decode steps form the async
        cross-step chain (step N reads the AsyncPreResults parked by step N-1).

        Bounded + value-insensitive: only shapes matter, so token/position VALUES
        are irrelevant. Wrapped so any failure degrades to "no warmup" (the first
        request just pays the bounded recompiles, as before) rather than breaking
        serving.
        """
        if os.environ.get("SPEC_WARMUP", "1") == "0":
            return  # escape hatch / A-B toggle
        if self.enforce_eager:
            return
        if not (self._is_async_drafter):
            return
        if self.input_batch.num_reqs != 0:
            logger.warning("skip spec-decode warmup: input_batch not empty")
            return

        n0 = self.num_xla_graphs
        # These SchedulerOutputs are built locally and therefore have no
        # scheduler-generated kv_connector_metadata. They must not start real
        # PD transfers or bind synthetic requests into the connector.
        with _suspend_kv_transfer_group(), self._precompile_timed(
                "spec-decode real warmup"):
            # (1) First-pass / first-decode shapes depend on the prompt length:
            # sweep one synthetic request per prompt-token bucket at nr=1. Each
            # runs the REAL two-phase propose+verify+sampling dispatch, so
            # torch-tpu emits the same fused programs serving will hit.
            for i, P in enumerate(self.num_tokens_paddings):
                # A single request's prompt can't exceed max_model_len; skip
                # token buckets above it (they only arise as batched totals,
                # covered by the num_reqs sweep below). Without this, a P >
                # max_model_len synthetic prompt overflows token_ids_cpu
                # (shape [.., max_model_len]) when max_num_batched_tokens >
                # max_model_len, e.g. small max_model_len configs.
                if int(P) > self.max_model_len:
                    continue
                # Greedy and non-greedy (do_sampling=True) verify+sample are
                # DISTINCT fused programs, so sweep both at every prompt bucket.
                for greedy in (True, False):
                    if not self._warmup_one_spec_request(
                            int(P), idx=i, quiet=True, greedy=greedy):
                        # First attempt cold-compiles the draft forward; a
                        # SymInt from that cold compile can leak into the
                        # dynamic=False gather wrapper on the first bucket. Retry
                        # once now that the forward is compiled and returns
                        # concrete-shaped tensors (this attempt logs loudly if it
                        # also fails).
                        self._warmup_one_spec_request(int(P),
                                                      idx=i,
                                                      greedy=greedy)
            # (2) The propose+verify+sampling fusions are also keyed on the
            # decode batch size (num_reqs). Only relevant when serving actually
            # batches (>1 concurrent request); sweep real multi-request batches
            # so nr>1 serving steps are cached too -- both greedy and non-greedy.
            if self.scheduler_config.max_num_seqs > 1:
                P_small = int(self.num_tokens_paddings[0])
                for R in self.num_reqs_paddings:
                    if int(R) > 1:
                        self._warmup_spec_batch(int(R), P_small, greedy=True)
                        self._warmup_spec_batch(int(R), P_small, greedy=False)
        logger.info("spec-decode warmup compiled %d graphs",
                    self.num_xla_graphs - n0)

    def _warmup_spec_batch(self, R: int, P: int, greedy: bool = True) -> None:
        """Warm an nr=R decode batch: prefill R synthetic requests then decode
        them together so the multi-request fused programs are compiled.

        greedy=False routes the non-greedy (do_sampling=True) verify+sample
        fusion, which is a distinct set of programs from the greedy path."""
        from vllm.sampling_params import SamplingParams
        from vllm.v1.core.sched.output import (CachedRequestData,
                                               NewRequestData, SchedulerOutput)

        K = self.speculative_config.num_speculative_tokens
        room = self.max_model_len - P
        n_decode = min(max(5, K + 2), max(0, room // (1 + K) - 1))
        max_pos = min(self.max_model_len, P + n_decode * (1 + K))
        # Per-group block counts at each group's own block size (see
        # _warmup_one_spec_request); the per-request stride uses the widest
        # group so every group's ranges stay disjoint across requests.
        group_nblks = []
        for group in self.kv_cache_config.kv_cache_groups:
            gbs = int(group.kv_cache_spec.block_size)
            group_nblks.append(
                min(cdiv(max_pos, gbs), cdiv(self.max_model_len, gbs)))
        nblk = max(group_nblks)
        top = int(self.kv_cache_config.num_blocks)
        if top <= R * nblk + 1:
            logger.warning(
                "skip spec-decode warmup R=%d: not enough kv blocks", R)
            return
        rids = [f"__spec_warmup_b{R}_{j}__" for j in range(R)]
        sp = SamplingParams(temperature=0.0 if greedy else 1.0)
        try:
            # ---- batched prefill of R requests (disjoint top-of-pool blocks) --
            new_reqs = []
            for j, rid in enumerate(rids):
                lo = top - (j + 1) * nblk
                new_reqs.append(
                    NewRequestData(req_id=rid,
                                   prompt_token_ids=[0] * P,
                                   mm_features=[],
                                   sampling_params=sp,
                                   pooling_params=None,
                                   block_ids=tuple(
                                       list(range(lo, lo + g_nblk))
                                       for g_nblk in group_nblks),
                                   num_computed_tokens=0,
                                   lora_request=None))
            # Prefill in chunks that respect the pre-allocated host token buffers.
            # A single R*P prefill overflows positions_np ([max_num_tokens]) when
            # Decode below schedules R*(1+K) <= max_num_tokens, so only
            # the prefill needs chunking; the nr=R decode shape is unchanged.
            reqs_per_chunk = max(1, self.max_num_tokens // P)
            for c0 in range(0, R, reqs_per_chunk):
                chunk = new_reqs[c0:c0 + reqs_per_chunk]
                so = SchedulerOutput.make_empty()
                so.scheduled_new_reqs = chunk
                so.num_scheduled_tokens = {r.req_id: P for r in chunk}
                so.total_num_scheduled_tokens = len(chunk) * P
                assert self.execute_model(so) is None
                self.sample_tokens(None)
                self.take_draft_token_ids()
            # ---- decode all R together: real nr=R propose + verify + sample ---
            nct = [P] * R
            for t in range(n_decode):
                all_token_ids = _spec_warmup_all_token_ids(rids, nct)
                creq = CachedRequestData(req_ids=list(rids),
                                         resumed_req_ids=set(),
                                         new_token_ids=[],
                                         all_token_ids=all_token_ids,
                                         new_block_ids=[None] * R,
                                         num_computed_tokens=list(nct),
                                         num_output_tokens=[t + 1] * R)
                so = SchedulerOutput.make_empty()
                so.scheduled_cached_reqs = creq
                so.num_scheduled_tokens = {rid: 1 + K for rid in rids}
                so.total_num_scheduled_tokens = R * (1 + K)
                so.scheduled_spec_decode_tokens = {
                    rid: [0] * K
                    for rid in rids
                }
                assert self.execute_model(so) is None
                self.sample_tokens(None)
                self.take_draft_token_ids()
                nct = [x + 1 for x in nct]
        except Exception:
            logger.exception("spec-decode warmup R=%d failed; continuing", R)
        finally:
            self._warmup_spec_decode_cleanup(rids)

    def _warmup_one_spec_request(self,
                                 P: int,
                                 idx: int,
                                 quiet: bool = False,
                                 greedy: bool = True) -> bool:
        """Run one synthetic spec-decode request (prefill P + decodes).

        Returns True on success, False if the warmup raised (and was caught).
        The caller retries a failed bucket once: the first attempt cold-compiles
        the draft forward, and a SymInt from that cold compile can leak into the
        dynamic=False gather wrapper on the very first bucket; the retry runs with
        the forward already compiled (returning concrete-shaped tensors).
        """
        from vllm.sampling_params import SamplingParams
        from vllm.v1.core.sched.output import (CachedRequestData,
                                               NewRequestData, SchedulerOutput)

        rid = f"__spec_warmup_{idx}__"
        K = self.speculative_config.num_speculative_tokens
        # The contexts (num_computed_tokens) to run decode steps at: a short walk
        # near the prefill.
        room = self.max_model_len - P
        n_dec = min(max(5, K + 2), max(0, room // (1 + K) - 1))
        decode_ncts = [P + t for t in range(n_dec)]
        # Blocks from the TOP of the pool so the first real allocations (low ids)
        # don't collide. Page CONTENTS are value-insensitive and KV writes use
        # empty_slot_mappings, so stale pages are safe; each request is fully torn
        # down before the next. Cover the highest context this request reaches.
        # Count blocks per kv-cache group at that group's own block size:
        # groups differ on hybrid models (mamba vs attention), and the
        # runner-level self.block_size predates the executor's post-load
        # block-size finalization.
        max_pos = min(self.max_model_len, max([P] + decode_ncts) + (1 + K))
        top = int(self.kv_cache_config.num_blocks)
        block_ids_per_group = []
        max_nblk = 0
        for group in self.kv_cache_config.kv_cache_groups:
            gbs = int(group.kv_cache_spec.block_size)
            g_nblk = min(cdiv(max_pos, gbs), cdiv(self.max_model_len, gbs))
            block_ids_per_group.append(list(range(top - g_nblk, top)))
            max_nblk = max(max_nblk, g_nblk)
        if top <= max_nblk + 1:
            logger.warning(
                "skip spec-decode warmup P=%d: not enough kv blocks", P)
            return
        block_ids = tuple(block_ids_per_group)
        sp = SamplingParams(temperature=0.0 if greedy else 1.0)
        try:
            # ---- prefill ----
            new_req = NewRequestData(req_id=rid,
                                     prompt_token_ids=[0] * P,
                                     mm_features=[],
                                     sampling_params=sp,
                                     pooling_params=None,
                                     block_ids=block_ids,
                                     num_computed_tokens=0,
                                     lora_request=None)
            so = SchedulerOutput.make_empty()
            so.scheduled_new_reqs = [new_req]
            so.num_scheduled_tokens = {rid: P}
            so.total_num_scheduled_tokens = P
            assert self.execute_model(so) is None
            self.sample_tokens(None)
            d = self.take_draft_token_ids()
            cur = d.draft_token_ids[0] if d and d.draft_token_ids else [0] * K
            # ---- decode steps (real propose + verify + sampling) ----
            for t, nct in enumerate(decode_ncts):
                all_token_ids = _spec_warmup_all_token_ids([rid], [nct])
                creq = CachedRequestData(req_ids=[rid],
                                         resumed_req_ids=set(),
                                         new_token_ids=[],
                                         all_token_ids=all_token_ids,
                                         new_block_ids=[None],
                                         num_computed_tokens=[nct],
                                         num_output_tokens=[t + 1])
                so = SchedulerOutput.make_empty()
                so.scheduled_cached_reqs = creq
                so.num_scheduled_tokens = {rid: 1 + K}
                so.total_num_scheduled_tokens = 1 + K
                so.scheduled_spec_decode_tokens = {
                    rid: [int(x) for x in list(cur)[:K]]
                }
                assert self.execute_model(so) is None
                self.sample_tokens(None)
                d = self.take_draft_token_ids()
                if d and d.draft_token_ids:
                    cur = d.draft_token_ids[0]
        except Exception:
            if quiet:
                # First attempt: a SymInt from the draft forward's cold compile
                # can leak into the dynamic=False gather wrapper. The caller
                # retries once (forward now compiled -> concrete shapes), so log
                # quietly and let the retry surface a genuine failure.
                # WARNING, not DEBUG: this except catches a *rank-local*
                # exception raised inside a *collective* region. The raising
                # rank unwinds and retries while its peers stay blocked in
                # the collective, so a swallowed failure here surfaces as a
                # cluster-wide hang with no log line to explain it. Keep it
                # visible even on the quiet first attempt.
                logger.warning(
                    "spec-decode warmup P=%d first attempt failed; retrying. "
                    "Under DP/EP this can desynchronize ranks — if startup "
                    "hangs after this line, that is the cause.",
                    P,
                    exc_info=True)
            else:
                logger.exception("spec-decode warmup P=%d failed; continuing",
                                 P)
            return False
        finally:
            self._warmup_spec_decode_cleanup(rid)
        return True

    def _warmup_spec_decode_cleanup(self, rids) -> bool:
        """Tear down the synthetic warmup request(s) so serving starts clean."""
        from vllm.v1.core.sched.output import SchedulerOutput
        cleanup_succeeded = True
        if isinstance(rids, str):
            rids = [rids]
        # Reset the two-phase guard: a half-finished step would otherwise make
        # the first real execute_model raise the "State error".
        self.execute_model_state = None
        # The last async warmup step parks an AsyncPreResults (in-flight D2H copy
        # + optimistic 1+K placeholders) with no follower step to drain it.
        # Finish the copy and null it BEFORE the drain below so the drain's
        # _flush_disjoint_async_results early-returns instead of issuing a stray
        # D2H / placeholder rollback on the torn-down synthetic request, and no
        # in-flight copy host buffer is freed mid-transfer (defensive: covers the
        # `if present:`-skip path). No-op in sync (always None).
        if self._pre_async_results is not None:
            try:
                self._pre_async_results.wait_for_copy()
            except Exception:
                cleanup_succeeded = False
                logger.exception("spec-decode warmup async copy wait failed")
            self._pre_async_results = None
        try:
            idx_map = self.input_batch.req_id_to_index
            present = {r for r in rids if r in self.requests or r in idx_map}
            if present:
                so = SchedulerOutput.make_empty()
                so.finished_req_ids = set(present)
                # total==0 -> _update_states removes + condenses, then early
                # return (no forward / sample_tokens needed).
                self.execute_model(so)
                self.execute_model_state = None
        except Exception:
            cleanup_succeeded = False
            logger.exception("spec-decode warmup drain failed")
        # Belt-and-suspenders: drop any synthetic state defensively.
        for r in rids:
            try:
                self.requests.pop(r, None)
            except Exception:
                cleanup_succeeded = False
                logger.exception("spec-decode warmup request cleanup failed")
        for attr in ("_draft_token_ids", ):
            try:
                if hasattr(self.spec_decode_manager, attr):
                    setattr(self.spec_decode_manager, attr, None)
            except Exception:
                cleanup_succeeded = False
                logger.exception("spec-decode warmup draft cleanup failed")
        try:
            if self.drafter is not None:
                self.drafter.draft_chunks = None
        except Exception:
            cleanup_succeeded = False
            logger.exception("spec-decode warmup chunk cleanup failed")
        self._pre_async_results = None
        self.mm_embed_inputs = None
        if self.input_batch.num_reqs != 0:
            cleanup_succeeded = False
            logger.warning("spec-decode warmup left batch non-empty (%d)",
                           self.input_batch.num_reqs)
        remaining = {
            r
            for r in rids
            if r in self.requests or r in self.input_batch.req_id_to_index
        }
        if remaining:
            cleanup_succeeded = False
            logger.warning("spec-decode warmup state still contains %s",
                           sorted(remaining))
        return cleanup_succeeded
