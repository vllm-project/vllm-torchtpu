# SPDX-License-Identifier: Apache-2.0
from typing import TYPE_CHECKING

import torch

from vllm_torchtpu import utils
from vllm_torchtpu.runner.padding import _get_padded_token_len

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


class DpLockstepMixin:
    """Data-parallel lockstep coordination."""

    def _dp_lockstep_enabled(self) -> bool:
        # EP combine reduces partial expert outputs across DP engines; those
        # ranks must enter the same collectives with matching token buckets.
        # Non-EP DP engines remain independently schedulable.
        return (utils.get_dp_size(self.parallel_config) > 1
                and self.parallel_config.enable_expert_parallel)

    def _count_input_chunks(
            self, scheduler_output: "SchedulerOutput") -> tuple[int, int]:
        """Match _prepare_inputs chunking without staging any tensors.

        Returns (num_chunks, max_chunk_reqs). The second value is the exact
        maximum request count of any chunk — a most-model-len chunk holds up
        to num_reqs_most_model_len requests, which can exceed
        num_reqs_max_model_len, so no batch-level clamp is a safe substitute.
        """
        if scheduler_output.total_num_scheduled_tokens == 0:
            return 0, 0

        num_reqs = self.input_batch.num_reqs
        start_index = 0
        num_chunks = 0
        max_chunk_reqs = 0
        while start_index < num_reqs:
            use_max_model_len = self.most_model_len is None
            for i in range(start_index, num_reqs):
                req_id = self.input_batch.req_ids[i]
                assert req_id is not None
                num_tokens = scheduler_output.num_scheduled_tokens[req_id]
                if (not use_max_model_len and self.most_model_len is not None
                        and num_tokens > self.most_model_len):
                    use_max_model_len = True

            if use_max_model_len:
                chunk_reqs = self.num_reqs_max_model_len
            else:
                assert self.num_reqs_most_model_len is not None
                chunk_reqs = self.num_reqs_most_model_len
            this_chunk_reqs = min(chunk_reqs, num_reqs - start_index)
            max_chunk_reqs = max(max_chunk_reqs, this_chunk_reqs)
            start_index += this_chunk_reqs
            num_chunks += 1
        return num_chunks, max_chunk_reqs

    def _dp_coordinated_step(
            self,
            local_num_tokens: int,
            local_num_chunks: int,
            local_num_reqs: int = 0) -> "tuple[int | None, int]":
        """Return the DP-wide padded token bucket and forward chunk count."""
        if not self._dp_lockstep_enabled():
            return None, local_num_chunks
        from vllm.distributed.parallel_state import get_dp_group

        # Token/chunk metadata is synchronized on vLLM's CPU DP group. The
        # model forward below receives this result explicitly as
        # num_tokens_across_dp, so set_forward_context does not run a second
        # DP synchronization.
        t = torch.tensor([
            int(local_num_tokens),
            int(local_num_chunks),
            int(local_num_reqs)
        ],
                         dtype=torch.int64,
                         device="cpu")
        torch.distributed.all_reduce(t,
                                     op=torch.distributed.ReduceOp.MAX,
                                     group=get_dp_group().cpu_group)
        bucket = _get_padded_token_len(self.num_tokens_paddings,
                                       int(t[0].item()))
        # Spec-decode draft forwards size themselves off the step's DP-wide
        # max per-chunk request count (see DFlashProposer), riding this same
        # all-reduce instead of paying the theoretical max bucket or a
        # second collective.
        self._dp_step_max_reqs = int(t[2].item())
        return bucket, int(t[1].item())

    def _run_dp_dummy_chunk(self, bucket: int) -> None:
        self._dummy_run(bucket,
                        self.num_reqs_max_model_len,
                        self.max_num_blocks_per_req,
                        use_max_model_len=True,
                        dp_lockstep=True)

    def _run_dp_idle_pairing(self, bucket: int, num_chunks: int) -> None:
        """Idle-rank EP-DP pairing: emit the same cross-DP collective stream a
        busy rank does — ALL target dummy forwards, then ALL draft dummy
        forwards.
        """
        # Set before any dummy fires: both _run_dp_dummy_chunk and the draft
        # run_dp_dummy_draft read _dp_target_bucket for the coordinated, equal-token shape.
        self._dp_target_bucket = bucket
        # Phase 1: all TARGET dummy forwards.
        for _ in range(num_chunks):
            self._run_dp_dummy_chunk(bucket)
        # Phase 2: all DRAFT dummy forwards (num_chunks * K).
        # use_eagle() covers every hidden-state drafter, dflash included.
        spec = self.speculative_config
        if num_chunks > 0 and spec is not None and spec.use_eagle():
            self.drafter.run_dp_dummy_draft(num_chunks)

    def _dp_num_tokens_across_dp(self, num_tokens: int) -> torch.Tensor | None:
        if not self._dp_lockstep_enabled():
            return None
        return torch.full((utils.get_dp_size(self.parallel_config), ),
                          num_tokens,
                          dtype=torch.int32,
                          device="cpu")
