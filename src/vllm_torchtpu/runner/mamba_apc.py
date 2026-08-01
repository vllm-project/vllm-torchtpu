import itertools

import torch
from vllm.distributed import get_dcp_group, get_pcp_group
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_torchtpu.layers.vllm.custom_ops.mamba_state_copy_op import (
    ensure_op_built, mamba_state_copy)


def _pad_to_pow2(values: list[int]) -> list[int]:
    # Callers always pass a non-empty list (see the per-group guard below);
    # guard defensively so an empty input doesn't produce `[0]` and issue a
    # spurious self-copy at slot 0.
    if not values:
        return values
    target = 1 << (len(values) - 1).bit_length()
    return values + [0] * (target - len(values))


class MambaApcStateCopier:
    """Move align-mode Mamba state between block-table slots before forward."""

    def __init__(self, runner):
        self.runner = runner
        self.base_block_size = runner.cache_config.mamba_block_size
        kv_cache_config = runner.kv_cache_config
        fwd_ctx = runner.vllm_config.compilation_config.static_forward_context

        self.mamba_group_ids: list[int] = []
        self.group_states: list[list[torch.Tensor]] = []
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            if not isinstance(group.kv_cache_spec, MambaSpec):
                continue
            self.mamba_group_ids.append(gid)
            states: list[torch.Tensor] = []
            for layer_name in group.layer_names:
                states.extend(fwd_ctx[layer_name].kv_cache)
            self.group_states.append(states)

        # req_id -> block-table column currently holding that request's state.
        self._state_block_idx: dict[str, int] = {}

        # Build the pallas op eagerly so the first block-crossing forward step
        # doesn't pay the JAX trace + XLA compile stall on the critical path.
        ensure_op_built()

    def _state_block_size(self) -> int:
        try:
            cp_world_size = (get_dcp_group().world_size *
                             get_pcp_group().world_size)
        except Exception:
            cp_world_size = 1
        return self.base_block_size * cp_world_size

    def preprocess(self, scheduler_output) -> None:
        runner = self.runner
        if not runner._unified_kv_layout:
            return

        input_batch = runner.input_batch
        block_size = self._state_block_size()

        resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
        for req_id in itertools.chain(scheduler_output.finished_req_ids,
                                      scheduler_output.preempted_req_ids or (),
                                      resumed_req_ids):
            self._state_block_idx.pop(req_id, None)

        num_scheduled = scheduler_output.num_scheduled_tokens
        srcs: list[list[int]] = [[] for _ in self.mamba_group_ids]
        dsts: list[list[int]] = [[] for _ in self.mamba_group_ids]
        block_tables = [
            input_batch.block_table[gid].get_cpu_tensor()
            for gid in self.mamba_group_ids
        ]

        for row in range(input_batch.num_reqs):
            req_id = input_batch.req_ids[row]
            assert req_id is not None
            n_sched = num_scheduled.get(req_id, 0)
            if n_sched == 0:
                continue

            num_computed = int(input_batch.num_computed_tokens_cpu[row])
            prev_idx = self._state_block_idx.get(req_id)
            if prev_idx is None:
                prev_idx = (num_computed - 1) // block_size
            curr_idx = cdiv(num_computed + n_sched, block_size) - 1
            self._state_block_idx[req_id] = curr_idx
            if prev_idx == -1 or prev_idx == curr_idx:
                continue

            for g in range(len(self.mamba_group_ids)):
                srcs[g].append(int(block_tables[g][row, prev_idx]))
                dsts[g].append(int(block_tables[g][row, curr_idx]))

        # Guard per-group so an empty group 0 doesn't silently drop group 1's
        # copies if a future change makes the groups diverge.
        for g, states in enumerate(self.group_states):
            if not srcs[g]:
                continue
            # `_pad_to_pow2` pads with block id 0. vLLM's BlockPool reserves
            # block 0 as the null_block and never allocates it to a real
            # request (block_pool.py: `null_block.is_null = True`), so real
            # dsts are always > 0 and padding writes `state[0] = state[0]` —
            # a self-copy that avoids the JAX scatter's implementation-defined
            # duplicate-index semantics.
            src = torch.tensor(_pad_to_pow2(srcs[g]), dtype=torch.int32)
            dst = torch.tensor(_pad_to_pow2(dsts[g]), dtype=torch.int32)
            src_dev = src.to(runner.device)
            dst_dev = dst.to(runner.device)
            for state in states:
                mamba_state_copy(state, src_dev, dst_dev)
