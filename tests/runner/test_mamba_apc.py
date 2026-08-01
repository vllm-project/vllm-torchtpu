from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

import vllm_torchtpu.runner.mamba_apc as mamba_apc_mod
from vllm_torchtpu.runner.mamba_apc import MambaApcStateCopier, _pad_to_pow2

BLOCK_SIZE = 256
NUM_BLOCKS = 32
STATE_DIM = 8


def _make_runner(num_reqs,
                 req_ids,
                 num_computed,
                 block_rows,
                 extra_mamba_group=False,
                 extra_block_rows=None):
    conv = torch.arange(NUM_BLOCKS, dtype=torch.float32).unsqueeze(1).repeat(
        1, STATE_DIM).clone()
    ssm = conv.clone() * 100.0

    block_table_cpu = torch.zeros((16, 8), dtype=torch.int32)
    for row, blocks in enumerate(block_rows):
        block_table_cpu[row, :len(blocks)] = torch.tensor(blocks,
                                                          dtype=torch.int32)

    mamba_table = SimpleNamespace(get_cpu_tensor=lambda: block_table_cpu)
    attn_table = SimpleNamespace(
        get_cpu_tensor=lambda: torch.zeros((16, 8), dtype=torch.int32))

    block_tables_list = [attn_table, mamba_table]

    input_batch = SimpleNamespace(
        num_reqs=num_reqs,
        req_ids=req_ids,
        num_computed_tokens_cpu=np.array(num_computed + [0] *
                                         (16 - len(num_computed))),
        block_table=block_tables_list,
    )

    attn_group = SimpleNamespace(kv_cache_spec=SimpleNamespace(),
                                 layer_names=["layers.0.attn"])
    from vllm.v1.kv_cache_interface import MambaSpec
    mamba_spec = MambaSpec(shapes=((STATE_DIM, ), (STATE_DIM, )),
                           dtypes=(torch.float32, torch.float32),
                           block_size=BLOCK_SIZE)
    mamba_group = SimpleNamespace(kv_cache_spec=mamba_spec,
                                  layer_names=["layers.1.linear_attn"])

    layer = MagicMock()
    layer.kv_cache = (conv, ssm)
    static_ctx = {"layers.1.linear_attn": layer}
    kv_cache_groups = [attn_group, mamba_group]

    extras: dict = {}
    if extra_mamba_group:
        conv2 = (torch.arange(NUM_BLOCKS,
                              dtype=torch.float32).unsqueeze(1).repeat(
                                  1, STATE_DIM).clone()) * 1000.0
        block_table2_cpu = torch.zeros((16, 8), dtype=torch.int32)
        for row, blocks in enumerate(extra_block_rows or block_rows):
            block_table2_cpu[row, :len(blocks)] = torch.tensor(
                blocks, dtype=torch.int32)
        mamba_table2 = SimpleNamespace(get_cpu_tensor=lambda: block_table2_cpu)
        block_tables_list.append(mamba_table2)
        mamba_group2 = SimpleNamespace(kv_cache_spec=mamba_spec,
                                       layer_names=["layers.2.linear_attn"])
        kv_cache_groups.append(mamba_group2)
        layer2 = MagicMock()
        layer2.kv_cache = (conv2, )
        static_ctx["layers.2.linear_attn"] = layer2
        extras["conv2"] = conv2

    runner = SimpleNamespace(
        cache_config=SimpleNamespace(mamba_block_size=BLOCK_SIZE),
        kv_cache_config=SimpleNamespace(kv_cache_groups=kv_cache_groups),
        vllm_config=SimpleNamespace(compilation_config=SimpleNamespace(
            static_forward_context=static_ctx)),
        input_batch=input_batch,
        device="cpu",
        _unified_kv_layout=True,
    )
    if extras:
        return runner, conv, ssm, block_table_cpu, extras
    return runner, conv, ssm, block_table_cpu


def _sched_output(num_scheduled, finished=(), preempted=None, resumed=()):
    return SimpleNamespace(
        num_scheduled_tokens=num_scheduled,
        finished_req_ids=set(finished),
        preempted_req_ids=preempted,
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=list(resumed)),
    )


class TestPadToPow2:

    @pytest.mark.parametrize(("values", "expected"), [
        ([], []),
        ([5], [5]),
        ([5, 6], [5, 6]),
        ([5, 6, 7], [5, 6, 7, 0]),
        ([5, 6, 7, 8], [5, 6, 7, 8]),
        ([5, 6, 7, 8, 9], [5, 6, 7, 8, 9, 0, 0, 0]),
    ])
    def test_padding(self, values, expected):
        assert _pad_to_pow2(values) == expected


class TestMambaApcStateCopier:

    @pytest.fixture(autouse=True)
    def _cpu_state_copy(self, monkeypatch):
        """CPU stand-in for the TPU donating copy op.

        Class-scoped so an unrelated top-level test added later to this module
        isn't silently monkeypatched.
        """
        monkeypatch.setattr(
            mamba_apc_mod,
            "mamba_state_copy",
            lambda state, src, dst: state.index_copy_(
                0, dst.long(), state.index_select(0, src.long())),
        )
        # Skip the real jax.export.export path in unit tests — the copy op is
        # already stubbed by the monkeypatch above.
        monkeypatch.setattr(mamba_apc_mod, "ensure_op_built", lambda: None)

    def test_first_prefill_chunk_no_copy(self):
        runner, conv, _ssm, _ = _make_runner(1, ["a"], [0], [[5, 6, 7, 8]])
        copier = MambaApcStateCopier(runner)
        before = conv.clone()

        copier.preprocess(_sched_output({"a": 2 * BLOCK_SIZE}))

        assert torch.equal(conv, before)
        assert copier._state_block_idx["a"] == 1

    def test_chunk_advance_copies_state(self):
        runner, conv, ssm, _ = _make_runner(1, ["a"], [2 * BLOCK_SIZE],
                                            [[5, 6, 7, 8]])
        copier = MambaApcStateCopier(runner)
        copier._state_block_idx["a"] = 1

        copier.preprocess(_sched_output({"a": BLOCK_SIZE}))

        assert copier._state_block_idx["a"] == 2
        assert torch.equal(conv[7], torch.full((STATE_DIM, ), 6.0))
        assert torch.equal(ssm[7], torch.full((STATE_DIM, ), 600.0))
        assert torch.equal(conv[0], torch.zeros(STATE_DIM))

    def test_cache_hit_resume_copies_from_checkpoint(self):
        runner, conv, _ssm, _ = _make_runner(1, ["b"], [2 * BLOCK_SIZE],
                                             [[9, 10, 11, 12]])
        copier = MambaApcStateCopier(runner)

        copier.preprocess(_sched_output({"b": BLOCK_SIZE}))

        assert copier._state_block_idx["b"] == 2
        assert torch.equal(conv[11], torch.full((STATE_DIM, ), 10.0))

    def test_decode_within_block_no_copy(self):
        runner, conv, _ssm, _ = _make_runner(1, ["a"], [3 * BLOCK_SIZE - 1],
                                             [[5, 6, 7, 8]])
        copier = MambaApcStateCopier(runner)
        copier._state_block_idx["a"] = 2
        before = conv.clone()

        copier.preprocess(_sched_output({"a": 1}))

        assert torch.equal(conv, before)
        assert copier._state_block_idx["a"] == 2

    def test_decode_boundary_crossing_copies(self):
        runner, conv, _ssm, _ = _make_runner(1, ["a"], [3 * BLOCK_SIZE],
                                             [[5, 6, 7, 8]])
        copier = MambaApcStateCopier(runner)
        copier._state_block_idx["a"] = 2

        copier.preprocess(_sched_output({"a": 1}))

        assert copier._state_block_idx["a"] == 3
        assert torch.equal(conv[8], torch.full((STATE_DIM, ), 7.0))

    def test_finished_preempted_and_resumed_are_forgotten(self):
        runner, _conv, _ssm, _ = _make_runner(0, [], [], [])
        copier = MambaApcStateCopier(runner)
        copier._state_block_idx = {"done": 3, "kicked": 1, "back": 2}

        copier.preprocess(
            _sched_output({},
                          finished=("done", ),
                          preempted={"kicked"},
                          resumed=("back", )))

        assert copier._state_block_idx == {}

    def test_unscheduled_request_untouched(self):
        runner, conv, _ssm, _ = _make_runner(1, ["a"], [2 * BLOCK_SIZE],
                                             [[5, 6, 7, 8]])
        copier = MambaApcStateCopier(runner)
        copier._state_block_idx["a"] = 1
        before = conv.clone()

        copier.preprocess(_sched_output({}))

        assert torch.equal(conv, before)
        assert copier._state_block_idx["a"] == 1

    def test_context_parallel_uses_effective_block_size(self):
        runner, conv, _ssm, _ = _make_runner(1, ["a"], [2 * BLOCK_SIZE],
                                             [[5, 6, 7, 8]])
        copier = MambaApcStateCopier(runner)

        with patch(
                "vllm_torchtpu.runner.mamba_apc.get_dcp_group"
        ) as mock_dcp, patch(
                "vllm_torchtpu.runner.mamba_apc.get_pcp_group") as mock_pcp:
            mock_dcp.return_value.world_size = 2
            mock_pcp.return_value.world_size = 1
            copier.preprocess(_sched_output({"a": 1}))

        assert copier._state_block_idx["a"] == 1
        assert torch.equal(conv[6], torch.full((STATE_DIM, ), 5.0))
        assert torch.equal(conv[7], torch.full((STATE_DIM, ), 7.0))

    def test_three_requests_triggers_pow2_padding(self):
        # 3 real (src,dst) pairs pad to length 4. Padding writes state[0] to
        # state[0] (a self-copy) — must not clobber the real copies.
        runner, conv, _ssm, _ = _make_runner(
            3,
            ["a", "b", "c"],
            [2 * BLOCK_SIZE, 2 * BLOCK_SIZE, 2 * BLOCK_SIZE],
            [[5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        )
        copier = MambaApcStateCopier(runner)
        for rid in ("a", "b", "c"):
            copier._state_block_idx[rid] = 1
        before_zero = conv[0].clone()

        copier.preprocess(
            _sched_output({
                "a": BLOCK_SIZE,
                "b": BLOCK_SIZE,
                "c": BLOCK_SIZE,
            }))

        assert torch.equal(conv[7], torch.full((STATE_DIM, ), 6.0))
        assert torch.equal(conv[11], torch.full((STATE_DIM, ), 10.0))
        assert torch.equal(conv[15], torch.full((STATE_DIM, ), 14.0))
        assert torch.equal(conv[0], before_zero)

    def test_block_zero_as_destination(self):
        # A real dst == 0 collides with the pow2 padding sentinel. In production
        # vLLM's BlockPool reserves block 0 as the null_block so this doesn't
        # happen, but assert the copy still lands correctly when it does — a
        # regression in the op's `dst[0] >= 0` guard would silently drop it.
        runner, conv, _ssm, _ = _make_runner(1, ["a"], [BLOCK_SIZE], [[3, 0]])
        copier = MambaApcStateCopier(runner)
        copier._state_block_idx["a"] = 0

        copier.preprocess(_sched_output({"a": 1}))

        assert copier._state_block_idx["a"] == 1
        assert torch.equal(conv[0], torch.full((STATE_DIM, ), 3.0))

    def test_multiple_mamba_groups_get_independent_copies(self):
        runner, conv, _ssm, _, extras = _make_runner(
            1,
            ["a"],
            [2 * BLOCK_SIZE],
            [[5, 6, 7, 8]],
            extra_mamba_group=True,
            extra_block_rows=[[15, 16, 17, 18]],
        )
        conv2 = extras["conv2"]
        copier = MambaApcStateCopier(runner)
        copier._state_block_idx["a"] = 1

        copier.preprocess(_sched_output({"a": BLOCK_SIZE}))

        # group 1 (linear_attn) copies within its own block table.
        assert torch.equal(conv[7], torch.full((STATE_DIM, ), 6.0))
        # group 2 uses its own block table (15..18 instead of 5..8) — dst=17,
        # src=16 → state2[17] should hold state2[16]'s value (16*1000 = 16000).
        assert torch.equal(conv2[17], torch.full((STATE_DIM, ), 16000.0))
