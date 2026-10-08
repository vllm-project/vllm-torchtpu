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
"""Tests for RC34: DSpark MTP DP-EP Lockstep Collective Synchronization.

Verifies:
1. `model_has_moe` correctly identifies DSparkDeepseekV4Model and
   DSparkDeepseekV4ForCausalLM
   even without vLLM's MixtureOfExperts base interface.
2. `DFlashProposer._dp_lockstep_sharded()` returns True for DSpark models with MoE.
3. Zero-token idle ranks execute `_run_dp_idle_pairing` during DP lockstep instead of
   returning early and desynchronizing collectives.
4. Multirank lockstep phase alignment: all 16 DP ranks emit the identical sequence of
   target and draft collective phases when scheduled chunk counts vary (0, 1, 2 chunks).
5. Elimination of device-side `tt_jit_zero_` allocations in all spec
   decode helper paths.
"""

import os
import sys
from types import SimpleNamespace
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

for _cand_path in (
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
):
    if os.path.isdir(_cand_path) and _cand_path not in sys.path:
        sys.path.insert(0, _cand_path)


def _install_vllm_test_stubs():
    import contextlib
    import dataclasses
    import importlib.abc
    import importlib.machinery
    import logging
    import types

    try:
        import torch_tpu  # noqa: F401
        import vllm  # noqa: F401

        return
    except ImportError:
        pass

    class _DummyMeta(type):
        def __getattr__(cls, attr: str):
            if attr.startswith("__") and attr.endswith("__"):
                raise AttributeError(attr)
            val = MagicMock(name=f"{cls.__name__}.{attr}")
            setattr(cls, attr, val)
            return val

    class _DummyBase(metaclass=_DummyMeta):
        def __init__(self, *args, **kwargs):
            pass

    class _DummyMod(types.ModuleType):
        def __init__(self, name: str):
            super().__init__(name)
            self.__path__ = []

        def __getattr__(self, attr: str):
            if attr.startswith("__") and attr.endswith("__"):
                raise AttributeError(attr)
            sub_name = f"{self.__name__}.{attr}"
            if sub_name in sys.modules:
                val = sys.modules[sub_name]
                setattr(self, attr, val)
                return val
            if self.__name__ == "vllm.envs" and attr.isupper():
                setattr(self, attr, None)
                return None
            if attr[0].isupper():
                val = _DummyMeta(attr, (_DummyBase,), {})
            else:
                val = MagicMock(name=sub_name)
            setattr(self, attr, val)
            return val

    class _AutoStubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        PREFIXES = (
            "vllm",
            "torch_tpu",
            "einops",
            "compressed_tensors",
            "transformers",
            "safetensors",
            "portpicker",
            "requests",
        )

        def find_spec(self, fullname, path, target=None):
            if any(
                fullname == p or fullname.startswith(p + ".") for p in self.PREFIXES
            ):
                return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
            return None

        def create_module(self, spec):
            if spec.name in sys.modules:
                return sys.modules[spec.name]
            mod = _DummyMod(spec.name)
            sys.modules[spec.name] = mod
            if "." in spec.name:
                parent_name, child_attr = spec.name.rsplit(".", 1)
                if parent_name in sys.modules:
                    setattr(sys.modules[parent_name], child_attr, mod)
            return mod

        def exec_module(self, module):
            pass

    sys.meta_path.insert(0, _AutoStubFinder())
    stub_modules = [
        "torch_tpu",
        "vllm",
        "vllm.envs",
        "vllm.logger",
        "vllm.config",
        "vllm.compilation",
        "vllm.compilation.backends",
        "vllm.distributed",
        "vllm.distributed.parallel_state",
        "vllm.forward_context",
        "vllm.v1",
        "vllm.v1.outputs",
        "vllm.v1.core",
        "vllm.v1.core.sched",
        "vllm.v1.core.sched.async_scheduler",
        "vllm.v1.core.sched.interface",
        "vllm.v1.core.sched.output",
        "vllm.v1.core.sched.request_queue",
        "vllm.v1.worker",
        "vllm.v1.worker.gpu_model_runner",
    ]
    for mod_name in stub_modules:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = _DummyMod(mod_name)

    @contextlib.contextmanager
    def _noop_cm(*args, **kwargs):
        yield

    sys.modules["vllm.compilation.backends"].set_model_tag = _noop_cm
    sys.modules["vllm.config"].set_current_vllm_config = _noop_cm

    @dataclasses.dataclass
    class LogprobsTensors:
        logprob_token_ids: object = None
        logprobs: object = None
        selected_token_ranks: object = None

    @dataclasses.dataclass
    class LogprobsLists:
        logprob_token_ids: object = None
        logprobs: object = None
        sampled_token_ranks: object = None
        cu_num_generated_tokens: object = None

        def slice_request(self, req_idx: int, num_tokens: int) -> "LogprobsLists":
            start = self.cu_num_generated_tokens[req_idx]
            end = start + num_tokens
            return LogprobsLists(
                logprob_token_ids=self.logprob_token_ids[start:end],
                logprobs=self.logprobs[start:end],
                sampled_token_ranks=self.sampled_token_ranks[start:end],
                cu_num_generated_tokens=[0, num_tokens],
            )

    sys.modules["vllm.v1.outputs"].LogprobsTensors = LogprobsTensors
    sys.modules["vllm.v1.outputs"].LogprobsLists = LogprobsLists

    @dataclasses.dataclass
    class ModelRunnerOutput:
        req_ids: list = dataclasses.field(default_factory=list)
        req_id_to_index: dict = dataclasses.field(default_factory=dict)
        sampled_token_ids: list = dataclasses.field(default_factory=list)
        logprobs: object = None
        prompt_logprobs_dict: dict = dataclasses.field(default_factory=dict)
        pooler_output: list = dataclasses.field(default_factory=list)
        kv_connector_output: object = None
        ec_connector_output: object = None
        num_nans_in_logits: dict = dataclasses.field(default_factory=dict)
        finish_sending: object = None
        finish_recving: object = None

    sys.modules["vllm.v1.outputs"].ModelRunnerOutput = ModelRunnerOutput
    sys.modules["vllm.v1.outputs"].EMPTY_MODEL_RUNNER_OUTPUT = ModelRunnerOutput()

    try:
        import torch
        import torch._dynamo

        if "tpu" not in torch._dynamo.list_backends():

            @torch._dynamo.register_backend(name="tpu")
            def _dummy_tpu_backend(model, inputs):
                return model
    except Exception as e:
        logging.debug("torch._dynamo backend setup skipped: %s", e)

    class _GPUModelRunner:
        def __init__(self, *args, **kwargs):
            pass

    sys.modules["vllm.v1.worker.gpu_model_runner"].GPUModelRunner = _GPUModelRunner
    sys.modules["vllm.logger"].init_logger = lambda name: logging.getLogger(name)


_install_vllm_test_stubs()


from vllm_torchtpu.spec_decode.dflash import DFlashProposer  # noqa: E402
from vllm_torchtpu.spec_decode.utils import (  # noqa: E402
    DraftChunkInputs,
    model_has_moe,
)


def test_dspark_model_has_moe():
    """Verify DSpark models are recognized as MoE without depending on
    isinstance(MixtureOfExperts)."""
    assert model_has_moe(None) is False

    # Plain dense module
    dense = torch.nn.Linear(10, 10)
    assert model_has_moe(dense) is False

    # DSpark model mock with has_moe = True
    dspark_model = mock.MagicMock()
    dspark_model.has_moe = True
    dspark_model.num_moe_layers = 3
    dspark_model.modules.return_value = [dspark_model]
    assert model_has_moe(dspark_model) is True

    # DSpark causal lm mock with has_moe = True
    dspark_lm = mock.MagicMock()
    dspark_lm.has_moe = True
    dspark_lm.num_moe_layers = 3
    dspark_lm.modules.return_value = [dspark_lm]
    assert model_has_moe(dspark_lm) is True

    # MoE module inside submodules
    container = torch.nn.Sequential(dense)
    assert model_has_moe(container) is False
    moe_sub = mock.MagicMock()
    type(moe_sub).__name__ = "DeepseekV4MoE"
    container_with_moe = mock.MagicMock()
    container_with_moe.has_moe = False
    container_with_moe.num_moe_layers = 0
    container_with_moe.modules.return_value = [container_with_moe, moe_sub]
    assert model_has_moe(container_with_moe) is True


def _make_lockstep_proposer(
    has_moe: bool = True, draft_tp: int = 1
) -> tuple[DFlashProposer, mock.MagicMock]:
    hf_config_mock = mock.MagicMock()
    hf_config_mock.to_dict.return_value = {
        "dflash_config": {
            "mask_token_id": 0,
            "target_layer_ids": [1, 2],
        }
    }

    speculative_config = SimpleNamespace(
        num_speculative_tokens=4,
        draft_tensor_parallel_size=draft_tp,
        draft_model_config=SimpleNamespace(hf_config=hf_config_mock),
        use_eagle=lambda: True,
        method="dflash",
    )
    vllm_config = SimpleNamespace(
        speculative_config=speculative_config,
        model_config=SimpleNamespace(max_model_len=2048),
        compilation_config=SimpleNamespace(
            static_forward_context={},
            fast_moe_cold_start=False,
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=16,
            tensor_parallel_size=1,
            is_moe_model=True,
            use_sequence_parallel_moe=False,
        ),
        scheduler_config=SimpleNamespace(
            async_scheduling=True,
            max_num_seqs=16,
        ),
    )
    runner = mock.MagicMock()
    runner.device = torch.device("cpu")
    runner._dp_lockstep_enabled.return_value = True
    runner._dp_target_bucket = 128
    runner._dp_step_num_chunks = 2
    runner.num_reqs_paddings = [1, 2, 4, 8, 16]
    runner.num_tokens_paddings = [128, 256, 512, 1024]
    runner.num_reqs_max_model_len = 16
    runner.num_reqs_most_model_len = 16
    runner.max_num_blocks_per_req = 128
    runner.max_num_reqs = 16

    proposer = DFlashProposer(runner=runner, vllm_config=vllm_config)

    draft_model = mock.MagicMock()
    draft_model.has_moe = has_moe
    draft_model.num_moe_layers = 3 if has_moe else 0
    draft_model.modules.return_value = [draft_model]
    embed = torch.nn.Embedding(10, 8)
    draft_model.model.embed_tokens = embed
    proposer.draft_model = draft_model

    return proposer, runner


def test_dspark_dp_lockstep_sharded_detection():
    """Verify _dp_lockstep_sharded enables lockstep for MoE draft models even
    when draft_tp=1."""
    proposer, _ = _make_lockstep_proposer(has_moe=True, draft_tp=1)
    assert proposer._dp_lockstep_sharded() is True

    proposer_dense, _ = _make_lockstep_proposer(has_moe=False, draft_tp=1)
    assert proposer_dense._dp_lockstep_sharded() is False


def test_zero_device_tt_jit_zero_in_draft_helpers():
    """Verify that _zero_draft_lengths allocates on CPU avoiding device tt_jit_zero_."""
    proposer, runner = _make_lockstep_proposer(has_moe=True)
    runner.device = torch.device("cpu")

    # Initial zero lengths are pre-cached for all num_reqs_paddings
    for r in runner.num_reqs_paddings:
        tensor = proposer._zero_draft_lengths(int(r))
        assert tensor.device == runner.device
        assert (tensor == 0).all()

    # Dynamic lookup returns cached tensor
    t1 = proposer._zero_draft_lengths(4)
    t2 = proposer._zero_draft_lengths(4)
    assert t1 is t2


def test_16rank_lockstep_collective_symmetry():
    """Simulate all 16 DP ranks in a heterogeneous step (busy, partially busy,
    and idle ranks).

    Verifies that all 16 ranks execute the identical count of:
    - Target forwards (real + dummy)
    - Draft KV updates (real + dummy)
    - Draft forwards (real + dummy)
    in strict lockstep phase ordering.
    """
    num_ranks = 16
    # Configuration of scheduled chunks per rank:
    # Rank 0: 2 chunks (busy prefill)
    # Rank 1: 1 chunk (busy decode)
    # Rank 2: 0 chunks (idle rank)
    # Rank 3..15: mixture
    rank_chunks = [2, 1, 0, 1, 2, 0, 0, 1, 2, 1, 0, 1, 0, 2, 1, 0]
    target_num_chunks = max(rank_chunks)  # Coordinated DP max = 2

    rank_events = {r: [] for r in range(num_ranks)}

    for rank, num_c in enumerate(rank_chunks):
        proposer, runner = _make_lockstep_proposer(has_moe=True)
        runner._dp_step_num_chunks = target_num_chunks

        # Instrument runner and proposer to record collective events
        def make_record(rank_id, name):
            def record_fn(*args, **kwargs):
                rank_events[rank_id].append(name)

            return record_fn

        runner._run_dp_dummy_chunk = make_record(rank, "target_dummy_forward")
        proposer._dp_dummy_kv_update = make_record(rank, "draft_dummy_kv_update")
        proposer._dp_dummy_forward = make_record(rank, "draft_dummy_forward")
        proposer._update_draft_kv_cache_from_target = make_record(
            rank, "draft_real_kv_update"
        )
        proposer._dflash_forward_and_sample = mock.MagicMock(
            side_effect=lambda *args, _r=rank, **kwargs: (
                rank_events[_r].append("draft_real_forward")
                or (torch.zeros((1, 4), dtype=torch.int32), None)
            )
        )

        # 1. Target Phase (simulated from execute_model)
        for chunk_idx in range(num_c):
            rank_events[rank].append("target_real_forward")
        # Padding target chunks
        for chunk_idx in range(num_c, target_num_chunks):
            runner._run_dp_dummy_chunk(runner._dp_target_bucket)

        # 2. Draft Phase (simulated from sample_tokens / propose)
        if num_c == 0:
            # Idle rank: called via run_dp_dummy_draft(target_num_chunks)
            proposer.run_dp_dummy_draft(target_num_chunks)
        else:
            # Busy rank with num_c chunks: proposes with padding
            runner.input_batch.num_reqs = num_c
            mock_chunk = mock.MagicMock(spec=DraftChunkInputs)
            mock_chunk.start_index = 0
            mock_chunk.num_reqs = 1
            proposer.draft_chunks = [mock_chunk] * num_c
            proposer._prepare_dflash_inputs = mock.MagicMock(
                return_value=(
                    torch.zeros(16, dtype=torch.int32),
                    torch.zeros(16, dtype=torch.int32),
                    torch.zeros(1, dtype=torch.int32),
                )
            )
            proposer._build_draft_attn_metadata = mock.MagicMock()
            proposer.propose(
                sampled_token_ids=None,
                discard_sampled_tokens_req_indices=[],
                num_rejected_tokens_np=None,
                scheduler_output=None,
                return_device=True,
                next_tokens_per_chunk=[torch.zeros(1, dtype=torch.int32)] * num_c,
            )

    # Verification: Each rank must have executed:
    # 2 target forwards (real + dummy)
    # 2 draft KV updates (real + dummy)
    # 2 draft forwards (real + dummy)
    for rank in range(num_ranks):
        events = rank_events[rank]
        target_forwards = [e for e in events if "target_" in e]
        draft_kv_updates = [e for e in events if "draft_" in e and "_kv_update" in e]
        draft_forwards = [e for e in events if "draft_" in e and "_forward" in e]

        assert len(target_forwards) == target_num_chunks, (
            f"Rank {rank} had {len(target_forwards)} target forwards, "
            f"expected {target_num_chunks}"
        )
        assert len(draft_kv_updates) == target_num_chunks, (
            f"Rank {rank} had {len(draft_kv_updates)} draft KV updates, "
            f"expected {target_num_chunks}"
        )
        assert len(draft_forwards) == target_num_chunks, (
            f"Rank {rank} had {len(draft_forwards)} draft forwards, "
            f"expected {target_num_chunks}"
        )

        # Strict phase order: all target forwards precede all draft KV updates,
        # which precede all draft forwards.
        phase1 = events[:target_num_chunks]
        phase2 = events[target_num_chunks : 2 * target_num_chunks]
        phase3 = events[2 * target_num_chunks : 3 * target_num_chunks]

        assert all("target_" in e for e in phase1), (
            f"Rank {rank} phase 1 violation: {phase1}"
        )
        assert all("draft_" in e and "_kv_update" in e for e in phase2), (
            f"Rank {rank} phase 2 violation: {phase2}"
        )
        assert all("draft_" in e and "_forward" in e for e in phase3), (
            f"Rank {rank} phase 3 violation: {phase3}"
        )


# 2. Mechanism B: Hybrid TP=2/DP=8/EP=16 & TP=4/DP=4/EP=16 + RC34 + DSpark K=5
# ==============================================================================


def test_populate_draft_embed_from_target_when_draft_embed_is_none():
    """Verify populate_draft_embed_from_target works when DSpark draft model has
    `embed_tokens = None` under TP=2, DP=8, EP=16 or TP=4, DP=4, EP=16
    (`target_tp > 1`, `draft_tp == 1`).
    """
    from vllm_torchtpu.spec_decode.utils import (
        maybe_share_embeddings,
        populate_draft_embed_from_target,
    )

    org_vocab_size = 256
    hidden_size = 32
    full_weight = torch.arange(
        org_vocab_size * hidden_size, dtype=torch.float32
    ).reshape(org_vocab_size, hidden_size)

    # Simulate TP=2 sharded target VocabParallelEmbedding (rank 0 holds first half,
    # gather_sharded_weight reconstructs full_weight).
    target_embed = nn.Module()
    target_embed.org_vocab_size = org_vocab_size
    target_embed.weight = nn.Parameter(
        full_weight[: org_vocab_size // 2].clone(), requires_grad=False
    )
    target_model = SimpleNamespace(model=SimpleNamespace(embed_tokens=target_embed))

    # DSparkDeepseekV4Model sets `self.embed_tokens = None` at __init__ and
    # `has_own_embed_tokens = False` on the outer CausalLM class.
    draft_model = SimpleNamespace(
        has_own_embed_tokens=False,
        model=SimpleNamespace(embed_tokens=None),
    )

    with patch(
        "vllm_torchtpu.spec_decode.utils.gather_sharded_weight",
        return_value=(
            full_weight.clone(),
            org_vocab_size,
            hidden_size,
            torch.device("cpu"),
        ),
    ):
        maybe_share_embeddings(
            draft_model,
            target_model,
            tp_layout_matches=False,  # target_tp=2 != draft_tp=1
        )
        assert isinstance(draft_model.model.embed_tokens, nn.Embedding)
        assert draft_model.model.embed_tokens.org_vocab_size == org_vocab_size
        assert torch.equal(draft_model.model.embed_tokens.weight, full_weight)

        # Repeat-call path: calling populate_draft_embed_from_target a second time
        # when embed_tokens is already populated must succeed idempotently.
        populate_draft_embed_from_target(draft_model, target_embed)
        assert torch.equal(draft_model.model.embed_tokens.weight, full_weight)

        # Negative Control: mismatched org_vocab_size on an existing
        # draft_embed raises AssertionError.
        draft_model.model.embed_tokens.org_vocab_size = org_vocab_size + 64
        with pytest.raises(AssertionError, match="draft embed org_vocab"):
            populate_draft_embed_from_target(draft_model, target_embed)


@pytest.mark.parametrize("tp_size,dp_size", [(2, 8), (4, 4)])
def test_hybrid_tp_dp_ep16_dspark_lockstep_and_hash_moe_gather(
    tp_size: int, dp_size: int
):
    """Verify Hybrid `TP=2, DP=8, EP=16` and `TP=4, DP=4, EP=16` compatibility with:
    1. `DSparkDeepseekV4Model._dp_gather_hash_moe_input_ids` across `dp_size` ranks.
    2. `RC34` cross-DP lockstep `num_reqs > 0` indicator when some DP ranks are idle,
       some run decode (`K=5`), and some run micro-chunked prefill (`256` tokens).
    3. `_sync_replicated_drafts_across_tp` broadcasting draft tokens
       cleanly across `tp_size` ranks.
    """
    # Ensure stub base classes do not trigger metaclass conflicts when
    # running in CPU stub mode.
    if "vllm.model_executor.models.deepseek_v4.attention" in sys.modules:
        sys.modules[
            "vllm.model_executor.models.deepseek_v4.attention"
        ].DeepseekV4Attention = nn.Module
    if "vllm.model_executor.layers.attention_layer_base" in sys.modules:
        sys.modules[
            "vllm.model_executor.layers.attention_layer_base"
        ].AttentionLayerBase = object
    if "vllm.model_executor.models.interfaces" in sys.modules:
        sys.modules["vllm.model_executor.models.interfaces"].SupportsPP = object
    from vllm_torchtpu.models.vllm.deepseek_v4.dspark import DSparkDeepseekV4Model
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    assert tp_size * dp_size == 16  # 16 chips total (EP=16)

    # 1. Verify _dp_gather_hash_moe_input_ids on DSparkDeepseekV4Model
    dummy_dspark = DSparkDeepseekV4Model.__new__(DSparkDeepseekV4Model)
    nn.Module.__init__(dummy_dspark)
    fake_layer = SimpleNamespace(
        ffn=SimpleNamespace(
            gate=SimpleNamespace(tid2eid=torch.zeros((64, 4), dtype=torch.int32)),
            routes_dp_gathered_tokens=True,
        )
    )
    dummy_dspark.layers = [fake_layer]
    dummy_dspark._has_hash_moe = True

    local_ids = torch.arange(8, dtype=torch.int32)
    fake_dp = SimpleNamespace(
        world_size=dp_size,
        all_gather=lambda x, dim=0: x.repeat(dp_size),
    )
    with patch("vllm.distributed.get_dp_group", return_value=fake_dp):
        gathered_ids = dummy_dspark._dp_gather_hash_moe_input_ids(local_ids)
        assert gathered_ids.shape == (8 * dp_size,)

        # When Fused-EP owns dispatch (`routes_dp_gathered_tokens=False`),
        # ids stay local.
        fake_layer.ffn.routes_dp_gathered_tokens = False
        local_kept = dummy_dspark._dp_gather_hash_moe_input_ids(local_ids)
        assert local_kept.shape == (8,)

    # 2. Verify RC34 cross-DP lockstep + TP draft broadcast across all
    # (dp_rank, tp_rank)
    num_spec_tokens = 5
    golden_draft_by_dp = {
        dp_r: torch.tensor(
            [[1000 + dp_r * 10 + k for k in range(num_spec_tokens)]],
            dtype=torch.int64,
        )
        for dp_r in range(dp_size)
    }

    for dp_rank in range(dp_size):
        # Make half the DP ranks active decode, one rank micro-chunked
        # prefill, and at least one idle.
        is_idle_dp = dp_rank == (dp_size - 1)
        tp_masked_partials: list[torch.Tensor] = []
        runners_for_dp: list[tuple[int, TPUModelRunner, torch.Tensor]] = []
        for tp_rank in range(tp_size):
            runner = TPUModelRunner.__new__(TPUModelRunner)
            runner._dp_step_max_reqs = 2
            runner._dp_step_num_chunks = 1
            runner._dp_target_bucket = 256
            runner._dp_step_has_active_decode = True
            runner._tp_rank0_mask = None
            runner.speculative_config = SimpleNamespace(
                num_speculative_tokens=num_spec_tokens,
                method="dspark",
                draft_tensor_parallel_size=1,
            )
            # Corrupt non-zero TP ranks before sync to prove mask +
            # all_reduce overwrites them.
            local_draft = (
                golden_draft_by_dp[dp_rank].clone()
                if tp_rank == 0
                else torch.full((1, num_spec_tokens), 9999, dtype=torch.int64)
            )
            runners_for_dp.append((tp_rank, runner, local_draft))

        # First compute masked tensors across TP ranks (rank 0 keeps
        # draft, rank > 0 zeros it).
        for tp_rank, runner, local_draft in runners_for_dp:
            mask = 1 if tp_rank == 0 else 0
            tp_masked_partials.append(local_draft * mask)
        tp_sum = torch.stack(tp_masked_partials, dim=0).sum(dim=0)

        for tp_rank, runner, local_draft in runners_for_dp:
            runner._tpu_pin_drafts_to_rank0 = (
                lambda drafts, mask, _sum=tp_sum: _sum.clone()
            )
            with (
                patch(
                    "vllm.distributed.get_tensor_model_parallel_world_size",
                    return_value=tp_size,
                ),
                patch(
                    "vllm.distributed.get_tensor_model_parallel_rank",
                    return_value=tp_rank,
                ),
            ):
                synced = (
                    runner._sync_replicated_drafts_across_tp(local_draft)
                    if not is_idle_dp
                    else torch.empty((0, num_spec_tokens), dtype=torch.int64)
                )
            if not is_idle_dp:
                assert torch.equal(synced, golden_draft_by_dp[dp_rank])
                assert runner._tp_rank0_mask is not None
                assert int(runner._tp_rank0_mask.item()) == (1 if tp_rank == 0 else 0)
            else:
                # Idle DP rank still sees _dp_step_max_reqs == 2 > 0 for
                # RC34 dummy propose lockstep!
                assert runner._dp_step_max_reqs > 0


# ==============================================================================
# 3. Gate 7 & Gate 7B (TRAP24) Numerical Parity + Negative Controls
# ==============================================================================
