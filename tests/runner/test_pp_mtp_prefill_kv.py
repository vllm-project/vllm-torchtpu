# SPDX-License-Identifier: Apache-2.0
"""MTP for prefill KV under pipeline parallelism (TPU_PP_MTP_PREFILL_KV).

CPU tests of the runner side: only the last stage builds the drafter, the
scheduler never receives drafts, a step that verifies drafts is refused, the
first stage's embedding shard reaches the last stage's draft, and the offload
connector registers the drafter layer's KV with the last stage's own layers.
"""

import os
import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp

import vllm_torchtpu.runner.tpu_runner as tr
from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingConnector
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.spec_decode import utils as spec_utils

MTP1 = SimpleNamespace(method="mtp", num_speculative_tokens=1, use_eagle=lambda: True)


class _FakeProposer:
    def __init__(self, runner, vllm_config):
        self.runner = runner


def _runner(monkeypatch, *, pp=2, last=True, speculative=MTP1):
    runner = object.__new__(TPUModelRunner)
    runner.parallel_config = SimpleNamespace(pipeline_parallel_size=pp)
    runner.speculative_config = speculative
    runner.vllm_config = SimpleNamespace(
        parallel_config=runner.parallel_config, speculative_config=speculative
    )
    runner.device = torch.device("cpu")
    runner._pp_mtp_kv_only = tr.pp_mtp_prefill_kv_only(runner.vllm_config)
    monkeypatch.setattr(tr, "pp_rank_flags", lambda _config: (not last, last))
    monkeypatch.setattr(tr, "Eagle3Proposer", _FakeProposer)
    monkeypatch.setattr(tr, "RejectionSampler", lambda *_args: object())
    runner._init_speculative_decoding()
    return runner


def test_only_the_last_stage_builds_the_drafter(monkeypatch):
    first = _runner(monkeypatch, last=False)
    assert first._pp_mtp_kv_only
    assert first.drafter is None
    assert not first._is_async_drafter

    last = _runner(monkeypatch, last=True)
    assert isinstance(last.drafter, _FakeProposer)
    assert last._is_async_drafter


def test_a_single_stage_keeps_its_drafter(monkeypatch):
    runner = _runner(monkeypatch, pp=1, last=False)
    assert not runner._pp_mtp_kv_only
    assert isinstance(runner.drafter, _FakeProposer)
    assert runner._is_async_drafter


class _Manager:
    def __init__(self):
        self.taken = 0

    def take_draft_token_ids(self):
        self.taken += 1
        return "drafts"


@pytest.mark.parametrize("pp, expected", [(2, None), (1, "drafts")])
def test_the_scheduler_gets_no_drafts_under_a_pipeline(monkeypatch, pp, expected):
    runner = _runner(monkeypatch, pp=pp)
    runner.spec_decode_manager = _Manager()
    assert runner.take_draft_token_ids() == expected
    # The cached drafts are consumed either way.
    assert runner.spec_decode_manager.taken == 1


def _scheduled(spec):
    return SimpleNamespace(scheduled_spec_decode_tokens=spec)


def test_a_step_without_drafts_passes():
    tr._refuse_pp_spec_tokens(_scheduled({}))
    tr._refuse_pp_spec_tokens(_scheduled({"a": [], "b": []}))


def test_a_step_that_verifies_drafts_is_refused():
    with pytest.raises(RuntimeError, match=r"never verifies drafts.*\['b'\]"):
        tr._refuse_pp_spec_tokens(_scheduled({"a": [], "b": [7]}))


# ---------------------------------------------------------------------------
# Embedding hand-off
# ---------------------------------------------------------------------------


def _target_with_embed(embed):
    return SimpleNamespace(model=SimpleNamespace(embed_tokens=embed))


def test_target_embed_on_another_stage():
    from vllm.model_executor.models.utils import PPMissingLayer

    assert spec_utils.target_embed_on_another_stage(
        _target_with_embed(PPMissingLayer())
    )
    assert not spec_utils.target_embed_on_another_stage(
        _target_with_embed(torch.nn.Embedding(4, 2))
    )


def _draft_with_embed(rows, dim):
    embed = torch.nn.Embedding(rows, dim)
    return SimpleNamespace(model=SimpleNamespace(embed_tokens=embed))


def test_install_draft_embed_shard():
    draft = _draft_with_embed(6, 4)
    shard = torch.arange(24, dtype=torch.float32).view(6, 4)
    spec_utils.install_draft_embed_shard(draft, shard)
    assert torch.equal(draft.model.embed_tokens.weight, shard)
    assert not draft.model.embed_tokens.weight.requires_grad


def test_install_draft_embed_shard_refuses_another_tp_layout():
    draft = _draft_with_embed(12, 4)
    with pytest.raises(RuntimeError, match="tensor parallel size"):
        spec_utils.install_draft_embed_shard(draft, torch.zeros(6, 4))


def _hand_off_worker(rank, init_file, dtype, out_dir):
    import torch.distributed as dist
    import vllm.distributed.parallel_state as ps

    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2
    )
    ps.get_pp_group = lambda: SimpleNamespace(ranks=[0, 1], cpu_group=None)
    weight = (torch.arange(5 * 3, dtype=torch.float32).view(5, 3) / 7).to(dtype)
    target = _target_with_embed(SimpleNamespace(weight=weight))
    got = spec_utils.hand_embedding_to_last_stage(
        target, is_first=rank == 0, is_last=rank == 1
    )
    if rank == 0:
        assert got is None
    else:
        torch.save(got, os.path.join(out_dir, "got.pt"))
    dist.destroy_process_group()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_the_first_stage_hands_its_embedding_to_the_last(dtype):
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(
            _hand_off_worker,
            args=(os.path.join(tmp, "init"), dtype, tmp),
            nprocs=2,
            join=True,
        )
        got = torch.load(os.path.join(tmp, "got.pt"))
    expected = (torch.arange(5 * 3, dtype=torch.float32).view(5, 3) / 7).to(dtype)
    assert got.dtype == dtype
    assert torch.equal(got, expected)


# ---------------------------------------------------------------------------
# Offload registration
# ---------------------------------------------------------------------------

NUM_BLOCKS = 4


def test_offload_registers_the_drafter_layer_of_the_last_stage():
    """GLM-5.2 keeps one MLA and one indexer-K cache per layer; on the last
    stage of a 42,36 partition the drafter's layer 78 sits beside 42-77, and
    its caches must be registered like theirs."""
    captured = []
    worker = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            num_blocks=NUM_BLOCKS, kv_cache_groups=[object(), object()]
        ),
        _init_worker=captured.append,
    )
    connector = SimpleNamespace(connector_worker=worker)

    kv_caches = {}
    storages = {}
    for layer in [*range(42, 78), 78]:
        mla = torch.zeros(NUM_BLOCKS, 32, dtype=torch.uint8)
        idx = torch.zeros(NUM_BLOCKS, 8, dtype=torch.uint8)
        prefix = "model.layers.78.mtp_block" if layer == 78 else f"model.layers.{layer}"
        kv_caches[f"{prefix}.self_attn.attn"] = mla
        kv_caches[f"{prefix}.self_attn.indexer.k_cache"] = idx
        storages[layer] = (mla.untyped_storage(), idx.untyped_storage())

    TPURaidenOffloadingConnector.register_kv_caches(connector, kv_caches)

    registered = {t.tensor.untyped_storage().data_ptr() for t in captured[0].tensors}
    assert len(captured[0].tensors) == 2 * 37
    for mla_storage, idx_storage in storages.values():
        assert mla_storage.data_ptr() in registered
        assert idx_storage.data_ptr() in registered


def test_offload_capacity_counts_the_drafter_layer_of_the_last_stage():
    from vllm_torchtpu.offload import raiden_store

    def worker(layers):
        page = 40 * layers
        return SimpleNamespace(
            num_blocks=NUM_BLOCKS,
            kv_cache_tensors=[SimpleNamespace(size=page * NUM_BLOCKS)],
        )

    # Stage 0 holds 42 layers; the last stage 36 plus the drafter's.
    total = raiden_store.record_all_ranks_kv_bytes_per_block([worker(42), worker(37)])
    assert total == 40 * (42 + 37)
    assert raiden_store.all_ranks_kv_bytes_per_block() == total
