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
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

from vllm_torchtpu.spec_decode.dspark import DSparkProposer


class _FakeHFConfig:
    """to_dict() reflects attribute writes, like a transformers config."""

    def __init__(self, d):
        self._d = dict(d)

    def to_dict(self):
        out = dict(self._d)
        if hasattr(self, "dflash_config"):
            out["dflash_config"] = self.dflash_config
        return out


def _make_proposer(hf_dict, num_speculative_tokens=4, **spec_fields):
    # The vLLM 0.29 drafting fields at their defaults; tests override them.
    fields = dict(
        draft_sample_method="greedy",
        enable_adaptive_verification=False,
        dspark_draft_topk=None,
    )
    fields.update(spec_fields)
    speculative_config = SimpleNamespace(
        num_speculative_tokens=num_speculative_tokens,
        draft_tensor_parallel_size=1,
        draft_model_config=SimpleNamespace(hf_config=_FakeHFConfig(hf_dict)),
        **fields,
    )
    vllm_config = SimpleNamespace(
        speculative_config=speculative_config,
        model_config=SimpleNamespace(max_model_len=2048),
        compilation_config=SimpleNamespace(
            static_forward_context={
                "model.layers.0.self_attn": mock.MagicMock(),
            },
            fast_moe_cold_start=False,
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            tensor_parallel_size=1,
            is_moe_model=False,
            use_sequence_parallel_moe=False,
        ),
    )
    runner = mock.MagicMock()
    runner.device = torch.device("cpu")
    return DSparkProposer(runner=runner, vllm_config=vllm_config)


_LAYER_IDS = {"dspark_target_layer_ids": [1, 15, 30]}


def test_noise_resolution_and_dense_block():
    p = _make_proposer(
        {
            "dspark_noise_token_id": 7,
            "dspark_block_size": 4,
            **_LAYER_IDS,
        }
    )
    assert p.mask_token_id == 7
    assert p.sample_from_anchor
    assert p._query_block_size(4) == 4
    # propose() sizes the query block via this property, so the dense
    # override must reach it (K slots, not DFlash's 1 + K).
    assert p.block_size == 4


def test_bonus_anchor_keeps_dflash_layout():
    p = _make_proposer(
        {
            "mask_token_id": 5,
            "sample_from_anchor": False,
            **_LAYER_IDS,
        }
    )
    assert not p.sample_from_anchor
    assert p._query_block_size(4) == 5
    assert p.block_size == 5


@pytest.mark.parametrize(
    "field,value",
    [
        ("enable_adaptive_verification", True),
        ("dspark_draft_topk", 8),
    ],
)
def test_warns_on_unsupported_drafting_options(field, value):
    with mock.patch("vllm_torchtpu.spec_decode.dspark.logger") as log:
        p = _make_proposer(
            {
                "dspark_noise_token_id": 7,
                "dspark_block_size": 4,
                **_LAYER_IDS,
            },
            **{field: value},
        )
    # Drafting stays over the fixed K-token block.
    assert p.block_size == 4
    log.warning.assert_called_once()
    assert "does not support" in log.warning.call_args.args[0]


@pytest.mark.parametrize("rejection_sample_method", ["standard", "synthetic"])
def test_probabilistic_drafting_under_any_rejection_method(rejection_sample_method):
    # As upstream: synthetic rejection still gets q for its recovery step.
    with mock.patch("vllm_torchtpu.spec_decode.dspark.logger") as log:
        p = _make_proposer(
            {"dspark_noise_token_id": 7, **_LAYER_IDS},
            draft_sample_method="probabilistic",
            rejection_sample_method=rejection_sample_method,
        )
    assert p.probabilistic is True
    # Supported now: no "does not support" warning either way.
    log.warning.assert_not_called()
    assert p._draft_probs is None


def test_greedy_drafting_is_the_default():
    p = _make_proposer({"dspark_noise_token_id": 7, **_LAYER_IDS})
    assert p.probabilistic is False


@pytest.mark.parametrize("method", ["probabilistic", "greedy"])
def test_load_model_creates_sampling_generator_with_nothing_in_flight(method):
    # torch_tpu's Generator constructor can hang behind an in-flight
    # non-blocking upload, so probabilistic drafting creates the runner's
    # generator at load, after a device sync, not inside a draft forward.
    p = _make_proposer(
        {"dspark_noise_token_id": 7, **_LAYER_IDS}, draft_sample_method=method
    )
    p.vllm_config.model_config.get_vocab_size = lambda: 5
    p.draft_model = SimpleNamespace(
        compute_draft_logits=None,
        markov_embed=None,
        markov_bias=None,
        map_draft_to_target=None,
        has_own_lm_head=False,
        has_own_embed_tokens=False,
    )
    calls = []
    p.runner._get_sampling_generator = lambda: calls.append("generator")
    with (
        mock.patch("vllm_torchtpu.spec_decode.dflash.DFlashProposer.load_model"),
        mock.patch(
            "vllm_torchtpu.spec_decode.dspark.synchronize_device",
            side_effect=lambda: calls.append("sync"),
        ),
    ):
        p.load_model(target_model=None)
    assert calls == (["sync", "generator"] if method == "probabilistic" else [])


def _probabilistic_proposer(num_reqs, K=3, V=5, all_greedy=False):
    p = _make_proposer(
        {"dspark_noise_token_id": 7, **_LAYER_IDS},
        num_speculative_tokens=K,
        draft_sample_method="probabilistic",
        rejection_sample_method="standard",
    )
    p.vllm_config.model_config.get_vocab_size = lambda: V
    p.runner.device = torch.device("cpu")
    p.runner.input_batch = SimpleNamespace(
        num_reqs=num_reqs,
        req_ids=[f"r{i}" for i in range(num_reqs)],
        temperature_cpu_tensor=torch.full((num_reqs,), 0.5),
        all_greedy=all_greedy,
    )
    del p.runner._get_sampling_generator  # MagicMock: no device generator
    return p


def test_run_draft_forward_samples_and_records_q_per_chunk():
    K, V = 3, 5
    p = _probabilistic_proposer(num_reqs=3, K=K, V=V)
    block_size = p.block_size  # dense layout: K
    # Two chunks: requests [0, 2) and [2, 3); row buckets 4 and 2.
    chunks = [
        SimpleNamespace(start_index=0, num_reqs=2),
        SimpleNamespace(start_index=2, num_reqs=1),
    ]
    rows_per_chunk = [4, 2]
    calls = []

    def fake_logits_graph(input_ids, positions, block_size):
        rows = input_ids.shape[0] // block_size
        greedy = torch.zeros((rows, K), dtype=torch.int64)
        return greedy, None, torch.zeros((input_ids.shape[0], V))

    def fake_sample_graph(logits, input_ids, block_size, temperatures, uniform):
        rows = logits.shape[0] // block_size
        calls.append((rows, temperatures.squeeze(1).tolist(), tuple(uniform.shape)))
        probs = torch.arange(rows * K * V, dtype=torch.float32).view(rows, K, V)
        return torch.zeros((rows, K), dtype=torch.int64), probs

    p._dflash_forward_logits = fake_logits_graph
    p._markov_sample_probs = fake_sample_graph
    p._dflash_forward_and_sample = mock.MagicMock(side_effect=AssertionError)
    for chunk, rows in zip(chunks, rows_per_chunk):
        ids = torch.zeros(rows * block_size, dtype=torch.int32)
        # Sampling inputs are built before the forward context; the forward
        # yields logits; sampling runs after it.
        p._prepare_draft_sampling(chunk, ids, block_size)
        assert p._pending_sampling_inputs is not None
        out = p._run_draft_forward(ids, ids, block_size, chunk)
        out = p._finish_draft_forward(out, chunk)
        assert out.shape == (rows, K)
        assert p._pending_sampling_inputs is None
    # The host side of each non-blocking upload stays referenced.
    assert len(p._host_keepalive) == 2

    # Live rows carry the request temperature, padding rows draft greedily;
    # the noise is [K, rows, V] so every Markov step reads a contiguous slice.
    assert calls[0] == (4, [0.5, 0.5, 0.0, 0.0], (K, 4, V))
    assert calls[1] == (2, [0.5, 0.0], (K, 2, V))
    assert p._draft_probs_rows == {"r0": 0, "r1": 1, "r2": 4}

    # Verify-time gather: request-major, the first n slots of each request,
    # in the verify chunk's order; zero-draft requests take no slots.
    q = p.gather_draft_probs(["r2", "r0", "r1"], np.array([2, 0, 3]), 8)
    flat = torch.cat([t.reshape(-1, V) for t in p._draft_probs])
    # r2 -> flat row 4, slots 0-1; r0 -> no slots; r1 -> flat row 1, slots 0-2.
    expected = flat[[4 * K + 0, 4 * K + 1, 1 * K + 0, 1 * K + 1, 1 * K + 2]]
    assert torch.equal(q[:5], expected)
    assert q.shape == (8, V)
    # A request this proposer did not draft disables q for the chunk.
    assert p.gather_draft_probs(["r9"], np.array([1]), 4) is None
    assert p.gather_draft_probs(["r0"], np.array([0]), 4) is None


def test_run_draft_forward_stays_greedy_for_greedy_batches():
    p = _probabilistic_proposer(num_reqs=2, all_greedy=True)
    greedy_tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
    p._dflash_forward_and_sample = mock.MagicMock(return_value=(greedy_tokens, None))
    p._dflash_forward_logits = mock.MagicMock(side_effect=AssertionError)
    p._markov_sample_probs = mock.MagicMock(side_effect=AssertionError)
    chunk = SimpleNamespace(start_index=0, num_reqs=2)
    ids = torch.zeros(2 * p.block_size, dtype=torch.int32)
    p._prepare_draft_sampling(chunk, ids, p.block_size)
    assert p._pending_sampling_inputs is None  # nothing to build for greedy
    out = p._run_draft_forward(ids, ids, p.block_size, chunk)
    out = p._finish_draft_forward(out, chunk)
    assert out is greedy_tokens
    assert p._draft_probs is None
    assert p.gather_draft_probs(["r0"], np.array([1]), 4) is None


def test_propose_resets_q():
    p = _probabilistic_proposer(num_reqs=1, all_greedy=False)
    p._draft_probs = [torch.zeros(1)]
    p._draft_probs_rows = {"stale": 0}
    with mock.patch(
        "vllm_torchtpu.spec_decode.dflash.DFlashProposer.propose", return_value=[]
    ) as base:
        p.propose(None, [], None, None)
    base.assert_called_once()
    assert p._draft_probs is None and p._draft_probs_rows == {}


@pytest.mark.parametrize(
    "sample_from_anchor,trained_block,K,query_block",
    [
        (True, 5, 6, 6),  # DeepSeek-V4-Pro-0813 served at K=6
        (True, 7, 4, 4),
        (False, 7, 4, 5),  # bonus-anchor layout keeps 1 + K
    ],
)
def test_k_need_not_match_checkpoint_block(
    sample_from_anchor, trained_block, K, query_block
):
    # The query block follows K whatever block the checkpoint was trained
    # at; the mismatch is allowed but warned about.
    hf = {
        "dspark_noise_token_id": 7,
        "dspark_block_size": trained_block,
        "sample_from_anchor": sample_from_anchor,
        **_LAYER_IDS,
    }
    with mock.patch("vllm_torchtpu.spec_decode.dspark.logger") as log:
        p = _make_proposer(hf, num_speculative_tokens=K)
    assert p.block_size == query_block
    log.warning.assert_called_once()
    assert "trained with block size" in log.warning.call_args.args[0]


@pytest.mark.parametrize(
    "block_key,sample_from_anchor,trained_block,K",
    [
        ("dspark_block_size", True, 4, 4),
        # Speculators-format checkpoints count the bonus anchor in block_size
        # and draft block_size - 1 tokens (e.g. dspark_qwen3_8b_block7, K=6).
        ("block_size", False, 7, 6),
    ],
)
def test_matching_checkpoint_block_does_not_warn(
    block_key, sample_from_anchor, trained_block, K
):
    hf = {
        "dspark_noise_token_id": 7,
        block_key: trained_block,
        "sample_from_anchor": sample_from_anchor,
        **_LAYER_IDS,
    }
    with mock.patch("vllm_torchtpu.spec_decode.dspark.logger") as log:
        _make_proposer(hf, num_speculative_tokens=K)
    log.warning.assert_not_called()


def test_missing_noise_token():
    with pytest.raises(ValueError, match="noise/mask token"):
        _make_proposer({})


def test_missing_target_layer_ids():
    with pytest.raises(ValueError, match="target layer ids"):
        _make_proposer({"dspark_noise_token_id": 7})


def test_target_layer_id_resolution_chain():
    # Aux ids (target + 1) take precedence and are shifted back down.
    p = _make_proposer(
        {
            "dspark_noise_token_id": 7,
            "eagle_aux_hidden_state_layer_ids": [2, 16, 31],
            "dspark_target_layer_ids": [9, 9, 9],
        }
    )
    assert p.dflash_config["target_layer_ids"] == [1, 15, 30]

    # Dense-checkpoint spelling.
    p = _make_proposer(
        {
            "dspark_noise_token_id": 7,
            "target_layer_ids": [3, 7],
        }
    )
    assert p.dflash_config["target_layer_ids"] == [3, 7]

    # An explicit dflash_config wins unchanged.
    p = _make_proposer(
        {
            "dspark_noise_token_id": 7,
            "dflash_config": {"target_layer_ids": [4, 8]},
            "dspark_target_layer_ids": [9, 9],
        }
    )
    assert p.dflash_config["target_layer_ids"] == [4, 8]


def test_no_forced_embedding_sharing():
    # A reduced-vocab DSpark draft owns its lm_head; load_model must honor
    # has_own_embed_tokens / has_own_lm_head instead of force-sharing.
    p = _make_proposer({"dspark_noise_token_id": 7, **_LAYER_IDS})
    assert p._force_share_target_embeddings is False
