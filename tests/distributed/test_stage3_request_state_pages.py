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
"""Exercise state-page extents through the real connector scheduler methods."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from vllm.v1.request import RequestStatus

from tests.runner.test_qwen4_exp_pd import cache_groups
from vllm_torchtpu.distributed.kv_transfer.tpu_connector import _Stage3TransferGroup
from vllm_torchtpu.models.vllm.qwen4_exp.pd import (
    TransferExtent,
    build_transfer_groups,
)

from .test_tpu_connector import _make_raiden_scheduler


@pytest.fixture(autouse=True)
def transport(monkeypatch):
    for name, value in {
        "TPU_KV_RESHARD_TRANSPORT": "raiden",
        "TPU_RAIDEN_CONTROLLER_ADDRESS": "prefill-controller.test:27000",
        "TPU_RAIDEN_JOB_NAME": "prefill",
        "TPU_RAIDEN_ENGINE_ID": "producer-engine",
        "TPU_RAIDEN_TRANSFER_PARALLELISM": 1,
    }.items():
        monkeypatch.setenv(name, str(value))


def setup_plan(*, producer, state_first=False, tokens=8191):
    groups = cache_groups()
    if state_first:
        groups = groups[2:] + groups[:2]
    model_plan = build_transfer_groups(groups)
    scheduler = _make_raiden_scheduler(is_producer=producer, block_size=128)
    # Translate the model-owned plan at the boundary; no native admission is
    # implied by this scheduler-only test.
    scheduler._stage3_transfer_groups = tuple(
        _Stage3TransferGroup(
            group.group_index,
            group.block_size,
            request_state=group.extent is TransferExtent.REQUEST_STATE,
        )
        for group in model_plan
    )
    scheduler._stage3_fa_group_index = 0
    pages = []
    offset = 1
    for group in model_plan:
        count = group.page_count(tokens)
        pages.append(list(range(offset, offset + count)))
        offset += count
    req = SimpleNamespace(
        request_id="state-pages",
        status=RequestStatus.FINISHED_LENGTH_CAPPED,
        num_prompt_tokens=tokens,
        num_computed_tokens=tokens,
        kv_transfer_params={"_p_side_truncated": True},
    )
    return scheduler, req, pages


@pytest.mark.parametrize("state_first", [False, True])
@pytest.mark.parametrize("tokens", [1, 4, 129, 8191])
def test_producer_publishes_token_pages_and_one_page_per_state(state_first, tokens):
    scheduler, req, pages = setup_plan(
        producer=True, state_first=state_first, tokens=tokens
    )
    assert scheduler._stage3_truncates_prompt()
    delay, params = scheduler.request_finished(req, pages[0], group_block_ids=pages)
    assert delay and params["num_tokens"] == tokens
    send = scheduler.build_connector_meta().reqs_to_send[req.request_id]
    assert send.group_block_ids == pages
    assert send.local_block_ids == pages[0]
    assert send.mamba_state_block_ids is None
    # A repeated finish retains its generation, without issuing a second send.
    again, repeated = scheduler.request_finished(req, pages[0], group_block_ids=pages)
    assert again and repeated == params
    assert not scheduler.build_connector_meta().reqs_to_send


@pytest.mark.parametrize("state_first", [False, True])
def test_consumer_admits_complete_mixed_extent_tables(state_first):
    producer, req, pages = setup_plan(producer=True, state_first=state_first)
    _, params = producer.request_finished(req, pages[0], group_block_ids=pages)
    consumer, dst, _ = setup_plan(producer=False, state_first=state_first)
    dst.kv_transfer_params = params
    blocks = MagicMock()
    blocks.get_block_ids.return_value = pages
    consumer._update_stage3_state_after_alloc(dst, blocks, 8191)
    load = consumer.reqs_to_load[dst.request_id]
    assert load.group_block_ids == pages
    assert load.local_block_ids == pages[0]
    assert load.num_tokens == 8191


@pytest.mark.parametrize("bad_pages", [[], [0], [True], [7, 8], [7, 7]])
def test_bad_state_pages_never_publish_or_load(bad_pages):
    producer, req, pages = setup_plan(producer=True)
    pages[2] = bad_pages
    assert producer.request_finished(req, pages[0], group_block_ids=pages) == (
        False,
        {},
    )
    assert not producer.reqs_to_send
    consumer, dst, _ = setup_plan(producer=False)
    dst.kv_transfer_params = {
        "req_id": "source",
        "uuid": 5,
        "num_tokens": 8191,
        "src_controller_address": "source:27000",
    }
    blocks = MagicMock()
    blocks.get_block_ids.return_value = pages
    with patch.object(consumer, "_fail_load") as failure:
        consumer._update_stage3_state_after_alloc(dst, blocks, 8191)
    failure.assert_called_once()
    assert not consumer.reqs_to_load


@pytest.mark.parametrize("delta", [-1, 1])
def test_producer_rejects_state_at_another_committed_prefix(delta):
    producer, req, pages = setup_plan(producer=True)
    assert producer._stage3_publish_send(
        req,
        pages[0],
        None,
        8191,
        computed_tokens=8191 + delta,
        group_block_ids=pages,
    ) == (False, {})
    assert not producer.reqs_to_send


def test_producer_rejects_missing_group_tables_and_cp():
    producer, req, pages = setup_plan(producer=True)
    assert producer.request_finished(req, pages[0]) == (False, {})
    producer.vllm_config.parallel_config.prefill_context_parallel_size = 2
    assert producer.request_finished(req, pages[0], group_block_ids=pages) == (
        False,
        {},
    )


@pytest.mark.parametrize("external,dcp", [(8190, 1), (8191, 2)])
def test_consumer_rejects_partial_prefix_or_context_sharding(external, dcp):
    producer, req, pages = setup_plan(producer=True)
    _, params = producer.request_finished(req, pages[0], group_block_ids=pages)
    consumer, dst, _ = setup_plan(producer=False)
    dst.kv_transfer_params = params
    consumer.vllm_config.parallel_config.decode_context_parallel_size = dcp
    blocks = MagicMock()
    blocks.get_block_ids.return_value = pages
    with patch.object(consumer, "_fail_load") as failure:
        consumer._update_stage3_state_after_alloc(dst, blocks, external)
    failure.assert_called_once()
    assert not consumer.reqs_to_load
