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
"""The offloading scheduler's host-tier hit estimate.

``PrefixAwareScheduler._external_warmth`` ranks waiting requests by it, so a
request whose prefix is in the host tier is not sorted as a cold prompt. These
run the real lookup (upstream OffloadingConnectorScheduler) and the real
RaidenOffloadingManager over the fake store, on a full-attention group and a
Mamba group, as in TestDivergedHybridHitLookup.
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_raiden_store_offload as tro  # noqa: E402

CONN = "vllm_torchtpu.offload.raiden_connector"


class TestExternalHitEstimate(unittest.TestCase):
    # The lookup tests' harness, reached through the module so that pytest
    # does not collect those tests a second time here.
    _Lookup = tro.TestDivergedHybridHitLookup
    BLOCK = _Lookup.BLOCK
    PROMPT = _Lookup.PROMPT
    LOCAL = _Lookup.LOCAL
    ATTN = _Lookup.ATTN
    MAMBA = _Lookup.MAMBA

    setUp = _Lookup.setUp
    _key = _Lookup._key
    _seed = _Lookup._seed
    _hit = _Lookup._hit

    def _estimate(self, local: int | None = None) -> int:
        return self.scheduler.estimate_external_hit_tokens(
            self.request, self.LOCAL if local is None else local
        )

    def _host_warm(self) -> None:
        # Attention chunks 1-3 and the Mamba state at the end of chunk 3.
        self._seed(self.ATTN, 1, 2, 3)
        self._seed(self.MAMBA, 3)

    def test_estimate_is_the_hit_the_lookup_then_loads(self):
        self._host_warm()
        self.assertEqual(self._estimate(), 3 * self.BLOCK)
        self.assertEqual(self._hit(), 3 * self.BLOCK)

    def test_estimate_ends_at_the_newest_recurrent_state(self):
        self._seed(self.ATTN, 1, 2, 3)
        self._seed(self.MAMBA, 2)
        self.assertEqual(self._estimate(), 2 * self.BLOCK)

    def test_no_estimate_without_the_recurrent_state(self):
        self._seed(self.ATTN, 1, 2, 3)
        self.assertEqual(self._estimate(), 0)

    def test_estimate_counts_from_an_unaligned_local_hit(self):
        # The lookup starts at the chunk-aligned local hit; the answer is what
        # lies past the caller's own local hit.
        self._host_warm()
        self.assertEqual(self._estimate(self.LOCAL + 4), 3 * self.BLOCK - 4)

    def test_estimate_changes_nothing(self):
        self._host_warm()
        req_status = self.scheduler._req_status["req"]
        req_status.num_locally_computed_tokens = 7
        req_status.partial_tail_boundary = 5
        tracker = self.scheduler._events_tracker
        record_lookup = tracker.record_lookup = MagicMock()
        record_partial = tracker.record_partial_lookup = MagicMock()
        num_calls = len(self.store.calls)

        self.assertEqual(self._estimate(), 3 * self.BLOCK)

        calls = self.store.calls[num_calls:]
        self.assertTrue(calls)
        # Only unpinned probes: no pin, no release (a touch), no insert.
        self.assertEqual({c.name for c in calls}, {"lookup"})
        self.assertTrue(all(c.args[2] is False for c in calls))
        self.assertEqual(self.store.pinned_hashes(), {})
        record_lookup.assert_not_called()
        record_partial.assert_not_called()
        self.assertEqual(req_status.num_locally_computed_tokens, 7)
        self.assertEqual(req_status.partial_tail_boundary, 5)
        self.assertEqual(self.scheduler._jobs, {})
        self.assertIsNone(self.scheduler._step_region)

    def test_estimate_is_reused_for_its_ttl(self):
        self._host_warm()
        with patch(f"{CONN}.time.monotonic", return_value=100.0):
            self.assertEqual(self._estimate(), 3 * self.BLOCK)
        tro.displace_entries(self.store, self._key(self.MAMBA, 3))
        with patch(f"{CONN}.time.monotonic", return_value=100.2):
            self.assertEqual(self._estimate(), 3 * self.BLOCK)
        with patch(f"{CONN}.time.monotonic", return_value=100.3):
            self.assertEqual(self._estimate(), 0)

    def test_estimate_is_asked_again_for_another_local_hit(self):
        self._host_warm()
        with patch(f"{CONN}.time.monotonic", return_value=100.0):
            self.assertEqual(self._estimate(), 3 * self.BLOCK)
            tro.displace_entries(self.store, self._key(self.MAMBA, 3))
            self.assertEqual(self._estimate(2 * self.BLOCK), 0)

    def test_no_estimate_for_a_request_the_connector_does_not_track(self):
        self._host_warm()
        del self.scheduler._req_status["req"]
        num_calls = len(self.store.calls)
        self.assertEqual(self._estimate(), 0)
        self.assertEqual(self.store.calls[num_calls:], [])

    def test_no_estimate_while_a_transfer_is_in_flight(self):
        self._host_warm()
        self.scheduler._req_status["req"].transfer_jobs.add(1)
        self.assertEqual(self._estimate(), 0)

    def test_no_estimate_when_the_request_skips_the_prefix_cache(self):
        self._host_warm()
        self.request.skip_reading_prefix_cache = True
        self.assertEqual(self._estimate(), 0)

    def test_no_estimate_within_a_hash_block_of_the_end(self):
        self._host_warm()
        self.assertEqual(self._estimate(self.PROMPT - 1), 0)

    def test_estimate_inside_the_step_region_keeps_it_open(self):
        self._host_warm()
        self._hit()  # get_num_new_matched_tokens opens schedule()'s region.
        region = self.scheduler._step_region
        self.assertIsNotNone(region)
        with (
            patch(f"{CONN}.logger") as log,
            patch(
                "vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler.logger"
            ) as upstream_log,
        ):
            self.assertEqual(self._estimate(), 3 * self.BLOCK)
        self.assertIs(self.scheduler._step_region, region)
        log.error.assert_not_called()
        upstream_log.error.assert_not_called()

    def test_answers_of_finished_requests_are_dropped(self):
        self._host_warm()
        self.scheduler._external_hit_estimates = {
            f"gone-{i}": (0, 0, 0.0, 0) for i in range(100)
        }
        self.assertEqual(self._estimate(), 3 * self.BLOCK)
        self.assertEqual(set(self.scheduler._external_hit_estimates), {"req"})

    def test_connector_reports_its_scheduler_estimate(self):
        from vllm_torchtpu.offload.raiden_connector import (
            TPURaidenOffloadingConnector,
        )

        self._host_warm()
        connector = object.__new__(TPURaidenOffloadingConnector)
        connector.connector_scheduler = self.scheduler
        self.assertEqual(
            connector.estimate_external_hit_tokens(self.request, self.LOCAL),
            3 * self.BLOCK,
        )
        connector.connector_scheduler = None
        self.assertEqual(
            connector.estimate_external_hit_tokens(self.request, self.LOCAL), 0
        )


if __name__ == "__main__":
    unittest.main()
