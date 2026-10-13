# SPDX-License-Identifier: Apache-2.0
"""CPU regression tests for natural serving counter/window alignment."""

import sys
import unittest
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts/dsv41_bench/server_steps"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
pytestmark = pytest.mark.cpu_test

from bench_decode import summarize  # noqa: E402
from server_step_worker import SCENARIOS, classify_rows, token_counts  # noqa: E402


def records(count: int = 20) -> list:
    scenario = SCENARIOS["decode"]
    ranks = []
    for rank in range(8):
        steps = []
        for i in range(count):
            stage = {
                "forward_id": i + 1,
                "stage": "encoder",
                "phase": "prefill" if i == 0 else "decode",
                "requests": 16,
                "rows": 96,
                "padded_rows": 128,
                "starts": [65536 + i * 3] * 16,
                "ends": [65542 + i * 3] * 16,
                "prompts": [65536] * 16,
            }
            steps.append(
                {
                    "start_ns": i * 200_000_000,
                    "observed_ns": i * 200_000_000 + 100_000_000 + rank * 1_000_000,
                    "observation": "native_output_consumption",
                    "stages": [
                        stage,
                        {**stage, "stage": "decoder"},
                        {
                            "stage": "draft_inputs",
                            "requests": 16,
                            "rows": 80,
                            "padded_rows": 128,
                        },
                    ],
                    "scheduled_draft_lengths": {f"r{rank}-{j}": 5 for j in range(16)},
                    "accepted_output_tokens": 16 * (1 + i % 3),
                    "output_requests": 16,
                    "output_request_ids": [f"r{rank}-{j}" for j in range(16)],
                    "accepted_output_counts": [1 + i % 3] * 16,
                }
            )
        ranks.append(
            {
                "rank": rank,
                "hostname": "node",
                "scenario": scenario.to_dict(),
                "steps": steps,
            }
        )
    return ranks


class AccountingTest(unittest.TestCase):
    def test_counts_align_with_observation_intervals(self):
        result = summarize(records(), "decode")["categories"]["decode"]
        expected = sum(128 * (1 + i % 3) for i in range(2, 18)) + 16
        self.assertEqual(result["steps"], 17)
        self.assertEqual(result["effective_tokens"], expected)
        self.assertAlmostEqual(result["global_tokens_s"], expected / (17 * 0.2 - 0.007))
        self.assertAlmostEqual(
            result["per_physical_chip_tokens_s"], expected / ((17 * 0.2 - 0.007) * 4)
        )

    def test_counter_includes_bonus(self):
        data = records()
        for rank in data:
            for step in rank["steps"]:
                step["accepted_output_counts"] = [6] * 16
                step["accepted_output_tokens"] = 96
        result = summarize(data, "decode")["categories"]["decode"]
        self.assertEqual(result["mean_tokens_per_request_step"], 6)

    def test_gap_is_not_joined(self):
        data = records(22)
        data[0]["steps"][11]["stages"][0]["requests"] = 15
        result = summarize(data, "decode")
        self.assertEqual(len(result["windows"]), 2)
        self.assertFalse(
            any(
                s["forward_id"] in (12, 13)
                for s in result["categories"]["decode"]["samples"]
            )
        )

    def test_length_cap_neighborhood_excluded(self):
        data = records(100)
        for rank in data:
            for step in rank["steps"]:
                step["accepted_output_counts"] = [6] * 16
                step["accepted_output_tokens"] = 96
        result = summarize(data, "decode")["categories"]["decode"]
        self.assertTrue(
            all(
                max(s["contexts_from_accepted_outputs"][0]) + 6 < 65536 + 512
                for s in result["samples"]
            )
        )
        self.assertLess(result["samples"][-1]["forward_id"], 85)

    def test_missing_or_duplicate_rank_rejected(self):
        for data in (records()[:-1], records() + records()[:1]):
            with self.assertRaisesRegex(ValueError, "every DP rank"):
                summarize(data, "decode")

    def test_legacy_data_rejected(self):
        data = records()
        del data[0]["steps"][0]["observation"]
        with self.assertRaisesRegex(ValueError, "legacy"):
            summarize(data, "decode")

    def test_partial_c_rejected(self):
        data = records()
        for step in data[0]["steps"]:
            step["stages"][0]["requests"] = 15
        with self.assertRaisesRegex(ValueError, "full-DP"):
            summarize(data, "decode")

    def test_invalid_counter_rejected(self):
        data = records()
        data[0]["steps"][4]["accepted_output_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "per-request"):
            summarize(data, "decode")

    def test_observation_before_dispatch_rejected(self):
        data = records()
        data[0]["steps"][5]["observed_ns"] = 0
        with self.assertRaisesRegex(ValueError, "precedes dispatch"):
            summarize(data, "decode")

    def test_immediate_empty_step_can_overtake_async_output(self):
        data = records()
        data[0]["steps"].insert(
            1,
            {
                "start_ns": 50_000_000,
                "observed_ns": 60_000_000,
                "observation": "native_output_consumption",
                "stages": [],
                "output_request_ids": [],
                "output_requests": 0,
                "accepted_output_counts": [],
                "accepted_output_tokens": 0,
            },
        )
        report = summarize(data, "decode")
        self.assertEqual(report["categories"]["decode"]["steps"], 17)

    def test_cross_node_rejected(self):
        data = records()
        data[0]["hostname"] = "other"
        with self.assertRaisesRegex(ValueError, "single node"):
            summarize(data, "decode")

    def test_missing_decoder_does_not_qualify(self):
        data = records()
        for step in data[0]["steps"]:
            step["stages"].pop(1)
        with self.assertRaisesRegex(ValueError, "full-DP"):
            summarize(data, "decode")

    def test_wrong_verify_shape_rejected(self):
        data = records()
        for step in data[0]["steps"]:
            step["stages"][0]["rows"] = 16
        with self.assertRaisesRegex(ValueError, "full-DP"):
            summarize(data, "decode")

    def test_wrong_draft_shape_rejected(self):
        data = records()
        for step in data[0]["steps"]:
            step["stages"][2]["rows"] = 16
        with self.assertRaisesRegex(ValueError, "full-DP"):
            summarize(data, "decode")

    def test_phase_and_replay_accounting(self):
        self.assertEqual(classify_rows([65535], [65540], [65536]), "mixed")
        self.assertEqual(token_counts([65408], [67452], 65536)["new_tokens"], 1916)


if __name__ == "__main__":
    unittest.main()
