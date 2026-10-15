# SPDX-License-Identifier: Apache-2.0
"""CPU regression tests for natural serving counter/window alignment."""

import datetime
import sys
import unittest
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts/dsv41_bench/server_steps"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
pytestmark = pytest.mark.cpu_test

from bench_cached_extend import assert_no_compilation  # noqa: E402
from bench_decode import summarize  # noqa: E402
from server_step_worker import SCENARIOS, classify_rows, token_counts  # noqa: E402


@pytest.mark.parametrize("color", ["", "\x1b[90m"])
def test_compilation_audit_with_server_log_colors(color):
    start = int(datetime.datetime(2026, 10, 15, 7, 8).timestamp() * 1e9)
    intervals = [[start, start + 10_000_000_000]]

    def line(stamp):
        return (
            f"(Worker_DP0_EP0 pid=761) INFO {color}{stamp}\x1b[0m "
            "[monitor.py:40] [v41_encoder] Starting torch.compile"
        )

    assert_no_compilation(line("10-15 06:44:06"), intervals)
    assert_no_compilation(line("10-15 07:10:00"), intervals)
    with pytest.raises(ValueError, match="Compilation in formal serving window"):
        assert_no_compilation(line("10-15 07:08:05"), intervals)
    with pytest.raises(ValueError, match="Cannot timestamp compiler log event"):
        assert_no_compilation(line("missing timestamp"), intervals)


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
                "requests": 48,
                "rows": 288,
                "padded_rows": 384,
                "starts": [98304 + i * 3] * 48,
                "ends": [98310 + i * 3] * 48,
                "prompts": [98304] * 48,
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
                            "requests": 48,
                            "rows": 240,
                            "padded_rows": 384,
                        },
                    ],
                    "scheduled_draft_lengths": {f"r{rank}-{j}": 5 for j in range(48)},
                    "accepted_output_tokens": 48 * (1 + i % 3),
                    "output_requests": 48,
                    "output_request_ids": [f"r{rank}-{j}" for j in range(48)],
                    "accepted_output_counts": [1 + i % 3] * 48,
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
        expected = sum(384 * (1 + i % 3) for i in range(2, 18)) + 48
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
                step["accepted_output_counts"] = [6] * 48
                step["accepted_output_tokens"] = 288
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
                step["accepted_output_counts"] = [6] * 48
                step["accepted_output_tokens"] = 288
        result = summarize(data, "decode")["categories"]["decode"]
        self.assertTrue(
            all(
                max(s["contexts_from_accepted_outputs"][0]) + 6 < 98304 + 512
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
        self.assertEqual(classify_rows([98303], [98308], [98304]), "mixed")
        self.assertEqual(token_counts([65408], [67452], 65536)["new_tokens"], 1916)


if __name__ == "__main__":
    unittest.main()
