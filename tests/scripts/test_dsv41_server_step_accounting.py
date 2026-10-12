# SPDX-License-Identifier: Apache-2.0
"""Adversarial checks for extend interval/counter alignment."""

import copy
import sys
import unittest
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts/dsv41_bench/server_steps"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from scenarios import SCENARIOS, token_counts  # noqa: E402
from summarize_extend import summarize_extend  # noqa: E402

pytestmark = pytest.mark.cpu_test


def fixture():
    ranks = []
    for rank in range(8):
        steps = []
        start = 65408
        for index in range(9):
            end = min(start + 2044, 81920)
            stage = dict(
                stage="encoder",
                phase="prefill",
                starts=[start],
                ends=[end],
                prompts=[81920],
                requests=1,
                rows=end - start,
                padded_rows=2048,
                forward_id=index + 1,
                **token_counts([start], [end], 65536),
            )
            steps.append(
                dict(
                    start_ns=1000000000 + index * 200000000,
                    observed_ns=1100000000 + index * 200000000 + rank * 1000,
                    observation="native_output_consumption",
                    output_path="async_discard_without_wait",
                    stages=[stage],
                )
            )
            start = end
        ranks.append(
            dict(
                rank=rank,
                hostname="one",
                scenario=SCENARIOS["encoder_extend"].to_dict(),
                steps=steps,
            )
        )
    return ranks


class AccountingTest(unittest.TestCase):
    def test_counter_interval_alignment(self):
        report = summarize_extend(fixture())
        middle = report["categories"]["encoder_middle"]
        self.assertEqual(middle["steps"], 7)
        self.assertEqual(middle["effective_tokens"], 7 * 8 * 2044)
        self.assertAlmostEqual(middle["total_seconds"], 1.4)
        self.assertAlmostEqual(middle["global_tokens_s"], 81760)
        self.assertFalse(report["capacity_qualified"])

    def test_reject_corruption(self):
        for mutation in [
            lambda r: r.pop(),
            lambda r: r[0]["steps"].pop(3),
            lambda r: r[0]["steps"][2]["stages"].append({"stage": "decoder"}),
            lambda r: r[0]["steps"][2]["stages"][0].update(new_tokens=2048),
            lambda r: r[0]["steps"][2]["stages"][0].update(padded_rows=4096),
            lambda r: r[0]["steps"][2].update(observed_ns=0),
            lambda r: r[0]["steps"][2].update(observation="synchronized"),
        ]:
            ranks = copy.deepcopy(fixture())
            mutation(ranks)
            with self.assertRaises(ValueError):
                summarize_extend(ranks)


if __name__ == "__main__":
    unittest.main()
