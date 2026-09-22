#!/usr/bin/env python3
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Compare two already-analyzed MoE UT runs, requiring identical routing/input."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("standard", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    a = json.loads((args.standard / "summary.json").read_text())
    b = json.loads((args.candidate / "summary.json").read_text())
    config_a, config_b = dict(a["configuration"]), dict(b["configuration"])
    config_a.pop("moe_impl", None)
    config_b.pop("moe_impl", None)
    assert config_a == config_b, "Different benchmark configurations"
    assert set(a["cases"]) == set(b["cases"]), "Different routing pools"
    result = {
        "configuration": config_a,
        "standard": str(args.standard),
        "candidate": str(args.candidate),
        "scope": "TPU:0 mean TensorCore XLA Ops duration over captured forwards",
        "cases": {},
    }
    for case in a["cases"]:
        with (
            np.load(args.standard / case / "fixture.npz") as fa,
            np.load(args.candidate / case / "fixture.npz") as fb,
        ):
            for key in [
                "hidden",
                "router_weight",
                "router_bias",
                "topk_ids",
                "topk_weights",
                "expert_counts",
            ]:
                np.testing.assert_array_equal(
                    fa[key], fb[key], err_msg=f"{case}: {key} differs"
                )
            expected, actual = fa["baseline_output"], fb["baseline_output"]
            assert np.isfinite(actual).all()
            rel = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
            worst = float(
                np.max(
                    np.linalg.norm(actual - expected, axis=1)
                    / np.maximum(np.linalg.norm(expected, axis=1), 1e-20)
                )
            )
            assert rel < 0.008 and worst < 0.015, (case, rel, worst)
        da = a["cases"][case]["device_timing"]["per_device"]["/device:TPU:0"]
        db = b["cases"][case]["device_timing"]["per_device"]["/device:TPU:0"]
        ta, tb = da["mean"]["op_sum_us"], db["mean"]["op_sum_us"]
        result["cases"][case] = {
            "inputs_and_routes_equal": True,
            "relative_l2": rel,
            "worst_token_relative_l2": worst,
            "standard_op_us": ta,
            "candidate_op_us": tb,
            "latency_ratio": tb / ta,
            "latency_change_percent": (tb / ta - 1) * 100,
            "standard_module_us": da["mean"]["total_us"],
            "candidate_module_us": db["mean"]["total_us"],
            "standard_sort_count": da["sort_count_per_step"],
            "candidate_sort_count": db["sort_count_per_step"],
            "standard_details": da["mean"],
            "candidate_details": db["mean"],
        }
        print(
            case,
            f"{ta:.2f} -> {tb:.2f} us ({(tb / ta - 1) * 100:+.1f}%), relative L2={rel:.5f}",
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
