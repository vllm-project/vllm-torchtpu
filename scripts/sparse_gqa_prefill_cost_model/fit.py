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
"""Reproduce the frozen prefill cost fit from saved calibration timings."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

# Numerical conditioning scales; divided back out before reporting coefficients.
QUERY_SCALE = 8192
MASK_WORD_SCALE = 2048
TC_VISIT_SCALE = 32768
SC_TILE_SCALE = 9216
MIN_WEIGHT_TIME_MS = 0.25
CASES_PER_DTYPE = 30

rows = json.loads(Path("calibration/results.json").read_text())
fits = {}
for dtype in ("bf16", "fp8"):
    data = [x for x in rows if x["dtype"] == dtype]
    if len(data) != CASES_PER_DTYPE:
        continue
    fits[dtype] = {}
    for arm in ("tc", "sc"):
        features = []
        for r in data:
            f = r["features"]
            q = f["queries"]
            features.append(
                [
                    1,
                    q / QUERY_SCALE,
                    q * f["words"] / (QUERY_SCALE * MASK_WORD_SCALE),
                    f["tile_visits"] / TC_VISIT_SCALE,
                ]
                if arm == "tc"
                else [1, q / QUERY_SCALE, f["attention_tiles"] / SC_TILE_SCALE]
            )
        x = np.asarray(features)
        y = np.asarray([r["median_ms"][arm] for r in data])
        # Relative-error weighting keeps small prefill buckets represented.
        weight = 1 / np.maximum(y, MIN_WEIGHT_TIME_MS)
        coefficients, _ = nnls(x * weight[:, None], y * weight)
        scales = np.array(
            [1, QUERY_SCALE, QUERY_SCALE * MASK_WORD_SCALE, TC_VISIT_SCALE]
            if arm == "tc"
            else [1, QUERY_SCALE, SC_TILE_SCALE]
        )
        raw = (coefficients / scales).tolist()
        pred = x @ coefficients
        fits[dtype][arm] = {
            "coefficients_ms": raw,
            "mean_relative_error": float(np.mean(abs(pred - y) / y)),
            "max_relative_error": float(np.max(abs(pred - y) / y)),
            "cases": [
                {
                    "tokens": r["tokens"],
                    "context": r["context"],
                    "pattern": r["pattern"],
                    "active": r["active"],
                    "actual_ms": float(a),
                    "estimated_ms": float(b),
                }
                for r, a, b in zip(data, y, pred)
            ],
        }
Path("fit.json").write_text(json.dumps(fits, indent=2))
print(
    json.dumps(
        {
            dt: {
                arm: {k: v for k, v in a.items() if k != "cases"}
                for arm, a in d.items()
            }
            for dt, d in fits.items()
        },
        indent=2,
    )
)
