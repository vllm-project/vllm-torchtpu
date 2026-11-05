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
"""The prefill-layout expert pre-shard (``preshard prefill-experts``)
against the regrouping loader, on a small four-rank pre-shard."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_test

# Small stand-ins of the four expert families: [layers, experts, ...], in the
# decode pre-shard's file split.
FAMILIES = {
    "expert_gu": (
        (4, 4, 3, 8),
        "uint32",
        ["expert_gu_0.bin", "expert_gu_1.bin"],
        [[0, 1], [1, 4]],
    ),
    "expert_gus": ((4, 4, 2, 8), "uint8", ["expert_gus.bin"], [[0, 4]]),
    "expert_down": (
        (4, 4, 3, 6),
        "uint32",
        ["expert_down_0.bin", "expert_down_1.bin"],
        [[0, 3], [3, 4]],
    ),
    "expert_ds": ((4, 4, 2, 6), "uint8", ["expert_ds.bin"], [[0, 4]]),
}


def write_source(root, ranks):
    rng = np.random.default_rng(0)
    for rank in ranks:
        path = root / f"rank{rank:02d}"
        path.mkdir(parents=True)
        families = {}
        for name, (shape, dtype, files, ranges) in FAMILIES.items():
            for filename, (lo, hi) in zip(files, ranges, strict=True):
                values = rng.integers(0, 255, (hi - lo, *shape[1:])).astype(dtype)
                (path / filename).write_bytes(values.tobytes())
            families[name] = {
                "shape": list(shape),
                "dtype": dtype,
                "files": files,
                "layer_ranges": ranges,
            }
        (path / "index.json").write_text(
            json.dumps({"rank": rank, "layers": 93, "families": families})
        )


def test_prefill_experts_are_the_regrouped_experts(tmp_path):
    from tools.megakernel import preshard
    from vllm_torchtpu.megakernel.kimi_k3.kernels import rank_loader as rl

    source, output = tmp_path / "source", tmp_path / "prefill"
    write_source(source, range(4))
    args = SimpleNamespace(
        source=str(source), output=str(output), ranks="0-3", in_flight=2, verify=False
    )
    preshard.write_prefill_experts(args, print)
    assert preshard.verify_prefill_experts(args, print)

    # A pre-shard view (the serve view's symlinks) finds the layout.
    view = tmp_path / "view"
    view.mkdir()
    for rank in range(4):
        (view / f"rank{rank:02d}").symlink_to(source / f"rank{rank:02d}")
    (view / rl.PREFILL_EXPERTS).symlink_to(output / rl.PREFILL_EXPERTS)
    for rank in range(4):
        prefill = rl.prefill_expert_dir(view, rank)
        assert prefill is not None
        own = rl.RankDir(source, rank)
        quarters, part, parts = rl.prefill_quarters(rank)
        quarter_dirs = [rl.RankDir(source, q) for q in quarters]
        for name in rl.EXPERT_FAMILIES:
            pairs = zip(
                own.regrouped_expert_parts(name, quarter_dirs, part, parts),
                prefill.expert_parts(name),
                strict=True,
            )
            for (label, nbytes, want), (got_label, got_bytes, got) in pairs:
                assert (label, nbytes) == (got_label, got_bytes)
                expected, found = want(), got()
                assert expected.shape == found.shape and expected.dtype == found.dtype
                assert expected.tobytes() == found.tobytes()
    assert rl.prefill_expert_dir(source, 0) is None


def test_prefill_layout_of_another_rank_is_refused(tmp_path):
    from tools.megakernel import preshard
    from vllm_torchtpu.megakernel.kimi_k3.kernels import rank_loader as rl

    source, output = tmp_path / "source", tmp_path / "prefill"
    write_source(source, range(4))
    preshard.write_prefill_experts(
        SimpleNamespace(
            source=str(source),
            output=str(output),
            ranks="0-3",
            in_flight=1,
            verify=False,
        ),
        print,
    )
    index = output / rl.PREFILL_EXPERTS / "rank01" / "index.json"
    layout = json.loads(index.read_text())
    layout["families"]["expert_gu"]["shape"][1] += 1
    index.write_text(json.dumps(layout))
    view = tmp_path / "view"
    view.mkdir()
    for rank in range(4):
        (view / f"rank{rank:02d}").symlink_to(source / f"rank{rank:02d}")
    (view / rl.PREFILL_EXPERTS).symlink_to(output / rl.PREFILL_EXPERTS)
    with pytest.raises(ValueError, match="prefill layout"):
        rl.prefill_expert_dir(view, 1)
