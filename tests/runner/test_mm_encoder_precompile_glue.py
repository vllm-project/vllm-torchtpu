# Copyright 2025 Google LLC
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
"""Boot-time precompilation must cover the glue, not just the encoder graph.

``_capture_budget_graph`` hands the device-resident template straight to the
compiled forward. A serving request also runs ``_pad_to_template`` before it
and ``postprocess_encoder_output`` after it, and those are plain lazy-tensor
ops -- so any shape they reach that the capture did not compiles at runtime.
``_capture_budget_glue`` enumerates them at boot; these tests check the
enumeration is actually a superset of what requests produce, which is the only
property that makes it worth running.
"""

import types

import pytest
import torch

from vllm_torchtpu.runner import mm_encoder_manager
from vllm_torchtpu.runner.mm_encoder_manager import (MMEncoderManager, _brief,
                                                     _tensors_of)

HIDDEN = 8
PACKED_ROWS = 1024  # rows the compiled forward returns for this budget
BUDGET = 2048


class BucketedModel:
    """The duck-typed contract ``_capture_budget_glue`` primes against.

    The manager never names a model class -- it looks for ``mm_embed_bucket``
    and calls ``postprocess_encoder_output``. Spelling the contract out here
    keeps this test on the generic machinery; that a real model implements it
    is the model's own test to make.
    """

    MM_EMBED_BUCKET_MIN = 64

    @classmethod
    def mm_embed_bucket(cls, num_tokens: int) -> int:
        bucket = cls.MM_EMBED_BUCKET_MIN
        while bucket < num_tokens:
            bucket *= 2
        return bucket

    def postprocess_encoder_output(self,
                                   outputs,
                                   indices,
                                   per_item_out_tokens,
                                   dest,
                                   clone=False,
                                   **kw):
        output = outputs["default"]
        last_row = output.shape[0] - 1
        offset = 0
        for idx, num_tokens in zip(indices, per_item_out_tokens):
            src_idx = torch.arange(offset,
                                   offset + self.mm_embed_bucket(num_tokens),
                                   dtype=torch.long)
            src_idx.clamp_(max=last_row)
            dest[idx] = output.index_select(0, src_idx.to(output.device))
            offset += num_tokens


def make_manager(record):
    """A manager with the fields the two glue paths touch, and a stub model.

    Every ``postprocess_encoder_output`` call records its (source rows,
    gathered rows) pair. That pair is the XLA program key, so two calls with
    the same pair share a program.
    """
    stub = BucketedModel()
    gather = stub.postprocess_encoder_output

    def postprocess(outputs,
                    indices,
                    per_item_out_tokens,
                    dest,
                    clone=False,
                    **kw):
        gather(outputs, indices, per_item_out_tokens, dest, clone, **kw)
        rows = outputs["default"].shape[0]
        for idx in indices:
            record.add((rows, dest[idx].shape[0]))

    stub.postprocess_encoder_output = postprocess

    mgr = object.__new__(MMEncoderManager)
    mgr.model = stub
    mgr.device = torch.device("cpu")
    mgr.dtype = torch.float32
    mgr.config = types.SimpleNamespace(padding_logics={})
    mgr._host_stage = {}
    mgr._host_stage_dirty = {}
    mgr.budget_templates = {
        "default": {
            BUDGET: {
                "pixel_values": torch.randn(BUDGET, 16),
                "grid_thws": torch.zeros(8, 3, dtype=torch.long),
            }
        }
    }
    return mgr


def test_primed_shapes_cover_real_requests():
    primed = set()
    mgr = make_manager(primed)
    out = torch.randn(PACKED_ROWS, HIDDEN)
    mgr._capture_budget_glue(out, BUDGET)
    assert primed, "the warmup primed nothing"

    # Batches a request can actually produce at this budget: one big item, a
    # mix of sizes, and counts that are not powers of two.
    requested = set()
    mgr2 = make_manager(requested)
    for per_item in ([1000], [7], [64], [300, 200, 130], [1, 1, 1,
                                                          1], [500,
                                                               129], [1024]):
        dest: dict = {}
        mgr2.model.postprocess_encoder_output({"default": out},
                                              list(range(len(per_item))),
                                              per_item,
                                              dest,
                                              clone=True)
    assert requested <= primed, (
        f"shapes a request reaches but boot never compiled: "
        f"{sorted(requested - primed)}")


def test_pad_warmup_leaves_a_clean_staging_buffer():
    """The padding contract is 'everything past the real rows is zero'."""
    mgr = make_manager(set())
    mgr._capture_budget_glue(torch.randn(PACKED_ROWS, HIDDEN), BUDGET)

    assert mgr._host_stage, "the warmup never routed through the staging path"
    for key, stage in mgr._host_stage.items():
        assert not stage.any(), f"{key} left dirty rows behind"
        assert mgr._host_stage_dirty[key] == 0

    # A short request after the warmup must still see zeros past its own rows.
    padded = mgr._pad_to_template({"pixel_values": torch.ones(3, 16)}, BUDGET)
    assert padded["pixel_values"].shape[0] == BUDGET
    assert not padded["pixel_values"][3:].any()


def test_optional_none_entries_do_not_abort_the_warmup(monkeypatch, caplog):
    """An optional ``None`` in a template must not sink the whole prime.

    ``_pad_to_template`` passes template entries through untouched when the
    request does not supply them, so a template holding ``None`` for an
    optional input puts a ``None`` in the padded dict. Handing that to
    ``synchronize_tensors`` fails the pybind signature for the *entire* list,
    which silently disabled the padding half of the warmup on Kimi K3.
    """
    mgr = make_manager(set())
    mgr.budget_templates["default"][BUDGET]["optional_thing"] = None

    seen = []
    monkeypatch.setattr(mm_encoder_manager,
                        "synchronize_tensors",
                        lambda tensors, wait=False: seen.append(tensors))

    with caplog.at_level("WARNING"):
        mgr._capture_budget_glue(torch.randn(PACKED_ROWS, HIDDEN), BUDGET)

    assert "Failed to precompile" not in caplog.text
    assert seen, "nothing was synchronized, so nothing was primed"
    for batch in seen:
        assert all(isinstance(t, torch.Tensor) for t in batch)
    # Still zeroed the staging buffer despite the extra template entry.
    assert all(not stage.any() for stage in mgr._host_stage.values())


def test_a_failed_prime_drains_the_lazy_graph(monkeypatch, caplog):
    """A warmup step that fails must not leave its ops queued.

    Traced-but-unsynchronized ops stay pending and fuse into the next budget's
    encoder program, so each budget compiles a strictly larger graph. That is
    not a slow warmup, it is a divergent one: it took the encoder precompile
    from 52s total to 5, then 8, then more minutes per budget.
    """
    mgr = make_manager(set())

    def explode(*a, **kw):
        raise RuntimeError("nope")

    mgr.model.postprocess_encoder_output = explode

    drained = []
    monkeypatch.setattr(
        mm_encoder_manager,
        "synchronize_tensors",
        lambda tensors=None, wait=False: drained.append(tensors))

    with caplog.at_level("WARNING"):
        mgr._capture_budget_glue(torch.randn(PACKED_ROWS, HIDDEN), BUDGET)

    assert "Failed to precompile the postprocess path" in caplog.text
    assert None in drained, "the failure left the pending graph queued"


def test_tensors_of_keeps_only_tensors():
    values = {"a": torch.zeros(2), "b": None, "c": 7, "d": torch.ones(1)}
    assert [t.shape[0] for t in _tensors_of(values)] == [2, 1]


def test_brief_truncates_and_flattens():
    long = _brief(RuntimeError("x\ny\n" + "z" * 1000))
    assert len(long) < 320
    assert "\n" not in long
    assert long.startswith("x y zzz")
    assert long.endswith("[...]")
    assert _brief(RuntimeError("short")) == "short"


@pytest.mark.parametrize("packed_rows", [64, 256, 4096])
def test_priming_stops_at_the_packed_output(packed_rows):
    """Buckets larger than the source tensor cannot occur, so skip them."""
    primed = set()
    mgr = make_manager(primed)
    mgr._capture_budget_glue(torch.randn(packed_rows, HIDDEN), BUDGET)
    assert primed == {(packed_rows, b)
                      for b in (64, 128, 256, 512, 1024, 2048, 4096)
                      if b <= packed_rows}
