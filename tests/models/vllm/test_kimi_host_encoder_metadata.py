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
"""The vision encoder's replay metadata must be computed on the host.

Two of its outputs are derived from tables that live on the TPU:
``rope_freqs_cis`` slices the rope table to the image's ``[:h, :w]`` and
``pos_embeds`` bicubic-interpolates the learned position embedding to the
image's exact ``(h, w)``. Both are device arithmetic keyed on the image size,
so each distinct image compiles a fresh XLA program -- one per image, measured
over the first 40 images of an MMMU run, created downstream of the budget
padding where no boot-time precompile reaches them.

``_host_vision_tower`` mirrors those tables on the host so the model's own code
runs against CPU copies. That only helps if it is numerically identical to what
it replaces, which is what these check; the mirror plumbing is checked
separately because a silent fall-through to the real module would keep the
numbers right and the recompilation intact.
"""

import types

import pytest
import torch
from vllm.model_executor.models.kimi_k25_vit import (
    Learnable2DInterpPosEmbDivided_fixed, Rope2DPosEmbRepeated)

from vllm_torchtpu.models.vllm.kimi_k3.model import (
    KimiK3ForConditionalGeneration, _HostMirror)

POS_EMB_SIDE = 14
DIM = 16
NUM_FRAMES = 4
ROPE_MAX = 64


def make_pos_emb():
    return Learnable2DInterpPosEmbDivided_fixed(height=POS_EMB_SIDE,
                                                width=POS_EMB_SIDE,
                                                num_frames=NUM_FRAMES,
                                                dim=DIM,
                                                interpolation_mode="bicubic")


def make_rope():
    return Rope2DPosEmbRepeated(dim=DIM,
                                max_height=ROPE_MAX,
                                max_width=ROPE_MAX)


def make_model():
    """Just enough of the model for ``_host_vision_tower`` to walk."""
    model = types.SimpleNamespace(vision_tower=types.SimpleNamespace(
        patch_embed=types.SimpleNamespace(pos_emb=make_pos_emb()),
        encoder=types.SimpleNamespace(rope_2d=make_rope()),
    ))
    model._host_vision_tower = types.MethodType(
        KimiK3ForConditionalGeneration._host_vision_tower, model)
    return model


# -- the mirror itself --


def test_an_override_wins_over_the_wrapped_module():
    module = types.SimpleNamespace(table=torch.ones(3))
    mirror = _HostMirror(module, table=torch.zeros(3))

    assert torch.equal(mirror.table, torch.zeros(3))
    assert torch.equal(module.table, torch.ones(3))


def test_an_attribute_with_no_override_falls_through():
    module = types.SimpleNamespace(table=torch.ones(3), width=7)

    assert _HostMirror(module, table=torch.zeros(3)).width == 7


def test_a_method_reached_through_the_mirror_sees_the_overrides():
    """The whole point: the model's own code must read the host copies.

    Without rebinding, ``mirror.method`` is still bound to the wrapped module,
    so every ``self.`` inside it reads the device table and the mirror silently
    does nothing.
    """

    class Table:

        def __init__(self):
            self.rows = torch.ones(3)

        def total(self):
            return self.rows.sum()

    module = Table()

    assert _HostMirror(module, rows=torch.full((3, ), 5.0)).total() == 15.0


def test_a_nested_mirror_is_reached_through_an_outer_method():

    class Inner:

        def __init__(self):
            self.rows = torch.ones(3)

        def total(self):
            return self.rows.sum()

    class Outer:

        def __init__(self):
            self.inner = Inner()

        def total(self):
            return self.inner.total()

    outer = Outer()
    mirror = _HostMirror(outer,
                         inner=_HostMirror(outer.inner,
                                           rows=torch.full((3, ), 2.0)))

    assert mirror.total() == 6.0


def test_a_plain_function_attribute_is_not_rebound():
    """Only methods bound to the wrapped module get re-pointed at the mirror."""
    module = types.SimpleNamespace(helper=lambda x: x + 1)

    assert _HostMirror(module).helper(1) == 2


def test_a_missing_attribute_still_raises():
    with pytest.raises(AttributeError):
        _HostMirror(types.SimpleNamespace()).nope


# -- the Kimi tables --


@pytest.mark.parametrize("grid", [
    [[1, 14, 14]],
    [[1, 20, 32]],
    [[1, 37, 41]],
    [[2, 20, 32]],
    [[1, 20, 32], [1, 37, 41], [1, 14, 14]],
])
def test_host_pos_embeds_match_the_module(grid):
    model = make_model()
    pos_emb = model.vision_tower.patch_embed.pos_emb

    mirrored = model._host_vision_tower().patch_embed.pos_emb.get_pos_embeds(
        grid)

    torch.testing.assert_close(mirrored, pos_emb.get_pos_embeds(grid))


@pytest.mark.parametrize("grid", [
    [[1, 20, 32]],
    [[2, 37, 41]],
    [[1, 20, 32], [1, 37, 41]],
])
def test_host_rope_freqs_match_the_module(grid):
    model = make_model()
    rope = model.vision_tower.encoder.rope_2d

    mirrored = model._host_vision_tower().encoder.rope_2d.get_freqs_cis(
        grid, device=torch.device("cpu"))

    torch.testing.assert_close(
        mirrored, rope.get_freqs_cis(grid, device=torch.device("cpu")))


def test_the_mirrored_tables_are_copies_on_the_host():
    model = make_model()
    pos_emb = model.vision_tower.patch_embed.pos_emb

    mirrored = model._host_vision_tower().patch_embed.pos_emb

    assert mirrored.weight.device.type == "cpu"
    assert mirrored.weight is not pos_emb.weight
    assert mirrored.time_weight is not pos_emb.time_weight


def test_the_rope_table_is_built_without_touching_the_module():
    """It is created lazily on the device by whoever calls first, so the mirror
    cannot assume the buffer already exists to be copied."""
    model = make_model()

    freqs = model._host_vision_tower().encoder.rope_2d.freqs_cis

    assert freqs.device.type == "cpu"
    assert not hasattr(model.vision_tower.encoder.rope_2d, "freqs_cis")


def test_the_mirror_is_built_once():
    model = make_model()

    assert model._host_vision_tower() is model._host_vision_tower()
