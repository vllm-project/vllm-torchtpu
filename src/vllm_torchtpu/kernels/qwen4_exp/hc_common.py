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
"""Shared HC wrapper checks and the BF16 up-projection boundary."""

from collections.abc import Callable
from typing import TypeVar

import jax
import jax.numpy as jnp


def validate_mix_geometry(
    branches: int,
    hidden: int,
    token_tile: int,
    name: str,
    *,
    positive_tile: bool = True,
) -> None:
    if branches <= 0 or hidden <= 0 or hidden % 128:
        raise ValueError(f"{name} requires positive branches and 128-aligned hidden")
    if token_tile <= 0 or token_tile % 128:
        adjective = "positive " if positive_tile else ""
        raise ValueError(f"{name} requires {adjective}128-aligned token tiles")


def invalid_mix_rows(
    tokens: int, lowrank: int, token_tile: int, width: int, expected_width: int
) -> bool:
    return (
        tokens == 0
        or lowrank == 0
        or tokens % token_tile != 0
        or width != expected_width
    )


def invalid_mix_weights(
    weight: jax.Array, down: jax.Array, up: jax.Array, width: int, lowrank: int
) -> bool:
    return (
        weight.shape != (width,)
        or down.shape != (lowrank, width)
        or up.shape != (width, lowrank)
    )


def project_up(gate: jax.Array, up: jax.Array) -> jax.Array:
    """Round the FP32 dot to the BF16 Linear output before FP32 sigmoid."""
    return (
        jax.lax.dot_general(
            gate,
            up,
            dimension_numbers=(((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        .astype(jnp.bfloat16)
        .astype(jnp.float32)
    )


_MixState = TypeVar("_MixState")


def run_mix_tiles(
    one_step: Callable[[int | jax.Array, int, _MixState | None], _MixState],
    tiles: int,
) -> None:
    """Peel resident-state initialization before paired tile steps."""
    initial = one_step(0, 0, None)
    if tiles > 1:
        one_step(1, 1, initial)

    def step_pair(pair, carry):
        one_step(2 * pair, 0, initial)
        one_step(2 * pair + 1, 1, initial)
        return carry

    jax.lax.fori_loop(1, tiles // 2, step_pair, jnp.int32(0))
    if tiles > 1 and tiles % 2:
        one_step(tiles - 1, 0, initial)
