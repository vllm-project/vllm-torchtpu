# SPDX-License-Identifier: Apache-2.0
"""q38 L4 VRNG-VPS (VLLM_TPU_VPS_INGRAPH_RNG): the VPS verify's uniforms drawn inside its compiled program.

Today `_sample_spec_verify_vps` draws three eager `torch.rand` arrays on the sampling generator (bonus [B, V],
accept [T], recovery [T, V] at the full vocabulary V under VLLM_TPU_VPS_DRAWS=full), each its own uniform_ program
(T98g census, M: tt_jit_uniform_ 4.75 per step, 0.103-0.106 ms device per step), and the VPS program reads only the
entries at the merged candidate ids (plus F whole rows for the fallback). Here the program computes exactly the
entries it reads, from a device int32 counter, with dongruizi's counter-based hash (counter_uniform.py, cherry-picked
by DEC-FB-3; credited): the uniform of element ``i`` of one call is ``counter_uniform_at(counter, i)``, which equals
``counter_uniform((n,), counter)[i]`` bit for bit. Element indices follow DEC-FB-3's layout of the fused verify's
three draws as one draw split in the eager order (bonus, accept, recover):

    bonus   (row r, token j): r * V + j                      r < B, j < V
    accept  (slot t)        : B * V + t                      t < T_a = draft_token_ids.numel()
    recover (row t, token j): B * V + T_a + t * V + j        t < T

so the regions are disjoint, no element is used twice within a call, and the program is op for op today's
VLLM_TPU_VPS_DRAWS=full verify fed with those full arrays (the tests check this bit for bit). Same distribution as
the generator's draws, different random stream: class R (needs the accuracy gate on the stack that carries it).

``draws = (counter, vocab, counter_uniform_at)`` is what vp_sampler's verify functions take (vp_sampler itself imports
nothing from here, so the stub-package VPS test harness keeps loading it unchanged).
"""

import torch

from vllm_torchtpu.layers.adapter.sample.counter_uniform import (
    _FMIX_C1,  # noqa: F401  (re-exported for the tests' constant check)
    _GOLDEN,
    _KEY2_SALT,
    _fmix32,
    counter_uniform,
    initial_counter,
)

# The VPS stream's start: initial_counter(seed ^ _VPS_SALT), so it never equals DEC-FB-3's fused-verify stream.
_VPS_SALT = 0x56505352  # "VPSR"
_INT32_SPAN = 1 << 31


def counter_uniform_at(counter: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """float32 uniforms in [0, 1) at int32 element indices ``idx`` (any shape) of the draw keyed by ``counter``
    (a 0-d int32 tensor): ``counter_uniform((n,), counter)[idx]``, computed without the other elements."""
    k1 = _fmix32(counter * _GOLDEN)
    k2 = _fmix32(counter ^ _KEY2_SALT)
    h = _fmix32(_fmix32(idx ^ k1) + k2)
    return ((h >> 8) & 0xFFFFFF).to(torch.float32) * (1.0 / 16777216.0)


def vps_initial_counter(seed: int) -> int:
    """The VPS counter's start value for a model seed (int32)."""
    return initial_counter((int(seed) ^ _VPS_SALT) & 0xFFFFFFFF)


def layout(num_bonus: int, num_accept: int, num_target: int, vocab: int) -> tuple[int, int, int]:
    """(accept base, recover base, total elements) of one call; raises if the indices would leave int32."""
    total = (num_bonus + num_target) * vocab + num_accept
    if total >= _INT32_SPAN:
        raise ValueError(
            f"VRNG-VPS: {num_bonus} bonus + {num_target} target rows x vocab {vocab} + {num_accept} accept slots "
            f"= {total} elements, past int32"
        )
    return num_bonus * vocab, num_bonus * vocab + num_accept, total


def full_draws(counter: torch.Tensor, num_bonus: int, accept_shape: tuple, num_target: int, vocab: int):
    """The three full arrays the program reads from (bonus [B, V], accept ``accept_shape``, recover [T, V]): DEC-FB-3's
    `_verify_counter_uniforms`, one counter_uniform draw split in the eager order. Eager; check mode and tests only."""
    na = 1
    for d in accept_shape:
        na *= d
    a0, r0, total = layout(num_bonus, na, num_target, vocab)
    u = counter_uniform((total,), counter)
    return u[:a0].view(num_bonus, vocab), u[a0:r0].view(accept_shape), u[r0:].view(num_target, vocab)
