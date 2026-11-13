# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_PREFLAT_FWD_ARGS and the owned-argument fix behind
VLLM_TPU_SUBST_STATIC_ARGS. Pure CPU: torch, envs, token_padding, owned_args.

Run: PYTHONPATH=src python3 -m pytest --noconftest tests/runner/test_preflat_fwd_args.py
Bench: PYTHONPATH=src python3 tests/runner/test_preflat_fwd_args.py
"""

import random
import time

import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.runner.owned_args import (
    ensure_owned,
    is_owned,
    is_stamped,
    static_shape_args_owned,
)


def _load_token_padding():
    """The module itself; without torch_tpu (local CPU) the `layers` package
    cannot be imported, so load the file, which needs only torch and envs."""
    try:
        from vllm_torchtpu.layers.adapter import token_padding as mod
    except ModuleNotFoundError:
        import importlib.util
        import pathlib
        import sys

        import vllm_torchtpu

        path = (
            pathlib.Path(vllm_torchtpu.__file__).parent
            / "layers"
            / "adapter"
            / "token_padding.py"
        )
        spec = importlib.util.spec_from_file_location("_token_padding_cpu", path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    return mod


token_padding = _load_token_padding()
TokenPaddingState = token_padding.TokenPaddingState
PREFLAT_MAX_PADDED_TOKENS = token_padding.PREFLAT_MAX_PADDED_TOKENS

# Served geometry: 8192-token max bucket; the c4 crash buffer was 8192 + 48.
MAX_TOKENS = 8192
BUCKETS = [4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 384, 512, 1024, 2048, 8192]
K = 3


def _bucket(n: int) -> int:
    return next(b for b in BUCKETS if b >= n)


def _flags(monkeypatch, preflat: bool, reuse: bool) -> None:
    monkeypatch.setattr(envs, "VLLM_TPU_PREFLAT_FWD_ARGS", preflat, raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_TOKEN_PADDING_REUSE", reuse, raising=False)


def _reference(key: tuple[int, int]) -> torch.Tensor:
    ref = torch.zeros(MAX_TOKENS, dtype=torch.bool)
    ref[key[0] : key[1]] = True
    return ref


def _random_keys(seed: int, steps: int) -> list[tuple[int, int]]:
    """Target-forward (valid, padded) rows over a served-like run: steady
    decode/verify at c1..c4 (n_req * (1 + K) rows), request churn, and the
    occasional prefill / mixed chunk past the cache bound."""
    rng = random.Random(seed)
    keys = []
    n_req = 1
    for _ in range(steps):
        r = rng.random()
        if r < 0.1:
            n_req = rng.randint(1, 4)  # churn
        if r < 0.15:
            valid = rng.randint(1, 4096)  # prefill / mixed chunk
        elif r < 0.2:
            valid = rng.randint(1, 64)  # short prefill inside the bound
        else:
            valid = n_req * (1 + K)
        keys.append((valid, _bucket(valid)))
    return keys


@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_padding_mask_preflat_matches_rebuild(monkeypatch, seed, reuse):
    """Same mask values, shape and dtype as the rebuild path after every
    update, over random steps, bucket changes and churn; every buffer handed
    out is owned and full-size (never a view into another buffer)."""
    keys = _random_keys(seed, 600)
    _flags(monkeypatch, preflat=False, reuse=False)
    old = TokenPaddingState.create(MAX_TOKENS, "cpu")
    new = TokenPaddingState.create(MAX_TOKENS, "cpu")
    base = new.local_padding_mask
    for key in keys:
        _flags(monkeypatch, preflat=False, reuse=False)
        old.update(*key)
        _flags(monkeypatch, preflat=True, reuse=reuse)
        new.update(*key)
        got = new.local_padding_mask
        assert got.shape == old.local_padding_mask.shape == (MAX_TOKENS,)
        assert got.dtype == torch.bool
        assert torch.equal(got, old.local_padding_mask)
        assert torch.equal(got, _reference(key))
        assert is_owned(got)
        n = key[1]
        assert torch.equal(new.get_local_padding_mask(n), old.get_local_padding_mask(n))
        # Past the bound the original buffer is written, as without the flag.
        assert (got is base) == (key[1] > PREFLAT_MAX_PADDED_TOKENS)
    # No fallback write ever landed in a cached per-key buffer.
    for key, (dev, host) in new._preflat_masks.items():
        assert torch.equal(dev, _reference(key))
        assert torch.equal(host, _reference(key))


def test_padding_mask_preflat_skips_unchanged_and_never_writes_in_place(
    monkeypatch,
):
    """A steady step hands back the same owned buffer; a new key swaps in a
    different one instead of writing the buffer a forward may still read."""
    _flags(monkeypatch, preflat=True, reuse=False)
    state = TokenPaddingState.create(MAX_TOKENS, "cpu")
    state.update(4, 4)
    first = state.local_padding_mask
    first_values = first.clone()
    state.update(4, 4)
    assert state.local_padding_mask is first
    state.update(12, 16)
    assert state.local_padding_mask is not first
    assert torch.equal(first, first_values)  # the in-flight buffer is untouched
    state.update(4, 4)
    assert state.local_padding_mask is first  # cached per key


def test_padding_mask_fallback_after_swap_rewrites_base(monkeypatch):
    """REUSE must not trust a key the original buffer never held: after a
    swap, a past-the-bound step rewrites the original buffer."""
    _flags(monkeypatch, preflat=False, reuse=True)
    state = TokenPaddingState.create(MAX_TOKENS, "cpu")
    state.update(100, 128)  # base holds (100, 128), REUSE key (100, 128)
    _flags(monkeypatch, preflat=True, reuse=True)
    state.update(4, 4)  # swapped
    state.update(100, 128)  # base again: may skip, base already holds it
    assert torch.equal(state.local_padding_mask, _reference((100, 128)))
    state.update(8, 8)
    state.update(200, 256)  # base must be rewritten
    assert torch.equal(state.local_padding_mask, _reference((200, 256)))
    # Flag turned off mid-run: back on the original buffer, exact values.
    state.update(4, 4)
    _flags(monkeypatch, preflat=False, reuse=True)
    state.update(4, 4)
    assert state.local_padding_mask is state._preflat_base
    assert torch.equal(state.local_padding_mask, _reference((4, 4)))


def test_padding_mask_flag_off_is_noop(monkeypatch):
    """Flag off: the one original buffer is written in place every step and
    no per-key state is created."""
    _flags(monkeypatch, preflat=False, reuse=False)
    state = TokenPaddingState.create(MAX_TOKENS, "cpu")
    base = state.local_padding_mask
    for key in _random_keys(3, 50):
        state.update(*key)
        assert state.local_padding_mask is base
        assert torch.equal(base, _reference(key))
    assert not any(k.startswith("_preflat") for k in state.__dict__)


def test_routing_reads_follow_swapped_mask(monkeypatch):
    """Consumers read `_padding_state.local_padding_mask` at call time, so a
    swapped buffer gives the same routing as a rewritten one."""
    weights = torch.rand(16, 2)
    ids = torch.zeros(16, 2, dtype=torch.long)
    outs = []
    for preflat in (False, True):
        _flags(monkeypatch, preflat=preflat, reuse=False)
        state = TokenPaddingState.create(MAX_TOKENS, "cpu")
        token_padding.set_padding_state(state)
        try:
            state.update(5, 16)
            outs.append(
                token_padding.zero_routing_weights_for_padding(
                    ids, weights, is_local_tensor=True
                )[1]
            )
        finally:
            token_padding.set_padding_state(None)
    assert torch.equal(outs[0], outs[1])


# --- view-vs-owned (the c4 boot crash class) -------------------------------


def test_view_of_token_buffer_is_not_owned():
    """The crash signature: a [64] slice of an [8192 + 48] buffer is a view
    whose backing buffer is the whole thing (torch_tpu launches the program
    with s32[8240]). ensure_owned gives a [64] tensor owning exactly 64."""
    big = torch.arange(MAX_TOKENS + 48, dtype=torch.int32)
    view = big[:64]
    assert not is_owned(view)
    assert view.untyped_storage().nbytes() == (MAX_TOKENS + 48) * 4
    owned = ensure_owned(view)
    assert is_owned(owned)
    assert owned.shape == (64,) and owned.is_contiguous()
    assert owned.untyped_storage().nbytes() == 64 * 4
    assert torch.equal(owned, view)
    # Offset slices and full-size views are caught too; owned passes through.
    assert not is_owned(big[16:80])
    assert not is_owned(big[:])
    assert ensure_owned(owned) is owned


def _stamped(t: torch.Tensor) -> torch.Tensor:
    torch._dynamo.mark_dynamic(t, 0)
    assert is_stamped(t)
    return t


def test_old_static_args_strip_hands_a_view():
    """`eagle3._static_shape_args` strips the stamp with `a[:]`: a view (of
    the caller's buffer, or of a larger one when the caller holds a slice)."""
    big = torch.zeros(MAX_TOKENS + 48, dtype=torch.int32)
    ids = _stamped(big[:64])
    stripped = ids[:]  # what _static_shape_args passes on
    assert stripped._base is not None
    assert not is_owned(stripped)


def test_static_shape_args_owned_hands_owned_unstamped_and_returns_caller_tensor():
    seen = []

    def fn(x, y):
        seen.append(x)
        x.add_(y)
        return x

    wrapped = static_shape_args_owned(fn)
    ids = _stamped(torch.arange(16, dtype=torch.int32))
    out = wrapped(ids, torch.ones(16, dtype=torch.int32))
    (inner,) = seen
    assert not is_stamped(inner) and is_owned(inner)
    assert inner.shape == ids.shape and inner.dtype == ids.dtype
    assert out is ids  # the caller's own object, as with the flag off
    assert torch.equal(ids, torch.arange(1, 17, dtype=torch.int32))
    assert is_stamped(ids)  # the forward's stamp is left as it was

    # A stamped view of a larger buffer reaches fn as an owned [64] copy.
    big = torch.zeros(MAX_TOKENS + 48, dtype=torch.int32)
    seen.clear()
    out = wrapped(_stamped(big[:64]), torch.ones(64, dtype=torch.int32))
    (inner,) = seen
    assert is_owned(inner) and inner.shape == (64,)
    assert out is inner and torch.equal(out, torch.ones(64, dtype=torch.int32))

    # Unstamped arguments pass through untouched.
    plain = torch.zeros(4, dtype=torch.int32)
    seen.clear()
    assert wrapped(plain, torch.ones(4, dtype=torch.int32)) is plain
    assert seen[0] is plain


def _substitute(input_ids, cur_idx, pre_idx, next_tokens):
    """tpu_runner._substitute_placeholder_token's body (tpu_runner needs vllm)."""
    mask = pre_idx > -1
    safe_idx = torch.clamp_min(pre_idx, 0)
    new_values = next_tokens[safe_idx].to(input_ids.dtype)
    original = input_ids[cur_idx]
    input_ids.scatter_(0, cur_idx, torch.where(mask, new_values, original))
    return input_ids


@pytest.mark.parametrize("seed", [0, 1])
def test_substitution_owned_static_args_bitwise(seed):
    """The compiled substitution through `static_shape_args_owned` (stamped
    ids, as after a forward) gives the same ids as the plain call, mutates
    the caller's tensor in place and returns it, over random steps."""
    compiled = torch.compile(_substitute, backend="aot_eager", fullgraph=True)
    owned = static_shape_args_owned(compiled)
    rng = torch.Generator().manual_seed(seed)
    for _ in range(20):
        n_req = int(torch.randint(1, 5, (1,), generator=rng))
        n = _bucket(n_req * (1 + K))
        real = n_req * (1 + K)
        next_tokens = torch.randint(0, 1000, (16 * (1 + K),), generator=rng)
        cur = torch.cat([torch.randperm(n, generator=rng)])
        pre = torch.full((n,), -1, dtype=torch.int32)
        pre[:real] = torch.randint(0, next_tokens.numel(), (real,), generator=rng).to(
            torch.int32
        )
        ids = torch.randint(0, 1000, (n,), generator=rng, dtype=torch.int32)
        want = _substitute(ids.clone(), cur, pre, next_tokens)
        stamped = _stamped(ids.clone())
        got = owned(stamped, cur, pre, next_tokens)
        assert got is stamped
        assert torch.equal(got, want) and got.dtype == want.dtype


# --- microbench -------------------------------------------------------------


def _bench(fn, iters: int = 2000) -> float:
    for _ in range(50):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1e6


def microbench() -> dict[str, float]:
    """µs per step of `TokenPaddingState.update` (CPU stand-in for the H2D)."""
    res = {}
    shapes = {
        "c1": [(4, 4)],  # steady verify, 1 request
        # 4 requests with churn across buckets 4/8/16
        "c4": [(16, 16), (16, 16), (12, 16), (16, 16), (8, 8), (16, 16), (4, 4)],
    }
    for name, keys in shapes.items():
        for label, preflat, reuse in (
            ("old", False, False),
            ("reuse", False, True),
            ("preflat", True, False),
        ):
            envs.VLLM_TPU_PREFLAT_FWD_ARGS = preflat
            envs.VLLM_TPU_TOKEN_PADDING_REUSE = reuse
            state = TokenPaddingState.create(MAX_TOKENS, "cpu")
            i = [0]

            def step(state=state, keys=keys, i=i):
                state.update(*keys[i[0] % len(keys)])
                i[0] += 1

            res[f"{name}_{label}"] = _bench(step)
    envs.VLLM_TPU_PREFLAT_FWD_ARGS = False
    envs.VLLM_TPU_TOKEN_PADDING_REUSE = False
    return res


if __name__ == "__main__":
    for k, v in microbench().items():
        print(f"{k:14s} {v:8.2f} us/step")
