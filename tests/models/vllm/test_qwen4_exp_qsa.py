# SPDX-License-Identifier: Apache-2.0
"""Independent QSA grouping, causal selection and request-owned cache oracles."""
import os

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
import torch

from vllm_torchtpu.kernels.qwen4_exp.qsa import (index_rope, main_rope,
                                                 qsa_attention, select_groups,
                                                 update_caches)
from vllm_torchtpu.models.vllm.qwen4_exp.qsa import (Qwen4ExpQSA,
                                                     make_rope_cache)

TPU = pytest.mark.skipif(os.environ.get("Q38_TEST_TPU") != "1",
                         reason="Requires exclusive TPU window")


def bf(x):
    return np.asarray(x, ml_dtypes.bfloat16).astype(np.float32)


def norm(x, w):
    x = np.asarray(x, np.float32)
    return bf(x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6) *
              (1 + np.asarray(w, np.float32)))


def round64(x):
    """Direct FP64->BF16 rounding, avoiding FP32 midpoint double rounding."""
    x = np.asarray(x, np.float64)
    m = x.astype(np.float32)
    tie = (m.view(np.uint32) & 65535) == 32768
    m = np.where(
        tie & (m.astype(np.float64) < x), np.nextafter(m, np.float32(np.inf)),
        np.where(tie & (m.astype(np.float64) > x),
                 np.nextafter(m, np.float32(-np.inf)), m))
    return bf(m)


def rope_oracle(x, positions, cache, index=True):
    cs = np.asarray(cache, np.float32)[positions]
    shape = (len(x), ) + (1, ) * (x.ndim - 2) + (32, )
    c, s = cs[:, 0].reshape(shape), cs[:, 1].reshape(shape)
    a, b = x[..., :32], x[..., 32:64]
    if index:
        left = round64(a.astype(np.float64) * c - bf(b * s))
        right = round64(a.astype(np.float64) * s + bf(b * c))
    else:
        left, right = bf(a * c - b * s), bf(b * c + a * s)
    return np.concatenate([left, right, x[..., 64:]], -1)


def compress_oracle(raw, positions, weight, trig):
    rows = []
    for begin in range(0, len(raw) - 3, 4):
        pooled = bf(np.mean(raw[begin:begin + 4], axis=0, dtype=np.float32))
        rows.append(
            rope_oracle(norm(pooled[None], weight), positions[begin:begin + 1],
                        trig)[0])
    return np.asarray(rows, np.float32).reshape(-1, 128)


def selection_oracle(q, compressed, position):
    scores = np.maximum(
        np.asarray(q, np.float32) @ np.asarray(compressed, np.float32).T,
        0).sum(0) / np.sqrt(128)
    visible = (position + 1) // 4
    groups = sorted(range(visible), key=lambda g: (-scores[g], g))[:512]
    tokens = [4 * g + i for g in groups for i in range(4)]
    tokens += list(range(visible * 4, position + 1))
    return np.asarray(tokens + [-1] * (2051 - len(tokens)), np.int32)


@pytest.mark.parametrize(
    "length", [1, 3, 4, 5, 127, 128, 129, 2047, 2048, 2049, 2052, 8192])
def test_selection_causal_ties_tail_and_future_poison(length):
    rng = np.random.default_rng(length)
    c = bf(rng.normal(size=(max(512, (length + 3) // 4 + 2), 128)))
    q = bf(rng.normal(size=(2, 4, 128)))
    q[0] = 0
    c[(length + 1) // 4:] = 10000  # Future groups must never win.
    actual = np.asarray(
        select_groups(jnp.asarray(q, jnp.bfloat16),
                      jnp.asarray(c, jnp.bfloat16)[None].repeat(2, 0),
                      jnp.array([length - 1] * 2)))
    for i in range(2):
        expected = selection_oracle(q[i], c, length - 1)
        np.testing.assert_array_equal(np.sort(actual[i]), np.sort(expected))
        valid = actual[i][actual[i] >= 0]
        assert len(valid) == len(set(valid)) and np.all(valid < length)
    np.testing.assert_array_equal(actual[0],
                                  selection_oracle(q[0], c, length - 1))


def test_independent_rope_contracts_and_compression_order():
    rng = np.random.default_rng(33)
    raw = bf(rng.normal(size=(12, 128)))
    weight = bf(rng.normal(size=128) * .2)
    pos = np.arange(97, 109)
    trig = make_rope_cache(128).float().numpy()
    normalized = norm(raw, weight)
    args = (jnp.asarray(normalized, jnp.bfloat16), jnp.asarray(pos),
            jnp.asarray(trig, jnp.bfloat16))
    np.testing.assert_array_equal(
        np.asarray(index_rope(*args)).astype(np.float32),
        rope_oracle(normalized, pos, trig))
    np.testing.assert_array_equal(
        np.asarray(main_rope(*args)).astype(np.float32),
        rope_oracle(normalized, pos, trig, False))
    expected = compress_oracle(raw, pos, weight, trig)
    wrong = bf(rope_oracle(normalized, pos, trig).reshape(3, 4, 128).mean(1))
    assert np.max(np.abs(expected - wrong)) > .1


def fixture_cache(pages=4, slots=4, kv=1):
    return (jnp.zeros((pages, 128, kv, 256), jnp.bfloat16),
            jnp.zeros((pages, 128, kv, 256), jnp.bfloat16),
            jnp.zeros((pages, 32, 1, 128),
                      jnp.bfloat16), jnp.zeros((slots, 8, 128), jnp.bfloat16),
            jnp.full((slots, 8, 2), -1, jnp.int32))


def test_raw_ring_three_then_two_and_slot_reuse():
    rng = np.random.default_rng(4)
    raw = bf(rng.normal(size=(5, 128)))
    weight = bf(rng.normal(size=128) * .1)
    trig = make_rope_cache(256).float().numpy()
    state = fixture_cache()
    table = jnp.array([[2, 0]], jnp.int32)
    ctable = jnp.array([[3, 1]], jnp.int32)
    for begin, end in [(0, 3), (3, 5)]:
        count = end - begin
        state = update_caches(
            jnp.ones((count, 1, 256), jnp.bfloat16),
            jnp.ones((count, 1, 256), jnp.bfloat16),
            jnp.asarray(raw[begin:end], jnp.bfloat16),
            jnp.arange(97 + begin, 97 + end, dtype=jnp.int32), *state,
            jnp.asarray(weight, jnp.bfloat16), jnp.asarray(trig, jnp.bfloat16),
            jnp.array([0, count]), jnp.array([begin]), jnp.array([2]), table,
            ctable)
        assert np.count_nonzero(np.asarray(state[2]).astype(
            np.float32)) == (0 if end == 3 else 128)
    expected = compress_oracle(raw, np.arange(97, 102), weight, trig)
    np.testing.assert_array_equal(
        np.asarray(state[2][3, 0, 0]).astype(np.float32), expected[0])
    np.testing.assert_array_equal(
        np.asarray(state[3][2, :4]).astype(np.float32), raw[[4, 1, 2, 3]])
    np.testing.assert_array_equal(np.asarray(state[4][2, :4, 0]),
                                  [101, 98, 99, 100])
    # Reused slot starts at b=0, discarding the previous request's ring tags.
    reused = update_caches(jnp.ones((1, 1, 256), jnp.bfloat16),
                           jnp.ones((1, 1, 256), jnp.bfloat16),
                           jnp.asarray(raw[:1], jnp.bfloat16), jnp.array([11]),
                           *state, jnp.asarray(weight, jnp.bfloat16),
                           jnp.asarray(trig, jnp.bfloat16), jnp.array([0, 1]),
                           jnp.array([0]), jnp.array([2]), table, ctable)
    np.testing.assert_array_equal(np.asarray(reused[4][2, :4, 0]),
                                  [11, -1, -1, -1])


def test_sparse_pages_no_other_request_leak_or_padding_write():
    k, v, c, ring, tags = fixture_cache()
    v = v.at[3].set(10000).at[2].set(2)
    q = jnp.zeros((3, 3, 256), jnp.bfloat16)
    iq = jnp.zeros((3, 4, 128), jnp.bfloat16)
    gate = jnp.zeros_like(q)
    output, ids = qsa_attention(q, iq, gate, k, v, c, jnp.array([0, 1, 2, 2]),
                                jnp.array([4, 4, 0]), jnp.array([2, 1, -1]),
                                jnp.array([[2], [3], [0]]),
                                jnp.array([[1], [0], [2]]))
    np.testing.assert_array_equal(np.asarray(output[0]).astype(np.float32), 1)
    np.testing.assert_array_equal(
        np.asarray(output[1]).astype(np.float32), 4992)
    np.testing.assert_array_equal(np.asarray(output[2]), 0)
    np.testing.assert_array_equal(np.asarray(ids[2]), -1)


def test_empty_step_keeps_all_owners():
    state = fixture_cache()
    actual = update_caches(jnp.zeros((0, 1, 256), jnp.bfloat16),
                           jnp.zeros((0, 1, 256), jnp.bfloat16),
                           jnp.zeros((0, 128), jnp.bfloat16),
                           jnp.zeros((0, ), jnp.int32), *state,
                           jnp.zeros(128, jnp.bfloat16),
                           jnp.zeros((8192, 2, 32), jnp.bfloat16),
                           jnp.array([0]), jnp.zeros(0, jnp.int32),
                           jnp.zeros(0, jnp.int32),
                           jnp.zeros((0, 64), jnp.int32),
                           jnp.zeros((0, 64), jnp.int32))
    for a, b in zip(actual, state):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("tp,rank", [(1, 0), (2, 1), (4, 1), (4, 2), (8, 3),
                                     (8, 4), (8, 7)])
def test_checkpoint_head_packing_and_kv_replication(tp, rank, monkeypatch):
    # Only replace distributed environment discovery; real upstream loaders
    # still perform every tested slice and copy. No CPU process group needed.
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
        lambda: rank)
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
        lambda: tp)
    with torch.device("cpu"):
        module = Qwen4ExpQSA(tp_size=tp, tp_rank=rank)
        q = torch.arange(12288, dtype=torch.float32)[:, None].expand(
            -1, 2560).to(torch.bfloat16)
        k = torch.arange(512, dtype=torch.float32)[:, None].expand(
            -1, 2560).to(torch.bfloat16)
        index = torch.arange(640, dtype=torch.float32)[:, None].expand(
            -1, 2560).to(torch.bfloat16)
        o = torch.arange(6144, dtype=torch.float32)[None].expand(2560, -1).to(
            torch.bfloat16)
        module.load_weights([("q_proj.weight", q), ("k_proj.weight", k),
                             ("v_proj.weight", k),
                             ("indexer.index_qk_proj.weight", index),
                             ("o_proj.weight", o)])
        width = 12288 // tp
        torch.testing.assert_close(module.q_proj.weight,
                                   q[rank * width:(rank + 1) * width],
                                   rtol=0,
                                   atol=0)
        kv_rank = rank // max(tp // 2, 1)
        kw = max(512 // tp, 256)
        torch.testing.assert_close(module.k_proj.weight,
                                   k[kv_rank * kw:(kv_rank + 1) * kw],
                                   rtol=0,
                                   atol=0)
        torch.testing.assert_close(module.indexer.index_qk_proj.weight,
                                   index,
                                   rtol=0,
                                   atol=0)
        torch.testing.assert_close(module.o_proj.weight,
                                   o[:, rank * width // 2:(rank + 1) * width //
                                     2],
                                   rtol=0,
                                   atol=0)
        assert module.get_attn_backend().get_min_page_size(None) == 128
        assert module.get_attn_backend().is_sparse()
        assert module.get_auxiliary_cache_specs(
        )["compressed_k"].tokens_per_state == 4
        with pytest.raises(ValueError, match="Missing"):
            module.finalize_weights()


def assert_metrics(actual, expected):
    a, e = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    error = np.linalg.norm(a - e) / max(np.linalg.norm(e), 1e-30)
    cosine = np.dot(a.ravel(), e.ravel()) / max(
        np.linalg.norm(a) * np.linalg.norm(e), 1e-30)
    assert error <= .01, error
    assert cosine >= .999, cosine


def attention_oracle(q, k, v, gate, selected):
    result = []
    for qq, gg, ii in zip(q, gate, selected):
        ii = ii[ii >= 0]
        scores = np.einsum('hd,shd->hs', qq, k[ii]) / 16
        prob = np.exp(scores - scores.max(-1, keepdims=True))
        prob /= prob.sum(-1, keepdims=True)
        out = bf(np.einsum('hs,shd->hd', prob, v[ii]))
        result.append(bf(out * bf(1 / (1 + np.exp(-gg)))))
    return np.asarray(result)


def streaming_contract(device=False):
    """Independent whole-sequence oracle vs <=128-token steps and reordered slots."""
    from vllm_torchtpu.kernels.qwen4_exp.qsa import build_qsa_op, qsa_step
    rng = np.random.default_rng(386)
    trig = make_rope_cache().float().numpy()
    weights = [bf(rng.normal(size=n) * .1) for n in (256, 256, 128, 128)]
    sequences = []
    for request, length in enumerate((132, 7)):
        qg = bf(rng.normal(size=(length, 1536)))
        k = bf(rng.normal(size=(length, 256)))
        v = bf(rng.normal(size=(length, 256)) + (100 if request else 0))
        ix = bf(rng.normal(size=(length, 640)))
        positions = np.arange(length, dtype=np.int32) + 97 + request * 256
        query = rope_oracle(
            norm(qg.reshape(length, 3, 512)[..., :256], weights[0]), positions,
            trig, False)
        key = rope_oracle(norm(k.reshape(length, 1, 256), weights[1]),
                          positions, trig, False)
        indexq = rope_oracle(
            norm(ix[:, :512].reshape(length, 4, 128), weights[2]), positions,
            trig)
        compressed = compress_oracle(ix[:, 512:], positions, weights[3], trig)
        ids = np.stack([
            selection_oracle(indexq[i], compressed, i) for i in range(length)
        ])
        expected = attention_oracle(query, key.repeat(3, 1),
                                    v[:, None].repeat(3, 1),
                                    qg.reshape(length, 3, 512)[..., 256:], ids)
        sequences.append(
            (qg, k, v, ix, positions, key, compressed, ids, expected))
    main_table = np.array([[3, 1], [0, 2]], np.int32)
    compressed_table = np.array([[2, 0], [1, 3]], np.int32)
    slots = np.array([2, 1], np.int32)
    initial = [
        np.zeros((4, 128, 1, 256), ml_dtypes.bfloat16),
        np.zeros((4, 128, 1, 256), ml_dtypes.bfloat16),
        np.zeros((4, 32, 1, 128), ml_dtypes.bfloat16),
        np.zeros((4, 8, 128), ml_dtypes.bfloat16),
        np.full((4, 8), -1, np.int64)
    ]
    initial[-1][3] = (
        1 << 40) + 17  # Untouched tags must survive the int32 bridge exactly.
    if device:

        def tensor(x):
            a = np.asarray(x)
            t = torch.from_numpy(
                a.astype(np.float32) if a.dtype ==
                ml_dtypes.bfloat16 else a.copy())
            if a.dtype == ml_dtypes.bfloat16:
                t = t.bfloat16()
            return t.to('tpu')

        state = [tensor(x) for x in initial]
        owners = list(state)
        op = torch.compile(build_qsa_op(),
                           backend='tpu',
                           fullgraph=True,
                           dynamic=False)
    else:
        state = [jnp.asarray(x) for x in initial[:-1]] + [
            jnp.asarray(initial[-1].view(np.int32).reshape(4, 8, 2))
        ]
        op = jax.jit(qsa_step)
    offsets = [0, 0]
    for step, lengths in enumerate([(3, 3), (2, 2), (122, 0), (1, 1), (1, 1),
                                    (3, 0)]):
        order = [0, 1] if step % 2 == 0 else [1, 0]
        starts = np.array(
            [0, lengths[order[0]],
             sum(lengths), sum(lengths)], np.int32)
        packed = []
        for field in range(5):
            rows = [
                sequences[r][field][offsets[r]:offsets[r] + lengths[r]]
                for r in order
            ]
            arr = np.concatenate(rows)
            arr = np.concatenate(
                [arr, np.zeros((1, ) + arr.shape[1:], arr.dtype)])
            packed.append(arr)
        qg, k, v, ix, pos = packed
        norm_inputs = [x.astype(ml_dtypes.bfloat16) for x in weights]
        meta = [
            starts,
            np.array([offsets[r] for r in order] + [0], np.int32),
            np.array([slots[r] for r in order] + [-1], np.int32),
            np.concatenate([main_table[order], [[0, 0]]]).astype(np.int32),
            np.concatenate([compressed_table[order], [[0,
                                                       0]]]).astype(np.int32)
        ]
        inputs = [x.astype(ml_dtypes.bfloat16) for x in (qg, k, v, ix)
                  ] + norm_inputs + [pos, trig.astype(ml_dtypes.bfloat16)]
        if device:
            output, ids = op(*[tensor(x) for x in inputs], *state,
                             *[tensor(x) for x in meta])
            output, ids = output.cpu().float().numpy(), ids.cpu().numpy()
            assert all(a is b for a, b in zip(state, owners))
            current = [
                x.cpu().float().numpy() if i < 4 else x.cpu().numpy()
                for i, x in enumerate(state)
            ]
        else:
            *state, output, ids = op(*[jnp.asarray(x) for x in inputs], *state,
                                     *[jnp.asarray(x) for x in meta])
            output, ids = np.asarray(output, np.float32), np.asarray(ids)
            current = [np.asarray(x, np.float32) for x in state[:-1]] + [
                np.asarray(state[-1]).copy().view(np.int64).reshape(4, 8)
            ]
        expected = np.concatenate([
            sequences[r][8][offsets[r]:offsets[r] + lengths[r]] for r in order
        ]).reshape(sum(lengths), -1)
        assert_metrics(output[:-1], expected)
        np.testing.assert_array_equal(output[-1], 0)
        np.testing.assert_array_equal(ids[-1], -1)
        for batch, r in enumerate(order):
            for i in range(lengths[r]):
                np.testing.assert_array_equal(
                    np.sort(ids[starts[batch] + i]),
                    np.sort(sequences[r][7][offsets[r] + i]))
            offsets[r] += lengths[r]
            # Main, compressed and raw owners use three intentionally different mappings.
            for logical in range(offsets[r]):
                np.testing.assert_array_equal(
                    current[1][main_table[r, logical // 128], logical % 128,
                               0], sequences[r][2][logical])
            groups = offsets[r] // 4
            if groups:
                produced = np.stack([
                    current[2][compressed_table[r, g // 32], g % 32, 0]
                    for g in range(groups)
                ])
                assert_metrics(produced, sequences[r][6][:groups])
            for logical in range(max(0, offsets[r] - 4), offsets[r]):
                np.testing.assert_array_equal(
                    current[3][slots[r], logical % 4], sequences[r][3][logical,
                                                                       512:])
                assert current[4][slots[r],
                                  logical % 4] == sequences[r][4][logical]
        np.testing.assert_array_equal(current[4][3], initial[4][3])
        np.testing.assert_array_equal(current[3][0], 0)
    return state


def test_whole_vs_chunked_mixed_requests_and_page_boundaries():
    streaming_contract()


@TPU
def test_tpu_compiled_sparse_cache_lifecycle():
    streaming_contract(device=True)


@TPU
def test_tpu_context8192_uses_steps128_and_exact_selection_boundaries():
    from vllm_torchtpu.kernels.qwen4_exp.qsa import build_qsa_op

    def tensor(x, dtype):
        return torch.as_tensor(x, dtype=dtype, device='cpu').to('tpu')

    state = [
        torch.zeros((64, 128, 1, 256), dtype=torch.bfloat16, device='tpu')
        for _ in range(2)
    ]
    state += [
        torch.zeros((64, 32, 1, 128), dtype=torch.bfloat16, device='tpu'),
        torch.zeros((2, 8, 128), dtype=torch.bfloat16, device='tpu'),
        torch.full((2, 8), -1, dtype=torch.int64, device='tpu')
    ]
    main_table = np.random.default_rng(32).permutation(64)[None]
    ctable = np.random.default_rng(33).permutation(64)[None]
    mt, ct = tensor(main_table, torch.int32), tensor(ctable, torch.int32)
    q = torch.zeros((128, 1536), dtype=torch.bfloat16, device='tpu')
    k = torch.zeros((128, 256), dtype=torch.bfloat16, device='tpu')
    index = torch.zeros((128, 640), dtype=torch.bfloat16, device='tpu')
    weights = [
        torch.zeros(n, dtype=torch.bfloat16, device='tpu')
        for n in (256, 256, 128, 128)
    ]
    trig = make_rope_cache().to('tpu')
    op = torch.compile(build_qsa_op(),
                       backend='tpu',
                       fullgraph=True,
                       dynamic=False)
    b = 0
    for target in (1, 3, 4, 5, 127, 128, 129, 2047, 2048, 2049, 2052, 8192):
        while b < target:
            length = min(128, target - b)
            positions = np.minimum(b + np.arange(128), 8191)
            # Distinct token values reveal stale/wrong/noncontiguous page reads.
            v = tensor(
                np.broadcast_to((positions % 97)[:, None], (128, 256)).copy(),
                torch.bfloat16)
            output, ids = op(q, k, v, index, *weights,
                             tensor(positions, torch.int64), trig, *state,
                             tensor([0, length], torch.int32),
                             tensor([b], torch.int64),
                             tensor([1], torch.int32), mt, ct)
            actual = ids.cpu().numpy()
            for i in (0, length - 1):
                logical = b + i
                count = min((logical + 1) // 4, 512)
                expected = list(range(count * 4)) + list(
                    range((logical + 1) // 4 * 4, logical + 1))
                expected += [-1] * (2051 - len(expected))
                np.testing.assert_array_equal(actual[i], expected)
                selected = np.asarray(expected)
                selected = selected[selected >= 0]
                value = bf(np.mean((selected % 97).astype(np.float32))) * .5
                row = output[i].cpu().float().numpy()
                if value == 0:
                    np.testing.assert_array_equal(row, 0)
                else:
                    assert_metrics(row, np.full_like(row, value))
            if length < 128:
                np.testing.assert_array_equal(actual[length:], -1)
                np.testing.assert_array_equal(
                    output[length:].cpu().float().numpy(), 0)
            b += length
        print(f'QSA context boundary {target}: exact selected IDs, step <=128',
              flush=True)
    np.testing.assert_array_equal(state[4][1, :4].cpu().numpy(),
                                  [8188, 8189, 8190, 8191])
    np.testing.assert_array_equal(state[4][0].cpu().numpy(), -1)


def test_empty_complete_step_has_no_selected_tokens_or_owner_mutation():
    from vllm_torchtpu.kernels.qwen4_exp.qsa import qsa_step
    state = fixture_cache()
    args = [jnp.zeros((0, n), jnp.bfloat16) for n in (1536, 256, 256, 640)]
    args += [jnp.zeros(n, jnp.bfloat16) for n in (256, 256, 128, 128)]
    args += [jnp.zeros(0, jnp.int32), jnp.zeros((8192, 2, 32), jnp.bfloat16)]
    result = qsa_step(*args, *state, jnp.array([0]), jnp.zeros(0, jnp.int32),
                      jnp.zeros(0, jnp.int32), jnp.zeros((0, 64), jnp.int32),
                      jnp.zeros((0, 64), jnp.int32))
    for actual, expected in zip(result[:5], state):
        np.testing.assert_array_equal(actual, expected)
    assert result[5].shape == (0, 768) and result[6].shape == (0, 2051)
