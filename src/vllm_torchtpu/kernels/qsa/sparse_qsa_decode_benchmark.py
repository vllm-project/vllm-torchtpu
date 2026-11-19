"""Microbenchmarks for the SC gather -> HBM -> TC attention QSA decode.

Same shape as `sparse_gqa_decode/benchmark.py` (the fused single-kernel
decode) so the two can be compared directly.
"""

import functools
import glob
import os
import shutil
import statistics
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import test_util as jtu
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.qsa import qsa_ref
from vllm_torchtpu.kernels.qsa import sparse_qsa

# Name prefix of the jitted decode module in the device trace.
_MODULE_NAME = "jit_sparse_qsa"

_WARMUP_REPS = 2
_TIMED_REPS = 3

# Matches sparse_gqa_decode/benchmark.py: heads=16 / kv_heads=1, dim=256,
# ratio=4, top-k budget 512 groups, page 128, fp8.
#
# Spec-decode workload: `requests` requests, each with `gamma + 1` query tokens
# (flattened onto the kernel's `batch` axis, so batch = requests * (gamma + 1)).
# The query tokens of one request share its page table, have consecutive
# lengths (context - gamma, ..., context), and each token's top-k selection
# overlaps the previous query token's selection (in the same request) by a
# fraction `overlap`.
_SHAPE = dict(
    heads=16,
    kv_heads=1,
    ratio=4,
    groups=512,
    page=128,
    requests=16,
    gamma=3,
    overlap=0.9,
    context=131072,
)
# Two KV heads x 16 query heads each.
_SHAPE_KV2 = dict(_SHAPE, heads=32, kv_heads=2)
# Four KV heads x 16 query heads each.
_SHAPE_KV4 = dict(_SHAPE, heads=64, kv_heads=4)
# Same as `_SHAPE` with 24 / 12 requests.
_SHAPE_REQ24 = dict(_SHAPE, requests=24)
_SHAPE_REQ12 = dict(_SHAPE, requests=12)

BENCHMARK_CONFIGS = [
    dict(_SHAPE, chunk=64, kbs=8),
    dict(_SHAPE_REQ24, chunk=64, kbs=8),
    dict(_SHAPE_REQ12, chunk=64, kbs=8),
    dict(_SHAPE_KV2, chunk=64, kbs=4),
    dict(_SHAPE_KV4, chunk=64, kbs=2),
]


def _generate_test_cases():
  cases = []
  for idx, cfg in enumerate(BENCHMARK_CONFIGS):
    name = (
        f"{idx}_req{cfg['requests']}_gamma{cfg['gamma']}"
        f"_ov{round(cfg['overlap'] * 100)}"
        f"_ctx{cfg['context']}_h{cfg['heads']}"
        f"_kv{cfg['kv_heads']}"
        f"_r{cfg['ratio']}_g{cfg['groups']}_p{cfg['page']}"
        f"_chunk{cfg['chunk']}_kbs{cfg['kbs']}"
    )
    cases.append((name, cfg))
  return cases


def _device_time_us(fn, args, reps: int = _TIMED_REPS) -> float:
  """Returns the median on-device duration (us) of `fn(*args)` on TPU:0.

  Times the `_MODULE_NAME` events on the "XLA Modules" line of a JAX profiler
  trace, so host dispatch overhead is excluded.

  Args:
    fn: jitted function to time.
    args: positional arguments of `fn`.
    reps: number of profiled runs.
  """
  for _ in range(_WARMUP_REPS):
    jax.block_until_ready(fn(*args))

  trace_dir = tempfile.mkdtemp()
  try:
    with jax.profiler.trace(trace_dir):
      out = None
      for _ in range(reps):
        out = fn(*args)
      jax.block_until_ready(out)

    pbs = glob.glob(f"{trace_dir}/**/*.xplane.pb", recursive=True)
    plane = jax.profiler.ProfileData.from_file(
        max(pbs, key=os.path.getmtime)
    ).find_plane_with_name("/device:TPU:0")
    durs = [
        stat[1]
        for line in plane.lines
        if line.name == "XLA Modules"
        for e in line.events
        if e.name.startswith(_MODULE_NAME)
        for stat in e.stats
        if stat[0] == "device_duration_ps"
    ]
    if not durs:
      raise RuntimeError(f"No {_MODULE_NAME} device time found in profile.")
    return statistics.median(durs) / 1e6
  finally:
    shutil.rmtree(trace_dir, ignore_errors=True)


def _print_table(header, rows, *, col_width_extra=None):
  """Prints `rows` as a box-drawn table under `header`."""
  col_width_extra = col_width_extra or {}
  widths = [len(h) + 3 + col_width_extra.get(h, 0) for h in header]
  fmt = "│" + "│".join(f"{{:<{w}}}" for w in widths) + "│"

  def sep(left, mid, right):
    return left + mid.join("═" * w for w in widths) + right

  print(sep("╒", "╤", "╕"))
  print(fmt.format(*header))
  for row in rows:
    assert len(row) == len(header), (row, header)
    print(sep("╞", "╪", "╡"))
    print(fmt.format(*row))
  print(sep("╘", "╧", "╛"))


def make_spec_decode_inputs(
    requests: int,
    gamma: int,
    overlap: float,
    context: int,
    page: int,
    *,
    heads: int,
    kv_heads: int,
    ratio: int,
    groups: int,
    seed: int = 6317,
):
  """Builds (q, gate, blocks, lengths, table, cache) spec-decode inputs.

  Each request contributes `gamma + 1` consecutive query tokens (rows of the
  flat `batch` axis). Tokens of the same request share the page table and have
  consecutive lengths `context - gamma, ..., context`. Token 0 picks
  `min(len // ratio, groups)` random groups; every later token `t` keeps a
  random `round(overlap * n)` of token `t - 1`'s groups and fills the rest with
  groups that token `t - 1` did not select.

  Args:
    requests: number of decode requests.
    gamma: number of speculative tokens (query tokens per request - 1).
    overlap: fraction in [0, 1] of each later token's top-k shared with the
      top-k of the previous query token in the same request.
    context: length (tokens) of the last query token of each request.
    page: physical KV page size (tokens).
    heads: number of query heads.
    kv_heads: number of KV heads.
    ratio: compression ratio (tokens per selected group).
    groups: top-k budget (number of selectable groups per query token).
    seed: RNG seed.

  Returns:
    Tuple of jax arrays `(q, gate, blocks, lengths, table, cache)`.
  """
  assert 0.0 <= overlap <= 1.0, overlap
  assert context > gamma, (context, gamma)
  tokens_per_req = gamma + 1
  batch = requests * tokens_per_req
  # The tokens of a request share its page table, so the cache only needs
  # `requests * columns` pages: build it for `requests` rows. q / gate, table,
  # lengths and blocks are built below for all `batch` rows.
  *_, cache = qsa_ref.make_inputs(
      requests,
      context,
      page,
      heads=heads,
      kv_heads=kv_heads,
      ratio=ratio,
      groups=groups,
      seed=seed,
      ragged=False,
      poison=False,
      device_cache=True,
  )
  rng = np.random.default_rng(seed + requests + gamma)
  columns = max(1, -(-context // page))
  q = jnp.asarray(
      rng.normal(0, 0.5, (batch, heads, sparse_qsa.HEAD_DIM)), jnp.bfloat16
  )
  gate = jnp.asarray(rng.normal(0, 3, q.shape), jnp.bfloat16)
  # One page table per request (distinct physical pages), shared by its tokens.
  req_table = (
      rng.permutation(requests * columns)
      .reshape(requests, columns)
      .astype(np.int32)
  )
  table = np.repeat(req_table, tokens_per_req, axis=0)
  lengths = np.tile(
      np.arange(context - gamma, context + 1, dtype=np.int32), requests
  )

  blocks = np.full((batch, groups), -1, np.int32)
  for r in range(requests):
    row0 = r * tokens_per_req
    n0 = min(int(lengths[row0]) // ratio, groups)
    prev = rng.choice(int(lengths[row0]) // ratio, n0, replace=False)
    blocks[row0, :n0] = prev
    for t in range(1, tokens_per_req):
      length = int(lengths[row0 + t])
      n = min(length // ratio, groups)
      shared = min(round(overlap * n), prev.size)
      kept = rng.choice(prev, shared, replace=False)
      # Fresh groups must avoid all of the previous token's picks to keep the
      # overlap with it exact.
      pool = np.setdiff1d(np.arange(length // ratio), prev, assume_unique=True)
      fresh = rng.choice(pool, n - shared, replace=False)
      sel = np.concatenate([kept, fresh])
      rng.shuffle(sel)
      blocks[row0 + t, :n] = sel
      prev = sel
  return (
      q,
      gate,
      jnp.asarray(blocks),
      jnp.asarray(lengths),
      jnp.asarray(table),
      cache,
  )


def run_one(cfg) -> tuple[str, ...]:
  """Runs a single microbenchmark configuration and returns a table row."""
  device_name = jax.devices()[0].device_kind
  inputs = make_spec_decode_inputs(
      cfg["requests"],
      cfg["gamma"],
      cfg["overlap"],
      cfg["context"],
      cfg["page"],
      heads=cfg["heads"],
      kv_heads=cfg["kv_heads"],
      ratio=cfg["ratio"],
      groups=cfg["groups"],
  )
  opts = dict(ratio=cfg["ratio"], kv_heads=cfg["kv_heads"])
  run_fn = jax.jit(
      functools.partial(
          sparse_qsa.sparse_qsa,
          **opts,
          chunk_size=cfg["chunk"],
          attention_kernel_batch_size=cfg["kbs"],
      )
  )

  out = np.asarray(run_fn(*inputs), np.float32)
  expected = np.asarray(
      qsa_ref.ref_sparse_qsa_decode(*inputs, **opts), np.float32
  )
  max_err = float(np.max(np.abs(out - expected)))

  step_us = _device_time_us(run_fn, inputs)

  # Bytes of selected K/V tokens (one row per token per KV head).
  lengths = np.asarray(inputs[3])
  tokens = np.minimum(lengths // cfg["ratio"], cfg["groups"]) * cfg["ratio"] + (
      lengths % cfg["ratio"]
  )
  kv_bytes = int(tokens.sum()) * cfg["kv_heads"] * sparse_qsa.ROW_WORDS * 4

  config_str = (
      f"req{cfg['requests']}x{cfg['gamma'] + 1}_ov{cfg['overlap']:g}"
      f"_ctx{cfg['context']}"
      f"_h{cfg['heads']}/{cfg['kv_heads']}_d256"
      f"_r{cfg['ratio']}_g{cfg['groups']}_p{cfg['page']}"
  )
  return (
      device_name,
      config_str,
      str(cfg["chunk"]),
      str(cfg["kbs"]),
      f"{step_us:.1f}",
      f"{kv_bytes / (step_us * 1e3):.1f}",
      f"{max_err:.2e}",
  )


class SparseQsaDecodeBenchmark(jtu.JaxTestCase):
  """Microbenchmarks for the SC gather -> TC attention QSA decode."""

  _results = []

  @classmethod
  def tearDownClass(cls):
    super().tearDownClass()
    if not cls._results:
      return
    header = [
        "device",
        "config",
        "chunk",
        "kbs",
        "decode step (us)",
        "KV GB/s",
        "max err",
    ]
    _print_table(
        header,
        cls._results,
        col_width_extra={"config": 30},
    )

  @parameterized.named_parameters(*_generate_test_cases())
  def test_benchmark(self, cfg):
    self._results.append(run_one(cfg))


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
