# Compact next-shape benchmark (no production change)

## Source audit

- `compress_rope._compact` dispatches to `compact_records` only on TPU,
  `records.shape == (288,)`, and `tiles * tb == 288`. Do not widen that gate
  on CPU evidence. T288 is an existing-path control, not a new speedup.
- The actual `default_config(n)` selects `(n, tb, tiles)`:
  `(96,96,1)`, `(192,192,1)`, `(240,240,1)`, `(256,256,1)`,
  `(288,144,2)`, `(384,192,2)`. This deliberately uses **default_config**, not
  `select_config`: the TPU7 tuned T288 ratio-2/hidden5120 entry instead uses
  tb288. Do not describe these results as tuned full-compressor timing.
- `compact_records` derives n from `records.size`; its offset boundaries are
  `arange(n // tb + 1) * tb`. Both paths require the caller's n divisible by tb.
  The candidate's rank-match intermediate is quadratic in n; generic math is
  not evidence of arbitrary-shape TPU compilation or performance support.
- Both implementations admit records iff `0 <= record < total_records`.
  Zero is valid, -1 is the missing-record sentinel, all other negative and
  out-of-range values drop. Duplicates remain in input-token order: compaction
  does not deduplicate. The compressor's write-record uniqueness contract is
  separate; duplicate fixtures do not authorize duplicate ring writes.
- `fused_compress_rope` rounds token count to 16, pads both ring vectors with
  **-1**, and derives tiles from padded tokens. It pads valid flags with zero
  and has_prev with one. Precomputed tables disallow token padding and require
  matching offsets. Table tail is zero-filled; only the stable valid prefix
  up to offsets[-1] is consumed by ring_start/unstage_reads.

## Measurement and correctness

The script imports the real kernel modules while bypassing unrelated package
startup, and calls the actual `_compact` by default. The checked-in baseline
is a verbatim function snapshot (including the TPU T288 branch); AST drift
refuses the default run. After a later reviewed gate integration, explicitly
use `--baseline frozen` so newly expanded production behavior cannot silently
replace the comparison baseline. Candidate and helper source text/hash, JAX,
Python, devices and Git HEAD are included in JSON.

The independent NumPy reference loops over tokens and tile boundaries, without
using the implementation's cumsum/scatter/rank-match algorithms. Both results
are checked exactly, including int32 dtype, stable prefix/order, zero tail and
all offsets. Per shape: empty, full, duplicates, invalid/out-of-range, mixed,
tile-boundary, 15-row sentinel padding and deterministic random fixtures;
additional zero-capacity coverage. CPU runs use the unchanged candidate body
with only the TPU Pallas wrapper replaced by JAX jit. They do **not** validate
Mosaic lowering, TPU layout padding, performance, or dispatch integration.

TPU runs lower/compile both paths and check all fixtures before timing. Warmup
is outside synchronized ABBA; every measured invocation blocks on all outputs.
JSON keeps every host sample and medians (dispatch plus completion, not pure
device latency). Optional profiles are separate baseline/candidate sessions,
16 calls each, after timing; no compile or NumPy validation occurs in traces.
StableHLO and compiled HLO are saved for each path and shape. Use fresh output
directories per run so old trace artifacts cannot be mistaken for new data.

## Commands

Local CPU oracle (no accelerator jobs):

```bash
JAX_PLATFORMS=cpu uv run --no-project --python 3.12 --with numpy --with 'jax==0.8.1' \
  python scripts/dsv41_bench/compact_shapes.py --mode cpu \
  --output /tmp/dsv41-compact-shapes-cpu
pre-commit run ruff-check --files scripts/dsv41_bench/compact_shapes.py scripts/dsv41_bench/compact_baseline.py
pre-commit run ruff-format --files scripts/dsv41_bench/compact_shapes.py scripts/dsv41_bench/compact_baseline.py
```

Future TPU command in the existing pinned accelerator environment (**not run**):

```bash
python scripts/dsv41_bench/compact_shapes.py --mode tpu --rounds 50 --warmup 8 \
  --profiles --output /tmp/dsv41-compact-shapes-tpu
```

No production files are changed, no jobs submitted, no commit created. This
checkpoint is a correctness/harness preparation, not a performance conclusion.
