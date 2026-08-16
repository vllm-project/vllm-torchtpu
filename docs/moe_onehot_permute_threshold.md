# The MoE one-hot permute threshold, and why it is not a row count

The MoE layer moves tokens twice per layer — a **permute** that gathers routed rows into
expert-major order before the grouped matmuls, and a **combine** that scatters and reduces them
back. Both have two implementations, and `ONEHOT_MOE_PERMUTE_THRESHOLD` picks between them:

| path | permute | combine | runs on |
|---|---|---|---|
| SparseCore | `sc_ragged_gather_v2` | `sc_ragged_gather_reduce_v2` | SparseCore |
| one-hot | `one_hot(sorted_idx, T) @ hidden` | `(one_hot(revert, R)·w·mask).sum(1) @ gmm2` | MXU |

## The mechanism: a flat cost against a quadratic one

**The SparseCore gather pads to a whole block and charges for it whatever the occupancy.**
`ragged_gather_v2` splits work into base blocks of `num_lanes × num_cores × num_subcores` rows and
processes `num_row_subchunks ∈ [1, 4]` of them at a time. On tpu7x that is `16 × 2 × 16 = 512` and
so up to **2048 rows per block**, and the kernel's grid is
`num_blocks = ceil(valid_rows / block_size)` — a block of 2048 rows costs the same with 10 valid
rows in it as with 2048. So over the whole interesting range the SparseCore cost is *flat*.

**The one-hot path builds a `[routed_rows, num_tokens]` operator and pushes it through the MXU.**
Its cost is therefore the **product** `routed_rows × num_tokens × hidden`, i.e. quadratic in the
token count at fixed `top_k`, and it is the same product on the combine side (the `[T, top_k, R]`
one-hot reduces to a `[T, R]` operator against the same `[R, H]` matrix).

A flat cost and a quadratic one cross exactly once, and where they cross is a function of
`(tokens, top_k, hidden, SparseCore block geometry)`. **It is not a constant in routed rows** — a
row threshold is only equivalent to the real rule at a fixed `top_k` and an uncompacted row count.

## The rule

```
one-hot is cheaper  ⟺  routed_rows × num_tokens × hidden  ≤  K × sc_permute_block_rows
```

`K` = 9.6e6 is one measured machine constant: the MXU flops that cost the same as moving one
padded SparseCore row (5.13e-8 s/row against the 0.65-0.75 of bf16 peak the one-hot matmul
achieves). `sc_permute_block_rows` comes straight from `pltpu.get_tpu_info().sparse_core`, so the
rule travels to other chips and other models instead of being a second magic constant.

At Qwen3.5-397B (`hidden=4096`, `top_k=10`) on tpu7x the rule fires up to **node-wide T = 692**,
i.e. **6,920 routed rows**.

## The measurement

Two terms, measured separately, then intersected:

* **one-hot** — AOT-compiled at Qwen3.5 geometry over the token ladder on a CPU host. The program
  is a single MXU-bound fusion (LLO static per-bundle occupancy `MXU = 0.86 … 0.98`), so its time
  is its flops over the achieved MXU rate.
* **SparseCore** — the W1 device ladder's `permute + combine` attribution, `102-112 µs/layer`
  across node-wide T = 256…1024, which is exactly the plateau the block model predicts (one 2048-row
  block covers the balanced valid count over that whole range).

That the *padding* is what is being charged — and not merely the occupancy — is testable on the
same eight cells, because the two hypotheses have different functional forms:

| model | fit | R² |
|---|---|--:|
| cost ∝ `ceil(valid/block)·block` padded rows | `t = 15.5 µs + 0.0458·padded_rows` | **0.943** |
| cost ∝ valid rows (occupancy) | `t = 57.8 µs + 0.0511·valid_rows` | 0.438 |

The staircase as `num_row_subchunks` climbs 1→4 is visible in the device data.

Fitting a single free parameter (the achieved MXU fraction) reproduces **all eight** measured
`prod1h` cells of that ladder within 8%, six of eight within 3%, including the sign flip between
T=512 and T=1024. The crossover is **T\* ≈ 700 node-wide tokens ≈ 7,000 routed rows**,
and it is insensitive to how the shared sort cost is attributed (690 vs 697) because that choice
moves both terms together. A ±20% error in *either* term moves T\* by only √1.2 ≈ 10%.

## Choosing a value: the loss is asymmetric

Below the crossover the one-hot path saves at most the SparseCore block cost (~100 µs/layer).
Above it, it loses as `T²` — at node-wide 2048 the one-hot pair costs ~917 µs/layer against
~190 µs. **Set the threshold below the crossover, not at it.** The shipped recipes use
**6144 routed rows** (node-wide T ≤ 614), the bottom of the measured bracket; every reachable
bucket is far from that boundary, so the choice is not sensitive to the last 10%.

## Routing skew

The one-hot cost depends only on `T` and is routing-independent. The SparseCore cost depends on
`valid_rows`, which under a skewed expert load rises on the busiest rank — and a step is bound by
its slowest rank. But because the cost is quantised in blocks, **skew moves the crossover only when
the busiest rank crosses a block boundary**, and at the crossover the balanced busiest-rank count is
884 rows against a 2048-row block:

| skew multiplier | ×1.0 | ×1.5 | ×2.0 | ×2.37 | ×3.0 | ×8.0 |
|---|--:|--:|--:|--:|--:|--:|
| node-wide T\* | 707 | 707 | 707 | 964 | 964 | 1338 |

So the crossover is **exactly unchanged up to 2.37x skew** and moves **up** beyond it. A balanced
measurement is a lower bound on the true crossover, and a threshold set from it is safe under any
routing.

## Reproducing it

```bash
# op census, flag-on vs flag-off, no TPU:
docker run --rm -e JAX_PLATFORMS=cpu -e TPU_ACCELERATOR_TYPE=tpu7x \
  -v $PWD:/root/torchtpu-vllm:ro -v <gate>:/gate:ro -v <out>:/out \
  --entrypoint python3 <serving-image> /gate/moe_census.py \
  --tokens 512,2048 --onehot-threshold 0,6144,32768 --out /out/census.json
```

The two paths are distinguishable in any op table by name: the SparseCore arm shows
`sc_ragged_gather_v2` / `sc_ragged_gather_reduce_v2`, the one-hot arm shows the `moe_permute_onehot`
/ `moe_combine_onehot` scopes. Each arm asserts the *presence* of its own path.
