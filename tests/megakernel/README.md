# Megakernel benchmarks

## Whole-step call: `tpu_bench_wholestep_call.py`

Times the served whole-step call per bucket on random weights. Its inputs decide how far a
device-time delta carries over to a served run, so A/B a megakernel change with the
served-geometry recipe below, not with the bare defaults.

### Served-geometry recipe (default for MK A/Bs)

Qwen3.8 AgentX, c16 and c32, `VLLM_TPU_MK_SCOPES=2`, served env per concurrency:

```
python tests/megakernel/tpu_bench_wholestep_call.py \
  --buckets 4,8,16,24,32,48,64 --ctx {100000,200000} \
  --L 45 --tail --shard --loop --union --budget 512 --max-groups 65536 --kb 512 \
  --key-slots -1 --key-mp 4 --key-lanes 128 --dhash \
  --rand-pool --row-mix 0 \
  --profile <dir> --profile-iters 3
```

What each served property needs:

- **Union size.** This dominates. The union-attention cost scales with the distinct groups
  a request's 4 verify rows select.
  - Served: about 843 groups per request per QSA layer at c16, and about 426 per rank at
    c32 (DCP2). The value is flat across buckets and contexts.
  - `--rand-pool` alone gives independent rows: about 1560 at the last QSA layer.
  - `--row-mix 0` feeds the 4 rows of a request the same input, so they differ only by
    position: about 1270.
  - The served union is not reachable yet. Run both pools and extrapolate each bucket's
    delta linearly in the union. Use the `qsa_u_attend` ratio between the pools as the
    union ratio, to x = 843/1563 at c16 and 426/766 at c32.
- **Bucket fill.** The bench fills every bucket with `B / 4` requests. At c16, served
  buckets hold 0.82-0.96 of that, so scale each delta by the served live requests per slot:
  b4 1.0, b8 0.96, b16 0.82, b24 0.88, b32 0.89, b48 0.82, b64 0.87.
- **Context mix.** 100K/200K matches served b4-b32 (p50 about 120-145K). Served b48/b64 sit
  near 100K, with a third of the requests at or below 64K (the ranged-row path), so add a
  48K cell there.
- **Page table.** The random block permutation is fine. Union attention issues one
  micro-page copy per group whatever the table holds.
- **Weighting.** Weight buckets by the served step counts of the control arm, and report
  b4 on its own: served b4 carries a fixed cost the bench does not have.

Always measure the union the run actually had. Make one `--export --iters 2` call per pool
with `CDK_OUTPUT_DIR` set, then:

```
python tests/megakernel/bench_union_size.py $CDK_OUTPUT_DIR/sel_b16_ctx100000.npy 512
```

### What the recipe predicts

- The recipe reproduces served per-bucket deltas within about 20% for changes that cut
  bytes or vector work in union attention or selection, e.g. `VLLM_TPU_MK_FP8_DENSE_READ` and
  `VLLM_TPU_MK_QSA_LANE_MAJOR_SELECT`.
- For changes that cut DMA-issue (scalar) work, such as `wholephysmxu`, it is still about 4x
  too optimistic. Treat the bench result as an upper bound.
