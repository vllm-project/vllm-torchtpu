# Qwen3.5 expert placement from decode-step routing

Tools to build a `TPU_EXPERT_PLACEMENT` table (read by
`src/vllm_torchtpu/expert_placement.py`) from the routing a real workload
produces, and the tables built this way for the InferenceX AgentX replay.

- `../expert_placement_tp8_ep8_agentx_corr_c0.json`: placement only (64
  experts per rank, no copies), no extra memory.
- `../expert_placement_tp8_ep8_agentx_corr_c32.json`: the same plus one copy
  of 32 experts per layer (4 copy slots per rank); needs `TPU_MOE_LOWTOK=1`.
- `profiler/`: in-graph routing recorder, installed through `sitecustomize`.
- `solve_placement.py`: solver, profiler dumps in, placement JSON out.

Both tables are for `Qwen/Qwen3.5-397B-A17B-FP8` with TP8 and expert
parallelism (8 ranks x 64 experts, 60 MoE layers), logical checkpoint expert
ids.

## Why decode steps, and why correlation

In decode, the small-batch MoE kernel streams the weights of every distinct
expert a step routes to on a rank, and the post-MoE all-reduce waits for the
slowest rank. A layer's decode cost therefore follows
`max over ranks of (distinct experts hit by this step)`, not each rank's
long-run share of routed tokens. At the batch sizes AgentX keeps decoding
(4 to 16 rows per step, MTP 3), that maximum is driven by per-step randomness
and by experts that tend to fire in the same step; a placement that balances
average load does not change it.

The profiler keeps a ring of sampled per-step hit sets. `solve_placement.py`
replays them, with the kernel's own rule for copied experts (the holder with
fewer hit experts so far, hottest copy first), and:

1. swaps primaries between ranks to minimise the mean over steps of the
   largest per-rank distinct-expert count (experts that co-fire end up on
   different ranks);
2. gives the most frequently hit experts one copy each, each on the rank that
   minimises the same replayed objective;
3. reports the objective on held-out steps for the new table and for any
   `--baseline` tables.

Held-out result on the recorded AgentX steps (mean over 60 layers of the
per-step maximum distinct experts per rank; lower is better):

| Table | Extra HBM per device | Max experts per rank |
|---|---|---|
| Checkpoint order (no table) | 0 | 13.28 |
| Average-load permutation | 0 | 13.19 |
| `corr_c0` | 0 | 12.38 |
| 128 copies chosen by hit count | 11.3 GiB | 11.01 |
| `corr_c32` | 2.8 GiB | 11.05 |

## Serving with a table

```bash
export TPU_EXPERT_PLACEMENT=$SCRIPT_DIR/expert_placement_tp8_ep8_agentx_corr_c32.json
export TPU_MOE_LOWTOK=1   # copies are served only by the small-batch MoE kernel
```

A copy slot holds one expert's FP8 weights (about 12 MiB), so `corr_c32` adds
4 x 60 x 12 MiB = 2.8 GiB per device; lower `--kv-cache-memory-bytes` by that
much if it is set explicitly. Each worker logs
`Expert placement <path>: 60 layers, 8 ranks x 68 slots (64 primaries)` when the
table is applied.

## Building a table for another workload

1. Profile. Run the workload with this directory first on `PYTHONPATH`, in
   front of the existing entries (replacing `PYTHONPATH` loads a different
   `vllm_torchtpu` and records nothing useful), no placement table (the
   recorder must see logical ids), and a fresh compile cache:

   ```bash
   export PYTHONPATH=$REPO/scripts/vllm/benchmark/inferencex/qwen3.5/eplb/profiler:$PYTHONPATH
   export VLLM_TPU_ENABLE_MOE_PROFILER=1
   export TPU_MOE_SKIP_PADDED_TOKENS=1          # separates real rows from padding
   export VLLM_TPU_MOE_PROFILE_DIR=/path/to/out
   export VLLM_TPU_MOE_PROFILE_DUMP_INTERVAL=120
   ```

   Check the server log for `Patched moe_lowtok_scores.apply`; without it
   decode steps are not recorded. Every rank writes
   `moe_profile_rank_<r>_pid_<pid>_latest.npz`; all ranks see the same router
   input, so one dump is enough. Ten minutes of traffic per concurrency is
   plenty (the ring keeps the last 4096 sampled steps per layer, one in four).

2. Solve:

   ```bash
   python3 solve_placement.py /path/to/out --copies 32 --out table.json \
       --baseline ../expert_placement_tp8_ep8_agentx_corr_c32.json \
       --recording "<job, workload, concurrencies>"
   ```

   `--copies` must be a multiple of the rank count (0 for placement only).
   About 5 s per layer per table on one CPU core.

Recorded per layer, for forwards of at most 128 rows:

| Key | Shape | Meaning |
|---|---|---|
| `dec_steps` | `[1]` | decode steps with at least one real row |
| `dec_step_hit` | `[E]` | steps in which the expert was hit |
| `dec_cohit` | `[E, E]` | steps in which both experts were hit |
| `dec_tok_coact` | `[E, E]` | real rows that picked both experts |
| `dec_padhit` | `[E]` | steps in which only padded rows hit the expert |
| `dec_rows_hist` | `[129]` | real rows per step |
| `dec_ring` | `[4096, E]` | sampled step hit sets (uint8) |
| `dec_ring_rows` | `[4096]` | real rows of each sampled step |
