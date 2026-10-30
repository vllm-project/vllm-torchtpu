# Qwen4Exp pool capacity

The native dual-store pool reserves one null block and enough manager blocks for
`max_num_seqs` requests at `max_model_len`, including attention, index and state
regions. The available HBM check validates this capacity after model allocation;
it does not grow the pool from profiled free memory or allow oversubscription.
Longer contexts can therefore require a smaller `max_num_seqs`.

When choosing serving limits, set `--max-model-len` and `--max-num-seqs` together.
Allocation also retains an 8 GiB temporary allowance. This is a manual planning
reserve, not a measured bound on compiler or device peak memory.

Physical page extents come from the complete main/index/state layout. They are
not a model-wide fixed page size, and revalidating the same configuration must
preserve the derived geometry. BF16 and FP8 carriers use their dtype packing.

Host PLE capacity checks cap Linux `MemAvailable` by remaining cgroup-v2 memory
when a finite limit is exposed. The estimate includes a loader allowance;
it is not a guarantee against every concurrent host allocation.
