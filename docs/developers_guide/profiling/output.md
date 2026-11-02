# Output layout

## Phased runs

Traces are grouped by phase, and within a phase all ranks merge into one
timestamped run directory:

```text
/tmp/vllm_phased_profile/
├── prefill_only/
│   ├── plugins/profile/2026_08_03_10_15_42/
│   │   ├── rank0_t1v-n-abccf5c4-w-0.xplane.pb
│   │   ├── rank1_t1v-n-abccf5c4-w-0.xplane.pb
│   │   ├── rank2_t1v-n-abccf5c4-w-0.xplane.pb
│   │   └── rank3_t1v-n-abccf5c4-w-0.xplane.pb
│   ├── rank_0/
│   │   ├── batch_composition_stats_<timestamp>.json   (one per profiled step)
│   │   └── <host>_<pid>.pt.trace.json.gz
│   ├── rank_1/...
│   └── ...
├── prefill_heavy/
│   └── ...
└── decode_only/
    └── ...
```

At TP=4 that is 4 XPlane files and 4 Chrome-format traces per captured phase,
plus one `batch_composition_stats_*.json` per rank per profiled step — so a run
capturing three phases with `max_iterations=5` writes 12, 12, and 72 files
respectively.

The `rank_<N>/` directories are the per-rank capture sandboxes. Traces are moved
out of them into the shared `plugins/profile/<timestamp>/` directory with a
`rank<N>_` filename prefix; the sandbox survives only because the batch
composition stats are left behind in it. Those JSON files record, for every
profiled step, the batch id, scheduled/prefill/decode token counts, request
count, minimum KV length, and the phase that was assigned — which is what lets
you tie a trace back to the batch that produced it.

## Standard (non-phased) runs

Same merge, without the phase level:

```text
<torch_profiler_dir>/
├── plugins/profile/<timestamp>/
│   ├── rank0_<host>.xplane.pb
│   └── ... (one .xplane.pb per rank)
├── rank_0/
│   └── <host>_<pid>.pt.trace.json.gz
├── rank_1/...
└── <host>_<pid>.async_llm.pt.trace.json.gz
```

## Why the rank prefix exists

XPlane files are named after the *host*, so ranks co-located on one host would
overwrite each other if they wrote into a shared directory. Each rank therefore
captures into its own sandbox first, and the prefix is added on the way out. The
prefix is applied when the slice-global world size (TP × PP × DP) is greater
than 1 — so it is present on every file of a run or on none of them.

The rank is deliberately slice-global rather than the TP×PP-scoped rank from
`parallel_config`: with DP > 1, every replica would otherwise call itself rank 0
and their traces would collide on merge.

> [!WARNING]
> **Multi-host slices.** Rank 0 publishes the canonical run timestamp through a
> marker file that the other ranks poll for (5 s timeout). On a multi-host
> slice, the trace directory must be on a filesystem that every rank shares with
> rank 0. If a rank never sees the marker it falls back to its own timestamp —
> no trace is lost, but that rank lands in a separate run directory. A
> `did not find rank 0's canonical-ts marker` warning in the logs is the signal.
