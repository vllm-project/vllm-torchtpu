# Kimi-K3 megakernel (Phase 1 port)

This package holds the Kimi-K3 decode megakernel, the sequence-parallel prefill
kernels, the DSpark draft kernels, and the program that drives them. It is
ported as-is from `inferact/k3-megakernel@30c9ba966b` to set the Phase 1
baseline on main (RFC #2595). Cleanup and the move onto the shared megakernel
library come in follow-up PRs.

The package is opt-in. Nothing outside `models/kimi_k3/megakernel/` imports it,
and it changes no existing model, runner, or scheduler path.

## How Kimi-K3 runs on it

- **Decode**: one op per batch size, `k3_decode_owned_b{B}`. Its core is one
  Pallas kernel, `kimi_decoder_stack`, that runs all 93 layers in one call.
  The embedding before it and the final norm, LM head, and sampling after it
  are XLA code inside the same op.
- **Prefill**: 13 ops per window: `first`, `group` 11 times, and `final`. Each
  op covers 4 or 8 layers, and each layer is one or more fused kernels with
  the cross-chip communication inside the kernels.
- **Program**: one Kimi-K3 object replaces the model runner's forward, the
  sampler, and the speculative-decoding drafter. Each step it receives the
  scheduled requests and returns tokens. It owns the weights, KV writes, KDA
  state, sampling, and DSpark speculation.

## Directory map

| Path | What it holds | RFC #2595 row | Stack PR |
| --- | --- | --- | --- |
| `config.py`, `layout.py` | kernel config, cache layout | B4 | 1/7 |
| `common/` | 32-rank collectives, VMEM pool, memref reinterpret hook | A1 | 1/7 |
| `runtime_shim/` | backend capabilities, torch bridge, compile and export cache | B1 | 1/7 |
| `kernels/decode_megakernel.py`, `kernels/mla_dcp.py`, `kernels/paged.py`, `kernels/load.py` | decode megakernel, DCP MLA attention, paged serving step, weight layout | A1, A4 | 2/7 |
| `kernels/prefill_megakernel.py`, `kernels/prefill_moe/`, `kernels/helpers.py` | sequence-parallel prefill kernels and prefill MoE | A2 | 3/7 |
| `kernels/dspark*.py` | DSpark draft kernels | A3 | 4/7 |
| `kernels/rank_loader.py`, `tools/preshard.py` | per-rank weight streaming and the offline preshard tool | A4 | 5/7 |
| `program.py`, `decode*.py`, `prefill.py`, `pools.py`, `sampling.py`, `draft*.py` | the program: decode and prefill steps, pool binding, KDA slots, sampling, DSpark | B1 to B4 | 6/7 |
| `tools/` (other files) | `check`, `bench`, and `coverage` tools for silicon bring-up | D3 | 7/7 |

## Provenance

Every ported file starts with a `Source provenance:` line that names its path
at `30c9ba966b`. Source paths map as follows:

| Source path | Path here |
| --- | --- |
| `src/vllm_torchtpu/megakernel/kimi_k3/` | `./` |
| `src/vllm_torchtpu/megakernel/common/` | `common/` |
| `src/vllm_torchtpu/megakernel/*.py` | `runtime_shim/` |
| `tools/megakernel/` | `tools/` |

To check a file against its source, compare the two revisions directly, for
example:

```bash
git diff 30c9ba966b:src/vllm_torchtpu/megakernel/kimi_k3/kernels/paged.py \
    HEAD:src/vllm_torchtpu/models/kimi_k3/megakernel/kernels/paged.py
```

Most files differ only by the license header, the provenance line, and import
paths. Each stack PR lists the files that carry further adaptations.

## Phase 1 scope

In scope: RFC rows A1 to A4, B1 to B4, and D3.

Not in this package: the runner subclass, scheduler patches, and platform
hooks (rows C1 to C3), and the KV connector fixes (row D1). Those land in
separate PRs.
