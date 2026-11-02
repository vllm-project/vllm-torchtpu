# Split-SparseCore MoE preprocessing

The fused EP FP4 MoE adapter supports an opt-in `topology` preprocessing
backend for TPU7x with eight or sixteen EP ranks. Eight ranks need an affine
rank mapping over a 2x2x1 chip topology with both dies and use the three-phase
activation all-gather kernel below. Sixteen ranks use XLA's own all-gather for
the activation, see "Sixteen EP ranks"; it was validated on two hosts with a
2x2x2 topology, and the adapter checks only the device count and kind.
Unsupported topology, shape, or scratch-memory requirements retain native
preprocessing. Other weight formats retain the native path.

## Enablement

Set these before initializing the TPU runtime, preserving any other libtpu flags:

```bash
export MOE_FUSED_EP_PREPROCESS_BACKEND=topology
export LIBTPU_INIT_ARGS="$LIBTPU_INIT_ARGS --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=0 --xla_mosaic_unsafe_allow_multicore_remote_dma=true"
```

The default backend is `native`. The DSV4.1 CI serving recipe selects `topology`
for the decode scenario and `native` for prefill/cached-extend. Large prefill
buckets within a decode server can still fall back through the resource guards.
The SparseCore queue-assignment compiler pass remains enabled.

## Execution and compilation

A complete three-phase activation all-gather keeps its SPMEM scratch within one
Pallas kernel. Metadata uses the native all-gather. A second Pallas kernel
scatters token indices and activation-scale bits together, with disjoint output
ownership across vector subcores and integer add/drop semantics.

The adapter resolves physical topology before tracing and registers one
PRE_SCHEDULER HLO transformation per process. For each eligible FP4 MoE call,
the transformation follows the kernel's activation and token-index operands to
identify its own collectives. It pins activation communication to SC1 and
metadata communication plus scatter to SC0, using immutable core assignments.
It adds control edges from activation submission to independent routing work,
and places the two independent FP4 weight-scale layout copies between metadata
submission and completion. It does not make routing depend on activation
communication completion. Before advancing activation submission, it orders
preceding multi-SC offload completions (in the original computation order)
before that submission unless already implied by dependencies. This prevents
independent work that needs both SCs from straddling the split-SC region.
Cycle checks still apply. This boundary can reduce overlap with unrelated SC
work; it does not serialize activation communication with MoE routing/scatter.

TorchTPU can drop outer `compute_on` scopes and placement options while retaining
Pallas kernel names. The transformation recognizes both representations.
Unrelated compilations and ambiguous patterns are left unchanged; final HLO and
trace inspection are required when upgrading the compiler. The transformation
preserves protobuf fields it does not inspect and retains other backend options.
There is no JAX lowering monkeypatch or scheduling-group dependency.

## Sixteen EP ranks

With sixteen devices there is no topology kernel: `prepare_moe_preprocessing`
returns a topology with `custom_gather=False` and the activation gather is
`lax.all_gather` under `compute_on(tpu_sparsecore, core_ids=[1])`, submitted
before the router like the kernel above. The schedule transformation matches it
by its `all_gather` operation name, pins it to SC1 and the metadata gather to
SC0, and orders them as for eight ranks. A plain `all_gather` also matches the
activation marker in a call that went back to native preprocessing, so the
transformation rewrites a graph only when it has the `moe_early_ag_route` scope
that the split branch alone emits; native fallbacks keep their schedule.

The SC scatter is not used at sixteen ranks: its scratch grows with the global
routed rows and has to fit in half a subcore's VMEM, 256 KiB on TPU7x, and at
4608 global tokens x top-k 6 it needs about 346 KB by the scratch formula. The
shapes where it would fit were not measured with it, so the scatter keeps XLA's
implementation and only the gather placement and submission order change.
`split_sc_plan` also bounds the path to `XLA_GATHER_MAX_LOCAL_TOKENS` (512)
local tokens: it was measured at 288 (48 requests x (5 speculative + 1)), and
larger prefill buckets in a decode server keep native preprocessing, as the
resource guards do for eight ranks. The libtpu flags above are required here
too.

Measured on two v7x 2x2x2 hosts with a whole-layer benchmark (not in this
repository) at 288 local tokens with DSV4.1 shapes, synthetic FP4 weights and
uniform routing; each value is the maximum over TensorCores, then the median
over three calls:

| Metric | Native | Split SC |
| --- | ---: | ---: |
| Module body | 408.3 us | 364.8 us |
| Before the fused kernel | 190.1 us | 162.1 us |
| Fused EP kernel | 196.3 us | 180.6 us |
| After the fused kernel | 22.7 us | 22.7 us |

Outputs are bitwise equal on all sixteen ranks. The gain is smaller than the
activation gather's earlier submission suggests (it is ready at 5 us instead of
61 us): the metadata gather runs on SC0 while the activation gather is in flight
and stretches from about 26 us to about 100 us, so both finish near 126 us,
matching the standalone all-gather microbenchmark where the two gathers run
back to back. The scatters, which wait for the metadata, then run after it. The
fused kernel is also 15.7 us shorter on every TensorCore; that is not explained.
This is a single-layer benchmark, not serving. Only SC0 was traced, so the
activation gather's SC1 execution is inferred from SC0 showing one collective
instead of two.

## Validation and limits

Validation used JAX/jaxlib 0.11.2, libtpu 0.0.50 and TorchTPU 0.2.0rc1, with
FP4 weights, FP8 activations, EP8, 288 local tokens, hidden size 5120, 384 experts,
top-k 6, logical intermediate size 2304 padded to 2560, and one shared expert.
Six interleaved ABBAAB blocks each measured 30 calls. Profile counters verified
P7. Each call used the maximum over eight TensorCores, then the median over
90 calls per backend. No assumed device frequency was used.

| Metric | Native | Topology |
| --- | ---: | ---: |
| Full device execution, excluding entry wait | 434.40 us | 407.06 us |
| Preprocessing | 118.78 us | 91.60 us |
| Exposed activation AG completion wait | 23.28 us | 0.0024 us |

All 24 cross-rank/candidate output comparisons were bitwise equal. All three
candidate compilations retained SC1 activation placement, SC0 metadata/scatter,
early activation submission, and both weight-scale copies within the metadata
asynchronous window. Metadata completion still exposed about 8.51 us of wait;
this is not a claim that all communication waits disappear or that the entire
preprocessing path reaches an ICI roofline.

These measurements cover the production helper and kernel through a TorchTPU
whole-layer benchmark, including routing and the shared MLP. They do not replace
full vLLM serving validation through the production adapter. Decode CI provides
that integration check. Revalidate final HLO, correctness and performance after
changing runtime versions, shapes, physical topology, or compilation settings.

### Full serving graph scheduling regression

Decode CI build 5652 captured the complete failing graph: 6,967 entry
instructions and 20 MoEs, using the compiler versions above. The original
callback added 380 early-submission control edges. There were no scale-copy
edges in this graph because the scale operands were parameters.

Local PRE_SCHEDULER replay reproduced the CI scheduler failure exactly
(5,391 of 6,967 instructions scheduled). Keeping only SC placement compiled
successfully. The HLO dependency graph and initial scheduler graph were both
acyclic; acyclicity alone did not establish schedulability with SC resources.
An independent earlier `candidate_union_metadata_sc` offload used both SCs.
Ordering its completion before activation submission allowed scheduling to
finish without removing any early-submission edges. The production rule uses
resource assignments, computation order and reachability, not this operator's
name or a layer index. The resulting graph added one completion boundary and
passed full backend AOT. Final HLO retained SC1 activation and SC0
metadata/scatter for all 20 MoEs, with activation argument submission before
metadata argument submission and scatter submission before activation completion.
The captured 3,038- and 5,499-instruction graphs also passed scheduler replay.

Full-graph AOT validates compilation and final ordering, not serving latency or
numerical correctness. The additional SC completion boundary requires fresh
decode CI and runtime profiling before claiming the whole-layer speedup above
for serving.
