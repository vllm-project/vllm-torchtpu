# Deployment-local expert load balancing

The same controller implementation supports separate prefill and decode
instances. Set the policy before loading the model, consistently on every
rank in one EP group. Each instance owns its expert tables, weight slots,
staging buffers, load history and convergence state; no layout is exchanged
through KV transfer.

| Mode | Samples and optimizes | Default |
| --- | --- | --- |
| `decode_only` | Clean decode steps | Yes |
| `prefill_only` | Clean prefill steps | No |
| `off` | Neither phase | No |

The existing `TPU_MOE_EPLB_DYNAMIC=1` remains the master enable switch.
With the switch off, no dynamic balancing runs. `TPU_MOE_EPLB_MODE=off`
also disables automatic dynamic placement and load taps, even when the
master switch is set.

## PD deployment

On every rank of the prefill instance:

```bash
export TPU_MOE_EPLB_DYNAMIC=1
export TPU_MOE_EPLB_MODE=prefill_only
```

On every rank of the decode instance:

```bash
export TPU_MOE_EPLB_DYNAMIC=1
export TPU_MOE_EPLB_MODE=decode_only
```

Decode-only ignores prefill warmup traffic. Prefill-only ignores decode
traffic, including first-token generation. Dirty/padded steps reset the
sampling window. Prefill samples fresh eight-clean-step windows after each
installed or rejected plan and keeps sampling when no layer clears the gain
gate. Decode retains its configured window and existing first-window
behavior: it reuses a captured logical load across budgeted migrations until
no layer clears the gain gate, then freezes. Outstanding migrations are
safely drained; phase filtering does not cancel a transfer in flight.

The policy is fixed for the process lifetime. It does not provide two
independent layouts in one mixed-role instance, and frozen decode layouts do
not automatically relearn a changed workload. Do not share persisted
physical slot tables between different EP topologies.

The DP8 prefill CI selects `prefill_only`; the DP16 decode launcher explicitly
selects and checks `decode_only`, overriding any inherited prefill policy.
The input3/cache32 kernel admission remains separate from this policy.
