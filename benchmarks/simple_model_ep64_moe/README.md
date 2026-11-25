# simple_model EP64 single-layer MoE reproduction

This branch benchmarks Fused EP MoE v2 W4A8 with synthetic data. The routed and shared
experts, routing projection, communication, and final combine are timed.
The workload is eight TPU hosts with eight devices each, DP16 × TP4,
48 valid tokens per DP rank, 768 experts, Top-10, hidden size 8192, and
block-256 FP4 expert weights.

Use a compatible TorchTPU environment on all eight hosts. From this checkout
on each host, pass the same comma-separated list of worker IPs in slice order
and a distinct host rank (0–7). For example:

```sh
GLOO_SOCKET_IFNAME=eth0 python benchmarks/simple_model_ep64_moe/launch.py   --host-rank "$HOST_RANK" --hosts "$WORKER_IPS" --root /tmp/simple_model-ep64-ut
```

Ten dependent residual MoE calls are compiled in one graph. The test checks
this graph against serial calls, checks token permutation, and checks exact
agreement of TP replicas. The synchronized host timer reports microseconds
per call. Device traces from ranks 0–7 are saved under
`b48/rank*/profile/`; divide an XLA module's device duration by ten to obtain
the per-layer duration. Use the largest duration across devices for each
graph and discard the first five complete graphs and the final edge graph.
