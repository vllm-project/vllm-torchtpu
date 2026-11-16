#!/usr/bin/env python3
"""Replay the fork's decode-local gate probes against a decode server and compare token-for-token.
Reference = prefill__pd_gates.json from the fork's validated run (decode-local greedy outputs)."""
import json, sys
sys.path.insert(0, "/tmp")
import pd_gates as G
from tokenizers import Tokenizer
ref = json.load(open(sys.argv[1])); dec = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:9400"
tok = Tokenizer.from_file("/dev/shm/DeepSeek-V4.1-Flash/tokenizer.json")
out = {"cases": []}
all_eq = True
for c in ref["gate_token_equality"]["cases"]:
    n = c["prompt_tokens"]
    ids = G.build_prompt_ids(tok, n, f"eq{n}")
    assert len(ids) == n, (len(ids), n)
    m0 = G.scrape_spec_metrics(dec); got = G.complete(dec, ids, len(c["local_token_ids"])); m1 = G.scrape_spec_metrics(dec)
    same = got["token_ids"] == c["local_token_ids"]; all_eq &= same
    div = G.first_divergence(got["token_ids"], c["local_token_ids"])
    d = m1["num_drafts"] - m0["num_drafts"]; a = m1["num_accepted"] - m0["num_accepted"]
    out["cases"].append({"prompt_tokens": n, "equal_to_fork": same, "first_divergence": div, "drafts": d, "accepted": a})
    print(f"[PORT-vs-FORK] n={n:>5} 64 greedy tokens identical to fork: {same} first_div={div} | this request: drafts={d:.0f} accepted={a:.0f} tau={1 + a / d if d else 0:.3f}", flush=True)
usa_q = "what is usa? how many states it has? list all the states by name"
m0 = G.scrape_spec_metrics(dec); usa = G.chat(dec, usa_q, 512); m1 = G.scrape_spec_metrics(dec)
d = m1["num_drafts"] - m0["num_drafts"]; a = m1["num_accepted"] - m0["num_accepted"]
tau = round(1.0 + (a / d if d else 0.0), 4)
ref_usa = ref["gate_usa"]["local_output"]; same_usa = usa == ref_usa; all_eq &= same_usa
i = next((k for k, (x, y) in enumerate(zip(usa, ref_usa)) if x != y), min(len(usa), len(ref_usa)))
print(f"[PORT-vs-FORK] USA answer text identical to fork: {same_usa} (len port={len(usa)} fork={len(ref_usa)}, first differing char={None if same_usa else i}) states={sum(s.lower() in usa.lower() for s in G.US_STATES)}/50", flush=True)
print(f"[PORT-vs-FORK] USA request acceptance: port drafts={d:.0f} accepted={a:.0f} tau={tau} | fork tau={ref['gate_usa']['pd_mean_accept_len']} (fork figure was measured through the proxy)", flush=True)
if not same_usa: print("  port:", repr(usa[max(0, i - 60):i + 100])); print("  fork:", repr(ref_usa[max(0, i - 60):i + 100]))
out.update({"usa_equal_to_fork": same_usa, "usa_tau_port": tau, "usa_tau_fork": ref["gate_usa"]["pd_mean_accept_len"], "usa_port": usa, "all_equal": all_eq})
json.dump(out, open("/tmp/port_vs_fork_tokens.json", "w"), indent=1)
print(f"[PORT-vs-FORK] ALL IDENTICAL: {all_eq}")
