# CPU statistical check of the sampled-draft rejection test (no TPU):
#   TORCH_DEVICE_BACKEND_AUTOLOAD=0 VLLM_PLUGINS="" .venv-epic/bin/python -I scripts/dsv41_pd/test_sampled_draft_rejection.py
# Expect TV(output, p) ~ 0.01 for both greedy and sampled drafts and sampled accept == sum(min(p, q)).
import ast, torch
src = open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "../../src/vllm_torchtpu/layers/adapter/sample/rejection_sampler.py")).read()
tree = ast.parse(src)
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_random_rejection_sample_with_segment")
fn.decorator_list = []
ns = {"torch": torch, "TEMPERATURE_EPS": 1e-9, "PLACEHOLDER_TOKEN_ID": -1,
      "apply_top_k_top_p": lambda logits, k, p: logits}
exec(compile(ast.Module([fn], []), "rs", "exec"), ns)
rs = ns["_random_rejection_sample_with_segment"]

torch.manual_seed(0)
V, K, N = 12, 3, 20000          # vocab, drafts per request, requests (batched)
T = 1.0
tl = torch.randn(K, V) * 1.5     # target logits per position (same for all requests)
dl = tl + torch.randn(K, V) * 0.8  # draft logits: correlated but different
p = torch.softmax(tl / T, -1); q = torch.softmax(dl / T, -1)

def run(sampled):
    # drafts: sampled from q (sampled mode) or argmax q (greedy draft)
    if sampled:
        d = torch.multinomial(q.repeat(N, 1), 1).view(N, K)
    else:
        d = dl.argmax(-1).repeat(N, 1)
    total = N * K
    out = rs(
        d.reshape(-1).to(torch.int32),
        tl.repeat(N, 1),
        torch.full((N,), K, dtype=torch.int32),
        torch.zeros(N, dtype=torch.int32),          # bonus token (unused for 1st pos)
        torch.arange(N).repeat_interleave(K),
        torch.arange(K, dtype=torch.int32).repeat(N),
        K,
        torch.full((total, 1), T), torch.zeros((total, 1), dtype=torch.int32), torch.ones((total, 1)),
        torch.rand(total), torch.rand(total, V),
        None, False,
        dl.repeat(N, 1) if sampled else None,
        torch.ones(total, dtype=torch.bool) if sampled else None,
    )
    first = out[:, 0]
    emp = torch.bincount(first.long(), minlength=V).float() / N
    acc1 = (out[:, 0] == d[:, 0]).float().mean().item()
    return emp, acc1, out, d

for sampled in (False, True):
    emp, acc1, out, d = run(sampled)
    tv = 0.5 * (emp - p[0]).abs().sum().item()
    print(f"{'sampled' if sampled else 'greedy '} draft: TV(output pos0, p) = {tv:.4f}  accept@pos0 = {acc1:.3f}")
print(f"theory: greedy accept = p(argmax q) = {p[0, dl[0].argmax()].item():.3f}, sampled accept = sum min(p,q) = {torch.minimum(p[0], q[0]).sum().item():.3f}")
# second position conditional check: given pos0 accepted == d0, output pos1 distribution should be p[1]
emp, acc1, out, d = run(True)
m = out[:, 0] == d[:, 0]
emp1 = torch.bincount(out[m, 1].long().clamp(min=0), minlength=V).float() / m.sum()
print(f"sampled pos1 | pos0 accepted: TV = {0.5 * (emp1 - p[1]).abs().sum().item():.4f}")
# draft_valid=False rows must reproduce greedy-draft behaviour exactly in distribution
