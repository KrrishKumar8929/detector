import time
import torch

d = torch.load(
    "poketwo_full_reference_features.pt",
    map_location="cpu",
    weights_only=False
)

f = d["features"].float().contiguous()
f = f / f.norm(dim=-1, keepdim=True).clamp(min=1e-12)
q = f[0:1].contiguous()

print("Images:", f.shape[0])

t = time.perf_counter()
s = torch.mm(q, f.T)
values, indices = torch.topk(s, 100)
print("Comparison time:", round(time.perf_counter() - t, 4), "seconds")

print("Benchmark complete")