"""Compare two training checkpoints bitwise (model + optimizer + loader state)."""
import sys

import torch

a = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
b = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
same_model = all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
same_opt = all(torch.equal(a["optimizer"]["state"][k][s], b["optimizer"]["state"][k][s])
               for k in a["optimizer"]["state"] for s in ("exp_avg", "exp_avg_sq"))
same_loader = a["loader"]["cursor"] == b["loader"]["cursor"] and a["loader"]["rng"] == b["loader"]["rng"]
maxdiff = max((a["model"][k].float() - b["model"][k].float()).abs().max().item() for k in a["model"])
print(f"step {a['step']} vs {b['step']} | model identical: {same_model} | optimizer identical: {same_opt} | "
      f"loader identical: {same_loader} | max |dw| = {maxdiff:.3e}")
sys.exit(0 if (same_model and same_opt and same_loader) else 1)
