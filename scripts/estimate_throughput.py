"""FLOP-based throughput / GPU-hour ESTIMATES for candidate configs on Kaggle T4s.

These are estimates, not measurements. Measure on Kaggle with:
  torchrun --standalone --nproc_per_node=2 -m trilingua.train --config_file configs/pretrain_base.json --benchmark_steps 10
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trilingua.config import PRESETS, flops_per_token, get_config, param_counts  # noqa: E402

T4_PEAK = 65e12          # fp16 tensor-core peak per T4 (NVIDIA spec)
DDP_EFF = 0.95           # assumed 2-GPU scaling efficiency (grad all-reduce ~0.3 GB / step over PCIe)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=float, default=1.25e9)
    p.add_argument("--gpus", type=int, default=2)
    p.add_argument("--mfu", nargs="+", type=float, default=[0.20, 0.30, 0.40])
    p.add_argument("--json_out", default=None)
    a = p.parse_args()
    rows = []
    hdr = (f"| config | d_model | layers | heads | ffn | params | non-emb | GFLOP/tok | 20xN tokens | "
           + " | ".join(f"tok/s @{int(m*100)}% MFU" for m in a.mfu) + " | "
           + " | ".join(f"h for {a.tokens/1e9:.2f}B @{int(m*100)}%" for m in a.mfu) + " |")
    print(hdr)
    print("|" + "---|" * (hdr.count("|") - 1))
    for name in PRESETS:
        if name == "tiny":
            continue
        c = get_config(name)
        pc = param_counts(c)
        f = flops_per_token(c)
        tps = [a.gpus * T4_PEAK * m * DDP_EFF / f for m in a.mfu]
        hrs = [a.tokens / t / 3600 for t in tps]
        rows.append(dict(config=name, d_model=c.d_model, n_layer=c.n_layer, n_head=c.n_head, ffn=c.ffn_hidden,
                         params=pc["total"], non_embedding=pc["non_embedding"], flops_per_token=f,
                         tok_per_sec=dict(zip(map(str, a.mfu), tps)), hours=dict(zip(map(str, a.mfu), hrs))))
        print(f"| {name} | {c.d_model} | {c.n_layer} | {c.n_head} | {c.ffn_hidden} | {pc['total']/1e6:.1f}M | "
              f"{pc['non_embedding']/1e6:.1f}M | {f/1e9:.3f} | {20*pc['total']/1e9:.2f}B | "
              + " | ".join(f"{t/1e3:.0f}k" for t in tps) + " | " + " | ".join(f"{h:.1f}" for h in hrs) + " |")
    if a.json_out:
        json.dump(dict(assumptions=dict(t4_peak=T4_PEAK, gpus=a.gpus, ddp_eff=DDP_EFF, tokens=a.tokens), rows=rows),
                  open(a.json_out, "w"), indent=2)


if __name__ == "__main__":
    main()
