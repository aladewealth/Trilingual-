"""Pretraining loop. Single process or DDP (torchrun), fp16+GradScaler on T4,
bf16 on Ampere+, fp16 + GradScaler on T4/P100, fp32 on CPU. Fully resumable across Kaggle sessions.

  python -m trilingua.train --config_file configs/pretrain_base.json
  torchrun --standalone --nproc_per_node=2 -m trilingua.train --config_file configs/pretrain_base.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist

from .config import flops_per_token, get_config, param_counts
from .data import MixtureLoader, ValSet, load_meta
from .model import Transformer
from .utils import CSVLogger, atomic_save, cosine_lr, find_latest_checkpoint, strip_compile_prefix

TRAIN_FIELDS = ["step", "tokens", "loss", "lr", "grad_norm", "tok_per_sec", "loss_scale", "elapsed_s", "mfu"]
VAL_FIELDS = (["step", "tokens", "val_loss"] + [f"val_loss_{l}" for l in ("en", "es", "fr")]
              + [f"ppl_{l}" for l in ("en", "es", "fr")] + [f"bpb_{l}" for l in ("en", "es", "fr")])
PEAK_FLOPS = {"T4": 65e12, "P100": 18.7e12, "V100": 125e12, "A100": 312e12, "L4": 121e12}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config_file", default=None, help="JSON with any of the args below (CLI wins)")
    # model / data
    p.add_argument("--model", default="base", help="preset name or model-config JSON")
    p.add_argument("--vocab_size", type=int, default=None)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--tokenizer", default=None, help="tokenizer.json for bpb + samples (default data_dir/tokenizer.json)")
    p.add_argument("--langs", nargs="+", default=["en", "es", "fr"])
    p.add_argument("--mix", default="en=0.5,es=0.25,fr=0.25", help="sampling weights or 'auto' (by token count)")
    p.add_argument("--seed", type=int, default=1337)
    # batch / schedule
    p.add_argument("--micro_batch", type=int, default=16)
    p.add_argument("--global_batch_tokens", type=int, default=524288)
    p.add_argument("--total_tokens", type=float, default=1.25e9)
    p.add_argument("--max_steps", type=int, default=None, help="overrides total_tokens")
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad_clip", type=float, default=1.0)
    # system
    p.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    p.add_argument("--compile", default="auto", choices=["auto", "on", "off"])
    p.add_argument("--device", default="auto")
    # eval / log / ckpt
    p.add_argument("--out_dir", default="out")
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--eval_chunks", type=int, default=64, help="val windows per language")
    p.add_argument("--eval_batch", type=int, default=None)
    p.add_argument("--sample_tokens", type=int, default=48)
    p.add_argument("--ckpt_interval", type=int, default=100)
    p.add_argument("--resume", default="auto", help="auto | none | path to ckpt or dir")
    p.add_argument("--resume_dirs", nargs="*", default=[], help="extra dirs searched by --resume auto (e.g. /kaggle/input)")
    p.add_argument("--stop_step", type=int, default=None, help="stop (and checkpoint) at this step, schedule unchanged")
    p.add_argument("--max_hours", type=float, default=None, help="session time budget; saves and exits before it")
    p.add_argument("--wandb_project", default="trilingua")
    p.add_argument("--benchmark_steps", type=int, default=0, help="measure throughput for N steps and exit")
    return p


def parse_args(argv=None):
    p = build_parser()
    a, _ = p.parse_known_args(argv)
    if a.config_file:
        with open(a.config_file) as f:
            p.set_defaults(**json.load(f))
    return p.parse_args(argv)


def _parse_mix(s, langs):
    if s == "auto":
        return None
    d = {k: float(v) for k, v in (kv.split("=") for kv in s.split(","))}
    return {l: d[l] for l in langs}


def _carry_logs(src_ckpt: str, out_dir: str, step: int):
    """When resuming from a previous Kaggle session's output, bring its logs along (trimmed to step)."""
    src_dir = os.path.dirname(os.path.abspath(src_ckpt))
    if os.path.abspath(src_dir) == os.path.abspath(out_dir):
        return
    for name in ("train_log.csv", "val_log.csv", "samples.txt"):
        s, d = os.path.join(src_dir, name), os.path.join(out_dir, name)
        if os.path.exists(s) and not os.path.exists(d):
            shutil.copy(s, d)
    for name in ("train_log.csv", "val_log.csv"):
        d = os.path.join(out_dir, name)
        if os.path.exists(d):
            lines = open(d).read().splitlines()
            keep = [lines[0]] + [l for l in lines[1:] if l and int(l.split(",")[0]) <= step]
            open(d, "w").write("\n".join(keep) + "\n")


def main(argv=None):
    a = parse_args(argv)
    ddp = int(os.environ.get("RANK", -1)) != -1
    if a.device == "auto":
        a.device = "cuda" if torch.cuda.is_available() else "cpu"
    if ddp:
        rank, local_rank, world = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
        if a.device.startswith("cuda"):
            torch.cuda.set_device(local_rank)
            device = f"cuda:{local_rank}"
            dist.init_process_group("nccl")
        else:
            device = "cpu"
            dist.init_process_group("gloo")
    else:
        rank, local_rank, world, device = 0, 0, 1, a.device
    master = rank == 0
    dev_type = "cuda" if device.startswith("cuda") else "cpu"
    torch.manual_seed(a.seed + rank)
    if dev_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if a.dtype == "auto":
        # Native bf16 needs compute capability >= 8.0 (Ampere+). torch.cuda.is_bf16_supported()
        # returns True on T4/P100 via slow emulation, which cut throughput ~5x, so check capability.
        a.dtype = (("bf16" if torch.cuda.get_device_capability()[0] >= 8 else "fp16")
                   if dev_type == "cuda" else "fp32")
    ptdtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]
    autocast = torch.autocast(device_type=dev_type, dtype=ptdtype) if a.dtype != "fp32" else nullcontext()
    os.makedirs(a.out_dir, exist_ok=True)

    tok_path = a.tokenizer or os.path.join(a.data_dir, "tokenizer.json")
    tokenizer = None
    if os.path.exists(tok_path):
        from .tokenizer import TrilinguaTokenizer
        tokenizer = TrilinguaTokenizer.from_file(tok_path)
        if a.vocab_size is None:
            a.vocab_size = tokenizer.vocab_size
    cfg = get_config(a.model, vocab_size=a.vocab_size, max_seq_len=a.seq_len, dropout=a.dropout)

    T = a.seq_len
    per_micro = a.micro_batch * T * world
    assert a.global_batch_tokens % per_micro == 0, f"global_batch_tokens must be a multiple of {per_micro}"
    accum = a.global_batch_tokens // per_micro
    max_steps = a.max_steps or math.ceil(a.total_tokens / a.global_batch_tokens)

    model = Transformer(cfg).to(device)
    raw_model = model
    optimizer = model.configure_optimizer(a.weight_decay, a.lr, (a.beta1, a.beta2), dev_type)
    try:
        scaler = torch.amp.GradScaler(dev_type, enabled=(a.dtype == "fp16"))
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=(a.dtype == "fp16"))
    loader = MixtureLoader(a.data_dir, a.langs, T, a.micro_batch, _parse_mix(a.mix, a.langs),
                           rank, world, a.seed, "train", device)
    fpt = flops_per_token(cfg, T)
    gpu_name = torch.cuda.get_device_name(0) if dev_type == "cuda" else "cpu"
    peak = next((v for k, v in PEAK_FLOPS.items() if k in gpu_name), None)

    step, tokens_seen, prev_elapsed, wandb_id = 0, 0, 0.0, None
    ckpt_path = None
    if a.benchmark_steps == 0 and a.resume != "none":
        if a.resume == "auto":
            ckpt_path = find_latest_checkpoint([a.out_dir] + list(a.resume_dirs))
        else:
            ckpt_path = find_latest_checkpoint([a.resume])
    if ckpt_path:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        raw_model.load_state_dict(strip_compile_prefix(ck["model"]))
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        loader.load_state_dict(ck["loader"])
        step, tokens_seen, prev_elapsed = ck["step"], ck["tokens"], ck.get("elapsed_total", 0.0)
        wandb_id = ck.get("wandb_id")
        torch.set_rng_state(ck["torch_rng"])
        if master:
            _carry_logs(ckpt_path, a.out_dir, step)
            print(f"resumed from {ckpt_path} at step {step} ({tokens_seen/1e6:.1f}M tokens)", flush=True)
        del ck

    if master:
        pc = param_counts(cfg)
        print(f"model {cfg.to_dict()}")
        print(f"params total {pc['total']/1e6:.2f}M | non-embedding {pc['non_embedding']/1e6:.2f}M | "
              f"FLOPs/token {fpt/1e6:.0f}M | device {gpu_name} x{world} | dtype {a.dtype}")
        print(f"batch {a.global_batch_tokens} tok = {a.micro_batch} x {T} x {world} ranks x {accum} accum | "
              f"max_steps {max_steps} ({max_steps*a.global_batch_tokens/1e9:.2f}B tokens) | "
              f"train tokens available {loader.tokens_per_lang()}", flush=True)
        cfg.save(os.path.join(a.out_dir, "model_config.json"))
        json.dump(vars(a), open(os.path.join(a.out_dir, "train_args.json"), "w"), indent=2)

    # torch.compile (falls back to eager if it fails on this GPU / torch build)
    use_compile = a.compile == "on" or (a.compile == "auto" and dev_type == "cuda")
    if use_compile:
        try:
            cmodel = torch.compile(model)
            # warm up with the real micro-batch shape so the compiled graph stays static
            x = y = torch.zeros(a.micro_batch, T, dtype=torch.long, device=device)
            with autocast:
                _, l = cmodel(x, y)
            l.backward()
            model.zero_grad(set_to_none=True)
            model = cmodel
            if master:
                print("torch.compile: enabled", flush=True)
        except Exception as e:  # pragma: no cover - depends on GPU stack
            model.zero_grad(set_to_none=True)
            if master:
                print(f"torch.compile failed, using eager: {type(e).__name__}: {str(e)[:300]}", flush=True)
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank] if dev_type == "cuda" else None)

    run = None
    if master and os.environ.get("WANDB_API_KEY") and a.benchmark_steps == 0:
        try:
            import wandb
            wandb_id = wandb_id or wandb.util.generate_id()
            run = wandb.init(project=a.wandb_project, id=wandb_id, resume="allow", config=vars(a))
        except Exception as e:
            print(f"wandb disabled: {e}")
    tlog = CSVLogger(os.path.join(a.out_dir, "train_log.csv"), TRAIN_FIELDS) if master and not a.benchmark_steps else None
    vlog = CSVLogger(os.path.join(a.out_dir, "val_log.csv"), VAL_FIELDS) if master and not a.benchmark_steps else None
    token_bytes = torch.tensor(tokenizer.token_bytes(), device=device) if tokenizer else None
    val_sets = {}
    for l in a.langs:
        try:
            val_sets[l] = ValSet(a.data_dir, l, T, a.eval_chunks)
        except AssertionError:
            pass

    @torch.no_grad()
    def evaluate():
        raw_model.eval()
        out = {}
        for l, vs in val_sets.items():
            s = torch.zeros(3, device=device, dtype=torch.float64)  # loss sum, token count, byte count
            for x, y in vs.batches(a.eval_batch or a.micro_batch, rank, world, device):
                with autocast:
                    _, loss = raw_model(x, y, reduction="none")
                s[0] += loss.double().sum()
                s[1] += y.numel()
                if token_bytes is not None:
                    s[2] += token_bytes[y].double().sum()
            if ddp:
                dist.all_reduce(s)
            out[f"val_loss_{l}"] = (s[0] / s[1]).item()
            out[f"ppl_{l}"] = math.exp(out[f"val_loss_{l}"])
            if token_bytes is not None and s[2] > 0:
                out[f"bpb_{l}"] = (s[0] / math.log(2) / s[2]).item()
        if out:
            out["val_loss"] = float(np.mean([out[f"val_loss_{l}"] for l in val_sets]))
        raw_model.train()
        return out

    def samples():
        if tokenizer is None or a.sample_tokens <= 0:
            return
        from .sampling import generate
        g = torch.Generator(device=device).manual_seed(0)
        lines = [f"--- step {step} ---"]
        for l in a.langs:
            ids = generate(raw_model, [tokenizer.lang_ids[l]], max_new_tokens=a.sample_tokens, temperature=0.8,
                           top_k=50, top_p=0.95, repetition_penalty=1.1, stop_ids=[tokenizer.eot_id], generator=g)
            lines.append(f"[{l}] " + tokenizer.decode(ids).replace("\n", " "))
        raw_model.train()
        with open(os.path.join(a.out_dir, "samples.txt"), "a") as f:
            f.write("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)

    last_saved = [-1]

    def save_ckpt():
        if a.benchmark_steps or last_saved[0] == step:
            return
        last_saved[0] = step
        if master:
            ck = dict(model=raw_model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                      loader=loader.state_dict(), step=step, tokens=tokens_seen, model_config=cfg.to_dict(),
                      train_args=vars(a), torch_rng=torch.get_rng_state(), wandb_id=wandb_id,
                      elapsed_total=prev_elapsed + (time.time() - t_start))
            p = os.path.join(a.out_dir, "ckpt_latest.pt")
            atomic_save(ck, p)
            json.dump({"step": step, "tokens": tokens_seen}, open(p[:-3] + ".json", "w"))
            print(f"checkpoint saved at step {step}", flush=True)
        if ddp:
            dist.barrier()

    model.train()
    t_start = time.time()
    t_log, tok_log = time.time(), 0
    max_seconds = a.max_hours * 3600 if a.max_hours else None
    bench_t0, bench_steps_done = None, 0
    stop_reason = "done"
    while step < max_steps:
        if a.stop_step is not None and step >= a.stop_step:
            stop_reason = "stop_step"; break
        if max_seconds is not None:
            over = torch.tensor(float(time.time() - t_start > max_seconds), device=device)
            if ddp:
                dist.all_reduce(over, op=dist.ReduceOp.MAX)
            if over.item() > 0:
                stop_reason = "time_budget"; break
        lr = cosine_lr(step, max_steps, a.lr, a.warmup_steps, a.min_lr_ratio)
        for g in optimizer.param_groups:
            g["lr"] = lr
        loss_acc = torch.zeros((), device=device)
        for micro in range(accum):
            x, y, _ = loader.next_batch()
            sync_ctx = model.no_sync() if (ddp and micro < accum - 1) else nullcontext()
            with sync_ctx:
                with autocast:
                    _, loss = model(x, y)
                loss = loss / accum
                scaler.scale(loss).backward()
            loss_acc += loss.detach()
        scaler.unscale_(optimizer)
        gnorm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), a.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        tokens_seen += a.global_batch_tokens
        tok_log += a.global_batch_tokens

        if a.benchmark_steps:
            if step == 3:  # skip compile / warmup steps
                if dev_type == "cuda":
                    torch.cuda.synchronize()
                bench_t0 = time.time()
            if step >= 3 + a.benchmark_steps:
                if dev_type == "cuda":
                    torch.cuda.synchronize()
                dt = time.time() - bench_t0
                tps = a.benchmark_steps * a.global_batch_tokens / dt
                res = dict(model=a.model, params=raw_model.num_params(), device=gpu_name, world=world,
                           dtype=a.dtype, compile=use_compile, micro_batch=a.micro_batch, seq_len=T,
                           tok_per_sec=tps, flops_per_token=fpt,
                           mfu=(tps * fpt / (peak * world)) if peak else None,
                           peak_mem_gb=(torch.cuda.max_memory_allocated() / 1e9) if dev_type == "cuda" else None,
                           hours_per_1B_tokens=1e9 / tps / 3600)
                if master:
                    print("BENCHMARK " + json.dumps(res), flush=True)
                    with open(os.path.join(a.out_dir, "benchmark.jsonl"), "a") as f:
                        f.write(json.dumps(res) + "\n")
                stop_reason = "benchmark"; break
            continue

        if step % a.log_interval == 0 or step == max_steps:
            if ddp:
                dist.all_reduce(loss_acc)
                loss_acc /= world
            dt = time.time() - t_log
            tps = tok_log / dt
            t_log, tok_log = time.time(), 0
            row = dict(step=step, tokens=tokens_seen, loss=round(loss_acc.item(), 5), lr=f"{lr:.3e}",
                       grad_norm=round(float(gnorm), 4), tok_per_sec=round(tps), loss_scale=scaler.get_scale() if scaler.is_enabled() else 1,
                       elapsed_s=round(prev_elapsed + time.time() - t_start, 1),
                       mfu=round(tps * fpt / (peak * world), 4) if peak else "")
            if master:
                tlog.log(row)
                print(" | ".join(f"{k} {v}" for k, v in row.items()), flush=True)
                if run:
                    run.log({k: float(v) for k, v in row.items() if v != ""}, step=step)
        if step % a.eval_interval == 0 or step == max_steps:
            ev = evaluate()
            if master and ev:
                vlog.log(dict(step=step, tokens=tokens_seen, **{k: round(v, 5) for k, v in ev.items()}))
                print("eval " + " ".join(f"{k} {v:.4f}" for k, v in ev.items()), flush=True)
                if run:
                    run.log(ev, step=step)
                samples()
        if step % a.ckpt_interval == 0 or step == max_steps:
            save_ckpt()

    if stop_reason in ("stop_step", "time_budget"):
        save_ckpt()
    if master:
        print(f"finished: {stop_reason} at step {step}/{max_steps}, {tokens_seen/1e6:.1f}M tokens", flush=True)
        if tlog:
            tlog.close(); vlog.close()
        if run:
            run.finish()
    if ddp:
        dist.destroy_process_group()
    return dict(step=step, stop_reason=stop_reason, model=raw_model)


if __name__ == "__main__":
    main()
