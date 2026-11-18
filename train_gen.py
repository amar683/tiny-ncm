"""Stage 2: train the NCM generator on frozen VQ codes.

  python train_gen.py --phase pretrain --ctx 1024 --minutes 150
  python train_gen.py --phase task --ctx 4096 --batch 2 --accum 4 --init runs/pretrain/last.pt --minutes 60
"""
import argparse
import time

import numpy as np
import torch

from ncm.bpe import BPE
from ncm.chat_format import load_registry, prompt_tokens, system_prompt
from ncm.config import get_config
from ncm.data import Data
from ncm.generator import NCMGenerator
from ncm.muon import build_optimizers, set_lr, wsd
from ncm.sampling import generate
from ncm.utils import Logger, autocast, load_ckpt, load_vq, pick_device, save_ckpt, sync


@torch.no_grad()
def evaluate(gen, data, dev, T, B=8, n=8, bf16=True):
    gen.eval()
    rng = np.random.RandomState(0)
    out = {}
    for src in ("pretrain", "task"):
        if (src == "pretrain" and "val" not in data.streams) or (src == "task" and "val" not in data.tasks):
            continue
        tot, per = 0.0, {}
        for _ in range(n):
            t, c, bm = data.gen_batch(B, T, src, "val", rng)
            with autocast(dev, bf16):
                loss, st = gen(t.to(dev), c.to(dev), None if bm is None else bm.to(dev), corrupt=False)
            tot += loss.item()
            for k, v in st.items():
                per[k] = per.get(k, 0.0) + v.item() / n
        out[f"{src}_loss"] = tot / n
        out.update({f"{src}_{k}": v for k, v in per.items()})
    gen.train()
    return out


@torch.no_grad()
def show_samples(gen, bpe, dev, phase):
    import datetime as dt
    gen.eval()
    if phase == "pretrain":
        prompts = ["Once upon a time, there was a little", "Dear team,\n\nI wanted to let you know that",
                   "The water cycle is"]
        for p in prompts:
            ids = bpe.encode(p)
            out = generate(gen, ids, max_blocks=3, temperature=0.8, stop_ids=(bpe.eos,), pad_id=bpe.pad, seed=0)
            print(f"  [{p!r}] -> {bpe.decode(out)!r}")
    else:
        tools = load_registry()
        now = dt.datetime(2026, 11, 17, 9, 14)
        for q in ["remind me to call mom tomorrow at 6pm", "wake me up at 7", "who was Ada Lovelace?",
                  "hi! how are you today?"]:
            msgs = [{"role": "system", "content": system_prompt(now, tools)}, {"role": "user", "content": q}]
            ids = prompt_tokens(bpe, msgs, gen.cfg.block)
            out = generate(gen, ids, max_blocks=4, temperature=0.0, stop_ids=(bpe.end,), pad_id=bpe.pad)
            print(f"  [{q}] -> {bpe.decode(out, skip_special=False).replace('<|pad|>', '')!r}")
    gen.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--vq", default="runs/vq/best.pt")
    ap.add_argument("--preset", default="base")
    ap.add_argument("--phase", choices=["pretrain", "task"], default="pretrain")
    ap.add_argument("--init", default=None, help="generator checkpoint to start from")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--minutes", type=float, default=150)
    ap.add_argument("--task-frac", type=float, default=0.8, help="task phase: share of task batches")
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-bf16", action="store_true")
    args = ap.parse_args()
    out = args.out or f"runs/{args.phase}"
    bf16 = not args.no_bf16

    dev = pick_device(args.device)
    bpe = BPE.load(f"{args.data}/bpe.json")
    vq, vcfg = load_vq(args.vq, dev)
    cfg = get_config(args.preset, **{k: getattr(vcfg, k) for k in
                                     ("vocab_size", "block", "levels", "vq_dim", "vq_depth", "vq_heads", "code_dim", "codebook")},
                     ctx=args.ctx)
    gen = NCMGenerator(cfg, vq).to(dev)
    if args.init:
        sd = load_ckpt(args.init)["model"]
        gen.load_state_dict(sd)
        gen.vq.load_state_dict(vq.state_dict())
    data = Data(args.data, cfg)
    opts = build_optimizers(gen, lr_muon=0.02 * args.lr, lr_adam=3e-3 * args.lr)
    step, total = 0, args.steps
    if args.resume:
        ck = load_ckpt(f"{out}/last.pt")
        gen.load_state_dict(ck["model"])
        for o, s in zip(opts, ck["opts"]):
            o.load_state_dict(s)
        step, total = ck["step"], (args.steps or ck["extra"].get("total", total))
    log = Logger(f"{out}/log.jsonl")
    print(f"generator params (trainable): {gen.n_params() / 1e6:.2f}M  ctx={args.ctx} device={dev}")
    rng = np.random.RandomState(4321 + step)
    t0, best = time.time(), float("inf")
    warmup = args.warmup if not args.init else 50
    while True:
        if total and step >= total:
            break
        set_lr(opts, wsd(step, total or 10**9, warmup=warmup, decay_frac=0.2 if args.phase == "pretrain" else 0.3))
        for o in opts:
            o.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            src = "task" if (args.phase == "task" and rng.rand() < args.task_frac) else "pretrain"
            t, c, bm = data.gen_batch(args.batch, args.ctx, src, "train", rng)
            with autocast(dev, bf16):
                loss, st = gen(t.to(dev), c.to(dev), None if bm is None else bm.to(dev))
            (loss / args.accum).backward()
        gn = torch.nn.utils.clip_grad_norm_([p for p in gen.parameters() if p.requires_grad], 1.0)
        for o in opts:
            o.step()
        step += 1
        if not total and step == 20:
            sync(dev)
            per = (time.time() - t0) / 20
            total = max(200, int(args.minutes * 60 / per))
            print(f"{per:.2f} s/step -> {total} steps for {args.minutes} min "
                  f"({total * args.batch * args.accum * args.ctx / 1e6:.0f}M tokens)")
        if step % 25 == 0:
            log(step, loss=loss.item(), gnorm=gn.item(), lr=opts[0].param_groups[0]["lr"], src=src,
                **{k: v.item() for k, v in st.items()})
        if step % args.eval_every == 0 or step == total:
            ev = evaluate(gen, data, dev, min(args.ctx, 1024), bf16=bf16)
            log(step, **{f"val_{k}": v for k, v in ev.items()})
            save_ckpt(f"{out}/last.pt", gen, cfg, step, opts, {"total": total})
            key = ev.get("task_loss" if args.phase == "task" else "pretrain_loss", 0)
            if key < best:
                best = key
                save_ckpt(f"{out}/best.pt", gen, cfg, step, None, ev)
            show_samples(gen, bpe, dev, args.phase)
    print("done")


if __name__ == "__main__":
    main()
