"""Stage 1: train the multiscale residual VQ-VAE text tokenizer.

  python train_vq.py --data data --preset base --minutes 45
"""
import argparse
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from ncm.bpe import BPE
from ncm.config import get_config
from ncm.data import Data
from ncm.muon import build_optimizers, set_lr, wsd
from ncm.utils import Logger, autocast, load_ckpt, pick_device, save_ckpt, sync
from ncm.vq import MultiscaleVQ


@torch.no_grad()
def tool_segments(tok_rows, tool_id, end_id, block, limit=300):
    """Yield arrays of block indices (row, blk) covering each tool-call segment."""
    segs = []
    for r in range(len(tok_rows)):
        row = np.asarray(tok_rows[r])
        starts = np.where(row == tool_id)[0]
        for s in starts:
            e = s + np.argmax(row[s:] == end_id)
            segs.append((r, s // block, e // block))
            if len(segs) >= limit:
                return segs
    return segs


@torch.no_grad()
def evaluate(vq, data, device, bpe=None, n=2048):
    vq.eval()
    cfg = vq.cfg
    rng = np.random.RandomState(0)
    res = {}
    blocks = data.vq_blocks(n, "val", rng, task_frac=0.0).to(device)
    codes = vq.encode(blocks)
    for k in range(1, len(cfg.levels) + 1):
        acc = (vq.decode(codes, k) == blocks).float().mean().item()
        res[f"acc_l{k}"] = acc
    res["acc"] = res[f"acc_l{len(cfg.levels)}"]
    res["block_exact"] = (vq.decode(codes) == blocks).all(1).float().mean().item()
    if "val" in data.tasks:
        tb = data.vq_blocks(n, "val", rng, task_frac=1.0).to(device)
        res["task_acc"] = (vq.decode(vq.encode(tb)) == tb).float().mean().item()
        if bpe is not None:
            tok = data.tasks["val"][0]
            segs = tool_segments(tok, bpe.ids["<|tool|>"], bpe.end, cfg.block)
            ok = 0
            for r, b0, b1 in segs:
                blk = torch.from_numpy(np.asarray(tok[r]).reshape(-1, cfg.block)[b0:b1 + 1].astype(np.int64)).to(device)
                ok += int((vq.decode(vq.encode(blk)) == blk).all())
            res["tool_exact"] = ok / max(1, len(segs))
    res["usage"] = vq.usage().mean().item()
    vq.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--preset", default="base")
    ap.add_argument("--out", default="runs/vq")
    ap.add_argument("--steps", type=int, default=0, help="0 = derive from --minutes")
    ap.add_argument("--minutes", type=float, default=45)
    ap.add_argument("--batch", type=int, default=512, help="blocks per step")
    ap.add_argument("--lr", type=float, default=1.0, help="multiplier on default LRs")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-bf16", action="store_true")
    args = ap.parse_args()

    dev = pick_device(args.device)
    bpe = BPE.load(f"{args.data}/bpe.json")
    cfg = get_config(args.preset, vocab_size=len(bpe))
    data = Data(args.data, cfg, need_codes=False)
    vq = MultiscaleVQ(cfg).to(dev)
    opts = build_optimizers(vq, lr_muon=0.02 * args.lr, lr_adam=2e-3 * args.lr)
    step, total = 0, args.steps
    log = Logger(f"{args.out}/log.jsonl")
    if args.resume:
        ck = load_ckpt(f"{args.out}/last.pt")
        vq.load_state_dict(ck["model"])
        for o, s in zip(opts, ck["opts"]):
            o.load_state_dict(s)
        step, total = ck["step"], (args.steps or ck["extra"].get("total", total))
    print(f"VQ params: {sum(p.numel() for p in vq.parameters()) / 1e6:.2f}M  device={dev}")
    rng = np.random.RandomState(1234 + step)
    nl = len(cfg.levels)
    t_start, best = time.time(), -1.0
    while True:
        if total and step >= total:
            break
        x = data.vq_blocks(args.batch, "train", rng).to(dev)
        cutoff = nl if rng.rand() < 0.5 else rng.randint(1, nl + 1)  # nested dropout
        set_lr(opts, wsd(step, total or 10**9, warmup=200))
        with autocast(dev, not args.no_bf16):
            logits, commit, _ = vq(x, cutoff=cutoff)
            rec = F.cross_entropy(logits.float().flatten(0, 1), x.flatten())
            loss = rec + commit
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vq.parameters(), 1.0)
        for o in opts:
            o.step()
        step += 1
        if step % 100 == 0:
            vq.reset_dead()
        if not total and step == 30:
            sync(dev)
            per = (time.time() - t_start) / 30
            total = max(300, int(args.minutes * 60 / per))
            print(f"{per * 1000:.0f} ms/step -> {total} steps for {args.minutes} min")
        if step % 50 == 0:
            log(step, loss=loss.item(), rec=rec.item(), commit=commit.item(), cutoff=cutoff,
                lr=opts[0].param_groups[0]["lr"])
        if step % args.eval_every == 0 or step == total:
            ev = evaluate(vq, data, dev, bpe)
            log(step, **{f"val_{k}": v for k, v in ev.items()})
            save_ckpt(f"{args.out}/last.pt", vq, cfg, step, opts, {"total": total})
            if ev["acc"] > best:
                best = ev["acc"]
                save_ckpt(f"{args.out}/best.pt", vq, cfg, step, None, ev)
    print(f"done. best val acc {best:.4f}")


if __name__ == "__main__":
    main()
