"""Measure training throughput on this machine and recommend step counts for a time budget.

  python bench.py
"""
import argparse
import time

import torch
import torch.nn.functional as F

from ncm import layers
from ncm.config import get_config
from ncm.generator import NCMGenerator
from ncm.muon import build_optimizers
from ncm.utils import autocast, pick_device, sync
from ncm.vq import MultiscaleVQ


def timeit(fn, n=6, warm=2, dev=None):
    for _ in range(warm):
        fn()
    sync(dev)
    t = time.time()
    for _ in range(n):
        fn()
    sync(dev)
    return (time.time() - t) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="base")
    ap.add_argument("--configs", default="1024:16,1024:8,4096:4,4096:2")
    args = ap.parse_args()
    dev = pick_device()
    cfg = get_config(args.preset)
    vq = MultiscaleVQ(cfg).to(dev)
    opts = build_optimizers(vq)
    print(f"VQ {sum(p.numel() for p in vq.parameters()) / 1e6:.1f}M params")
    for bf16 in (True, False):
        for B in (256, 512):
            x = torch.randint(0, cfg.vocab_size, (B, cfg.block), device=dev)

            def step():
                with autocast(dev, bf16):
                    lg, cm, _ = vq(x)
                    loss = F.cross_entropy(lg.float().flatten(0, 1), x.flatten()) + cm
                loss.backward()
                for o in opts:
                    o.step()
                    o.zero_grad()
            s = timeit(step, dev=dev)
            print(f"  VQ bf16={bf16} B={B}: {s * 1000:.0f} ms/step, {B * cfg.block / s / 1e3:.0f}k tok/s")

    gen = NCMGenerator(cfg, MultiscaleVQ(cfg)).to(dev)
    gopts = build_optimizers(gen)
    print(f"generator {gen.n_params() / 1e6:.1f}M trainable params")
    for impl in ("auto", "sdpa"):
        layers.ATTN_IMPL = impl
        for bf16 in (True, False):
            for spec in args.configs.split(","):
                T, B = map(int, spec.split(":"))
                t = torch.randint(0, cfg.vocab_size, (B, T), device=dev)
                c = torch.randint(0, cfg.codebook, (B, T // cfg.block, cfg.n_codes), device=dev)

                def step():
                    with autocast(dev, bf16):
                        loss, _ = gen(t, c)
                    loss.backward()
                    for o in gopts:
                        o.step()
                        o.zero_grad()
                try:
                    s = timeit(step, n=4, warm=1, dev=dev)
                    print(f"  gen attn={impl} bf16={bf16} T={T} B={B}: {s:.2f} s/step, {B * T / s / 1e3:.1f}k tok/s, "
                          f"{B * T / s * 3600 / 1e6:.0f}M tok/h", flush=True)
                except RuntimeError as e:
                    print(f"  gen attn={impl} bf16={bf16} T={T} B={B}: FAILED {str(e)[:120]}")
                if dev.type == "mps":
                    torch.mps.empty_cache()


if __name__ == "__main__":
    main()
