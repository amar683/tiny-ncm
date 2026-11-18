"""Write the release file: model weights + config only (no optimizer, no training paths).

python export_weights.py --ckpt runs/task6/best.pt --out weights/tiny-ncm.pt
"""
import argparse
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/task6/best.pt")
    ap.add_argument("--out", default="weights/tiny-ncm.pt")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": ck["cfg"], "model": {k: v.contiguous() for k, v in ck["model"].items()}, "step": ck.get("step", 0), "extra": {}}, a.out)
    n = sum(v.numel() for v in ck["model"].values())
    print(f"{a.out}: {n / 1e6:.1f}M parameters, {Path(a.out).stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
