"""End-to-end check of the whole pipeline on the `nano` preset with synthetic data (~3 min).

  python smoke_test.py
"""
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from ncm.chat_format import format_conv, parse_segment
from ncm.config import get_config
from ncm.generator import NCMGenerator
from ncm.sampling import generate
from ncm.synth_tools import Synth
from ncm.utils import load_generator, load_vq, pick_device
from prepare_data import write_dataset

PY = sys.executable
ok = True


def check(name, cond, info=""):
    global ok
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}")
    ok &= bool(cond)


def fake_pools(rng):
    words = "the a team meeting report budget lunch project friday update client invoice plan schedule review".split()
    emails = [{"body": " ".join(rng.choice(words) for _ in range(rng.randint(30, 80))),
               "subject": " ".join(rng.choice(words) for _ in range(rng.randint(2, 5)))} for _ in range(200)]
    wiki = [{"title": f"Topic{i}", "text": ". ".join(" ".join(rng.choice(words) for _ in range(10)) for _ in range(4))}
            for i in range(100)]
    chats = [[{"role": "user", "content": "hello how are you"}, {"role": "assistant", "content": "I'm good, thanks for asking!"}]] * 5
    arts = [{"article": " ".join(rng.choice(words) for _ in range(150)), "highlights": "the plan is good.\nthe team met."}] * 20
    docs = [" ".join(rng.choice(words) for _ in range(rng.randint(40, 200))) + "." for _ in range(3000)]
    return emails, wiki, chats, arts, docs


def main():
    rng = random.Random(0)
    dev = pick_device()
    tmp = Path(tempfile.mkdtemp(prefix="ncm_smoke_"))
    emails, wiki, chats, arts, docs = fake_pools(rng)
    syn = Synth(seed=0, emails=emails, articles=arts, wiki=wiki, chats=chats)
    convs = [syn.sample() for _ in range(1500)]
    val = [syn.sample() for _ in range(100)]
    cfg = get_config("nano")
    bpe = write_dataset(tmp, docs[:-100], docs[-100:], convs, val, cfg.vocab_size, 256, cfg.block)

    # 1. format round trip
    c = syn.alarm()
    toks, loss = format_conv(bpe, c, cfg.block)
    check("segments are block aligned", len(toks) % cfg.block == 0)
    seg = format_conv(bpe, [c[2]], cfg.block)[0]
    role, call = parse_segment(bpe, seg)
    check("tool call round trip", role == "tool" and call["name"] == "set_alarm", str(call))

    # 2. VQ training script (short)
    r = subprocess.run([PY, "train_vq.py", "--data", str(tmp), "--preset", "nano", "--out", str(tmp / "vq"),
                        "--steps", "400", "--batch", "256", "--eval-every", "200"], capture_output=True, text=True)
    print(r.stdout[-800:], r.stderr[-2000:])
    check("train_vq.py runs", r.returncode == 0)
    vq, _ = load_vq(tmp / "vq" / "best.pt", dev)
    blocks = torch.tensor(toks, device=dev).view(-1, cfg.block)
    acc = (vq.decode(vq.encode(blocks)) == blocks).float().mean().item()
    check("VQ reconstructs (smoke threshold 0.5)", acc > 0.5, f"acc={acc:.3f}")

    # 3. encode codes
    r = subprocess.run([PY, "encode_codes.py", "--data", str(tmp), "--vq", str(tmp / "vq" / "best.pt")], capture_output=True, text=True)
    check("encode_codes.py runs", r.returncode == 0, r.stderr[-500:])

    # 4. generator training script, both phases
    r = subprocess.run([PY, "train_gen.py", "--data", str(tmp), "--vq", str(tmp / "vq" / "best.pt"), "--preset", "nano",
                        "--phase", "pretrain", "--ctx", "128", "--batch", "8", "--steps", "60", "--eval-every", "30",
                        "--out", str(tmp / "pre")], capture_output=True, text=True)
    print(r.stdout[-1200:], r.stderr[-2000:])
    check("train_gen.py pretrain runs", r.returncode == 0)
    r = subprocess.run([PY, "train_gen.py", "--data", str(tmp), "--vq", str(tmp / "vq" / "best.pt"), "--preset", "nano",
                        "--phase", "task", "--ctx", "256", "--batch", "4", "--steps", "30", "--eval-every", "30",
                        "--init", str(tmp / "pre" / "last.pt"), "--out", str(tmp / "task")], capture_output=True, text=True)
    print(r.stdout[-1200:], r.stderr[-2000:])
    check("train_gen.py task runs", r.returncode == 0)
    r = subprocess.run([PY, "train_gen.py", "--data", str(tmp), "--vq", str(tmp / "vq" / "best.pt"), "--preset", "nano",
                        "--phase", "pretrain", "--ctx", "128", "--batch", "8", "--steps", "70", "--eval-every", "35",
                        "--out", str(tmp / "pre"), "--resume"], capture_output=True, text=True)
    check("resume works", r.returncode == 0 and "step=70" in r.stdout, r.stderr[-500:])

    # 5. training forward == generation path (no leaks, cache correct)
    gen, gcfg = load_generator(tmp / "task" / "last.pt", dev)
    gen = gen.float()
    T = 4 * gcfg.block
    tk = torch.tensor(toks[:T], device=dev)[None]
    if tk.size(1) < T:
        tk = torch.nn.functional.pad(tk, (0, T - tk.size(1)))
    codes = vq.encode(tk.view(-1, gcfg.block)).view(1, -1, gcfg.n_codes)
    with torch.no_grad():
        _, _, lg = gen(tk, codes, corrupt=False, return_logits=True, chunk=2)
        maxdiff = 0.0
        for b in range(T // gcfg.block):
            cache = [None] * len(gen.blocks)
            if b:
                cache = gen.extend(cache, tk[:, : b * gcfg.block], 0)
            parts = vq.split(codes[0, b][None])
            zhat = vq.embed.new_zeros(1, gcfg.block, gcfg.code_dim)
            for k in range(len(gcfg.levels)):
                l2 = gen.level_logits(cache, zhat, k, b)
                maxdiff = max(maxdiff, (l2[0] - lg[k][0, b]).abs().max().item())
                zhat = zhat + vq.contrib(k, vq.lookup(k, parts[k]))
        # incremental cache == one-shot prefill
        c1 = gen.extend([None] * len(gen.blocks), tk, 0)
        c2 = gen.extend([None] * len(gen.blocks), tk[:, :gcfg.block], 0)
        for b in range(1, T // gcfg.block):
            c2 = gen.extend(c2, tk[:, b * gcfg.block:(b + 1) * gcfg.block], b * gcfg.block)
        cdiff = max((a[0] - b_[0]).abs().max().item() for a, b_ in zip(c1, c2))
    check("training logits == generation logits", maxdiff < 2e-3, f"maxdiff={maxdiff:.2e}")
    check("incremental KV cache == prefill", cdiff < 2e-3, f"maxdiff={cdiff:.2e}")

    # 6. sampling
    out = generate(gen, toks[:2 * gcfg.block], max_blocks=3, temperature=0.0, stop_ids=(bpe.end,), pad_id=bpe.pad)
    out2 = generate(gen, toks[:2 * gcfg.block], max_blocks=3, temperature=0.0, stop_ids=(bpe.end,), pad_id=bpe.pad)
    check("greedy generation deterministic", out == out2 and len(out) > 0, f"{len(out)} tokens")
    print("\nALL PASS" if ok else "\nSOME CHECKS FAILED", f"(tmp: {tmp})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
