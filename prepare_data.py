"""Download text, train the BPE, build the pretraining stream and the packed chat/tool task set.

  python prepare_data.py --out data --pretrain-tokens 200e6
"""
import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np

from ncm.bpe import BPE
from ncm.chat_format import format_conv
from ncm.data import pack_rows, write_u16
from ncm.synth_tools import Synth

REFUSAL = re.compile(r"(I can't|I cannot|I'm sorry, but|I am sorry, but|As an AI|I'm not able to|I am not able to|"
                     r"I won't|I'm unable to|I am unable to|not appropriate|against my)", re.I)
CODEY = re.compile(r"(```|\bdef \w+\(|function\s*\(|#include|console\.log|\bSELECT\b.*\bFROM\b|<\/?div)")


def stream(name, config=None, split="train"):
    from datasets import load_dataset
    return load_dataset(name, config, split=split, streaming=True)


def take_chars(it, budget, field="text", filt=None, min_len=50):
    out, n = [], 0
    for ex in it:
        t = ex[field] if isinstance(field, str) else field(ex)
        if not t or len(t) < min_len or (filt and not filt(ex)):
            continue
        out.append(t)
        n += len(t)
        if n >= budget:
            break
    return out


def safe(fn, label):
    t = time.time()
    try:
        r = fn()
        print(f"  {label}: {len(r)} items ({time.time() - t:.0f}s)", flush=True)
        return r
    except Exception as e:  # noqa: BLE001 - a missing source must not stop the run
        print(f"  {label}: FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
        return []


# ---- sources ---------------------------------------------------------------------------
def fineweb(chars):
    return take_chars(stream("HuggingFaceFW/fineweb-edu", "sample-10BT"), chars,
                      filt=lambda e: e.get("int_score", 3) >= 4 and len(e["text"]) < 12000)


def tinystories(chars):
    return take_chars(stream("roneneldan/TinyStories"), chars)


def enron(chars):
    def body(e):
        for k in ("body", "text", "content", "message"):
            if k in e and e[k]:
                return e[k]
        return None
    return take_chars(stream("corbt/enron-emails"), chars, field=body)


def aeslc():
    out = []
    for split in ("train", "validation"):
        for e in stream("Yale-LILY/aeslc", split=split):
            b, s = e.get("email_body"), e.get("subject_line")
            if b and s and 80 < len(b) < 6000:
                out.append({"body": b, "subject": s})
    return out


def cnn(n):
    out = []
    for e in stream("abisee/cnn_dailymail", "3.0.0"):
        if len(e["article"]) < 9000:
            out.append({"article": e["article"], "highlights": e["highlights"]})
        if len(out) >= n:
            break
    return out


def wiki(n):
    out = []
    for e in stream("wikimedia/wikipedia", "20231101.en"):
        t = e["text"].split("\n\n")[0]
        if 200 < len(t) and "may refer to" not in t:
            out.append({"title": e["title"], "text": t[:1500]})
        if len(out) >= n:
            break
    return out


def chats_from(name, n, split="train", config=None):
    out = []
    for e in stream(name, config, split):
        msgs = e.get("messages") or e.get("conversations")
        if not msgs:
            continue
        conv = []
        okc = True
        for m in msgs:
            role = m.get("role") or m.get("from")
            text = m.get("content") or m.get("value") or ""
            role = {"human": "user", "gpt": "assistant"}.get(role, role)
            if role == "system":
                continue
            if role not in ("user", "assistant") or CODEY.search(text) or (role == "assistant" and REFUSAL.search(text)):
                okc = False
                break
            conv.append({"role": role, "content": text.strip()})
        if okc and len(conv) >= 2 and conv[0]["role"] == "user" and sum(len(m["content"]) for m in conv) < 6000:
            out.append(conv)
        if len(out) >= n:
            break
    return out


def xlam(n):
    from ncm.chat_format import tool_call_text
    out = []
    for e in stream("Salesforce/xlam-function-calling-60k"):
        try:
            tools = json.loads(e["tools"])
            answers = json.loads(e["answers"])
        except Exception:
            continue
        sig = " ".join(f"{t['name']}({','.join(t.get('parameters', {}).keys())})" for t in tools)
        conv = [{"role": "user", "content": e["query"]}]
        conv += [{"role": "tool", "content": tool_call_text(a["name"], a.get("arguments", {}))} for a in answers]
        out.append(("Tools: " + sig, conv))
        if len(out) >= n:
            break
    return out


# ---- dataset writing (also used by smoke_test.py) --------------------------------------
def write_dataset(root, docs, val_docs, convs, val_convs, vocab, R, block, bpe=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if bpe is None:
        print("training BPE...", flush=True)
        rng = random.Random(0)
        sample = rng.sample(docs, min(len(docs), 60000))
        conv_text = ["\n".join(m["content"] for m in c) for c in rng.sample(convs, min(len(convs), 20000))]
        bpe = BPE.train(sample + conv_text, vocab_size=vocab)
        bpe.save(root / "bpe.json")
    for name, ds in (("pretrain", docs), ("pretrain_val", val_docs)):
        print(f"encoding {name} ({len(ds)} docs)...", flush=True)
        ids = []
        for i in range(0, len(ds), 20000):
            for e in bpe.encode_batch(ds[i:i + 20000]):
                ids += e + [bpe.eos]
        write_u16(root / f"{name}.bin", ids)
        print(f"  {name}: {len(ids) / 1e6:.1f}M tokens", flush=True)
    for name, cs in (("task", convs), ("task_val", val_convs)):
        fm = [format_conv(bpe, c, block, max_len=R, max_segment=R - 4 * block) for c in cs]
        fm = [f for f in fm if f is not None]
        rows, bm = pack_rows(fm, R, bpe.pad, block)
        rows.tofile(root / f"{name}.bin")
        bm.tofile(root / f"{name}_bmask.bin")
        print(f"  {name}: {len(fm)} convs -> {len(rows)} rows of {R} ({rows.size / 1e6:.1f}M tokens, "
              f"{bm.mean():.0%} trained blocks)", flush=True)
    (root / "meta.json").write_text(json.dumps({"R": R, "block": block, "vocab": len(bpe)}))
    return bpe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--pretrain-tokens", type=float, default=200e6)
    ap.add_argument("--synth", type=int, default=60000)
    ap.add_argument("--vocab", type=int, default=16384)
    ap.add_argument("--R", type=int, default=4096)
    ap.add_argument("--block", type=int, default=32)
    args = ap.parse_args()
    chars = args.pretrain_tokens * 4.2
    print("downloading sources...", flush=True)
    fw = safe(lambda: fineweb(0.55 * chars), "fineweb-edu")
    ts = safe(lambda: tinystories(0.25 * chars), "tinystories")
    en = safe(lambda: enron(0.10 * chars), "enron")
    ae = safe(aeslc, "aeslc")
    cn = safe(lambda: cnn(20000), "cnn_dailymail")
    wk = safe(lambda: wiki(60000), "wikipedia")
    ev = safe(lambda: chats_from("HuggingFaceTB/everyday-conversations-llama3.1-2k", 10000, "train_sft"), "everyday")
    st = safe(lambda: chats_from("HuggingFaceTB/smol-smoltalk", 40000), "smol-smoltalk")
    xl = safe(lambda: xlam(15000), "xlam")

    # dialogue as plain text also goes into pretraining (10%)
    dialog_txt = ["\n".join(f"{m['role']}: {m['content']}" for m in c) for c in st[:20000] + ev]
    docs = fw + ts + en + [e["body"] for e in ae] + dialog_txt
    random.Random(0).shuffle(docs)
    nval = max(200, len(docs) // 200)
    val_docs, docs = docs[:nval], docs[nval:]

    # task conversations
    random.Random(1).shuffle(ae)
    ae_val, ae_tr = ae[:300], ae[300:]
    syn = Synth(seed=0, emails=ae_tr, articles=cn[:-300], wiki=wk[:-500], chats=ev + st[20000:])
    syn_val = Synth(seed=1, emails=ae_val, articles=cn[-300:], wiki=wk[-500:], chats=ev[:200])
    convs = [syn.sample() for _ in range(args.synth)]
    convs += [[syn.system(syn.now())] + c for c in st[:20000]]
    for sysmsg, c in xl:
        convs.append([{"role": "system", "content": syn.system(syn.now())["content"].split(" Tools:")[0] + " " + sysmsg}] + c)
    random.Random(2).shuffle(convs)
    val_convs = [syn_val.sample() for _ in range(1500)]
    write_dataset(args.out, docs, val_docs, convs, val_convs, args.vocab, args.R, args.block)
    print("done")


if __name__ == "__main__":
    main()
