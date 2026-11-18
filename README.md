# tiny-ncm

A small everyday assistant, **trained from scratch on one MacBook**, built on **Next-Concept Modeling (NCM)**
([Kumbong, Khare et al., 2026](https://arxiv.org/abs/2610.14162)): text is generated coarse-to-fine as levels of
discrete codes, with all codes of a level predicted in parallel. In this implementation, text is written 32 tokens at
a time, each block over 5 levels (2 → 4 → 8 → 16 → 32 codes), so a block takes 6 passes instead of 32.

It sets alarms, reminders and timers, manages the calendar, reads Gmail (read-only), looks things up on the web
and Wikipedia, checks the weather, takes notes, does everyday math, and chats a little. It is **not** a coding or
reasoning model.

- **36M parameters** (21M generator + 15M multiscale VQ tokenizer), context 4096
- trained with PyTorch on Apple-silicon MPS: about 12 hours in total (tokenizer 1.3 h, generator 10.5 h)
- the model picks the action; small **app-side checks** make sure the details (times, names, places, numbers)
  match what you typed. See [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) for every step and why.

```
you: set an alarm for seven thirty and remind me to call mom at 9
  → set_alarm({"time": "7:30"})
  → set_reminder({"text": "call mom", "time": "9"})
tiny-ncm: Your alarm is set for Thu 19 Nov 07:30.
I'll remind you to call mom on Wed 18 Nov 21:00.

you: how do I unclog a sink?
  → web_search({"query": "how do I unclog a sink"})
tiny-ncm: Here's what to do (from wikihow.com):
1. Plunge the sink with a regular bell plunger.
2. Boil a pot of water and pour it down the drain.
...
```

## Based on the NCM paper

This model is an independent, from-scratch implementation of **Next-Concept Modeling (NCM)** from

> **Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation**
> Hermann Kumbong, Ishan S. Khare, Iddah Mlauzi, Andrew Woen, Charles Ding, E. Kelly Buchanan, Simran Arora, Azalia Mirhoseini, Christopher Ré
> Stanford University and Together AI, 2026. [arXiv:2610.14162](https://arxiv.org/abs/2610.14162) · official code: [HazyResearch/ncm](https://github.com/HazyResearch/ncm)

All credit for the method goes to the authors. This project is not affiliated with them, and no code was taken from the official repository.

**From the paper:** a multiscale residual VQ-VAE turns text into coarse-to-fine levels of discrete codes; a transformer generates
one level at a time, predicting all codes of a level in parallel; nested dropout in the tokenizer; conditioning (code) corruption
and per-level loss weighting when training the generator.

**Changed or added here:** text is split into 32-token blocks that are generated one after another with a KV cache (each block
coarse-to-fine over 5 levels), so a chat can grow to 4096 tokens; level positions attend to a 128-token window of context to fit
Apple-silicon speed; Muon optimizer; a small size (36M) trained in about 12 hours on one MacBook; chat and tool-use training,
plus the app-side checks in `assistant/`.

If you use this, please cite the paper:

```bibtex
@article{kumbong2026ncm,
  title   = {Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation},
  author  = {Kumbong, Hermann and Khare, Ishan S. and Mlauzi, Iddah and Woen, Andrew and Ding, Charles and
             Buchanan, E. Kelly and Arora, Simran and Mirhoseini, Azalia and R{\'e}, Christopher},
  journal = {arXiv preprint arXiv:2610.14162},
  year    = {2026}
}
```

## Quick start (macOS)

```bash
git clone https://github.com/amar683/tiny-ncm && cd tiny-ncm
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
hf download amartomar683/tiny-ncm tiny-ncm.pt --local-dir weights   # 146 MB (or download it from the page below)
.venv/bin/python chat.py --dry-run          # tools that change things only say what they'd do
.venv/bin/python chat.py --show-levels      # watch each block form coarse -> fine
.venv/bin/python chat.py                    # real Reminders, Calendar, notifications, web
.venv/bin/python chat.py --connect-gmail    # one-time read-only Gmail sign-in (steps in assistant/gmail.py)
```

Alarms and timers ring only while the chat is open. Sending email opens a draft in Mail; you press Send.
Calls and texts are phone-only.

## Weights

`weights/tiny-ncm.pt` (146 MB, fp32) is on Hugging Face: https://huggingface.co/amartomar683/tiny-ncm (CC BY-NC 4.0)

## How well it works

All tests are in this repo and run offline (`eval_*.py`). Scores are for the model **plus** the app's checks.

| Test | Score |
|---|---|
| 500 new requests (300 everyday, 30 small talk, 70 math, 100 hard) | 490 / 500 |
| 60 hard requests written before the fixes (typos, "seven thirty", two requests in one, …) | 57 / 60 |
| 153 everyday requests: right tool and every detail exact | 153 / 153 |
| Math (60 + 30 new) | 59 / 60, 30 / 30 |
| Model alone, every detail exact (no app checks) | ~45% |
| Coding | ~5% (not trained for it) |

**Limits.** English is rough, so most replies after a tool are built from the tool's result rather than
written by the model. It can't do multi-step math or write code. Long rambling messages sometimes fail. Web answers are picked
from the top pages by word matching, so they're only as good as those pages. There is no safety or refusal training;
actions that send or call always ask you first.

## Platforms

| | Status |
|---|---|
| macOS terminal chat (`chat.py`) | works |
| Desktop app (window UI), Windows/Linux tools | not built yet |
| ONNX export (one ~35 MB file for every platform) | not built yet |
| Android / iOS app (Flutter) | not built yet |

## Repo map

| | |
|---|---|
| `ncm/` | tokenizer (BPE + multiscale VQ), generator, sampling, chat format, optimizer |
| `assistant/` | chat loop, Mac tools, Gmail, web answers, the detail checks, text clean-up |
| `prepare_data.py`, `train_vq.py`, `encode_codes.py`, `train_gen.py`, `make_copy_data.py` | the training pipeline |
| `eval_*.py`, `test_grounding.py`, `smoke_test.py` | tests |
| `tools/registry.json` | every tool's name and arguments, shared by all runtimes |

## License

Code: MIT (see [LICENSE](LICENSE)). Weights: CC BY-NC 4.0, research and non-commercial use, because of the training-data terms; see [MODEL_CARD.md](MODEL_CARD.md).
