# Model card: tiny-ncm

## Summary
- **Type:** NCM (coarse-to-fine, block-wise) language model for tool use, trained from scratch.
- **Method:** Next-Concept Modeling, Kumbong, Khare et al., 2026 ([arXiv:2610.14162](https://arxiv.org/abs/2610.14162)); see below.
- **Size:** 36.4M parameters: generator 20.9M (d=384, 8 layers, 3 query heads / 1 KV head) and multiscale residual VQ
  tokenizer 15.5M (5 levels, 1024-entry codebooks). Byte-level BPE vocabulary of 16,384. Context 4096.
- **Checkpoint:** `runs/task6` (released as `tiny-ncm.pt`, fp32, 146 MB). Code: https://github.com/amar683/tiny-ncm (MIT).
- **Hardware:** one MacBook (Apple M5, 24 GB), PyTorch MPS, bf16. About 12 hours of training in total (VQ tokenizer 1.3 h, generator 10.5 h).

## Method: the NCM paper

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

## Intended use
A personal assistant that turns everyday requests into tool calls (alarms, reminders, timers, calendar, read-only Gmail,
web and Wikipedia lookups, weather, notes, calculator) and makes small talk. Use it together with the app in
`assistant/`, which checks every tool call against the user's words. **Not** for coding, reasoning, medical, legal or
financial advice, or anything where a wrong answer matters.

## Training data
| Data | License | Used for |
|---|---|---|
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (sample-10BT) | ODC-By 1.0 | general English |
| [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | CDLA-Sharing 1.0 | simple English |
| [Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) (20231101.en) | CC BY-SA 3.0 / GFDL | text, lookup answers |
| [CNN/DailyMail](https://huggingface.co/datasets/abisee/cnn_dailymail) | Apache-2.0 | summaries |
| [Enron emails](https://huggingface.co/datasets/corbt/enron-emails), [AESLC](https://huggingface.co/datasets/Yale-LILY/aeslc) | not stated (public record) | email style, subject lines |
| [everyday-conversations](https://huggingface.co/datasets/HuggingFaceTB/everyday-conversations-llama3.1-2k), [smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) | Apache-2.0 | small talk (non-coding only) |
| [xLAM function-calling 60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | CC-BY-4.0, "research purposes only" | reading tool schemas |
| Synthetic tool conversations (`ncm/synth_tools.py`, `make_copy_data.py`) | made for this project | tool calls |

The weights were **not** trained on web pages collected by `harvest_web.py` (that experiment, `runs/web1`, was not adopted).

**Weights license: CC BY-NC 4.0.** Because of the xLAM terms and the data with no stated license, the weights are released for
**research and non-commercial use**, with credit to the datasets above. xLAM asks users to cite APIGen:
Liu et al., *APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets*, 2024.

## Training
1. BPE tokenizer (16k) on the mix.
2. VQ tokenizer: 97.6% exact text reconstruction, 95% exact tool JSON.
3. Generator pretraining at context 1024 (Muon + AdamW, warmup-stable-decay schedule).
4. Chat/tool phases at 4096: synthetic tool data, then "copy-focused" data with fresh random names, times and places in every
   example, then 2h45m of mostly English, then 2 × 1h of copy-focused data.
Details and every run: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md).

## Evaluation
| Test | Model + app checks | Model alone |
|---|---|---|
| 500 new requests | 490 / 500 | n/a |
| 60 hard requests written before the fixes | 57 / 60 | n/a |
| 153 everyday requests, right tool | 153 / 153 | n/a |
| 153 everyday requests, every detail exact | 153 / 153 | 64 / 153 |
| Math (90) | 89 / 90 | n/a |
| Coding (40) | 5 / 40 | n/a |

## Limitations and risks
- **Copying.** The model often garbles names, numbers and times ("London" → "Singapore"); the app fixes them from the
  user's words. Without the app, expect about half of the tool calls to have a wrong detail.
- **English.** Free-form replies are rough; the app replaces replies that don't match the tool result.
- **Reasoning.** No multi-step math, no code.
- **Web answers** are picked from the top search results by word matching; they can be wrong or mix methods.
- **No safety training.** The model has no refusal or content filtering. The app asks before any action that sends
  messages or calls someone, and never guesses those tools from a garbled name.
- **Privacy.** Everything runs locally. Web questions go to DuckDuckGo and the pages' sites; Gmail access is read-only.
