"""Build a copy-focused chat/tool task set: every example draws fresh random names, places, times,
numbers, app names and note texts, so the model must copy details from the user's message
instead of recalling them from a small fixed list. Adds short small-talk turns that use no tool.

Writes a new data dir that links the original pretraining stream and mixes in old task rows:

  python make_copy_data.py --src data --out data_copy --n 30000 --old-rows 5000 --vq runs/vq/best.pt

It encodes the new rows with the frozen VQ and reuses the original codes for the old rows.
"""
import argparse
import datetime as dt
import json
import os
import random
from pathlib import Path

import numpy as np

import ncm.synth_tools as st
from ncm.bpe import BPE
from ncm.chat_format import format_conv
from ncm.data import pack_rows, read_u16
from prepare_data import aeslc, safe, wiki

SYL = ["ka", "ri", "to", "man", "el", "sa", "vi", "no", "ber", "lin", "dra", "quo", "zen", "pa", "li", "ta",
       "mor", "gan", "shi", "ro", "ve", "ul", "ix", "an", "dor", "fen", "ola", "bri", "chu", "kel", "nas", "tur"]
FIRST = ["Ada", "Hiro", "Fatima", "Mateo", "Ingrid", "Kwame", "Yuki", "Nadia", "Oscar", "Leila", "Bjorn", "Zara",
         "Diego", "Anya", "Felix", "Imani", "Kai", "Lucia", "Musa", "Noor", "Pavel", "Rosa", "Tariq", "Vera",
         "Elif", "Jonas", "Keiko", "Amara", "Dev", "Greta", "Hamza", "Isla", "Jun", "Kofi", "Mila", "Nikhil"]
PLACES = ["Lisbon", "Nairobi", "Oslo", "Lima", "Hanoi", "Cairo", "Prague", "Austin", "Denver", "Kyoto", "Accra",
          "Dublin", "Vienna", "Manila", "Perth", "Quito", "Riga", "Seattle", "Tunis", "Zurich", "Pune", "Jaipur",
          "Boston", "Osaka", "Porto", "Bogota", "Havana", "Helsinki", "Montreal", "Marrakech", "Chennai", "Kochi"]
VERBS = ["call", "buy", "email", "pay", "book", "renew", "cancel", "pick up", "drop off", "order", "return",
         "check", "clean", "fix", "print", "send", "text", "visit", "water", "charge", "pack", "wash", "sign"]
OBJECTS = ["the tickets", "the car insurance", "a cake", "the documents", "new shoes", "the lease", "the plants",
           "my glasses", "the package", "batteries", "the form", "the laptop", "flowers", "the keys", "a gift",
           "the bike", "the passport photos", "coffee beans", "the rent", "the phone bill", "the parcel", "a taxi"]
TOPICS = ["the {e} moving to {d}", "dinner on {d}", "the {w} invoice", "my trip to {p}", "the {e} on {d}",
          "the {w} project", "a quick call {d}", "the budget for {p}", "{n}'s birthday party", "the new {w} plan"]
GREET = [
    ("hi", ["Hi! How can I help?", "Hello! What can I do for you?", "Hey! What do you need?"]),
    ("hello", ["Hello! How can I help today?", "Hi there! What can I do for you?"]),
    ("hey", ["Hey! What's up?", "Hi! How can I help?"]),
    ("good morning", ["Good morning! What can I do for you today?", "Morning! How can I help?"]),
    ("good afternoon", ["Good afternoon! How can I help?"]),
    ("good evening", ["Good evening! What can I do for you?"]),
    ("good night", ["Good night! Sleep well.", "Night! Talk tomorrow."]),
    ("how are you", ["I'm doing well, thanks! How can I help?", "All good here! What do you need?"]),
    ("how are you today", ["I'm great, thanks for asking! What can I do for you?"]),
    ("what's up", ["Not much! Ready to help. What do you need?"]),
    ("who are you", ["I'm your assistant. I can set alarms and reminders, check your email and calendar, look things up and more."]),
    ("what can you do", ["I can set alarms, timers and reminders, manage your calendar, read and summarize your email, look things up, check the weather and take notes."]),
    ("thanks", ["You're welcome!", "Anytime!", "Happy to help."]),
    ("thank you", ["You're welcome!", "No problem!", "Glad I could help."]),
    ("thanks, that's all", ["You're welcome! Have a good one.", "Anytime!"]),
    ("bye", ["Bye! Have a great day.", "See you!"]),
    ("ok", ["Okay! Anything else?", "Great. Let me know if you need anything."]),
    ("nice", ["Glad to hear it!", "Great!"]),
    ("lol", ["Ha! Anything else I can do?"]),
    ("you're awesome", ["Thank you! Happy to help."]),
    ("i'm bored", ["Want me to find something fun to read, or set a reminder to take a walk?"]),
    ("i'm tired", ["Sounds like you need some rest. Want me to set an alarm for a short nap?"]),
]


def rand_word(r, n=None):
    return "".join(r.choice(SYL) for _ in range(n or r.randint(2, 3)))


def rand_name(r):
    x = r.random()
    if x < 0.45:
        return r.choice(FIRST)
    if x < 0.8:
        return rand_word(r).capitalize()
    return f"{r.choice(FIRST)} {rand_word(r).capitalize()}"


def rand_place(r):
    x = r.random()
    if x < 0.5:
        return r.choice(PLACES + st.CITIES)
    w = rand_word(r).capitalize()
    return r.choice([w, f"New {w}", f"Port {w}", f"{w}ville", f"San {w}"])


def rand_task(r):
    x = r.random()
    if x < 0.25:
        return r.choice(st.TASKS)
    if x < 0.5:
        return f"{r.choice(['call', 'text', 'email', 'meet'])} {rand_name(r)}"
    if x < 0.75:
        return f"{r.choice(VERBS)} {r.choice(OBJECTS)}"
    return f"{r.choice(VERBS)} the {rand_word(r)} {r.choice(['order', 'report', 'tickets', 'forms', 'package'])}"


def rand_note(r):
    n = r.randrange(10, 99999)
    return r.choice([f"{rand_word(r)} code is {n}", f"locker number {n}", f"wifi password is {rand_word(r)}{n % 100}",
                     f"parking spot {r.choice('ABCDEFG')}{n % 50}", f"{rand_name(r)}'s number is 555-{n % 10000:04d}",
                     f"gift idea for {rand_name(r)}: {r.choice(OBJECTS)}", f"order {n} ships on {r.choice(st.DAYS)}"])


def sample_when2(now, rng):
    """Like st.sample_when, but 40% of the time any hour and minute, so times must be copied."""
    if rng.random() < 0.6:
        return _orig_when(now, rng)
    if rng.random() < 0.25:
        n = rng.randint(1, 120)
        if rng.random() < 0.5:
            h = rng.randint(1, 12)
            return rng.choice([f"in {h} hours" if h > 1 else "in 1 hour", f"in {h}h"]), now + dt.timedelta(hours=h)
        return rng.choice([f"in {n} minutes", f"in {n} mins"]), now + dt.timedelta(minutes=n)
    t = now.replace(hour=rng.randrange(24), minute=rng.randrange(60), second=0, microsecond=0)
    if t <= now:
        t += dt.timedelta(days=1)
    phrase = st.fmt_clock(t, rng) if rng.random() < 0.7 else t.strftime("%H:%M")
    if rng.random() < 0.3:
        phrase = rng.choice(["tomorrow at ", "at "]) + phrase
        if phrase.startswith("tomorrow") and t.date() == now.date():
            t += dt.timedelta(days=1)
    return phrase, t


_orig_when = st.sample_when


class CopySynth(st.Synth):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.gens = [
            (self.alarm, 12), (self.reminder, 12), (self.timer, 8), (self.cal_add, 5), (self.cal_add2, 7),
            (self.cal_list, 2), (self.list_emails, 4), (self.draft_free, 5), (self.lookup, 16), (self.web_search, 6),
            (self.weather, 8), (self.notes_free, 6), (self.calc, 12), (self.time_q, 4), (self.open_app, 6),
            (self.call_sms, 6), (self.multi, 3), (self.greet, 14),
        ]

    def sample(self):
        r = self.rng
        # fresh pools for this conversation; the base generators read these module globals
        st.NAMES[:] = [rand_name(r) for _ in range(20)] + ["mom", "dad", "my boss", "Grandma"]
        st.CITIES[:] = [rand_place(r) for _ in range(10)]
        st.TASKS[:] = [rand_task(r) for _ in range(10)]
        st.EVENTS[:] = [r.choice([f"meeting with {rand_name(r)}", f"{rand_word(r)} review", f"lunch with {rand_name(r)}",
                                  f"{r.choice(['yoga', 'piano', 'swim', 'spanish'])} class", f"call with {rand_name(r)}"])
                        for _ in range(8)]
        st.APPS[:] = [r.choice(_APPS) if r.random() < 0.6 else rand_word(r).capitalize() for _ in range(8)]
        st.sample_when = sample_when2
        return super().sample()

    def greet(self):
        r, now = self.rng, self.now()
        conv = [self.system(now)]
        for _ in range(r.choice([1, 1, 1, 2])):
            u, replies = r.choice(GREET)
            if r.random() < 0.3:
                u = u.capitalize() + r.choice(["!", "?", "", "."])
            conv += [{"role": "user", "content": u}, self.say(r.choice(replies))]
        if r.random() < 0.3:  # then a real request, so greetings don't cancel tool use
            conv += self.alarm()[1:] if r.random() < 0.5 else self.reminder()[1:]
        return conv

    def cal_add2(self):
        """Calendar events phrased like reminders ("add X on friday at 3pm"), so the two are told apart."""
        r, now = self.rng, self.now()
        ev = r.choice([r.choice(st.EVENTS), f"{r.choice(['dentist', 'doctor', 'vet', 'bank', 'salon'])} appointment",
                       f"{r.choice(['coffee', 'lunch', 'dinner', 'drinks'])} with {rand_name(r)}"])
        phrase, t = st.sample_when(now, r)
        u = r.choice(["add {e} {p}", "add {e} to my calendar {p}", "book {e} {p}", "{e} {p}, put it in my calendar",
                      "calendar: {e} {p}", "schedule {e} {p}", "i have {e} {p}, add it"]).format(e=ev, p=phrase)
        return [self.system(now), self.user(u), self.call("calendar_add", {"title": ev, "time": st.clean_time_arg(phrase)}),
                self.result({"ok": True, "event": ev, "time": st.fmt_dt(t)}),
                self.say(r.choice(["{a} Added {e} on {t}.", "{e} is on your calendar for {t}."]).format(
                    a=r.choice(st.ACKS), e=ev, t=st.fmt_dt(t)))]

    def notes_free(self):
        r, now = self.rng, self.now()
        txt = rand_note(r)
        return [self.system(now), self.user(r.choice(["note: {t}", "save a note {t}", "remember that {t}", "write down {t}",
                                                      "make a note that {t}"]).format(t=txt)),
                self.call("notes_add", {"text": txt}), self.result({"ok": True}),
                self.say(r.choice(["Saved.", "Noted!", "Got it, note saved.", f"Saved: {txt}."]))]

    def draft_free(self):
        r, now = self.rng, self.now()
        to = rand_name(r)
        topic = r.choice(TOPICS).format(e=r.choice(["meeting", "review", "call", "demo", "interview"]), d=r.choice(st.DAYS).capitalize(),
                                        w=rand_word(r).capitalize(), p=rand_place(r), n=rand_name(r))
        text = f"Hi {to},\n\nI wanted to let you know about {topic}.\n\nBest regards"
        args = {"to": to, "subject": topic[0].upper() + topic[1:], "body": text}
        return [self.system(now), self.user(f"{r.choice(['write', 'draft'])} an email to {to} about {topic}"),
                self.call("draft_email", args), self.result({"ok": True, "draft_id": f"d{r.randrange(10**5)}"}),
                self.say(f"I saved a draft to {to}:\n\n{text}")]


_APPS = list(st.APPS)


def link(src, dst):
    if not dst.exists():
        os.symlink(src.resolve(), dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data")
    ap.add_argument("--out", default="data_copy")
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--old-rows", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--vq", default="runs/vq/best.pt")
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((src / "meta.json").read_text())
    R, L = meta["R"], meta["block"]
    bpe = BPE.load(src / "bpe.json")
    print("downloading lookup and email sources...", flush=True)
    wk = safe(lambda: wiki(60000), "wikipedia")
    ae = safe(aeslc, "aeslc")
    syn = CopySynth(seed=args.seed, emails=ae, wiki=wk)
    convs = [syn.sample() for _ in range(args.n)]
    fm = [f for f in (format_conv(bpe, c, L, max_len=R, max_segment=R - 4 * L) for c in convs) if f is not None]
    rows, bm = pack_rows(fm, R, bpe.pad, L)
    print(f"  new: {len(fm)} convs -> {len(rows)} rows ({bm.mean():.0%} trained blocks)", flush=True)

    # mix in a random subset of the original task rows (tokens, masks and their codes) to avoid forgetting
    old_tok = read_u16(src / "task.bin", (-1, R))
    old_bm = np.fromfile(src / "task_bmask.bin", dtype=np.uint8).reshape(len(old_tok), R // L)
    pick = np.sort(np.random.RandomState(args.seed).choice(len(old_tok), min(args.old_rows, len(old_tok)), replace=False))
    np.concatenate([rows, old_tok[pick]]).tofile(out / "task.bin")
    np.concatenate([bm, old_bm[pick]]).tofile(out / "task_bmask.bin")
    np.save(out / "old_pick.npy", pick)
    (out / "new_rows.txt").write_text(str(len(rows)))
    for name in ("bpe.json", "meta.json", "pretrain.bin", "pretrain_val.bin", "pretrain.codes", "pretrain_val.codes",
                 "task_val.bin", "task_val_bmask.bin", "task_val.codes"):
        link(src / name, out / name)
    print(f"  total task rows: {len(rows) + len(pick)}", flush=True)

    from encode_codes import encode_array
    from ncm.utils import load_vq, pick_device
    dev = pick_device("auto")
    vq, vcfg = load_vq(args.vq, dev)
    n = vcfg.n_codes
    new_codes = encode_array(vq, rows.reshape(-1, L), dev).reshape(len(rows), R // L, n)
    old_codes = read_u16(src / "task.codes", (len(old_tok), R // L, n))[pick]
    np.concatenate([new_codes, old_codes]).astype(np.uint16).tofile(out / "task.codes")
    print("  codes written", flush=True)
    for c in convs[:6]:
        print("  sample:", " | ".join(f"{m['role']}: {m['content'][:70]!r}" for m in c[1:]))
    print("done")


if __name__ == "__main__":
    main()
