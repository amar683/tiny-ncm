"""Clean up what you typed before the model and the checks see it:
  - typos in command words      "remnd me", "set an alrm", "wheather"  -> remind, alarm, weather
  - numbers written as words    "seven thirty", "twenty five"           -> 7:30, 25
  - spoken times                "quarter past 7", "half past six", "0630", "half an hour", "the day after tomorrow"
Only words that look like typos of a small list of command words are changed; real words, names and anything else stay as typed.
"""
import datetime as dt
import difflib
import re
from pathlib import Path

KEYWORDS = """remind reminder alarm alarms timer timers weather calendar email emails mail inbox unread search google open launch
start check note notes tomorrow tonight today minutes minute hours hour seconds second times plus minus divided multiplied
percent schedule wake what what's whats where when how many much please morning evening night weather rain temperature
monday tuesday wednesday thursday friday saturday sunday week weekend next spotify zoom safari write draft""".split()
FIXES = {"wether": "weather", "calender": "calendar", "tims": "times", "wat": "what", "tmrw": "tomorrow", "tmr": "tomorrow",
         "tmrrw": "tomorrow", "pls": "please", "plz": "please", "u": "you", "ur": "your", "msg": "message", "mins": "minutes",
         "hrs": "hours", "secs": "seconds", "min": "minute", "hr": "hour", "sec": "second", "whats": "what's"}


def _dictionary():
    for p in ("/usr/share/dict/words", "/usr/dict/words"):
        try:
            return {w.strip().lower() for w in Path(p).read_text().splitlines()}
        except OSError:
            pass
    return set()


WORDS = _dictionary() | set(KEYWORDS)


def fix_typos(text):
    def fix(m):
        w = m.group()
        if re.search(r"'s$", w, re.I) and len(w) > 3:          # "tomorrow's", "Kenji's": fix the word, keep the 's
            return fix_word(w[:-2]) + w[-2:]
        return fix_word(w)

    def fix_word(w):
        lw = w.lower()
        if w.isupper() and len(w) <= 3:                       # HR, NY, OK: short forms, leave them
            return w
        if lw in FIXES:
            return _case(FIXES[lw], w)
        if len(lw) < 4 or lw in WORDS or (w[0].isupper() and not w.isupper()):
            return w                               # real word or a name: leave it
        swapped = [k for k in KEYWORDS if len(k) == len(lw) and sorted(k) == sorted(lw)]     # "opne" -> "open"
        if swapped:
            return _case(swapped[0], w)
        close = difflib.get_close_matches(lw, KEYWORDS, n=1, cutoff=0.8)
        return _case(close[0], w) if close else w
    return re.sub(r"[A-Za-z][A-Za-z']*", fix, text)


def _case(new, old):
    return new.upper() if old.isupper() and len(old) > 1 else new


ONES = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
                                    "fifteen sixteen seventeen eighteen nineteen".split())}
TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}


def words_to_numbers(text):
    """'twenty five' -> 25, 'a hundred' -> 100, 'seven thirty' -> 7:30 (only before am/pm or after at/for/alarm)."""
    num = r"(?:%s)" % "|".join(sorted(list(ONES) + list(TENS) + ["hundred", "thousand"], key=len, reverse=True))

    def val(phrase):
        total, cur = 0, 0
        for w in re.findall(r"[a-z]+", phrase.lower()):
            if w in ONES:
                cur += ONES[w]
            elif w in TENS:
                cur += TENS[w]
            elif w == "hundred":
                cur = max(cur, 1) * 100
            elif w == "thousand":
                total, cur = total + max(cur, 1) * 1000, 0
            elif w == "a":
                cur = cur or 1
        return total + cur
    # spoken clock: "seven thirty", "eight fifteen", "five thirty am"
    hours = "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve"
    clock = rf"\b(at|for|by|until|till|to)\s+({hours})\s+((?:twenty|thirty|forty|fifty)(?:[\s-](?:one|two|three|four|five|six|seven|eight|nine))?|fifteen|ten|five|o'?clock)\b"

    def clock_sub(m):
        h = val(m.group(2))
        mm = 0 if m.group(3).lower().startswith("o") else val(m.group(3))
        return f"{m.group(1)} {h}:{mm:02d}" if mm < 60 else m.group()
    text = re.sub(clock, clock_sub, text, flags=re.I)
    text = re.sub(rf"\b(?:a\s+)?(?:{num})(?:[\s-]+(?:and\s+)?(?:{num}))*\b",
                  lambda m: m.group() if m.group().lower() in ("a", "one", "a one") else str(val(m.group())), text, flags=re.I)
    # a lone "one" stays a word ("one ticket", "set one for 6:15"), except in math or before a time unit
    text = re.sub(r"\bone\b(?=\s+(?:plus|minus|times|divided|hour|minute|second|day|week)s?\b)", "1", text, flags=re.I)
    return text


def spoken_times(text, now=None):
    t = text
    t = re.sub(r"\b(?:a )?quarter past (\d{1,2})\b", lambda m: f"{m.group(1)}:15", t, flags=re.I)
    t = re.sub(r"\bhalf past (\d{1,2})\b", lambda m: f"{m.group(1)}:30", t, flags=re.I)
    t = re.sub(r"\b(?:a )?quarter to (\d{1,2})\b", lambda m: f"{(int(m.group(1)) - 2) % 12 + 1}:45", t, flags=re.I)
    t = re.sub(r"\b(?:an|1) hour and a half\b|\b1\.5 hours\b", "90 minutes", t, flags=re.I)
    t = re.sub(r"\bhalf an hour\b|\ba half hour\b", "30 minutes", t, flags=re.I)
    t = re.sub(r"\ba quarter of an hour\b|\ba quarter hour\b", "15 minutes", t, flags=re.I)
    t = re.sub(r"\b(?:at|for|by)\s+([01]\d|2[0-3])([0-5]\d)\b", lambda m: f"{m.group(0).split()[0]} {int(m.group(1))}:{m.group(2)}", t, flags=re.I)
    if now is not None:
        day = (now + dt.timedelta(days=2)).strftime("%A").lower()
        t = re.sub(r"\b(?:the )?day after tomorrow\b", day, t, flags=re.I)
    return t


def normalize(text, now=None):
    return spoken_times(words_to_numbers(fix_typos(text)), now)
