"""App-side check of the model's tool calls: fix the arguments using the user's own words.

The model is small and writes all codes of a level in parallel, so it picks the right tool
but sometimes garbles details ("London" -> "Singapore", "5pm" -> "5pmpm"). Before running a tool, the app:

  1. parses the call, repairing truncated JSON (keeps the tool name and every complete "key":"value"),
  2. checks the tool and its arguments against tools/registry.json (drops unknown args, converts types),
  3. checks each argument against what the user typed:
       grounded (appears in the message)   -> keep
       fixed by removing repeats ("5pmpm") -> use that
       else a rule for this tool/arg finds the span in the message ("remind me to X at T") -> use that
       else the closest run of words in the message (fuzzy match) -> use that
       else: optional -> drop, required -> keep the model's value.

No retraining, no extra parameters. `check_call` returns the fixed call plus a list of notes saying what changed.
"""
import difflib
import json
import re

DAYS = {"today", "tonight", "tomorrow", "monday", "tuesday", "wednesday", "thursday", "friday",
        "saturday", "sunday", "weekend", "morning", "afternoon", "evening", "night", "noon", "midnight"}
SOFT = {"next", "this"}                     # only part of a time when next to a real time word
LINK = {"at", "on", "in", "by"}             # allowed inside a time, trimmed from the edges
UNIT = r"(?:seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?)"
CLOCK = re.compile(r"^\d{1,2}(?::\d{2})?(?:am|pm|a\.m\.|p\.m\.)?$|^\d{1,2}(?:am|pm)$")
FREE_ARGS = set()          # args the model may write freely (none yet: its free writing is too rough at this size)
WORD_OPS = [(r"\bmultiplied by\b|\btimes\b|\bx\b|×", "*"), (r"\bdivided by\b|\bover\b|÷", "/"),
            (r"\bplus\b", "+"), (r"\bminus\b", "-"),
            (r"\bpercent of\b|% of\b", "/100*"), (r"\bsquared\b", "**2")]


# ---------- helpers ----------
def norm(s):
    return " ".join(re.sub(r"[^\w:%*/+.-]+", " ", str(s).lower()).split())


def words(text):
    """(word, start, end) with trailing punctuation removed."""
    out = []
    for m in re.finditer(r"\S+", text):
        w = m.group().rstrip("?,.!;")
        if w:
            out.append((w, m.start(), m.start() + len(w)))
    return out


def grounded(val, text):
    v = norm(re.sub(r"\b(?:from|to|is|subject|label|has|in):", "", str(val)))   # gmail syntax: "from:Sarah" -> "sarah"
    return len(v) >= 2 and f" {v} " in f" {norm(text)} "


def dedupe(val):
    """'at at 9' -> 'at 9', '5pmpm' -> '5pm', 'tomorrowtomorrow' -> 'tomorrow'."""
    s = re.sub(r"\b(\w+)(?:\s+\1\b)+", r"\1", str(val), flags=re.I)
    return re.sub(r"(\w{2,}?)\1+\b", r"\1", s)


def tidy(s):
    s = re.sub(r"\s+", " ", s).strip(" ,.?!")
    s = re.sub(r"\s+(?:please|pls|thanks|thank you)$", "", s, flags=re.I)
    return re.sub(r"^(?:please|pls)\s+", "", s, flags=re.I).strip(" ,.?!")


def drop_article(s):
    return re.sub(r"^(?:the|a|an)\s+", "", s, flags=re.I)


# ---------- time & duration spans ----------
def _is_time_word(ws, i):
    w = ws[i][0].lower()
    nxt = ws[i + 1][0].lower() if i + 1 < len(ws) else ""
    if w in DAYS or CLOCK.match(w) or w in {"am", "pm", "o'clock"}:
        return True
    if re.fullmatch(UNIT, w) and i > 0 and re.fullmatch(r"\d+|an?|one|half", ws[i - 1][0].lower()):
        return True
    if re.fullmatch(r"an?|one|half", w) and re.fullmatch(UNIT, nxt):
        return True
    return w in LINK | SOFT


def time_span(text):
    """Longest run of time words, e.g. 'tomorrow at 9', '7 tomorrow', 'in 2 hours'. Returns (value, start, end)."""
    ws, best, i = words(text), None, 0
    while i < len(ws):
        if not _is_time_word(ws, i):
            i += 1
            continue
        j = i
        while j + 1 < len(ws) and _is_time_word(ws, j + 1):
            j += 1
        run = ws[i:j + 1]
        # trim links/soft words from the right, links from the left (keep "in" before a number)
        while run and run[-1][0].lower() in LINK | SOFT:
            run = run[:-1]
        full_start = run[0][1] if run else None
        while run and run[0][0].lower() in LINK:
            keep_in = run[0][0].lower() == "in" and len(run) > 1 and re.match(r"\d|an?$", run[1][0].lower())
            if keep_in:
                break
            run = run[1:]
        real = [w for w, _, _ in run if w.lower() in DAYS or re.search(r"\d", w) or re.fullmatch(UNIT, w.lower())]
        if run and real and (best is None or len(run) > best[3]):
            best = (text[run[0][1]:run[-1][2]], full_start, run[-1][2], len(run))
        i = j + 1
    return best[:3] if best else None


def duration_span(text):
    m = re.search(rf"(?:\d+(?:\.\d+)?|an?|one|half an?)\s*-?\s*{UNIT}(?:\s*(?:and\s*)?\d+\s*{UNIT})*", text, re.I)
    return m.group() if m else None


def without_time(text):
    t = time_span(text)
    return (text[:t[1]] + text[t[2]:]) if t else text


def day_word(text):
    m = re.search(r"\b(?:today|tonight|tomorrow|this week|next week|this weekend|(?:next |this )?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b", text, re.I)
    return m.group() if m else None


# ---------- per-tool rules: find the value in the user's message ----------
def _after(pattern, text, flags=re.I):
    m = re.search(pattern, text, flags)
    return tidy(m.group(1)) if m and tidy(m.group(1)) else None


def _name_after(pattern, text):
    m = re.search(pattern + r"\s+([A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)?)", text)
    if m:
        return m.group(1)
    m = re.search(pattern + r"\s+([a-z][\w'-]*)\s*(?:about|saying|that|\?|$)", text, re.I)   # lower-case name
    return m.group(1).capitalize() if m else None


def _place(text):
    stop = r"(?=\s+(?:today|tomorrow|tonight|this|next|on|at|now|right now)\b|\s*[?.!]|$)"
    m = re.search(r"\b(?:in|for|at)\s+((?:[A-Z][\w'.-]*)(?:\s+[A-Z][\w'.-]*)*)", text)
    if m and m.group(1).lower() not in DAYS:
        return m.group(1)
    m = re.search(r"\bin\s+([a-z][\w' .-]*?)" + stop, text, re.I)
    if m and m.group(1).lower() not in DAYS and not re.match(r"\d|an?\b", m.group(1)):
        return m.group(1).title()
    return None


N = r"(-?\d+(?:\.\d+)?)"
UNITS = {  # unit -> (kind, size in a base unit)
    "second": ("t", 1), "sec": ("t", 1), "minute": ("t", 60), "min": ("t", 60), "hour": ("t", 3600), "hr": ("t", 3600),
    "day": ("t", 86400), "night": ("t", 86400), "week": ("t", 604800), "year": ("t", 31536000),
    "millimeter": ("l", 0.001), "mm": ("l", 0.001), "centimeter": ("l", 0.01), "cm": ("l", 0.01), "meter": ("l", 1), "metre": ("l", 1),
    "m": ("l", 1), "kilometer": ("l", 1000), "km": ("l", 1000), "inch": ("l", 0.0254), "foot": ("l", 0.3048), "feet": ("l", 0.3048),
    "mile": ("l", 1609.344), "gram": ("w", 1), "g": ("w", 1), "kilogram": ("w", 1000), "kg": ("w", 1000), "pound": ("w", 453.59237),
    "lb": ("w", 453.59237), "ounce": ("w", 28.349523), "oz": ("w", 28.349523),
    "milliliter": ("v", 0.001), "ml": ("v", 0.001), "liter": ("v", 1), "litre": ("v", 1), "l": ("v", 1),
}
U = r"([a-z]+)\b"


def _unit(w):
    w = w.lower()
    return UNITS.get(w) or UNITS.get(w[:-1] if w.endswith("s") else "") or (UNITS.get(w[:-2]) if w.endswith("es") else None)


def _convert(n, u1, u2):
    """n of u1 in u2 -> expression, or None if the units don't match."""
    a, b = _unit(u1), _unit(u2)
    if not a or not b or a[0] != b[0]:
        return None
    f = a[1] / b[1]
    return f"{n}*{f:g}" if f >= 1 else f"{n}/{1 / f:g}"


def _phrase_math(t):
    """Everyday math phrasings -> expression. t is lower-case."""
    t = t.replace(",", "")
    rules = [
        (rf"\badd {N} (?:and|to) {N}", lambda a, b: f"{a}+{b}"),
        (rf"\bsubtract {N} from {N}", lambda a, b: f"{b}-{a}"),
        (rf"\btake {N} (?:away )?from {N}", lambda a, b: f"{b}-{a}"),
        (rf"\bmultiply {N} (?:and|by) {N}", lambda a, b: f"{a}*{b}"),
        (rf"\bdivide {N} by {N}", lambda a, b: f"{a}/{b}"),
        (rf"\bsquare root of {N}|\bsqrt\s*\(?{N}", lambda a, b=None: f"sqrt({a or b})"),
        (rf"\bhalf of {N}", lambda a: f"{a}/2"), (rf"\b(?:a )?quarter of {N}", lambda a: f"{a}/4"),
        (rf"\b(?:double|twice) {N}", lambda a: f"{a}*2"), (rf"\btriple {N}", lambda a: f"{a}*3"),
        (rf"{N} squared", lambda a: f"{a}**2"), (rf"{N} cubed", lambda a: f"{a}**3"),
        (rf"{N} to the power of {N}", lambda a, b: f"{a}**{b}"),
        (rf"{N} (?:percent|%) off {N}", lambda p, a: f"{a}*(1-{p}/100)"),
        (rf"{N}\s*/\s*{N} as a (?:decimal|number)", lambda a, b: f"{a}/{b}"),
    ]
    for pat, fn in rules:
        m = re.search(pat, t)
        if m:
            return fn(*[g for g in m.groups() if g is not None])
    m = re.search(r"\b(?:average|mean) of ([\d.\s,and]+)", t)
    if m:
        xs = re.findall(r"\d+(?:\.\d+)?", m.group(1))
        if len(xs) >= 2:
            return f"({'+'.join(xs)})/{len(xs)}"
    m = re.search(rf"how many {U} (?:are )?(?:there )?in (?:{N}|an?|one) {U}", t)
    if m and len(re.findall(r"\d+(?:\.\d+)?", t)) <= 1:      # more numbers: it's a word problem, not a conversion
        return _convert(m.group(2) or 1, m.group(3), m.group(1))
    m = re.search(rf"(?:convert )?{N} ?{U} (?:to|in|into) {U}", t)
    if m:
        return _convert(m.group(1), m.group(2), m.group(3))
    return None


def _story_math(t):
    """Two-number word problems with one clear action word: 'had 50 and spent 18' -> 50-18."""
    t = t.lower().replace(",", "")
    m = re.search(rf"{N} dollars? with {N} (?:percent|%) off|{N} (?:percent|%) off .*?{N}", t)
    if m:
        a, p = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), m.group(3))
        return f"{a}*(1-{p}/100)"
    xs = re.findall(r"\d+(?:\.\d+)?", t)
    per = re.search(r"\b(?:a|per|each|every) (day|night|week|hour|month)\b", t)
    if len(xs) == 1 and per and re.search(r"\bin (?:a|one) (week|day|year)\b", t):      # "3 km every day ... in a week"
        w = re.search(r"\bin (?:a|one) (week|day|year)\b", t).group(1)
        f = UNITS[w][1] / UNITS[per.group(1)][1]
        return f"{xs[0]}*{f:g}" if f >= 1 else None
    if len(xs) != 2:
        return None
    a, b = xs
    if re.search(r"\b(?:split|share|shared|equally|groups?|teams?|per (?:group|team|person|kid|student)|each get)\b|how many \w+ (?:per|in each|for each) \w+", t):
        return f"{a}/{b}"
    if re.search(r"\b(?:spent|spend|spends|eat|eats|ate|leave|leaves|gave|give away|gives away|lost|lose|loses|sold|sells|used|uses|broke|fewer|less)\b", t):
        return f"{a}-{b}"
    if re.search(r"\b(?:more|gets?|got|buys?|bought|finds?|found|plus|another|older|added|adds|joins?|joined|gains?|taller|longer)\b", t):
        return f"{a}+{b}"
    if re.search(r"\b(?:each|every|per|a (?:day|night|week|hour)|how much do|how many \w+ in|times|costs?)\b", t):
        return f"{a}*{b}"
    return None


def _expression(text):
    t = text.lower()
    found = _phrase_math(t)
    if found:
        return found
    s = t
    for pat, op in WORD_OPS:
        s = re.sub(pat, f" {op} ", s)
    m = re.search(r"\(?\s*-?\d[\d.,]*(?:\s*(?:\*\*|/100\*|[-+*/^])\s*\(?\s*-?\d[\d.,]*\s*\)?)+", s)
    if m:
        e = re.sub(r"\s+", "", m.group()).replace(",", "").replace("^", "**")
        if e.count("(") != e.count(")"):
            e = e.strip("()") if e.count("(") + e.count(")") == 1 else e
        return e
    return _story_math(t)


def _email_query(text):
    parts = []
    if re.search(r"\bunread\b", text, re.I):
        parts.append("is:unread")
    if re.search(r"\bstarred\b", text, re.I):
        parts.append("is:starred")
    who = _name_after(r"\bfrom", text)
    m = re.search(r"\b(?:did|has|whether|if)\s+([A-Z][\w'-]*)\s+(?:e-?mail|mail|writ|wrote|messag|texted|replied|sent)", text)
    who = who or (m.group(1) if m else None)
    if who:
        parts.append(f"from:{who}")
    about = _after(r"\babout\s+(.+)$", text)
    if about:
        parts.append(about)
    return " ".join(parts) if parts else None


def _email_n(text):
    m = re.search(r"\b(?:last|latest|recent|top|first|newest)\s+(\d+)\b|\b(\d+)\s+(?:\w+\s+)?(?:emails?|mails?|messages?)\b", text, re.I)
    return int(m.group(1) or m.group(2)) if m else None


def _reminder_text(text):
    t = without_time(text)
    return _after(r"\bremind me\s+(?:to\s+|about\s+|that\s+)?(.+)$", t) or _after(r"\breminder\s+(?:to|for|about)\s+(.+)$", t)


def _calendar_title(text):
    t = without_time(text)
    t = re.sub(r"\s+(?:to|on|in)\s+(?:my\s+)?calendar\b", "", t, flags=re.I)
    s = _after(r"\b(?:add|schedule|put|book|create|set up)\s+(?:an?\s+|the\s+)?(?:event\s+|meeting\s+called\s+)?(.+)$", t)
    return s


def _note_text(text):
    """'I parked on level 4, please make a note of that' -> 'I parked on level 4'; 'I keep forgetting the code, it's 7781' -> all of it."""
    m = re.match(r"(.+?)[,.;]?\s*(?:(?:please|can you|could you)\s+)?make a note of (?:that|this|it)\b", text, re.I)
    if m:
        return tidy(re.sub(r"^(?:i just|just)\s+", "", m.group(1), flags=re.I))
    if re.search(r"\bkeep (?:forgetting|losing)\b", text, re.I):
        return tidy(text)
    return None


def _timer_label(text):
    return _after(r"\btimer\s+(?:for|called|named)\s+(?!\d|an?\s+(?:hour|minute))(.+)$", text) or \
        _after(r"\b(?:timer|minutes?|mins?|hours?|seconds?|secs?)\s+(?:for|called)\s+(?!\d)([a-z][\w ]*)$", text)


RULES = {
    ("set_alarm", "time"): lambda t: (time_span(t) or (None,))[0],
    ("set_alarm", "label"): lambda t: _after(r"\balarm\s+(?:for|called|named|labeled)\s+(?!\d)(\D+?)(?:\s+at\b.*)?$", t),
    ("set_reminder", "text"): _reminder_text,
    ("set_reminder", "time"): lambda t: (time_span(t) or (None,))[0],
    ("set_timer", "duration"): duration_span,
    ("set_timer", "label"): _timer_label,
    ("calendar_add", "title"): _calendar_title,
    ("calendar_add", "time"): lambda t: (time_span(t) or (None,))[0],
    ("calendar_list", "day"): lambda t: day_word(t) or "today",
    ("list_emails", "query"): _email_query,
    ("list_emails", "n"): _email_n,
    ("draft_email", "to"): lambda t: _name_after(r"\b(?:to|email|mail|write)", t),
    ("draft_email", "subject"): lambda t: _after(r"\babout\s+(.+)$", t) or _after(r"\bsaying\s+(?:that\s+)?(.+)$", t),
    ("send_email", "to"): lambda t: _name_after(r"\b(?:to|email|mail)", t),
    ("send_email", "subject"): lambda t: _after(r"\babout\s+(.+)$", t) or _after(r"\bsaying\s+(?:that\s+)?(.+)$", t),
    ("lookup", "topic"): lambda t: drop_article(_after(r"\b(?:who|what|where)\s*(?:'s|\s+is|\s+was|\s+are|\s+were)\s+(.+)$", t)
                                                or _after(r"\b(?:tell me about|look up|lookup|define|explain|info on|information about|what do you know about)\s+(.+)$", t) or "") or None,
    ("web_search", "query"): lambda t: (tidy(t) if re.match(r"(?:how|why|what should|what(?:'s| is) the best|tips?|ways? to|is it|can i|should i)\b", t, re.I) else None)
        or _after(r"\bfind\s+(?:me\s+)?(.+?)(?:\s+online)?$", t) or _after(r"\b(?:search|look)\s+(?:the\s+web\s+|online\s+|google\s+)?(?:for|up)\s+(.+?)(?:\s+online)?$", t)
                                       or _after(r"\b(?:google|search)\s+(.+)$", t),
    ("weather", "location"): _place,
    ("weather", "when"): day_word,
    ("notes_add", "text"): lambda t: _note_text(t) or _after(r"\b(?:note that|note down that|note down|note:|remember that|write down that|write down|make a note(?:\s+that)?|save a note(?:\s+that)?|take a note(?:\s+that)?|note)\s+(.+)$", t),
    ("notes_search", "query"): lambda t: _after(r"\b(?:notes?\s+(?:about|for|on)|note\s+about|did i note about|search (?:my )?notes for)\s+(.+)$", t),
    ("calculator", "expression"): _expression,
    ("get_time", "place"): _place,
    ("open_app", "app"): lambda t: "Music" if re.search(r"\b(?:listen to|play)\b.*\b(?:music|songs?|playlist)\b", t, re.I) else
        (lambda s: s.title() if s else None)(_after(r"\b(?:open|launch|start|run)\s+(?:the\s+|my\s+)?(.+?)(?:\s+app)?$", t)),
    ("call", "contact"): lambda t: _name_after(r"\b(?:call|phone|ring|dial)", t),
    ("sms", "to"): lambda t: _name_after(r"\b(?:text|message|sms|tell)", t),
    ("sms", "message"): lambda t: _after(r"\b(?:saying|that says|that)\s+(.+)$", t),
}


def _expr_ok(val, text):
    want = _expression(text)
    return want is not None and re.sub(r"\s+", "", str(val)) == want


def fuzzy(val, text, cutoff=0.6):
    """Closest run of words in the message to the model's value (fixes 'Pha Lovelace' -> 'Ada Lovelace')."""
    ws, v = words(text), norm(val)
    k = max(1, len(v.split()))
    best, score = None, cutoff
    for n in range(max(1, k - 1), k + 2):
        for i in range(len(ws) - n + 1):
            cand = text[ws[i][1]:ws[i + n - 1][2]]
            r = difflib.SequenceMatcher(None, v, norm(cand)).ratio()
            if r > score:
                best, score = cand, r
    return best


# ---------- parsing ----------
def parse_call(s):
    """Model output after <|tool|> -> dict. Repairs cut-off JSON by keeping the complete parts."""
    s = s.replace("<|tool|>", "").replace("<|end|>", "").replace("<|pad|>", "").strip()
    try:
        j = json.loads(s)
        if isinstance(j, dict) and "name" in j:
            return {"name": j["name"], "args": j.get("args") or {}}, True
    except json.JSONDecodeError:
        pass
    m = re.search(r'"name"\s*:\s*"(\w+)"', s)
    if not m:
        return None, False
    args = {}
    for k, v in re.findall(r'"(\w+)"\s*:\s*("(?:[^"\\]|\\.)*"(?=\s*[,}])|-?\d+(?=\s*[,}]))', s[m.end():]):
        try:
            args[k] = json.loads(v)
        except json.JSONDecodeError:
            args[k] = v.strip('"')          # bad escape inside a garbled value: keep the raw text
    return {"name": m.group(1), "args": args}, False


# ---------- strong phrases: when your words clearly name a tool ----------
NOT_CMD = r"(?!.*\b(?:set|remind|alarm|timer|open|search|weather|note|add|email|call|text|what|who|how|when|where|schedule|\d))"
ROUTES = [   # checked in order; only clear, common phrasings
    ("chat", r"^" + NOT_CMD + r"(?:hi|hello|hey|yo|bye|goodbye|good ?bye|see you|see ya|good (?:morning|afternoon|evening|night)|morning|"
             r"thanks|thank you|thx|cheers|nice to meet you|you're (?:great|awesome|really helpful|the best)|have a (?:nice|good|great) (?:day|night|evening)|"
             r"sounds good|ok(?:ay)?|cool|nice(?: one)?|great|perfect|awesome|talk (?:to you )?later|hmm+|good job|well done)\b[\w\s,'!.]{0,20}[!.?]*$"),
    ("chat", r"^(?:how are you|how's it going|how are things|how's your day|what's up|sup|who are you|what(?:'s| is) your name|what can you do|tell me a joke)\b[\w\s,'!.]{0,25}[!.?]*$"),
    ("set_reminder", r"\bremind me\b|\b(?:don't|do not) let me forget\b|\bmake sure i (?:don't|do not) forget\b"),
    ("notes_search", r"\b(?:what did i note|search (?:my )?notes|find (?:my )?note)\b"),
    ("notes_add", r"^(?:please\s+)?(?:note that|note down|take a note|save a note|write down|remember that|note:)|\bmake a note\b|\bkeep (?:forgetting|losing)\b"),
    ("set_timer", r"\btimers?\b|\bcount ?down\b|\btime (?:that|it|this)\b|\b(?:needs?|takes?|for) (?:about )?\d+ (?:minutes?|seconds?)\b(?=.*\b(?:oven|boil|cook|bake|egg|rice|pasta|cake|tea)|\s*$)"),
    ("set_alarm", r"\bwake me\b|\balarms?\b"),
    ("get_time", r"\b(?:what(?:'s| is) the time|what time is it|current time|time) in\b"),
    ("list_emails", r"\b(?:did|has|whether|if) [A-Z]\w* (?:e-?mail(?:ed)?|mail(?:ed)?|writ(?:e|ten)|wrote|messag(?:e|ed)|texted|replied|sent)(?: to)? me\b|"
                    r"\b(?:emails?|mails?) from\b|\binbox\b|\bunread\b|\bcheck (?:my )?(?:e-?mails?|mail)\b|\bmy (?:e-?mails?|mail)\b|"
                    r"\b(?:anyone|anybody|someone) (?:e-?mail(?:ed)?|writ(?:e|ten)|wrote|messag(?:e|ed)|sent)\b"),
    ("draft_email", r"^(?:please\s+)?(?:write|draft|send|compose)?\s*(?:an? )?(?:email|mail)\s+(?:to\s+)?[A-Z]\w*\s+(?:about|saying)\b"),
    ("calendar_list", r"\b(?:what(?:'s| is) on my calendar|my (?:day|schedule|agenda)|what do i have|any (?:meetings|events)|am i (?:free|busy)|do i have anything)\b"),
    ("calendar_add", r"^(?:please\s+)?(?:add|schedule|put|create an event)\s+(?!\d)(?!me\b).*\b(?:on|at|calendar|tomorrow|today|tonight|next)\b"),
    ("weather", r"\bweather\b|\b(?:will|is) it (?:going to )?(?:be )?(?:rain|snow|sunny|cold|hot|warm|windy)|\bhow (?:hot|cold|warm) is it\b|\bumbrella\b|\bwarm enough\b|\b(?:wear|need|bring|take) (?:a |an |my )?(?:jacket|coat|umbrella|sunscreen|shorts|sweater|scarf)\b"),
    ("open_app", r"^(?:please\s+)?(?:open|launch|start)\s+(?!a\b|an\b|\d)|\b(?:listen to|play)\b.*\b(?:music|songs?|playlist)\b"),
    ("web_search", r"^(?:search|google)\b|\bnear me\b|^(?:please\s+)?find\b|\b(?:look up|find) .+ online\b|"
                   r"^(?:how (?:do|can|should|would|to)|how to|why (?:do|does|is|are|did|can)|what should i|what(?:'s| is) the best way|"
                   r"tips? (?:for|on)|ways? to|is it (?:safe|ok|okay|bad|good) to|can i\b|should i\b)"),
    ("lookup", r"^(?:who (?:is|was)|tell me about)\b"),
]
CHAT_REPLIES = {"how": "I'm doing well, thanks! What can I do for you?", "what's up": "Not much! What can I do for you?",
                "sup": "Not much! What can I do for you?",
                "who are you": "I'm tiny-ncm, a small assistant that runs on your computer.",
                "what's your name": "I'm tiny-ncm, a small assistant that runs on your computer.",
                "what is your name": "I'm tiny-ncm, a small assistant that runs on your computer.",
                "what can you do": "I can set alarms, reminders and timers, manage your calendar, check your email, look things up, "
                                   "check the weather, take notes and do math.",
                "tell me a joke": "Why did the scarecrow win an award? Because he was outstanding in his field.",
                "bye": "Bye! Talk soon.", "goodbye": "Goodbye!", "see": "See you!", "talk": "Talk soon!", "thank": "You're welcome!",
                "thx": "You're welcome!", "cheers": "Cheers!", "nice to": "Nice to meet you too!", "you're": "Thank you! Happy to help.",
                "have a": "You too!", "good night": "Good night!", "good morning": "Good morning! What can I do for you?",
                "morning": "Good morning! What can I do for you?", "good evening": "Good evening! How can I help?",
                "good afternoon": "Good afternoon! How can I help?", "hmm": "Take your time. What can I do for you?",
                "sounds": "Great!", "ok": "Great!", "cool": "Great!", "nice": "Thanks!", "great": "Great!", "perfect": "Great!",
                "awesome": "Great!", "good job": "Thanks!", "well done": "Thanks!"}
CAN_DO = ("I can't do that one yet. I can set alarms, reminders and timers, manage your calendar, check your email, "
          "look things up, search the web, check the weather, take notes and do math.")
ASK_FOR = {("set_alarm", "time"): "What time should I set the alarm for?", ("set_reminder", "text"): "What should I remind you about?",
           ("set_timer", "duration"): "How long should the timer be?", ("calendar_add", "time"): "When is it?",
           ("calendar_add", "title"): "What should I call the event?", ("web_search", "query"): "What should I search for?",
           ("lookup", "topic"): "What should I look up?", ("open_app", "app"): "Which app should I open?",
           ("notes_add", "text"): "What should the note say?", ("weather", "location"): "For which place?",
           ("get_time", "place"): "For which place?", ("calculator", "expression"): "What should I calculate?"}


def question_like(text):
    return bool(re.search(r"\?\s*$|^(?:what|how|why|where|when|which|who|is|are|can|should|do|does)\b", text.strip(), re.I))


def route(user_text):
    """The tool your words clearly ask for, "chat" for small talk, or None if it isn't clear."""
    t = user_text.strip()
    if not re.search(r"[A-Za-z0-9]", t):
        return "chat"
    for tool, pat in ROUTES:
        if re.search(pat, t, 0 if tool in ("list_emails", "draft_email") else re.I):
            return tool
    asks = re.search(r"^(?:what|what's|whats|how|calculate|compute|convert|work out|add|subtract|multiply|divide|take)\b|\bwhat(?:'s| is)\b|"
                     r"\bhow (?:many|much|far|old|long)\b|^[\d(.][\d\s+\-*/^().%x×÷]*\??$", t, re.I)
    if asks and _expression(t):
        return "calculator"
    return None


def check_chat(answer, user_text):
    """Small talk with no tool: no made-up facts (numbers), and don't greet someone who is leaving."""
    t = user_text.lower()
    leaving = re.match(r"(?:bye|goodbye|good ?bye|see you|see ya|good night|thanks|thank you|thx|cheers)", t)
    if re.search(r"\d", answer) or (leaving and re.match(r"(?:hello|hi|hey|good (?:morning|afternoon|evening))\b", answer, re.I)):
        return chat_reply(user_text), f"reply: {answer[:50]!r} didn't fit small talk"
    return answer, None


def plain_reply_ok(answer, user_text):
    """A reply with no tool behind it: no repeated words, no numbers or names you didn't say, not rambling."""
    if "<|" in answer or re.search(r"\b(\w+)\s+\1\b", answer, re.I) or re.search(r"(\S)\1\1", answer) or len(answer) > 300:
        return False
    return answer_ok(answer, [user_text])


def chat_reply(user_text):
    t = user_text.lower().strip()
    for k, v in sorted(CHAT_REPLIES.items(), key=lambda kv: -len(kv[0])):
        if t.startswith(k):
            return v
    return "Hi! What can I do for you?"


# ---------- main entry ----------
def check_call(model_out, user_text, registry):
    """Returns (call, notes). call is {"name", "args", "confirm"} or None if the tool is unknown."""
    notes = []
    call, valid = parse_call(model_out) if isinstance(model_out, str) else (model_out, True)
    if call is None:
        return None, ["could not read a tool name"]
    if not valid:
        notes.append("repaired cut-off JSON")
    tools = {t["name"]: t for t in registry}
    name = call["name"]
    if name not in tools:
        safe = [t for t in tools if not tools[t].get("confirm")]     # never guess a tool that sends or calls
        close = difflib.get_close_matches(name, safe, n=1, cutoff=0.75)
        if not close:
            return None, notes + [f"unknown tool {name!r}"]
        notes.append(f"tool {name!r} -> {close[0]!r}")
        name = close[0]
    spec = tools[name]["args"]
    args_in = {k: v for k, v in call["args"].items() if k in spec}
    for k in set(call["args"]) - set(spec):
        notes.append(f"dropped unknown arg {k!r}")

    out, missing, unsure = {}, [], []
    for arg, typ in spec.items():
        optional, base = typ.endswith("?"), typ.rstrip("?")
        val = args_in.get(arg)
        rule = RULES.get((name, arg))
        new = _ground_one(name, arg, base, val, user_text, rule)
        if new is None:
            if not optional and val not in (None, ""):
                new = val                       # required and nothing better: keep the model's value
                notes.append(f"{arg}: kept {val!r} (not found in message)")
                unsure.append(arg)
            elif not optional:
                notes.append(f"{arg}: missing")
                missing.append(arg)
                continue
            else:
                if val not in (None, ""):
                    notes.append(f"{arg}: dropped {val!r} (not in message)")
                continue
        if base == "int":
            try:
                new = int(new)
            except (TypeError, ValueError):
                continue
        if new != val and val is not None:
            notes.append(f"{arg}: {val!r} -> {new!r}")
        elif val is None:
            notes.append(f"{arg}: filled {new!r}")
        out[arg] = new
    if name in ("draft_email", "send_email") and "to" in out:
        body = f"Hi {out['to']},\n\nI wanted to get in touch about {out.get('subject', 'this')}.\n\nBest regards"
        if out.get("body") != body:
            notes.append("body: plain text from your words (the model's own writing is too rough)")
        out["body"] = body
        missing = [m for m in missing if m != "body"]
    return {"name": name, "args": out, "confirm": bool(tools[name].get("confirm")), "missing": missing,
            "unsure": unsure}, notes


def _ground_one(tool, arg, base, val, text, rule):
    if (tool, arg) in FREE_ARGS:
        return val
    if base == "int":
        if str(val).lstrip("-").isdigit() and re.search(rf"\b{int(val)}\b", text):
            return val
        return rule(text) if rule else None
    if val not in (None, ""):
        if (tool, arg) == ("calculator", "expression"):
            if _expr_ok(val, text):
                return val
        else:
            cand = val if grounded(val, text) else dedupe(val) if grounded(dedupe(val), text) else None   # "5pmpm" -> "5pm"
            if cand is not None:
                full = rule(text) if rule and base == "str" else None
                if full and norm(full) != norm(cand):
                    if arg in ("location", "place", "app"):          # "Perth right" when the rule finds "Perth"
                        return full
                    if norm(cand) in norm(full) and len(norm(full)) > len(norm(cand)):
                        return full                                   # "standup" when you said "standup review"
                v = str(cand).strip()
                return re.sub(r"^(?:at|on|for)\s+", "", v, flags=re.I) if arg in ("time", "when", "day") else v
    found = rule(text) if rule else None
    if found:
        return found
    if val not in (None, "") and (tool, arg) != ("calculator", "expression"):
        f = fuzzy(val, text)
        if f:
            return f
    if val == "" and (tool, arg) == ("list_emails", "query"):
        return ""
    return None


# ---------- answers: check the model's reply against the tool result ----------
STOP = set("""a an the and or but if of to in on at for from by with about as is are was were be been it its it's this that
these those you your i i'm i'll me my we our they their he she his her them there here what which who when where how all any
some no not yes done okay ok sure got set all one two more most also just very so than then into over up out""".split())
COPY_TOOLS = {"lookup", "web_search", "read_email", "list_emails", "notes_search", "calendar_list"}
REPLY_WORDS = set("""done all set okay sure got it no problem i added a reminder to saved i'll remind you on timer for let know in
started is your calendar opening open alarm you'll get will ring wake up at the note noted that comes it's added event set: reminder:
here are latest emails sent wrote about short top result says""".split())   # words the model was trained to reply with
ACTION_TOOLS = {"set_alarm", "set_reminder", "set_timer", "calendar_add", "notes_add", "open_app", "get_time", "calculator"}
KEY_ARG = {"set_timer": "duration", "open_app": "app", "get_time": "place", "weather": "location", "calendar_add": "title",
           "set_reminder": "text", "draft_email": "to", "send_email": "to"}   # the reply must name this


def _nums(s):
    return re.findall(r"\d+(?::\d\d)?", str(s))


def answer_ok(answer, sources, copy=False, vocab=False):
    """Numbers and names in the reply must come from the sources; for copy tools most words must too."""
    src = norm(" ".join(map(str, sources)))
    src_words, src_nums = set(src.split()), set(_nums(src))
    if any(n not in src_nums for n in _nums(answer)):
        return False
    for sent in re.split(r"[.!?:\n]\s*", answer):
        for w in re.findall(r"[A-Za-z][\w']*", sent)[1:]:      # capitalised words after the first one
            if w[0].isupper() and w.lower() not in src_words and w.lower() not in STOP:
                return False
    if re.search(r"(\S)\1\1|°°", answer):                  # "Adaa" is fine, "°°" / "!!!" is garbled
        return False
    if re.search(r"\b(\w+)\s+\1\b", answer, re.I):         # "3 3 hours", "project project review"
        return False
    if vocab:      # action tools: every word must come from your words, the result, or the trained reply phrases
        for w in re.findall(r"[a-z][a-z']+", answer.lower()):
            if len(w) > 2 and w not in src_words and w not in REPLY_WORDS and w not in STOP and w.rstrip("s") not in src_words:
                return False
    if copy:
        content = [w for w in re.findall(r"[a-z][\w']+", answer.lower()) if w not in STOP and len(w) > 2]
        if content and sum(w in src_words for w in content) / len(content) < 0.85:
            return False
    return True


def _sentences(text, n=2):
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
    return " ".join(parts[:n])


def template_reply(name, args, res):
    """Plain reply built straight from the tool result (used when the model's reply doesn't match it)."""
    if not res.get("ok", True):
        return f"Sorry, I couldn't do that: {res.get('error', 'something went wrong')}."
    t = res.get("time")
    if name == "set_alarm":
        return f"Alarm set for {t}."
    if name == "set_reminder":
        return f"I'll remind you to {res['reminder']}" + (f" on {t}." if t else ".")
    if name == "set_timer":
        return f"Timer set for {args.get('duration')}; it ends at {res['ends']}."
    if name == "calendar_add":
        return f"Added {res['event']} on {t}."
    if name == "calendar_list":
        evs = res.get("events", [])
        return "\n".join([f"You have {len(evs)} event{'s' * (len(evs) != 1)} {args.get('day', '')}:"] +
                         [f"- {e['time']}: {e['title']}" for e in evs]) if evs else f"Nothing scheduled {args.get('day', '')}."
    if name == "list_emails":
        em = res.get("emails", [])
        return "\n".join([f"{len(em)} email{'s' * (len(em) != 1)}:"] + [f"- {e['from']}: {e['subject']}" for e in em]) if em else "No emails found."
    if name == "read_email":
        return f"From {res['from']}, \"{res['subject']}\": {_sentences(res['body'], 3)}"
    if name in ("draft_email", "send_email"):
        return f"I opened an email to {args.get('to')} about {args.get('subject')} in Mail. Edit it and press Send."
    if name == "lookup":
        return _sentences(res["extract"])
    if name == "web_search":
        src = res.get("source", {})
        site = re.sub(r"^www\.", "", re.sub(r"^https?://([^/]+).*", r"\1", src.get("url", "")))
        if res.get("steps"):
            return f"Here's what to do (from {site}):\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(res["steps"], 1))
        if res.get("answer"):
            return f"{res['answer']} (from {site})"
        r = res.get("results", [])
        return f"Top result: {r[0]['title']}. {_sentences(r[0]['snippet'], 1)}" if r else "No results."
    if name == "weather":
        return (f"{res['location']}, {res['when']}: {res['summary']}, {res['temp_c']}°C "
                f"(high {res['high_c']}°C, low {res['low_c']}°C).")
    if name == "notes_add":
        return "Saved."
    if name == "notes_search":
        n = res.get("notes", [])
        return f"Your note says: {n[-1]['text']}." if n else "I couldn't find a note about that."
    if name == "calculator":
        return f"{args.get('expression')} = {res['result']:g}" if isinstance(res.get("result"), (int, float)) else str(res)
    if name == "get_time":
        return f"It's {res['time']} in {res['place']}."
    if name == "open_app":
        return f"Opening {args.get('app')}."
    return "Done."


def check_answer(answer, name, args, res, user_text):
    """Returns (reply, note). Keeps the model's reply if it matches the result, else a plain one from the result."""
    if not res.get("ok", True):
        return template_reply(name, args, res), "reply: error message from the tool"
    if not answer.strip() or "<|" in answer:
        return template_reply(name, args, res), "reply: empty or not a reply, used the result"
    if name in ("draft_email", "send_email"):
        return template_reply(name, args, res), "reply: plain confirmation for the draft"
    if name == "web_search" and (res.get("steps") or res.get("answer")):
        return template_reply(name, args, res), "reply: the answer picked from the web pages"
    if name == "calculator" and isinstance(res.get("result"), (int, float)):
        if not any(abs(float(x) - res["result"]) < 1e-6 for x in re.findall(r"-?\d+(?:\.\d+)?", answer.replace(",", ""))):
            return template_reply(name, args, res), f"reply: {answer[:50]!r} doesn't contain the result"
    sources = [user_text, json.dumps(res, ensure_ascii=False), json.dumps(args, ensure_ascii=False)]
    key = args.get(KEY_ARG.get(name, ""), "")
    names_key = not key or norm(key) in norm(answer)
    if names_key and answer_ok(answer, sources, copy=name in COPY_TOOLS, vocab=name in ACTION_TOOLS):
        return answer, None
    return template_reply(name, args, res), f"reply: {answer[:60]!r} didn't match the result, used the result directly"
