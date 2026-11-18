"""Tool back ends for macOS. Each tool takes the checked args and returns a small dict,
shaped like the results the model saw in training (ncm/synth_tools.py), so it can answer from them.

Times stay in the user's words ("tomorrow at 9") and are turned into a datetime here with dateparser.
Network tools use only public, key-free APIs: Wikipedia, Open-Meteo, DuckDuckGo.
dry_run=True: tools that change something (reminders, alarms, calendar, notes, apps, email) only describe what they would do.
"""
import ast
import html
import datetime as dt
import json
import operator
import re
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

import dateparser

HOME = Path.home() / ".tiny-ncm"
UA = {"User-Agent": "tiny-ncm/0.1 (local assistant)"}


def fmt_dt(t):
    return t.strftime("%a %d %b %H:%M")


def get_json(url, timeout=8, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code != 429 or i == tries - 1:          # rate limited: wait 1, 2, 4 s and retry
                raise
            time.sleep(2 ** i)


def osa(script):
    p = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=20)
    if p.returncode:
        raise RuntimeError(p.stderr.strip() or "osascript failed")
    return p.stdout.strip()


def as_str(s):
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


def as_date(t):
    """AppleScript date built from parts (works in every locale)."""
    return (f"set d to current date\nset day of d to 1\nset year of d to {t.year}\nset month of d to {t.month}\n"
            f"set day of d to {t.day}\nset time of d to {t.hour * 3600 + t.minute * 60}\n")


def notify(title, text, sound="Glass"):
    try:
        osa(f"display notification {as_str(text)} with title {as_str(title)} sound name {as_str(sound)}")
    except Exception:
        print(f"\n🔔 {title}: {text}")


class MacTools:
    def __init__(self, now=None, dry_run=False, gmail=None):
        self.now = now or dt.datetime.now
        self.dry = dry_run
        self.gmail = gmail
        self.timers = []
        HOME.mkdir(exist_ok=True)
        self.notes_file = HOME / "notes.json"

    # ---- time words -> datetime ----
    def when(self, phrase, future=True, alarm=False):
        """Your words -> datetime. Own rules for the common cases; dateparser for the rest."""
        now, p = self.now(), (phrase or "").lower().strip()
        m = re.search(r"\bin\s+(\d+|an?|half an?)\s*(min|hour|hr|h\b|day|sec)", p)
        if m:
            return (now + dt.timedelta(seconds=self.seconds(p.replace(" h", " hour")) or 3600)).replace(microsecond=0)
        day = now.date()
        if "tomorrow" in p:
            day += dt.timedelta(days=1)
        wd = re.search(r"\b(mon|tues|wednes|thurs|fri|satur|sun)day\b", p)
        if wd:
            i = ["mon", "tues", "wednes", "thurs", "fri", "satur", "sun"].index(wd.group(1))
            ahead = (i - now.weekday()) % 7 or 7
            day = now.date() + dt.timedelta(days=ahead)
        clock = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?(?!\s*(?:min|hour|sec|day))", p)
        h = mi = None
        if clock:
            h, mi, ap = int(clock.group(1)), int(clock.group(2) or 0), (clock.group(3) or "")[:1]
            if h > 23 or mi > 59:
                h = None
            elif ap == "p" and h < 12:
                h += 12
            elif ap == "a" and h == 12:
                h = 0
            elif not ap and 1 <= h <= 6 and not alarm and "morning" not in p:
                h += 12                         # "remind me at 5" usually means 5pm; an alarm at 6:30 means morning
        elif "noon" in p:
            h, mi = 12, 0
        elif "midnight" in p:
            h, mi = 0, 0
        elif "tonight" in p or "evening" in p:
            h, mi = 20, 0
        elif "morning" in p:
            h, mi = 8, 0
        elif "afternoon" in p:
            h, mi = 15, 0
        elif day != now.date():
            h, mi = 9, 0                        # just a day ("on thursday", "tomorrow"): 9 in the morning
        if h is not None:
            if "tonight" in p and h < 12:
                h += 12
            t = dt.datetime.combine(day, dt.time(h, mi))
            if future and t <= now:
                bare = clock and not clock.group(3) and h < 12 and not alarm
                t = t + dt.timedelta(hours=12) if bare and t + dt.timedelta(hours=12) > now else t + dt.timedelta(days=1)
            return t
        t = dateparser.parse(p, settings={"RELATIVE_BASE": now, "PREFER_DATES_FROM": "future"})
        if t is None:
            return None
        t = t.replace(second=0, microsecond=0)
        return t + dt.timedelta(days=1) if future and t <= now else t

    def seconds(self, duration):
        total = 0
        for n, unit in re.findall(r"(\d+(?:\.\d+)?|an?|half an?)\s*(sec|min|hour|hr|day)", duration or "", re.I):
            n = 0.5 if n.lower().startswith("half") else 1 if n.lower() in ("a", "an") else float(n)
            total += n * {"sec": 1, "min": 60, "hour": 3600, "hr": 3600, "day": 86400}[unit.lower()]
        return int(total)

    # ---- tools ----
    def set_alarm(self, time, label=None):
        t = self.when(time, alarm=True)
        if t is None:
            return {"ok": False, "error": f"could not understand the time '{time}'"}
        if not self.dry:
            self._schedule((t - self.now()).total_seconds(), "Alarm", label or f"It's {t:%H:%M}")
        return {"ok": True, "time": fmt_dt(t)}

    def set_timer(self, duration, label=None):
        s = self.seconds(duration)
        if s <= 0:
            return {"ok": False, "error": f"could not understand '{duration}'"}
        end = self.now() + dt.timedelta(seconds=s)
        if not self.dry:
            self._schedule(s, "Timer", f"{label or 'Timer'} is done")
        return {"ok": True, "ends": end.strftime("%H:%M:%S")}

    def _schedule(self, secs, title, text):
        tm = threading.Timer(max(0, secs), notify, args=(title, text, "Radar"))
        tm.daemon = True
        tm.start()
        self.timers.append(tm)

    def set_reminder(self, text, time=None):
        t = self.when(time) if time else None
        if time and t is None:
            return {"ok": False, "error": f"could not understand the time '{time}'"}
        if not self.dry:
            props = f"{{name:{as_str(text)}" + (", remind me date:d}" if t else "}")
            osa((as_date(t) if t else "") + f'tell application "Reminders" to make new reminder with properties {props}')
        return {"ok": True, "reminder": text, "time": fmt_dt(t)} if t else {"ok": True, "reminder": text}

    def calendar_add(self, title, time, duration=None):
        t = self.when(time)
        if t is None:
            return {"ok": False, "error": f"could not understand the time '{time}'"}
        mins = self.seconds(duration) // 60 if duration else 60
        if not self.dry:
            osa(as_date(t) + f'tell application "Calendar" to tell (first calendar whose writable is true) to '
                f"make new event with properties {{summary:{as_str(title)}, start date:d, end date:d + {mins * 60}}}")
        return {"ok": True, "event": title, "time": fmt_dt(t)}

    def calendar_list(self, day="today"):
        now = self.now()
        week = "week" in day.lower()
        start = now.replace(hour=0, minute=0) if day.lower() in ("today", "this week") else self.when(day + " 00:00") or now
        start = start.replace(hour=0, minute=0, second=0, microsecond=0)
        days = 7 if week else 1
        script = (as_date(start) + f"set e to d + {days * 86400}\nset out to {{}}\n"
                  'tell application "Calendar"\nrepeat with c in calendars\n'
                  "repeat with ev in (every event of c whose start date ≥ d and start date < e)\n"
                  'set end of out to ((start date of ev) as «class isot» as string) & "|" & summary of ev\n'
                  "end repeat\nend repeat\nend tell\n"
                  'set AppleScript\'s text item delimiters to linefeed\nreturn out as string')
        evs = []
        for line in filter(None, osa(script).splitlines()):
            iso, _, title = line.partition("|")
            evs.append({"title": title, "time": fmt_dt(dt.datetime.fromisoformat(iso))})
        return {"events": sorted(evs, key=lambda e: e["time"])}

    def list_emails(self, query="", n=5):
        if not self.gmail:
            return {"ok": False, "error": "Gmail is not connected (run: python chat.py --connect-gmail)"}
        return {"emails": self.gmail.list(query or "", n or 5)}

    def read_email(self, id):
        if not self.gmail:
            return {"ok": False, "error": "Gmail is not connected"}
        return self.gmail.read(id)

    def draft_email(self, to, subject, body):
        """Opens a pre-filled message in the Mail app; you press Send yourself."""
        url = "mailto:" + urllib.parse.quote(to) + "?" + urllib.parse.urlencode({"subject": subject, "body": body}, quote_via=urllib.parse.quote)
        if not self.dry:
            subprocess.run(["open", url])
        return {"ok": True, "draft_id": "mail-app"}

    send_email = draft_email

    def lookup(self, topic):
        hits = get_json("https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
            {"action": "query", "list": "search", "srsearch": topic, "srlimit": 1, "format": "json"}))["query"]["search"]
        if not hits:
            return {"ok": False, "error": f"nothing found for '{topic}'"}
        title = hits[0]["title"]
        s = get_json("https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(title.replace(" ", "_")))
        return {"title": s.get("title", title), "extract": " ".join(s.get("extract", "").split())[:700]}

    def web_search(self, query):
        """Search the internet, read the top pages and pull out the answer (assistant/websearch.py)."""
        from .websearch import web_answer
        res = web_answer(query)
        if res.get("steps") or res.get("answer"):      # same shape the model was fine-tuned on (make_web_data.py)
            return {"kind": res["kind"], **({"steps": res["steps"]} if res.get("steps") else {"answer": res["answer"]}),
                    "source": {"title": res["source"]["title"][:80], "url": res["source"]["url"]}}
        return res

    def _place(self, name):
        r = get_json("https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode({"name": name, "count": 1}))
        return (r.get("results") or [None])[0]

    def weather(self, location=None, when=None):
        p = self._place(location) if location else None
        if location and not p:
            return {"ok": False, "error": f"unknown place '{location}'"}
        if not p:      # no place given: rough location from the IP address
            ip = get_json("https://ipapi.co/json/")
            p = {"name": ip.get("city", "your location"), "latitude": ip["latitude"], "longitude": ip["longitude"]}
        w = get_json("https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode({
            "latitude": p["latitude"], "longitude": p["longitude"], "current": "temperature_2m,weather_code",
            "daily": "temperature_2m_max,temperature_2m_min,weather_code", "timezone": "auto", "forecast_days": 3}))
        i = 1 if when and "tomorrow" in when.lower() else 0
        code = w["daily"]["weather_code"][i] if i else w["current"]["weather_code"]
        temp = round(w["daily"]["temperature_2m_max"][i] if i else w["current"]["temperature_2m"])
        return {"location": location or p["name"], "when": when or "now", "summary": WMO.get(code, "mixed"), "temp_c": temp,
                "high_c": round(w["daily"]["temperature_2m_max"][i]), "low_c": round(w["daily"]["temperature_2m_min"][i])}

    def _notes(self):
        try:
            return json.loads(self.notes_file.read_text())
        except Exception:
            return []

    def notes_add(self, text):
        if not self.dry:
            notes = self._notes() + [{"text": text, "time": fmt_dt(self.now())}]
            self.notes_file.write_text(json.dumps(notes, indent=1))
        return {"ok": True}

    def notes_search(self, query):
        q = set(re.findall(r"\w+", query.lower()))
        hits = [n for n in self._notes() if q & set(re.findall(r"\w+", n["text"].lower()))]
        return {"notes": hits[-3:]}

    def calculator(self, expression):
        try:
            return {"result": round(safe_eval(expression), 6)}
        except Exception:
            return {"ok": False, "error": f"can't calculate '{expression}'"}

    def get_time(self, place):
        p = self._place(place)
        if not p or not p.get("timezone"):
            return {"ok": False, "error": f"unknown place '{place}'"}
        return {"place": p["name"], "time": dt.datetime.now(ZoneInfo(p["timezone"])).strftime("%H:%M")}

    def open_app(self, app):
        if not self.dry:
            r = subprocess.run(["open", "-a", app], capture_output=True, text=True)
            if r.returncode:
                return {"ok": False, "error": f"no app called '{app}'"}
        return {"ok": True}

    def call(self, contact):
        return {"ok": False, "error": "calls work on the phone app only"}

    def sms(self, to, message):
        return {"ok": False, "error": "text messages work on the phone app only"}

    def run(self, name, args):
        fn = getattr(self, name, None)
        if fn is None or name.startswith("_"):
            return {"ok": False, "error": f"tool '{name}' is not available here"}
        try:
            return fn(**args)
        except Exception as e:
            return {"ok": False, "error": str(e)[:200]}


WMO = {0: "clear sky", 1: "mostly clear", 2: "partly cloudy", 3: "cloudy", 45: "fog", 48: "fog", 51: "light drizzle", 53: "drizzle",
       55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain", 71: "light snow", 73: "snow", 75: "heavy snow",
       80: "rain showers", 81: "rain showers", 82: "heavy rain showers", 95: "thunderstorms", 96: "thunderstorms", 99: "thunderstorms"}

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
       ast.Pow: operator.pow, ast.Mod: operator.mod, ast.USub: operator.neg, ast.UAdd: operator.pos}


def safe_eval(expr):
    """Arithmetic only: numbers, + - * / ** % and brackets, sqrt(). Never runs code."""
    expr = expr.replace("^", "**").replace("×", "*").replace("÷", "/")

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            if isinstance(n.op, ast.Pow) and abs(ev(n.right)) > 100:
                raise ValueError("exponent too big")
            return OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in OPS:
            return OPS[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "sqrt" and len(n.args) == 1:
            return ev(n.args[0]) ** 0.5
        raise ValueError("not arithmetic")
    return ev(ast.parse(expr, mode="eval"))
