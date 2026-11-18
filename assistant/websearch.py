"""Answer a question from the internet: search, read the top pages, pull out the part that answers it.

    web_answer("how do I change a tire?")
      1. search DuckDuckGo (no API key; Brave Search if BRAVE_API_KEY is set; Wikipedia if both fail)
      2. read the top 3 pages at the same time (text only: scripts, menus, footers are dropped)
      3. "how do I / how to / what should I do" questions -> the best list of steps on those pages
         other questions -> the 2-3 sentences that best match the question
      4. return the answer with its source, and save the question + answer to ~/.tiny-ncm/web_answers.jsonl

The 21M model can't read and summarise whole web pages, so the picking is done here (scoring words that match
your question), and the model only phrases the reply. Saved answers can become training data later.
Page text is treated as data only: nothing from a page is run or followed.
"""
import concurrent.futures as cf
import datetime as dt
import html
import json
import math
import os
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
      "Accept-Language": "en"}
LOG = Path.home() / ".tiny-ncm" / "web_answers.jsonl"
STOP = set("""a an the and or but if of to in on at for from by with about as is are was were be been it its this that these those
you your i me my we our they their he she his her them what which who when where why how do does did can could should would will
shall may might must have has had not no yes so than then there here just very really get got make some any all more most much
many way best good""".split())
HOWTO = re.compile(r"^(?:how (?:do|can|should|would|to|does one)|how to|what (?:should|do|can) (?:i|we|you) do|steps? (?:to|for)|"
                   r"ways? to|tips? (?:for|on|to)|what(?:'s| is) the best way to|best way to)\b", re.I)
FLUFF = re.compile(r"\b(?:learn (?:what|how|more|why)|read on|keep reading|in this (?:article|guide|post)|find out|here's (?:how|what|why)|"
                   r"we'll|we've|we're|let's|you'll learn|overwhelming|wondering|ever wonder|check out|our (?:guide|forum|readers)|"
                   r"wikihow readers|click|video|subscribe|cookie|sign up|newsletter|advertis|read more|this article|article|takes a look|bitesize)\b", re.I)
VERBS = set("""add adjust allow apply arrange ask avoid bake beat begin blend blot boil bring brush buy call change check choose clean
clear close combine connect cook cool cover create cut dab decide dilute dip do drain drink drop dry eat empty enable enter exercise
fill find finish flip fold follow gather get give go grab grease heat hold insert install keep knead launch layer leave let lift
limit line list loosen lower make mark measure melt mix move note open pick place plan plug pour prepare preheat press pull put raise
read reduce remove repeat replace rinse roll rub run save scrape scrub season select serve set shake share sign sit slice soak spray
spread sprinkle squeeze start stay stick stir stop store stretch switch take talk tap test tighten top try turn unplug use wait walk
wash watch whisk wipe write""".split())
SKIP_TAGS = {"script", "style", "nav", "footer", "header", "aside", "form", "noscript", "svg", "button", "select", "iframe", "template"}
BLOCK_TAGS = {"p", "li", "h1", "h2", "h3", "h4", "dd", "blockquote", "td"}


# ---------- search ----------
def _get(url, data=None, timeout=8, limit=2_000_000, headers=None):
    req = urllib.request.Request(url, data=data, headers={**UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        ctype = r.headers.get("Content-Type", "")
        raw = r.read(limit)
    m = re.search(r"charset=([\w-]+)", ctype)
    return raw.decode(m.group(1) if m else "utf-8", "replace"), ctype


def search(query, n=6):
    """-> [{"title", "url", "snippet"}]"""
    key = os.environ.get("BRAVE_API_KEY")
    if key:
        try:
            body, _ = _get("https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode({"q": query, "count": n}),
                           headers={"X-Subscription-Token": key, "Accept": "application/json"})
            return [{"title": r["title"], "url": r["url"], "snippet": _clean(r.get("description", ""))}
                    for r in json.loads(body)["web"]["results"][:n]]
        except Exception:
            pass
    try:
        body, _ = _get("https://lite.duckduckgo.com/lite/?" + urllib.parse.urlencode({"q": query}))
        out = []
        for m in re.finditer(r"<a[^>]+href=\"([^\"]+)\"[^>]*class='result-link'[^>]*>(.*?)</a>(.*?)class='result-snippet'>(.*?)</td>", body, re.S):
            url = html.unescape(m.group(1))
            if "duckduckgo.com" in url:          # ads go through a duckduckgo.com redirect
                continue
            out.append({"title": _clean(m.group(2)), "url": url, "snippet": _clean(m.group(4))})
        if out:
            return out[:n]
    except Exception:
        pass
    body, _ = _get("https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
        {"action": "query", "list": "search", "srsearch": query, "srlimit": n, "format": "json"}))
    return [{"title": h["title"], "url": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(h["title"].replace(" ", "_")),
             "snippet": _clean(h["snippet"])} for h in json.loads(body)["query"]["search"]]


def _clean(s):
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", s)).split())


# ---------- reading a page ----------
class _Reader(HTMLParser):
    """Collects text blocks: headings, paragraphs and list items (with the list they belong to)."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks, self.skip, self.cur, self.tag = [], 0, [], None
        self.lists, self.list_id, self.heading = [], 0, ""

    def handle_starttag(self, tag, attrs):
        if tag in SKIP_TAGS:
            self.skip += 1
        elif tag in ("ol", "ul"):
            self.list_id += 1
            self.lists.append((self.list_id, tag))
        elif tag in BLOCK_TAGS and not self.skip:
            self._flush()
            self.tag = tag
        elif tag == "br":
            self.cur.append(" ")

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS:
            self.skip = max(0, self.skip - 1)
        elif tag in ("ol", "ul"):
            self._flush()
            if self.lists:
                self.lists.pop()
        elif tag in BLOCK_TAGS:
            self._flush()

    def handle_data(self, data):
        if not self.skip and self.tag:
            self.cur.append(data)

    def _flush(self):
        text = " ".join("".join(self.cur).split())
        if text and self.tag:
            kind = "h" if self.tag[0] == "h" else self.tag
            lst = self.lists[-1] if self.lists and kind == "li" else None
            if kind == "h":
                self.heading = text
            self.blocks.append({"kind": kind, "text": text, "list": lst, "heading": self.heading})
        self.cur, self.tag = [], None


def read_page(url, timeout=7):
    try:
        body, ctype = _get(url, timeout=timeout)
    except Exception:
        return []
    if "html" not in ctype:
        return []
    r = _Reader()
    try:
        r.feed(body)
        r._flush()
    except Exception:
        pass
    return r.blocks


# ---------- picking the answer ----------
def _terms(text):
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 1]


def _stem(w):
    for suf in ("ing", "ies", "es", "ed", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def _sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text) if 30 <= len(s.strip()) <= 320]


def _steps(q, pages):
    """The best list of steps: pick the page whose step lists match best, then start from its FIRST good list
    (how-to pages split steps into Part 1, Part 2…) and carry on into the next ones, up to 8 steps."""
    qt = set(_terms(q))
    best = None                                         # (page score, page, [lists in page order])
    for rank, (page, blocks) in enumerate(pages):
        lists = {}
        for b in blocks:
            if b["kind"] == "li" and b["list"]:
                lists.setdefault(b["list"], {"items": [], "heading": b["heading"]})["items"].append(b["text"])
        good = []
        for (lid, tag), L in lists.items():
            items = [i for i in L["items"] if 15 <= len(i) <= 600]
            if len(items) < 3 or len(items) > 25:
                continue
            words = [len(i.split()) for i in items]
            if sum(words) / len(words) < 4:            # menus and link lists: very short items
                continue
            text_terms = set(_terms(L["heading"] + " " + " ".join(items)))
            quantity = sum(bool(re.match(r"[\d½¼¾/⅓.-]+\s+(?:cups?|tbsp|tsp|tablespoons?|teaspoons?|quarts?|oz|g|ml|pounds?|lbs?|grams?)\b", i))
                           for i in items) / len(items)                       # ingredient lists: "1 quart water"
            action = sum(re.sub(r"\W", "", _unnumber(i).split()[0].lower()) in VERBS for i in items) / len(items)
            titles = sum(sum(w[:1].isupper() for w in i.split()) >= 0.6 * len(i.split()) and not re.search(r"[.!?]$", i)
                         for i in items) / len(items)                    # contents lists, recipe-card titles
            score = (len(qt & text_terms) / (len(qt) or 1) + (0.3 if tag == "ol" else 0) + 0.8 * action - 0.8 * quantity - 0.8 * titles
                     + (0.2 if re.search(r"\b(?:step|how to|instructions|directions|method)", L["heading"], re.I) else 0))
            if score >= 0.6:
                good.append((score, lid, items))
        if good:
            page_score = max(g[0] for g in good) - 0.1 * rank          # earlier search results are usually better
            if best is None or page_score > best[0]:
                best = (page_score, page, sorted(good, key=lambda g: g[1]))
    if not best:
        return None, None
    _, page, good = best
    top = max(g[0] for g in good)
    start = next(i for i, g in enumerate(good) if g[0] >= 0.8 * top)   # first list that is nearly as good as the best
    steps = []
    for _, _, items in good[start:]:
        items = [i for i in items if not FLUFF.search(i)]
        if not items:
            continue
        acts = [i for i in items if re.sub(r"\W", "", _unnumber(i).split()[0].lower()) in VERBS]
        if len(acts) < 2 and len(good) > 1:
            continue                                   # a list of tips, not steps
        steps += [_short(i) for i in (acts if len(acts) >= 2 else items)]     # keep the "do this" lines, drop the tips
        if len(steps) >= 8:
            break
    return steps[:8], page


def _unnumber(item):
    return re.sub(r"^\s*(?:step\s*)?\d+[.):]?\s+", "", item, flags=re.I)


def _short(item):
    """One step: its first sentence, at most ~200 characters."""
    item = re.sub(r"\[\d+\]\s*(?:X\s+(?:Research|Trustworthy|Expert)\b.*)?", "", _unnumber(item)).strip()   # "[9] X Research source"
    s = re.split(r"(?<=[.!?])\s+", item)[0]
    return s if len(s) <= 200 else s[:197].rsplit(" ", 1)[0] + "…"


def _best_sentences(q, pages, results, k=3):
    qt = _terms(q)
    cands = []                                         # (sentence, page)
    for r in results[:4]:
        for s in _sentences(r["snippet"]):
            cands.append((s, r))
    pos = {}
    for page, blocks in pages:
        n = 0
        for b in blocks:
            if b["kind"] in ("p", "dd", "blockquote"):
                for s in _sentences(b["text"]):
                    cands.append((s, page))
                    pos[len(cands) - 1] = n
                    n += 1
    if not cands:
        return None, None
    docs = [set(_terms(s)) for s, _ in cands]
    N = len(docs)
    idf = {t: math.log(1 + N / (1 + sum(t in d for d in docs))) for t in set(qt)}
    why = re.match(r"\s*why\b", q, re.I)
    scored = []
    for i, ((s, page), d) in enumerate(zip(cands, docs)):
        score = sum(idf[t] for t in set(qt) if t in d) / (1 + 0.02 * len(d))
        score *= 1 / (1 + pos.get(i, 0) / 25)                     # explanations usually come early on a page
        if why and re.search(r"\b(?:because|due to|caused by|since|results? from|the reason)\b", s, re.I):
            score *= 1.5
        if FLUFF.search(s) or re.search(r"\b(?:we|our|us)\b", s, re.I) or s.endswith("?"):
            score *= 0.2                                           # page filler: "Learn what happens…", "Planning a trip can feel…"
        if re.match(r"(?:this|that|it|these|they)\b", s, re.I):        # sentences that lean on the one before
            score *= 0.7
        scored.append((score, i, s, page))
    by_page = {}
    for score, i, s, page in sorted(scored, key=lambda x: -x[0]):
        by_page.setdefault(id(page), (page, []))[1].append((score, i, s))
    page, best = max(by_page.values(), key=lambda pv: sum(x[0] for x in pv[1][:k]))   # the page whose best k sentences score most
    picked, seen = [], []
    for score, i, s in best:
        if score <= 0:
            break
        key = set(_terms(s))
        if any(len(key & o) / (len(key | o) or 1) > 0.6 for o in seen):
            continue
        picked.append((i, s))
        seen.append(key)
        if len(picked) == k:
            break
    picked.sort()
    return " ".join(s for _, s in picked) or None, page


def web_answer(question, read=3, log=True):
    q = question.strip().rstrip("?")
    results = search(q)
    if not results:
        return {"ok": False, "error": "no search results"}
    with cf.ThreadPoolExecutor(max_workers=read) as ex:
        texts = list(ex.map(read_page, [r["url"] for r in results[:read]]))
    pages = [(r, t) for r, t in zip(results[:read], texts) if t]
    out = {"results": [{"title": r["title"], "snippet": r["snippet"][:220]} for r in results[:3]]}
    steps, page = _steps(q, pages) if HOWTO.search(q) else (None, None)
    if steps:
        out.update(kind="steps", steps=steps, source={"title": page["title"], "url": page["url"]})
    else:
        ans, page = _best_sentences(q, pages, results)
        if ans:
            out.update(kind="answer", answer=ans, source={"title": page["title"], "url": page["url"]})
    out["sources"] = [{"title": r["title"], "url": r["url"]} for r in results[:read]]
    if log and ("steps" in out or "answer" in out):
        try:
            LOG.parent.mkdir(exist_ok=True)
            with LOG.open("a") as f:
                f.write(json.dumps({"time": dt.datetime.now().isoformat(timespec="seconds"), "question": question,
                                    **{k: out[k] for k in ("kind", "steps", "answer", "source") if k in out}}, ensure_ascii=False) + "\n")
        except OSError:
            pass
    return out


def domain(url):
    return re.sub(r"^www\.", "", urllib.parse.urlparse(url).netloc)
