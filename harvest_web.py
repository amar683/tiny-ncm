"""Collect web answers to everyday questions, for fine-tuning the model to copy answers out of search results.

python harvest_web.py --out data_web/harvest.jsonl [--max 1800] [--delay 1.0]

Each question TOPIC is searched once (assistant/websearch.web_answer); its answer is later reused for several phrasings.
About 1 in 9 topics is held out for testing (data_web/holdout.jsonl) and never used for training.
Safe to stop and restart: topics already in the output file are skipped.
"""
import argparse
import itertools
import json
import random
import time
from pathlib import Path

from assistant.websearch import web_answer

COOK = ["boil rice", "make an omelette", "roast a chicken", "bake bread", "make coffee in a french press", "cook quinoa", "poach an egg",
        "make pancakes", "cook pasta al dente", "grill salmon", "make guacamole", "make hummus", "cook steak in a pan", "make rice pudding",
        "make a smoothie", "bake potatoes in the oven", "make mashed potatoes", "hard boil eggs", "make scrambled eggs", "cook lentils",
        "make tomato soup", "make pizza dough", "caramelize onions", "make iced tea", "brew green tea", "make popcorn on the stove",
        "cook brown rice", "make a vinaigrette", "make banana bread", "cook frozen vegetables", "make chicken soup", "sharpen a kitchen knife",
        "store fresh herbs", "ripen avocados faster", "soften butter quickly", "make whipped cream", "cook tofu", "make french toast",
        "make cold brew coffee", "steam broccoli", "roast vegetables", "make a cheese sauce", "cook bacon in the oven", "make oatmeal"]
CLEAN = ["clean an oven", "descale a kettle", "clean a microwave", "remove limescale from a shower", "clean grout", "wash a pillow",
         "clean a mattress", "get rid of fruit flies", "clean a cast iron pan", "remove rust from metal", "clean windows without streaks",
         "clean a washing machine", "unclog a shower drain", "clean a dishwasher", "remove mold from a bathroom", "clean a fridge",
         "get rid of ants in the kitchen", "clean silver jewelry", "clean white sneakers", "remove pet hair from a couch", "clean a laptop screen",
         "clean a keyboard", "deodorize a carpet", "clean a burnt pot", "wash a wool sweater", "clean glasses lenses", "remove sticker residue"]
STAINS = ["coffee", "red wine", "grass", "blood", "ink", "grease", "sweat", "tomato sauce", "chocolate", "makeup", "oil"]
SURFACES = ["a shirt", "carpet", "a couch", "jeans", "a mattress"]
HOME = ["fix a leaky faucet", "hang a picture straight", "unclog a toilet", "fix a running toilet", "reset a tripped breaker",
        "change a light bulb safely", "patch a hole in drywall", "paint a room", "bleed a radiator", "fix a squeaky door",
        "lower my electricity bill", "keep my house cool without ac", "stop condensation on windows", "assemble flat pack furniture",
        "fix a wobbly table", "insulate a drafty window", "change a furnace filter", "fix a slow draining sink", "remove a stripped screw",
        "choose a houseplant for low light", "water succulents", "repot a plant", "get rid of weeds naturally", "grow tomatoes in pots",
        "start a compost bin", "care for an orchid", "grow herbs indoors", "mow the lawn properly"]
DEVICES = ["iphone", "android phone", "mac", "windows laptop"]
TECH_ACT = ["take a screenshot", "free up storage", "update the software", "turn on dark mode", "change the wallpaper",
            "clear the cache", "find my wifi password", "speed up", "back up photos", "turn off notifications", "check the battery health"]
TECH = ["reset my router", "make my wifi faster", "make a strong password", "spot a phishing email", "convert a word file to pdf",
        "record my screen", "connect bluetooth headphones", "share my location", "block spam calls", "set up two factor authentication",
        "fix a frozen app", "print from my phone", "transfer photos to a new phone", "use a vpn", "zip a folder"]
HEALTH = ["stop hiccups", "treat a sunburn", "fall asleep faster", "get rid of a headache without pills", "treat a minor burn",
          "stay hydrated", "improve my posture", "stretch my lower back", "relieve a stiff neck", "reduce stress", "start running",
          "do a proper push up", "boost my energy in the afternoon", "treat a bee sting", "clean a small cut", "soothe a sore throat",
          "stop a nosebleed", "beat jet lag", "drink more water", "build a morning routine", "meditate for beginners", "do a plank"]
LIFE = ["make a monthly budget", "save money on groceries", "write a cover letter", "prepare for a job interview", "pack a carry on bag",
        "tie a tie", "iron a shirt", "fold a fitted sheet", "sew a button", "wrap a gift neatly", "write a thank you note",
        "learn a language fast", "study for an exam", "remember names better", "be more productive", "plan a road trip",
        "jump start a car", "check tire pressure", "change a car tire", "defrost a windshield", "parallel park", "improve my credit score",
        "negotiate a salary", "start a vegetable garden", "train a puppy to sit", "trim a cat's nails", "bathe a dog", "keep a dog cool"]
WHY = ["why is the ocean salty", "why do we yawn", "why do cats purr", "why do cats knead", "why is the sky blue", "why do leaves change color",
       "why do onions make you cry", "why do we get goosebumps", "why do dogs wag their tails", "why is the sunset red", "why do we dream",
       "why does bread go stale", "why do bananas turn brown", "why does ice float", "why do we hiccup", "why do knuckles crack",
       "why do mosquitoes bite", "why does the moon change shape", "why do birds migrate", "why is my phone battery draining fast",
       "why does my wifi keep disconnecting", "why do my ears pop on a plane", "why do we get brain freeze", "why does coffee keep you awake",
       "why do plants need sunlight", "why do fingers wrinkle in water", "why do we sneeze", "why do stars twinkle", "why is grass green",
       "why does my laptop get hot", "why do cut apples turn brown", "why does popcorn pop", "why do we have seasons", "why does salt melt ice",
       "why do bubbles form in boiling water", "why do we get tired after eating", "why do cats sleep so much", "why do owls hoot at night",
       "why does milk go sour", "why does my car battery die in the cold", "why is the dead sea salty", "why do zebras have stripes"]
CAN = ["can I freeze milk", "can I freeze cooked rice", "can I eat eggs after the expiry date", "can I charge my phone overnight",
       "can dogs eat grapes", "can cats drink milk", "can I reuse cooking oil", "can I put a cast iron pan in the dishwasher",
       "can I microwave plastic containers", "can I freeze bread", "can I water plants with tap water", "can I drink coffee while pregnant",
       "can I wash a down jacket", "can I use baking soda instead of baking powder", "can I eat raw oats", "can I paint over wallpaper",
       "is it safe to eat raw cookie dough", "is it safe to leave a laptop plugged in", "is it bad to sleep with wet hair",
       "is it ok to run every day", "should I stretch before running", "should I turn off my router at night", "should I wash new clothes",
       "can I freeze cheese", "can I put a phone in rice", "is it safe to reheat rice", "can I mix bleach and vinegar",
       "should I refrigerate tomatoes", "can dogs eat chocolate", "can I recycle pizza boxes"]
HOWLONG = ["boil an egg", "steep green tea", "bake a potato", "cook rice", "roast a chicken", "boil pasta", "let dough rise",
           "cook salmon in the oven", "charge a new phone", "keep leftovers in the fridge", "steam broccoli", "grill a burger",
           "microwave a sweet potato", "soak beans", "cook a hard boiled egg", "keep cooked chicken in the fridge", "rest a steak",
           "brew cold brew coffee", "nap for", "boil corn on the cob"]


FOODS = ["chicken wings", "frozen fries", "salmon", "sweet potatoes", "broccoli", "chicken breast", "pork chops", "shrimp", "tofu",
         "brussels sprouts", "meatballs", "corn on the cob", "a whole fish", "bacon", "eggs"]
METHODS = ["in an air fryer", "in the oven", "on the grill"]
STORE = ["bananas", "fresh bread", "cut onions", "avocados", "berries", "coffee beans", "potatoes", "leftover rice", "cheese", "lettuce",
         "fresh fish", "cooked pasta", "herbs", "garlic", "apples", "tomatoes", "mushrooms", "eggs"]
PESTS = ["ants", "fruit flies", "cockroaches", "mice", "moths", "spiders", "wasps", "bed bugs", "silverfish", "aphids on plants"]
APPS = ["excel", "google docs", "gmail", "whatsapp", "zoom", "google sheets", "word", "instagram"]
APP_ACT = ["undo a mistake", "share a file", "change the font size", "add a signature", "mute notifications", "make a table"]
CARE = ["a goldfish", "a hamster", "a rabbit", "a parrot", "a cactus", "a snake plant", "a fiddle leaf fig", "a bonsai tree",
        "a peace lily", "a kitten", "a puppy", "a tortoise"]


def bases():
    """-> [(kind, base question)]"""
    B = [("howto", t) for t in COOK + CLEAN + HOME + TECH + HEALTH + LIFE]
    B += [("howto", f"remove {s} stains from {f}") for s, f in itertools.product(STAINS, SURFACES)]
    B += [("howto", f"{a} on my {d}") for a, d in itertools.product(TECH_ACT, DEVICES)]
    B += [("howto", f"cook {f} {m}") for f, m in itertools.product(FOODS, METHODS)]
    B += [("howto", f"store {f}") for f in STORE]
    B += [("howto", f"get rid of {p}") for p in PESTS]
    B += [("howto", f"{x} in {ap}") for x, ap in itertools.product(APP_ACT, APPS)]
    B += [("howto", f"take care of {c}") for c in CARE]
    B += [("why", w) for w in WHY] + [("can", c) for c in CAN] + [("howlong", h) for h in HOWLONG]
    return B


def question(kind, base, r):
    if kind == "howto":
        return r.choice(["how do I {b}?", "how to {b}", "how can I {b}?", "what's the best way to {b}?", "how should I {b}?"]).format(b=base)
    if kind == "howlong":
        return r.choice(["how long should I {b}?", "how long does it take to {b}?", "how long to {b}"]).format(b=base)
    return base + "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_web/harvest.jsonl")
    ap.add_argument("--max", type=int, default=2000)
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    B = bases()
    r = random.Random(a.seed)
    r.shuffle(B)
    hold = set(range(0, len(B), 9))                      # about 1 in 9 topics: test only
    done = set()
    if out.exists():
        done = {json.loads(l)["base"] for l in out.read_text().splitlines() if l.strip()}
    print(f"{len(B)} topics ({len(hold)} held out), {len(done)} already done", flush=True)
    n_ok = n_fail = 0
    with out.open("a") as f:
        for i, (kind, base) in enumerate(B[: a.max]):
            if base in done:
                continue
            q = question(kind, base, r)
            for attempt in range(3):
                try:
                    res = web_answer(q, log=False)
                    break
                except Exception as e:
                    res = {"ok": False, "error": str(e)[:100]}
                    time.sleep(5 * (attempt + 1))
            rec = {"base": base, "kind": kind, "question": q, "split": "test" if i in hold else "train",
                   **{k: res[k] for k in ("kind", "steps", "answer", "source", "results") if k in res}}
            rec["kind"] = res.get("kind", "none")
            rec["qkind"] = kind
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            ok = rec["kind"] in ("steps", "answer")
            n_ok += ok
            n_fail += not ok
            if (n_ok + n_fail) % 25 == 0:
                print(f"  {n_ok + n_fail} searched, {n_ok} with an answer, {n_fail} without", flush=True)
            time.sleep(a.delay)
    print(f"done: {n_ok} answers, {n_fail} without")


if __name__ == "__main__":
    main()
