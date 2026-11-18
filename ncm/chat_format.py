"""Chat/tool conversation format shared by training, the desktop runtime and (ported) mobile.

Every message becomes one segment, padded with <|pad|> to a whole number of blocks:
    system:    <|system|> {date/time + tool list} <|end|>
    user:      <|user|> {text} <|end|>
    assistant: <|assistant|> {reply} <|end|>          (trained)
    tool:      <|tool|> {"name": ..., "args": {...}} <|end|>   (trained)
    result:    <|result|> {json} <|end|>
The model generates the next segment starting with its role token, so it decides by itself
whether to answer (<|assistant|>) or call a tool (<|tool|>).
"""
import json
from pathlib import Path

REGISTRY = Path(__file__).resolve().parent.parent / "tools" / "registry.json"
ROLE_TOKEN = {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>",
              "tool": "<|tool|>", "result": "<|result|>"}
TRAINED = {"assistant", "tool"}


def load_registry(path=REGISTRY):
    return json.loads(Path(path).read_text())["tools"]


def signature(tool):
    args = ",".join(a for a in tool["args"])
    return f"{tool['name']}({args})"


def system_prompt(now, tools):
    """now: datetime. tools: list of registry entries."""
    when = now.strftime("%A %d %B %Y, %H:%M")
    return f"Now: {when}. Tools: " + " ".join(signature(t) for t in tools)


def tool_call_text(name, args):
    return json.dumps({"name": name, "args": args}, ensure_ascii=False, separators=(",", ":"))


def format_conv(bpe, msgs, block, max_len=None, max_segment=None):
    """msgs: list of {"role", "content"}. Returns (tokens, loss_per_token) with every segment
    padded to a block multiple. loss=1 on trained segments (role token, text, <|end|>, pads)."""
    toks, loss = [], []
    end = bpe.end
    for m in msgs:
        role = m["role"]
        body = bpe.encode(m["content"])
        if max_segment and len(body) > max_segment:
            body = body[:max_segment]
        seg = [bpe.ids[ROLE_TOKEN[role]]] + body + [end]
        seg += [bpe.pad] * ((-len(seg)) % block)
        toks += seg
        loss += [1 if role in TRAINED else 0] * len(seg)
        if max_len and len(toks) > max_len:
            return None
    return toks, loss


def prompt_tokens(bpe, msgs, block):
    """Tokens for a conversation so far (the model continues with the next segment)."""
    return format_conv(bpe, msgs, block)[0]


def parse_segment(bpe, ids):
    """Decode a generated segment -> ("assistant", text) or ("tool", {"name", "args"}) or
    ("tool_invalid", raw text)."""
    ids = [i for i in ids if i != bpe.pad]
    role = None
    if ids and ids[0] == bpe.ids["<|tool|>"]:
        role, ids = "tool", ids[1:]
    elif ids and ids[0] == bpe.ids["<|assistant|>"]:
        role, ids = "assistant", ids[1:]
    if ids and ids[-1] == bpe.end:
        ids = ids[:-1]
    text = bpe.decode(ids).strip()
    if role == "tool":
        try:
            call = json.loads(text)
            if isinstance(call, dict) and "name" in call:
                call.setdefault("args", {})
                return "tool", call
        except json.JSONDecodeError:
            pass
        return "tool_invalid", text
    return "assistant", text
