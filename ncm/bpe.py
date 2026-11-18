"""Byte-level BPE text tokenizer with chat/tool special tokens."""
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIALS = ["<|pad|>", "<|eos|>", "<|system|>", "<|user|>", "<|assistant|>", "<|tool|>",
            "<|result|>", "<|end|>"]


class BPE:
    def __init__(self, tok):
        self.tok = tok
        self.ids = {s: tok.token_to_id(s) for s in SPECIALS}
        self.pad, self.eos, self.end = self.ids["<|pad|>"], self.ids["<|eos|>"], self.ids["<|end|>"]

    def __len__(self):
        return self.tok.get_vocab_size()

    @classmethod
    def train(cls, texts, vocab_size=16384):
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        tr = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIALS, min_frequency=2,
                                 initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                 show_progress=False)
        tok.train_from_iterator(texts, tr)
        return cls(tok)

    @classmethod
    def load(cls, path):
        return cls(Tokenizer.from_file(str(path)))

    def save(self, path):
        self.tok.save(str(path))

    def encode(self, text):
        return self.tok.encode(text).ids

    def encode_batch(self, texts):
        return [e.ids for e in self.tok.encode_batch(texts)]

    def decode(self, ids, skip_special=True):
        if skip_special:
            ids = [i for i in ids if i != self.pad]
        return self.tok.decode(ids, skip_special_tokens=skip_special)
