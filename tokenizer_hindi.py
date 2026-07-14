import os
import torch
import re

def split_into_aksharas(text):
    """
    Hindi Akshara tokenizer.
    """

    text = text.strip()

    tokens = []

    i = 0
    n = len(text)

    while i < n:

        ch = text[i]

        if ch.isspace():
            tokens.append("<SP>")
            i += 1
            continue

        cluster = ch
        i += 1

        while i < n:

            curr = text[i]
            code = ord(curr)

            # virama
            if curr == "्":
                cluster += curr

                if i + 1 < n:
                    cluster += text[i + 1]
                    i += 2
                else:
                    i += 1

            # matras
            elif (
                0x093E <= code <= 0x094C
                or curr in ["ं", "ः", "ँ", "़", "ॉ", "ॆ", "ॊ"]
            ):
                cluster += curr
                i += 1

            else:
                break

        tokens.append(cluster)

    return tokens


class HindiTokenizer:
    PAD_TOKEN = "<PAD>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"
    SPACE_TOKEN = "<SP>"

    def __init__(self, vocab_path=None):
        self.token2id = {}
        self.id2token = {}

        if vocab_path and os.path.exists(vocab_path):
            self.load(vocab_path)
        else:
            self._init_special_tokens()

    def _init_special_tokens(self):
        specials = [
            self.PAD_TOKEN,
            self.EOS_TOKEN,
            self.UNK_TOKEN,
            self.SPACE_TOKEN,
        ]

        self.token2id = {t: i for i, t in enumerate(specials)}
        self.id2token = {i: t for i, t in enumerate(specials)}

    @property
    def vocab_size(self):
        return len(self.token2id)

    @property
    def pad_token_id(self):
        return self.token2id[self.PAD_TOKEN]

    @property
    def eos_token_id(self):
        return self.token2id[self.EOS_TOKEN]

    @property
    def unk_token_id(self):
        return self.token2id[self.UNK_TOKEN]


    def build_from_texts(self, texts):

        self._init_special_tokens()

        vocab = set()

        for text in texts: 
            vocab.update(split_into_aksharas(text))

        for tok in sorted(vocab):

            if tok not in self.token2id:

                idx = len(self.token2id)

                self.token2id[tok] = idx
                self.id2token[idx] = tok

        print(f"Vocabulary built: {len(self.token2id)}")

    def encode(self, text, return_tensor=True):

        tokens = split_into_aksharas(text)

        ids = [
            self.token2id.get(t, self.unk_token_id)
            for t in tokens
        ]

        ids.append(self.eos_token_id)

        if return_tensor:
            return torch.tensor(ids, dtype=torch.long)

        return ids

    def decode(self, ids, skip_special=True):

        special = {
            self.PAD_TOKEN,
            self.EOS_TOKEN,
            self.UNK_TOKEN,
        }

        out = []

        for idx in ids:

            tok = self.id2token.get(int(idx), self.UNK_TOKEN)

            if skip_special and tok in special:
                continue

            if tok == self.SPACE_TOKEN:
                out.append(" ")
            else:
                out.append(tok)

        return "".join(out)

    def save(self, path):

        torch.save(
            {
                "token2id": self.token2id,
                "id2token": self.id2token,
            },
            path,
        )

        print(f"Saved vocab -> {path}")

    def load(self, path):

        data = torch.load(path, map_location="cpu")

        self.token2id = data["token2id"]
        self.id2token = data["id2token"]

        print(
            f"Vocab loaded from {path} "
            f"(size={len(self.token2id)})"
        )