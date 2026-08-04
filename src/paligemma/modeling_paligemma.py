import json

import numpy as np
from torch._decomp.decompositions import pad_sequence

SEQ_LEN = 150


def pad_sequence(tokens):
    vec = np.zeros(SEQ_LEN)
    vec[: len(tokens)] = tokens
    return vec


def create_vocab():
    vocals = ["a", "e", "i", "o", "u"]
    consonants = [
        "b",
        "c",
        "d",
        "f",
        "g",
        "h",
        "j",
        "k",
        "l",
        "m",
        "n",
        "p",
        "q",
        "r",
        "s",
        "t",
        "v",
        "w",
        "x",
        "y",
        "z",
    ]
    vocab = {
        "<BOS>": 0,
        "<EOS>": 1,
        "<PAD>": 2,
    }
    cont = 3
    for consonant in consonants:
        for vocal in vocals:
            token = consonant + vocal
            vocab[token] = cont
            cont += 1
    # Save vocab as json
    with open("vocab.json", "w") as f:
        json.dump(vocab, f)


def forward_pass():
    tokens = [0, 23, 43, 53, 6]
    tokens = pad_sequence(tokens)
    print(tokens)


if __name__ == "__main__":
    forward_pass()
