import zipfile, urllib.request
from pathlib import Path
import numpy as np


def load_enwik8(L, split="train", path="enwik8.zip"):
    if not Path(path).exists():
        urllib.request.urlretrieve("http://mattmahoney.net/dc/enwik8.zip", path)
    with zipfile.ZipFile(path) as z, z.open("enwik8") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    a, b = {"train": (0, 90_000_000), "val": (90_000_000, 95_000_000), "test": (95_000_000, 100_000_000)}[split]
    d = data[a:b]; n = len(d) // L
    return d[:n * L].reshape(n, L)


class Sampler:
    def __init__(self, data, bsz, seed=0):
        self.data, self.bsz, self.rng = data, bsz, np.random.default_rng(seed)
        self.order, self.pos = self.rng.permutation(len(data)), 0
    def next(self):
        if self.pos + self.bsz > len(self.data):
            self.order, self.pos = self.rng.permutation(len(self.data)), 0
        sel = self.order[self.pos:self.pos + self.bsz]; self.pos += self.bsz
        return np.asarray(self.data[sel], dtype=np.int32)
