"""
Loading Aalto keystroke data as fixed-length sequences for the keystroke verifier.

Each row of the dataset is one typed sentence ("section") with one entry per key:
    keycode_ids  key code of each key
    hl           hold latency       (key press -> its release)
    il           inter-key latency  (release -> next press)
    pl           press latency      (press -> next press)
    rl           release latency    (release -> next release)

A sentence becomes an array of shape (max_len, 5): [keycode/255, hl, il, pl, rl] per key,
zero-padded or truncated to max_len, plus its true length.
"""
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FEATURES = ("hl", "il", "pl", "rl")
N_FEATURES = 1 + len(FEATURES)       # keycode + 4 timings
DEFAULT_MAX_LEN = 50                 # TypeNet uses 50 keys per sequence


@dataclass
class KeystrokeSet:
    """Sentences as padded arrays, indexed by section id."""
    x: np.ndarray                    # (n, max_len, N_FEATURES) float32
    lengths: np.ndarray              # (n,) int, number of real keys (>= 1)
    participants: np.ndarray         # (n,) str
    sections: np.ndarray             # (n,) str
    index: dict                      # section id -> row

    def rows(self, section_ids):
        return np.array([self.index[str(s)] for s in section_ids])

    def by_participant(self):
        groups = defaultdict(list)
        for i, p in enumerate(self.participants):
            groups[p].append(i)
        return dict(groups)


def _to_array(keycodes, timings, max_len):
    n = min(len(keycodes), max_len)
    out = np.zeros((max_len, N_FEATURES), dtype=np.float32)
    out[:n, 0] = np.clip(np.asarray(keycodes[:n], dtype=np.float32), 0, 255) / 255.0
    for j, values in enumerate(timings, start=1):
        out[:n, j] = np.asarray(values[:n], dtype=np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), max(n, 1)


def load_sequences(parquet_dir: Path, split: str, participants=None, sections=None,
                   max_len=DEFAULT_MAX_LEN) -> KeystrokeSet:
    """
    Read the '<split>-*.parquet' files, keeping only the given participants and/or sections.
    Only the needed rows are converted, so selecting 10,000 participants fits easily in memory.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    columns = ["participant_id", "section_id", "keycode_ids", *FEATURES]
    keep_p = {str(p) for p in participants} if participants is not None else None
    keep_s = {str(s) for s in sections} if sections is not None else None
    xs, lengths, pids, sids = [], [], [], []
    files = sorted(Path(parquet_dir).glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {split}-*.parquet files in {parquet_dir}")
    for f in files:
        table = pq.read_table(f, columns=columns)
        mask = None
        if keep_p is not None:
            mask = pc.is_in(pc.cast(table["participant_id"], "string"), value_set=pa.array(sorted(keep_p)))
        if keep_s is not None:
            m = pc.is_in(pc.cast(table["section_id"], "string"), value_set=pa.array(sorted(keep_s)))
            mask = m if mask is None else pc.and_(mask, m)
        if mask is not None:
            table = table.filter(mask)
        d = table.to_pydict()
        for i in range(len(d["section_id"])):
            arr, n = _to_array(d["keycode_ids"][i], [d[k][i] for k in FEATURES], max_len)
            xs.append(arr)
            lengths.append(n)
            pids.append(str(d["participant_id"][i]))
            sids.append(str(d["section_id"][i]))
    x = np.stack(xs) if xs else np.zeros((0, max_len, N_FEATURES), dtype=np.float32)
    sids = np.array(sids)
    return KeystrokeSet(x, np.array(lengths), np.array(pids), sids, {s: i for i, s in enumerate(sids)})


@dataclass
class Standardizer:
    """Per-feature mean / std of the timing features, computed on training keys only."""
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, data: KeystrokeSet):
        mask = np.arange(data.x.shape[1])[None, :] < data.lengths[:, None]
        keys = data.x[mask]                               # real keys only, no padding
        mean, std = keys.mean(axis=0), keys.std(axis=0)
        mean[0], std[0] = 0.0, 1.0                        # keycode is already scaled to 0..1
        std[std < 1e-6] = 1.0
        return cls(mean.astype(np.float32), std.astype(np.float32))

    def transform(self, data: KeystrokeSet) -> KeystrokeSet:
        x = (data.x - self.mean) / self.std
        mask = np.arange(x.shape[1])[None, :] < data.lengths[:, None]
        x[~mask] = 0.0                                    # keep padding at zero
        return KeystrokeSet(x.astype(np.float32), data.lengths, data.participants, data.sections, data.index)

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(np.array(d["mean"], dtype=np.float32), np.array(d["std"], dtype=np.float32))


def pk_batches(groups: dict, p: int, k: int, rng: random.Random):
    """
    Yield batches of row indices with p participants x k sentences each (for triplet loss),
    covering every eligible participant once per epoch.
    """
    eligible = [pid for pid, rows in groups.items() if len(rows) >= 2]
    rng.shuffle(eligible)
    for start in range(0, len(eligible) - p + 1, p):
        batch = []
        for pid in eligible[start:start + p]:
            rows = groups[pid]
            batch += rng.sample(rows, k) if len(rows) >= k else [rng.choice(rows) for _ in range(k)]
        yield batch
