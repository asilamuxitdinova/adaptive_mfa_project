"""
Building verification pairs: "same person" (genuine, label 1) and "different people" (impostor, label 0).

All functions take a random.Random so results are reproducible with a fixed seed.
"""
import random
from itertools import combinations


def split_ids(ids, test_fraction, rng: random.Random):
    """Split identities into disjoint train / test sets, so no person appears in both."""
    ids = sorted(set(ids))
    rng.shuffle(ids)
    n_test = max(1, round(len(ids) * test_fraction))
    if n_test >= len(ids):
        raise ValueError("need at least one identity left for training")
    return sorted(ids[n_test:]), sorted(ids[:n_test])


def make_pairs(groups, rng: random.Random, max_genuine_per_id=None):
    """
    Build a balanced set of genuine and impostor pairs.

    groups: {identity: [item, ...]} where items are e.g. file paths or sample ids.
    max_genuine_per_id: cap on genuine pairs per identity (None = all combinations).
    Returns a list of (item_a, item_b, label, id_a, id_b), with as many impostor
    pairs as genuine pairs, in random order.
    """
    groups = {k: sorted(v) for k, v in groups.items() if v}
    if len(groups) < 2:
        raise ValueError("need at least two identities to build impostor pairs")

    genuine = []
    for pid in sorted(groups):
        combos = list(combinations(groups[pid], 2))
        if max_genuine_per_id is not None and len(combos) > max_genuine_per_id:
            combos = rng.sample(combos, max_genuine_per_id)
        genuine += [(a, b, 1, pid, pid) for a, b in combos]

    ids = sorted(groups)
    max_impostor = sum(len(groups[a]) * len(groups[b]) for a, b in combinations(ids, 2))
    target = min(len(genuine), max_impostor)
    seen, impostor = set(), []
    while len(impostor) < target:
        id_a, id_b = rng.sample(ids, 2)
        a, b = rng.choice(groups[id_a]), rng.choice(groups[id_b])
        key = frozenset((a, b))
        if key in seen:
            continue
        seen.add(key)
        impostor.append((a, b, 0, id_a, id_b))

    pairs = genuine[:target] if target < len(genuine) else genuine
    pairs = pairs + impostor
    rng.shuffle(pairs)
    return pairs
