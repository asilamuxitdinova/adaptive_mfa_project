import random

import pytest

from src.pairs import make_pairs, split_ids


GROUPS = {f"p{i}": [f"p{i}_s{j}" for j in range(5)] for i in range(6)}


def test_pairs_are_balanced_and_labelled_correctly():
    pairs = make_pairs(GROUPS, random.Random(0))
    genuine = [p for p in pairs if p[2] == 1]
    impostor = [p for p in pairs if p[2] == 0]
    assert len(genuine) == len(impostor) == 6 * 10          # C(5,2) = 10 per person
    assert all(p[3] == p[4] for p in genuine)
    assert all(p[3] != p[4] for p in impostor)


def test_no_duplicate_impostor_pairs():
    pairs = make_pairs(GROUPS, random.Random(0))
    keys = [frozenset(p[:2]) for p in pairs if p[2] == 0]
    assert len(keys) == len(set(keys))


def test_genuine_cap_per_identity():
    pairs = make_pairs(GROUPS, random.Random(0), max_genuine_per_id=3)
    assert sum(p[2] for p in pairs) == 6 * 3


def test_same_seed_same_pairs():
    assert make_pairs(GROUPS, random.Random(42)) == make_pairs(GROUPS, random.Random(42))


def test_split_ids_is_disjoint_and_complete():
    train, test = split_ids(range(100), 0.2, random.Random(1))
    assert len(test) == 20 and len(train) == 80
    assert not set(train) & set(test)
    assert set(train) | set(test) == set(range(100))


def test_needs_two_identities():
    with pytest.raises(ValueError):
        make_pairs({"only": ["a", "b"]}, random.Random(0))
