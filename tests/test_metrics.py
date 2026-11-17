import numpy as np
import pytest

from src.metrics import compute_eer, far_frr


def test_perfect_separation_has_zero_eer():
    scores = [0.9, 0.8, 0.95, 0.1, 0.2, 0.05]
    labels = [1, 1, 1, 0, 0, 0]
    result = compute_eer(scores, labels)
    assert result.eer == 0.0
    assert result.accuracy == 1.0


def test_far_frr_at_threshold():
    scores = [0.9, 0.6, 0.4, 0.7, 0.3, 0.1]
    labels = [1, 1, 1, 0, 0, 0]
    far, frr = far_frr(scores, labels, threshold=0.5)
    assert far == pytest.approx(1 / 3)   # impostor 0.7 accepted
    assert frr == pytest.approx(1 / 3)   # genuine 0.4 rejected


def test_known_eer():
    # one genuine below one impostor at the crossing: FAR = FRR = 1/4
    scores = [0.9, 0.8, 0.7, 0.35, 0.4, 0.3, 0.2, 0.1]
    labels = [1, 1, 1, 1, 0, 0, 0, 0]
    result = compute_eer(scores, labels)
    assert result.eer == pytest.approx(0.25)
    assert result.far == pytest.approx(result.frr)


def test_overlapping_gaussians_match_theory():
    # genuine ~ N(1, 1), impostor ~ N(-1, 1): theoretical EER = Phi(-1) ~ 15.87%
    rng = np.random.default_rng(0)
    genuine = rng.normal(1, 1, 50_000)
    impostor = rng.normal(-1, 1, 50_000)
    scores = np.concatenate([genuine, impostor])
    labels = np.concatenate([np.ones(50_000), np.zeros(50_000)])
    assert compute_eer(scores, labels).eer == pytest.approx(0.1587, abs=0.005)


@pytest.mark.parametrize("scores,labels", [
    ([0.1, 0.2], [1, 1]),          # no impostors
    ([0.1, 0.2], [0, 2]),          # bad label
    ([0.1], [1, 0]),               # length mismatch
])
def test_invalid_input_raises(scores, labels):
    with pytest.raises(ValueError):
        compute_eer(scores, labels)
