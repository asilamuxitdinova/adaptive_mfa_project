"""
Face verifier tests on synthetic images: no LFW download and no pretrained weights needed.
Each fake "person" is a distinct random pattern; their photos are noisy copies of it.
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.verifiers.face import best_threshold, center_crop, cosine_scores, kfold_accuracy, pixel_features

pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parent.parent


def test_best_threshold_and_kfold_accuracy_on_separable_scores():
    rng = np.random.default_rng(0)
    labels = np.tile([1, 0], 100)
    scores = np.where(labels == 1, rng.uniform(0.6, 1.0, 200), rng.uniform(0.0, 0.4, 200))
    t = best_threshold(scores, labels)
    assert 0.4 < t <= scores[labels == 1].min() + 1e-12
    acc, std = kfold_accuracy(scores, labels, np.repeat(np.arange(10), 20))
    assert acc == 1.0 and std == 0.0
    flipped = kfold_accuracy(-scores, labels, np.repeat(np.arange(10), 20))[0]
    assert flipped <= 0.55                                    # the wrong way round is no better than chance


def test_pixel_features_and_center_crop():
    from PIL import Image
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (250, 250, 3), dtype=np.uint8))
    crop = center_crop(img)
    assert crop.shape == (3, 160, 160) and -1.0 <= crop.min() and crop.max() <= 1.0
    feats = pixel_features(np.stack([crop, crop * 0.5]))
    assert feats.shape == (2, 1024)
    assert np.allclose(np.linalg.norm(feats, axis=1), 1.0, atol=1e-5)
    assert np.allclose(cosine_scores(feats, [0], [1]), 1.0, atol=1e-5)   # brightness change does not matter


def test_frr_at_far_uses_impostor_quantile():
    sys.path.insert(0, str(ROOT / "scripts"))
    import evaluate_face
    labels = np.array([1] * 10 + [0] * 100)
    scores = np.r_[np.linspace(0.5, 1.0, 10), np.linspace(0.0, 0.9, 100)]
    # at FAR 1% only the single highest impostor (0.9) may pass; genuine scores below it are rejected
    assert evaluate_face.frr_at_far(scores, labels, 0.01) == pytest.approx(0.8)


def write_fake_lfw(root: Path, people=8, photos=4, seed=0):
    from PIL import Image
    rng = np.random.default_rng(seed)
    paths = {}
    for p in range(people):
        base = rng.integers(0, 255, (25, 25, 3)).astype(np.float32)
        for k in range(photos):
            img = np.kron(base, np.ones((10, 10, 1))) + rng.normal(0, 20, (250, 250, 3))
            path = root / "data/raw/face/train" / f"person_{p}" / f"person_{p}_{k:04d}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path)
            paths.setdefault(p, []).append(path.relative_to(root).as_posix())
    rows = []
    for fold in range(10):
        p = fold % people
        q = (p + 1) % people
        rows.append((1, fold, paths[p][0], paths[p][1 + fold % (photos - 1)], f"person_{p}", f"person_{p}"))
        rows.append((0, fold, paths[p][0], paths[q][0], f"person_{p}", f"person_{q}"))
    out = root / "data/pairs/face_test.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "fold", "path_a", "path_b", "person_a", "person_b"])
        w.writerows(rows)


def test_evaluate_face_end_to_end_without_weights(tmp_path):
    write_fake_lfw(tmp_path)
    # image paths in the CSV are relative to the project root, so point them at tmp_path
    pairs = tmp_path / "data/pairs/face_test.csv"
    text = pairs.read_text().replace("data/raw/face", str(tmp_path / "data/raw/face"))
    pairs.write_text(text)
    results = tmp_path / "results"
    run = subprocess.run([sys.executable, "scripts/evaluate_face.py", "--data", str(tmp_path / "data"),
                          "--results", str(results), "--weights", "none", "--device", "cpu", "--batch-size", "8"],
                         cwd=ROOT, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    summary = json.loads((results / "face_results.json").read_text())
    assert summary["pairs"] == 20 and summary["face_detection_rate"] == 0.0   # no real faces: central crops
    assert summary["baseline"]["eer"] <= 0.1                                   # distinct patterns are easy
    assert {"facenet", "baseline"} <= set(summary)
    assert "LFW accuracy" in (results / "face_results.md").read_text()
