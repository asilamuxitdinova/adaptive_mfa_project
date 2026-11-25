"""
Stage 2: measure EER / FAR / FRR of the face verifiers on LFW's official 6,000 pairs.

Evaluates data/pairs/face_test.csv (from prepare_data.py: 3,000 same-person and 3,000
different-person pairs in 10 folds) with:
  - the baseline (32x32 grey pixels of the aligned face, no learning)
  - pretrained FaceNet (MTCNN alignment + Inception-ResNet-v1 trained on VGGFace2), not trained here
and compares the results with the face values the paper took from the literature.

Also reports the standard LFW accuracy (10-fold: threshold chosen on 9 folds, tested on the
10th), so the numbers can be checked against published LFW results.

Run from the project root (uses the GPU automatically when available). The first run downloads
the FaceNet weights (~110 MB) into the torch cache (~/.cache/torch/checkpoints):
    python scripts/evaluate_face.py

Outputs:
    results/face_results.json    all numbers
    results/face_results.md      readable summary table
    results/face_scores.png      same-person vs different-person score distributions
"""
import argparse
import csv
import json
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.metrics import compute_eer, far_frr  # noqa: E402
from src.verifiers.face import FaceEmbedder, cosine_scores, kfold_accuracy  # noqa: E402

# Face values in the paper's Table I (from earlier literature)
PAPER = {"far": 0.01, "frr": 0.20, "lip_eer": 0.068, "texture_eer": 0.10, "brow_cheek_eer": 0.15}


def show(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def read_pairs(path: Path):
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return ([r["path_a"] for r in rows], [r["path_b"] for r in rows],
            np.array([int(r["label"]) for r in rows]), np.array([int(r["fold"]) for r in rows]))


def frr_at_far(scores, labels, far):
    """FRR when the threshold lets in only `far` of impostors (the paper's FAR of 1%)."""
    impostor = np.sort(scores[labels == 0])[::-1]
    threshold = np.nextafter(impostor[int(len(impostor) * far)], np.inf)   # strictly above that impostor
    return far_frr(scores, labels, threshold)[1]


def far_at_frr(scores, labels, frr):
    genuine = np.sort(scores[labels == 1])
    return far_frr(scores, labels, genuine[int(len(genuine) * frr)])[0]


def evaluate(name, scores, labels, folds):
    res = compute_eer(scores, labels)
    acc, acc_std = kfold_accuracy(scores, labels, folds)
    out = {"eer": res.eer, "far": res.far, "frr": res.frr, "threshold": res.threshold,
           "lfw_accuracy": acc, "lfw_accuracy_std": acc_std,
           "frr_at_paper_far": frr_at_far(scores, labels, PAPER["far"]),
           "far_at_paper_frr": far_at_frr(scores, labels, PAPER["frr"]),
           "genuine_pairs": res.n_genuine, "impostor_pairs": res.n_impostor}
    print(f"  {name:<10} EER {res.eer:.2%} | LFW accuracy {acc:.2%} ± {acc_std:.2%} | "
          f"FRR at FAR 1%: {out['frr_at_paper_far']:.2%}")
    return out, res


def plot(path, curves, labels):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed: skipping the score plot)")
        return
    fig, axes = plt.subplots(1, len(curves), figsize=(6 * len(curves), 4))
    for ax, (name, (scores, res)) in zip(np.atleast_1d(axes), curves.items()):
        bins = np.linspace(scores.min(), scores.max(), 61) if np.ptp(scores) > 1e-6 else 10
        ax.hist(scores[labels == 1], bins=bins, alpha=0.6, density=True, label="same person")
        ax.hist(scores[labels == 0], bins=bins, alpha=0.6, density=True, label="different people")
        ax.axvline(res.threshold, color="k", linestyle="--", linewidth=1, label=f"EER threshold ({res.eer:.1%})")
        ax.set_title(name)
        ax.set_xlabel("cosine similarity")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"  wrote {show(path)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--weights", default="vggface2", choices=("vggface2", "casia-webface", "none"),
                        help="FaceNet weights (none = random, for tests only)")
    parser.add_argument("--selection", default="center_weighted_size",
                        choices=("center_weighted_size", "largest", "probability", "largest_over_threshold"),
                        help="which face to use when a photo shows several people (run 1 used 'largest')")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    import torch
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    path_a, path_b, labels, folds = read_pairs(args.data / "pairs" / "face_test.csv")
    images = sorted(set(path_a) | set(path_b))
    row = {p: i for i, p in enumerate(images)}
    rows_a, rows_b = np.array([row[p] for p in path_a]), np.array([row[p] for p in path_b])
    print(f"device {device} | {len(labels):,} pairs ({labels.sum():,} same person) | "
          f"{len(images):,} distinct images | {len(np.unique(folds))} folds")

    t0 = time.time()
    embedder = FaceEmbedder(device, pretrained=None if args.weights == "none" else args.weights,
                            selection=args.selection)
    emb = embedder.embed_files([ROOT / p if not Path(p).is_absolute() else Path(p) for p in images],
                               batch_size=args.batch_size)
    seconds = time.time() - t0
    detection_rate = float(emb.detected.mean())
    print(f"  embedded {len(images):,} images in {seconds:.0f}s; face found in {detection_rate:.2%}")

    results, curves = {}, {}
    names = {"baseline": "Baseline (32x32 pixels, no learning)",
             "facenet": f"FaceNet, pretrained on {args.weights.upper() if args.weights != 'none' else 'nothing'}"}
    for key, vectors in (("baseline", emb.pixels), ("facenet", emb.embeddings)):
        scores = cosine_scores(vectors, rows_a, rows_b)
        results[key], res = evaluate(key, scores, labels, folds)
        curves[names[key]] = (scores, res)

    both_found = emb.detected[rows_a] & emb.detected[rows_b]
    args.results.mkdir(parents=True, exist_ok=True)
    summary = {"date": str(date.today()), "pairs": int(len(labels)), "images": len(images),
               "weights": args.weights, "face_selection": args.selection, "device": device, "embedding_seconds": round(seconds, 1),
               "face_detection_rate": detection_rate, "pairs_with_both_faces_found": int(both_found.sum()),
               "paper_table_i": PAPER, **results}
    (args.results / "face_results.json").write_text(json.dumps(summary, indent=2))

    lines = ["# Face verification results (Stage 2)", "",
             f"LFW's official {len(labels):,} pairs ({labels.sum():,} same person, {len(labels) - labels.sum():,} "
             f"different people, 10 folds), {len(images):,} distinct images. MTCNN found a face in "
             f"{detection_rate:.2%} of the images (the central crop was used for the rest); when a photo showed "
             f"several faces, the `{args.selection}` one was used. "
             f"Generated {summary['date']} by `scripts/evaluate_face.py`.", "",
             "| Verifier | EER | LFW accuracy (10-fold) | FRR when FAR = 1% | FAR when FRR = 20% |",
             "|---|---|---|---|---|"]
    for key, r in results.items():
        lines.append(f"| {names[key]} | {r['eer']:.2%} | {r['lfw_accuracy']:.2%} ± {r['lfw_accuracy_std']:.2%} | "
                     f"{r['frr_at_paper_far']:.2%} | {r['far_at_paper_frr']:.2%} |")
    lines += [f"| *Paper, Table I (from literature)* | *lip {PAPER['lip_eer']:.1%}, texture {PAPER['texture_eer']:.0%}, "
              f"brow/cheek {PAPER['brow_cheek_eer']:.0%}* | | *FRR {PAPER['frr']:.0%} at FAR {PAPER['far']:.0%}* | |", "",
              "The paper's FAR 1% / FRR 20% is one operating point for the whole face modality, so the last two "
              "columns read our score distribution at that point from both sides. The paper's values come from "
              "earlier studies with different data and features (lip, eye, brow, texture), so this shows how far "
              "a modern pretrained model is from the assumed values, not a like-for-like benchmark.", "",
              "Note: FaceNet's VGGFace2 training set contains celebrities, and some LFW people may appear in it; "
              "published LFW results share this limitation."]
    (args.results / "face_results.md").write_text("\n".join(lines) + "\n")
    print(f"  wrote {show(args.results / 'face_results.md')} and face_results.json")
    plot(args.results / "face_scores.png", curves, labels)
    return 0


if __name__ == "__main__":
    sys.exit(main())
