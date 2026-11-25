"""
Stage 2: measure EER / FAR / FRR of the keystroke verifiers on the held-out test participants.

Evaluates data/pairs/keystroke_test.csv (participants never seen in training) with:
  - the baseline (sentence timing statistics, no training)
  - the trained TypeNet-style model from scripts/train_keystroke.py (if the model file exists)
and compares the results with the keystroke values the paper took from the literature.

Run from the project root:
    python scripts/evaluate_keystroke.py

Outputs:
    results/keystroke_results.json    all numbers
    results/keystroke_results.md      readable summary table
    results/keystroke_scores.png      genuine vs impostor score distributions (if matplotlib is installed)
"""
import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.keystroke_data import DEFAULT_MAX_LEN, Standardizer, load_sequences  # noqa: E402
from src.metrics import compute_eer, far_frr  # noqa: E402
from src.verifiers.keystroke import (baseline_features, baseline_scores, build_model, embed,  # noqa: E402
                                     embedding_scores)

# Keystroke values in the paper's Table I (collected from earlier literature)
PAPER = {"far": 0.07, "frr": 0.001, "crossover": 0.018}


def show(path: Path) -> str:
    """Path relative to the project when possible, for readable messages."""
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def read_pairs(path: Path):
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return ([r["section_a"] for r in rows], [r["section_b"] for r in rows],
            np.array([int(r["label"]) for r in rows]))


def summarize(name, scores, labels):
    res = compute_eer(scores, labels)
    print(f"  {name:<10} {res}")
    return {"eer": res.eer, "far": res.far, "frr": res.frr, "threshold": res.threshold,
            "genuine_pairs": res.n_genuine, "impostor_pairs": res.n_impostor}, res


def plot(path, curves, labels):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed: skipping the score plot)")
        return
    fig, axes = plt.subplots(1, len(curves), figsize=(6 * len(curves), 4))
    axes = np.atleast_1d(axes)
    for ax, (name, (scores, res)) in zip(axes, curves.items()):
        scores = scores.astype(np.float64)
        spread = float(np.ptp(scores))
        if spread < 1e-6:            # (almost) identical scores: the model cannot tell people apart
            print(f"  warning: {name}: all scores are nearly identical (spread {spread:.2e}); "
                  "the embeddings have collapsed")
            ax.text(0.5, 0.5, "scores nearly identical\n(embeddings collapsed)", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(name)
            continue
        bins = np.linspace(scores.min(), scores.max(), 61)
        ax.hist(scores[labels == 1], bins=bins, alpha=0.6, density=True, label="same person")
        ax.hist(scores[labels == 0], bins=bins, alpha=0.6, density=True, label="different people")
        ax.axvline(res.threshold, color="k", linestyle="--", linewidth=1, label=f"EER threshold ({res.eer:.1%})")
        ax.set_title(name)
        ax.set_xlabel("similarity score")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"  wrote {show(path)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "keystroke_typenet.pt")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    sec_a, sec_b, labels = read_pairs(args.data / "pairs" / "keystroke_test.csv")
    checkpoint = None
    if args.model.exists():
        import torch
        device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
        checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    max_len = checkpoint["max_len"] if checkpoint else DEFAULT_MAX_LEN

    data = load_sequences(args.data / "raw" / "keystroke" / "data", "test", sections=set(sec_a) | set(sec_b), max_len=max_len)
    if checkpoint:
        data = Standardizer.from_dict(checkpoint["standardizer"]).transform(data)   # training statistics only
    else:
        data = Standardizer.fit(data).transform(data)
    rows_a, rows_b = data.rows(sec_a), data.rows(sec_b)
    print(f"{len(labels):,} test pairs from {len(set(data.participants)):,} unseen participants")

    results, curves = {}, {}
    scores = baseline_scores(baseline_features(data), rows_a, rows_b)
    results["baseline"], res = summarize("baseline", scores, labels)
    curves["Baseline (timing statistics, no training)"] = (scores, res)

    if checkpoint:
        model = build_model(**checkpoint["config"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        scores = embedding_scores(embed(model, data, device), rows_a, rows_b)
        results["typenet"], res = summarize("TypeNet", scores, labels)
        results["typenet"].update({"trained_epochs": checkpoint["epoch"], "val_eer": checkpoint["val_eer"]})
        curves["TypeNet-style LSTM (trained)"] = (scores, res)
    else:
        print(f"  (no trained model at {show(args.model)}: run scripts/train_keystroke.py first)")

    # FAR when the threshold is set so that only 0.1% of genuine users are rejected (the paper's FRR)
    for name, (scores, res) in zip(results, curves.values()):
        genuine = np.sort(scores[labels == 1])
        threshold = genuine[int(len(genuine) * PAPER["frr"])]
        results[name]["far_at_paper_frr"], _ = far_frr(scores, labels, threshold)

    args.results.mkdir(parents=True, exist_ok=True)
    summary = {"date": str(date.today()), "test_pairs": int(len(labels)),
               "test_participants": int(len(set(data.participants))), "paper_table_i": PAPER, **results}
    (args.results / "keystroke_results.json").write_text(json.dumps(summary, indent=2))

    lines = ["# Keystroke verification results (Stage 2)", "",
             f"{len(labels):,} test pairs from {summary['test_participants']:,} participants never seen in training "
             f"(Aalto dataset). Generated {summary['date']} by `scripts/evaluate_keystroke.py`.", "",
             "| Verifier | EER | FAR at EER | FRR at EER | FAR when FRR = 0.1% |", "|---|---|---|---|---|"]
    names = {"baseline": "Baseline (no training)", "typenet": "TypeNet-style LSTM (trained)"}
    for key, r in results.items():
        lines.append(f"| {names[key]} | {r['eer']:.2%} | {r['far']:.2%} | {r['frr']:.2%} | {r['far_at_paper_frr']:.2%} |")
    lines += [f"| *Paper, Table I (from literature)* | *crossover {PAPER['crossover']:.1%}* | | | "
              f"*FAR {PAPER['far']:.0%} at FRR {PAPER['frr']:.1%}* |", "",
              "Each test pair compares two single typed sentences. The paper's values come from earlier studies "
              "with different data and protocols, so the comparison shows how far real measurements are from "
              "the assumed values, not a like-for-like benchmark."]
    (args.results / "keystroke_results.md").write_text("\n".join(lines) + "\n")
    print(f"  wrote {show(args.results / 'keystroke_results.md')} and keystroke_results.json")
    plot(args.results / "keystroke_scores.png", curves, labels)
    return 0


if __name__ == "__main__":
    sys.exit(main())
