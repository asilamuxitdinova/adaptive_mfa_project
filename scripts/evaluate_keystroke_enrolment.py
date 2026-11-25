"""
Stage 2: keystroke verification with enrolment (several sentences per user), as real systems work.

scripts/evaluate_keystroke.py compares one typed sentence with one other sentence. A real
system first records several sentences from the user (enrolment) and compares each new
sentence with all of them. This script measures how the error drops as the user enrols more.

Protocol (similar to TypeNet, Acien et al. 2021), on the same 2,000 test participants as
evaluate_keystroke.py (never seen in training):
  - Each participant's sentences are shuffled once. The last `--queries` (default 5) are
    their login attempts; the first G of the rest are their enrolment sentences.
  - Genuine score: one of the user's own login sentences vs their enrolment.
  - Impostor score: a login sentence from another test participant vs this user's enrolment
    (one from each of `--queries` different people).
  - Score = average similarity between the login sentence and each enrolment sentence.
  - The same login sentences are used for every G, so only the amount of enrolment changes.
  - Only participants with at least max(G) + queries sentences are used (Aalto: 15 each).

Run from the project root, after training:
    python scripts/evaluate_keystroke_enrolment.py
    python scripts/evaluate_keystroke_enrolment.py --enrol 1 2 3 5 7 10

Outputs:
    results/keystroke_enrolment.json   all numbers
    results/keystroke_enrolment.md     readable summary table
    results/keystroke_enrolment.png    EER vs number of enrolment sentences (if matplotlib is installed)
"""
import argparse
import csv
import json
import random
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.keystroke_data import Standardizer, load_sequences  # noqa: E402
from src.metrics import compute_eer, far_frr  # noqa: E402
from src.verifiers.keystroke import baseline_features, build_model, embed  # noqa: E402

PAPER = {"far": 0.07, "frr": 0.001, "crossover": 0.018}   # keystroke values in the paper's Table I


def show(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def test_participants(pairs_csv: Path):
    with pairs_csv.open(encoding="utf-8") as f:
        return sorted({r[k] for r in csv.DictReader(f) for k in ("participant_a", "participant_b")})


def build_protocol(groups: dict, max_enrol: int, n_queries: int, rng: random.Random):
    """
    Fix enrolment candidates and login attempts for every eligible participant.

    Returns (users, enrol, queries, impostor_queries):
        users             eligible participant ids
        enrol             (U, max_enrol) rows; the first G columns are the enrolment for G sentences
        queries           (U, n_queries) the user's own login sentences
        impostor_queries  (U, n_queries) login sentences of n_queries different other users
    """
    users = sorted(p for p, rows in groups.items() if len(rows) >= max_enrol + n_queries)
    if len(users) <= n_queries:
        raise ValueError(f"need more than {n_queries} participants with at least {max_enrol + n_queries} "
                         f"sentences, found {len(users)}")
    enrol, queries = [], []
    for p in users:
        rows = list(groups[p])
        rng.shuffle(rows)
        queries.append(rows[-n_queries:])
        enrol.append(rows[:max_enrol])
    queries = np.array(queries)
    impostor = []
    for i in range(len(users)):
        others = rng.sample([j for j in range(len(users)) if j != i], n_queries)
        impostor.append([queries[j, rng.randrange(n_queries)] for j in others])
    return users, np.array(enrol), queries, np.array(impostor)


def enrolment_scores(similarity, enrol, queries, impostor, g):
    """
    Genuine and impostor scores when each user has enrolled their first g sentences.
    `similarity(q_rows, e_rows)` returns the similarity of each q row with each of its e rows: (N, g).
    """
    e = enrol[:, :g]                                           # (U, g)
    n_q = queries.shape[1]
    e_rep = np.repeat(e, n_q, axis=0)                          # one enrolment per login attempt
    genuine = similarity(queries.reshape(-1), e_rep).mean(axis=1)
    impostor_s = similarity(impostor.reshape(-1), e_rep).mean(axis=1)
    scores = np.concatenate([genuine, impostor_s])
    labels = np.concatenate([np.ones(len(genuine), int), np.zeros(len(impostor_s), int)])
    return scores, labels


def cosine_similarity(embeddings):
    return lambda q, e: np.einsum("nd,ngd->ng", embeddings[q], embeddings[e])


def baseline_similarity(features):
    scale = features.std(axis=0)
    scale[scale < 1e-6] = 1.0
    f = features / scale
    return lambda q, e: -np.abs(f[q][:, None, :] - f[e]).sum(axis=2)


def far_at_frr(scores, labels, frr):
    genuine = np.sort(scores[labels == 1])
    return far_frr(scores, labels, genuine[int(len(genuine) * frr)])[0]


def plot(path, table, enrol_sizes):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed: skipping the plot)")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, rows in table.items():
        ax.plot(enrol_sizes, [100 * rows[str(g)]["eer"] for g in enrol_sizes], marker="o", label=name)
    ax.axhline(100 * PAPER["crossover"], color="k", linestyle="--", linewidth=1, label="paper, Table I (1.8%)")
    ax.set_xlabel("enrolment sentences per user")
    ax.set_ylabel("EER (%)")
    ax.set_xticks(enrol_sizes)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"  wrote {show(path)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "keystroke_typenet.pt")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--enrol", type=int, nargs="+", default=[1, 2, 5, 7, 10],
                        help="numbers of enrolment sentences to evaluate")
    parser.add_argument("--queries", type=int, default=5, help="login attempts per user (genuine and impostor each)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"no trained model at {show(args.model)}: run scripts/train_keystroke.py first")
    import torch
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)

    enrol_sizes = sorted(set(args.enrol))
    participants = test_participants(args.data / "pairs" / "keystroke_test.csv")
    data = load_sequences(args.data / "raw" / "keystroke" / "data", "test", participants=participants,
                          max_len=checkpoint["max_len"])
    data = Standardizer.from_dict(checkpoint["standardizer"]).transform(data)   # training statistics only
    users, enrol, queries, impostor = build_protocol(data.by_participant(), max(enrol_sizes), args.queries,
                                                     random.Random(args.seed))
    print(f"{len(users):,} of {len(participants):,} test participants have at least "
          f"{max(enrol_sizes) + args.queries} sentences; {len(users) * args.queries:,} genuine + "
          f"{len(users) * args.queries:,} impostor login attempts per setting")

    model = build_model(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    verifiers = {"TypeNet-style LSTM (trained)": cosine_similarity(embed(model, data, device)),
                 "Baseline (no training)": baseline_similarity(baseline_features(data))}

    table = {}
    for name, similarity in verifiers.items():
        table[name] = {}
        for g in enrol_sizes:
            scores, labels = enrolment_scores(similarity, enrol, queries, impostor, g)
            res = compute_eer(scores, labels)
            table[name][str(g)] = {"eer": res.eer, "far": res.far, "frr": res.frr, "threshold": res.threshold,
                                   "far_at_paper_frr": far_at_frr(scores, labels, PAPER["frr"])}
            print(f"  {name:<30} G={g:<3} EER {res.eer:.2%} | FAR when FRR = 0.1%: "
                  f"{table[name][str(g)]['far_at_paper_frr']:.2%}")

    args.results.mkdir(parents=True, exist_ok=True)
    summary = {"date": str(date.today()), "test_participants": len(users), "queries_per_user": args.queries,
               "genuine_attempts": len(users) * args.queries, "impostor_attempts": len(users) * args.queries,
               "model_epoch": checkpoint["epoch"], "model_val_eer": checkpoint["val_eer"],
               "paper_table_i": PAPER, "results": table}
    (args.results / "keystroke_enrolment.json").write_text(json.dumps(summary, indent=2))

    head = "| Verifier | " + " | ".join(f"G = {g}" for g in enrol_sizes) + " |"
    sep = "|---|" + "---|" * len(enrol_sizes)
    lines = ["# Keystroke verification with enrolment (Stage 2)", "",
             f"{len(users):,} test participants never seen in training (Aalto dataset); each makes "
             f"{args.queries} genuine login attempts and receives {args.queries} impostor attempts from other "
             f"participants ({summary['genuine_attempts']:,} + {summary['impostor_attempts']:,} per setting). "
             f"G = number of enrolment sentences per user. Generated {summary['date']} by "
             "`scripts/evaluate_keystroke_enrolment.py`.", "", "**EER**", "", head, sep]
    lines += [f"| {n} | " + " | ".join(f"{r[str(g)]['eer']:.2%}" for g in enrol_sizes) + " |" for n, r in table.items()]
    lines += ["", "**FAR when FRR = 0.1%** (the paper's operating point)", "", head, sep]
    lines += [f"| {n} | " + " | ".join(f"{r[str(g)]['far_at_paper_frr']:.2%}" for g in enrol_sizes) + " |"
              for n, r in table.items()]
    lines += ["", f"Paper, Table I (from literature): crossover {PAPER['crossover']:.1%}, FAR {PAPER['far']:.0%} "
              f"at FRR {PAPER['frr']:.1%}.", "",
              "Score = average similarity between the login sentence and each enrolment sentence. The same login "
              "sentences are used for every G, so the columns differ only in the amount of enrolment."]
    (args.results / "keystroke_enrolment.md").write_text("\n".join(lines) + "\n")
    print(f"  wrote {show(args.results / 'keystroke_enrolment.md')} and keystroke_enrolment.json")
    plot(args.results / "keystroke_enrolment.png", table, enrol_sizes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
