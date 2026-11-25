"""
Stage 2: train the TypeNet-style keystroke verifier on the Aalto training participants.

Uses the participants listed in data/pairs/keystroke_train.csv (from prepare_data.py), or with
--all-participants every participant in the training split, and all of their typed sentences. A small group of them is held out for validation: after every epoch
the validation EER is measured, and the best model is kept (with early stopping).

Run from the project root (uses the GPU automatically when available):
    python scripts/train_keystroke.py
    python scripts/train_keystroke.py --epochs 5 --limit-users 500     # quick check
    python scripts/train_keystroke.py --all-participants --epochs 200 --patience 15   # largest run

Outputs:
    models/keystroke_typenet.pt              best model + settings (gitignored)
    results/keystroke_training_log.csv       loss and validation EER per epoch
"""
import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.keystroke_data import DEFAULT_MAX_LEN, N_FEATURES, Standardizer, load_sequences, pk_batches  # noqa: E402
from src.metrics import compute_eer  # noqa: E402
from src.pairs import make_pairs, split_ids  # noqa: E402
from src.verifiers.keystroke import (batch_hard_triplet_loss, build_model, embed, embedding_scores,  # noqa: E402
                                     embedding_spread, supervised_contrastive_loss)


def read_participants(pairs_csv: Path):
    with pairs_csv.open(encoding="utf-8") as f:
        return sorted({r[k] for r in csv.DictReader(f) for k in ("participant_a", "participant_b")})


def participants_in_split(parquet_dir: Path, split: str):
    """All participant ids in the '<split>-*.parquet' files (reads only that one column)."""
    import pyarrow.parquet as pq

    ids = set()
    for f in sorted(Path(parquet_dir).glob(f"{split}-*.parquet")):
        ids.update(str(p) for p in pq.read_table(f, columns=["participant_id"]).column(0).to_pylist())
    return sorted(ids)


def subset(data, rows):
    """A KeystrokeSet with only the given rows (indices re-numbered from 0)."""
    sections = data.sections[rows]
    return type(data)(data.x[rows], data.lengths[rows], data.participants[rows], sections,
                      {s: i for i, s in enumerate(sections)})


def validation_pairs(groups, rng):
    """Row-index pairs for validation participants (5 genuine + 5 impostor per person)."""
    pairs = make_pairs(groups, rng, max_genuine_per_id=5)
    return np.array([a for a, *_ in pairs]), np.array([b for _, b, *_ in pairs]), np.array([lab for _, _, lab, *_ in pairs])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data", help="data folder from prepare_data.py")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "keystroke_typenet.pt")
    parser.add_argument("--log", type=Path, default=ROOT / "results" / "keystroke_training_log.csv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8, help="stop after this many epochs without improvement")
    parser.add_argument("--participants-per-batch", type=int, default=64)
    parser.add_argument("--sentences-per-participant", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss", choices=("supcon", "triplet"), default="supcon",
                        help="supcon = supervised contrastive (default, stable); triplet = batch-hard triplet (TypeNet)")
    parser.add_argument("--temperature", type=float, default=0.1, help="supervised contrastive temperature")
    parser.add_argument("--margin", type=float, default=0.2, help="triplet loss margin")
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN, help="keys per sentence")
    parser.add_argument("--val-users", type=int, default=500, help="training participants held out for validation")
    parser.add_argument("--all-participants", action="store_true",
                        help="train on every participant in the training split (~78,000) instead of the "
                             "10,000 listed in keystroke_train.csv; the test participants are unaffected")
    parser.add_argument("--limit-users", type=int, default=0, help="use only this many participants (0 = all)")
    parser.add_argument("--device", default="auto", help="auto, cuda or cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    parquet_dir = args.data / "raw" / "keystroke" / "data"
    participants = (participants_in_split(parquet_dir, "train") if args.all_participants
                    else read_participants(args.data / "pairs" / "keystroke_train.csv"))
    if args.limit_users:
        participants = sorted(rng.sample(participants, min(args.limit_users, len(participants))))
    n_val = max(2, min(args.val_users, len(participants) // 5))      # validation pairs need 2+ people
    if len(participants) < n_val + 2:
        raise SystemExit(f"need at least {n_val + 2} participants, found {len(participants)}")
    val_fraction = n_val / len(participants)
    train_ids, val_ids = split_ids(participants, val_fraction, rng)
    print(f"device {device} | {len(train_ids)} training + {len(val_ids)} validation participants")

    t0 = time.time()
    data = load_sequences(parquet_dir, "train", participants=participants, max_len=args.max_len)
    print(f"loaded {len(data.x):,} sentences in {time.time() - t0:.0f}s")

    train_mask = np.isin(data.participants, train_ids)
    scaler = Standardizer.fit(subset(data, np.flatnonzero(train_mask)))
    data = scaler.transform(data)

    groups = data.by_participant()
    train_groups = {p: groups[p] for p in train_ids if p in groups}
    val_data = subset(data, np.flatnonzero(np.isin(data.participants, val_ids)))   # only these are embedded per epoch
    val_a, val_b, val_y = validation_pairs(val_data.by_participant(), rng)
    label_of = {p: i for i, p in enumerate(sorted(groups))}
    labels_all = np.array([label_of[p] for p in data.participants])

    model = build_model(N_FEATURES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    x_all, len_all = torch.from_numpy(data.x), torch.from_numpy(data.lengths)

    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    best = {"eer": float("inf"), "epoch": 0}
    with args.log.open("w", newline="", encoding="utf-8") as log_file:
        log = csv.writer(log_file)
        log.writerow(["epoch", "train_loss", "val_eer", "embedding_spread", "seconds"])
        for epoch in range(1, args.epochs + 1):
            model.train()
            t0, losses = time.time(), []
            for batch in pk_batches(train_groups, args.participants_per_batch, args.sentences_per_participant, rng):
                idx = torch.tensor(batch)
                emb = model(x_all[idx].to(device), len_all[idx])
                labels = torch.from_numpy(labels_all[batch]).to(device)
                loss = (supervised_contrastive_loss(emb, labels, args.temperature) if args.loss == "supcon"
                        else batch_hard_triplet_loss(emb, labels, args.margin))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            emb = embed(model, val_data, device)
            val = compute_eer(embedding_scores(emb, val_a, val_b), val_y)
            spread = embedding_spread(emb)
            seconds = time.time() - t0
            log.writerow([epoch, f"{np.mean(losses):.4f}", f"{val.eer:.4f}", f"{spread:.4f}", f"{seconds:.0f}"])
            log_file.flush()
            improved = val.eer < best["eer"]
            print(f"epoch {epoch:3d} | loss {np.mean(losses):.4f} | val EER {val.eer:.2%} | spread {spread:.4f} | {seconds:.0f}s"
                  f"{'  <- best' if improved else ''}", flush=True)
            if spread < 1e-3:
                print("  warning: embeddings have collapsed (almost identical for everyone); "
                      "try --loss supcon or a lower --lr", flush=True)
            if improved:
                best = {"eer": val.eer, "epoch": epoch}
                torch.save({"state_dict": model.state_dict(), "config": model.config, "max_len": args.max_len,
                            "standardizer": scaler.to_dict(), "val_eer": val.eer, "epoch": epoch,
                            "train_participants": len(train_ids), "args": {k: str(v) for k, v in vars(args).items()}},
                           args.out)
            elif epoch - best["epoch"] >= args.patience:
                print(f"no improvement for {args.patience} epochs, stopping")
                break

    print(f"\nbest validation EER {best['eer']:.2%} at epoch {best['epoch']}; saved {args.out}")
    print(json.dumps({"best_val_eer": round(best["eer"], 4), "best_epoch": best["epoch"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
