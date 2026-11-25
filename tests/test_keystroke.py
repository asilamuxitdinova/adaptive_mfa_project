"""
Keystroke verifier tests on synthetic typists: each fake participant has their own typing
rhythm, so a working verifier must separate them. No real data is needed.
"""
import csv
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.keystroke_data import N_FEATURES, Standardizer, load_sequences, pk_batches
from src.pairs import make_pairs
from src.verifiers.keystroke import baseline_features, baseline_scores, build_model, embedding_scores

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parent.parent


def fake_participants(ids, sentences=8, seed=0):
    """Rows like the Aalto parquet: each participant types with their own speed and rhythm."""
    rng = np.random.default_rng(seed)
    rows = {k: [] for k in ("participant_id", "section_id", "keycode_ids", "hl", "il", "pl", "rl")}
    for p in ids:
        hold, gap = rng.uniform(0.05, 0.25), rng.uniform(0.05, 0.4)
        for s in range(sentences):
            n = int(rng.integers(20, 60))
            rows["participant_id"].append(p)
            rows["section_id"].append(p * 1000 + s)
            rows["keycode_ids"].append(rng.integers(32, 127, n).tolist())
            hl = rng.normal(hold, 0.02, n)
            il = rng.normal(gap, 0.03, n)
            il[0] = np.nan                              # first key has no previous key
            rows["hl"].append(hl.tolist())
            rows["il"].append(il.tolist())
            rows["pl"].append((hl + il).tolist())
            rows["rl"].append((hl + il).tolist())
    return pa.table(rows)


def write_dataset(root: Path, train_ids, test_ids):
    raw = root / "data/raw/keystroke/data"
    raw.mkdir(parents=True)
    pq.write_table(fake_participants(train_ids, seed=1), raw / "train-00000-of-00001.parquet")
    pq.write_table(fake_participants(test_ids, seed=2), raw / "test-00000-of-00001.parquet")
    for split, ids in (("train", train_ids), ("test", test_ids)):
        groups = {str(p): [str(p * 1000 + s) for s in range(8)] for p in ids}
        pairs = make_pairs(groups, random.Random(0), max_genuine_per_id=5)
        path = root / f"data/pairs/keystroke_{split}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["label", "section_a", "section_b", "participant_a", "participant_b"])
            w.writerows([(lab, a, b, ia, ib) for a, b, lab, ia, ib in pairs])
    return raw


def test_load_sequences_filters_pads_and_cleans(tmp_path):
    raw = write_dataset(tmp_path, train_ids=[1, 2, 3], test_ids=[4, 5])
    data = load_sequences(raw, "train", participants=["1", "3"], max_len=30)
    assert set(data.participants) == {"1", "3"} and len(data.x) == 16
    assert data.x.shape == (16, 30, N_FEATURES)
    assert data.lengths.max() <= 30                      # long sentences truncated
    assert np.isfinite(data.x).all()                     # NaN (first key) replaced
    assert 0 <= data.x[..., 0].min() and data.x[..., 0].max() <= 1   # keycodes scaled
    only = load_sequences(raw, "train", sections=["2001", "2005"])
    assert sorted(only.sections) == ["2001", "2005"]


def test_standardizer_keeps_padding_zero(tmp_path):
    data = load_sequences(write_dataset(tmp_path, [1, 2, 3], [4, 5]), "train")
    scaled = Standardizer.fit(data).transform(data)
    pad = np.arange(scaled.x.shape[1])[None, :] >= scaled.lengths[:, None]
    assert (scaled.x[pad] == 0).all()
    back = Standardizer.from_dict(Standardizer.fit(data).to_dict())
    assert np.allclose(back.mean, Standardizer.fit(data).mean)


def test_pk_batches_shape():
    groups = {f"p{i}": list(range(i * 10, i * 10 + 6)) for i in range(10)}
    batches = list(pk_batches(groups, p=4, k=3, rng=random.Random(0)))
    assert len(batches) == 2 and all(len(b) == 12 for b in batches)


def test_baseline_separates_distinct_typists(tmp_path):
    data = load_sequences(write_dataset(tmp_path, list(range(1, 11)), [98, 99]), "train")
    groups = data.by_participant()
    pairs = make_pairs(groups, random.Random(0), max_genuine_per_id=5)
    a, b, y = (np.array(c) for c in zip(*[(p[0], p[1], p[2]) for p in pairs]))
    from src.metrics import compute_eer
    assert compute_eer(baseline_scores(baseline_features(data), a, b), y).eer < 0.25


def test_model_outputs_unit_embeddings():
    model = build_model(N_FEATURES, hidden=16, embedding=8)
    x = torch.randn(4, 12, N_FEATURES)
    out = model(x, torch.tensor([12, 5, 3, 1]))
    assert out.shape == (4, 8)
    assert torch.allclose(out.norm(dim=1), torch.ones(4), atol=1e-5)
    e = out.detach().numpy()
    assert np.allclose(embedding_scores(e, [0], [0]), 1.0, atol=1e-5)


def test_train_and_evaluate_scripts_end_to_end(tmp_path):
    write_dataset(tmp_path, train_ids=list(range(1, 41)), test_ids=list(range(100, 112)))
    model, results = tmp_path / "models/k.pt", tmp_path / "results"
    run = lambda *a: subprocess.run([sys.executable, *a], cwd=ROOT, capture_output=True, text=True)  # noqa: E731
    train = run("scripts/train_keystroke.py", "--data", str(tmp_path / "data"), "--out", str(model),
                "--log", str(results / "log.csv"), "--epochs", "8", "--participants-per-batch", "8",
                "--sentences-per-participant", "4", "--val-users", "8", "--device", "cpu")
    assert train.returncode == 0, train.stderr
    assert model.exists() and "best validation EER" in train.stdout

    ev = run("scripts/evaluate_keystroke.py", "--data", str(tmp_path / "data"), "--model", str(model),
             "--results", str(results), "--device", "cpu")
    assert ev.returncode == 0, ev.stderr
    import json
    summary = json.loads((results / "keystroke_results.json").read_text())
    assert summary["test_participants"] == 12
    assert summary["typenet"]["eer"] < 0.4               # learned something on unseen typists
    assert (results / "keystroke_results.md").exists()


def test_supervised_contrastive_loss_prefers_separated_people():
    from src.verifiers.keystroke import supervised_contrastive_loss
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    separated = torch.nn.functional.normalize(torch.eye(6, 3)[[0, 0, 1, 1, 2, 2]] + 1e-3 * torch.randn(6, 3), dim=1)
    collapsed = torch.nn.functional.normalize(torch.ones(6, 3), dim=1)
    good, bad = supervised_contrastive_loss(separated, labels), supervised_contrastive_loss(collapsed, labels)
    assert good < bad
    assert bad == pytest.approx(np.log(5), abs=1e-4)      # all similarities equal: 1 positive among 5 others


def test_embedding_spread_detects_collapse():
    from src.verifiers.keystroke import embedding_spread
    assert embedding_spread(np.ones((10, 8), dtype=np.float32)) == 0.0
    assert embedding_spread(np.random.default_rng(0).normal(size=(10, 8))) > 0.5


def test_training_log_records_spread_and_triplet_option_still_runs(tmp_path):
    write_dataset(tmp_path, train_ids=list(range(1, 21)), test_ids=[100, 101])
    result = subprocess.run(
        [sys.executable, "scripts/train_keystroke.py", "--data", str(tmp_path / "data"), "--out", str(tmp_path / "m.pt"),
         "--log", str(tmp_path / "log.csv"), "--epochs", "2", "--participants-per-batch", "4",
         "--sentences-per-participant", "4", "--val-users", "4", "--device", "cpu", "--loss", "triplet"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    header = (tmp_path / "log.csv").read_text().splitlines()[0]
    assert header == "epoch,train_loss,val_eer,embedding_spread,seconds"


def test_all_participants_reads_training_split_not_pair_list(tmp_path):
    write_dataset(tmp_path, train_ids=list(range(1, 31)), test_ids=[100, 101])
    pairs = tmp_path / "data/pairs/keystroke_train.csv"            # pair list mentions only some participants
    rows = pairs.read_text().splitlines()
    keep = [rows[0]] + [r for r in rows[1:] if r.split(",")[3] in {"1", "2", "3", "4", "5"}
                        and r.split(",")[4] in {"1", "2", "3", "4", "5"}]
    pairs.write_text("\n".join(keep) + "\n")
    common = ["scripts/train_keystroke.py", "--data", str(tmp_path / "data"), "--epochs", "1",
              "--participants-per-batch", "4", "--sentences-per-participant", "4", "--val-users", "4", "--device", "cpu"]
    listed = subprocess.run([sys.executable, *common, "--out", str(tmp_path / "a.pt"), "--log", str(tmp_path / "a.csv")],
                            cwd=ROOT, capture_output=True, text=True)
    everyone = subprocess.run([sys.executable, *common, "--all-participants", "--out", str(tmp_path / "b.pt"),
                               "--log", str(tmp_path / "b.csv")], cwd=ROOT, capture_output=True, text=True)
    assert listed.returncode == 0 and everyone.returncode == 0, listed.stderr + everyone.stderr
    assert "loaded 40 sentences" in listed.stdout          # 5 participants x 8 sentences
    assert "loaded 240 sentences" in everyone.stdout       # all 30 participants
