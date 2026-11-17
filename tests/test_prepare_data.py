"""
Runs every builder in scripts/prepare_data.py on tiny fake datasets that mimic the real
file layouts, so the script is tested without downloading anything.
"""
import argparse
import csv
import importlib.util
import random
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "prepare_data.py"


@pytest.fixture
def prep(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("prepare_data", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "DATA", tmp_path / "data")
    monkeypatch.setattr(module, "RAW", tmp_path / "data" / "raw")
    monkeypatch.setattr(module, "PAIRS", tmp_path / "data" / "pairs")
    return module


@pytest.fixture
def args():
    return argparse.Namespace(test_fraction=1 / 3, noise_test_fraction=0.3, keystroke_train_users=0,
                              keystroke_test_users=0, keystroke_pairs_per_user=5)


def touch(path: Path, content=b""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def read(path: Path):
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_face_uses_official_pairs_and_skips_missing_images(prep, args, tmp_path):
    raw = tmp_path / "data/raw/face"
    for name in ("Ann_Lee_0001", "Ann_Lee_0002", "Bob_Ray_0001"):
        touch(raw / f"train/images/000/{name}.jpg")
    touch(raw / "pairs.csv", (
        "pair_id,fold_id,fold_position,is_same,image_a,image_b,image_a_path,image_b_path\n"
        "0,1,0,1,Ann_Lee_0001.jpg,Ann_Lee_0002.jpg,images/000/Ann_Lee_0001.jpg,images/000/Ann_Lee_0002.jpg\n"
        "1,1,1,0,Ann_Lee_0001.jpg,Bob_Ray_0001.jpg,images/000/Ann_Lee_0001.jpg,images/000/Bob_Ray_0001.jpg\n"
        "2,2,0,1,Cy_Do_0001.jpg,Cy_Do_0002.jpg,images/000/Cy_Do_0001.jpg,images/000/Cy_Do_0002.jpg\n").encode())

    stats = prep.build_face(raw, random.Random(0), args)

    rows = read(tmp_path / "data/pairs/face_test.csv")
    assert stats == {"face_test_pairs": 2, "missing_pairs": 1}
    assert [r["label"] for r in rows] == ["1", "0"]
    assert rows[1]["person_a"] == "Ann_Lee" and rows[1]["person_b"] == "Bob_Ray"
    assert (tmp_path / rows[0]["path_a"]).exists()


def test_voice_resplits_by_speaker(prep, args, tmp_path):
    raw = tmp_path / "data/raw/voice"
    for split in ("train", "valid", "test"):          # same speakers in every split, like the real repo
        for spk in range(6):
            for clip in range(2):
                touch(raw / f"{split}/id1000{spk}-vid{split}-0000{clip}.wav")

    stats = prep.build_voice(raw, random.Random(0), args)

    train = read(tmp_path / "data/pairs/voice_train.csv")
    test = read(tmp_path / "data/pairs/voice_test.csv")
    train_spk = {r["speaker_a"] for r in train} | {r["speaker_b"] for r in train}
    test_spk = {r["speaker_a"] for r in test} | {r["speaker_b"] for r in test}
    assert stats["speakers"] == 6 and stats["clips"] == 36
    assert not train_spk & test_spk                   # no speaker in both train and test
    for rows in (train, test):
        assert sum(r["label"] == "1" for r in rows) == sum(r["label"] == "0" for r in rows)


def _write_keystrokes(path: Path, participants, sections_each=4):
    p_ids = [p for p in participants for _ in range(sections_each)]
    s_ids = [p * 100 + i for p in participants for i in range(sections_each)]
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"participant_id": p_ids, "section_id": s_ids}), path)


def test_keystroke_keeps_disjoint_repo_splits(prep, args, tmp_path):
    raw = tmp_path / "data/raw/keystroke"
    _write_keystrokes(raw / "data/train-00000-of-00001.parquet", [1, 2, 3])
    _write_keystrokes(raw / "data/test-00000-of-00001.parquet", [4, 5])

    stats = prep.build_keystroke(raw, random.Random(0), args)

    assert stats["repo_splits_shared_participants"] == 0
    assert stats["keystroke_train_participants"] == 3 and stats["keystroke_test_participants"] == 2
    test = read(tmp_path / "data/pairs/keystroke_test.csv")
    assert {r["participant_a"] for r in test} <= {"4", "5"}
    assert sum(r["label"] == "1" for r in test) == 2 * 5   # 5 genuine pairs per participant (C(4,2)=6 capped)


def test_keystroke_resplits_when_repo_splits_overlap(prep, args, tmp_path):
    raw = tmp_path / "data/raw/keystroke"
    _write_keystrokes(raw / "data/train-00000-of-00001.parquet", [1, 2, 3, 4, 5, 6])
    _write_keystrokes(raw / "data/test-00000-of-00001.parquet", [1, 2])

    stats = prep.build_keystroke(raw, random.Random(0), args)

    train = read(tmp_path / "data/pairs/keystroke_train.csv")
    test = read(tmp_path / "data/pairs/keystroke_test.csv")
    ids = lambda rows: {r["participant_a"] for r in rows} | {r["participant_b"] for r in rows}  # noqa: E731
    assert stats["repo_splits_shared_participants"] == 2
    assert not ids(train) & ids(test)


def test_noise_splits_each_environment_by_time(prep, args, tmp_path):
    raw = tmp_path / "data/raw/noise"
    # mimic the repo: environments spread over its own splits, 10 clips each, numbered in time order
    for split, env in (("feature_scorer_train", "PCAFETER"), ("benchmark_test", "TBUS"), ("scorer_val", "OOFFICE")):
        for i in range(10):
            touch(raw / f"data/{split}/{env}/{env}_ch01_{i:06d}.wav")

    stats = prep.build_noise(raw, random.Random(0), args)

    train = read(tmp_path / "data/pairs/noise_train.csv")
    test = read(tmp_path / "data/pairs/noise_test.csv")
    assert stats == {"noise_environments": 3, "noise_clips": 30, "noise_train_clips": 21, "noise_test_clips": 9}
    assert {r["environment"] for r in train} == {r["environment"] for r in test} == {"PCAFETER", "TBUS", "OOFFICE"}
    for env in ("PCAFETER", "TBUS", "OOFFICE"):           # test clips come after all train clips in time
        last_train = max(r["path"] for r in train if r["environment"] == env)
        first_test = min(r["path"] for r in test if r["environment"] == env)
        assert last_train < first_test
    cafe = next(r for r in test if r["environment"] == "PCAFETER")
    assert (cafe["category"], cafe["description"]) == ("public", "cafeteria")


def test_motion_unzips_without_macos_junk_and_marks_moving(prep, args, tmp_path):
    raw = tmp_path / "data/raw/motion"
    raw.mkdir(parents=True)
    with zipfile.ZipFile(raw / "raw_data.zip", "w") as z:
        z.writestr("raw data/labels.txt", "1 1 1 0 100\n1 1 4 100 200\n")
        z.writestr("raw data/acc_exp01_user01.txt", "0.0 0.0 0.0\n")
        z.writestr("raw data/gyro_exp01_user01.txt", "0.0 0.0 0.0\n")
        z.writestr("__MACOSX/raw data/._labels.txt", "junk")

    stats = prep.build_motion(raw, random.Random(0), args)

    rows = read(tmp_path / "data/pairs/motion_segments.csv")
    assert stats == {"segments": 2, "users": 1, "experiments": 1, "moving_segments": 1}
    assert [(r["activity"], r["moving"]) for r in rows] == [("walking", "1"), ("sitting", "0")]
    assert not any("MACOSX" in str(p) or p.name.startswith("._") for p in (raw / "extracted").iterdir())
