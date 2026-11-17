"""
Stage 1: download the datasets and build verification pairs / label lists.

Datasets are downloaded from Hugging Face at pinned revisions (so every machine gets
identical data) into data/raw/<name>/, and the derived files go into data/pairs/.
Nothing in data/ is committed to git; run this script on each machine instead.

Run from the project root:
    python scripts/prepare_data.py                     # everything (~2.4 GB)
    python scripts/prepare_data.py --only face voice   # just some datasets
    python scripts/prepare_data.py --dry-run           # show what would be downloaded
    python scripts/prepare_data.py --skip-download     # rebuild pairs from existing downloads

Outputs (paths in the CSVs are relative to the project root):
    data/pairs/face_test.csv          LFW official 6,000 pairs (10 folds)
    data/pairs/voice_{train,test}.csv speaker pairs, speaker-disjoint train/test
    data/pairs/keystroke_{train,test}.csv sentence pairs, participant-disjoint train/test
    data/pairs/scenes_{train,test}.csv acoustic scene labels, location-disjoint train/test
    data/pairs/motion_segments.csv     labelled activity segments (moving vs still)
    data/MANIFEST.json                 what was downloaded, revisions and counts

Set HF_HOME to a large disk on servers; downloads are skipped when already present.
LFW and the scene dataset are ~10,000 small files each, and anonymous Hugging Face
downloads are rate-limited: log in once with `hf auth login` (free token) for faster
downloads. An interrupted download resumes where it stopped when re-run.
"""
import argparse
import csv
import json
import random
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pairs import make_pairs, split_ids  # noqa: E402

DATA = ROOT / "data"
RAW = DATA / "raw"
PAIRS = DATA / "pairs"


@dataclass(frozen=True)
class Source:
    repo_id: str
    revision: str          # pinned commit, so results are reproducible
    purpose: str
    licence: str


SOURCES = {
    "face": Source("marcelohaps/lfw", "12a61458b56d0433d07269dc1d64368abf4f6b4d",
                   "Face verification (LFW)", "research use"),
    "voice": Source("s3prl/mini_voxceleb1", "43c1579f7bc6c72f7e6ff55bf2bd14703f0bbbaa",
                    "Voice verification (mini VoxCeleb1)", "research use (VoxCeleb terms)"),
    "keystroke": Source("onepunchdimash/aalto_keystroke_features_normalized", "7190311c9635848d0f603913c5bc03ca47c25fbc",
                        "Keystroke dynamics (Aalto 136M keystrokes)", "research use (Aalto study)"),
    "scenes": Source("quinnlue/tau-urban-acoustic-scenes-2022-mobile-split5", "f8279077c020195bc8828ffe11e8a2647929d3a6",
                     "Acoustic scenes / noise (TAU 2022 mobile)", "non-commercial research"),
    "motion": Source("DiFronzo/Human_Activity_Recognition", "578af74e6d9abf50e091ad2292a79dda85998e0f",
                     "Human activity / motion (smartphone IMU)", "not stated"),
}

ACTIVITIES = {1: "walking", 2: "walking_upstairs", 3: "walking_downstairs", 4: "sitting", 5: "standing"}
MOVING = {"walking", "walking_upstairs", "walking_downstairs"}


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"    wrote {rel(path)} ({len(rows):,} rows)")
    return len(rows)


def write_pairs(path: Path, pairs, id_name):
    return write_csv(path, ["label", "path_a", "path_b", f"{id_name}_a", f"{id_name}_b"],
                     [(lab, a, b, ia, ib) for a, b, lab, ia, ib in pairs])


# ---------------------------------------------------------------- download

def remote_size(src: Source):
    from huggingface_hub import HfApi
    info = HfApi().dataset_info(src.repo_id, revision=src.revision, files_metadata=True)
    return sum(s.size or 0 for s in info.siblings or [])


def download(name: str, src: Source, workers: int) -> Path:
    from huggingface_hub import snapshot_download
    target = RAW / name
    print(f"  downloading {src.repo_id}@{src.revision[:8]} -> {rel(target)}")
    snapshot_download(src.repo_id, repo_type="dataset", revision=src.revision, local_dir=target, max_workers=workers)
    return target


# ---------------------------------------------------------------- builders

def build_face(raw: Path, rng, args):
    """LFW's official pairs: 10 folds x 600 pairs (300 same, 300 different)."""
    rows, missing = [], 0
    with (raw / "pairs.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            a, b = raw / "train" / r["image_a_path"], raw / "train" / r["image_b_path"]
            if not (a.exists() and b.exists()):
                missing += 1
                continue
            person_a = r["image_a"].rsplit("_", 1)[0]
            person_b = r["image_b"].rsplit("_", 1)[0]
            rows.append((r["is_same"], r["fold_id"], rel(a), rel(b), person_a, person_b))
    if missing:
        print(f"    warning: {missing} pairs skipped (image files missing)")
    n = write_csv(PAIRS / "face_test.csv", ["label", "fold", "path_a", "path_b", "person_a", "person_b"], rows)
    return {"face_test_pairs": n, "missing_pairs": missing}


def build_voice(raw: Path, rng, args):
    """The repo's train/valid/test splits share the same 30 speakers, so re-split by speaker."""
    clips = {}
    for wav in sorted(raw.glob("*/*.wav")):
        clips.setdefault(wav.name.split("-")[0], []).append(rel(wav))
    train_ids, test_ids = split_ids(clips, args.test_fraction, rng)
    stats = {"speakers": len(clips), "clips": sum(map(len, clips.values())),
             "train_speakers": len(train_ids), "test_speakers": len(test_ids)}
    for split, ids in (("train", train_ids), ("test", test_ids)):
        pairs = make_pairs({s: clips[s] for s in ids}, rng)
        stats[f"voice_{split}_pairs"] = write_pairs(PAIRS / f"voice_{split}.csv", pairs, "speaker")
    return stats


def build_keystroke(raw: Path, rng, args):
    """Pairs of typed sentences (section_id). Train and test use different participants."""
    import pyarrow.parquet as pq

    def sections(split):
        groups = {}
        for f in sorted((raw / "data").glob(f"{split}-*.parquet")):
            t = pq.read_table(f, columns=["participant_id", "section_id"]).to_pydict()
            for p, s in zip(t["participant_id"], t["section_id"]):
                groups.setdefault(str(p), []).append(str(s))
        return groups

    hf_train, hf_test = sections("train"), sections("test")
    overlap = set(hf_train) & set(hf_test)
    if overlap:
        # repo splits share participants: pool them and re-split by participant
        pooled = {p: hf_train.get(p, []) + hf_test.get(p, []) for p in set(hf_train) | set(hf_test)}
        train_ids, test_ids = split_ids(pooled, args.test_fraction, rng)
        hf_train = {p: pooled[p] for p in train_ids}
        hf_test = {p: pooled[p] for p in test_ids}
    stats = {"participants_train_available": len(hf_train), "participants_test_available": len(hf_test),
             "repo_splits_shared_participants": len(overlap)}

    caps = {"train": args.keystroke_train_users, "test": args.keystroke_test_users}
    for split, groups in (("train", hf_train), ("test", hf_test)):
        ids = sorted(p for p, s in groups.items() if len(s) >= 2)
        if caps[split] and len(ids) > caps[split]:
            ids = sorted(rng.sample(ids, caps[split]))
        pairs = make_pairs({p: groups[p] for p in ids}, rng, max_genuine_per_id=args.keystroke_pairs_per_user)
        stats[f"keystroke_{split}_participants"] = len(ids)
        stats[f"keystroke_{split}_pairs"] = write_csv(
            PAIRS / f"keystroke_{split}.csv",
            ["label", "section_a", "section_b", "participant_a", "participant_b"],
            [(lab, a, b, ia, ib) for a, b, lab, ia, ib in pairs])
    return stats


def scene_location(filename: str) -> str:
    """'audio/airport-barcelona-203-6129-0-a.wav' -> 'airport-barcelona-203' (one recording location)."""
    return "-".join(Path(filename).stem.split("-")[:3])


def build_scenes(raw: Path, rng, args):
    """
    Scene labels from the tab-separated DCASE lists. The Hugging Face copy contains only
    10,000 of the listed clips (1,894 of the train list, 8,106 of the test list), so the
    downloaded clips are pooled and re-split, keeping every recording location entirely
    in train or in test, separately for each scene so all scenes appear in both.
    """
    labels, missing = {}, 0
    for listing in ("fold1_train.csv", "fold1_evaluate.csv"):
        with (raw / "evaluation_setup" / listing).open(encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if (raw / r["filename"]).exists():
                    labels[r["filename"]] = r["scene_label"]
                else:
                    missing += 1

    by_scene = {}
    for filename, scene in labels.items():
        by_scene.setdefault(scene, {}).setdefault(scene_location(filename), []).append(filename)

    split = {"train": [], "test": []}
    for scene in sorted(by_scene):
        locations = by_scene[scene]
        if len(locations) < 2:          # cannot hold out a location; keep it for training
            train_locs, test_locs = sorted(locations), []
        else:
            train_locs, test_locs = split_ids(locations, args.scenes_test_fraction, rng)
        for name, locs in (("train", train_locs), ("test", test_locs)):
            split[name] += [(rel(raw / f), scene, loc) for loc in locs for f in sorted(locations[loc])]

    stats = {"scenes_clips_downloaded": len(labels), "scenes_listed_but_not_in_repo": missing,
             "scenes_classes": len(by_scene), "scenes_locations": sum(map(len, by_scene.values()))}
    for name, rows in split.items():
        stats[f"scenes_{name}_clips"] = write_csv(PAIRS / f"scenes_{name}.csv", ["path", "scene", "location"], rows)
    return stats


def build_motion(raw: Path, rng, args):
    """Unzip the raw IMU logs (skipping macOS metadata) and list labelled activity segments."""
    out = raw / "extracted"
    with zipfile.ZipFile(raw / "raw_data.zip") as z:
        for member in z.namelist():
            if member.startswith("__MACOSX") or member.endswith("/"):
                continue
            target = out / Path(member).name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(member))
    rows = []
    for line in (out / "labels.txt").read_text().split("\n"):
        if not line.strip():
            continue
        exp, user, act, start, end = (int(x) for x in line.split())
        activity = ACTIVITIES[act]
        acc, gyro = out / f"acc_exp{exp:02d}_user{user:02d}.txt", out / f"gyro_exp{exp:02d}_user{user:02d}.txt"
        rows.append((exp, user, activity, int(activity in MOVING), start, end, rel(acc), rel(gyro)))
    n = write_csv(PAIRS / "motion_segments.csv",
                  ["experiment", "user", "activity", "moving", "start_sample", "end_sample", "acc_path", "gyro_path"], rows)
    return {"segments": n, "users": len({r[1] for r in rows}), "experiments": len({r[0] for r in rows}),
            "moving_segments": sum(r[3] for r in rows)}


BUILDERS = {"face": build_face, "voice": build_voice, "keystroke": build_keystroke,
            "scenes": build_scenes, "motion": build_motion}


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=SOURCES, help="datasets to prepare (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="show what would be downloaded, then stop")
    parser.add_argument("--skip-download", action="store_true", help="only rebuild pairs from existing downloads")
    parser.add_argument("--workers", type=int, default=16, help="parallel downloads (default 16)")
    parser.add_argument("--seed", type=int, default=42, help="random seed for splits and pairs (default 42)")
    parser.add_argument("--test-fraction", type=float, default=1 / 3,
                        help="fraction of people held out for testing (voice, keystroke if re-split; default 1/3)")
    parser.add_argument("--scenes-test-fraction", type=float, default=0.3,
                        help="fraction of recording locations per scene held out for testing (default 0.3)")
    parser.add_argument("--keystroke-train-users", type=int, default=10_000, help="max participants for training pairs (0 = all)")
    parser.add_argument("--keystroke-test-users", type=int, default=2_000, help="max participants for test pairs (0 = all)")
    parser.add_argument("--keystroke-pairs-per-user", type=int, default=5, help="genuine pairs per participant")
    args = parser.parse_args()

    names = args.only or list(SOURCES)
    if args.dry_run:
        total = 0
        for name in names:
            size = remote_size(SOURCES[name])
            total += size
            print(f"  {name:<10} {SOURCES[name].repo_id:<58} {size / 2**30:6.2f} GB")
        print(f"  {'total':<10} {'':<58} {total / 2**30:6.2f} GB")
        return 0

    manifest_path = DATA / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name in names:
        src = SOURCES[name]
        print(f"\n[{name}] {src.purpose}")
        raw = RAW / name if args.skip_download else download(name, src, args.workers)
        if not raw.exists():
            print(f"  error: {rel(raw)} not found; run without --skip-download first")
            return 1
        rng = random.Random(f"{args.seed}-{name}")   # independent, reproducible stream per dataset
        stats = BUILDERS[name](raw, rng, args)
        manifest[name] = {"repo_id": src.repo_id, "revision": src.revision, "licence": src.licence,
                          "seed": args.seed, "prepared_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                          **stats}

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nDone. Summary in {rel(manifest_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
