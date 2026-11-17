"""
Stage 1: inspect the candidate Hugging Face datasets without downloading them in full.

For each dataset it reports: licence, total size, configs/splits, columns (features),
a few example rows, and whether it looks usable for verification, i.e. whether a
person/speaker/user column exists with several samples per person in the sample.

Run from the project root:
    python scripts/inspect_datasets.py                 # all datasets, 200 sample rows each
    python scripts/inspect_datasets.py --rows 500
    python scripts/inspect_datasets.py --only marcelohaps/lfw

Writes a report to results/dataset_inspection.md.
"""
import argparse
import asyncio
import itertools
import sys
from collections import Counter
from datetime import date
from pathlib import Path

from datasets import Audio, get_dataset_config_names, get_dataset_split_names, load_dataset
from huggingface_hub import HfApi

DATASETS = {
    "marcelohaps/lfw": "Face verification",
    "s3prl/mini_voxceleb1": "Voice verification",
    "ProgramComputer/voxceleb": "Voice verification (full)",
    "onepunchdimash/aalto_keystroke_features_normalized": "Keystroke dynamics",
    "rokset3/keystrokes136M_normalized_features": "Keystroke dynamics (large)",
    "dishamodi/mouse-dynamics": "Mouse dynamics",
    "SPARCO-project/benchmark_DEMAND_noise": "Background noise (DEMAND)",
    "DiFronzo/Human_Activity_Recognition": "Motion",
}

# Column names that usually identify the person (or the class, for context datasets)
ID_HINTS = ("speaker", "person", "subject", "user", "participant", "identity", "name", "label", "class", "activity", "scene")

REPORT = Path(__file__).resolve().parent.parent / "results" / "dataset_inspection.md"


def human_size(n):
    if not n:
        return "unknown"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def describe_value(v):
    """Short, readable description of one cell (images, audio and long values are summarized)."""
    if isinstance(v, dict) and "array" in v and "sampling_rate" in v:
        return f"<audio {len(v['array']) / v['sampling_rate']:.1f}s @ {v['sampling_rate']} Hz>"
    if isinstance(v, dict) and "bytes" in v and "path" in v:            # undecoded audio
        return f"<audio file {Path(v['path'] or '').name}>"
    if hasattr(v, "size") and hasattr(v, "mode"):          # PIL image
        return f"<image {v.size[0]}x{v.size[1]} {v.mode}>"
    if isinstance(v, (list, tuple)):
        return f"<list len={len(v)}>"
    s = str(v)
    return s if len(s) <= 40 else s[:37] + "..."


def describe_files(info):
    """Fallback when samples cannot be streamed: summarize the repo's file layout."""
    files = [s.rfilename for s in (info.siblings or []) if not s.rfilename.startswith(".")]
    if not files:
        return ["- **Files:** not visible (gated or empty)", ""]
    exts = Counter(Path(f).suffix.lower() or "(none)" for f in files)
    folders = Counter(f.split("/")[0] for f in files if "/" in f)
    return [f"- **Files:** {len(files)} ({', '.join(f'{e} ×{n}' for e, n in exts.most_common(5))})",
            f"- **Top-level folders:** {', '.join(f'{d}/ ×{n}' for d, n in folders.most_common(6)) or 'none'}",
            f"- **Example files:** {', '.join(f'`{f}`' for f in files[:4])}", ""]


def guess_id_column(features):
    names = list(features)
    for hint in ID_HINTS:
        for col in names:
            if hint in col.lower():
                return col
    return None


def inspect(repo_id, purpose, n_rows, api):
    lines = [f"## {purpose}: `{repo_id}`", "", f"https://huggingface.co/datasets/{repo_id}", ""]
    try:
        info = api.dataset_info(repo_id, files_metadata=True)
        card = info.card_data.to_dict() if info.card_data else {}
        licence = card.get("license") or "not stated"
        size = sum((s.size or 0) for s in (info.siblings or []))
        lines += [f"- **Licence:** {licence}", f"- **Repo size:** {human_size(size)}",
                  f"- **Gated (needs approval):** {'yes' if info.gated else 'no'}"]
    except Exception as e:  # noqa: BLE001 - report and continue with the next dataset
        lines += [f"- **Could not read dataset info:** {e}", ""]
        return lines, "error"

    try:
        configs = get_dataset_config_names(repo_id)
        config = configs[0] if configs else None
        splits = get_dataset_split_names(repo_id, config)
        lines += [f"- **Configs:** {', '.join(configs) if configs else 'default'}",
                  f"- **Splits ({config or 'default'}):** {', '.join(splits)}"]
        split = "train" if "train" in splits else splits[0]
        stream = load_dataset(repo_id, config, split=split, streaming=True)
        # decoding audio needs torchcodec + torch; for inspection the file path is enough
        for col, feat in (stream.features or {}).items():
            if isinstance(feat, Audio):
                stream = stream.cast_column(col, Audio(decode=False))
        rows = list(itertools.islice(stream, n_rows))
    except Exception as e:  # noqa: BLE001
        lines += [f"- **Could not stream samples:** {type(e).__name__}: {str(e)[:300]}"]
        lines += describe_files(info)
        return lines, "gated" if info.gated else "files only"

    if not rows:
        lines += ["- **Sample:** empty", ""]
        return lines, "empty"

    features = rows[0].keys()
    lines += [f"- **Columns:** {', '.join(f'`{c}`' for c in features)}", "",
              f"First 3 rows (of {len(rows)} sampled from `{split}`):", "",
              "| " + " | ".join(features) + " |", "|" + "---|" * len(features)]
    for r in rows[:3]:
        lines.append("| " + " | ".join(describe_value(r[c]).replace("|", "/") for c in features) + " |")
    lines.append("")

    id_col = guess_id_column(features)
    if id_col is None:
        lines += ["- **Verification check:** no person/label column found, so check manually.", ""]
        return lines, "check manually"
    counts = Counter(str(r[id_col]) for r in rows)
    multi = sum(1 for c in counts.values() if c >= 2)
    lines += [f"- **ID column (guessed):** `{id_col}`",
              f"- **In the sample:** {len(counts)} distinct values; {multi} have 2+ samples "
              f"(most common: {', '.join(f'{k} ×{v}' for k, v in counts.most_common(3))})"]
    verdict = "usable" if multi >= 2 else "check manually"
    lines += [f"- **Verdict:** {verdict} "
              f"({'can build same/different pairs' if verdict == 'usable' else 'too few repeats in the sample'})", ""]
    return lines, verdict


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=200, help="rows to sample per dataset (default 200)")
    parser.add_argument("--only", help="inspect a single dataset id")
    args = parser.parse_args()

    # Python 3.14 no longer creates an event loop implicitly; streaming image/audio files needs one
    asyncio.set_event_loop(asyncio.new_event_loop())

    targets = {args.only: DATASETS.get(args.only, "Custom")} if args.only else DATASETS
    api = HfApi()
    sections, summary = [], []
    for repo_id, purpose in targets.items():
        print(f"Inspecting {repo_id} ...", flush=True)
        lines, verdict = inspect(repo_id, purpose, args.rows, api)
        sections += lines
        summary.append((purpose, repo_id, verdict))
        print(f"  -> {verdict}", flush=True)

    report = [f"# Dataset inspection (Stage 1)", "",
              f"Generated by `scripts/inspect_datasets.py` on {date.today()}, sampling {args.rows} rows per dataset.", "",
              "| Purpose | Dataset | Verdict |", "|---|---|---|"]
    report += [f"| {p} | `{r}` | {v} |" for p, r, v in summary]
    report += ["", "Verdicts are based on a small sample; \"check manually\" means read the dataset card.", ""]
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(report + sections), encoding="utf-8")
    print(f"\nReport written to {REPORT.relative_to(Path.cwd()) if REPORT.is_relative_to(Path.cwd()) else REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
