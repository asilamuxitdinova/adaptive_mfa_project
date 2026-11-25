"""
Summarize a GPU usage log recorded with nvidia-smi during a training run.

Record the log in a separate terminal while training runs (one row every 5 seconds):
    nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
               --format=csv -l 5 | tee results/keystroke_gpu_log.csv

Then summarize it (optionally with the training log, to report epochs and training time):
    python scripts/summarize_gpu_log.py results/keystroke_gpu_log.csv \
        --training-log results/keystroke_training_log.csv --out results/keystroke_gpu_summary.md

Rows where the GPU is idle (0% use) at the start and end are trimmed, so the numbers
describe the training itself, not the minutes before and after it.
"""
import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path

COLUMNS = {   # nvidia-smi header prefix -> short name
    "timestamp": "time", "name": "gpu", "utilization.gpu": "gpu_util", "utilization.memory": "mem_util",
    "memory.used": "mem_used", "memory.total": "mem_total", "power.draw": "power", "temperature.gpu": "temp",
}


def number(text):
    """' 45 %' -> 45.0, ' 1234 MiB' -> 1234.0, '[N/A]' -> None."""
    match = re.search(r"-?\d+(\.\d+)?", text or "")
    return float(match.group()) if match else None


def read_log(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        reader = csv.reader(f)
        header = None
        for raw in reader:
            cells = [c.strip() for c in raw]
            if not cells or not any(cells):
                continue
            if cells[0].startswith("timestamp"):            # header (repeated if logging was restarted)
                header = [next((v for k, v in COLUMNS.items() if c.startswith(k)), c) for c in cells]
                continue
            if header is None or len(cells) != len(header):
                continue
            row = dict(zip(header, cells))
            try:
                row["time"] = datetime.strptime(row["time"], "%Y/%m/%d %H:%M:%S.%f")
            except (KeyError, ValueError):
                continue
            for key in ("gpu_util", "mem_util", "mem_used", "mem_total", "power", "temp"):
                row[key] = number(row.get(key))
            rows.append(row)
    return rows


def trim_idle(rows):
    busy = [i for i, r in enumerate(rows) if (r["gpu_util"] or 0) > 0]
    return rows[busy[0]:busy[-1] + 1] if busy else []


def stats(values):
    values = [v for v in values if v is not None]
    return (sum(values) / len(values), max(values)) if values else (None, None)


def fmt(value, unit, digits=0):
    return "n/a" if value is None else f"{value:,.{digits}f} {unit}".strip()


def summarize(rows, training_log=None):
    active = trim_idle(rows)
    if not active:
        return None
    seconds = (active[-1]["time"] - active[0]["time"]).total_seconds()
    gpu_avg, gpu_max = stats(r["gpu_util"] for r in active)
    _, mem_max = stats(r["mem_used"] for r in active)
    power_avg, power_max = stats(r["power"] for r in active)
    _, temp_max = stats(r["temp"] for r in active)
    summary = {
        "gpu": active[0].get("gpu", "unknown"), "samples": len(active), "duration_min": seconds / 60,
        "gpu_util_avg": gpu_avg, "gpu_util_max": gpu_max, "mem_used_max_mib": mem_max,
        "mem_total_mib": active[0]["mem_total"], "power_avg_w": power_avg, "power_max_w": power_max,
        "temp_max_c": temp_max, "energy_wh": (power_avg or 0) * seconds / 3600,
    }
    if training_log and Path(training_log).exists():
        with Path(training_log).open(encoding="utf-8") as f:
            epochs = list(csv.DictReader(f))
        if epochs:
            summary["epochs"] = len(epochs)
            summary["train_minutes"] = sum(float(e["seconds"]) for e in epochs) / 60
            best = min(epochs, key=lambda e: float(e["val_eer"]))
            summary["best_epoch"], summary["best_val_eer"] = int(best["epoch"]), float(best["val_eer"])
    return summary


def to_markdown(s, title):
    lines = [f"## {title}", "", f"Recorded with `nvidia-smi` every few seconds; idle time before and after training trimmed.", "",
             "| Metric | Value |", "|---|---|",
             f"| GPU | {s['gpu']} |",
             f"| Active duration | {fmt(s['duration_min'], 'min', 1)} ({s['samples']} samples) |",
             f"| GPU utilization | average {fmt(s['gpu_util_avg'], '%')}, peak {fmt(s['gpu_util_max'], '%')} |",
             f"| GPU memory used (peak) | {fmt(s['mem_used_max_mib'], 'MiB')} of {fmt(s['mem_total_mib'], 'MiB')} |",
             f"| Power draw | average {fmt(s['power_avg_w'], 'W', 1)}, peak {fmt(s['power_max_w'], 'W', 1)} |",
             f"| Energy used (approx.) | {fmt(s['energy_wh'], 'Wh', 1)} |",
             f"| Max temperature | {fmt(s['temp_max_c'], '°C')} |"]
    if "epochs" in s:
        lines += [f"| Epochs run | {s['epochs']} (best: epoch {s['best_epoch']}, validation EER {s['best_val_eer']:.2%}) |",
                  f"| Training time (sum of epochs) | {fmt(s['train_minutes'], 'min', 1)} |"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", type=Path, help="CSV written by nvidia-smi --format=csv")
    parser.add_argument("--training-log", type=Path, help="per-epoch training log (adds epochs and training time)")
    parser.add_argument("--out", type=Path, help="write the summary as Markdown to this file")
    parser.add_argument("--title", default="GPU usage during keystroke training")
    args = parser.parse_args()

    summary = summarize(read_log(args.log), args.training_log)
    if summary is None:
        print("no GPU activity found in the log (all rows idle or unreadable)")
        return 1
    text = to_markdown(summary, args.title)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
