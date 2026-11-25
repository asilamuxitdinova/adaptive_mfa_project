import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "summarize_gpu_log.py"
spec = importlib.util.spec_from_file_location("summarize_gpu_log", SCRIPT)
gpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu)

HEADER = ("timestamp, name, utilization.gpu [%], utilization.memory [%], memory.used [MiB], "
          "memory.total [MiB], power.draw [W], temperature.gpu")


def row(second, util, mem, power, temp=50):
    return f"2026/11/25 10:00:{second:02d}.123, NVIDIA L4, {util} %, {util // 2} %, {mem} MiB, 23034 MiB, {power:.2f} W, {temp}"


@pytest.fixture
def log(tmp_path):
    lines = [HEADER, row(0, 0, 0, 16.0),                      # idle before training
             row(5, 60, 2000, 50.0), row(10, 80, 3000, 60.0, 61), row(15, 70, 2500, 55.0),
             row(20, 0, 0, 17.0), "", HEADER]                  # idle after, blank line, repeated header
    path = tmp_path / "gpu.csv"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_parses_nvidia_smi_units(log):
    rows = gpu.read_log(log)
    assert len(rows) == 5
    assert rows[1]["gpu_util"] == 60 and rows[1]["mem_used"] == 2000 and rows[1]["power"] == 50.0


def test_summary_trims_idle_time(log):
    s = gpu.summarize(gpu.read_log(log))
    assert s["samples"] == 3 and s["duration_min"] == pytest.approx(10 / 60)
    assert s["gpu_util_avg"] == pytest.approx(70) and s["gpu_util_max"] == 80
    assert s["mem_used_max_mib"] == 3000 and s["temp_max_c"] == 61
    assert s["power_avg_w"] == pytest.approx(55.0)


def test_training_log_adds_epochs(log, tmp_path):
    train = tmp_path / "train.csv"
    train.write_text("epoch,train_loss,val_eer,seconds\n1,0.5,0.20,30\n2,0.4,0.15,30\n3,0.35,0.16,30\n")
    s = gpu.summarize(gpu.read_log(log), train)
    assert s["epochs"] == 3 and s["best_epoch"] == 2 and s["best_val_eer"] == 0.15
    assert s["train_minutes"] == pytest.approx(1.5)
    md = gpu.to_markdown(s, "GPU usage")
    assert "| GPU | NVIDIA L4 |" in md and "peak 80 %" in md and "epoch 2" in md


def test_all_idle_log_gives_none(tmp_path):
    path = tmp_path / "idle.csv"
    path.write_text("\n".join([HEADER, row(0, 0, 0, 16.0), row(5, 0, 0, 16.0)]) + "\n")
    assert gpu.summarize(gpu.read_log(path)) is None
