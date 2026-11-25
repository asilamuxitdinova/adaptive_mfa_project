# Project Progress: Adaptive Multi-Factor Authentication

Tracks the progress of turning the adaptive MFA approach from the paper below into a working, data-driven system.

**Based on:** A. K. Nag, A. Roy, D. Dasgupta, *"An Adaptive Approach Towards the Selection of Multi-factor Authentication"*, IEEE Symposium Series on Computational Intelligence (SSCI), 2015. DOI: [10.1109/SSCI.2015.75](https://doi.org/10.1109/SSCI.2015.75)

**Last updated:** 2026-11-17

## Status overview

| Step | Description | Status |
|---|---|---|
| 1 | Simulation demo of the paper's adaptive selection approach | ✅ Done |
| 2 | Real ML-based adaptive MFA trained on public datasets | 🚧 In progress: Stage 1 done; Stage 2 keystroke verifier trained (test EER 9.94%); face and voice next |

---

## Step 1: Simulation demo ✅

**File:** [`adaptive_mfa_demo.py`](adaptive_mfa_demo.py) (Python 3, standard library only)

**Run:**
```bash
python3 adaptive_mfa_demo.py            # default seed 7, reproducible output
python3 adaptive_mfa_demo.py --seed 3   # different random choices, same overall pattern
```

### What it does
At each authentication trigger, the demo follows the paper's logic:

1. **Senses the context:** device (desktop / laptop / phone), network medium (wired / Wi-Fi / cellular), light (lux), noise (dB) and motion (moving / still).
2. **Filters out factors that cannot work:** face and iris need 100–1000 lux and a still user; voice needs a quiet environment (≤ 60 dB) and a still user; iris and hand factors need a scanner the device actually has.
3. **Scores candidate sets of 1–3 factors** using the paper's Table II trust values, capped at U_max = 3.97.
4. **Selects a set** using the paper's two objectives: high trust with few factors (objective 1), and never the same set as the previous selection (objective 2, the Markov property: only the previous selection matters).

It compares three strategies, as the paper does:

| Strategy | Behavior |
|---|---|
| **Adaptive** | Senses the environment and avoids repeating the previous set |
| **Optimal (static)** | Highest trust only; ignores the environment and selection history |
| **Random** | Any factors the device supports, chosen at random |

The simulation runs 8 triggers: office desk (×2), dark room, noisy café, walking outside on a phone (×2), bedroom at night, and back at the office.

### Results (default seed 7)

| Strategy | Avg trust | Failed logins | Same set twice in a row |
|---|---|---|---|
| **Adaptive** | **3.76** | **0** | **0** |
| Optimal | 3.36 | 2 | 3 |
| Random | 2.18 | 4 | 2 |

### Conclusions
- **Adaptive selection gives the highest trust, with zero failed logins and zero repeated sets.** This matches the paper's claims (Fig. 2: highest total trust; Table V: non-repeating selections).
- **Ignoring the environment causes failures.** The static and random strategies pick face recognition while the user is walking or in the dark, and voice in a noisy café.
- **Ignoring history makes selection predictable.** In the same office three times, the static strategy asked for *Iris + Voice* every time; the adaptive strategy rotated between different high-trust sets.
- **Adaptive sometimes trades a little trust for unpredictability** (t8: 3.94 vs 3.97), which is consistent with the paper's note that optimal selection can match adaptive in the best case.
- **The weakest setting is phone + cellular**, mirroring the lowest value (0.047) in the paper's T-matrix.

### Limitations
- The trust values come from the paper's Table II, which is based on **assumed** pairwise preferences, not measured data.
- Some scoring rules are simplifications made for the demo: a factor's trust is the mean of its device and medium values, a set's trust is the sum, voice needs ≤ 60 dB, and a login needs ≥ 2.5 trust. The paper's combined-factor formula could not be reproduced exactly from its own equations.
- No real sensors or biometric matching are used.

---

## Step 2: Real ML-based adaptive MFA 🚧

**Goal:** replace the paper's assumed trust values with **trust values learned from real data**, including how each factor's accuracy drops in darkness, noise and motion, and then reuse the adaptive selector.

### Project setup ✅

```
adaptive_mfa_project/
├── adaptive_mfa_demo.py        # Step 1 simulation
├── PROGRESS.md
├── requirements.txt            # Stage 1 libraries + Stage 2-4 libraries
├── data/                       # created by prepare_data.py on the server (gitignored)
├── src/
│   ├── metrics.py              # shared EER / FAR / FRR calculation
│   ├── pairs.py                # same-person / different-person pair building
│   ├── keystroke_data.py       # Aalto sentences -> padded key-timing sequences
│   └── verifiers/
│       └── keystroke.py        # baseline + TypeNet-style LSTM verifier (Stage 2)
├── scripts/
│   ├── inspect_datasets.py     # Stage 1 dataset inspection
│   ├── prepare_data.py         # Stage 1 download + pair building
│   ├── train_keystroke.py      # Stage 2 keystroke training
│   ├── evaluate_keystroke.py   # Stage 2 keystroke EER / FAR / FRR
│   └── summarize_gpu_log.py    # GPU usage summary from an nvidia-smi log
├── results/
│   └── dataset_inspection.md   # generated inspection report
└── tests/                      # 32 tests, all passing
    ├── test_metrics.py
    ├── test_pairs.py
    ├── test_prepare_data.py    # runs prepare_data.py on tiny fake datasets
    └── test_keystroke.py       # keystroke pipeline on synthetic typists
```

**Set up and run:**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install datasets huggingface_hub numpy pytest pillow soundfile   # Stage 1 part of requirements.txt
python -m pytest -q tests                                          # all tests (no downloads needed)
python scripts/inspect_datasets.py                                 # writes results/dataset_inspection.md
```

- **`src/metrics.py`:** computes EER, FAR and FRR from verifier scores (1 = same person, 0 = different person). Every factor will use it, so results are comparable. Tested against known cases, including the theoretical EER of two overlapping normal distributions (15.87%).
- **`src/pairs.py`:** builds balanced genuine / impostor pairs and splits people into disjoint train / test sets, with a fixed random seed for reproducibility.

### Data preparation (`scripts/prepare_data.py`)

**Datasets are never stored in git or on the development laptop.** They are downloaded on the server by one script, at pinned dataset versions, so every run uses identical data.

```bash
# on the server, from the project root (set HF_HOME to a large disk first)
python scripts/prepare_data.py --dry-run           # list datasets and sizes (~1.3 GB total)
python scripts/prepare_data.py                     # download everything and build pairs
python scripts/prepare_data.py --only face voice   # or just some datasets
```

| Output (in `data/pairs/`) | Contents |
|---|---|
| `face_test.csv` | LFW's official 6,000 pairs in 10 folds (standard benchmark) |
| `voice_train.csv`, `voice_test.csv` | Same / different speaker clip pairs; **speakers re-split** 20 train / 10 test, because the repo's own splits share all 30 speakers |
| `keystroke_train.csv`, `keystroke_test.csv` | Same / different participant sentence pairs; **no participant in both** (default caps: 10,000 train and 2,000 test participants, 5 genuine pairs each) |
| `noise_train.csv`, `noise_test.csv` | Real background-noise clips (DEMAND, 17 environments) to mix into voice clips in Stage 3; **each environment split by time**, first ~70% train / last ~30% test, so every environment is in both and no audio is shared |
| `motion_segments.csv` | Labelled activity segments with a `moving` flag (walking vs sitting / standing) |

A summary of what was downloaded (revisions, counts, seed) is written to `data/MANIFEST.json`.

- **Download time:** LFW is ~13,000 small files. Anonymous Hugging Face downloads are rate-limited (about 1 hour for LFW), so log in once with `hf auth login` (free read token) on the server. Interrupted downloads resume when the script is re-run.
- **Testing without data:** `tests/test_prepare_data.py` runs every builder on tiny fake datasets that mimic the real file layouts.

### Server run results (2026-11-17)

Run on the GPU server (Python 3.12 via `uv`, PyTorch 2.15 + CUDA 13.2, NVIDIA L4) after `hf auth login`; all 5 datasets downloaded and processed without errors. From `data/MANIFEST.json`:

| Dataset | Result | Check |
|---|---|---|
| Face (LFW) | 6,000 test pairs, 0 missing images | ✅ Exactly the official benchmark |
| Voice (mini VoxCeleb) | 30 speakers, 900 clips → 20 train / 10 test speakers; 17,400 train / 8,700 test pairs | ✅ 435 same-speaker pairs per speaker (30 clips), plus as many different-speaker pairs |
| Keystroke (Aalto) | 77,873 train / 77,874 test participants available, **0 shared**; capped to 10,000 / 2,000 → 100,000 / 20,000 pairs | ✅ Person-disjoint splits |
| Scenes (TAU) | Only **10,000 of the listed clips are in the Hugging Face copy**, and they cover **only 4 of the 10 scenes** (bus 3,670, airport 3,660, metro 2,470, metro station 200): the uploader kept the first 10,000 files alphabetically | ❌ **Replaced with DEMAND** (see below) |
| Motion (HAR) | 79 labelled segments, 8 users, 12 experiments, 47 moving | ✅ Small, but enough for moving vs still |

**Why TAU was replaced:** a re-split by recording location confirmed only 4 scenes (all transport / indoor), with no park, street, café or home noise. The paper's condition is the **noise level**, not the type of place, so the project now:
- **measures noise level directly in dB** from the microphone, like lux from the light sensor (no model needed);
- uses **DEMAND** as real background noise to mix into voice clips for the Stage 3 degradation curves;
- treats a scene classifier as an **optional extra** (the complete 10-scene TUT 2018 dataset, 15 GB, could be added later on the server).
- **Compute:** development and small subsets on a laptop (CPU); full evaluations, degradation sweeps and any fine-tuning on a server with an **NVIDIA L4 GPU (24 GB)**. The code will use the GPU automatically when available.

### Stage 2: Keystroke verifier (trained: test EER 9.94%)

**Approach**, following TypeNet (Acien et al., *IEEE T-BIOM* 2021), which introduced deep keystroke verification on this Aalto dataset:
- Each typed sentence becomes a sequence of up to 50 keys; each key has its code and four timings: hold time (`hl`) and three gaps to the next key (`il`, `pl`, `rl`). Timings are standardized with training statistics only.
- A 2-layer **LSTM** turns the sequence into a 128-number embedding (a "typing fingerprint"). Each batch has 64 participants × 5 sentences, and the model learns to place one person's sentences closer together than anyone else's.
- **Loss:** **supervised contrastive loss** (Khosla et al., 2020) by default. The first real-data run used TypeNet's batch-hard triplet loss: its loss got stuck at the margin value (0.2003 vs margin 0.2) and the scores were squeezed into a tiny range (a near-collapse), so the default was switched to this more stable loss; triplet loss is still available with `--loss triplet`. The training log now records an **embedding spread** each epoch and warns if it drops towards 0 (collapse).
- About 500 of the 10,000 training participants are held out for **validation**; the model with the best validation EER is kept, with early stopping.
- A **baseline with no training** (mean / std / median of each timing, compared by distance) is evaluated alongside it, to show what training adds.
- Testing uses `keystroke_test.csv`: **2,000 participants never seen in training**, 20,000 pairs of single sentences. Results are compared with the paper's Table I keystroke values (FAR 7%, FRR 0.1%, crossover 1.8%).

**Run on the server:**
```bash
python scripts/train_keystroke.py --epochs 3 --limit-users 500   # quick check that everything runs
python scripts/train_keystroke.py                                # full training on the GPU
python scripts/evaluate_keystroke.py                             # EER / FAR / FRR on test participants
```

| Output | Contents |
|---|---|
| `models/keystroke_typenet.pt` | Best model + settings (gitignored) |
| `results/keystroke_training_log.csv` | Loss, validation EER and embedding spread per epoch |
| `results/keystroke_results.md` / `.json` | Test EER / FAR / FRR, baseline vs trained model vs paper |
| `results/keystroke_scores.png` | Genuine vs impostor score distributions |

**Run 1 on real data (2026-11-25, batch-hard triplet loss):** 20,000 test pairs from 2,000 unseen participants.

| Verifier | EER | FAR at EER | FRR at EER | FAR when FRR = 0.1% |
|---|---|---|---|---|
| Baseline (no training) | 26.52% | 26.52% | 26.52% | 99.81% |
| TypeNet-style LSTM, triplet loss | 22.07% | 21.13% | 23.00% | 65.75% |
| *Paper, Table I (from literature)* | *crossover 1.8%* | | | *FAR 7% at FRR 0.1%* |

- Training helped only a little (26.5% → 22.1%). The training loss stayed at 0.2003, the triplet margin, for the last epochs and validation EER fluctuated between 21% and 25%: the near-collapse described above. Epochs took about 2 seconds each on the L4.
- Each test pair compares **one sentence with one sentence**, the hardest setting; systems that compare a new sentence with a profile of several enrolled sentences report much lower errors.
- At the paper's operating point (only 0.1% of real users rejected), 66% of impostors would be accepted, so the paper's FAR of 7% looks very optimistic for single sentences.
- **Next:** retrain with supervised contrastive loss (run 2).

**Run 2 on real data (2026-11-25, supervised contrastive loss):** same 20,000 test pairs from 2,000 unseen participants.

| Verifier | EER | FAR at EER | FRR at EER | FAR when FRR = 0.1% |
|---|---|---|---|---|
| Baseline (no training) | 26.52% | 26.52% | 26.52% | 99.81% |
| TypeNet-style LSTM, triplet loss (run 1) | 22.07% | 21.13% | 23.00% | 65.75% |
| **TypeNet-style LSTM, supervised contrastive (run 2)** | **9.94%** | 9.94% | 9.94% | 52.16% |
| *Paper, Table I (from literature)* | *crossover 1.8%* | | | *FAR 7% at FRR 0.1%* |

- **The loss change fixed the near-collapse:** EER more than halved (22.07% → 9.94%), the embedding spread stayed steady around 0.08, and the training loss kept falling.
- **No overfitting:** validation EER 9.48% (best, epoch 47) vs test EER 9.94% on participants never seen in training.
- **Not finished learning:** training stopped at the 50-epoch limit with the loss still decreasing (3.4148 at epoch 50), so more epochs are likely to help.
- The gap to the paper's 1.8% remains large; part of it is the single-sentence vs single-sentence protocol.

**GPU usage during run 2** (`nvidia-smi` every 5 s, summarized with `scripts/summarize_gpu_log.py`):

| Metric | Value |
|---|---|
| GPU | NVIDIA L4 |
| Training time | 1.7 min (50 epochs, ~2 s each) |
| GPU utilization | average 42%, peak 72% |
| GPU memory used (peak) | 902 MiB of 23,034 MiB |
| Power draw | average 52.5 W, peak 59.4 W |
| Energy used (approx.) | 1.5 Wh |
| Max temperature | 70 °C |

The model is small, so the GPU is far from fully used; much larger runs (more epochs, all ~78,000 participants) are cheap.

**Possible next runs:** (1) more epochs; (2) train on all ~78,000 participants (`prepare_data.py --only keystroke --keystroke-train-users 0`, then retrain); (3) evaluate with **enrolment**: compare a new sentence with a profile averaged from several of the user's sentences, as real systems and TypeNet do.

**Sanity check on synthetic data (laptop):** 80 fake typists with distinct rhythms, 30 unseen test typists → baseline EER 10%, trained model **4%**. This only shows the pipeline learns; synthetic typing is far easier than real typing.

**Options:** `prepare_data.py --keystroke-train-users 0` rebuilds the pairs with all ~78,000 training participants (TypeNet used ~68,000) for a larger training run.

### Planned stages

- [x] **Stage 1: Dataset preparation.**
  - [x] Inspect the candidate datasets and their licences ([results/dataset_inspection.md](results/dataset_inspection.md))
  - [x] Write the download + pair-building script ([scripts/prepare_data.py](scripts/prepare_data.py)), tested on fake data
  - [x] Set up the server (Python 3.12 via `uv`, `hf auth login`) and run `prepare_data.py` there
  - [x] Check the generated pairs and `data/MANIFEST.json` on the server
  - [x] Found that the TAU copy has only 4 of 10 scenes; replaced it with DEMAND
  - [x] Download DEMAND on the server (`--only noise`): 17 environments, 1,275 clips, 901 train / 374 test ✅
  - [ ] Delete the unused TAU files on the server (`data/raw/scenes`, `data/pairs/scenes_*.csv`)
  - [ ] Request access to the gated datasets (full VoxCeleb), only if needed later
- [ ] **Stage 2: Per-factor verifiers.** *(in progress)* Measure real **EER, FAR and FRR** for each factor. Face and voice use pretrained models (evaluation only); keystroke is the only model trained here.
  - [x] Keystroke verifier code: data loader, baseline, TypeNet-style LSTM, training and evaluation scripts, tested on synthetic typists (see below)
  - [x] Train the keystroke model on the server and evaluate it on the test participants: **EER 9.94%** (run 2), baseline 26.52%
  - [ ] Optional keystroke improvements: more epochs, all ~78,000 participants, enrolment with several sentences
  - [ ] Face verifier on LFW (pretrained model, evaluation only)
  - [ ] Voice verifier on mini VoxCeleb (pretrained model, evaluation only)
- [ ] **Stage 3: Degradation curves.** Re-evaluate each verifier under simulated conditions (darkened or blurred faces, voice mixed with real DEMAND noise such as cafeteria, street or metro at increasing levels) to learn **how EER changes with light, noise and motion**. This is the learned replacement for the paper's T-matrix.
- [ ] **Stage 4: Context sensing.** Train a human-activity classifier (walking vs still) on the motion data. Light (lux) and noise level (dB) are measured directly from the device's sensors, so no model is needed for them. A scene classifier is optional.
- [ ] **Stage 5: Learned trust function.** Convert measured accuracy into trust values per factor and per context, replacing the hard-coded Table II.
- [ ] **Stage 6: Adaptive selector.** Plug the learned trust function into the selector from Step 1 (same two objectives). Later: try a contextual-bandit model that learns from login outcomes.
- [ ] **Stage 7: Evaluation.** Compare adaptive, static and random selection using measured trust and failure rates, and compare against the paper's results.

### Datasets: inspection results (2026-11-17)

Full generated report: [results/dataset_inspection.md](results/dataset_inspection.md). The table below adds what was confirmed by reading file lists and dataset cards.

| Purpose | Dataset | Size | What's inside | Verdict |
|---|---|---|---|---|
| Face verification | [`marcelohaps/lfw`](https://huggingface.co/datasets/marcelohaps/lfw) | 182 MB | 13,233 face images (250×250) of 5,749 people; 1,680 people have 2+ images. Includes the official `pairs.csv` (6,000 same/different pairs in 10 folds) | ✅ **Use.** Ready-made verification pairs |
| Voice verification | [`s3prl/mini_voxceleb1`](https://huggingface.co/datasets/s3prl/mini_voxceleb1) | 228 MB | 900 `.wav` clips from 30 speakers (30 clips each), split into train / valid / test (300 each) **but all three splits contain the same 30 speakers**; the speaker ID is in the file name (`id10012-...`) | ✅ **Use.** Small enough for a laptop. Decoding audio with `datasets` needs `torchcodec` (installed with PyTorch in Stage 2) |
| Voice verification (full) | [`ProgramComputer/voxceleb`](https://huggingface.co/datasets/ProgramComputer/voxceleb) | 716 GB | VoxCeleb1 + VoxCeleb2 as zip archives | 🔒 **Gated, optional.** Needs Hugging Face login and accepting terms; only worth it on the GPU server with enough disk space |
| Keystroke dynamics | [`onepunchdimash/aalto_keystroke_features_normalized`](https://huggingface.co/datasets/onepunchdimash/aalto_keystroke_features_normalized) | 781 MB | One row per typed sentence: `participant_id`, key codes and four timing features per key (`hl`, `il`, `pl`, `rl`: hold and between-key latencies); about 15 sentences per participant; train / test splits | ✅ **Use.** Can build same/different-participant pairs |
| Keystroke dynamics (large) | [`rokset3/keystrokes136M_normalized_features`](https://huggingface.co/datasets/rokset3/keystrokes136M_normalized_features) | 781 MB | Not visible without access; same size as the Aalto dataset above, so possibly the same data | 🔒 **Gated, probably not needed** |
| Mouse dynamics | [`dishamodi/mouse-dynamics`](https://huggingface.co/datasets/dishamodi/mouse-dynamics) | 277 MB | 3,571 PNG images of mouse paths labeled only **bot** (2,616) or **human** (955); no user IDs | ❌ **Not usable for verification.** It detects bots, not users. MIT licence |
| Background noise | [`SPARCO-project/benchmark_DEMAND_noise`](https://huggingface.co/datasets/SPARCO-project/benchmark_DEMAND_noise) | 0.15 GB | 1,275 four-second clips (16 kHz, mono) from the DEMAND database: 75 per environment, 17 environments: kitchen, living room, laundry, sports field, park, river, hallway, meeting room, office, **cafeteria**, restaurant, train station, public square, traffic, bus, car, metro | ✅ **Use** as real background noise to mix into voice clips (Stage 3). Licence CC BY 4.0 or CC BY-SA 3.0, depending on the listing (see DEMAND on Zenodo) |
| ~~Noise / environment~~ | [`quinnlue/tau-urban-acoustic-scenes-2022-mobile-split5`](https://huggingface.co/datasets/quinnlue/tau-urban-acoustic-scenes-2022-mobile-split5) | 1.2 GB | Label lists name 10 scenes, but the audio covers **only 4** (airport, bus, metro, metro station) | ❌ **Replaced by DEMAND** |
| Motion | [`DiFronzo/Human_Activity_Recognition`](https://huggingface.co/datasets/DiFronzo/Human_Activity_Recognition) | 7 MB | `raw_data.zip`: accelerometer + gyroscope at 50 Hz from a Samsung Galaxy S8 in a pocket; the zip contains 8 users / 12 experiments (the README mentions 16 volunteers), with a labels file; 5 activities: walking, walking upstairs, walking downstairs, sitting, standing | ✅ **Use** for a moving-vs-still classifier. Needs unzipping and parsing the labels file |

### Known gaps
- **Fingerprint, iris and hand biometrics:** no real public datasets on Hugging Face (they are privacy-restricted and usually need signed licence agreements). The "iris" dataset on Hugging Face is the flower dataset. These factors will stay simulated with the paper's values.
- **Device and network effects:** no public dataset records the same users logging in on different devices and networks. These effects will stay simulated unless we collect our own data.
- **Licences:** LFW and VoxCeleb are for research use only, which suits a university project but not a commercial product.
- **Voice evaluation may be optimistic:** mini VoxCeleb's speakers (IDs like `id10012`) come from the VoxCeleb1 *development* set, which popular pretrained speaker models (e.g. SpeechBrain ECAPA) were trained on. The standard unseen-speaker test set is VoxCeleb1-O (speakers `id10270`–`id10309`), available in the gated full VoxCeleb dataset.
- **Mouse dynamics:** the only mouse dataset found is for bot detection, so mouse dynamics will stay simulated unless a per-user dataset is found.
- **Compute:** the minimal version runs on a laptop CPU; full VoxCeleb, degradation sweeps and fine-tuning will run on the L4 GPU server.
