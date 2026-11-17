# Project Progress: Adaptive Multi-Factor Authentication

Tracks the progress of turning the adaptive MFA approach from the paper below into a working, data-driven system.

**Based on:** A. K. Nag, A. Roy, D. Dasgupta, *"An Adaptive Approach Towards the Selection of Multi-factor Authentication"*, IEEE Symposium Series on Computational Intelligence (SSCI), 2015. DOI: [10.1109/SSCI.2015.75](https://doi.org/10.1109/SSCI.2015.75)

**Last updated:** 2026-11-17

## Status overview

| Step | Description | Status |
|---|---|---|
| 1 | Simulation demo of the paper's adaptive selection approach | ✅ Done |
| 2 | Real ML-based adaptive MFA trained on public datasets | 🚧 In progress: project set up, datasets inspected, data preparation script ready (Stage 1) |

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
│   └── verifiers/              # face, voice, keystroke verifiers (Stage 2)
├── scripts/
│   ├── inspect_datasets.py     # Stage 1 dataset inspection
│   └── prepare_data.py         # Stage 1 download + pair building
├── results/
│   └── dataset_inspection.md   # generated inspection report
└── tests/                      # 19 tests, all passing
    ├── test_metrics.py
    ├── test_pairs.py
    └── test_prepare_data.py    # runs prepare_data.py on tiny fake datasets
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
python scripts/prepare_data.py --dry-run           # list datasets and sizes (~2.4 GB total)
python scripts/prepare_data.py                     # download everything and build pairs
python scripts/prepare_data.py --only face voice   # or just some datasets
```

| Output (in `data/pairs/`) | Contents |
|---|---|
| `face_test.csv` | LFW's official 6,000 pairs in 10 folds (standard benchmark) |
| `voice_train.csv`, `voice_test.csv` | Same / different speaker clip pairs; **speakers re-split** 20 train / 10 test, because the repo's own splits share all 30 speakers |
| `keystroke_train.csv`, `keystroke_test.csv` | Same / different participant sentence pairs; **no participant in both** (default caps: 10,000 train and 2,000 test participants, 5 genuine pairs each) |
| `scenes_train.csv`, `scenes_test.csv` | Acoustic scene labels for the noise / environment classifier |
| `motion_segments.csv` | Labelled activity segments with a `moving` flag (walking vs sitting / standing) |

A summary of what was downloaded (revisions, counts, seed) is written to `data/MANIFEST.json`.

- **Download time:** LFW and the scene dataset are ~10,000 small files each. Anonymous Hugging Face downloads are rate-limited (about 1 hour each), so log in once with `hf auth login` (free read token) on the server. Interrupted downloads resume when the script is re-run.
- **Testing without data:** `tests/test_prepare_data.py` runs every builder on tiny fake datasets that mimic the real file layouts.
- **Compute:** development and small subsets on a laptop (CPU); full evaluations, degradation sweeps and any fine-tuning on a server with an **NVIDIA L4 GPU (24 GB)**. The code will use the GPU automatically when available.

### Planned stages

- [ ] **Stage 1: Dataset preparation.** *(in progress)*
  - [x] Inspect the candidate datasets and their licences ([results/dataset_inspection.md](results/dataset_inspection.md))
  - [x] Write the download + pair-building script ([scripts/prepare_data.py](scripts/prepare_data.py)), tested on fake data
  - [ ] Set up the server (Python 3.11/3.12, `HF_HOME`, `hf auth login`) and run `prepare_data.py` there
  - [ ] Check the generated pairs and `data/MANIFEST.json` on the server
  - [ ] Request access to the gated datasets (full VoxCeleb), only if needed later
- [ ] **Stage 2: Per-factor verifiers.** Use pretrained embedding models (e.g. a face-embedding model for faces, SpeechBrain ECAPA for speaker verification) and train a small classifier for keystroke dynamics. Measure real **EER, FAR and FRR** for each factor.
- [ ] **Stage 3: Degradation curves.** Re-evaluate each verifier under simulated conditions (darkened or blurred faces, voice mixed with café or street noise at increasing levels) to learn **how EER changes with light, noise and motion**. This is the learned replacement for the paper's T-matrix.
- [ ] **Stage 4: Context models.** Train or evaluate an acoustic-scene classifier (environment and noise) and a human-activity classifier (walking vs still). Light comes directly from the device's light sensor, so no model is needed.
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
| Noise / environment | [`quinnlue/tau-urban-acoustic-scenes-2022-mobile-split5`](https://huggingface.co/datasets/quinnlue/tau-urban-acoustic-scenes-2022-mobile-split5) | 1.2 GB | ~10,000 `.wav` clips; labels in tab-separated files under `evaluation_setup/` (6,980 labeled training clips). 10 scenes: airport, bus, metro, metro station, park, public square, shopping mall, street (pedestrian), street (traffic), tram | ✅ **Use** for the noise classifier and as **real background noise** to mix into voice clips. Non-commercial research use only. No "café" class |
| Motion | [`DiFronzo/Human_Activity_Recognition`](https://huggingface.co/datasets/DiFronzo/Human_Activity_Recognition) | 7 MB | `raw_data.zip`: accelerometer + gyroscope at 50 Hz from a Samsung Galaxy S8 in a pocket; the zip contains 8 users / 12 experiments (the README mentions 16 volunteers), with a labels file; 5 activities: walking, walking upstairs, walking downstairs, sitting, standing | ✅ **Use** for a moving-vs-still classifier. Needs unzipping and parsing the labels file |

### Known gaps
- **Fingerprint, iris and hand biometrics:** no real public datasets on Hugging Face (they are privacy-restricted and usually need signed licence agreements). The "iris" dataset on Hugging Face is the flower dataset. These factors will stay simulated with the paper's values.
- **Device and network effects:** no public dataset records the same users logging in on different devices and networks. These effects will stay simulated unless we collect our own data.
- **Licences:** LFW and VoxCeleb are for research use only, which suits a university project but not a commercial product.
- **Voice evaluation may be optimistic:** mini VoxCeleb's speakers (IDs like `id10012`) come from the VoxCeleb1 *development* set, which popular pretrained speaker models (e.g. SpeechBrain ECAPA) were trained on. The standard unseen-speaker test set is VoxCeleb1-O (speakers `id10270`–`id10309`), available in the gated full VoxCeleb dataset.
- **Mouse dynamics:** the only mouse dataset found is for bot detection, so mouse dynamics will stay simulated unless a per-user dataset is found.
- **Compute:** the minimal version runs on a laptop CPU; full VoxCeleb, degradation sweeps and fine-tuning will run on the L4 GPU server.
