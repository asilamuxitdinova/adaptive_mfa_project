# Project Progress: Adaptive Multi-Factor Authentication

Tracks the progress of turning the adaptive MFA approach from the paper below into a working, data-driven system.

**Based on:** A. K. Nag, A. Roy, D. Dasgupta, *"An Adaptive Approach Towards the Selection of Multi-factor Authentication"*, IEEE Symposium Series on Computational Intelligence (SSCI), 2015. DOI: [10.1109/SSCI.2015.75](https://doi.org/10.1109/SSCI.2015.75)

**Last updated:** 2026-11-13

## Status overview

| Step | Description | Status |
|---|---|---|
| 1 | Simulation demo of the paper's adaptive selection approach | ✅ Done |
| 2 | Real ML-based adaptive MFA trained on public datasets | 🔜 Planned (useful datasets found) |

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

## Step 2: Real ML-based adaptive MFA 🔜

**Goal:** replace the paper's assumed trust values with **trust values learned from real data**, including how each factor's accuracy drops in darkness, noise and motion, and then reuse the adaptive selector.

### Planned stages

- [ ] **Stage 1: Dataset preparation.** Download and inspect the datasets below, check their licences, and build verification pairs (same person / different person).
- [ ] **Stage 2: Per-factor verifiers.** Use pretrained embedding models (e.g. a face-embedding model for faces, SpeechBrain ECAPA for speaker verification) and train a small classifier for keystroke dynamics. Measure real **EER, FAR and FRR** for each factor.
- [ ] **Stage 3: Degradation curves.** Re-evaluate each verifier under simulated conditions (darkened or blurred faces, voice mixed with café or street noise at increasing levels) to learn **how EER changes with light, noise and motion**. This is the learned replacement for the paper's T-matrix.
- [ ] **Stage 4: Context models.** Train or evaluate an acoustic-scene classifier (environment and noise) and a human-activity classifier (walking vs still). Light comes directly from the device's light sensor, so no model is needed.
- [ ] **Stage 5: Learned trust function.** Convert measured accuracy into trust values per factor and per context, replacing the hard-coded Table II.
- [ ] **Stage 6: Adaptive selector.** Plug the learned trust function into the selector from Step 1 (same two objectives). Later: try a contextual-bandit model that learns from login outcomes.
- [ ] **Stage 7: Evaluation.** Compare adaptive, static and random selection using measured trust and failure rates, and compare against the paper's results.

### Useful datasets found on Hugging Face

| Purpose | Dataset | Used for | Notes |
|---|---|---|---|
| Face verification | [`marcelohaps/lfw`](https://huggingface.co/datasets/marcelohaps/lfw) | Face EER; degradation under darkness and blur | Labeled Faces in the Wild, a standard benchmark. Research use |
| Voice verification | [`s3prl/mini_voxceleb1`](https://huggingface.co/datasets/s3prl/mini_voxceleb1) | Speaker EER; degradation under noise | Small subset, good for a laptop. Research use |
| Voice verification (full) | [`ProgramComputer/voxceleb`](https://huggingface.co/datasets/ProgramComputer/voxceleb) | Larger-scale speaker evaluation | Large download; GPU recommended. Research use |
| Keystroke dynamics | [`onepunchdimash/aalto_keystroke_features_normalized`](https://huggingface.co/datasets/onepunchdimash/aalto_keystroke_features_normalized) | Keystroke verifier and EER | Derived from the Aalto keystroke study. Contents still to be checked |
| Keystroke dynamics (large) | [`rokset3/keystrokes136M_normalized_features`](https://huggingface.co/datasets/rokset3/keystrokes136M_normalized_features) | Larger-scale keystroke training | Contents still to be checked |
| Mouse dynamics | [`dishamodi/mouse-dynamics`](https://huggingface.co/datasets/dishamodi/mouse-dynamics) | Mouse-dynamics verifier | Small; contents still to be checked |
| Noise / environment | [`quinnlue/tau-urban-acoustic-scenes-2022-mobile-split5`](https://huggingface.co/datasets/quinnlue/tau-urban-acoustic-scenes-2022-mobile-split5) | Recognizing café, street, metro, etc. | Based on the DCASE TAU Urban Acoustic Scenes challenge |
| Motion | [`DiFronzo/Human_Activity_Recognition`](https://huggingface.co/datasets/DiFronzo/Human_Activity_Recognition) | Walking vs still from phone sensors | Contents still to be checked |

### Known gaps
- **Fingerprint, iris and hand biometrics:** no real public datasets on Hugging Face (they are privacy-restricted and usually need signed licence agreements). The "iris" dataset on Hugging Face is the flower dataset. These factors will stay simulated with the paper's values.
- **Device and network effects:** no public dataset records the same users logging in on different devices and networks. These effects will stay simulated unless we collect our own data.
- **Licences:** LFW and VoxCeleb are for research use only, which suits a university project but not a commercial product.
- **Compute:** the minimal version (pretrained models evaluated on dataset subsets) runs on a laptop CPU; training from scratch on full VoxCeleb would need a GPU.
