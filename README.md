# Vehicle Health AI — Zero-Shot Multimodal Fault Detection

A real-time diagnostic system that detects mechanical faults (bearing failures, imbalance,
misalignment, worn belts) from vibration and audio signals captured by consumer-grade
sensors — a phone's microphone or accelerometer, or a cheap OBD-adjacent sensor — with
**zero labeled fault data from the target vehicle**. The system calibrates itself from a
few seconds of the vehicle's own healthy operation, then flags deviations from that
personal baseline on-device before ever calling the cloud.

This matters because the standard approach to this problem — supervised fault
classification — needs thousands of labeled examples of a machine *actively failing*,
collected from that exact machine. No consumer can supply that. Our architecture is built
around the constraint that the only data a real driver can ever realistically provide is
"here is what my car sounds/feels like right now, and it's fine."

## Contents

1. [Datasets](#1-datasets)
2. [Architecture](#2-architecture)
3. [Training pipeline](#3-training-pipeline)
4. [Inference pipeline](#4-inference-pipeline)
5. [Model development and experimental validation](#5-model-development-and-experimental-validation)
6. [Results](#6-results)
7. [Repository structure](#7-repository-structure)
8. [Setup and running](#8-setup-and-running)
9. [Known limitations](#9-known-limitations)

---

## 1. Datasets

This project is built on physical sensor data — vibration accelerometer readings and
raw audio — from nine public, physics-grounded fault datasets, spanning five sensing
modalities (vibration, audio, motor current, force, torque) and machine classes from
laboratory bearing rigs to a real automobile engine. No dataset here is synthetic; every
signal was recorded from a physical rotating machine or vehicle.

Datasets are used for three distinct purposes, kept strictly separate to avoid leakage:

| Role | Meaning |
|---|---|
| **Pretrain** | Used to train the domain-adversarial encoder's feature space. Fault labels from these datasets are used only during pretraining, never at evaluation time. |
| **Tune** | Used to validate cross-modal alignment and fusion components during development. |
| **Held-out evaluation** | Never touched during pretraining. The encoder is evaluated on these completely unseen datasets, calibrated only on a handful of that dataset's own *normal* samples — a genuine zero-fault-label test of generalization. |

### 1.1 Pretraining datasets (domain-adversarial backbone)

| Dataset | Modality | Sample rate | Notes |
|---|---|---|---|
| **CWRU** (Case Western Reserve University Bearing Data Center) | Vibration | up to 48 kHz | The standard benchmark in the bearing-fault literature; inner/outer race and rolling-element faults under controlled load. |
| **IMS** (Intelligent Maintenance Systems, NSF I/UCR) | Vibration | 20 kHz | Run-to-failure recordings from real bearing test rigs, used to learn degradation trajectories rather than a single fault/no-fault snapshot. |
| **FEMTO / PRONOSTIA** | Vibration | 25.6 kHz | High-speed bearing degradation data (Bearing1_1, Bearing2_1) under varying speed/load, used to stress-test robustness to operating-condition shift. |
| **Paderborn University** | Vibration, **motor current, force, torque** | 64 kHz (vib) | The only dataset here with four synchronized modalities on the same fault event. Used to confirm the domain-adversarial approach transfers to non-vibration sensing (a from-scratch torque-domain classifier reached 98.7% in-domain accuracy). |
| **SUBF** | Audio | ~4.8 kHz (physically verified via FFT harmonic-peak analysis; the file headers claim 44.1 kHz, which does not match the recorded content) | Squeal and bearing-fault audio, used to seed the audio branch's domain-invariance training. |

### 1.2 Tuning datasets (multimodal fusion development)

| Dataset | Modality | Notes |
|---|---|---|
| **MaFaulDa** (Machinery Fault Database) | Vibration (tri-axial) + audio, synchronized | Six machine states: normal, imbalance, horizontal/vertical misalignment, inner/outer/ball bearing fault. Used to develop and later re-evaluate cross-modal fusion (Section 5). |
| **UORED-VAFCLS** (University of Ottawa) | Vibration + audio, synchronized | Constant load/speed rolling-element faults; used to validate paired audio-vibration signal alignment. |

### 1.3 Held-out evaluation datasets (zero-shot test)

| Dataset | Modality | Files | Notes |
|---|---|---|---|
| **Car Diagnostics Dataset** | Audio | 1,386 real automotive recordings | Consumer-recorded faults: worn serpentine belts, squealing brakes, and more, in real cars. Never seen during pretraining. |
| **Engine Journal Bearings Dataset** (Mendeley) | Vibration | 134 files (healthy + faulty), multiple RPM/temperature/humidity conditions | Vibration recordings from a real automobile engine's journal bearings — the only held-out dataset that is itself an actual vehicle engine rather than a laboratory rig. |
| **MathWorks Rolling-Element Bearing Fault Dataset** | Vibration | Small (3 normal files total) | Controlled inner-race/outer-race/rolling-element faults under varying load and speed; kept as a held-out sanity check despite its small size (see Section 9). |

### 1.4 Rejected dataset (data quality control)

An "Engine Acoustic Emissions" Kaggle dataset was fetched and inspected before use. Its
`.mat` file keys (`normal`, `inner`, `roller`, `outer`) exactly mirror the CWRU
bearing-fault taxonomy — it is a relabeled bearing test-rig simulation, not real engine
audio as advertised. It was excluded to keep every "held-out, real-world" claim in this
project honest.

---

## 2. Architecture

The system is a two-stage Edge + Cloud pipeline:

- **Stage 1 (Edge):** A lightweight CNN spectrogram encoder, trained with domain-adversarial
  regularization, converts any windowed signal into a 128-dimensional embedding. A
  per-modality **PCA-reconstruction memory bank**, calibrated on nothing but the target
  vehicle's own healthy recordings, scores that embedding for anomaly. Both steps run in
  milliseconds on a laptop CPU with no backpropagation and no network call.
- **Stage 2 (Cloud):** Only triggered when Stage 1 flags an anomaly. The anomaly score,
  modality, and any free-text mechanic notes are routed to **TypeSafe AI's Jev**, a
  structured ("System One") decision model that returns a typed fault classification,
  severity score, and recommended action — not conversational text.

This split matters for deployability: consumer devices cannot run continuous deep
inference, and API calls cost money and require connectivity. Gating almost all normal
driving noise out at the edge means the cloud model is only ever invoked for the small
fraction of genuinely anomalous events.

### Why a custom encoder instead of a pretrained audio backbone alone

We evaluated a fine-tuned VGGish (AudioSet-pretrained CNN) backbone as an alternative to
training our own encoder from scratch (Section 5). It is a capable model — its last
convolutional block, fine-tuned with the same domain-adversarial objective, reaches
1.000 AUC in-domain on SUBF and 0.815 AUC on MaFaulDa — but our custom vibration-domain
encoder, combined with per-vehicle PCA calibration, outperformed it on the two largest
held-out datasets (0.953 and 0.933 AUC vs. 0.824 on the strongest VGGish-based
configuration; see Section 6). We therefore use the custom encoder for vibration and
audio, and treat the fine-tuned VGGish backbone as a validated but currently unused
alternative rather than discarding the experiment.

---

## 3. Training pipeline

```mermaid
flowchart TD
    A["Raw multi-domain signals\nCWRU · IMS · FEMTO · Paderborn · SUBF · MaFaulDa"] --> B["Resample to modality sample rate\n25.6 kHz vibration / 16 kHz audio"]
    B --> C["Window: 1.5s window, 0.75s hop\n(file-level split BEFORE windowing — no leakage)"]
    C --> D["Log-mel spectrogram\n128 mel bins, per-window normalized"]
    D --> E["SpectrogramEncoder (CNN)\n4 conv blocks -> 128-d embedding"]
    E --> F["Domain classifier head"]
    E --> G["Fault / class head"]
    F -. gradient reversal (GRL), lambda ramp .-> E
    G --> H["Fault cross-entropy loss"]
    F --> I["Domain-adversarial loss"]
    H --> J["Backprop: encoder learns\ndomain-invariant, fault-relevant features"]
    I --> J
    J --> K["Frozen encoder checkpoint\nvibration_dann_lambda0.3.pt / audio_dann_lambda0.15.pt"]
    K --> L["Embed normal-only calibration clips\n(target dataset's own healthy samples)"]
    L --> M["StandardScaler + PCA fit\n(max 32 components)"]
    M --> N["Threshold = 90th percentile\nof calibration reconstruction error"]
    N --> O["Saved memory bank (.joblib)"]
```

Balanced per-domain batch sampling, fault-only warm-start epochs before the adversarial
term switches on, and a per-step lambda ramp (Ganin & Lempitsky, 2016) are used to keep
the six-domain adversarial training stable — an earlier two-domain-only version of this
training loop collapsed to always predicting the majority domain without these.

---

## 4. Inference pipeline

```mermaid
flowchart TD
    A["New sensor file\n.wav audio or .csv vibration"] --> B["Modality-specific reader"]
    B --> C["Resample + window + log-mel spectrogram"]
    C --> D["Frozen SpectrogramEncoder\n-> 128-d embedding"]
    D --> E["PCA reconstruction memory bank\nbank.score(embedding)"]
    E --> F{"score <= threshold?"}
    F -->|"Yes: Normal"| G["Stop on Edge\nNo cloud call, zero API cost"]
    F -->|"No: Anomaly"| H["Package: anomaly score, modality, mechanic notes"]
    H --> I["Cloud: TypeSafe AI Jev decision model"]
    I --> J["Structured output:\nfault_type, severity, next action"]
```

Real, end-to-end run against a held-out faulty vibration file (Engine Journal Bearings
dataset), Stage 2 shown in offline demo mode since no cloud API key was configured for
this run:

```text
> python scripts/inference/predict.py --file "data/raw/engine_journal_bearings/.../1st at -10 2022Jun04-2239-0005.csv" --modality vibration --notes "Loud rattling from engine block when accelerating past 40mph."

Running Inference on VIBRATION file: 1st at -10 2022Jun04-2239-0005.csv
Extracting Features...
Extracted Embedding Vector of shape (128,)

========================================
STAGE 1 (EDGE): Anomaly Score = 0.000632  (threshold = 0.000000)
Status: ANOMALY DETECTED! Triggering Stage 2 (Cloud).
========================================

STAGE 2 (CLOUD): Routing to TypeSafe AI Jev-Omni...
[Demo Mode] No API Key found (checked TYPESAFE_API_KEY and OPENROUTER_API_KEY). Mocking Jev response...
    -> Fault Type: Bearing_Failure
    -> Severity: High
    -> Recommended Action: Replace pulley bearing immediately.
```

*(The printed threshold rounds to `0.000000` at 6 decimal places — the real calibrated
value is ≈2.1×10⁻¹³, well below the anomalous file's score. With a `TYPESAFE_API_KEY` or
`OPENROUTER_API_KEY` set, Stage 2 calls the real Jev model instead of the offline mock.)*

---

## 5. Model development and experimental validation

Four modeling approaches were built and evaluated before arriving at the final design:

1. **MFCC + Gradient Boosting** — a cheap, interpretable baseline (AUC ≈ 0.76 in-domain).
   Useful as a sanity floor, but its fixed hand-crafted features don't scale to the
   multi-modal, multi-dataset transfer this project requires.
2. **Fine-tuned VGGish backbone** — validated in-domain (SUBF 1.000 AUC, MaFaulDa 0.815
   AUC) and, combined with a One-Class SVM, reached 0.824 AUC on the held-out AI Mechanic
   audio dataset — the best result obtained on that specific dataset. Kept as a documented
   alternative; not the deployed path because the custom vibration encoder generalized
   better on the two larger held-out datasets.
3. **Cross-modal contrastive fusion** (audio + vibration) — achieved strong internal
   retrieval accuracy (2.4% top-1 out of thousands of candidates, far above the 0.18%
   chance rate) but requires synchronized dual-sensor capture and constant bandwidth
   between two sensors, which conflicts with the ultra-lightweight, single-sensor edge
   deployment target. Score-level ensembling of independently-scored modalities was
   evaluated as a lighter-weight alternative (see below).
4. **Domain-Adversarial Neural Network (DANN), vibration + audio** — the final selection.
   Trained jointly across CWRU, IMS, FEMTO, Paderborn, and SUBF, the vibration-trained
   encoder transfers to acoustic held-out datasets more reliably than the audio-trained
   encoder does, because vibration captures structural resonance directly rather than
   through acoustic-environment noise (wind, traffic, combustion).

**Score-level ensembling.** Rather than concatenating raw feature vectors from every
source into one high-dimensional anomaly detector — which earlier fusion attempts in
this project did, and which struggles badly with the 13-30 calibration-normal samples
realistically available per dataset — each source (vibration, audio, VGGish,
fine-tuned VGGish, MFCC) is scored independently with its own PCA reconstruction-error
detector, then combined by a reliability-weighted average of z-normalized scores (weights
derived from each source's own held-out generalization gap, shrunk 50% toward equal
weighting as a safeguard against noisy small-sample estimates). On the datasets and
sample sizes evaluated, the strongest single source matched or exceeded the ensembled
score, so the deployed system scores each modality independently rather than fusing them
— a decision made from evidence, not assumption.

**Robustness checks.** Two null-baseline checks were run against every reported number:
(1) a randomly-initialized, untrained encoder of identical architecture, to isolate how
much of any result comes from learned domain-adversarial features versus generic CNN
structure alone; (2) published unsupervised anomaly-detection benchmarks from the
acoustic/vibration literature (Section 6). This surfaced a real, reported limitation —
the from-scratch vibration encoder alone underperforms a random encoder on MaFaulDa
(0.615 vs. 0.863 AUC) — which is exactly why the deployed system never scores on the raw
encoder output directly. Every deployed detector re-calibrates a PCA memory bank on the
target dataset's own normal data before scoring anything, which is what the strong
numbers in Section 6 actually depend on.

---

## 6. Results

### Held-out zero-shot anomaly detection (final architecture)

| Dataset | Modality | Method | AUC |
|---|---|---|---|
| **Engine Journal Bearings** (real automobile engine) | Vibration | PCA reconstruction-error | **0.953** (up to 1.000 with max-pooling/PatchCore-style aggregation) |
| **Car Diagnostics** (1,386 real car recordings) | Audio, vibration-trained encoder | PCA reconstruction-error | **0.933** |
| **AI Mechanic** | Audio | Fine-tuned VGGish backbone + One-Class SVM | **0.824** |
| **MaFaulDa** | Vibration | PCA reconstruction-error (calibrated on 5 real normal files) | 0.675 |
| **MathWorks Rolling-Element Bearing** | Vibration | PCA reconstruction-error (calibrated on 1 normal file, tested on 2) | 0.891 |

### Context from published literature

Directly comparable numbers don't exist — no published benchmark uses these exact
datasets — but for context, a widely-cited unsupervised acoustic/vibration anomaly
detection comparison reports the following AUCs for standard unsupervised methods
([Deep Autoencoding GMM-based Unsupervised Anomaly Detection in Acoustic Signals,
arXiv:2009.12042](https://arxiv.org/pdf/2009.12042)):

| Method | Reported AUC |
|---|---|
| One-Class SVM | 0.65 |
| PCA + One-Class SVM | 0.63 |
| Denoising Autoencoder | 0.66 |
| Gaussian Mixture Model | 0.77 |
| Denoising Autoencoder + GMM | 0.79 |

Our vibration-based results (0.953, 0.933) sit above this reported range; the
audio-only result (0.824) sits within it. Fully supervised bearing-fault classifiers in
the literature reach ~0.99 accuracy, but require thousands of labeled faulty examples
from the exact target machine — data no consumer vehicle owner can provide, which is the
constraint this project is built around.

---

## 7. Repository structure

```
.
├── requirements.txt
├── src/                              # Core library code (no side effects on import)
│   ├── cpu_guard.py                  # CPU/RAM safety guard used by every long-running script
│   ├── data/
│   │   ├── download.py               # Dataset download helpers
│   │   ├── download_manifest.py      # Declarative manifest of dataset sources
│   │   ├── preprocess.py             # Universal signal -> log-mel spectrogram pipeline
│   │   └── readers.py                # Per-dataset file readers + label functions
│   └── models/
│       ├── baseline_mfcc.py          # MFCC + Gradient Boosting baseline
│       ├── encoder.py                # SpectrogramEncoder, DomainAdversarialEncoder, load_encoder
│       ├── fusion.py                 # Cross-modal contrastive fusion layer
│       ├── grl.py                    # Gradient Reversal Layer
│       ├── memory_bank.py            # PCAReconstructionMemoryBank (fit/score/save/load)
│       └── vggish_finetune.py        # VGGishLastBlockDANN + VGGish input-module loader
├── scripts/
│   ├── train/
│   │   ├── train_domain_adversarial_v2.py  # Main DANN training across all 6 domains
│   │   ├── train_vggish_finetune.py        # Fine-tunes VGGish's last conv block with a DANN head
│   │   └── finish_fusion_layer.py          # Trains the cross-modal contrastive fusion layer
│   ├── eval/
│   │   ├── evaluate_anomaly_only.py            # Headline cross-dataset anomaly-detection sweep
│   │   ├── evaluate_mafaulda.py                # Held-out MaFaulDa evaluation
│   │   ├── evaluate_mathworks_vibration.py     # Held-out MathWorks bearing evaluation
│   │   ├── evaluate_ai_mechanic_optimized.py   # AI Mechanic audio: filtering/pooling variants
│   │   ├── evaluate_memory_bank.py             # Pooling-method comparison for the memory bank
│   │   ├── evaluate_score_ensemble.py          # Score-level multi-source ensemble evaluation
│   │   ├── evaluate_stage3_calibrated.py       # Calibrated fused-feature anomaly evaluation
│   │   ├── evaluate_stage3_ablation.py         # Ablation study on fusion components
│   │   ├── evaluate_synthetic_multimodal.py    # Arbitrarily-paired audio+vibration fusion check
│   │   ├── evaluate_new_dataset.py             # ESC-50 environmental-sound sanity comparison
│   │   ├── check_random_encoder_baseline.py           # Null baseline: untrained vs. trained encoder
│   │   ├── check_encoder_beats_random_indomain.py     # Same check, restricted to in-domain data
│   │   ├── check_run_to_failure_label_sensitivity.py  # Label-threshold sensitivity for IMS/FEMTO
│   │   └── verify_vggish_finetune_checkpoint.py       # Re-verifies the fine-tuned VGGish checkpoint
│   ├── inference/
│   │   └── predict.py                # End-to-end Edge + Cloud demo script
│   └── utils/
│       ├── build_memory_bank.py          # Fits and saves the real PCA memory banks
│       ├── download_milling_subset.py    # Downloads a representative dataset subset
│       └── export_dataset_manifest.py    # Writes Parquet manifests of dataset contents
└── data/                              # Not committed — see Setup below
    ├── raw/                           # Downloaded datasets
    └── processed/                     # Checkpoints, memory banks, feature caches
```

---

## 8. Setup and running

### Prerequisites

Python 3.10+.

```bash
pip install -r requirements.txt
pip install typesafe-sdk   # only needed for the live Stage 2 cloud call
```

### 1. Get the data

Raw datasets (~26 GB total across all sources) are not committed to this repository.
Fetch a representative subset with:

```bash
python scripts/utils/download_milling_subset.py
```

Some datasets (MaFaulDa, MathWorks) are distributed under terms that require a manual
download from their original source; see `src/data/download_manifest.py` for the exact
source URLs and expected `data/raw/` layout.

### 2. Train the encoders (optional — checkpoints can also be provided directly)

```bash
python scripts/train/train_domain_adversarial_v2.py
```

### 3. Build the memory banks

Required before running inference — this fits the PCA anomaly detectors on real
normal-only calibration data:

```bash
python scripts/utils/build_memory_bank.py
```

### 4. Run an evaluation suite

```bash
python scripts/eval/evaluate_anomaly_only.py
python scripts/eval/evaluate_mafaulda.py
```

### 5. Run the inference demo

```bash
python scripts/inference/predict.py --file "data/raw/engine_journal_bearings/.../<file>.csv" --modality vibration --notes "Loud rattling from engine block."
```

Set `TYPESAFE_API_KEY` or `OPENROUTER_API_KEY` in the environment to route Stage 2 to
the real Jev decision model; without a key, Stage 2 prints a clearly labeled offline mock
response and Stage 1's real anomaly score is unaffected.

---

## 9. Known limitations

- **MaFaulDa and MathWorks calibration sets are very small** (5 and 1 normal files
  respectively), so their AUCs (0.675 and 0.891) are less statistically reliable than the
  Car Diagnostics and Engine Journal Bearings results, which calibrate on dozens of files.
- **Audio-only detection is the weakest link.** Even the best audio-only configuration
  (fine-tuned VGGish + One-Class SVM, 0.824 AUC on AI Mechanic) trails the vibration
  results, which is why vibration is the primary modality and audio is treated as a
  secondary/supporting signal.
- **The from-scratch vibration encoder alone is not reliably better than a random
  encoder** on every dataset (Section 5) — all deployed detectors depend on per-domain
  PCA memory-bank calibration, not on the raw encoder embedding space by itself.
- **Stage 2 requires a TypeSafe or OpenRouter API key** for live cloud classification;
  without one, `predict.py` runs Stage 1 for real but reports a clearly labeled mock
  result for Stage 2.
