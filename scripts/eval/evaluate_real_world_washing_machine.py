"""
First real-world validation (README Section 2.7): scores two self-collected
recordings from an actual faulty washing-machine dryer (IIT Guwahati hostel
corridor) through the exact deployed pipeline (real trained encoders + the
real pre-built PCA memory banks from scripts/utils/build_memory_bank.py) —
the same code path predict.py uses, not a re-fit special-cased evaluation.

Two machines, two recordings each, all self-collected with the same phone
and app settings (see sample_data/real_world_validation/LOG.md for the full
collection + preprocessing write-up):
  - sample_data/real_world_validation/washing_machine_dryer_faulty/  — the
    FAULTY dryer: "running" (on, known faulty, not induced) / "silent"
    (same spot, off — ambient/idle reference only)
  - sample_data/real_world_validation/washing_machine_dryer_healthy/ — a
    second, HEALTHY machine: "running" (healthy, on — the real normal-
    operating-condition) / "silent" (off — ambient/idle reference only)

Three checks are reported:
  1. Deployment check: score the faulty machine's recordings against the
     REAL memory banks built from Car Diagnostics (audio) / Engine Journal
     Bearings (vibration) healthy data — the same banks predict.py loads.
     A genuine cross-machine, cross-sensor zero-shot test.
  2. Self-consistency check: fit a tiny PCA detector on the faulty
     machine's own silent windows, score its running windows against it —
     is "running" distinguishable from this exact machine's own quiet
     state, independent of any external calibration.
  3. Train-on-healthy / test-on-faulty (this project's actual deployed
     architecture, run for real for the first time): fit the PCA memory
     bank on the HEALTHY machine's running windows — a genuine normal-
     operating-condition calibration, not silence — then score the FAULTY
     machine's running windows against it. This is the real
     "calibrate-on-healthy, flag-the-anomaly" test the whole system is
     built around.

Audio denoising: the faulty machine's "running" recording was made in a
hostel corridor with real ambient noise (people, hallway acoustics)
contaminating the target machine sound. Its audio is spectral-gate
denoised (noisereduce, stationary mode) against its own paired "silent"
clip (same spot, same session, machine off) as the noise profile, and used
in Check 1 only, which scores it against externally-calibrated memory
banks (no circularity — the noise profile and the calibration source are
unrelated recordings). Checks 2 and 3 use raw audio throughout. Vibration
is untouched in all checks — spectral audio denoising doesn't have a
meaningful analogue for a ~400Hz accelerometer trace.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

import warnings
from pathlib import Path

import noisereduce as nr

import numpy as np
import torch

from cpu_guard import CPUGuard
from data.preprocess import SpectrogramConfig, signal_to_spectrogram_batch
from data.readers import (
    washing_machine_dryer_audio_reader,
    washing_machine_dryer_vibration_reader,
)
from models.encoder import load_encoder
from models.memory_bank import PCAReconstructionMemoryBank

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
# Committed (not gitignored) — anyone cloning the repo can re-run this
# exact check without needing to collect their own recordings first.
VALIDATION_DIR = REPO_ROOT / "sample_data" / "real_world_validation"
FAULTY_DIR = VALIDATION_DIR / "washing_machine_dryer_faulty"
HEALTHY_DIR = VALIDATION_DIR / "washing_machine_dryer_healthy"
CHECKPOINT_DIR = REPO_ROOT / "data" / "processed" / "checkpoints"
MEMORY_BANK_DIR = REPO_ROOT / "data" / "processed" / "memory_banks"


def denoise_with_reference(signal: np.ndarray, sr: int, noise_signal: np.ndarray, noise_sr: int) -> np.ndarray:
    """Spectral-gate denoise `signal` using `noise_signal` (a separate,
    machine-off recording from the same spot/session) as the noise profile.
    `stationary=True` computes one fixed noise threshold from the reference
    clip rather than adapting per-frame, appropriate since the noise clip
    isn't time-aligned with the target clip."""
    assert sr == noise_sr, f"sample rate mismatch: {sr} vs {noise_sr}"
    return nr.reduce_noise(y=signal, sr=sr, y_noise=noise_signal, stationary=True)


def embed(encoder, signal, orig_sr, cfg):
    specs = signal_to_spectrogram_batch(signal, orig_sr, cfg)
    batch = torch.from_numpy(specs).float().unsqueeze(1)
    with torch.no_grad():
        embs = encoder(batch).numpy()
    return embs  # (n_windows, 128) — one embedding per 1.5s window


def pca_reconstruction_error_score(fit_X: np.ndarray, X: np.ndarray) -> np.ndarray:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(fit_X)
    fit_scaled = scaler.transform(fit_X)
    n_components = max(min(32, fit_X.shape[0] - 1, fit_X.shape[1]), 1)
    pca = PCA(n_components=n_components, random_state=42).fit(fit_scaled)
    X_scaled = scaler.transform(X)
    reduced = pca.transform(X_scaled)
    reconstructed = pca.inverse_transform(reduced)
    return np.mean((X_scaled - reconstructed) ** 2, axis=1)


def main():
    guard = CPUGuard()
    if not guard.check("real-world washing machine eval setup"):
        return

    print("Loading encoders...")
    vib_encoder = load_encoder(CHECKPOINT_DIR / "vibration_dann_lambda0.3.pt")
    vib_encoder.eval()
    aud_encoder = load_encoder(CHECKPOINT_DIR / "audio_dann_lambda0.15.pt")
    aud_encoder.eval()

    vib_cfg = SpectrogramConfig(sample_rate=25600)
    aud_cfg = SpectrogramConfig(sample_rate=16000)

    print("Reading recordings...")
    def read_pair(root):
        vib_sig, vib_sr = washing_machine_dryer_vibration_reader(root / "TotalAcceleration.csv")
        aud_sig, aud_sr = washing_machine_dryer_audio_reader(root / "Microphone.mp4")
        return vib_sig, vib_sr, aud_sig, aud_sr

    vib_running_sig, vib_running_sr, aud_running_sig, aud_running_sr = read_pair(FAULTY_DIR / "running")
    vib_silent_sig, vib_silent_sr, aud_silent_sig, aud_silent_sr = read_pair(FAULTY_DIR / "silent")
    vib_h_running_sig, vib_h_running_sr, aud_h_running_sig, aud_h_running_sr = read_pair(HEALTHY_DIR / "running")
    vib_h_silent_sig, vib_h_silent_sr, aud_h_silent_sig, aud_h_silent_sr = read_pair(HEALTHY_DIR / "silent")

    print(f"  faulty:  vibration running {len(vib_running_sig)/vib_running_sr:.1f}s, silent {len(vib_silent_sig)/vib_silent_sr:.1f}s "
          f"| audio running {len(aud_running_sig)/aud_running_sr:.1f}s, silent {len(aud_silent_sig)/aud_silent_sr:.1f}s")
    print(f"  healthy: vibration running {len(vib_h_running_sig)/vib_h_running_sr:.1f}s, silent {len(vib_h_silent_sig)/vib_h_silent_sr:.1f}s "
          f"| audio running {len(aud_h_running_sig)/aud_h_running_sr:.1f}s, silent {len(aud_h_silent_sig)/aud_h_silent_sr:.1f}s")

    print("Denoising faulty machine's 'running' audio against its own paired 'silent' clip (Check 1 only)...")
    aud_running_sig_denoised = denoise_with_reference(aud_running_sig, aud_running_sr, aud_silent_sig, aud_silent_sr)

    print("Extracting embeddings...")
    vib_running_emb = embed(vib_encoder, vib_running_sig, vib_running_sr, vib_cfg)
    vib_silent_emb = embed(vib_encoder, vib_silent_sig, vib_silent_sr, vib_cfg)
    # RAW audio embeddings — used by Checks 2 and 3.
    aud_running_emb = embed(aud_encoder, aud_running_sig, aud_running_sr, aud_cfg)
    aud_silent_emb = embed(aud_encoder, aud_silent_sig, aud_silent_sr, aud_cfg)
    vib_h_running_emb = embed(vib_encoder, vib_h_running_sig, vib_h_running_sr, vib_cfg)
    vib_h_silent_emb = embed(vib_encoder, vib_h_silent_sig, vib_h_silent_sr, vib_cfg)
    aud_h_running_emb = embed(aud_encoder, aud_h_running_sig, aud_h_running_sr, aud_cfg)
    aud_h_silent_emb = embed(aud_encoder, aud_h_silent_sig, aud_h_silent_sr, aud_cfg)
    # DENOISED audio embedding — used by Check 1 only.
    aud_running_emb_dn = embed(aud_encoder, aud_running_sig_denoised, aud_running_sr, aud_cfg)
    print(f"  faulty windows:  vibration running={len(vib_running_emb)} silent={len(vib_silent_emb)} "
          f"| audio running={len(aud_running_emb)} (denoised={len(aud_running_emb_dn)}) silent={len(aud_silent_emb)}")
    print(f"  healthy windows: vibration running={len(vib_h_running_emb)} silent={len(vib_h_silent_emb)} "
          f"| audio running={len(aud_h_running_emb)} silent={len(aud_h_silent_emb)}")

    # -----------------------------------------------------------------
    # Check 1: score against the REAL deployed memory banks (same ones
    # predict.py loads) — a genuine cross-machine zero-shot test.
    # -----------------------------------------------------------------
    print(f"\n{'='*70}\nCHECK 1 — Deployed memory banks (Car Diagnostics / Engine Journal Bearings)\n{'='*70}")
    vib_bank = PCAReconstructionMemoryBank.load(MEMORY_BANK_DIR / "vibration_memory_bank.joblib")
    aud_bank = PCAReconstructionMemoryBank.load(MEMORY_BANK_DIR / "audio_memory_bank.joblib")

    for name, bank, running_emb, silent_emb, note in [
        ("Vibration", vib_bank, vib_running_emb, vib_silent_emb, ""),
        ("Audio", aud_bank, aud_running_emb_dn, aud_silent_emb, " (running audio denoised)"),
    ]:
        running_scores = bank.score(running_emb)
        silent_scores = bank.score(silent_emb)
        print(f"\n[{name}]{note}  threshold = {bank.threshold:.6g}")
        print(f"  running (dryer ON, faulty): mean score = {running_scores.mean():.6g}  "
              f"({(running_scores > bank.threshold).mean()*100:.0f}% of windows flagged ANOMALY)")
        print(f"  silent  (dryer OFF, ambient): mean score = {silent_scores.mean():.6g}  "
              f"({(silent_scores > bank.threshold).mean()*100:.0f}% of windows flagged ANOMALY)")

    # -----------------------------------------------------------------
    # Check 2: self-consistency — fit on silent's own windows, score
    # running against it. Narrower question, same machine only.
    # -----------------------------------------------------------------
    print(f"\n{'='*70}\nCHECK 2 — Self-consistency (silent windows as this machine's own calibration, RAW audio — see module docstring)\n{'='*70}")
    for name, running_emb, silent_emb in [
        ("Vibration", vib_running_emb, vib_silent_emb),
        ("Audio", aud_running_emb, aud_silent_emb),
    ]:
        if len(silent_emb) < 2:
            print(f"\n[{name}] too few silent windows ({len(silent_emb)}) to fit a self-consistency check — skipped.")
            continue
        silent_scores = pca_reconstruction_error_score(silent_emb, silent_emb)
        running_scores = pca_reconstruction_error_score(silent_emb, running_emb)
        threshold = float(np.percentile(silent_scores, 90.0))
        print(f"\n[{name}]  threshold (90th pct of silent's own scores) = {threshold:.6g}")
        print(f"  running vs. silent-calibration: mean score = {running_scores.mean():.6g}  "
              f"({(running_scores > threshold).mean()*100:.0f}% of windows flagged ANOMALY)")

    # -----------------------------------------------------------------
    # Check 3: the real deployed architecture — calibrate on a HEALTHY
    # machine's own running (normal-operating-condition) windows, then
    # score the FAULTY machine's running windows against that.
    # -----------------------------------------------------------------
    print(f"\n{'='*70}\nCHECK 3 — Train on healthy, test on faulty (real deployed architecture)\n{'='*70}")
    for name, healthy_running_emb, faulty_running_emb, healthy_silent_emb, faulty_silent_emb, note in [
        ("Vibration", vib_h_running_emb, vib_running_emb, vib_h_silent_emb, vib_silent_emb, ""),
        ("Audio", aud_h_running_emb, aud_running_emb, aud_h_silent_emb, aud_silent_emb, ""),
    ]:
        cal_scores = pca_reconstruction_error_score(healthy_running_emb, healthy_running_emb)
        threshold = float(np.percentile(cal_scores, 90.0))
        print(f"\n[{name}]{note}  calibrated on {len(healthy_running_emb)} healthy-running windows  "
              f"threshold (90th pct) = {threshold:.6g}")

        faulty_running_scores = pca_reconstruction_error_score(healthy_running_emb, faulty_running_emb)
        print(f"  FAULTY running (the actual target test): mean score = {faulty_running_scores.mean():.6g}  "
              f"({(faulty_running_scores > threshold).mean()*100:.0f}% of windows flagged ANOMALY)")

        healthy_silent_scores = pca_reconstruction_error_score(healthy_running_emb, healthy_silent_emb)
        print(f"  healthy silent (different regime, reference only): mean score = {healthy_silent_scores.mean():.6g}  "
              f"({(healthy_silent_scores > threshold).mean()*100:.0f}% of windows flagged ANOMALY)")

        faulty_silent_scores = pca_reconstruction_error_score(healthy_running_emb, faulty_silent_emb)
        print(f"  faulty silent (different regime, reference only): mean score = {faulty_silent_scores.mean():.6g}  "
              f"({(faulty_silent_scores > threshold).mean()*100:.0f}% of windows flagged ANOMALY)")

    print("\nDone.")


if __name__ == "__main__":
    main()
