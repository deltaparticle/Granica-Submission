# Real-world validation log — washing machine dryer

First real-world test of this project's pipeline end to end, on genuinely
self-collected data (README Section 2.7 / 2.2 filter #6: independent
inspection before trust). Two machines: a known-faulty hostel-corridor
washing machine dryer at IIT Guwahati, and a second, healthy machine of the
same kind, recorded to provide a real normal-operating-condition baseline.

## 1. Data collection

**Hardware:** two Android phones available (CMF Phone 1, Moto G57); one was
used per recording session, placed flat and stationary directly on the
dryer's top cap, screen up, not handheld.

**Software:** [Sensor Logger](https://play.google.com/store/apps/details?id=com.kelvin.sensorapp)
by Logger Labs Ltd (free, Play Store) — records accelerometer and
microphone audio simultaneously in one session and exports a bundle of
CSVs plus the raw audio file.

**Sensors enabled:** Accelerometer only (Linear Acceleration and the raw
Total Acceleration channels are both exported automatically once
Accelerometer is toggled on) and Microphone/Audio. Gyroscope, Orientation,
Magnetometer, Compass, Location, and Camera were left off.

**Sessions recorded**, each ~10-20 seconds, phone stationary on the same
spot on the machine for the duration:

| Machine | State | Duration | Files |
|---|---|---|---|
| Faulty dryer | Running (dryer on) | 14.3s | `washing_machine_dryer_faulty/running/` |
| Faulty dryer | Silent (dryer off, ambient reference) | 11.0s | `washing_machine_dryer_faulty/silent/` |
| Healthy dryer | Running (dryer on) | 18.9s | `washing_machine_dryer_healthy/running/` |
| Healthy dryer | Silent (dryer off, ambient reference) | 8.0s | `washing_machine_dryer_healthy/silent/` |

Each session folder contains the app's raw export: `TotalAcceleration.csv`,
`Accelerometer.csv` (linear, gravity removed), `AccelerometerUncalibrated.csv`,
`Metadata.csv`, `Microphone.csv` (a coarse ~100ms dBFS loudness meter log,
not audio), `Microphone.mp4` (the actual audio), `Annotation.csv` (unused).

**Known collection issue:** both healthy-machine sessions were recorded in
a hostel corridor with real ambient noise (people, hallway acoustics)
audible in the audio track. Addressed in preprocessing below.

## 2. Preprocessing

Two new reader functions were added to `src/data/readers.py`
(`washing_machine_dryer_vibration_reader`, `washing_machine_dryer_audio_reader`)
rather than reusing any existing dataset's reader, since the file formats
don't match any existing dataset:

- **Vibration:** `TotalAcceleration.csv` (raw accelerometer, gravity
  included, x/y/z in m/s², verified near-perfectly regular at ~399.4Hz on
  every session — std of consecutive sample gaps <0.02ms). Chosen over the
  app's own gravity-removed `Accelerometer.csv` because it's delivered at
  roughly 2x the rate on this phone (~399Hz vs ~196Hz); gravity's DC offset
  is already suppressed downstream by the existing mel-spectrogram's
  `fmin=20Hz` floor, so nothing was lost by not using the pre-gravity-removed
  channel. Resampled onto a uniform time grid via linear interpolation
  (real sensor timestamps aren't guaranteed evenly spaced, unlike a lab
  DAQ). Reported as the axis magnitude `sqrt(x²+y²+z²)`, not a single axis,
  since phone placement orientation wasn't fixed the way a bolted lab
  accelerometer's mounting is.
- **Audio:** `Microphone.mp4` (AAC, 16kHz, stereo) decoded via PyAV
  (`av` package) and downmixed to mono by averaging channels. `soundfile`
  cannot read AAC/MP4 containers directly, and no system ffmpeg install was
  needed since PyAV binds the decoder directly.
- **Denoising:** the faulty machine's "running" audio was spectral-gate
  denoised (`noisereduce`, stationary mode) using its own paired "silent"
  clip (same spot, same session, machine off) as the noise profile, applied
  only where there's no circularity (Check 1 below, which scores against
  an externally-calibrated bank unrelated to either clip).
- Both signals then go through the exact same pipeline as every other
  dataset in this project: `signal_to_spectrogram_batch` (resample to the
  target encoder's training rate — 25.6kHz vibration / 16kHz audio — window
  at 1.5s/0.75s hop, log-mel spectrogram), then the real trained
  `SpectrogramEncoder` checkpoints (`vibration_dann_lambda0.3.pt`,
  `audio_dann_lambda0.15.pt`) produce one 128-d embedding per window.

Code: `scripts/eval/evaluate_real_world_washing_machine.py`. Full run log:
[`evaluation_log.txt`](evaluation_log.txt).

## 3. Checks run

**Check 1 — deployed memory banks.** Score the faulty machine's recordings
against the real, already-built `vibration_memory_bank.joblib` /
`audio_memory_bank.joblib` (calibrated on Engine Journal Bearings /
Car Diagnostics respectively) — the exact banks `predict.py` loads. A
cross-machine, cross-sensor zero-shot test: this washing machine
contributed zero data to that calibration.

**Check 2 — self-consistency.** Fit a PCA detector on the faulty machine's
own silent windows only, score its running windows against that — is
"running" distinguishable from this exact machine's own quiet state,
independent of any external calibration.

**Check 3 — train on healthy, test on faulty.** Fit the PCA memory bank on
the HEALTHY machine's running windows (a genuine normal-operating-condition
calibration, not silence), then score the FAULTY machine's running windows
against it. This is the project's actual deployed architecture, run for
real for the first time on self-collected data.

## 4. Results

### Check 1 — deployed memory banks (faulty machine only)

| Modality | Threshold | Running (faulty, on) | Silent (off, ambient) |
|---|---|---|---|
| Vibration | 2.12e-13 | mean 0.00672 — 100% flagged anomaly | mean 0.00329 — 100% flagged anomaly |
| Audio (denoised) | 1.15e-5 | mean 4.59e-5 — 100% flagged anomaly | mean 1.70e-5 — 92% flagged anomaly |

### Check 2 — self-consistency (faulty machine only)

| Modality | Threshold (90th pct of silent) | Running vs. silent-calibration |
|---|---|---|
| Vibration | 1.03e-13 | mean 0.205 — 100% flagged anomaly |
| Audio | 3.53e-13 | mean 0.00087 — 100% flagged anomaly |

### Check 3 — train on healthy, test on faulty

| Modality | Threshold (90th pct of healthy-running) | **Faulty running (the target test)** | Healthy silent (reference) | Faulty silent (reference) |
|---|---|---|---|---|
| **Vibration** | 6.12e-13 | **mean 0.00583 — 100% flagged anomaly** | mean 0.000225 | mean 0.00109 |
| Audio | 1.53e-13 | mean 0.000396 — 100% flagged anomaly | mean 0.000465 | mean 0.000341 |

In Check 3, vibration ranks faulty-running as clearly the highest-scoring
of the three real recordings (5x above faulty-silent, 26x above
healthy-silent). Audio's three scores are close to each other, without a
clear ranking.
