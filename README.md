# ECG Arrhythmia Classifier (MIT-BIH)

A reproducible Python package that downloads the **real MIT-BIH Arrhythmia Database** from PhysioNet, extracts annotated heartbeat windows, and compares a compact 1D CNN with a handcrafted-feature random forest under an inter-patient train/validation/test split. This is a research demonstration, **not a medical device**.

## Measured patient-independent benchmark results

Held-out **DS2 only**, 50,612 beats (36,428 Normal; 14,184 Abnormal). No model selection or threshold tuning was done on DS2.

| Model | Accuracy | ROC-AUC | PR-AUC | Normal precision / recall / F1 | Abnormal precision / recall / F1 |
|---|---:|---:|---:|---:|---:|
| 1D CNN | 0.7979 | 0.8573 | 0.7163 | 0.8011 / 0.9567 / 0.8720 | 0.7781 / 0.3899 / 0.5195 |
| Random forest | 0.7514 | 0.6778 | 0.5224 | 0.7702 / 0.9329 / 0.8438 | 0.6234 / 0.2853 / 0.3914 |

CNN confusion matrix (actual rows; predicted columns; labels ordered Normal, Abnormal): `[[34851, 1577], [8654, 5530]]`. The main weakness is **low abnormal recall (38.99%)**: 8,654 of 14,184 abnormal beats were missed. Accuracy alone is therefore misleading. Training used 40,792 beats, validation 9,101 beats; early stopping retained epoch 6 by validation loss (9 epochs total). The Random Forest fit used a deterministic sample of 25,000 DS1 training beats.

See [`results/results.md`](results/results.md), [`results/metrics.json`](results/metrics.json), and the figures below. Metrics are from the executed corrected run, not estimates. An initial run used the common record split that places paired records 201 and 202 across DS1/DS2; it was discarded. The final run excludes record 201 and retains record 202 in DS2, and includes regression tests for this known same-subject pair.

## Method

- **Source:** PhysioNet MIT-BIH Arrhythmia Database, 360 Hz, downloaded via WFDB's targeted file client and read with WFDB `rdrecord`/`rdann`. MLII is selected when available. Paced records 102, 104, 107, and 217 are excluded.
- **Preprocessing:** fourth-order zero-phase Butterworth 0.5–40 Hz bandpass (baseline drift suppression), annotation-centered 252-sample (~0.70 s) windows, z-normalization per beat. `N` is Normal; every other annotated beat symbol is Abnormal. Annotation marker `+` and edge windows are discarded.
- **Split:** standard DS1/DS2 record lists, with validation records 115, 122, 220, 223 carved from DS1. DS1 record 201 is excluded from modeling because it shares a subject with held-out DS2 record 202. Final split: 17 training records, 4 validation records, and 22 test records (43 analyzed records); patient identity assertions and pytest checks guard all boundaries. DS2 is evaluation-only.
- **CNN:** compact three-stage 1D CNN, weighted cross-entropy, AdamW, ReduceLROnPlateau, early stopping. Seed 42; CPU threads capped at four.
- **Baseline:** random forest on morphology summaries, local annotation-to-annotation RR intervals, and shape descriptors; fit only on DS1 training.
- **Evaluation/explanations:** confusion matrices, per-class precision/recall/F1, ROC-AUC, average precision (PR-AUC), loss curves, six correct and six misclassified DS2 beats, and input-gradient saliency for three DS2 beats.

## Reproduce

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
bash scripts/run_all.sh
```

Run from the repository root. `scripts/run_all.sh` downloads the selected source files with WFDB (no synthetic fallback), preprocesses and caches `data/processed.npz`, trains/evaluates, then runs pytest. To rerun only the model pipeline with cached data: `PYTHONPATH=src python -m ecg_classifier.train`. Checkpoint: `models/cnn.pt`.

## Key figures

- Loss curves: [`results/figures/loss_curves.png`](results/figures/loss_curves.png)
- DS2 confusion matrices: [`results/figures/confusion_matrices.png`](results/figures/confusion_matrices.png)
- Correct/misclassified beats: [`results/figures/example_beats.png`](results/figures/example_beats.png)
- Three gradient-saliency examples: [`results/figures/saliency.png`](results/figures/saliency.png)

## Limitations

- Binary beat-level classification on a single ECG lead is **not** a patient-level diagnosis and is not a medical device or clinical decision-support system.
- MIT-BIH is a small historical cohort (48 recordings from 47 subjects). This analysis uses 43 records after omitting four paced records and record 201 to prevent same-subject overlap with DS2 record 202. The benchmark result does not establish generalization to current devices, other populations, hospitals, or rhythm prevalences.
- This is an imbalanced task. Abnormal recall is only 38.99%; a real clinical system needs prospective validation, clinically appropriate sensitivity targets, calibration, and external datasets.
- Labels collapse heterogeneous non-`N` beat types into a single Abnormal class. Windows and RR context use expert beat annotations; deployment would additionally require a robust R-peak detector.
- Saliency is qualitative, not causal or clinically validated.
- The user-provided `ecg_split.npz` was not used: it contains arrays but no record/patient provenance, so its split independence and labels cannot be audited against the required DS1/DS2 protocol.

## Resume bullets (based on this run)

- Built a reproducible PyTorch 1D CNN for patient-independent MIT-BIH heartbeat classification; achieved **0.857 ROC-AUC and 0.716 PR-AUC** on 50,612 held-out beats.
- Implemented WFDB acquisition, 0.5–40 Hz filtering, 0.7-second beat normalization, weighted-loss training, early stopping, and saliency visualization across 43 analyzed records.
- Benchmarked against a handcrafted RR/morphology random forest; improved DS2 accuracy from **75.14% to 79.79%**, while reporting the CNN's **38.99% abnormal recall** limitation.
