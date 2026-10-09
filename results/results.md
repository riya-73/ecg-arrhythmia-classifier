# MIT-BIH DS2 held-out results

Test beats: 50,612 (Normal 36,428; Abnormal 14,184).

| Model | Accuracy | ROC-AUC | PR-AUC | Normal P / R / F1 | Abnormal P / R / F1 |
|---|---:|---:|---:|---:|---:|
| 1D CNN | 0.7979 | 0.8573 | 0.7163 | 0.8011 / 0.9567 / 0.8720 | 0.7781 / 0.3899 / 0.5195 |
| Random forest | 0.7514 | 0.6778 | 0.5224 | 0.7702 / 0.9329 / 0.8438 | 0.6234 / 0.2853 / 0.3914 |

Confusion matrices are in `figures/confusion_matrices.png`. Metrics use DS2 only; model selection/early stopping uses DS1 validation.
