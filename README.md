# WiFiVision

## What is this project?

WiFiVision is a CSI-based human pose estimation project. The goal is to predict **17 human body joints in 2D coordinates** using WiFi Channel State Information (CSI).

Current best verified model:

**Topology Graph Pose Model**
- Best MPJPE: **0.138569**
- Reproducible amplitude baseline: **0.152493**
- Improvement: **9.13%**

---

# 1. What Did We Do Till Now?

## Dataset and preprocessing

Completed:
- Loaded and processed CSI data.
- Created temporal sliding windows.
- Prepared pose targets with **17 joints × 2 coordinates**.
- Created training, validation and test splits.
- Applied normalization for training and denormalization for evaluation.

Current dataset:

| Split | Windows |
|---|---:|
| Training | 3402 |
| Validation | 486 |
| Test | 972 |

Input shape:

```text
(30, 3, 114, 10)
```

Target shape:

```text
(17, 2)
```

---

## Experiments completed

### 1. Amplitude Baseline

This is the main reproducible baseline.

```text
MPJPE: 0.152493
```

Files:

```text
models/baseline_best.pt

data/processed/baseline/
├── baseline_predictions.npy
├── baseline_targets.npy
├── baseline_errors.npy
└── baseline_analysis.npz
```

---

### 2. Phase Experiment

A phase-based CSI model was tested.

```text
Checkpoint: models/phase_best.pt
```

It was not selected because it did not produce a verified improvement over the reproducible baseline.

---

### 3. Spatial Experiment

A spatial CNN-based architecture was tested.

```text
MPJPE: 0.201713
```

It performed worse than the baseline.

---

### 4. Spatial Attention Experiment

A spatial attention architecture was tested.

```text
Checkpoint: models/spatial_attention_best.pt
```

No verified improvement over the baseline.

---

### 5. Temporal Transformer Experiment

A Transformer-based temporal architecture was tested.

```text
Checkpoint: models/temporal_transformer_best.pt
```

It did not provide a verified improvement over the baseline.

---

### 6. PCA + STFT Experiments

Two checkpoints were generated:

```text
models/pca_stft_best.pt
models/pca_stft_corrected_best.pt
```

The corrected experiment addressed a preprocessing/evaluation inconsistency.

Neither was selected as the final model.

---

### 7. Experiment G — Advanced CSI Pose Model

Model parameters:

```text
217,027
```

Final result:

```text
MPJPE: 0.158880
```

Comparison:

```text
Baseline:     0.152493
Experiment G: 0.158880
```

Experiment G performed worse than the reproducible baseline.

---

# 2. Final Best Model — Topology Graph

The current best model is the **Topology Graph Pose Model**.

Architecture:

```text
CSI Input
   ↓
CNN Frame Feature Extraction
   ↓
GRU Temporal Modeling
   ↓
Joint Feature Projection
   ↓
Graph Convolution
   ↓
Attention
   ↓
17 × 2 Pose Prediction
```

Checkpoint:

```text
models/topology_graph_best.pt
```

Model parameters:

```text
67,554
```

Checkpoint information:

```text
Best Epoch: 2
Validation Loss: 1.1400491405171131
```

---

# 3. Final Results

Test windows:

```text
972
```

| Metric | Amplitude Baseline | Topology Graph |
|---|---:|---:|
| MPJPE | 0.152493 | 0.138569 |
| Improvement | — | 9.13% |

Topology Graph PCK:

| Threshold | PCK |
|---|---:|
| 0.05 | 27.08% |
| 0.10 | 55.42% |
| 0.15 | 69.21% |
| 0.20 | 78.09% |

Final conclusion:

```text
Topology Graph is the current best verified model.
```

---

# 4. Per-Joint Results

```text
Joint 01: 0.089723
Joint 02: 0.090741
Joint 03: 0.087598
Joint 04: 0.081320
Joint 05: 0.089460
Joint 06: 0.089562
Joint 07: 0.084970
Joint 08: 0.098195
Joint 09: 0.115101
Joint 10: 0.141703
Joint 11: 0.138345
Joint 12: 0.108900
Joint 13: 0.188275
Joint 14: 0.340026
Joint 15: 0.110574
Joint 16: 0.177564
Joint 17: 0.323622
```

Best predicted joint:

```text
Joint 04 → 0.081320
```

Most difficult joint:

```text
Joint 14 → 0.340026
```

---

# 5. Subject-Wise Evaluation

Test subjects:

```text
Subject 9
Subject 10
```

| Subject | MPJPE |
|---|---:|
| Subject 9 | 0.138242 |
| Subject 10 | 0.138897 |

```text
Mean subject MPJPE: 0.138569
MPJPE standard deviation: 0.000328
```

The model performed consistently across both test subjects.

---

# 6. Error Distribution

## Baseline

```text
Mean error:   0.152493
Median error: 0.102985
```

## Topology Graph

```text
Mean error:   0.138569
Median error: 0.085739
```

Improvement:

```text
Mean error improvement:   9.13%
Median error improvement: 16.75%
```

---

# 7. Statistical Significance

The models were compared on the same 972 test windows.

```text
Baseline MPJPE: 0.152493
Graph MPJPE:    0.138569

Absolute improvement: 0.013924
Relative improvement: 9.13%
```

Window-wise comparison:

```text
Improved windows: 694 / 972 (71.40%)
Worsened windows: 278 / 972 (28.60%)
```

Wilcoxon signed-rank test:

```text
Statistically significant
```

Bootstrap confidence interval for mean error reduction:

```text
95% CI: [0.012125, 0.015787]
```

Conclusion:

```text
The Topology Graph improvement over the amplitude baseline is statistically supported.
```

---

# 8. Important Files

## Models

```text
models/
├── baseline_best.pt
├── phase_best.pt
├── spatial_best.pt
├── spatial_attention_best.pt
├── temporal_transformer_best.pt
├── topology_graph_best.pt
├── experiment_G_best.pt
├── pca_stft_best.pt
└── pca_stft_corrected_best.pt
```

## Final Topology Graph Results

```text
data/processed/topology_graph/
├── predictions.npy
├── targets.npy
├── errors.npy
├── per_joint_error.npy
├── qualitative_predictions.png
├── results.txt
├── subject_wise_results.npz
└── statistical_significance_results.npz
```

## Final Notebooks

```text
notebooks/
├── WiFiVision_Final_Documentation_and_Visualization.ipynb
├── WiFiVision_Final_Project_Completion.ipynb
└── WiFiVision_Final_Report_Generator.ipynb
```

---

# 9. Dependencies and Installation

## Recommended Python Version

```text
Python 3.10 – 3.13
```

## Create a virtual environment

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

## Install dependencies

```powershell
pip install numpy pandas matplotlib scipy scikit-learn jupyter
pip install torch torchvision
```

Verify PyTorch:

```powershell
python -c "import torch; print(torch.__version__)"
```

Start Jupyter:

```powershell
jupyter notebook
```

---

# 10. How to Reproduce the Best Result

1. Ensure the processed dataset is available.
2. Load the same test split containing **972 windows**.
3. Load:

```text
models/topology_graph_best.pt
```

4. Use the same normalization and denormalization procedure.
5. Generate predictions.
6. Calculate MPJPE and PCK.

Expected result:

```text
MPJPE ≈ 0.138569
```

Note: During checkpoint loading, the following may appear:

```text
Missing keys: ['adjacency']
```

This was handled by reconstructing the adjacency structure in the corrected model definition.

---

# 11. Current Progress

| Task | Status |
|---|---|
| Dataset preparation | Completed |
| Train/validation/test split | Completed |
| Amplitude baseline | Completed |
| Phase experiment | Completed |
| Spatial experiment | Completed |
| Spatial attention experiment | Completed |
| Temporal Transformer experiment | Completed |
| PCA/STFT experiments | Completed |
| Experiment G | Completed |
| Topology Graph model | Completed |
| Baseline regeneration | Completed |
| Final evaluation | Completed |
| Per-joint analysis | Completed |
| Subject-wise analysis | Completed |
| Error distribution analysis | Completed |
| Statistical significance testing | Completed |
| Visualization notebooks | Completed |
| Final project packaging | Pending |
| Final report | Pending |
| Presentation/demo preparation | Pending |

---

# 12. What To Do Hereafter?

## Step 1 — Finalize Documentation

Run and verify:

```text
notebooks/WiFiVision_Final_Documentation_and_Visualization.ipynb
```

Ensure all final figures use:

```text
Amplitude Baseline MPJPE: 0.152493
Topology Graph MPJPE:     0.138569
```

---

## Step 2 — Generate the Final Report

Run:

```text
notebooks/WiFiVision_Final_Report_Generator.ipynb
```

The report should include:

- Problem statement
- Dataset description
- CSI preprocessing
- Model architectures
- Experiment comparison
- Evaluation metrics
- Final results
- Per-joint analysis
- Subject-wise analysis
- Statistical significance
- Limitations
- Conclusion

---

## Step 3 — Final Project Verification

Run:

```text
notebooks/WiFiVision_Final_Project_Completion.ipynb
```

Verify:

```text
✓ Required checkpoints exist
✓ Final result files exist
✓ Baseline comparison is reproducible
✓ Final MPJPE is correct
✓ Visualizations are generated
✓ Statistical analysis is available
```

---

## Step 4 — Clean the Repository

Before final submission:

1. Remove unnecessary temporary files.
2. Keep required checkpoints and final result files.
3. Ensure README.md is updated.
4. Verify `.gitignore`.

Recommended `.gitignore`:

```text
__pycache__/
*.pyc
.ipynb_checkpoints/
venv/
.env
```

---

## Step 5 — Prepare Final Presentation / Demo

Presentation flow:

```text
1. Problem
2. Motivation
3. CSI input
4. Data preprocessing
5. Amplitude baseline
6. Experiments attempted
7. Why unsuccessful approaches failed
8. Topology Graph architecture
9. Final results
10. Statistical significance
11. Limitations
12. Future work
```

Key result:

```text
The Topology Graph model achieved a 9.13% reduction in MPJPE
compared with the reproducible amplitude baseline.
```

---

# 13. Future Work

Not required for the current project, but possible future research:

- Train with more subjects and environments.
- Improve difficult joints, especially joints 14 and 17.
- Improve CSI calibration and phase sanitization.
- Explore self-supervised CSI representation learning.
- Test cross-environment generalization.
- Test larger graph-temporal architectures with more training data.
- Perform leave-one-subject-out validation if more subjects become available.

---

# 14. Important Notes for Team Members

Before changing the project:

```text
1. Do not overwrite topology_graph_best.pt.
2. Do not change preprocessing without creating a new experiment.
3. Evaluate baseline and new models on the same test split.
4. Always use identical normalization and denormalization.
5. Reproducible baseline MPJPE = 0.152493.
6. Current best verified MPJPE = 0.138569.
```

---

# Final Status

## Experimentation

```text
COMPLETED
```

## Current Best Model

```text
Topology Graph Pose Model
```

## Final Verified MPJPE

```text
0.138569
```

## Improvement Over Baseline

```text
9.13%
```

## Remaining Work

```text
Documentation
→ Final report generation
→ Final verification
→ Repository cleanup
→ Presentation/demo preparation
```
