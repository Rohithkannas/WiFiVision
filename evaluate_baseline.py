import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from baseline_dataset import MMFiBaselineDataset
from baseline_model import CSI2PoseBaseline


# ============================================================
# CONFIGURATION
# ============================================================

SPLIT_DIR = "data/processed/baseline_split"
DATA_DIR = "data/processed/mmfi_calibrated"

CHECKPOINT = "models/baseline_best.pt"

BATCH_SIZE = 8

DEVICE = torch.device("cpu")


# ============================================================
# LOAD TEST SPLIT
# ============================================================

test_idx = np.load(
    os.path.join(
        SPLIT_DIR,
        "test_idx.npy"
    )
)

test_subjects = np.load(
    os.path.join(
        SPLIT_DIR,
        "test_subjects.npy"
    )
)


# ============================================================
# LOAD TEST DATASET
# ============================================================

test_dataset = MMFiBaselineDataset(
    test_idx,
    window_size=30,
    stride=15
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# LOAD MODEL
# ============================================================

model = CSI2PoseBaseline().to(
    DEVICE
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# LOAD POSE NORMALIZATION
# ============================================================

pose_mean = np.load(
    os.path.join(
        SPLIT_DIR,
        "pose_mean.npy"
    )
).astype(np.float32)

pose_std = np.load(
    os.path.join(
        SPLIT_DIR,
        "pose_std.npy"
    )
).astype(np.float32)


# ============================================================
# EVALUATION
# ============================================================

all_predictions = []
all_targets = []


with torch.no_grad():

    for x, y in test_loader:

        x = x.to(DEVICE)

        prediction = model(x)

        all_predictions.append(
            prediction.cpu().numpy()
        )

        all_targets.append(
            y.numpy()
        )


predictions = np.concatenate(
    all_predictions,
    axis=0
)

targets = np.concatenate(
    all_targets,
    axis=0
)


# ============================================================
# DENORMALIZE
# ============================================================

predictions = (
    predictions * pose_std
    + pose_mean
)

targets = (
    targets * pose_std
    + pose_mean
)


# ============================================================
# MPJPE
# ============================================================

joint_errors = np.sqrt(
    np.sum(
        (predictions - targets) ** 2,
        axis=2
    )
)

mpjpe = np.mean(
    joint_errors
)


# ============================================================
# PCK
# ============================================================

# Thresholds are expressed in the original
# coordinate system.

thresholds = [
    0.05,
    0.10,
    0.15,
    0.20
]

pck_results = {}

for threshold in thresholds:

    pck = np.mean(
        joint_errors <= threshold
    )

    pck_results[threshold] = pck


# ============================================================
# PER-JOINT ERROR
# ============================================================

per_joint_error = np.mean(
    joint_errors,
    axis=0
)


# ============================================================
# OUTPUT
# ============================================================

print("========================================")
print("MM-Fi BASELINE TEST EVALUATION")
print("========================================")

print(
    "Checkpoint:",
    CHECKPOINT
)

print(
    "Test subjects:",
    test_subjects
)

print(
    "Test windows:",
    len(test_dataset)
)

print()

print(
    f"MPJPE: {mpjpe:.6f}"
)

print()

print("PCK:")

for threshold, pck in pck_results.items():

    print(
        f"  @ {threshold:.2f}: "
        f"{pck * 100:.2f}%"
    )


print()

print("Per-joint error:")

for joint, error in enumerate(
    per_joint_error
):

    print(
        f"  Joint {joint + 1:02d}: "
        f"{error:.6f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

np.save(
    os.path.join(
        SPLIT_DIR,
        "baseline_predictions.npy"
    ),
    predictions
)

np.save(
    os.path.join(
        SPLIT_DIR,
        "baseline_targets.npy"
    ),
    targets
)

np.save(
    os.path.join(
        SPLIT_DIR,
        "baseline_joint_errors.npy"
    ),
    joint_errors
)


print(
    "\nPredictions saved."
)