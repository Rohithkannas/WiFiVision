import os
import numpy as np


# ============================================================
# PATHS
# ============================================================

DATA_DIR = "data/processed/mmfi_calibrated"
SPLIT_DIR = "data/processed/baseline_split"

OUTPUT_DIR = "data/processed/baseline_split"

POSE_PATH = os.path.join(
    DATA_DIR,
    "pose.npy"
)

TRAIN_IDX_PATH = os.path.join(
    SPLIT_DIR,
    "train_idx.npy"
)


# ============================================================
# LOAD
# ============================================================

pose = np.load(
    POSE_PATH,
    mmap_mode="r"
)

train_idx = np.load(
    TRAIN_IDX_PATH
)


print("Pose shape:", pose.shape)
print("Training sequences:", len(train_idx))


# ============================================================
# TRAINING POSE ONLY
# ============================================================

train_pose = np.asarray(
    pose[train_idx],
    dtype=np.float32
)


# ============================================================
# NORMALIZATION STATISTICS
# ============================================================

# Calculate statistics independently for x and y.

pose_mean = np.mean(
    train_pose,
    axis=(0, 1)
)

pose_std = np.std(
    train_pose,
    axis=(0, 1)
)

# Prevent division by zero.

pose_std = np.maximum(
    pose_std,
    1e-6
)


# ============================================================
# SAVE
# ============================================================

np.save(
    os.path.join(
        OUTPUT_DIR,
        "pose_mean.npy"
    ),
    pose_mean
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "pose_std.npy"
    ),
    pose_std
)


# ============================================================
# OUTPUT
# ============================================================

print("\n========================================")
print("POSE NORMALIZATION")
print("========================================")

print(
    "Mean shape:",
    pose_mean.shape
)

print(
    "Std shape:",
    pose_std.shape
)

print(
    "\nX mean range:",
    pose_mean[:, 0].min(),
    "to",
    pose_mean[:, 0].max()
)

print(
    "Y mean range:",
    pose_mean[:, 1].min(),
    "to",
    pose_mean[:, 1].max()
)

print(
    "\nX std range:",
    pose_std[:, 0].min(),
    "to",
    pose_std[:, 0].max()
)

print(
    "Y std range:",
    pose_std[:, 1].min(),
    "to",
    pose_std[:, 1].max()
)

print(
    "\nSaved to:",
    OUTPUT_DIR
)