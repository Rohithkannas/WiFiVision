import os
import numpy as np


# ============================================================
# PATHS
# ============================================================

DATA_DIR = "data/processed/mmfi_calibrated"
OUTPUT_DIR = "data/processed/baseline_split"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD SUBJECT LABELS
# ============================================================

subject = np.load(
    os.path.join(DATA_DIR, "subject.npy")
)

action = np.load(
    os.path.join(DATA_DIR, "action.npy")
)

pose = np.load(
    os.path.join(DATA_DIR, "pose.npy"),
    mmap_mode="r"
)


print("Total sequences:", len(subject))
print("Subjects:", np.unique(subject))
print("Actions:", np.unique(action))


# ============================================================
# SUBJECT SPLIT
# ============================================================

TRAIN_SUBJECTS = [1, 2, 3, 4, 5, 6, 7]

VAL_SUBJECTS = [8]

TEST_SUBJECTS = [9, 10]


train_idx = np.where(
    np.isin(subject, TRAIN_SUBJECTS)
)[0]

val_idx = np.where(
    np.isin(subject, VAL_SUBJECTS)
)[0]

test_idx = np.where(
    np.isin(subject, TEST_SUBJECTS)
)[0]


# ============================================================
# SAVE INDICES
# ============================================================

np.save(
    os.path.join(
        OUTPUT_DIR,
        "train_idx.npy"
    ),
    train_idx
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "val_idx.npy"
    ),
    val_idx
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "test_idx.npy"
    ),
    test_idx
)


# ============================================================
# SAVE SUBJECT INFORMATION
# ============================================================

np.save(
    os.path.join(
        OUTPUT_DIR,
        "train_subjects.npy"
    ),
    np.array(TRAIN_SUBJECTS)
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "val_subjects.npy"
    ),
    np.array(VAL_SUBJECTS)
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "test_subjects.npy"
    ),
    np.array(TEST_SUBJECTS)
)


# ============================================================
# SUMMARY
# ============================================================

print("\n========================================")
print("BASELINE SUBJECT SPLIT")
print("========================================")

print(
    "Train subjects:",
    TRAIN_SUBJECTS
)

print(
    "Validation subjects:",
    VAL_SUBJECTS
)

print(
    "Test subjects:",
    TEST_SUBJECTS
)

print()

print(
    "Training sequences:",
    len(train_idx)
)

print(
    "Validation sequences:",
    len(val_idx)
)

print(
    "Test sequences:",
    len(test_idx)
)

print()

print(
    "Train subject IDs:",
    np.unique(subject[train_idx])
)

print(
    "Validation subject IDs:",
    np.unique(subject[val_idx])
)

print(
    "Test subject IDs:",
    np.unique(subject[test_idx])
)

print("\nSaved to:")
print(OUTPUT_DIR)