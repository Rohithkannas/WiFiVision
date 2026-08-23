import os
import numpy as np
import torch
from torch.utils.data import Dataset


# ============================================================
# CONFIGURATION
# ============================================================

CSI_PATH = "data/processed/mmfi_calibrated/amplitude_clean.npy"
POSE_PATH = "data/processed/mmfi_calibrated/pose.npy"

MEAN_PATH = "data/processed/baseline_split/pose_mean.npy"
STD_PATH = "data/processed/baseline_split/pose_std.npy"


# ============================================================
# DATASET
# ============================================================

class MMFiBaselineDataset(Dataset):

    def __init__(
        self,
        sequence_indices,
        window_size=30,
        stride=15
    ):

        self.sequence_indices = np.asarray(
            sequence_indices,
            dtype=np.int64
        )

        self.window_size = window_size
        self.stride = stride

        # Memory-mapped arrays.
        self.csi = np.load(
            CSI_PATH,
            mmap_mode="r"
        )

        self.pose = np.load(
            POSE_PATH,
            mmap_mode="r"
        )

        # Pose normalization statistics.
        self.pose_mean = np.load(
            MEAN_PATH
        ).astype(np.float32)

        self.pose_std = np.load(
            STD_PATH
        ).astype(np.float32)

        # ----------------------------------------------------
        # Build list of:
        #
        # (sequence_id, start_frame)
        #
        # without loading CSI into RAM.
        # ----------------------------------------------------

        self.windows = []

        num_frames = self.csi.shape[1]

        for seq in self.sequence_indices:

            start = 0

            while (
                start + window_size
                <= num_frames
            ):

                self.windows.append(
                    (
                        int(seq),
                        int(start)
                    )
                )

                start += stride


    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):

        return len(self.windows)


    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(self, index):

        seq, start = self.windows[index]

        end = (
            start
            + self.window_size
        )

        # ----------------------------------------------------
        # CSI
        #
        # Original:
        #
        # (30, 3, 114, 10)
        #
        # ----------------------------------------------------

        x = np.asarray(
            self.csi[
                seq,
                start:end
            ],
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Central frame
        # ----------------------------------------------------

        center = (
            start
            + self.window_size // 2
        )

        # ----------------------------------------------------
        # Pose target:
        #
        # (17, 2)
        # ----------------------------------------------------

        y = np.asarray(
            self.pose[
                seq,
                center
            ],
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Normalize pose using TRAIN statistics.
        # ----------------------------------------------------

        y = (
            y - self.pose_mean
        ) / self.pose_std

        # ----------------------------------------------------
        # Convert to PyTorch tensors.
        # ----------------------------------------------------

        x = torch.from_numpy(
            x.copy()
        )

        y = torch.from_numpy(
            y.copy()
        )

        return x, y


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    split_dir = (
        "data/processed/baseline_split"
    )

    train_idx = np.load(
        os.path.join(
            split_dir,
            "train_idx.npy"
        )
    )

    val_idx = np.load(
        os.path.join(
            split_dir,
            "val_idx.npy"
        )
    )

    test_idx = np.load(
        os.path.join(
            split_dir,
            "test_idx.npy"
        )
    )

    train_dataset = MMFiBaselineDataset(
        train_idx
    )

    val_dataset = MMFiBaselineDataset(
        val_idx
    )

    test_dataset = MMFiBaselineDataset(
        test_idx
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "========================================"
    )

    print(
        "MM-Fi BASELINE DATASET"
    )

    print(
        "========================================"
    )

    print(
        "Train windows:",
        len(train_dataset)
    )

    print(
        "Validation windows:",
        len(val_dataset)
    )

    print(
        "Test windows:",
        len(test_dataset)
    )


    # ========================================================
    # SAMPLE
    # ========================================================

    x, y = train_dataset[0]

    print(
        "\nSample CSI shape:",
        tuple(x.shape)
    )

    print(
        "Sample pose shape:",
        tuple(y.shape)
    )

    print(
        "CSI dtype:",
        x.dtype
    )

    print(
        "Pose dtype:",
        y.dtype
    )