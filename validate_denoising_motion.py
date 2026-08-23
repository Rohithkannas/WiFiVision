import os
import numpy as np
from scipy.signal import savgol_filter


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data/processed/mmfi_calibrated"

amplitude = np.load(
    os.path.join(
        DATA_DIR,
        "amplitude_clean.npy"
    ),
    mmap_mode="r"
)

pose = np.load(
    os.path.join(
        DATA_DIR,
        "pose.npy"
    ),
    mmap_mode="r"
)

WINDOWS = [7, 11, 15, 21]

NUM_SEQUENCES = 100

rng = np.random.default_rng(42)


# ============================================================
# CORRELATION STORAGE
# ============================================================

correlations = {
    w: []
    for w in WINDOWS
}


# ============================================================
# RANDOM SEQUENCES
# ============================================================

sequences = rng.choice(
    amplitude.shape[0],
    size=NUM_SEQUENCES,
    replace=False
)


# ============================================================
# PROCESS
# ============================================================

print("Testing denoising against human motion...")
print("Sequences:", NUM_SEQUENCES)
print()


for count, seq in enumerate(sequences):

    # --------------------------------------------------------
    # CSI
    #
    # Shape:
    # (297, 3, 114, 10)
    # --------------------------------------------------------

    csi = np.asarray(
        amplitude[seq],
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Reduce CSI dimensions to one temporal signal.
    #
    # We average absolute CSI amplitude across:
    #
    # antenna × subcarrier × inner CSI dimension
    # --------------------------------------------------------

    csi_signal = np.mean(
        csi,
        axis=(1, 2, 3)
    )

    # --------------------------------------------------------
    # Pose
    #
    # Shape:
    # (297, 17, 2)
    # --------------------------------------------------------

    p = np.asarray(
        pose[seq],
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Human movement speed.
    #
    # Frame-to-frame displacement of all 17 joints.
    # --------------------------------------------------------

    pose_velocity = np.sqrt(
        np.sum(
            np.diff(p, axis=0) ** 2,
            axis=2
        )
    )

    pose_velocity = np.mean(
        pose_velocity,
        axis=1
    )

    # --------------------------------------------------------
    # CSI temporal variation
    # --------------------------------------------------------

    csi_velocity = np.abs(
        np.diff(csi_signal)
    )

    # Make lengths identical
    min_len = min(
        len(csi_velocity),
        len(pose_velocity)
    )

    csi_velocity = csi_velocity[
        :min_len
    ]

    pose_velocity_local = pose_velocity[
        :min_len
    ]


    # --------------------------------------------------------
    # Test every denoising window
    # --------------------------------------------------------

    for window in WINDOWS:

        filtered = savgol_filter(
            csi_signal,
            window_length=window,
            polyorder=2
        )

        filtered_velocity = np.abs(
            np.diff(filtered)
        )

        filtered_velocity = (
            filtered_velocity[:min_len]
        )

        # ----------------------------------------------------
        # Pearson correlation
        # ----------------------------------------------------

        if (
            np.std(filtered_velocity) > 0
            and
            np.std(pose_velocity_local) > 0
        ):

            corr = np.corrcoef(
                filtered_velocity,
                pose_velocity_local
            )[0, 1]

            if np.isfinite(corr):
                correlations[window].append(
                    corr
                )


    if (
        (count + 1) % 20 == 0
        or count == 0
        or count + 1 == NUM_SEQUENCES
    ):

        print(
            f"Processed "
            f"{count + 1}/{NUM_SEQUENCES}"
        )


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 65)
print("DENOISING vs HUMAN-MOTION CORRELATION")
print("=" * 65)

for window in WINDOWS:

    values = np.asarray(
        correlations[window]
    )

    print(
        f"\nWindow {window}"
    )

    print(
        "Valid sequences:",
        len(values)
    )

    print(
        "Mean correlation:",
        f"{np.mean(values):.6f}"
    )

    print(
        "Median correlation:",
        f"{np.median(values):.6f}"
    )

    print(
        "Std:",
        f"{np.std(values):.6f}"
    )


# ============================================================
# BEST WINDOW
# ============================================================

mean_correlations = {
    w: np.mean(correlations[w])
    for w in WINDOWS
}

best_window = max(
    mean_correlations,
    key=mean_correlations.get
)

print("\n" + "=" * 65)

print(
    "Best motion-preserving window:",
    best_window
)

print(
    "Mean correlation:",
    f"{mean_correlations[best_window]:.6f}"
)

print("=" * 65)