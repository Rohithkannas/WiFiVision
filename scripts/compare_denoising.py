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

rng = np.random.default_rng(42)

NUM_SAMPLES = 100

WINDOWS = [7, 11, 15, 21]
POLYORDER = 2


# ============================================================
# STORAGE
# ============================================================

original_std = np.zeros(NUM_SAMPLES)

filtered_std = {
    w: np.zeros(NUM_SAMPLES)
    for w in WINDOWS
}

residual_std = {
    w: np.zeros(NUM_SAMPLES)
    for w in WINDOWS
}


# ============================================================
# RANDOM CSI TRACE SELECTION
# ============================================================

print("Testing denoising parameters...")
print("Samples:", NUM_SAMPLES)
print()


for i in range(NUM_SAMPLES):

    sequence = rng.integers(
        0,
        amplitude.shape[0]
    )

    antenna = rng.integers(
        0,
        amplitude.shape[2]
    )

    subcarrier = rng.integers(
        0,
        amplitude.shape[3]
    )

    inner = rng.integers(
        0,
        amplitude.shape[4]
    )

    signal = np.asarray(
        amplitude[
            sequence,
            :,
            antenna,
            subcarrier,
            inner
        ],
        dtype=np.float64
    )

    original_std[i] = np.std(
        signal
    )


    # ========================================================
    # TEST EACH FILTER
    # ========================================================

    for window in WINDOWS:

        filtered = savgol_filter(
            signal,
            window_length=window,
            polyorder=POLYORDER
        )

        residual = (
            signal - filtered
        )

        filtered_std[window][i] = np.std(
            filtered
        )

        residual_std[window][i] = np.std(
            residual
        )


# ============================================================
# RESULTS
# ============================================================

print("=" * 65)
print("DENOISING COMPARISON")
print("=" * 65)

mean_original = np.mean(
    original_std
)

print(
    f"\nOriginal average STD: "
    f"{mean_original:.6f}"
)

print()

for window in WINDOWS:

    mean_filtered = np.mean(
        filtered_std[window]
    )

    mean_residual = np.mean(
        residual_std[window]
    )

    reduction = (
        1
        - mean_filtered / mean_original
    ) * 100

    print(
        f"Window {window:2d} | "
        f"Filtered STD: {mean_filtered:.6f} | "
        f"Residual STD: {mean_residual:.6f} | "
        f"STD reduction: {reduction:.2f}%"
    )


# ============================================================
# CONSISTENCY
# ============================================================

print("\n" + "=" * 65)
print("FILTER CONSISTENCY")
print("=" * 65)

for window in WINDOWS:

    improved = np.sum(
        filtered_std[window]
        < original_std
    )

    print(
        f"Window {window:2d}: "
        f"{improved}/{NUM_SAMPLES} "
        f"traces have reduced STD"
    )


print("\nDone.")