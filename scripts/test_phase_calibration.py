import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# LOAD ONE RAW CSI FRAME
# ============================================================

DATA_DIR = "data/processed/mmfi_sequence"

phase = np.load(
    os.path.join(DATA_DIR, "phase.npy"),
    mmap_mode="r"
)

# ------------------------------------------------------------
# Select:
# sequence 0
# frame 0
# CSI sample 0
#
# Result:
# 3 antennas × 114 subcarriers
# ------------------------------------------------------------

raw = phase[0, 0, :, :, 0]

print("Raw phase shape:", raw.shape)


# ============================================================
# PHASE CALIBRATION
# ============================================================

subcarriers = np.arange(raw.shape[1])

calibrated = np.zeros_like(raw)


for antenna in range(raw.shape[0]):

    # --------------------------------------------------------
    # 1. Unwrap phase ACROSS subcarriers
    # --------------------------------------------------------

    unwrapped = np.unwrap(
        raw[antenna]
    )

    # --------------------------------------------------------
    # 2. Linear regression
    #
    # phase ≈ slope * subcarrier + intercept
    # --------------------------------------------------------

    slope, intercept = np.polyfit(
        subcarriers,
        unwrapped,
        1
    )

    # --------------------------------------------------------
    # 3. Remove linear phase component
    # --------------------------------------------------------

    calibrated[antenna] = (
        unwrapped
        - (
            slope * subcarriers
            + intercept
        )
    )


# ============================================================
# DISPLAY
# ============================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(12, 10)
)


for antenna in range(3):

    axes[antenna].plot(
        subcarriers,
        raw[antenna],
        label="Raw phase"
    )

    axes[antenna].plot(
        subcarriers,
        calibrated[antenna],
        label="Calibrated phase"
    )

    axes[antenna].set_title(
        f"Antenna {antenna + 1}"
    )

    axes[antenna].set_xlabel(
        "Subcarrier"
    )

    axes[antenna].set_ylabel(
        "Phase (rad)"
    )

    axes[antenna].legend()
    axes[antenna].grid(True)


plt.tight_layout()
plt.show()


# ============================================================
# NUMERICAL SUMMARY
# ============================================================

print("\nCalibration summary:")

for antenna in range(3):

    unwrapped = np.unwrap(
        raw[antenna]
    )

    slope, intercept = np.polyfit(
        subcarriers,
        unwrapped,
        1
    )

    print(
        f"Antenna {antenna + 1}: "
        f"slope={slope:.6f}, "
        f"intercept={intercept:.6f}"
    )