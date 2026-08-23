import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data/processed/mmfi_calibrated"

amplitude = np.load(
    os.path.join(DATA_DIR, "amplitude_clean.npy"),
    mmap_mode="r"
)


# ============================================================
# SELECT ONE TEMPORAL CSI TRACE
# ============================================================

sequence = 0
antenna = 0
subcarrier = 50
inner = 0

signal = amplitude[
    sequence,
    :,
    antenna,
    subcarrier,
    inner
].astype(np.float64)


# ============================================================
# SAVITZKY-GOLAY FILTER
# ============================================================

# 11-frame window, polynomial order 2.
#
# At ~100 Hz this corresponds to approximately
# 110 ms of temporal data.

denoised = savgol_filter(
    signal,
    window_length=11,
    polyorder=2
)


# ============================================================
# STATISTICS
# ============================================================

noise_residual = (
    signal - denoised
)

print("Original shape:", signal.shape)

print(
    "Original std:",
    np.std(signal)
)

print(
    "Denoised std:",
    np.std(denoised)
)

print(
    "Residual std:",
    np.std(noise_residual)
)


# ============================================================
# VISUALIZATION
# ============================================================

time = np.arange(
    len(signal)
) / 100.0


plt.figure(
    figsize=(12, 6)
)

plt.plot(
    time,
    signal,
    label="Clean CSI amplitude"
)

plt.plot(
    time,
    denoised,
    label="Savitzky-Golay filtered"
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "CSI amplitude"
)

plt.title(
    "CSI Amplitude Denoising"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()