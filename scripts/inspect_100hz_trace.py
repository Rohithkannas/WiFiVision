import os
import numpy as np
import matplotlib.pyplot as plt


DATA_DIR = "data/processed/mmfi_sequence"

amplitude = np.load(
    os.path.join(DATA_DIR, "amplitude.npy"),
    mmap_mode="r"
)

phase = np.load(
    os.path.join(DATA_DIR, "phase.npy"),
    mmap_mode="r"
)


# ============================================================
# Select one CSI path
# ============================================================

sequence = 0
antenna = 0
subcarrier = 50


# Shape before flattening:
# (297 frames, 10 CSI samples)
#
# Flatten in chronological order:
# 297 × 10 = 2970 samples

amp_trace = amplitude[
    sequence,
    :,
    antenna,
    subcarrier,
    :
].reshape(-1)

phase_trace = phase[
    sequence,
    :,
    antenna,
    subcarrier,
    :
].reshape(-1)


# ============================================================
# Sampling information
# ============================================================

FS = 100

time = np.arange(
    len(amp_trace)
) / FS


print("Amplitude trace:", amp_trace.shape)
print("Phase trace    :", phase_trace.shape)

print("Total samples  :", len(amp_trace))
print("Sampling rate  :", FS, "Hz")
print(
    "Duration       :",
    len(amp_trace) / FS,
    "seconds"
)


# ============================================================
# Plot
# ============================================================

fig, axes = plt.subplots(
    2,
    1,
    figsize=(12, 8)
)


axes[0].plot(
    time,
    amp_trace
)

axes[0].set_title(
    "CSI Amplitude — Reconstructed 100 Hz Signal"
)

axes[0].set_xlabel(
    "Time (seconds)"
)

axes[0].set_ylabel(
    "Amplitude"
)

axes[0].grid(True)


axes[1].plot(
    time,
    phase_trace
)

axes[1].set_title(
    "CSI Phase — Reconstructed 100 Hz Signal"
)

axes[1].set_xlabel(
    "Time (seconds)"
)

axes[1].set_ylabel(
    "Phase (radians)"
)

axes[1].grid(True)


plt.tight_layout()
plt.show()