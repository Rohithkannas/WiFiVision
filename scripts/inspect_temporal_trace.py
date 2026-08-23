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


# ------------------------------------------------------------
# Select one CSI path
# ------------------------------------------------------------

sequence = 0
antenna = 0
subcarrier = 50
channel = 0


amp_trace = amplitude[
    sequence,
    :,
    antenna,
    subcarrier,
    channel
]

phase_trace = phase[
    sequence,
    :,
    antenna,
    subcarrier,
    channel
]


# ------------------------------------------------------------
# Print information
# ------------------------------------------------------------

print("Amplitude trace shape:", amp_trace.shape)
print("Phase trace shape:", phase_trace.shape)

print(
    "Amplitude range:",
    np.min(amp_trace),
    np.max(amp_trace)
)

print(
    "Phase range:",
    np.min(phase_trace),
    np.max(phase_trace)
)


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, axes = plt.subplots(
    2,
    1,
    figsize=(12, 8)
)


axes[0].plot(
    amp_trace
)

axes[0].set_title(
    "CSI Amplitude Across 297 Frames"
)

axes[0].set_xlabel(
    "Frame"
)

axes[0].set_ylabel(
    "Amplitude"
)

axes[0].grid(True)


axes[1].plot(
    phase_trace
)

axes[1].set_title(
    "CSI Phase Across 297 Frames"
)

axes[1].set_xlabel(
    "Frame"
)

axes[1].set_ylabel(
    "Phase (radians)"
)

axes[1].grid(True)


plt.tight_layout()
plt.show()