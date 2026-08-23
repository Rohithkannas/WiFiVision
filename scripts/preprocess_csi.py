import os
import numpy as np
from scipy.signal import medfilt


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = "data/processed/mmfi_sequence"
OUTPUT_DIR = "data/processed/mmfi_preprocessed"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading CSI data...")

amplitude = np.load(
    os.path.join(INPUT_DIR, "amplitude.npy"),
    mmap_mode="r"
)

phase = np.load(
    os.path.join(INPUT_DIR, "phase.npy"),
    mmap_mode="r"
)

pose = np.load(
    os.path.join(INPUT_DIR, "pose.npy"),
    mmap_mode="r"
)

subject = np.load(
    os.path.join(INPUT_DIR, "subject.npy")
)

action = np.load(
    os.path.join(INPUT_DIR, "action.npy")
)

print("Amplitude:", amplitude.shape)
print("Phase    :", phase.shape)
print("Pose     :", pose.shape)


# ============================================================
# COPY TO WRITABLE ARRAYS
# ============================================================

amp = np.array(
    amplitude,
    dtype=np.float32
)

ph = np.array(
    phase,
    dtype=np.float32
)


# ============================================================
# STEP 1 — REPAIR INVALID AMPLITUDE VALUES
# ============================================================

print("\nStep 1: repairing invalid amplitude values...")

invalid = ~np.isfinite(amp)

print(
    "Invalid amplitude values:",
    invalid.sum()
)


# The invalid values are extremely sparse.
# Replace each invalid value with the median of
# neighboring valid values along the temporal axis.
#
# Shape:
# (sequence, frame, antenna, subcarrier, time)


for seq in range(amp.shape[0]):

    for antenna in range(amp.shape[2]):

        for subcarrier in range(amp.shape[3]):

            for t in range(amp.shape[4]):

                signal = amp[
                    seq,
                    :,
                    antenna,
                    subcarrier,
                    t
                ]

                bad = ~np.isfinite(signal)

                if not bad.any():
                    continue

                valid_indices = np.where(
                    np.isfinite(signal)
                )[0]

                if len(valid_indices) == 0:
                    signal[:] = 0.0
                    continue

                for idx in np.where(bad)[0]:

                    # nearest valid samples
                    previous = valid_indices[
                        valid_indices < idx
                    ]

                    following = valid_indices[
                        valid_indices > idx
                    ]

                    if (
                        len(previous) > 0
                        and len(following) > 0
                    ):

                        p = previous[-1]
                        f = following[0]

                        signal[idx] = (
                            signal[p] + signal[f]
                        ) / 2.0

                    elif len(previous) > 0:

                        signal[idx] = signal[
                            previous[-1]
                        ]

                    elif len(following) > 0:

                        signal[idx] = signal[
                            following[0]
                        ]


# Verify
print(
    "Remaining invalid amplitude values:",
    (~np.isfinite(amp)).sum()
)


# ============================================================
# STEP 2 — PHASE UNWRAPPING
# ============================================================

print("\nStep 2: phase unwrapping...")

# Phase is wrapped approximately within [-pi, pi].
#
# We unwrap along the temporal dimension of each
# CSI representation.
#
# axis=1 corresponds to the 297 consecutive
# frames within each action sequence.

ph_unwrapped = np.unwrap(
    ph,
    axis=1
).astype(np.float32)


# ============================================================
# STEP 3 — MEDIAN DENOISING
# ============================================================

print("\nStep 3: median denoising amplitude...")

# Median filtering along the temporal dimension.
#
# Kernel size = 3:
#   previous frame
#   current frame
#   next frame
#
# This removes isolated spikes while preserving
# relatively fast changes better than a large window.

amp_denoised = np.empty_like(amp)

for seq in range(amp.shape[0]):

    for antenna in range(amp.shape[2]):

        for subcarrier in range(amp.shape[3]):

            for t in range(amp.shape[4]):

                signal = amp[
                    seq,
                    :,
                    antenna,
                    subcarrier,
                    t
                ]

                amp_denoised[
                    seq,
                    :,
                    antenna,
                    subcarrier,
                    t
                ] = medfilt(
                    signal,
                    kernel_size=3
                )


# ============================================================
# STEP 4 — ROBUST NORMALIZATION
# ============================================================

print("\nStep 4: normalization...")

# Normalize each sequence independently using
# median and MAD.
#
# This is more robust to outliers than ordinary
# mean/std normalization.

for seq in range(amp_denoised.shape[0]):

    x = amp_denoised[seq]

    median = np.median(x)

    mad = np.median(
        np.abs(x - median)
    )

    if mad < 1e-6:
        mad = 1.0

    amp_denoised[seq] = (
        x - median
    ) / (
        1.4826 * mad
    )


# ============================================================
# SAVE
# ============================================================

print("\nSaving preprocessed data...")

np.save(
    os.path.join(
        OUTPUT_DIR,
        "amplitude.npy"
    ),
    amp_denoised
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "phase_unwrapped.npy"
    ),
    ph_unwrapped
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "pose.npy"
    ),
    pose
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "subject.npy"
    ),
    subject
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "action.npy"
    ),
    action
)


# ============================================================
# FINAL CHECK
# ============================================================

print("\n========================================")
print("CSI preprocessing completed")
print("========================================")

print(
    "Amplitude:",
    amp_denoised.shape
)

print(
    "Phase:",
    ph_unwrapped.shape
)

print(
    "Amplitude min:",
    amp_denoised.min()
)

print(
    "Amplitude max:",
    amp_denoised.max()
)

print(
    "Phase min:",
    ph_unwrapped.min()
)

print(
    "Phase max:",
    ph_unwrapped.max()
)

print(
    "Remaining NaN/Inf:",
    (~np.isfinite(amp_denoised)).sum()
    + (~np.isfinite(ph_unwrapped)).sum()
)

print(
    "\nSaved to:",
    OUTPUT_DIR
)