import os
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = "data/processed/mmfi_sequence"
OUTPUT_DIR = "data/processed/mmfi_calibrated"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading input data...")

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


# ============================================================
# DATASET DIMENSIONS
# ============================================================

N_SEQ = amplitude.shape[0]
N_FRAMES = amplitude.shape[1]
N_ANT = amplitude.shape[2]
N_SUB = amplitude.shape[3]
N_INNER = amplitude.shape[4]

print("\nDataset:")
print("Sequences       :", N_SEQ)
print("Frames          :", N_FRAMES)
print("Antennas        :", N_ANT)
print("Subcarriers     :", N_SUB)
print("Inner CSI samples:", N_INNER)


# ============================================================
# CREATE OUTPUT FILES
# ============================================================

print("\nCreating output files...")

clean_amplitude = np.lib.format.open_memmap(
    os.path.join(
        OUTPUT_DIR,
        "amplitude_clean.npy"
    ),
    mode="w+",
    dtype=np.float32,
    shape=amplitude.shape
)

calibrated_phase = np.lib.format.open_memmap(
    os.path.join(
        OUTPUT_DIR,
        "phase_calibrated.npy"
    ),
    mode="w+",
    dtype=np.float32,
    shape=phase.shape
)


# ============================================================
# LINEAR PHASE CALIBRATION SETUP
# ============================================================

# Subcarrier indices
k = np.arange(
    N_SUB,
    dtype=np.float64
)

# Design matrix:
#
# phase(k) = slope*k + intercept
#
A = np.column_stack(
    (
        k,
        np.ones(N_SUB)
    )
)

# Pseudoinverse for least-squares fitting
P = np.linalg.pinv(A)


# ============================================================
# PROCESS DATA
# ============================================================

total_invalid = 0

print("\nProcessing sequences...\n")


for seq in range(N_SEQ):

    # --------------------------------------------------------
    # Load one sequence
    #
    # Shape:
    # (297, 3, 114, 10)
    # --------------------------------------------------------

    amp = np.array(
        amplitude[seq],
        dtype=np.float32
    )

    ph = np.array(
        phase[seq],
        dtype=np.float32
    )


    # ========================================================
    # STEP 1 — REPAIR INVALID AMPLITUDE
    # ========================================================

    invalid = ~np.isfinite(amp)

    count = int(
        invalid.sum()
    )

    total_invalid += count

    if count > 0:

        # Process every:
        #
        # antenna × subcarrier × inner CSI sample
        #
        # along the 297-frame temporal axis.

        for antenna in range(N_ANT):

            for subcarrier in range(N_SUB):

                for inner in range(N_INNER):

                    signal = amp[
                        :,
                        antenna,
                        subcarrier,
                        inner
                    ]

                    bad = ~np.isfinite(
                        signal
                    )

                    if not bad.any():
                        continue

                    valid = np.isfinite(
                        signal
                    )

                    valid_indices = np.flatnonzero(
                        valid
                    )

                    if len(valid_indices) == 0:

                        signal[:] = 0.0

                    else:

                        # Linear interpolation across
                        # neighboring valid temporal samples.

                        signal[bad] = np.interp(
                            np.flatnonzero(bad),
                            valid_indices,
                            signal[valid]
                        )


    # ========================================================
    # STEP 2 — PHASE CALIBRATION
    # ========================================================

    # Original shape:
    #
    # (frames, antennas, subcarriers, inner)
    #
    # We move subcarrier to the LAST dimension:
    #
    # (frames, antennas, inner, subcarriers)

    ph_work = np.transpose(
        ph,
        (0, 1, 3, 2)
    )

    # --------------------------------------------------------
    # Unwrap phase ACROSS subcarriers
    # --------------------------------------------------------

    unwrapped = np.unwrap(
        ph_work,
        axis=-1
    )

    # --------------------------------------------------------
    # Flatten everything except subcarrier
    #
    # (297, 3, 10, 114)
    #
    # becomes
    #
    # (297*3*10, 114)
    # --------------------------------------------------------

    flat = unwrapped.reshape(
        -1,
        N_SUB
    )

    # --------------------------------------------------------
    # Fit:
    #
    # phase = slope * k + intercept
    #
    # for every temporal/antenna/inner sample.
    # --------------------------------------------------------

    coefficients = (
        flat @ P.T
    )

    slopes = coefficients[:, 0]
    intercepts = coefficients[:, 1]

    # --------------------------------------------------------
    # Construct fitted linear phase
    # --------------------------------------------------------

    fitted = (
        slopes[:, None] * k[None, :]
        + intercepts[:, None]
    )

    # --------------------------------------------------------
    # Remove linear phase component
    # --------------------------------------------------------

    calibrated = (
        flat - fitted
    )

    # Restore:
    #
    # (297, 3, 10, 114)

    calibrated = calibrated.reshape(
        N_FRAMES,
        N_ANT,
        N_INNER,
        N_SUB
    )

    # Return to original:
    #
    # (297, 3, 114, 10)

    calibrated = np.transpose(
        calibrated,
        (0, 1, 3, 2)
    )


    # ========================================================
    # SAVE
    # ========================================================

    clean_amplitude[seq] = amp

    calibrated_phase[seq] = (
        calibrated.astype(np.float32)
    )


    # ========================================================
    # PROGRESS
    # ========================================================

    if (
        (seq + 1) % 10 == 0
        or seq == 0
        or seq == N_SEQ - 1
    ):

        print(
            f"Processed "
            f"{seq + 1}/{N_SEQ}"
            f" | repaired values: "
            f"{total_invalid}"
        )


# ============================================================
# SAVE METADATA
# ============================================================

np.save(
    os.path.join(
        OUTPUT_DIR,
        "pose.npy"
    ),
    np.asarray(pose)
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
# FLUSH
# ============================================================

clean_amplitude.flush()
calibrated_phase.flush()


# ============================================================
# FINAL VALIDATION
# ============================================================

amp_bad = (
    np.isnan(clean_amplitude).sum()
    + np.isinf(clean_amplitude).sum()
)

phase_bad = (
    np.isnan(calibrated_phase).sum()
    + np.isinf(calibrated_phase).sum()
)


print("\n========================================")
print("CSI calibration completed")
print("========================================")

print(
    "Invalid amplitude values repaired:",
    total_invalid
)

print(
    "Remaining invalid amplitude:",
    amp_bad
)

print(
    "Remaining invalid phase:",
    phase_bad
)

print(
    "Amplitude shape:",
    clean_amplitude.shape
)

print(
    "Phase shape:",
    calibrated_phase.shape
)

print(
    "Amplitude range:",
    clean_amplitude.min(),
    "to",
    clean_amplitude.max()
)

print(
    "Calibrated phase range:",
    calibrated_phase.min(),
    "to",
    calibrated_phase.max()
)

print(
    "\nSaved to:",
    OUTPUT_DIR
)