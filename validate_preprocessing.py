import os
import numpy as np


# ============================================================
# PATHS
# ============================================================

RAW_DIR = "data/processed/mmfi_sequence"
CLEAN_DIR = "data/processed/mmfi_calibrated"


# ============================================================
# LOAD DATA
# ============================================================

raw_amp = np.load(
    os.path.join(RAW_DIR, "amplitude.npy"),
    mmap_mode="r"
)

raw_phase = np.load(
    os.path.join(RAW_DIR, "phase.npy"),
    mmap_mode="r"
)

clean_amp = np.load(
    os.path.join(CLEAN_DIR, "amplitude_clean.npy"),
    mmap_mode="r"
)

cal_phase = np.load(
    os.path.join(CLEAN_DIR, "phase_calibrated.npy"),
    mmap_mode="r"
)


print("========================================")
print("PREPROCESSING VALIDATION")
print("========================================")


# ============================================================
# 1. SHAPE VALIDATION
# ============================================================

print("\n[1] Shape validation")

print("Raw amplitude :", raw_amp.shape)
print("Clean amplitude:", clean_amp.shape)

print("Raw phase     :", raw_phase.shape)
print("Calibrated phase:", cal_phase.shape)

print(
    "Amplitude shape preserved:",
    raw_amp.shape == clean_amp.shape
)

print(
    "Phase shape preserved:",
    raw_phase.shape == cal_phase.shape
)


# ============================================================
# 2. INVALID VALUE VALIDATION
# ============================================================

print("\n[2] Invalid-value validation")

raw_bad_amp = (
    np.isnan(raw_amp).sum()
    + np.isinf(raw_amp).sum()
)

clean_bad_amp = (
    np.isnan(clean_amp).sum()
    + np.isinf(clean_amp).sum()
)

raw_bad_phase = (
    np.isnan(raw_phase).sum()
    + np.isinf(raw_phase).sum()
)

clean_bad_phase = (
    np.isnan(cal_phase).sum()
    + np.isinf(cal_phase).sum()
)

print(
    "Raw amplitude invalid:",
    raw_bad_amp
)

print(
    "Clean amplitude invalid:",
    clean_bad_amp
)

print(
    "Raw phase invalid:",
    raw_bad_phase
)

print(
    "Calibrated phase invalid:",
    clean_bad_phase
)


# ============================================================
# 3. AMPLITUDE STATISTICS
# ============================================================

print("\n[3] Amplitude statistics")

print(
    "Raw amplitude:")
print(
    "  min    :",
    np.nanmin(raw_amp)
)
print(
    "  max    :",
    np.nanmax(raw_amp)
)
print(
    "  mean   :",
    np.nanmean(raw_amp)
)
print(
    "  std    :",
    np.nanstd(raw_amp)
)

print(
    "\nClean amplitude:"
)

print(
    "  min    :",
    np.min(clean_amp)
)

print(
    "  max    :",
    np.max(clean_amp)
)

print(
    "  mean   :",
    np.mean(clean_amp)
)

print(
    "  std    :",
    np.std(clean_amp)
)


# ============================================================
# 4. PHASE STATISTICS
# ============================================================

print("\n[4] Phase statistics")

print("Raw phase:")

print(
    "  min    :",
    np.min(raw_phase)
)

print(
    "  max    :",
    np.max(raw_phase)
)

print(
    "  mean   :",
    np.mean(raw_phase)
)

print(
    "  std    :",
    np.std(raw_phase)
)

print("\nCalibrated phase:")

print(
    "  min    :",
    np.min(cal_phase)
)

print(
    "  max    :",
    np.max(cal_phase)
)

print(
    "  mean   :",
    np.mean(cal_phase)
)

print(
    "  std    :",
    np.std(cal_phase)
)


# ============================================================
# 5. PHASE LINEAR-TREND VALIDATION
# ============================================================

print("\n[5] Phase linear-trend validation")

rng = np.random.default_rng(42)

NUM_SAMPLES = 100

N_SUB = raw_phase.shape[3]

subcarriers = np.arange(
    N_SUB,
    dtype=np.float64
)

raw_rms = np.zeros(3)
cal_rms = np.zeros(3)

for _ in range(NUM_SAMPLES):

    seq = rng.integers(
        0,
        raw_phase.shape[0]
    )

    frame = rng.integers(
        0,
        raw_phase.shape[1]
    )

    inner = rng.integers(
        0,
        raw_phase.shape[4]
    )

    for antenna in range(3):

        # Raw phase
        raw = raw_phase[
            seq,
            frame,
            antenna,
            :,
            inner
        ]

        raw_unwrapped = np.unwrap(
            raw
        )

        slope, intercept = np.polyfit(
            subcarriers,
            raw_unwrapped,
            1
        )

        raw_fit = (
            slope * subcarriers
            + intercept
        )

        raw_residual = (
            raw_unwrapped
            - np.mean(raw_unwrapped)
        )

        # Calibrated phase
        calibrated = cal_phase[
            seq,
            frame,
            antenna,
            :,
            inner
        ]

        cal_residual = (
            calibrated
            - np.mean(calibrated)
        )

        raw_rms[antenna] += np.sqrt(
            np.mean(
                raw_residual ** 2
            )
        )

        cal_rms[antenna] += np.sqrt(
            np.mean(
                cal_residual ** 2
            )
        )


print(
    "\nAverage RMS after removing mean:"
)

for antenna in range(3):

    before = (
        raw_rms[antenna]
        / NUM_SAMPLES
    )

    after = (
        cal_rms[antenna]
        / NUM_SAMPLES
    )

    reduction = (
        1 - after / before
    ) * 100

    print(
        f"Antenna {antenna + 1}: "
        f"{before:.6f} → "
        f"{after:.6f} "
        f"({reduction:.2f}% reduction)"
    )


# ============================================================
# FINAL
# ============================================================

print("\n========================================")
print("VALIDATION COMPLETE")
print("========================================")