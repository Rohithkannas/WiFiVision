import os
import numpy as np


DATA_DIR = "data/processed/mmfi_sequence"

phase = np.load(
    os.path.join(DATA_DIR, "phase.npy"),
    mmap_mode="r"
)

rng = np.random.default_rng(42)

NUM_SAMPLES = 100

total_before = np.zeros(3)
total_after = np.zeros(3)

improved = np.zeros(3, dtype=int)

subcarriers = np.arange(114)


for _ in range(NUM_SAMPLES):

    # Random sequence and frame
    seq = rng.integers(0, phase.shape[0])
    frame = rng.integers(0, phase.shape[1])

    # 3 × 114
    raw = phase[
        seq,
        frame,
        :,
        :,
        0
    ]

    for antenna in range(3):

        # Unwrap across subcarriers
        unwrapped = np.unwrap(
            raw[antenna]
        )

        # Fit linear phase trend
        slope, intercept = np.polyfit(
            subcarriers,
            unwrapped,
            1
        )

        fitted = (
            slope * subcarriers
            + intercept
        )

        residual = (
            unwrapped - fitted
        )

        # RMS around mean before calibration
        rms_before = np.sqrt(
            np.mean(
                (
                    unwrapped
                    - np.mean(unwrapped)
                ) ** 2
            )
        )

        # RMS residual after calibration
        rms_after = np.sqrt(
            np.mean(
                residual ** 2
            )
        )

        total_before[antenna] += rms_before
        total_after[antenna] += rms_after

        if rms_after < rms_before:
            improved[antenna] += 1


print("Phase calibration — 100-frame validation")
print("=" * 55)

for antenna in range(3):

    avg_before = (
        total_before[antenna]
        / NUM_SAMPLES
    )

    avg_after = (
        total_after[antenna]
        / NUM_SAMPLES
    )

    reduction = (
        1
        - avg_after / avg_before
    ) * 100

    print(f"\nAntenna {antenna + 1}")
    print(f"Average RMS before : {avg_before:.6f}")
    print(f"Average RMS after  : {avg_after:.6f}")
    print(f"Average reduction  : {reduction:.2f}%")
    print(
        f"Frames improved    : "
        f"{improved[antenna]}/{NUM_SAMPLES}"
    )