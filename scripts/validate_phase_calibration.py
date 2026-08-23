import os
import numpy as np


DATA_DIR = "data/processed/mmfi_sequence"

phase = np.load(
    os.path.join(DATA_DIR, "phase.npy"),
    mmap_mode="r"
)

# Same frame used previously
raw = phase[0, 0, :, :, 0]

subcarriers = np.arange(raw.shape[1])


print("Phase calibration validation")
print("=" * 50)

for antenna in range(raw.shape[0]):

    # Unwrap across subcarriers
    unwrapped = np.unwrap(raw[antenna])

    # Linear phase model
    slope, intercept = np.polyfit(
        subcarriers,
        unwrapped,
        1
    )

    fitted = (
        slope * subcarriers
        + intercept
    )

    residual = unwrapped - fitted

    # RMS before removing linear trend
    rms_before = np.sqrt(
        np.mean(
            (unwrapped - np.mean(unwrapped)) ** 2
        )
    )

    # RMS residual after removing linear trend
    rms_after = np.sqrt(
        np.mean(
            residual ** 2
        )
    )

    print(f"\nAntenna {antenna + 1}")
    print(f"Linear slope : {slope:.6f}")
    print(f"RMS before   : {rms_before:.6f}")
    print(f"RMS residual : {rms_after:.6f}")

    print(
        "Residual range:",
        f"{residual.min():.4f}",
        "to",
        f"{residual.max():.4f}"
    )