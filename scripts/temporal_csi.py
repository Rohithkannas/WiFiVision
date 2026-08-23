import numpy as np
import matplotlib.pyplot as plt

X = np.load(
    "data/processed/mmfi/X.npy",
    mmap_mode="r"
)

# Select one sample
sample = X[0]

# sample shape = (3, 114, 10)
# Average across antennas and subcarriers
temporal_signal = sample.mean(axis=(0, 1))

print("Sample shape:", sample.shape)
print("Temporal signal shape:", temporal_signal.shape)
print("Temporal values:")
print(temporal_signal)

plt.figure(figsize=(10, 5))

plt.plot(
    range(10),
    temporal_signal,
    marker="o"
)

plt.xlabel("Temporal Frame")
plt.ylabel("Mean CSI Amplitude")
plt.title("MM-Fi CSI Temporal Variation")
plt.grid(True)

plt.tight_layout()
plt.show()